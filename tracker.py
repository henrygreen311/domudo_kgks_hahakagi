import ccxt
import re
import time
import json
import logging
import requests
import threading
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# Bybit and KuCoin block datacenter IPs, so they route through a PHP relay on
# our own domain instead of hitting the exchange API directly.
PROXY_BASE = "https://arb-bot.infinityfree.io/proxy.php"
PROXY_EXCHANGES = {'Bybit', 'KuCoin'}
PROXY_EXCHANGE_IDS = {name.lower() for name in PROXY_EXCHANGES}

# InfinityFree's bot-check wants these headers + a valid "__test" cookie or it
# blocks the request. The cookie itself lives in Supabase (api_keys.cookie,
# per-exchange row) rather than being hardcoded here — see route_through_proxy().
_PROXY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://arb-bot.infinityfree.io/",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

_proxy_session = requests.Session()

MIN_VOLUME_USDT = 20_000
MIN_GAP_PERCENT = 2.0
MAX_GAP_PERCENT = 100.0
ALT_ROUTE_MIN_GAP_PERCENT = MIN_GAP_PERCENT
MAX_PRICE_RATIO = 3.0
MIN_CONFIRMED_PAIR_GAP = 3.0

CAPITAL_USD     = 1000
DEPTH_CHECK_USD = 1000
MIN_STORE_PROFIT_USD = 0.1

DEFAULT_TAKER_FEE  = 0.001
LOG_CONFIRMED_ONLY = True

ORDER_BOOK_LIMIT = 50
ORDER_BOOK_LIMIT_OVERRIDES = {
    'KuCoin': 100,
}

def order_book_limit_for(exchange_name):
    return ORDER_BOOK_LIMIT_OVERRIDES.get(exchange_name, ORDER_BOOK_LIMIT)

RETRY_ATTEMPTS = 1
RETRY_DELAY = 1
SCAN_INTERVAL_SEC = 5 * 60

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("arb_scanner.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

# supabase-py uses httpx/httpcore under the hood, which log every HTTP request
# (including the full Supabase URL) at INFO level. Since our root logger is set
# to INFO, those would otherwise leak into the console/log file. Quiet them.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def with_retries(fn, label):
    last_err = None
    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
    raise last_err

def _load_db_config() -> dict:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "db.txt"),
        os.path.join(os.path.dirname(here), "db.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            config = {}
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, _, v = line.partition("=")
                        config[k.strip()] = v.strip().strip('"')
            return config
    raise FileNotFoundError(f"db.txt not found in any of: {candidates}")

def _get_supabase():
    from supabase import create_client
    config = _load_db_config()
    return create_client(config["SUPABASE_URL"], config["SUPABASE_KEY"])

_SUPABASE_CLIENT = None

def _get_supabase_cached():
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is None:
        _SUPABASE_CLIENT = _get_supabase()
    return _SUPABASE_CLIENT

def _clean_secret(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().upper() in ("", "NULL"):
        return None
    return value

def load_api_credentials() -> dict:
    try:
        sb = _get_supabase()
        rows = sb.table("api_keys").select("*").execute()
    except Exception as e:
        log.warning(f"  WARNING  Supabase credentials: {str(e)[:150]}")
        return {}
    creds = {}
    for row in rows.data or []:
        name = (row.get("exchange_name") or "").strip().lower()
        if not name:
            continue
        creds[name] = {
            'apiKey': _clean_secret(row.get("api_key")),
            'secret': _clean_secret(row.get("api_secret")),
            'password': _clean_secret(row.get("passphrase")),
            'cookie': _clean_secret(row.get("cookie")),
        }
    return creds

CREDENTIALS = load_api_credentials()

def ccxt_config(exchange_id_lower):
    proxied = exchange_id_lower in PROXY_EXCHANGE_IDS
    cfg = {
        'enableRateLimit': True,
        'timeout': 20_000 if proxied else TIMEOUT_MS,
        'options': {
            'adjustForTimeDifference': True,
        },
    }
    creds = CREDENTIALS.get(exchange_id_lower)
    if creds:
        if creds.get('apiKey'):
            cfg['apiKey'] = creds['apiKey']
        if creds.get('secret'):
            cfg['secret'] = creds['secret']
        if creds.get('password'):
            cfg['password'] = creds['password']
    else:
        log.warning(f"  WARNING  no API credentials found in Supabase for '{exchange_id_lower}' — public endpoints only")
    return cfg

TIMEOUT_MS = 10_000

def _build_kucoin():
    cfg = ccxt_config('kucoin')
    ex = route_through_proxy(ccxt.kucoin(cfg))
    ex.set_markets(ex.fetch_markets())
    return ex

def route_through_proxy(ex):
    """Monkeypatch ccxt's low-level fetch() so every request this exchange makes
    goes through our PHP relay instead of hitting the exchange API directly."""
    exchange_key = ex.id

    proxy_cookie = (CREDENTIALS.get(exchange_key) or {}).get('cookie')
    if not proxy_cookie:
        log.warning(
            f"  WARNING  no proxy cookie found in Supabase for '{exchange_key}' "
            f"(api_keys.cookie) — InfinityFree's bot-check will likely block every request"
        )
    proxy_cookies = {"__test": proxy_cookie} if proxy_cookie else {}

    def proxied_fetch(url, method='GET', headers=None, body=None):
        parsed = urlparse(url)
        request_headers = dict(headers or {})
        request_headers.update(_PROXY_HEADERS)
        request_headers['X-Proxy-Target-Host'] = parsed.netloc
        new_url = f"{PROXY_BASE}/{exchange_key}{parsed.path}"
        if parsed.query:
            new_url += f"?{parsed.query}"
        def do_request():
            resp = _proxy_session.request(
                method,
                new_url,
                headers=request_headers,
                cookies=proxy_cookies,
                data=body if method != 'GET' else None,
                timeout=20,
            )
            if resp.status_code >= 400:
                raise Exception(f"{exchange_key} {method} {new_url} -> {resp.status_code}: {resp.text[:300]}")
            try:
                return json.loads(resp.text)
            except ValueError:
                return resp.text
        try:
            return do_request()
        except Exception:
            return do_request()
    ex.fetch = proxied_fetch
    if exchange_key == 'bybit':
        ex.has['fetchCurrencies'] = False
    return ex

EXCHANGE_BUILDERS = {
    'Bybit':   lambda: route_through_proxy(ccxt.bybit(ccxt_config('bybit'))),
    'Bitget':  lambda: ccxt.bitget(ccxt_config('bitget')),
    'MEXC':    lambda: ccxt.mexc(ccxt_config('mexc')),
    'BingX':   lambda: ccxt.bingx(ccxt_config('bingx')),
    'KuCoin':  _build_kucoin,
    'CoinEx':  lambda: ccxt.coinex(ccxt_config('coinex')),
    'BitMart': lambda: ccxt.bitmart(ccxt_config('bitmart')),
    'OKX':     lambda: ccxt.okx(ccxt_config('okx')),
    'LBank':   lambda: ccxt.lbank(ccxt_config('lbank')),
}

EXTRA_PARAMS = {
    'Bybit': {'category': 'spot'},
}

EXCHANGES = {}

def ensure_exchange(name):
    if EXCHANGES.get(name) is not None:
        return EXCHANGES[name]
    try:
        ex = with_retries(EXCHANGE_BUILDERS[name], name)
    except Exception as e:
        log.warning(f"  WARNING  {name}: could not initialize — {str(e)[:400]}")
        EXCHANGES[name] = None
        return None

    if name in PROXY_EXCHANGES:
        try:
            ex.load_markets()
            if not ex.markets:
                log.warning(f"  WARNING  {name}: load_markets() returned 0 markets (no exception raised)")
        except Exception as e:
            log.warning(f"  WARNING  {name}: load_markets() failed — {str(e)[:400]}")

    EXCHANGES[name] = ex
    return ex

def init_exchanges():
    for name in EXCHANGE_BUILDERS:
        ensure_exchange(name)

def get_usdt_tickers(name):
    ex = ensure_exchange(name)
    if ex is None:
        return {}
    params = EXTRA_PARAMS.get(name, {})
    try:
        tickers = with_retries(lambda: ex.fetch_tickers(params=params), name)
    except Exception as e:
        log.warning(f"  WARNING  {name}: {str(e)[:400]}")
        EXCHANGES[name] = None
        return {}
    markets = getattr(ex, 'markets', None) or {}
    result = {}
    for symbol, t in tickers.items():
        if not symbol.endswith('/USDT'):
            continue
        last = t.get('last')
        if not last or last <= 0:
            continue
        market = markets.get(symbol)
        if market is not None:
            if market.get('active') is False:
                continue
            if market.get('spot') is False:
                continue
        result[symbol] = t

    if not result:
        sample = list(tickers.keys())[:5]
        log.warning(
            f"  WARNING  {name}: fetched {len(tickers)} raw tickers, 0 passed filtering  |  "
            f"{len(markets)} markets loaded  |  sample symbols: {sample}"
        )

    return result

def build_price_map(all_tickers):
    price_map = defaultdict(dict)
    for exchange_name, tickers in all_tickers.items():
        for symbol, t in tickers.items():
            last   = t.get('last',        0) or 0
            bid    = t.get('bid',         0) or 0
            ask    = t.get('ask',         0) or 0
            volume = t.get('quoteVolume', 0) or 0
            change = t.get('percentage',  0) or 0
            high   = t.get('high',        0) or 0
            low    = t.get('low',         0) or 0
            if bid <= 0: bid = last
            if ask <= 0: ask = last
            price_map[symbol][exchange_name] = {
                'price':  last,
                'volume': volume,
                'bid':    bid,
                'ask':    ask,
                'change': change,
                'high':   high,
                'low':    low,
            }
    return price_map

def find_opportunities(price_map):
    results = []
    for symbol, exchanges in price_map.items():
        if len(exchanges) < 2:
            continue
        valid_buys  = [(ex, d) for ex, d in exchanges.items() if d['ask'] > 0]
        valid_sells = [(ex, d) for ex, d in exchanges.items() if d['bid'] > 0]
        if not valid_buys or not valid_sells:
            continue
        buy_ex,  buy_data  = min(valid_buys,  key=lambda x: x[1]['ask'])
        sell_ex, sell_data = max(valid_sells, key=lambda x: x[1]['bid'])
        if buy_ex == sell_ex:
            continue
        buy_price  = buy_data['ask']
        sell_price = sell_data['bid']
        if buy_price <= 0 or sell_price <= 0:
            continue
        gap_pct = ((sell_price - buy_price) / buy_price) * 100
        if gap_pct < MIN_GAP_PERCENT:
            continue
        if gap_pct > MAX_GAP_PERCENT:
            continue
        if sell_price / buy_price > MAX_PRICE_RATIO:
            continue
        buy_vol  = buy_data['volume']
        sell_vol = sell_data['volume']
        if min(buy_vol, sell_vol) < MIN_VOLUME_USDT:
            continue
        results.append({
            'symbol':     symbol,
            'buy_ex':     buy_ex,
            'buy_price':  buy_price,
            'buy_vol':    buy_vol,
            'buy_chg':    buy_data['change'],
            'buy_high':   buy_data['high'],
            'buy_low':    buy_data['low'],
            'sell_ex':    sell_ex,
            'sell_price': sell_price,
            'sell_vol':   sell_vol,
            'sell_chg':   sell_data['change'],
            'gap_pct':    gap_pct,
            'n_exchanges': len(exchanges),
            'all_prices': {ex: {'price': d['price'], 'bid': d['bid'], 'ask': d['ask']} for ex, d in exchanges.items()},
            'depth_checked':  False,
            'depth_ok':       None,
            'buy_depth_price':  None,
            'sell_depth_price': None,
            'depth_gap_pct':  None,
            'meta_checked':     False,
            'buy_name':         None,
            'sell_name':        None,
            'name_match':       None,
            'name_suspect':     False,
            'common_networks':  [],
            'blocked_networks': [],
            'blocked_details':  [],
            'alt_route':        None,
            'confirmed_pair':   None,
            'networks_known':   False,
            'withdrawal_network':    None,
            'withdrawal_fee':        None,
            'withdrawal_min_tokens': None,
            'contract_buy':     None,
            'contract_sell':    None,
            'contract_match':   None,
            'verified':         False,
            'buy_meta_summary':  None,
            'sell_meta_summary': None,
            'extra_meta_summaries': [],
            'corrected_symbol':       None,
            'corrected_sell_price':   None,
            'corrected_gap_pct':      None,
            'corrected_depth_gap_pct': None,
            'corrected_sell_depth_price': None,
        })
    results.sort(key=lambda x: x['gap_pct'], reverse=True)
    return results

def _sort_book_side(levels, side):
    if not levels:
        return []
    return sorted(levels, key=lambda lvl: lvl[0], reverse=(side == 'bids'))

def walk_book(levels, target_usd):
    filled_quote = 0.0
    filled_base  = 0.0
    for level in levels:
        if len(level) < 2:
            continue
        price, amount = level[0], level[1]
        if price <= 0 or amount <= 0:
            continue
        level_quote = price * amount
        if filled_quote + level_quote >= target_usd:
            remaining_quote = target_usd - filled_quote
            remaining_base  = remaining_quote / price
            filled_quote += remaining_quote
            filled_base  += remaining_base
            return filled_quote / filled_base, filled_quote, True
        filled_quote += level_quote
        filled_base  += amount
    if filled_base <= 0:
        return None, 0.0, False
    return filled_quote / filled_base, filled_quote, False

DEPTH_FETCH_RETRIES    = 3
DEPTH_FETCH_RETRY_DELAY = 1

def _fetch_order_book_safe(exchange_name, venue, symbol):
    params   = EXTRA_PARAMS.get(exchange_name, {})
    last_err = None
    for attempt in range(1, DEPTH_FETCH_RETRIES + 1):
        try:
            return venue.fetch_order_book(
                symbol, limit=order_book_limit_for(exchange_name), params=params
            )
        except Exception as e:
            last_err = e
            if attempt < DEPTH_FETCH_RETRIES:
                time.sleep(DEPTH_FETCH_RETRY_DELAY)
    err_summary = f"{type(last_err).__name__}: {str(last_err)[:150]}"
    log.warning(
        f"  WARNING  depth check {symbol} on {exchange_name} "
        f"(failed all {DEPTH_FETCH_RETRIES} attempts): {err_summary}"
    )
    return None

def check_depth(r):
    buy_ex, sell_ex, symbol = r['buy_ex'], r['sell_ex'], r['symbol']
    buy_venue  = ensure_exchange(buy_ex)
    sell_venue = ensure_exchange(sell_ex)
    if buy_venue is None or sell_venue is None:
        return r

    buy_ob  = _fetch_order_book_safe(buy_ex,  buy_venue,  symbol)
    sell_ob = _fetch_order_book_safe(sell_ex, sell_venue, symbol)

    if buy_ob is None or sell_ob is None:
        r['depth_checked'] = True
        return r

    asks = _sort_book_side(buy_ob.get('asks', []) or [], 'asks')
    bids = _sort_book_side(sell_ob.get('bids', []) or [], 'bids')

    buy_price, _, buy_ok   = walk_book(asks, DEPTH_CHECK_USD)
    sell_price, _, sell_ok = walk_book(bids, DEPTH_CHECK_USD)

    r['depth_checked'] = True
    if buy_price is None or sell_price is None or buy_price <= 0:
        r['depth_ok'] = False
        return r
    r['buy_depth_price']  = buy_price
    r['sell_depth_price'] = sell_price
    r['depth_gap_pct']    = ((sell_price - buy_price) / buy_price) * 100
    r['depth_ok']         = buy_ok and sell_ok
    return r

def apply_depth_checks(results):
    if not results:
        return results
    with ThreadPoolExecutor(max_workers=min(len(results), 8)) as pool:
        futures = {pool.submit(check_depth, r): r for r in results}
        for future in as_completed(futures):
            r = futures[future]
            try:
                future.result()
            except Exception as e:
                log.warning(f"  WARNING  depth check {r.get('symbol')} ({r.get('buy_ex')}->{r.get('sell_ex')}): {type(e).__name__}: {str(e)[:200]}")
                r['depth_checked'] = True
                r['depth_ok'] = False
    return results

def _clean_networks_dict(networks):
    if not isinstance(networks, dict):
        return {}
    return {code: net for code, net in networks.items() if isinstance(net, dict)}

def _sanitize_currencies(data):
    if not isinstance(data, dict):
        return {}
    clean = {}
    for code, cur in data.items():
        if not isinstance(cur, dict):
            continue
        networks = cur.get('networks')
        if networks is not None:
            cur['networks'] = _clean_networks_dict(networks)
        clean[code] = cur
    return clean

CURRENCY_CACHE = {}
CURRENCY_CACHE_TTL = 1800
CURRENCY_LOCKS = defaultdict(threading.Lock)

def get_currencies(exchange_name):
    now = time.time()
    cached = CURRENCY_CACHE.get(exchange_name)
    if cached and (now - cached['ts']) < CURRENCY_CACHE_TTL:
        return cached['data']

    with CURRENCY_LOCKS[exchange_name]:
        cached = CURRENCY_CACHE.get(exchange_name)
        now = time.time()
        if cached and (now - cached['ts']) < CURRENCY_CACHE_TTL:
            return cached['data']

        ex = ensure_exchange(exchange_name)
        if ex is None:
            return cached['data'] if cached else {}
        try:
            data = ex.fetch_currencies() or {}
            data = _sanitize_currencies(data)
        except Exception as e:
            log.warning(f"  WARNING  {exchange_name} currencies: {str(e)[:120]}")
            data = {}
        if not data and cached:
            data = cached['data']
        CURRENCY_CACHE[exchange_name] = {'ts': now, 'data': data}
        if data:
            sample_code = next(iter(data))
            sample = data[sample_code]
            log.info(
                f"  [currencies] {exchange_name}: {len(data)} loaded  |  "
                f"sample '{sample_code}': name={sample.get('name')!r}  "
                f"networks={list((sample.get('networks') or {}).keys())}"
            )
        else:
            log.warning(f"  WARNING  {exchange_name} currencies: fetch_currencies() returned nothing")
        return data

NETWORK_ALIASES = {
    'ETH': 'ETH', 'ERC20': 'ETH', 'ETHEREUM': 'ETH', 'ETHER': 'ETH',
    'BSC': 'BSC', 'BEP20': 'BSC', 'BNB': 'BSC', 'BNBSMARTCHAIN': 'BSC', 'BNBBEP20': 'BSC',
    'TRX': 'TRX', 'TRC20': 'TRX', 'TRON': 'TRX',
    'MATIC': 'MATIC', 'POLYGON': 'MATIC', 'POLYGONPOS': 'MATIC',
    'ARBITRUM': 'ARB', 'ARB': 'ARB', 'ARBITRUMONE': 'ARB',
    'OPTIMISM': 'OP', 'OP': 'OP',
    'SOL': 'SOL', 'SOLANA': 'SOL',
    'AVAX': 'AVAX', 'AVALANCHE': 'AVAX', 'AVAXC': 'AVAX', 'AVALANCHEC': 'AVAX',
    'BTC': 'BTC', 'BITCOIN': 'BTC',
    'LIGHTNING': 'BTC-LN', 'LN': 'BTC-LN',
    'BASE': 'BASE',
    'TON': 'TON',
    'APT': 'APT', 'APTOS': 'APT',
    'SUI': 'SUI',
    'XRP': 'XRP', 'RIPPLE': 'XRP',
    'DOGE': 'DOGE', 'DOGECOIN': 'DOGE',
    'LTC': 'LTC', 'LITECOIN': 'LTC',
    'ZKSYNC': 'ZKSYNC', 'ZKSYNCERA': 'ZKSYNC',
}

_NETWORK_PAREN_RE = re.compile(r'\(([^)]+)\)\s*$')

def normalize_network(code):
    if not code:
        return None
    raw = str(code).strip()
    m = _NETWORK_PAREN_RE.search(raw)
    if m:
        raw = m.group(1)
    key = raw.upper().replace('-', '').replace('_', '').replace(' ', '')
    return NETWORK_ALIASES.get(key, key)

_NAME_JUNK_SUFFIXES = (' token', ' coin', ' protocol', ' network', ' finance', ' chain', ' project')

def normalize_name(name):
    if not name:
        return ''
    n = name.strip().lower()
    for junk in _NAME_JUNK_SUFFIXES:
        if n.endswith(junk):
            n = n[: -len(junk)]
    return re.sub(r'[^a-z0-9]', '', n)

def name_is_informative(name, code):
    if not name:
        return False
    return name.strip().upper() != (code or '').strip().upper()

def compare_names(buy_name, sell_name, code):
    buy_ok  = name_is_informative(buy_name, code)
    sell_ok = name_is_informative(sell_name, code)
    if buy_ok and sell_ok:
        return (normalize_name(buy_name) == normalize_name(sell_name)), False
    if buy_ok or sell_ok:
        real_name = buy_name if buy_ok else sell_name
        norm_real = normalize_name(real_name)
        norm_code = normalize_name(code)
        if norm_code and (norm_code in norm_real or norm_real in norm_code):
            return None, False
        return None, True
    return None, False

_CONTRACT_KEYS = ('contractAddress', 'contract_address', 'contract', 'tokenAddress', 'token_address', 'address')

def _extract_contract(network_data, currency_data=None):
    for source in (network_data, currency_data):
        if not isinstance(source, dict):
            continue
        info = (source or {}).get('info')
        if isinstance(info, dict):
            for key in _CONTRACT_KEYS:
                val = info.get(key)
                if val:
                    return val
    return None

def _fallback_networks(exchange_name, base):
    ex = ensure_exchange(exchange_name)
    if ex is None:
        return {}
    try:
        if getattr(ex, 'has', {}).get('fetchDepositWithdrawFee'):
            fee_data = ex.fetch_deposit_withdraw_fee(base)
            return _clean_networks_dict((fee_data or {}).get('networks') or {})
        if getattr(ex, 'has', {}).get('fetchDepositWithdrawFees'):
            fee_data = ex.fetch_deposit_withdraw_fees([base])
            return _clean_networks_dict(((fee_data or {}).get(base) or {}).get('networks') or {})
    except Exception as e:
        log.warning(f"  WARNING  {exchange_name} deposit/withdraw fee fallback for {base}: {str(e)[:400]}")
    return {}

def _to_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def _fmt_networks(networks):
    if not networks or not isinstance(networks, dict):
        return "(none)"
    parts = []
    for code, data in list(networks.items())[:10]:
        w   = data.get('withdraw', True)
        d   = data.get('deposit', True)
        act = data.get('active', True)
        fee = data.get('fee')
        min_wd = ((data.get('limits') or {}).get('withdraw') or {}).get('min')
        parts.append(f"{code}[w={w},d={d},act={act},fee={fee},min={min_wd}]")
    suffix = "" if len(networks) <= 10 else f"  (+{len(networks) - 10} more)"
    return ", ".join(parts) + suffix

def find_currency_by_name(exchange_name, target_name, exclude_code=None):
    currencies = get_currencies(exchange_name)
    norm_target = normalize_name(target_name)
    if not norm_target:
        return None
    for code, cur in currencies.items():
        if code == exclude_code:
            continue
        if not isinstance(cur, dict):
            continue
        cur_name = (cur.get('name') or '').strip()
        if not cur_name:
            continue
        if normalize_name(cur_name) == norm_target:
            return code, cur
    return None

def fetch_symbol_price(exchange_name, symbol, side):
    ex = ensure_exchange(exchange_name)
    if ex is None:
        return None
    try:
        params = EXTRA_PARAMS.get(exchange_name, {})
        t = ex.fetch_ticker(symbol, params=params)
    except Exception as e:
        log.warning(f"  WARNING  {exchange_name} fetch_ticker({symbol}): {str(e)[:400]}")
        return None
    val = t.get(side)
    return val if val else t.get('last')

def _network_entries_matching_contract(networks, cur, contract):
    if not isinstance(networks, dict):
        return list(networks.items()) if isinstance(networks, dict) else []
    if not contract:
        return list(networks.items())
    out = [
        (code, data) for code, data in networks.items()
        if (c := _extract_contract(data, cur)) and c.lower() == contract.lower()
    ]
    return out or list(networks.items())

def _route_profit(route, symbol):
    """Compute actual net P&L for an alt-route (gap% alone can be misleading —
    a wide gap can still be eaten alive by gas + trading fees)."""
    if route['direction'] == 'alt_buy':
        buy_ex_for_fee, sell_ex_for_fee = route['alt_ex'], route['sell_ex']
        buy_price, sell_price = route['alt_price'], route['sell_price_ref']
    else:
        buy_ex_for_fee, sell_ex_for_fee = route['buy_ex'], route['alt_ex']
        buy_price, sell_price = route['buy_price_ref'], route['alt_price']
    buy_rate  = get_trading_fee_rate(buy_ex_for_fee,  symbol)
    sell_rate = get_trading_fee_rate(sell_ex_for_fee, symbol)
    return calc_arb_profit(
        CAPITAL_USD, buy_price, sell_price,
        fee_tokens=route.get('fee'),
        min_withdrawal_tokens=route.get('min_withdrawal'),
        buy_taker_rate=buy_rate,
        sell_taker_rate=sell_rate,
    )

def evaluate_network_block(base, buy_ex, sell_ex, buy_price, sell_price, all_prices, blocked_details, contract=None, symbol=None):
    other_exchanges = sorted(ex for ex in all_prices if ex not in (buy_ex, sell_ex))
    if not other_exchanges:
        return None

    routes = []

    for entry in blocked_details:
        norm_key = entry['norm_key']

        # Case A: buy-side withdrawal blocked -> try an alternate BUY venue
        if not entry['can_withdraw'] and entry['can_deposit']:
            for alt_ex in other_exchanges:
                alt_price = all_prices.get(alt_ex, {}).get('ask')
                if not alt_price or alt_price <= 0 or alt_price >= sell_price:
                    continue
                gap_pct = ((sell_price - alt_price) / alt_price) * 100
                if gap_pct < ALT_ROUTE_MIN_GAP_PERCENT:
                    continue
                cur = get_currencies(alt_ex).get(base)
                if not isinstance(cur, dict):
                    continue
                networks = cur.get('networks') or {}
                if not networks:
                    networks = _fallback_networks(alt_ex, base)
                candidates = _network_entries_matching_contract(networks, cur, contract)
                matched = next((c_d for c_d in candidates if normalize_network(c_d[0]) == norm_key), None)
                if not matched:
                    continue
                code, data = matched
                withdraw_ok = data.get('withdraw', True) is not False and data.get('active', True) is not False
                if withdraw_ok:
                    routes.append({
                        'direction':      'alt_buy',
                        'blocked_ex':     buy_ex,
                        'alt_ex':         alt_ex,
                        'sell_ex':        sell_ex,
                        'network':        norm_key,
                        'gap_pct':        gap_pct,
                        'alt_price':      alt_price,
                        'sell_price_ref': sell_price,
                        'fee':            _to_float(data.get('fee')),
                        'min_withdrawal': _to_float(((data.get('limits') or {}).get('withdraw') or {}).get('min')),
                        'blocked_reason': f"{buy_ex} withdraw disabled for {norm_key}",
                    })

        # Case B: sell-side deposit blocked -> try an alternate SELL venue
        if entry['can_withdraw'] and not entry['can_deposit']:
            for alt_ex in other_exchanges:
                alt_price = all_prices.get(alt_ex, {}).get('bid')
                if not alt_price or alt_price <= buy_price:
                    continue
                gap_pct = ((alt_price - buy_price) / buy_price) * 100
                if gap_pct < ALT_ROUTE_MIN_GAP_PERCENT:
                    continue
                cur = get_currencies(alt_ex).get(base)
                if not isinstance(cur, dict):
                    continue
                networks = cur.get('networks') or {}
                if not networks:
                    networks = _fallback_networks(alt_ex, base)
                candidates = _network_entries_matching_contract(networks, cur, contract)
                matched = next((c_d for c_d in candidates if normalize_network(c_d[0]) == norm_key), None)
                if not matched:
                    continue
                code, data = matched
                deposit_ok = data.get('deposit', True) is not False and data.get('active', True) is not False
                if deposit_ok:
                    buy_data_entry = entry.get('buy_data') or {}
                    routes.append({
                        'direction':      'alt_sell',
                        'blocked_ex':     sell_ex,
                        'alt_ex':         alt_ex,
                        'buy_ex':         buy_ex,
                        'network':        norm_key,
                        'gap_pct':        gap_pct,
                        'alt_price':      alt_price,
                        'buy_price_ref':  buy_price,
                        'fee':            _to_float(buy_data_entry.get('fee')),
                        'min_withdrawal': _to_float(((buy_data_entry.get('limits') or {}).get('withdraw') or {}).get('min')),
                        'blocked_reason': f"{sell_ex} deposit disabled for {norm_key}",
                    })

    if not routes:
        return None

    # Evaluate every viable alt-route found across all blocked networks/exchanges,
    # scored by actual net P&L (gas + trading fees can eat a wide gap), and take
    # the most profitable one instead of stopping at the first hit or highest gap%.
    if symbol:
        for route in routes:
            profit = _route_profit(route, symbol)
            route['profit'] = profit
            route['net_pnl'] = profit['net_pnl'] if profit else float('-inf')
        best = max(routes, key=lambda route: route['net_pnl'])
        if best['net_pnl'] < 0:
            log.info(
                f"       VERIFY  ⚠️ Best alt-route ({best['alt_ex']}) still nets "
                f"{best['net_pnl']:+.4f} USDT after fees — gap alone looked good but isn't profitable"
            )
        return best

    # No symbol available (shouldn't normally happen) — fall back to gap-based pick.
    return max(routes, key=lambda route: route['gap_pct'])

def meta_summary_for(exchange_name, base):
    cur = get_currencies(exchange_name).get(base)
    if not isinstance(cur, dict):
        return f"{exchange_name}: '{base}' currency data is not a dict (type {type(cur).__name__})"
    name = (cur.get('name') or '').strip()
    networks = cur.get('networks') or {}
    if not networks:
        networks = _fallback_networks(exchange_name, base)
    contract = None
    for net_data in networks.values():
        if isinstance(net_data, dict):
            contract = _extract_contract(net_data, cur)
            if contract:
                break
    return f"{exchange_name}: name={name!r}  networks: {_fmt_networks(networks)}  contract={contract}"

def find_confirmed_pair(base, all_prices):
    info = {}
    for ex in all_prices:
        cur = get_currencies(ex).get(base)
        if not isinstance(cur, dict):
            continue
        name = (cur.get('name') or '').strip()
        if not name_is_informative(name, base):
            continue
        info[ex] = (name, cur)
    ex_list = list(info.keys())
    matched_pairs = []
    for i in range(len(ex_list)):
        for j in range(i + 1, len(ex_list)):
            ex1, ex2 = ex_list[i], ex_list[j]
            n1, cur1 = info[ex1]
            n2, cur2 = info[ex2]
            if normalize_name(n1) == normalize_name(n2):
                matched_pairs.append((ex1, ex2, cur1, cur2))
    if not matched_pairs:
        return None
    best = None
    for ex1, ex2, cur1, cur2 in matched_pairs:
        p1, p2 = all_prices.get(ex1, {}).get('price'), all_prices.get(ex2, {}).get('price')
        if not p1 or not p2 or p1 == p2:
            continue
        if p1 < p2:
            buy_c, sell_c, buy_p, sell_p, buy_cur, sell_cur = ex1, ex2, p1, p2, cur1, cur2
        else:
            buy_c, sell_c, buy_p, sell_p, buy_cur, sell_cur = ex2, ex1, p2, p1, cur2, cur1
        gap_pct = ((sell_p - buy_p) / buy_p) * 100
        if best is None or gap_pct > best['gap_pct']:
            best = {
                'buy_ex': buy_c, 'sell_ex': sell_c,
                'buy_price': buy_p, 'sell_price': sell_p,
                'gap_pct': gap_pct,
                'buy_cur': buy_cur, 'sell_cur': sell_cur,
            }
    if best is None or best['gap_pct'] < MIN_CONFIRMED_PAIR_GAP:
        return None
    buy_networks = best['buy_cur'].get('networks') or {}
    if not buy_networks:
        buy_networks = _fallback_networks(best['buy_ex'], base)
    sell_networks = best['sell_cur'].get('networks') or {}
    if not sell_networks:
        sell_networks = _fallback_networks(best['sell_ex'], base)
    buy_by_norm = defaultdict(list)
    for code, data in buy_networks.items():
        buy_by_norm[normalize_network(code)].append((code, data))
    sell_by_norm = defaultdict(list)
    for code, data in sell_networks.items():
        sell_by_norm[normalize_network(code)].append((code, data))
    tradable_network = None
    blocked_msgs = []
    for norm_key, buy_entries in buy_by_norm.items():
        sell_entries = sell_by_norm.get(norm_key)
        if not sell_entries:
            continue
        buy_code, buy_data   = buy_entries[0]
        sell_code, sell_data = sell_entries[0]
        can_withdraw = buy_data.get('withdraw', True) is not False and buy_data.get('active', True) is not False
        can_deposit  = sell_data.get('deposit', True) is not False and sell_data.get('active', True) is not False
        if can_withdraw and can_deposit:
            tradable_network = norm_key
            tradable_fee = _to_float(buy_data.get('fee'))
            break
        else:
            if not can_withdraw:
                blocked_msgs.append(f"{best['buy_ex']} withdraw disabled for {buy_code}")
            if not can_deposit:
                blocked_msgs.append(f"{best['sell_ex']} deposit disabled for {sell_code}")
    else:
        tradable_fee = None
    best['tradable_network'] = tradable_network
    best['blocked_msgs']     = blocked_msgs
    best['fee']              = tradable_fee
    if tradable_network:
        buy_by_norm_for_min = defaultdict(list)
        raw_buy_nets = best['buy_cur'].get('networks') or _fallback_networks(best['buy_ex'], base)
        for code, data in raw_buy_nets.items():
            buy_by_norm_for_min[normalize_network(code)].append((code, data))
        entries_for_min = buy_by_norm_for_min.get(tradable_network)
        if entries_for_min:
            best['min_withdrawal'] = _to_float(((entries_for_min[0][1].get('limits') or {}).get('withdraw') or {}).get('min'))
        else:
            best['min_withdrawal'] = None
    else:
        best['min_withdrawal'] = None
    return best

def other_deposit_blocks(base, norm_key, exclude, all_prices):
    blocked_others = []
    for ex in sorted(all_prices):
        if ex in exclude:
            continue
        cur = get_currencies(ex).get(base)
        if not isinstance(cur, dict):
            continue
        networks = cur.get('networks') or {}
        if not networks:
            networks = _fallback_networks(ex, base)
        by_norm = defaultdict(list)
        for code, data in networks.items():
            by_norm[normalize_network(code)].append((code, data))
        entries = by_norm.get(norm_key)
        if not entries:
            continue
        code, data = entries[0]
        can_deposit = data.get('deposit', True) is not False and data.get('active', True) is not False
        if not can_deposit:
            blocked_others.append((ex, code))
    return blocked_others

def check_metadata(r):
    base = r['symbol'].split('/')[0]
    buy_ex, sell_ex = r['buy_ex'], r['sell_ex']
    other_exchanges = sorted(ex for ex in r['all_prices'] if ex not in (buy_ex, sell_ex))
    r['extra_meta_summaries'] = [meta_summary_for(ex, base) for ex in other_exchanges]
    buy_currencies  = get_currencies(buy_ex)
    sell_currencies = get_currencies(sell_ex)
    buy_cur  = buy_currencies.get(base)
    sell_cur = sell_currencies.get(base)
    r['meta_checked'] = True

    # --- safeguard: ensure we have dicts; otherwise treat as missing ---
    if buy_cur is not None and not isinstance(buy_cur, dict):
        buy_cur = None
    if sell_cur is not None and not isinstance(sell_cur, dict):
        sell_cur = None

    if not buy_cur or not sell_cur:
        r['buy_meta_summary']  = f"{buy_ex}: '{base}' not found in currency list" if not buy_cur else f"{buy_ex}: currency data invalid"
        r['sell_meta_summary'] = f"{sell_ex}: '{base}' not found in currency list" if not sell_cur else f"{sell_ex}: currency data invalid"
        r['verified'] = None
        return r

    buy_name  = (buy_cur.get('name')  or '').strip()
    sell_name = (sell_cur.get('name') or '').strip()
    r['buy_name'], r['sell_name'] = buy_name, sell_name
    r['name_match'], r['name_suspect'] = compare_names(buy_name, sell_name, base)
    is_confirmed_conflict = r['name_match'] is False
    should_attempt_resolution = is_confirmed_conflict or (r['name_match'] is None and name_is_informative(buy_name, base))
    if should_attempt_resolution:
        found = find_currency_by_name(sell_ex, buy_name, exclude_code=base)
        if not found:
            if is_confirmed_conflict:
                r['verified'] = False
                r['buy_meta_summary']  = f"{buy_ex}: name={buy_name!r}"
                r['sell_meta_summary'] = (
                    f"{sell_ex}: '{base}' is name={sell_name!r} — CONFIRMED different token; "
                    f"'{buy_name}' does not exist on {sell_ex} under any ticker"
                )
                return r
        else:
            real_code, real_cur = found
            real_symbol = f"{real_code}/USDT"
            real_price  = fetch_symbol_price(sell_ex, real_symbol, 'bid')
            if not real_price:
                if is_confirmed_conflict:
                    r['verified'] = False
                    r['buy_meta_summary'] = (
                        f"{buy_ex}: name={buy_name!r} — this project also lists on {sell_ex} as "
                        f"'{real_code}', but its price couldn't be fetched to re-confirm any gap"
                    )
                    r['sell_meta_summary'] = f"{sell_ex}: '{base}' is name={sell_name!r} — a DIFFERENT token from {buy_ex}'s '{buy_name}'"
                    return r
            else:
                corrected_gap = ((real_price - r['buy_price']) / r['buy_price']) * 100
                r['corrected_symbol']     = real_symbol
                r['corrected_sell_price'] = real_price
                r['corrected_gap_pct']    = corrected_gap
                try:
                    sell_venue = ensure_exchange(sell_ex)
                    ob = sell_venue.fetch_order_book(real_symbol, limit=order_book_limit_for(sell_ex), params=EXTRA_PARAMS.get(sell_ex, {}))
                    corrected_bids = _sort_book_side(ob.get('bids', []) or [], 'bids')
                    corrected_sell_depth, _, _ = walk_book(corrected_bids, DEPTH_CHECK_USD)
                    if corrected_sell_depth and r.get('buy_depth_price'):
                        r['corrected_depth_gap_pct']    = ((corrected_sell_depth - r['buy_depth_price']) / r['buy_depth_price']) * 100
                        r['corrected_sell_depth_price'] = corrected_sell_depth
                except Exception as e:
                    log.warning(f"  WARNING  corrected depth check {real_symbol} on {sell_ex}: {str(e)[:400]}")
                sell_cur  = real_cur
                sell_name = (real_cur.get('name') or '').strip()
                r['sell_name'] = sell_name
                r['name_match'], r['name_suspect'] = True, False

    buy_networks  = buy_cur.get('networks')  or {}
    sell_networks = sell_cur.get('networks') or {}
    if not buy_networks:
        buy_networks = _fallback_networks(buy_ex, base)
    if not sell_networks:
        sell_networks = _fallback_networks(sell_ex, base)
    r['networks_known'] = bool(buy_networks) and bool(sell_networks)
    r['buy_meta_summary']  = f"{buy_ex}: name={buy_name!r}  networks: {_fmt_networks(buy_networks)}"
    r['sell_meta_summary'] = f"{sell_ex}: name={sell_name!r}  networks: {_fmt_networks(sell_networks)}"

    sell_contracts_by_addr = defaultdict(list)
    for s_code, s_data in sell_networks.items():
        c = _extract_contract(s_data, sell_cur)
        if c:
            sell_contracts_by_addr[c.lower()].append((s_code, s_data))
    contract_matches = []
    for b_code, b_data in buy_networks.items():
        c = _extract_contract(b_data, buy_cur)
        if not c:
            continue
        for s_code, s_data in sell_contracts_by_addr.get(c.lower(), []):
            contract_matches.append((b_code, b_data, s_code, s_data, c))

    buy_by_norm = defaultdict(list)
    for code, data in buy_networks.items():
        buy_by_norm[normalize_network(code)].append((code, data))
    sell_by_norm = defaultdict(list)
    for code, data in sell_networks.items():
        sell_by_norm[normalize_network(code)].append((code, data))
    common = []
    blocked = []
    blocked_details = []
    seen_pairs = set()
    for norm_key, buy_entries in buy_by_norm.items():
        sell_entries = sell_by_norm.get(norm_key)
        if not sell_entries:
            continue
        buy_code, buy_data   = buy_entries[0]
        sell_code, sell_data = sell_entries[0]
        seen_pairs.add((buy_code, sell_code))
        can_withdraw = buy_data.get('withdraw', True) is not False and buy_data.get('active', True) is not False
        can_deposit  = sell_data.get('deposit', True) is not False and sell_data.get('active', True) is not False
        if can_withdraw and can_deposit:
            common.append((norm_key, buy_code, buy_data, sell_code, sell_data))
        else:
            reasons = []
            if not can_withdraw:
                reasons.append(f"{buy_ex} withdraw disabled for {buy_code}")
            if not can_deposit:
                reasons.append(f"{sell_ex} deposit disabled for {sell_code}")
            blocked.append(f"{norm_key} ({', '.join(reasons)})")
            blocked_details.append({
                'norm_key':     norm_key,
                'buy_code':     buy_code,
                'buy_data':     buy_data,
                'sell_code':    sell_code,
                'can_withdraw': can_withdraw,
                'can_deposit':  can_deposit,
                'other_blocked': (
                    other_deposit_blocks(base, norm_key, {buy_ex, sell_ex}, r['all_prices'])
                    if can_withdraw and not can_deposit else []
                ),
            })

    for buy_code, buy_data, sell_code, sell_data, contract in contract_matches:
        if (buy_code, sell_code) in seen_pairs:
            continue
        seen_pairs.add((buy_code, sell_code))
        label = f"{buy_code}→{sell_code} [contract match]" if buy_code != sell_code else f"{buy_code} [contract match]"
        can_withdraw = buy_data.get('withdraw', True) is not False and buy_data.get('active', True) is not False
        can_deposit  = sell_data.get('deposit', True) is not False and sell_data.get('active', True) is not False
        if can_withdraw and can_deposit:
            common.append((label, buy_code, buy_data, sell_code, sell_data))
        else:
            reasons = []
            if not can_withdraw:
                reasons.append(f"{buy_ex} withdraw disabled for {buy_code}")
            if not can_deposit:
                reasons.append(f"{sell_ex} deposit disabled for {sell_code}")
            blocked.append(f"{label} ({', '.join(reasons)})")
            blocked_details.append({
                'norm_key':      label,
                'buy_code':      buy_code,
                'buy_data':      buy_data,
                'sell_code':     sell_code,
                'can_withdraw':  can_withdraw,
                'can_deposit':   can_deposit,
                'other_blocked': [],
            })

    r['common_networks'] = [c[0] for c in common]
    r['blocked_networks'] = blocked
    r['blocked_details'] = blocked_details

    if contract_matches:
        r['contract_buy']   = contract_matches[0][4]
        r['contract_sell']  = contract_matches[0][4]
        r['contract_match'] = True

    r['alt_route'] = None
    if not common and blocked_details and r['n_exchanges'] >= 3:
        r['alt_route'] = evaluate_network_block(
            base, buy_ex, sell_ex, r['buy_price'], r['sell_price'], r['all_prices'],
            blocked_details, contract=r['contract_buy'], symbol=r['symbol']
        )

    if common:
        norm_key, buy_code, buy_data, sell_code, sell_data = min(
            common,
            key=lambda c: _to_float(c[2].get('fee')) if c[2].get('fee') is not None else float('inf')
        )
        r['withdrawal_network'] = (
            norm_key if ('[contract match]' in norm_key or buy_code == sell_code)
            else f"{norm_key} ({buy_code}→{sell_code})"
        )
        r['withdrawal_fee']        = _to_float(buy_data.get('fee'))
        r['withdrawal_min_tokens'] = _to_float(((buy_data.get('limits') or {}).get('withdraw') or {}).get('min'))
        if r['contract_match'] is None:
            contract_buy  = _extract_contract(buy_data, buy_cur)
            contract_sell = _extract_contract(sell_data, sell_cur)
            r['contract_buy']  = contract_buy
            r['contract_sell'] = contract_sell
            if contract_buy and contract_sell:
                r['contract_match'] = contract_buy.lower() == contract_sell.lower()

    r['buy_meta_summary']  += f"  contract={r['contract_buy']}"
    r['sell_meta_summary'] += f"  contract={r['contract_sell']}"
    if r['contract_match'] is True:
        r['verified'] = bool(common)
    elif r['contract_match'] is False:
        r['verified'] = False
    elif r['name_match'] is False:
        r['verified'] = False
    elif r['name_suspect']:
        r['verified'] = None
    elif common and r['name_match'] is not False:
        r['verified'] = True
    elif not r['networks_known']:
        r['verified'] = None
    else:
        r['verified'] = False
    r['confirmed_pair'] = None
    if r['name_suspect'] and r['verified'] is not True and r['n_exchanges'] >= 3:
        r['confirmed_pair'] = find_confirmed_pair(base, r['all_prices'])
    return r

def apply_metadata_checks(results):
    if not results:
        return results
    with ThreadPoolExecutor(max_workers=min(len(results), 6)) as pool:
        futures = {pool.submit(check_metadata, r): r for r in results}
        for future in as_completed(futures):
            r = futures[future]
            try:
                future.result()
            except Exception as e:
                log.warning(f"  WARNING  metadata check {r.get('symbol')} ({r.get('buy_ex')}->{r.get('sell_ex')}): {type(e).__name__}: {str(e)[:200]}")
                r['meta_checked'] = True
                r['verified'] = None
    return results

def get_trading_fee_rate(exchange_name, symbol):
    ex = ensure_exchange(exchange_name)
    if ex is None:
        return DEFAULT_TAKER_FEE
    market = (getattr(ex, 'markets', None) or {}).get(symbol)
    if market and market.get('taker') is not None:
        return market['taker']
    fees    = getattr(ex, 'fees', {}) or {}
    trading = fees.get('trading') or {}
    taker   = trading.get('taker')
    if taker is not None:
        return taker
    return DEFAULT_TAKER_FEE

def _is_confirmed_tradable(r):
    if not r.get('meta_checked'):
        return False
    if r['verified'] is True:
        return True
    if r['verified'] is False and r.get('alt_route'):
        return True
    if r['verified'] is None:
        cp = r.get('confirmed_pair')
        if cp and cp.get('tradable_network'):
            return True
    return False

def fmt_price(p):
    if   p >= 1:      return f"${p:,.4f}"
    elif p >= 0.0001: return f"${p:.6f}"
    else:             return f"${p:.8f}"

def fmt_vol(v):
    if   v >= 1_000_000: return f"${v/1_000_000:.2f}M"
    elif v >= 1_000:     return f"${v/1_000:.1f}K"
    else:                return f"${v:.0f}"

def calc_arb_profit(capital_usd, buy_price, sell_price, fee_tokens=None,
                    min_withdrawal_tokens=None,
                    buy_taker_rate=0.0, sell_taker_rate=0.0):
    if not buy_price or buy_price <= 0 or not sell_price or sell_price <= 0:
        return None

    buy_taker_rate       = buy_taker_rate  or 0.0
    sell_taker_rate      = sell_taker_rate or 0.0
    fee_tokens           = _to_float(fee_tokens)
    min_withdrawal_tokens = _to_float(min_withdrawal_tokens)

    tokens_bought = capital_usd / buy_price
    buy_fee_usd   = capital_usd * buy_taker_rate

    min_withdrawal_met = (
        tokens_bought >= min_withdrawal_tokens
        if min_withdrawal_tokens is not None and min_withdrawal_tokens > 0
        else True
    )

    gas_tokens       = fee_tokens if fee_tokens is not None else 0.0
    tokens_remaining = tokens_bought - gas_tokens

    gross_sell_usd = tokens_remaining * sell_price if tokens_remaining > 0 else 0.0
    sell_fee_usd   = gross_sell_usd * sell_taker_rate
    total_received = gross_sell_usd - sell_fee_usd

    total_cost = capital_usd
    net_pnl    = total_received - total_cost
    roi_pct    = (net_pnl / capital_usd) * 100 if capital_usd else 0.0

    return {
        'capital':               capital_usd,
        'tokens_bought':          tokens_bought,
        'buy_fee_usd':            buy_fee_usd,
        'buy_taker_rate':         buy_taker_rate,
        'min_withdrawal_tokens':  min_withdrawal_tokens,
        'min_withdrawal_met':     min_withdrawal_met,
        'gas_tokens':             gas_tokens,
        'tokens_remaining':       tokens_remaining,
        'sell_fee_usd':           sell_fee_usd,
        'sell_taker_rate':        sell_taker_rate,
        'total_cost':             total_cost,
        'total_received':         total_received,
        'net_pnl':                net_pnl,
        'roi_pct':                roi_pct,
    }

def log_profit_block(profit):
    if not profit:
        return
    log.info(f"       PROFIT Capital= {profit['capital']:.0f}USDT")
    log.info(f"       PROFIT Tokens bought= {profit['tokens_bought']:.6f}")
    if profit['buy_taker_rate']:
        log.info(f"       PROFIT Buy trading fee=  -{profit['buy_fee_usd']:.4f}")
    if profit['min_withdrawal_tokens'] is not None:
        mark = "✅" if profit['min_withdrawal_met'] else "❌"
        note = "met" if profit['min_withdrawal_met'] else "NOT MET — trade size too small to withdraw"
        log.info(f"       PROFIT Min withdrawal= {profit['min_withdrawal_tokens']:.6f} tokens  [{mark} {note}]")
    log.info(f"       PROFIT Gas deducted= -{profit['gas_tokens']:.6f}")
    log.info(f"       PROFIT Tokens remaining= {profit['tokens_remaining']:.6f}")
    if profit['sell_taker_rate']:
        log.info(f"       PROFIT Sell trading fee=  -{profit['sell_fee_usd']:.4f}")
    log.info(f"       PROFIT Total cost= {profit['total_cost']:.4f} USDT")
    log.info(f"       PROFIT Total received= {profit['total_received']:.4f} USDT")
    log.info(f"       PROFIT Net P&L {profit['net_pnl']:+.4f} USDT")
    log.info(f"       PROFIT ROI {profit['roi_pct']:+.2f}%")

def store_arb_coin(symbol, real_gap_pct, buy_ex, buy_price, sell_ex, sell_price, profit_usdt, network):
    try:
        sb = _get_supabase_cached()
        existing = sb.table("arb_coins").select("id").eq("symbol", symbol).limit(1).execute()
        if existing.data:
            return  # already stored — skip duplicate
        sb.table("arb_coins").insert({
            "symbol":        symbol,
            "real_gap_pct":  round(real_gap_pct, 4) if real_gap_pct is not None else None,
            "buy_exchange":  buy_ex,
            "buy_price":     buy_price,
            "sell_exchange": sell_ex,
            "sell_price":    sell_price,
            "profit_usdt":   round(profit_usdt, 4),
            "network":       network,
        }).execute()
        log.info(f"       DB     ✅ stored {symbol} in arb_coins  |  profit {profit_usdt:+.4f} USDT  |  network {network}")
    except Exception as e:
        # Covers connectivity issues and duplicate-key races (unique constraint on symbol)
        log.warning(f"  WARNING  arb_coins store for {symbol}: {str(e)[:200]}")

def maybe_store_arb_coin(symbol, real_gap_pct, buy_ex, buy_price, sell_ex, sell_price, profit, network):
    if not profit or profit['net_pnl'] < MIN_STORE_PROFIT_USD:
        return
    network_plain = (network or '').split(' (')[0]  # drop "(buy_code→sell_code)" suffix if present
    store_arb_coin(symbol, real_gap_pct, buy_ex, buy_price, sell_ex, sell_price, profit['net_pnl'], network_plain)

def print_results(results, scan_num, duration, counts):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("")
    log.info(f"SCAN #{scan_num}  {now}  ({duration:.1f}s)")
    log.info("-" * 60)
    for name, count in counts.items():
        log.info(f"  {name:<10} {'FAILED' if count == 0 else f'{count} pairs'}")
    log.info("-" * 60)
    if not results:
        log.info("  No opportunities found this scan.")
        log.info("")
        return
    confirmed = [r for r in results if _is_confirmed_tradable(r)]
    if LOG_CONFIRMED_ONLY:
        log.info(
            f"  {len(confirmed)} confirmed tradable (out of {len(results)} found)  "
            f"(gap {MIN_GAP_PERCENT}%-{MAX_GAP_PERCENT}%  vol >${MIN_VOLUME_USDT//1000}K)"
        )
    else:
        log.info(f"  {len(results)} found  ({len(confirmed)} confirmed tradable)  "
                 f"(gap {MIN_GAP_PERCENT}%-{MAX_GAP_PERCENT}%  vol >${MIN_VOLUME_USDT//1000}K)")
    log.info(f"  Depth check: all {len(results)} verified against ${DEPTH_CHECK_USD} of real order-book depth")
    log.info(f"  Metadata check: all {len(results)} verified for name/network/contract match")
    log.info("")
    to_display = confirmed if LOG_CONFIRMED_ONLY else results
    for i, r in enumerate(to_display, 1):
        log.info("-" * 60)
        log.info(f"  #{i}  {r['symbol']}  |  gap {r['gap_pct']:.2f}%  |  {r['n_exchanges']} exchanges")
        log.info(f"       BUY   {r['buy_ex']:<10}  ask {fmt_price(r['buy_price'])}  vol {fmt_vol(r['buy_vol'])}  24h {r['buy_chg']:+.1f}%")
        log.info(f"       SELL  {r['sell_ex']:<10}  bid {fmt_price(r['sell_price'])}  vol {fmt_vol(r['sell_vol'])}  24h {r['sell_chg']:+.1f}%")
        prices = "  ".join(
            f"{ex}:{fmt_price(p['price'])}"
            for ex, p in sorted(r['all_prices'].items())
        )
        log.info(f"       ALL   {prices}")
        if r['depth_checked']:
            if r['buy_depth_price'] is None:
                log.info(f"       DEPTH  insufficient liquidity to verify — SKIP")
            else:
                status = "OK" if r['depth_ok'] else f"THIN (book ran out before ${DEPTH_CHECK_USD} filled)"
                log.info(
                    f"       DEPTH  buy avg {fmt_price(r['buy_depth_price'])}  "
                    f"sell avg {fmt_price(r['sell_depth_price'])}  "
                    f"real gap {r['depth_gap_pct']:.2f}%  [{status}]"
                )
        else:
            log.info(f"       DEPTH  not checked (exchange unavailable)")
        if r['meta_checked']:
            if r['corrected_symbol']:
                depth_note = (
                    f"  |  depth-adjusted: {r['corrected_depth_gap_pct']:.2f}%"
                    if r['corrected_depth_gap_pct'] is not None else ""
                )
                log.info(
                    f"       NOTE   Ticker collision resolved by name — '{r['buy_name']}' actually trades as "
                    f"{r['corrected_symbol']} on {r['sell_ex']}. Re-priced sell: {fmt_price(r['corrected_sell_price'])}  "
                    f"corrected gap: {r['corrected_gap_pct']:.2f}%{depth_note}  "
                    f"(the DEPTH line above was computed against the ORIGINAL, wrong-token pairing — ignore it, use this instead)"
                )
            if r['verified'] is True:
                basis = "contract address match" if r['contract_match'] is True else f"name '{r['buy_name']}' on both"
                min_wd = r['withdrawal_min_tokens']
                log.info(
                    f"       VERIFY  ✅ confirmed via {basis}  |  "
                    f"network {r['withdrawal_network']}  fee {r['withdrawal_fee']}  "
                    f"min_withdraw {min_wd if min_wd is not None else 'unknown'}"
                )
                buy_fee_rate  = get_trading_fee_rate(r['buy_ex'],  r['symbol'])
                sell_fee_rate = get_trading_fee_rate(r['sell_ex'], r['symbol'])
                profit = calc_arb_profit(
                    CAPITAL_USD, r['buy_price'], r['sell_price'],
                    fee_tokens=r['withdrawal_fee'],
                    min_withdrawal_tokens=r['withdrawal_min_tokens'],
                    buy_taker_rate=buy_fee_rate,
                    sell_taker_rate=sell_fee_rate,
                )
                log_profit_block(profit)
                real_gap = r['depth_gap_pct'] if r['depth_gap_pct'] is not None else r['gap_pct']
                maybe_store_arb_coin(
                    r['symbol'], real_gap, r['buy_ex'], r['buy_price'], r['sell_ex'], r['sell_price'],
                    profit, r['withdrawal_network'],
                )
            elif r['verified'] is False and r['alt_route']:
                ar = r['alt_route']
                n_plural = "exchange" if r['n_exchanges'] == 1 else "exchanges"
                log.info(f"       VERIFY  ✅ Same token existed on {r['n_exchanges']} {n_plural}.")
                log.info(f"       VERIFY  ❌ {ar['blocked_reason']}")
                if ar['direction'] == 'alt_buy':
                    log.info(f"       VERIFY  ✅ {ar['alt_ex']} withdraw allowed in {ar['network']}  |  ask {fmt_price(ar['alt_price'])}  |  gap {ar['gap_pct']:.2f}%")
                    log.info(f"       VERIFY  📉 Tradable at {ar['alt_ex']} → {ar['sell_ex']} | {ar['network']} 🚀")
                    buy_fee_rate  = get_trading_fee_rate(ar['alt_ex'],  r['symbol'])
                    sell_fee_rate = get_trading_fee_rate(r['sell_ex'],   r['symbol'])
                    profit = calc_arb_profit(
                        CAPITAL_USD, ar['alt_price'], r['sell_price'],
                        fee_tokens=ar.get('fee'),
                        min_withdrawal_tokens=ar.get('min_withdrawal'),
                        buy_taker_rate=buy_fee_rate,
                        sell_taker_rate=sell_fee_rate,
                    )
                    log_profit_block(profit)
                    maybe_store_arb_coin(
                        r['symbol'], ar['gap_pct'], ar['alt_ex'], ar['alt_price'], r['sell_ex'], r['sell_price'],
                        profit, ar['network'],
                    )
                else:
                    log.info(f"       VERIFY  ✅ {ar['alt_ex']} deposit allowed in {ar['network']}  |  bid {fmt_price(ar['alt_price'])}  |  gap {ar['gap_pct']:.2f}%")
                    log.info(f"       VERIFY  📉 Tradable at {ar['buy_ex']} → {ar['alt_ex']} | {ar['network']} 🚀")
                    buy_fee_rate  = get_trading_fee_rate(r['buy_ex'],   r['symbol'])
                    sell_fee_rate = get_trading_fee_rate(ar['alt_ex'],  r['symbol'])
                    profit = calc_arb_profit(
                        CAPITAL_USD, r['buy_price'], ar['alt_price'],
                        fee_tokens=ar.get('fee'),
                        min_withdrawal_tokens=ar.get('min_withdrawal'),
                        buy_taker_rate=buy_fee_rate,
                        sell_taker_rate=sell_fee_rate,
                    )
                    log_profit_block(profit)
                    maybe_store_arb_coin(
                        r['symbol'], ar['gap_pct'], r['buy_ex'], r['buy_price'], ar['alt_ex'], ar['alt_price'],
                        profit, ar['network'],
                    )
            elif r['verified'] is False:
                if r['name_match'] is False:
                    log.info(f"       VERIFY  ❌ Name mismatch ('{r['buy_name']}' vs '{r['sell_name']}') — likely different tokens sharing a symbol")
                elif r['contract_match'] is False:
                    log.info(f"       VERIFY  ❌ Contract mismatch ({r['contract_buy']} vs {r['contract_sell']})")
                elif r['blocked_details']:
                    log.info(f"       VERIFY  ✅ Same network exists on both sides")
                    withdraw_blocked = False
                    for bd in r['blocked_details']:
                        if not bd['can_withdraw']:
                            withdraw_blocked = True
                            log.info(f"       VERIFY  ❌ {r['buy_ex']} withdraw disabled for {bd['buy_code']}")
                        if not bd['can_deposit']:
                            log.info(f"       VERIFY  ❌ {r['sell_ex']} deposit disabled for {bd['sell_code']}")
                            for other_ex, other_code in bd['other_blocked']:
                                log.info(f"       VERIFY  ❌ {other_ex} deposit disabled for {other_code}")
                    if withdraw_blocked:
                        log.info(f"       VERIFY  🚫 Not tradable — no working transfer path, {r['buy_ex']} withdrawal is disabled.")
                    else:
                        log.info(f"       VERIFY  🚫 Not tradable — no working transfer path, all deposits are disabled.")
                else:
                    log.info(f"       VERIFY  ❌ No matching network between {r['buy_ex']} and {r['sell_ex']}")
            else:
                log.info(f"       VERIFY  ⚠️ Unverified — no contract ID to confirm match")
                if r['name_suspect']:
                    log.info(f"       VERIFY  ⚠️ {r['buy_ex']}='{r['buy_name']}' vs {r['sell_ex']}='{r['sell_name']}' — verify manually")
                cp = r['confirmed_pair']
                if cp:
                    log.info(f"       VERIFY  ✅ Confirmed same token: {cp['buy_ex']} & {cp['sell_ex']} (name match)  |  gap {cp['gap_pct']:.2f}%")
                    if cp['tradable_network']:
                        log.info(f"       VERIFY  ✅ Tradable via {cp['buy_ex']} → {cp['sell_ex']} | network {cp['tradable_network']} 🚀")
                        buy_fee_rate  = get_trading_fee_rate(cp['buy_ex'],  r['symbol'])
                        sell_fee_rate = get_trading_fee_rate(cp['sell_ex'], r['symbol'])
                        profit = calc_arb_profit(
                            CAPITAL_USD, cp['buy_price'], cp['sell_price'],
                            fee_tokens=cp.get('fee'),
                            min_withdrawal_tokens=cp.get('min_withdrawal'),
                            buy_taker_rate=buy_fee_rate,
                            sell_taker_rate=sell_fee_rate,
                        )
                        log_profit_block(profit)
                        maybe_store_arb_coin(
                            r['symbol'], cp['gap_pct'], cp['buy_ex'], cp['buy_price'], cp['sell_ex'], cp['sell_price'],
                            profit, cp['tradable_network'],
                        )
                    else:
                        for msg in cp['blocked_msgs']:
                            log.info(f"       VERIFY  ❌ {msg}")
                        log.info(f"       VERIFY  🚫 Not tradable — no working transfer path between confirmed exchanges")
            if r['buy_meta_summary']:
                log.info(f"       META   {r['buy_meta_summary']}")
            if r['sell_meta_summary']:
                log.info(f"       META   {r['sell_meta_summary']}")
            for extra_summary in r['extra_meta_summaries']:
                log.info(f"       META   {extra_summary}")
        else:
            log.info(f"       VERIFY  not checked (unexpected error — see warnings above)")
        log.info("")
    log.info("-" * 60)
    log.info(f"  Next scan in {SCAN_INTERVAL_SEC}s  |  Ctrl+C to stop")
    log.info("")

def scan_once(scan_num):
    start = time.time()
    all_tickers = {}
    counts      = {}
    with ThreadPoolExecutor(max_workers=len(EXCHANGE_BUILDERS)) as pool:
        futures = {pool.submit(get_usdt_tickers, name): name for name in EXCHANGE_BUILDERS}
        for future in as_completed(futures):
            name = futures[future]
            try:
                tickers = future.result()
            except Exception as e:
                log.warning(f"  WARNING  {name}: {str(e)[:400]}")
                tickers = {}
            all_tickers[name] = tickers
            counts[name]      = len(tickers)
    price_map = build_price_map(all_tickers)
    results   = find_opportunities(price_map)
    results   = apply_depth_checks(results)
    results   = apply_metadata_checks(results)
    duration  = time.time() - start
    print_results(results, scan_num, duration, counts)

def main():
    log.info("Crypto Arb Scanner started")
    log.info(f"Exchanges : {', '.join(EXCHANGE_BUILDERS.keys())}")
    log.info(f"Gap range : {MIN_GAP_PERCENT}% - {MAX_GAP_PERCENT}%")
    log.info(f"Min volume: ${MIN_VOLUME_USDT:,} USDT")
    log.info(f"Price cap : {MAX_PRICE_RATIO}x ratio between exchanges")
    log.info(f"Capital   : ${CAPITAL_USD} per trade (profit calc)")
    log.info(f"Taker fee : {DEFAULT_TAKER_FEE*100:.2f}% fallback (live rates fetched from exchange)")
    log.info(f"Log mode  : {'confirmed tradable only' if LOG_CONFIRMED_ONLY else 'all arb opportunities'}")
    log.info(f"Depth chk : ${DEPTH_CHECK_USD} on all candidates found")
    log.info(f"Meta chk  : name/network/contract on all candidates found")
    log.info("")
    init_exchanges()
    scan_num = 1
    while True:
        try:
            scan_once(scan_num)
            scan_num += 1
            time.sleep(SCAN_INTERVAL_SEC)
        except KeyboardInterrupt:
            log.info("Bot stopped.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            log.info("Retrying in 60s...")
            time.sleep(60)

if __name__ == "__main__":
    main()
