"""
historical_engine.py — 3‑candle pattern strategy (5‑minute candles).

Compares the latest three completed 5‑minute candles against historical patterns
(~1 year) and emits a LONG/SHORT signal if at least 70% of similar historical
patterns were followed by a meaningful, sustained trend lasting >3 candles.
"""

from __future__ import annotations
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from market_data import MarketDataStore, TradeStore, Signal, DEFAULT_SYMBOL_WHITELIST
from .base import StrategyContext, StrategyEngine

log = logging.getLogger("okx_futures.historical")
BASE = "https://www.okx.com"
HISTORY_CANDLES_PATH = "/api/v5/market/history-candles"

REQUIRED_TRADE_WINDOW_MS = 900_000

# Maps candle_interval_sec -> OKX 'bar' query param value.
_BAR_BY_INTERVAL_SEC = {
    60: "1m", 180: "3m", 300: "5m", 900: "15m", 1800: "30m",
    3600: "1H", 7200: "2H", 14400: "4H", 21600: "6H", 43200: "12H", 86400: "1D",
}


def bar_from_interval(interval_sec: int) -> str:
    return _BAR_BY_INTERVAL_SEC.get(interval_sec, "5m")


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class Candle:
    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class HistoricalEngineConfig:
    # Data download
    history_days: int = 365
    history_cache_dir: str = ".historical_cache"
    history_refresh_hours: float = 24.0
    request_timeout_sec: float = 20.0
    request_delay_sec: float = 0.12
    max_history_requests_per_refresh: int = 0
    require_requested_history: bool = False
    retry_cooldown_sec: float = 300.0   # 5 minutes after failure

    # Candles
    candle_interval_sec: int = 300           # 5 minutes
    live_window_ms: int = 900_000
    min_live_warmup_sec: float = 45.0
    min_live_trade_count: int = 20

    # Pattern matching
    pattern_length: int = 3
    min_historical_matches: int = 10
    min_directional_agreement: float = 0.70
    min_forward_trend_candles: int = 4
    min_forward_move_pct: float = 0.003
    min_forward_directional_ratio: float = 0.60
    min_pattern_similarity: float = 0.85
    min_candle_range_atr_multiplier: float = 0.5
    min_candle_body_atr_multiplier: float = 0.3
    atr_period: int = 14

    # Signal gate
    cooldown_sec: float = 60.0
    max_observation_minutes: float = 6.0
    symbol_whitelist: Optional[frozenset] = field(
        default_factory=lambda: DEFAULT_SYMBOL_WHITELIST
    )
    log_top_matches: int = 3


@dataclass
class Candidate:
    symbol: str
    started_at: float = field(default_factory=time.time)
    status: str = "OBSERVING"
    last_checked_at: float = 0.0
    data_ready: bool = False
    direction: str = ""

    @property
    def elapsed_sec(self):
        return time.time() - self.started_at


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def f(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def timestamp(v):
    x = f(v, float("nan"))
    if math.isfinite(x):
        return x / 1000 if x > 10_000_000_000 else x
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def side_of(item):
    s = str(item.get("side", "")).lower()
    if s in ("buy", "sell"):
        return s
    m = item.get("m")
    if isinstance(m, bool):
        return "sell" if m else "buy"
    if str(m).lower() in ("true", "1"):
        return "sell"
    if str(m).lower() in ("false", "0"):
        return "buy"
    try:
        w = int(item.get("way"))
        if 1 <= w <= 4:
            return "buy"
        if 5 <= w <= 8:
            return "sell"
    except Exception:
        pass
    return None


def okx_get(path, params, timeout):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "historical-engine/1.0", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return data


async def download_candles(symbol, cfg):
    """
    Download completed OHLCV candles directly from OKX's historical
    candlestick endpoint (/api/v5/market/history-candles), paginating
    backward in time until cfg.history_days of coverage is collected or
    OKX has no older data left.

    Returns (list[Candle] sorted oldest->newest, request_count).

    PAGINATION:
    OKX returns history-candles newest-first within each batch. Per the
    documented v5 semantics for this endpoint, `after=<ts_ms>` means
    "return records EARLIER than <ts_ms>" (moves backward in time), while
    `before=<ts_ms>` means "return records NEWER than <ts_ms>" (moves
    forward, and is not usable to reach older data). To walk backward
    through history we repeatedly set `after` to the timestamp of the
    OLDEST candle in the previous batch.
    See: https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-candlesticks-history
    """
    bar = bar_from_interval(cfg.candle_interval_sec)
    end = time.time()
    start = end - cfg.history_days * 86400

    candles_by_ts: Dict[float, Candle] = {}
    requests = 0
    consecutive_errors = 0
    max_consecutive_errors = 5
    after_cursor = None      # ts (seconds) cursor: next batch must be OLDER than this
    prev_cursor = None
    prev_oldest_ts = None
    last_progress_log = 0.0

    log.info("[historical] %s downloading 5m candles...", symbol)
    log.debug("[historical] %s   source = OKX history-candles bar=%s", symbol, bar)
    log.debug("[historical] %s   requested_history_days = %.1f", symbol, cfg.history_days)

    while True:
        if cfg.max_history_requests_per_refresh and requests >= cfg.max_history_requests_per_refresh:
            log.warning("[historical] %s download stopped: reached max requests (%d)", symbol, requests)
            break

        params = {"instId": symbol, "bar": bar, "limit": "100"}
        if after_cursor is not None:
            params["after"] = str(int(after_cursor * 1000))

        log.debug("[historical] %s API REQUEST endpoint=%s instId=%s bar=%s after=%s",
                  symbol, HISTORY_CANDLES_PATH, symbol, bar, after_cursor if after_cursor else "<none>")

        try:
            response = await asyncio.to_thread(
                okx_get,
                HISTORY_CANDLES_PATH,
                params,
                cfg.request_timeout_sec
            )
        except Exception as e:
            consecutive_errors += 1
            log.error("[historical] %s API request failed (%d/%d): %s",
                      symbol, consecutive_errors, max_consecutive_errors, e)
            if consecutive_errors >= max_consecutive_errors:
                log.error("[historical] %s too many consecutive errors, aborting download", symbol)
                break
            await asyncio.sleep(max(1.0, cfg.request_delay_sec * 10))
            continue

        requests += 1
        code = response.get("code")
        msg = response.get("msg")
        data = response.get("data", [])

        log.debug("[historical] %s API RESPONSE okx_code=%s okx_msg=%s data_count=%d",
                  symbol, code, msg, len(data))

        if code != "0":
            consecutive_errors += 1
            log.error("[historical] %s API error (%d/%d): code=%s msg=%s",
                      symbol, consecutive_errors, max_consecutive_errors, code, msg)
            if consecutive_errors >= max_consecutive_errors:
                log.error("[historical] %s too many consecutive API errors, aborting download", symbol)
                break
            await asyncio.sleep(max(1.0, cfg.request_delay_sec * 10))
            continue

        consecutive_errors = 0

        if not data:
            log.info("[historical] %s no older data available from OKX — download complete", symbol)
            break

        # Parse this batch: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        batch = []
        for row in data:
            try:
                ts = f(row[0]) / 1000.0
                o = f(row[1])
                h = f(row[2])
                l = f(row[3])
                c = f(row[4])
                vol = f(row[5])
                confirm = str(row[8]) if len(row) > 8 else "1"
            except (IndexError, TypeError, ValueError):
                continue
            if confirm != "1":
                continue  # skip the still-forming candle
            if ts <= 0 or o <= 0 or h <= 0 or l <= 0 or c <= 0:
                continue
            batch.append((ts, Candle(ts=ts, open=o, high=h, low=l, close=c, volume=vol)))

        if not batch:
            log.warning("[historical] %s batch had no valid candles, stopping", symbol)
            break

        # OKX returns candles newest-first within a batch.
        batch_newest_ts = batch[0][0]
        batch_oldest_ts = batch[-1][0]
        next_cursor = batch_oldest_ts

        # --- Pagination safety check: cursor must move strictly backward ---
        stalled = (prev_cursor is not None and next_cursor >= prev_cursor) or \
                  (prev_oldest_ts is not None and batch_oldest_ts >= prev_oldest_ts)
        if stalled:
            log.error("[historical] %s pagination stalled (prev_cursor=%s next_cursor=%s), stopping",
                      symbol, prev_cursor, next_cursor)
            break

        for ts, candle in batch:
            if start <= ts <= end:
                candles_by_ts[ts] = candle

        prev_cursor = next_cursor
        prev_oldest_ts = batch_oldest_ts
        after_cursor = next_cursor

        # Periodic progress at DEBUG only — INFO stays limited to start/finish (see log design below).
        now_t = time.time()
        if candles_by_ts and (now_t - last_progress_log >= 5.0 or requests % 25 == 0):
            oldest_ts = min(candles_by_ts)
            newest_ts = max(candles_by_ts)
            coverage_days = (newest_ts - oldest_ts) / 86400
            log.debug("[historical] %s progress: %d candles, coverage=%.1f days",
                      symbol, len(candles_by_ts), coverage_days)
            last_progress_log = now_t

        if batch_oldest_ts <= start:
            break

        await asyncio.sleep(max(0, cfg.request_delay_sec))

    candles_sorted = [candles_by_ts[ts] for ts in sorted(candles_by_ts)]

    if candles_sorted:
        coverage_days = (candles_sorted[-1].ts - candles_sorted[0].ts) / 86400
        log.info("[historical] %s download finished: %d candles (%d requests, coverage=%.1f days)",
                 symbol, len(candles_sorted), requests, coverage_days)
    else:
        log.warning("[historical] %s download finished: zero candles returned", symbol)

    return candles_sorted, requests


def build_candles_from_trades(trades, interval_sec=300):
    """Aggregate trades into candles of given interval (default 5 minutes)."""
    if not trades:
        return []
    candles = []
    trades_sorted = sorted(trades, key=lambda x: x["ts"])
    start_ts = trades_sorted[0]["ts"]
    end_ts = trades_sorted[-1]["ts"]
    current_bucket_start = math.floor(start_ts / interval_sec) * interval_sec
    bucket_end = current_bucket_start + interval_sec
    bucket_open = None
    bucket_high = -float("inf")
    bucket_low = float("inf")
    bucket_close = None
    bucket_volume = 0.0
    for t in trades_sorted:
        ts = t["ts"]
        if ts >= bucket_end:
            if bucket_open is not None and bucket_close is not None:
                candles.append(Candle(
                    ts=current_bucket_start,
                    open=bucket_open,
                    high=bucket_high,
                    low=bucket_low,
                    close=bucket_close,
                    volume=bucket_volume
                ))
            while ts >= bucket_end:
                current_bucket_start += interval_sec
                bucket_end += interval_sec
            bucket_open = t["price"]
            bucket_high = t["price"]
            bucket_low = t["price"]
            bucket_close = t["price"]
            bucket_volume = t["qty"]
        else:
            bucket_high = max(bucket_high, t["price"])
            bucket_low = min(bucket_low, t["price"])
            bucket_close = t["price"]
            bucket_volume += t["qty"]
            if bucket_open is None:
                bucket_open = t["price"]
    if bucket_open is not None and bucket_close is not None:
        candles.append(Candle(
            ts=current_bucket_start,
            open=bucket_open,
            high=bucket_high,
            low=bucket_low,
            close=bucket_close,
            volume=bucket_volume
        ))
    return candles


def compute_atr(candles, period=14):
    if len(candles) < period:
        return None
    ranges = [c.high - c.low for c in candles[-period:]]
    return sum(ranges) / len(ranges)


def candle_significant(candle, atr, cfg):
    if atr is None or atr <= 0:
        return False
    candle_range = candle.high - candle.low
    candle_body = abs(candle.close - candle.open)
    return (candle_range >= cfg.min_candle_range_atr_multiplier * atr and
            candle_body >= cfg.min_candle_body_atr_multiplier * atr)


def pattern_features(candles):
    if len(candles) != 3:
        raise ValueError("Pattern must have exactly 3 candles")
    base = candles[0].open
    if base == 0:
        base = 1e-6
    vec = []
    for c in candles:
        vec.extend([c.open / base, c.high / base, c.low / base, c.close / base])
    return vec


def pattern_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        return 0.0
    diff_sq = sum((a - b) ** 2 for a, b in zip(vec1, vec2))
    dist = math.sqrt(diff_sq)
    sim = max(0.0, 1.0 - dist / 2.0)
    return min(1.0, sim)


def classify_forward_trend(candles, start_idx, cfg):
    total_candles = len(candles)
    pattern_end = start_idx + cfg.pattern_length
    if pattern_end + cfg.min_forward_trend_candles > total_candles:
        return "neutral"

    forward_candles = candles[pattern_end:pattern_end + cfg.min_forward_trend_candles]
    start_price = candles[pattern_end - 1].close
    end_price = forward_candles[-1].close
    net_move_pct = (end_price - start_price) / start_price if start_price else 0.0

    if abs(net_move_pct) < cfg.min_forward_move_pct:
        return "neutral"

    direction = 1 if net_move_pct > 0 else -1
    dir_count = 0
    for c in forward_candles:
        move = c.close - c.open
        if (move > 0 and direction > 0) or (move < 0 and direction < 0):
            dir_count += 1
    ratio = dir_count / len(forward_candles)

    if ratio < cfg.min_forward_directional_ratio:
        return "neutral"

    return "bullish" if direction > 0 else "bearish"


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class Dataset:
    def __init__(self, candles: List[Candle], cfg: HistoricalEngineConfig):
        self.candles = candles
        self.cfg = cfg
        self.atr = compute_atr(candles, cfg.atr_period)
        self.patterns = []
        self._build_patterns()

    def _build_patterns(self):
        n = len(self.candles)
        if n < self.cfg.pattern_length + self.cfg.min_forward_trend_candles:
            return
        max_start = n - self.cfg.pattern_length - self.cfg.min_forward_trend_candles
        for i in range(max_start + 1):
            pattern_candles = self.candles[i:i+self.cfg.pattern_length]
            if self.atr is not None:
                if not all(candle_significant(c, self.atr, self.cfg) for c in pattern_candles):
                    continue
            vec = pattern_features(pattern_candles)
            self.patterns.append((i, vec))

    def find_matches(self, current_pattern_candles: List[Candle]) -> List[Tuple[int, float]]:
        if self.atr is None:
            return []
        if not all(candle_significant(c, self.atr, self.cfg) for c in current_pattern_candles):
            return []
        cur_vec = pattern_features(current_pattern_candles)
        matches = []
        for start_idx, hist_vec in self.patterns:
            sim = pattern_similarity(cur_vec, hist_vec)
            if sim >= self.cfg.min_pattern_similarity:
                matches.append((start_idx, sim))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def classify_forward(self, start_idx: int) -> str:
        return classify_forward_trend(self.candles, start_idx, self.cfg)


# ---------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------

class HistoricalEngine(StrategyEngine):
    name = "historical_engine"

    def __init__(self, trade_store, market_data, candle_fetcher=None, config=None):
        self._trade_store = trade_store
        self._market_data = market_data
        self._candle_fetcher = candle_fetcher
        self.config = config or HistoricalEngineConfig()
        self._candidates = {}
        self._datasets = {}
        self._ready = {}
        self._ready_at = {}  # symbol -> time.time() when dataset became ready (data-readiness clock)
        self._tasks = {}
        self._last_attempt = {}
        self._last_failure = {}
        self._last_signal = {}
        self._last_evaluated_ts = {}  # symbol -> last pattern candle timestamp (for avoiding repeated logs)
        self._cooldown_announced = {}  # symbol -> failure timestamp already logged as "cooldown started"
        self._lock = asyncio.Lock()

    def _cache_path(self, symbol):
        h = hashlib.sha1(symbol.encode()).hexdigest()[:10]
        return os.path.join(self.config.history_cache_dir, f"{symbol.replace('-', '_')}_{h}.csv")

    async def sync_watchlist(self, symbols):
        symbols = set(symbols)
        if self.config.symbol_whitelist:
            symbols &= set(self.config.symbol_whitelist)
        async with self._lock:
            for s in symbols:
                self._candidates.setdefault(s, Candidate(s))
            for s in list(self._candidates):
                if s not in symbols:
                    self._candidates.pop(s, None)

        now = time.time()
        for s in symbols:
            should_start = False
            if s not in self._tasks or self._tasks[s].done():
                last_fail = self._last_failure.get(s, 0)
                if now - last_fail < self.config.retry_cooldown_sec:
                    if self._cooldown_announced.get(s) != last_fail:
                        log.debug("[historical] %s retry cooldown started (duration=%.0fs)",
                                  s, self.config.retry_cooldown_sec)
                        self._cooldown_announced[s] = last_fail
                    continue
                should_start = True

            if should_start:
                if self._cooldown_announced.pop(s, None) is not None:
                    log.debug("[historical] %s retrying historical preparation", s)
                self._last_attempt[s] = now
                self._tasks[s] = asyncio.create_task(self._prepare(s))
                log.debug("[historical] %s preparation task CREATED", s)

    async def snapshot(self):
        async with self._lock:
            return list(self._candidates.values())

    async def _prepare(self, symbol):
        """Download 5m candles from OKX (or reuse cache), build the dataset."""
        cfg = self.config
        log.debug("[historical] %s preparation started", symbol)

        os.makedirs(cfg.history_cache_dir, exist_ok=True)
        cache_path = self._cache_path(symbol)

        try:
            if os.path.exists(cache_path):
                mtime = os.path.getmtime(cache_path)
                age_hours = (time.time() - mtime) / 3600
                if age_hours <= cfg.history_refresh_hours:
                    candles = self._load_candles(cache_path)
                    if candles:
                        ds = Dataset(candles, cfg)
                        self._datasets[symbol] = ds
                        self._ready[symbol] = True
                        self._ready_at[symbol] = time.time()
                        log.info("[historical] %s cache hit: %d candles ready (age=%.1fh)",
                                 symbol, len(candles), age_hours)
                        return
                    else:
                        log.warning("[historical] %s cache INVALID (corrupted/empty) — fresh download required", symbol)
                else:
                    log.debug("[historical] %s cache STALE (age=%.1fh > refresh=%.1fh) — fresh download required",
                              symbol, age_hours, cfg.history_refresh_hours)
            else:
                log.debug("[historical] %s cache MISS — fresh download required", symbol)
        except Exception as e:
            log.warning("[historical] %s cache INVALID (%s) — fresh download required", symbol, e)

        try:
            candles, requests = await download_candles(symbol, cfg)
        except Exception as e:
            log.error("[historical] %s PREPARATION FAILED: download_candles() raised exception", symbol, exc_info=True)
            self._ready[symbol] = False
            self._last_failure[symbol] = time.time()
            return

        if not candles:
            log.error("[historical] %s DOWNLOAD FAILED: no historical candles returned", symbol)
            self._ready[symbol] = False
            self._last_failure[symbol] = time.time()
            return

        coverage_days = (candles[-1].ts - candles[0].ts) / 86400
        log.debug("[historical] %s   candles=%d newest=%.0f oldest=%.0f coverage_days=%.1f target_days=%.1f",
                  symbol, len(candles), candles[-1].ts, candles[0].ts, coverage_days, cfg.history_days)

        if cfg.require_requested_history and coverage_days < cfg.history_days * 0.9:
            log.error("[historical] %s PREPARATION FAILED: insufficient coverage (%.1fd < %.1fd)",
                      symbol, coverage_days, cfg.history_days * 0.9)
            self._ready[symbol] = False
            self._last_failure[symbol] = time.time()
            return

        required_candles = cfg.pattern_length + cfg.min_forward_trend_candles
        if len(candles) < required_candles:
            log.error("[historical] %s PREPARATION FAILED: insufficient candles (need %d, got %d)",
                      symbol, required_candles, len(candles))
            self._ready[symbol] = False
            self._last_failure[symbol] = time.time()
            return

        self._save_candles(cache_path, candles)

        ds = Dataset(candles, cfg)
        self._datasets[symbol] = ds
        self._ready[symbol] = True
        self._ready_at[symbol] = time.time()
        log.info("[historical] %s dataset ready: %d candles (timeframe=%ds pattern_length=%d coverage=%.1fd)",
                 symbol, len(candles), cfg.candle_interval_sec, cfg.pattern_length, coverage_days)
        self._last_failure.pop(symbol, None)

    def _save_candles(self, path, candles):
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ts", "open", "high", "low", "close", "volume"])
            writer.writeheader()
            for c in candles:
                writer.writerow({
                    "ts": c.ts,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume
                })
        os.replace(tmp, path)

    @staticmethod
    def _load_candles(path):
        candles = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candles.append(Candle(
                    ts=f(row.get("ts")),
                    open=f(row.get("open")),
                    high=f(row.get("high")),
                    low=f(row.get("low")),
                    close=f(row.get("close")),
                    volume=f(row.get("volume"))
                ))
        return candles

    async def _get_current_pattern(self, symbol):
        if self._candle_fetcher is None:
            trades = await self._trade_store.get_window(symbol, self.config.live_window_ms)
            if len(trades) < 20:
                return None
            live_candles = build_candles_from_trades(trades, interval_sec=self.config.candle_interval_sec)
            if len(live_candles) < 3:
                return None
            return live_candles[-3:]
        try:
            raw = await self._candle_fetcher(symbol, "5m", 5)
        except Exception as e:
            log.warning("[historical] %s cannot fetch candles: %s", symbol, e)
            return None
        if not raw:
            return None
        completed = [c for c in raw if str(c.get("confirm", "1")) == "1"]
        if len(completed) < 3:
            return None
        completed_sorted = sorted(completed, key=lambda x: x.get("ts", 0))
        last_three = completed_sorted[-3:]
        candles = []
        for c in last_three:
            candles.append(Candle(
                ts=f(c.get("ts", 0)),
                open=f(c.get("open")),
                high=f(c.get("high")),
                low=f(c.get("low")),
                close=f(c.get("close")),
                volume=f(c.get("volume", 0))
            ))
        return candles

    async def evaluate(self, symbol):
        cfg = self.config
        c = self._candidates.get(symbol)
        if c is None:
            return None
        now = time.time()
        c.last_checked_at = now

        if not self._ready.get(symbol, False):
            # Historical dataset is still downloading/preparing. A 365-day 5m
            # download takes several minutes, so the observation window must
            # not start ticking until data is actually ready — otherwise the
            # candidate can expire before it ever gets a chance to evaluate.
            return None

        ready_at = self._ready_at.get(symbol, now)
        if now - ready_at >= cfg.max_observation_minutes * 60:
            c.status = "EXPIRED"
            async with self._lock:
                self._candidates.pop(symbol, None)
            return None

        ds = self._datasets.get(symbol)
        if ds is None:
            return None

        if now - self._last_signal.get(symbol, 0) < cfg.cooldown_sec:
            return None

        market = await self._market_data.get(symbol)
        if not market:
            return None
        price = f(market.get("last_price"))
        if price <= 0:
            return None

        trades = await self._trade_store.get_window(symbol, cfg.live_window_ms)
        if len(trades) < cfg.min_live_trade_count:
            return None
        c.data_ready = c.elapsed_sec >= cfg.min_live_warmup_sec and len(trades) >= cfg.min_live_trade_count
        if not c.data_ready:
            return None

        current_pattern = await self._get_current_pattern(symbol)
        if current_pattern is None or len(current_pattern) != cfg.pattern_length:
            return None

        # Check if we already evaluated this exact pattern (by timestamp of the oldest candle)
        pattern_ts = current_pattern[0].ts
        if self._last_evaluated_ts.get(symbol) == pattern_ts:
            return None  # already evaluated this candle
        self._last_evaluated_ts[symbol] = pattern_ts

        log.info("[historical] %s PATTERN ANALYSIS STARTED timeframe=%ds pattern_length=%d",
                 symbol, cfg.candle_interval_sec, cfg.pattern_length)

        log.info("[historical] %s searching historical matches ...", symbol)
        matches = ds.find_matches(current_pattern)
        match_count = len(matches)
        log.info("[historical] %s historical matches FOUND: %d", symbol, match_count)

        if match_count < cfg.min_historical_matches:
            log.info("[historical] %s NO_SIGNAL: insufficient matches (%d < %d)", symbol, match_count, cfg.min_historical_matches)
            return None

        log.info("[historical] %s analyzing forward outcomes (min_forward=%d, dir_ratio=%.0f%%)",
                 symbol, cfg.min_forward_trend_candles, cfg.min_forward_directional_ratio * 100)

        bullish = 0
        bearish = 0
        neutral = 0
        details = []
        top_n = min(cfg.log_top_matches, match_count)
        for idx, (start_idx, sim) in enumerate(matches):
            outcome = ds.classify_forward(start_idx)
            if outcome == "bullish":
                bullish += 1
            elif outcome == "bearish":
                bearish += 1
            else:
                neutral += 1
            if idx < top_n:
                details.append((sim, outcome))
        total = bullish + bearish + neutral
        bullish_ratio = bullish / total if total else 0.0
        bearish_ratio = bearish / total if total else 0.0

        log.info("[historical] %s HISTORICAL RESULT", symbol)
        log.info("[historical] %s   matches=%d bullish=%d bearish=%d neutral=%d", symbol, total, bullish, bearish, neutral)
        log.info("[historical] %s   bullish_agreement=%.1f%% bearish_agreement=%.1f%% required=%.0f%%",
                 symbol, bullish_ratio * 100, bearish_ratio * 100, cfg.min_directional_agreement * 100)
        if details:
            log.debug("[historical] %s   top matches: %s", symbol, details)

        direction = ""
        if bullish_ratio >= cfg.min_directional_agreement:
            direction = "long"
            log.info("[historical] %s SIGNAL = LONG (agreement=%.1f%%)", symbol, bullish_ratio * 100)
        elif bearish_ratio >= cfg.min_directional_agreement:
            direction = "short"
            log.info("[historical] %s SIGNAL = SHORT (agreement=%.1f%%)", symbol, bearish_ratio * 100)
        else:
            log.info("[historical] %s SIGNAL = NO_SIGNAL (agreement below threshold)", symbol)
            return None

        c.direction = direction
        self._last_signal[symbol] = now
        async with self._lock:
            self._candidates.pop(symbol, None)

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=max(bullish_ratio, bearish_ratio),
            entry_price=price,
            take_profit=price,
            stop_loss=price,
            timestamp=now,
            reasons=[
                "engine=historical_3candle_pattern",
                f"matches={total}",
                f"bullish={bullish}",
                f"bearish={bearish}",
                f"neutral={neutral}",
                f"agreement={max(bullish_ratio, bearish_ratio):.2f}"
            ]
        )


def build(ctx: StrategyContext) -> HistoricalEngine:
    cfg = ctx.build_config(HistoricalEngineConfig)
    return HistoricalEngine(ctx.trade_store, ctx.market_data, ctx.candle_fetcher, cfg)