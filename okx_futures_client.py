"""
OKX v5 USDT-margined Perpetual Swap REST client.

Supports both production and OKX's Demo Trading environment. Unlike BitMart
(which uses a separate demo host), OKX Demo Trading uses the SAME REST host
as production (https://www.okx.com) and is toggled purely with the
`x-simulated-trading: 1` header on every request. The WebSocket side is the
opposite — demo trading uses a different host (wspap.okx.com) — see
tracker.py / test_ws.py for that.

Auth (all private endpoints):
  OK-ACCESS-KEY, OK-ACCESS-SIGN, OK-ACCESS-TIMESTAMP, OK-ACCESS-PASSPHRASE
  headers, where:
    sign = base64(HMAC_SHA256(secret, f"{timestamp}{method}{requestPath}{body}"))
    timestamp is ISO8601 with milliseconds, e.g. 2024-01-01T00:00:00.000Z

Position mode
-------------
`position_mode` (constructor arg) must match whatever the OKX account is
actually set to under Trade Settings -> Position Mode:

  "long_short" (hedge mode, the default here) — every trade/leverage call
  includes `posSide` ("long"/"short"), derived from the BitMart-style
  `side` code below. `reduceOnly` is omitted, since OKX only accepts it
  in net mode; in hedge mode, side+posSide alone determine open vs close.

  "net" (one-way mode) — no `posSide` is sent, `side=buy`/`side=sell`
  alone opens/adds exposure, and closes go through the opposite side with
  `reduceOnly=true`.

Passing "long_short" while the account is actually in net mode (or vice
versa) is exactly what produces OKX's `code=51000 msg=Parameter posSide
error` — the two must match.

Symbol format
-------------
Callers pass OKX instIds directly, e.g. "BTC-USDT-SWAP", not "BTCUSDT".

Field-mapping note
-------------------
To keep execution_engine.py's business logic close to its original shape,
several methods below translate OKX's native field names into the same
keys the old BitMart client returned (e.g. `current_amount`, `mark_price`,
`deal_avg_price`, `paid_fees`). Where OKX's exact response shape for a
less-common field (e.g. per-fill realized PnL) couldn't be double-checked
against a live account, it's flagged with a comment — verify against a
real demo response before trusting the numbers in production.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger("okx_futures.client")

BASE_URL = "https://www.okx.com"

USER_AGENT = "okx-futures-bot/1.0"

INST_TYPE = "SWAP"


class OKXAPIError(Exception):
    """Raised when OKX returns a non-"0" `code`, or the request fails after
    all retries have been exhausted."""

    def __init__(self, message: str, code: Optional[str] = None, payload: Optional[dict] = None) -> None:
        super().__init__(message)
        self.code = code
        self.payload = payload


def _inner_error_detail(data: Any) -> str:
    """OKX's top-level `msg` on a rejected /trade/order (or order-algo)
    call is frequently a generic placeholder — e.g. "All operations
    failed" or "Operation failed" — while the *actual* reason (wrong
    account/position mode, insufficient margin, bad posSide, lot-size
    violation, etc.) sits on the per-item `sCode`/`sMsg` fields inside
    `data`. Without surfacing those, every rejection looks identical in
    the logs no matter the real cause. Returns "" if there's nothing more
    specific to add."""
    if not isinstance(data, dict):
        return ""
    rows = data.get("data")
    if not isinstance(rows, list):
        return ""
    parts = []
    for row in rows:
        if isinstance(row, dict) and row.get("sCode") not in (None, "0"):
            parts.append(f"sCode={row.get('sCode')} sMsg={row.get('sMsg')}")
    return " | ".join(parts)


def _first_inner_scode(data: Any) -> Optional[str]:
    """The actionable per-item `sCode` (e.g. "51155" for a compliance-
    restricted pair) that `_inner_error_detail` renders into the message
    text. Pulled out separately so callers can branch on the real reason
    programmatically instead of the generic top-level `code` (frequently
    just "1"/"All operations failed"), without having to regex the log
    string."""
    if not isinstance(data, dict):
        return None
    rows = data.get("data")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("sCode") not in (None, "0"):
            return str(row.get("sCode"))
    return None


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


class OKXFuturesClient:
    """Thin async wrapper around the OKX v5 Trade/Account/Public REST API,
    scoped to USDT-margined perpetual swaps.

    All network I/O happens via `requests`, executed in a worker thread via
    `asyncio.to_thread` so the async event loop is never blocked.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        demo_trading: bool = True,
        timeout_sec: float = 10.0,
        max_retries: int = 3,
        retry_base_delay_sec: float = 0.5,
        position_mode: str = "long_short",
    ) -> None:
        if position_mode not in ("long_short", "net"):
            raise ValueError(f"position_mode must be 'long_short' or 'net', got {position_mode!r}")
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self.demo_trading = demo_trading
        self.position_mode = position_mode
        self.base_url = BASE_URL
        self._timeout = timeout_sec
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay_sec
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Low-level request plumbing
    # ------------------------------------------------------------------

    def _sign(self, timestamp: str, method: str, request_path: str, body_str: str) -> str:
        payload = f"{timestamp}{method}{request_path}{body_str}"
        digest = hmac.new(self._api_secret.encode(), payload.encode(), hashlib.sha256).digest()
        return base64.b64encode(digest).decode()

    def _do_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Any:
        # OKX signs over the exact query string that goes on the wire, so
        # build it once and reuse it for both the signature and the request.
        query_str = ""
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                query_str = "?" + "&".join(f"{k}={v}" for k, v in clean.items())
        request_path = path + query_str
        body_str = json.dumps(body, separators=(",", ":")) if body else ""

        url = f"{self.base_url}{request_path}"
        headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
        if self.demo_trading:
            headers["x-simulated-trading"] = "1"

        if auth:
            timestamp = _iso_timestamp()
            headers["OK-ACCESS-KEY"] = self._api_key
            headers["OK-ACCESS-SIGN"] = self._sign(timestamp, method, request_path, body_str)
            headers["OK-ACCESS-TIMESTAMP"] = timestamp
            headers["OK-ACCESS-PASSPHRASE"] = self._passphrase

        resp = self._session.request(
            method,
            url,
            data=body_str if body_str else None,
            headers=headers,
            timeout=self._timeout,
        )

        # OKX returns a JSON body with the real code/msg even on 4xx
        # responses, so parse it before deciding anything -- raise_for_status()
        # would discard that body and mislabel a permanent, non-retryable
        # rejection as a transient network error.
        try:
            data = resp.json()
        except ValueError:
            resp.raise_for_status()
            raise OKXAPIError(f"{path} failed: HTTP {resp.status_code} with non-JSON body: {resp.text[:200]}")

        if 400 <= resp.status_code < 500:
            detail = _inner_error_detail(data)
            suffix = f" ({detail})" if detail else ""
            raise OKXAPIError(
                f"{path} failed: HTTP {resp.status_code} code={data.get('code')} msg={data.get('msg')}{suffix}",
                code=_first_inner_scode(data) or data.get("code"),
                payload=data,
            )
        resp.raise_for_status()

        code = data.get("code")
        if code != "0":
            detail = _inner_error_detail(data)
            suffix = f" ({detail})" if detail else ""
            raise OKXAPIError(
                f"{path} failed: code={code} msg={data.get('msg')}{suffix}",
                code=_first_inner_scode(data) or code,
                payload=data,
            )
        return data.get("data")

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Any:
        last_exc: Optional[Exception] = None
        delay = self._retry_base_delay
        for attempt in range(1, self._max_retries + 1):
            try:
                return await asyncio.to_thread(self._do_request, method, path, params, body, auth)
            except OKXAPIError as exc:
                # Application-level errors (bad params, insufficient margin,
                # etc.) are not retried — retrying would repeat the mistake.
                log.error(f"[okx-api] {method} {path} rejected: {exc}")
                raise
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                log.warning(
                    f"[okx-api] {method} {path} attempt {attempt}/{self._max_retries} failed: "
                    f"{type(exc).__name__}: {str(exc)[:200]}"
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        raise OKXAPIError(f"{method} {path} failed after {self._max_retries} attempts: {last_exc}")

    # ------------------------------------------------------------------
    # Market data / instrument metadata (no auth)
    # ------------------------------------------------------------------

    async def list_swap_instruments(self) -> List[dict]:
        """Returns all live USDT-margined perpetual swap instruments.
        Used at startup to build the WS subscription universe (OKX's public
        WS has no "all tickers" firehose channel — each instId must be
        subscribed individually)."""
        data = await self._request("GET", "/api/v5/public/instruments", params={"instType": INST_TYPE})
        return [
            d for d in (data or [])
            if d.get("settleCcy") == "USDT" and d.get("state") == "live" and str(d.get("instId", "")).endswith("-USDT-SWAP")
        ]

    async def get_contract_details(self, symbol: str) -> dict:
        """Returns contract metadata for `symbol` (an instId, e.g.
        "BTC-USDT-SWAP"), translated into the same keys the old BitMart
        client returned so execution_engine.py's sizing math is unchanged:
          max_leverage   <- lever
          contract_size  <- ctVal (value of 1 contract, in the base ccy)
          price_precision <- tickSz
          vol_precision  <- lotSz
          min_volume     <- minSz
        """
        data = await self._request("GET", "/api/v5/public/instruments", params={"instType": INST_TYPE, "instId": symbol})
        rows = data or []
        if not rows:
            raise OKXAPIError(f"No instrument details returned for {symbol}")
        row = rows[0]
        return {
            "max_leverage": row.get("lever"),
            "contract_size": row.get("ctVal"),
            "price_precision": row.get("tickSz"),
            "vol_precision": row.get("lotSz"),
            "min_volume": row.get("minSz"),
        }

    async def get_candles(self, symbol: str, bar: str = "5m", limit: int = 100) -> List[dict]:
        """Recent OHLCV candles for `symbol` via /api/v5/market/candles
        (public, no auth). OKX returns newest-first; `limit` maxes out at
        300 per OKX's docs, which comfortably covers both of this bot's
        current lookback windows (1h and 15h at 5m bars = 12 and 180
        candles) in a single call — no pagination/history-candles needed."""
        data = await self._request(
            "GET", "/api/v5/market/candles",
            params={"instId": symbol, "bar": bar, "limit": str(limit)},
        )
        candles = []
        for row in data or []:
            try:
                candles.append({
                    "ts": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "confirm": row[8] if len(row) > 8 else None,
                })
            except (IndexError, TypeError, ValueError):
                continue
        return candles

    # ------------------------------------------------------------------
    # Account / private GET
    # ------------------------------------------------------------------

    async def get_position_tier_mmr(self, symbol: str, open_type: str, notional_usdt: float) -> float:
        """Returns the maintenance margin rate (mmr, e.g. 0.005 = 0.5%) OKX
        would apply to a position of `notional_usdt` on `symbol`, via
        /api/v5/public/position-tiers.

        MMR is tiered by position notional — larger positions sit in
        higher tiers with a higher mmr — so this picks the tier whose
        [minSz, maxSz] bracket actually contains notional_usdt rather than
        always assuming tier 1. Falls back to the highest tier if
        notional_usdt exceeds every bracket.

        Used only by the pre-trade liquidation-distance guard
        (liquidation_guard.py) — never for anything that needs to be
        exact to the tick, since it's an estimate going into an estimate.
        VERIFY minSz/maxSz's exact units (USDT notional vs. contracts) on
        this endpoint against a live response before trusting this in
        production; not double-checked against a live demo response here."""
        mgn_mode = "isolated" if open_type == "isolated" else "cross"
        # OKX rejects this endpoint (code=50015) without instFamily or uly —
        # instId alone isn't enough. instFamily is just instId with the
        # "-SWAP" suffix dropped (e.g. "ETH-USDT-SWAP" -> "ETH-USDT").
        inst_family = symbol[: -len("-SWAP")] if symbol.endswith("-SWAP") else symbol
        data = await self._request(
            "GET", "/api/v5/public/position-tiers",
            params={"instType": INST_TYPE, "tdMode": mgn_mode, "instFamily": inst_family, "instId": symbol},
        )
        rows = data or []
        if not rows:
            raise OKXAPIError(f"No position-tier data returned for {symbol}")

        def _f(row, key, default=0.0):
            try:
                return float(row.get(key, default) or default)
            except (TypeError, ValueError):
                return default

        sorted_rows = sorted(rows, key=lambda r: _f(r, "minSz"))
        for row in sorted_rows:
            min_sz, max_sz = _f(row, "minSz"), _f(row, "maxSz")
            upper_bound = max_sz if max_sz > 0 else float("inf")
            if min_sz <= notional_usdt <= upper_bound:
                return _f(row, "mmr")
        # notional_usdt fell outside every bracket (e.g. above the top
        # tier's maxSz) — use the highest tier's rate rather than guessing.
        return _f(sorted_rows[-1], "mmr")

    async def get_trade_fee_rate(self, symbol: str) -> dict:
        data = await self._request(
            "GET", "/api/v5/account/trade-fee", params={"instType": INST_TYPE, "instId": symbol}, auth=True
        )
        rows = data or []
        row = rows[0] if rows else {}
        # OKX quotes taker fee as a negative string (e.g. "-0.0005" = 0.05%
        # paid). Normalize to a positive rate so callers can multiply
        # straight through, matching the old BitMart client's convention.
        taker = row.get("taker")
        try:
            taker_rate = abs(float(taker)) if taker is not None else 0.0
        except (TypeError, ValueError):
            taker_rate = 0.0
        return {"taker_fee_rate": taker_rate}

    async def get_position(self, symbol: Optional[str] = None) -> List[dict]:
        params = {"instType": INST_TYPE, "instId": symbol} if symbol else {"instType": INST_TYPE}
        data = await self._request("GET", "/api/v5/account/positions", params=params, auth=True)
        results = []
        for p in data or []:
            try:
                current_amount = abs(float(p.get("pos", 0)))
            except (TypeError, ValueError):
                current_amount = 0.0
            results.append({
                "current_amount": current_amount,
                "mark_price": p.get("markPx"),
                "unrealized_pnl": p.get("upl"),
                "liquidation_price": p.get("liqPx"),
                "avg_price": p.get("avgPx"),
                "raw": p,
            })
        return results

    async def get_order(self, symbol: str, order_id: str) -> dict:
        data = await self._request(
            "GET", "/api/v5/trade/order", params={"instId": symbol, "ordId": str(order_id)}, auth=True
        )
        rows = data or []
        row = rows[0] if rows else {}
        # OKX order states: live, partially_filled, filled, canceled,
        # mmp_canceled. Map "filled" to "4" to match the old BitMart
        # client's convention that callers (execution_engine.py) check for.
        state = "4" if row.get("state") == "filled" else row.get("state")
        return {
            "state": state,
            "deal_avg_price": row.get("avgPx"),
            "deal_size": row.get("accFillSz"),
            "order_id": row.get("ordId"),
            "client_order_id": row.get("clOrdId"),
            "raw": row,
        }

    async def get_trades(
        self, symbol: Optional[str] = None, order_id: Optional[str] = None
    ) -> List[dict]:
        """Returns fills, translated to the old field names
        (price/vol/paid_fees/create_time/side/realised_profit).

        NOTE: OKX's /trade/fills response does not carry a reliable
        per-fill realized-PnL field for swaps in all cases — verify this
        against a live demo response. If `pnl` isn't present, realized_pnl
        will come back as 0 here and should instead be read from
        get_order()'s `raw.pnl` for a fully-closed order."""
        params: Dict[str, Any] = {"instType": INST_TYPE}
        if symbol:
            params["instId"] = symbol
        if order_id:
            params["ordId"] = str(order_id)
        data = await self._request("GET", "/api/v5/trade/fills", params=params, auth=True)
        results = []
        for t in data or []:
            try:
                pnl = float(t.get("pnl", 0) or 0)
            except (TypeError, ValueError):
                pnl = 0.0
            try:
                fee = abs(float(t.get("fee", 0) or 0))
            except (TypeError, ValueError):
                fee = 0.0
            results.append({
                "price": t.get("fillPx"),
                "vol": t.get("fillSz"),
                "paid_fees": fee,
                "realised_profit": pnl,
                "create_time": t.get("ts"),
                "side": t.get("side"),
                "order_id": t.get("ordId"),
                "raw": t,
            })
        return results

    async def get_order_history(
        self, symbol: str, start_time: Optional[int] = None, end_time: Optional[int] = None
    ) -> List[dict]:
        """Returns recently filled/canceled orders for `symbol` (last 7
        days via /trade/orders-history). `start_time`/`end_time` are Unix
        seconds, converted to the ms timestamps OKX expects."""
        params: Dict[str, Any] = {"instType": INST_TYPE, "instId": symbol}
        if start_time is not None:
            params["begin"] = str(int(start_time) * 1000)
        if end_time is not None:
            params["end"] = str(int(end_time) * 1000)
        data = await self._request("GET", "/api/v5/trade/orders-history", params=params, auth=True)
        results = []
        for row in data or []:
            state = "4" if row.get("state") == "filled" else row.get("state")
            results.append({
                "order_id": row.get("ordId"),
                "client_order_id": row.get("clOrdId"),
                "state": state,
                "raw": row,
            })
        return results

    # ------------------------------------------------------------------
    # Trading (private POST)
    # ------------------------------------------------------------------

    async def submit_leverage(self, symbol: str, leverage: int, open_type: str, direction: Optional[str] = None) -> dict:
        """`direction` ("long"/"short") is required when the client is in
        hedge mode, since OKX's set-leverage endpoint requires `posSide`
        for isolated margin under long/short position mode (this is what
        produces `code=51000 msg=Parameter posSide error` if omitted)."""
        mgn_mode = "isolated" if open_type == "isolated" else "cross"
        body: Dict[str, Any] = {"instId": symbol, "lever": str(leverage), "mgnMode": mgn_mode}
        if self.position_mode == "long_short":
            if direction not in ("long", "short"):
                raise OKXAPIError(f"submit_leverage: hedge mode requires direction='long'/'short', got {direction!r}")
            body["posSide"] = direction
        data = await self._request("POST", "/api/v5/account/set-leverage", body=body, auth=True)
        rows = data or []
        return rows[0] if rows else {}

    async def submit_order(
        self,
        symbol: str,
        side: int,
        size: float,
        order_type: str = "market",
        price: Optional[str] = None,
        leverage: Optional[str] = None,
        open_type: Optional[str] = None,
        mode: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """`side` keeps the old BitMart numeric convention so
        execution_engine.py doesn't need to change its call sites:
          1 = buy_open_long   -> OKX side="buy",  posSide="long"
          2 = buy_close_short -> OKX side="buy",  posSide="short" (reduceOnly in net mode)
          3 = sell_close_long -> OKX side="sell", posSide="long"  (reduceOnly in net mode)
          4 = sell_open_short -> OKX side="sell", posSide="short"
        In hedge (long_short) mode, `posSide` is sent and `reduceOnly` is
        omitted (OKX only accepts reduceOnly in net mode — side+posSide
        alone determine open vs close in hedge mode). In net mode,
        `posSide` is omitted and `reduceOnly` is sent instead — see module
        docstring."""
        side_map = {1: ("buy", "long", False), 2: ("buy", "short", True), 3: ("sell", "long", True), 4: ("sell", "short", False)}
        okx_side, pos_side, reduce_only = side_map.get(side, ("buy", "long", False))
        mgn_mode = "isolated" if (open_type or "isolated") == "isolated" else "cross"

        body: Dict[str, Any] = {
            "instId": symbol,
            "tdMode": mgn_mode,
            "side": okx_side,
            "ordType": "market" if order_type == "market" else "limit",
            "sz": str(size),
        }
        if self.position_mode == "long_short":
            body["posSide"] = pos_side
        elif reduce_only:
            body["reduceOnly"] = "true"
        if order_type == "limit" and price is not None:
            body["px"] = price
        if client_order_id is not None:
            body["clOrdId"] = client_order_id

        data = await self._request("POST", "/api/v5/trade/order", body=body, auth=True)
        rows = data or []
        row = rows[0] if rows else {}
        if row.get("sCode") not in (None, "0"):
            raise OKXAPIError(f"order rejected: sCode={row.get('sCode')} sMsg={row.get('sMsg')}", code=row.get("sCode"), payload=row)
        return {"order_id": row.get("ordId"), "client_order_id": row.get("clOrdId")}

    async def submit_tp_sl_order(
        self,
        symbol: str,
        order_type: str,
        side: int,
        trigger_price: str,
        executive_price: str,
        price_type: int = 1,
        size: Optional[float] = None,
        plan_category: int = 2,
        category: str = "market",
        stop_loss_trigger_price: Optional[str] = None,
    ) -> dict:
        """Places a take-profit algo order via /api/v5/trade/order-algo.
        `side` uses the same BitMart-style close codes as submit_order
        (2=close short via buy, 3=close long via sell). In hedge mode,
        `posSide` is sent (identifying which position this closes) and
        `reduceOnly` is omitted; in net mode it's the reverse — see
        module docstring.

        If `stop_loss_trigger_price` is given, the order is placed as
        ordType="oco" (one-cancels-other) with both tpTriggerPx and
        slTriggerPx set on the same algo order — whichever side triggers
        first cancels the other automatically, so a closed position can
        never leave the other leg still live on the exchange. Without it,
        this places a TP-only ordType="conditional" order, same as
        before."""
        side_map = {2: ("buy", "short"), 3: ("sell", "long")}
        mapped = side_map.get(side)
        if mapped is None:
            raise OKXAPIError(f"submit_tp_sl_order: unsupported side {side!r}")
        okx_side, pos_side = mapped

        body: Dict[str, Any] = {
            "instId": symbol,
            "tdMode": "isolated",
            "side": okx_side,
            "ordType": "oco" if stop_loss_trigger_price is not None else "conditional",
            "tpTriggerPx": trigger_price,
            "tpOrdPx": "-1",  # -1 = execute the TP as a market order once triggered
        }
        if stop_loss_trigger_price is not None:
            body["slTriggerPx"] = stop_loss_trigger_price
            body["slOrdPx"] = "-1"  # -1 = execute the SL as a market order once triggered
        if self.position_mode == "long_short":
            body["posSide"] = pos_side
        else:
            body["reduceOnly"] = "true"
        if size is not None:
            body["sz"] = str(size)

        data = await self._request("POST", "/api/v5/trade/order-algo", body=body, auth=True)
        rows = data or []
        row = rows[0] if rows else {}
        if row.get("sCode") not in (None, "0"):
            raise OKXAPIError(f"tp/sl order rejected: sCode={row.get('sCode')} sMsg={row.get('sMsg')}", code=row.get("sCode"), payload=row)
        return {"order_id": row.get("algoId")}

    async def get_closed_position(self, symbol: str, opened_at_ms: float) -> Optional[dict]:
        """Returns the exchange's own record of the position on `symbol`
        that closed at or after `opened_at_ms` (Unix ms — our local
        timestamp for when the position was opened), via
        /api/v5/account/positions-history.

        Filters on the row's `uTime` (its close time), not `cTime` (its
        open time): `opened_at_ms` is captured locally only after the
        opening order's fill-wait, fee lookup, and TP placement all
        complete, so it can land slightly *after* OKX's own `cTime` for
        the same position — filtering on cTime would then wrongly drop
        the correct row every time. `uTime` doesn't have that problem:
        a position's close necessarily happens after it opens, so uTime
        is always safely after our local opened_at regardless of any
        skew on the open side.

        This is the endpoint to use for a closed position's realized PnL:
        unlike /trade/fills (see get_trades()), positions-history rows
        carry genuine `pnl` (price PnL, excluding fees) and `fee` fields
        for closed FUTURES/SWAP/OPTION positions, plus a `type` field
        that says exactly how the position closed (1/2 = closed normally,
        3/4 = liquidated, 5/6 = ADL) instead of having to infer it.
        Returns None if no matching row is found yet (the record can lag
        a little behind the position actually closing).

        IMPORTANT — `fee` here is CUMULATIVE for the position's entire
        lifecycle (the opening trade's fee *and* the closing trade's fee
        added together), not the closing trade's fee in isolation. OKX
        produces one positions-history row per position, covering
        open-to-close, so there is no separate "closing-only" fee field
        on this endpoint. `fundingFee` is reported separately and is
        included here too since it's a real cost of holding the position.
        Callers that already know the real opening fee (captured from the
        fill at open time) must subtract it back out of `total_fee` to
        isolate the true closing-side fee — treating `total_fee` itself
        as the closing fee double-counts the opening fee (this was the
        bug: it made closing_fee read ~2x too high and made net_pnl,
        computed downstream as realized_pnl - opening_fee - closing_fee,
        subtract the opening fee twice).

        This row also carries OKX's own `realizedPnl` field, which is
        `pnl` (price PnL) plus `fee`, `fundingFee`, AND `liqPenalty`
        (the liquidation penalty charged on a liquidated position) all
        netted together by OKX itself — returned here as `net_pnl`.
        Reconstructing net PnL locally from realized_pnl/opening_fee/
        closing_fee alone silently drops `liqPenalty`, which is exactly
        why locally-computed net_pnl read far less negative than reality
        on liquidated trades. Prefer this field over any local
        reconstruction whenever it's present."""
        data = await self._request(
            "GET",
            "/api/v5/account/positions-history",
            params={"instType": INST_TYPE, "instId": symbol},
            auth=True,
        )
        candidates = []
        for row in data or []:
            try:
                if float(row.get("uTime", 0) or 0) >= opened_at_ms:
                    candidates.append(row)
            except (TypeError, ValueError):
                continue
        if not candidates:
            return None
        row = max(candidates, key=lambda r: float(r.get("uTime", 0) or 0))
        try:
            pnl = float(row.get("pnl", 0) or 0)
        except (TypeError, ValueError):
            pnl = 0.0
        try:
            trading_fee = abs(float(row.get("fee", 0) or 0))
        except (TypeError, ValueError):
            trading_fee = 0.0
        try:
            funding_fee = abs(float(row.get("fundingFee", 0) or 0))
        except (TypeError, ValueError):
            funding_fee = 0.0
        raw_net_pnl = row.get("realizedPnl")
        try:
            net_pnl = float(raw_net_pnl) if raw_net_pnl not in (None, "") else None
        except (TypeError, ValueError):
            net_pnl = None
        return {
            "exit_price": row.get("closeAvgPx"),
            "realized_pnl": pnl,
            # Cumulative fee for the WHOLE position (open + close + funding).
            # See docstring above — this is NOT the closing-only fee. The
            # caller must subtract the known opening_fee from this to get
            # the true closing_fee.
            "total_fee": trading_fee + funding_fee,
            # OKX's own fully-netted realized PnL (pnl + fee + fundingFee +
            # liqPenalty). None if the field wasn't present on this row.
            "net_pnl": net_pnl,
            "close_type": row.get("type"),
            "raw": row,
        }

    async def cancel_order(self, symbol: str, order_id: Optional[str] = None) -> dict:
        body: Dict[str, Any] = {"instId": symbol}
        if order_id is not None:
            body["ordId"] = str(order_id)
        data = await self._request("POST", "/api/v5/trade/cancel-order", body=body, auth=True)
        rows = data or []
        return rows[0] if rows else {}

    async def get_algo_order_status(
        self, symbol: str, algo_id: str, attempts: int = 3, retry_delay_sec: float = 1.0
    ) -> Optional[dict]:
        """Looks up a TP/SL algo order's status via
        /api/v5/trade/orders-algo-history. Queries with ordType="oco"
        since execution_engine.py always places TP+SL together as a
        single one-cancels-other algo order (see
        DemoFuturesExecutionEngine._place_tp_sl) — querying with the
        wrong ordType (this used to say "conditional", left over from
        before SL support was added) makes this endpoint report
        code=51603 "Order does not exist" for an order that verifiably
        exists, every single time, since OKX filters this lookup by
        ordType server-side.

        NOTE: even with the right ordType, a successful "effective"
        result here only tells you the algo order triggered — NOT which
        leg (TP or SL) did, since both share the same algoId on an OCO
        order. execution_engine.py no longer uses this method to
        determine close_reason for that reason (see
        _infer_close_reason_from_exit_price) — this is kept for anything
        that only needs to know whether the order is still live/pending
        vs. triggered/canceled.

        Retries on code=51603 ("Order does not exist"): this endpoint's
        index can lag a couple of seconds behind an algo order that was
        JUST created, so querying it immediately after placing/triggering
        a TP can spuriously come back "not found" even though the order
        is live. `_request()` doesn't retry application-level errors by
        design (retrying a genuinely bad request just repeats the
        mistake), so that retry has to happen here, where we know 51603
        specifically can be a transient indexing-lag artifact rather than
        a real absence."""
        last_exc: Optional[OKXAPIError] = None
        for attempt in range(1, attempts + 1):
            try:
                data = await self._request(
                    "GET",
                    "/api/v5/trade/orders-algo-history",
                    params={"instType": INST_TYPE, "ordType": "oco", "algoId": str(algo_id), "instId": symbol},
                    auth=True,
                )
                rows = data or []
                row = rows[0] if rows else None
                if not row:
                    return None
                ord_id = row.get("ordId")
                if not ord_id:
                    ord_id_list = row.get("ordIdList") or []
                    ord_id = ord_id_list[0] if ord_id_list else None
                return {"state": row.get("state"), "ord_id": ord_id, "raw": row}
            except OKXAPIError as exc:
                last_exc = exc
                if str(exc.code) != "51603" or attempt >= attempts:
                    raise
                log.warning(
                    f"[okx-api] orders-algo-history 51603 for {symbol} algoId={algo_id} "
                    f"(attempt {attempt}/{attempts}) — likely index lag, retrying"
                )
                await asyncio.sleep(retry_delay_sec)
        if last_exc is not None:
            raise last_exc
        return None

    async def get_pending_algo_order(self, symbol: str, algo_id: str) -> Optional[dict]:
        """Reads back a currently-live (not yet triggered) algo order's
        own parameters via /api/v5/trade/orders-algo-pending. Deliberately
        NOT the same endpoint as get_algo_order_status (which queries
        orders-algo-history, a historical index documented there to lag a
        couple of seconds behind an order that was just created) — this
        queries the live pending-order book directly, so a just-placed
        order's actual accepted slTriggerPx/tpTriggerPx are readable
        immediately, with no indexing delay. This is the source of truth
        for confirming what OKX actually has resting right now, used by
        execution_engine.py's _ratchet_stop_loss verification loop.

        Returns None if the order isn't found pending — e.g. it already
        triggered in the moment between being placed and this being
        called, or genuinely doesn't exist. Callers should treat that as
        "can't verify right now", not as an error."""
        data = await self._request(
            "GET",
            "/api/v5/trade/orders-algo-pending",
            params={"instType": INST_TYPE, "ordType": "oco", "algoId": str(algo_id), "instId": symbol},
            auth=True,
        )
        rows = data or []
        row = rows[0] if rows else None
        if not row:
            return None
        return {
            "sl_trigger_price": row.get("slTriggerPx"),
            "tp_trigger_price": row.get("tpTriggerPx"),
            "state": row.get("state"),
            "raw": row,
        }

    async def cancel_algo_order(self, symbol: str, algo_id: str) -> dict:
        """Cancels a pending TP algo order (e.g. before manually closing a
        position). Not present on the old BitMart client's public surface,
        but needed since OKX TP orders are a separate algo-order object."""
        data = await self._request(
            "POST", "/api/v5/trade/cancel-algos", body=[{"instId": symbol, "algoId": str(algo_id)}], auth=True
        )
        rows = data or []
        return rows[0] if rows else {}