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

Position mode assumption
-------------------------
This client assumes the OKX (demo) account is set to **net (one-way)**
position mode (Trade Settings -> Position Mode -> Net). That matches how
the bot already behaves (it never holds simultaneous long+short on the same
symbol), and keeps every order call posSide-free: `side=buy` opens/adds
long exposure, `side=sell` opens/adds short exposure, and closes go through
the opposite side with `reduceOnly=true`. If the account is in long/short
(hedge) mode instead, OKX will reject orders here with a posSide-related
error — switch the account to net mode before running this.

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
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self.demo_trading = demo_trading
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
            raise OKXAPIError(
                f"{path} failed: HTTP {resp.status_code} code={data.get('code')} msg={data.get('msg')}",
                code=data.get("code"),
                payload=data,
            )
        resp.raise_for_status()

        code = data.get("code")
        if code != "0":
            raise OKXAPIError(f"{path} failed: code={code} msg={data.get('msg')}", code=code, payload=data)
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

    # ------------------------------------------------------------------
    # Account / private GET
    # ------------------------------------------------------------------

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

    async def submit_leverage(self, symbol: str, leverage: int, open_type: str) -> dict:
        mgn_mode = "isolated" if open_type == "isolated" else "cross"
        body = {"instId": symbol, "lever": str(leverage), "mgnMode": mgn_mode}
        data = await self._request("POST", "/api/v5/account/set-leverage", body=body, auth=True)
        rows = data or []
        return rows[0] if rows else {}

    async def submit_order(
        self,
        symbol: str,
        side: int,
        size: int,
        order_type: str = "market",
        price: Optional[str] = None,
        leverage: Optional[str] = None,
        open_type: Optional[str] = None,
        mode: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """`side` keeps the old BitMart numeric convention so
        execution_engine.py doesn't need to change its call sites:
          1 = buy_open_long   -> OKX side="buy"
          2 = buy_close_short -> OKX side="buy",  reduceOnly=true
          3 = sell_close_long -> OKX side="sell", reduceOnly=true
          4 = sell_open_short -> OKX side="sell"
        Assumes net (one-way) position mode — see module docstring."""
        side_map = {1: ("buy", False), 2: ("buy", True), 3: ("sell", True), 4: ("sell", False)}
        okx_side, reduce_only = side_map.get(side, ("buy", False))
        mgn_mode = "isolated" if (open_type or "isolated") == "isolated" else "cross"

        body: Dict[str, Any] = {
            "instId": symbol,
            "tdMode": mgn_mode,
            "side": okx_side,
            "ordType": "market" if order_type == "market" else "limit",
            "sz": str(size),
        }
        if reduce_only:
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
        size: Optional[int] = None,
        plan_category: int = 2,
        category: str = "market",
    ) -> dict:
        """Places a standalone take-profit algo order via
        /api/v5/trade/order-algo (ordType="conditional"), market-executed
        on trigger. `side` uses the same BitMart-style close codes as
        submit_order (2=close short via buy, 3=close long via sell)."""
        side_map = {2: "buy", 3: "sell"}
        okx_side = side_map.get(side)
        if okx_side is None:
            raise OKXAPIError(f"submit_tp_sl_order: unsupported side {side!r}")

        body: Dict[str, Any] = {
            "instId": symbol,
            "tdMode": "isolated",
            "side": okx_side,
            "ordType": "conditional",
            "reduceOnly": "true",
            "tpTriggerPx": trigger_price,
            "tpOrdPx": "-1",  # -1 = execute the TP as a market order once triggered
        }
        if size is not None:
            body["sz"] = str(size)

        data = await self._request("POST", "/api/v5/trade/order-algo", body=body, auth=True)
        rows = data or []
        row = rows[0] if rows else {}
        if row.get("sCode") not in (None, "0"):
            raise OKXAPIError(f"tp/sl order rejected: sCode={row.get('sCode')} sMsg={row.get('sMsg')}", code=row.get("sCode"), payload=row)
        return {"order_id": row.get("algoId")}

    async def cancel_order(self, symbol: str, order_id: Optional[str] = None) -> dict:
        body: Dict[str, Any] = {"instId": symbol}
        if order_id is not None:
            body["ordId"] = str(order_id)
        data = await self._request("POST", "/api/v5/trade/cancel-order", body=body, auth=True)
        rows = data or []
        return rows[0] if rows else {}

    async def get_algo_order_status(self, symbol: str, algo_id: str) -> Optional[dict]:
        """Looks up a TP/SL conditional algo order's status via
        /api/v5/trade/orders-algo-history. NOTE: the exact field OKX uses
        to report the ordId(s) a triggered conditional order spawned has
        not been verified here against a live response — this reads a
        best-guess `ordIdList`/`ordId` field defensively and returns
        `ord_id: None` if neither is present, in which case callers should
        fall back to a time/side-based fills scan instead of trusting this
        linkage blindly."""
        data = await self._request(
            "GET",
            "/api/v5/trade/orders-algo-history",
            params={"instType": INST_TYPE, "ordType": "conditional", "algoId": str(algo_id), "instId": symbol},
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

    async def cancel_algo_order(self, symbol: str, algo_id: str) -> dict:
        """Cancels a pending TP algo order (e.g. before manually closing a
        position). Not present on the old BitMart client's public surface,
        but needed since OKX TP orders are a separate algo-order object."""
        data = await self._request(
            "POST", "/api/v5/trade/cancel-algos", body=[{"instId": symbol, "algoId": str(algo_id)}], auth=True
        )
        rows = data or []
        return rows[0] if rows else {}
