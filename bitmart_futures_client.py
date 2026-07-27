"""
BitMart USDT-M Futures REST client.

Supports both the production and Demo Trading environments. Demo Trading uses
the same API key/secret/memo as production but a different base URL
(https://demo-api-cloud-v2.bitmart.com). All endpoint paths and signing rules
are identical between the two environments.

Auth levels used here:
  - NONE:   public market data, no headers required.
  - KEYED:  private GET endpoints, only the X-BM-KEY header is required.
  - SIGNED: private POST endpoints, require X-BM-KEY, X-BM-TIMESTAMP and
            X-BM-SIGN = HMAC_SHA256(secret, f"{timestamp_ms}#{memo}#{body}")
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger("bitmart_futures.client")

PROD_BASE_URL = "https://api-cloud-v2.bitmart.com"
DEMO_BASE_URL = "https://demo-api-cloud-v2.bitmart.com"

USER_AGENT = "bitmart-futures-bot/1.0"


class BitMartAPIError(Exception):
    """Raised when BitMart returns a non-success application code, or the
    request fails after all retries have been exhausted."""

    def __init__(self, message: str, code: Optional[int] = None, payload: Optional[dict] = None) -> None:
        super().__init__(message)
        self.code = code
        self.payload = payload


class BitMartFuturesClient:
    """Thin async wrapper around the BitMart Futures v2 REST API.

    All network I/O happens via `requests`, executed in a worker thread via
    `asyncio.to_thread` so the async event loop is never blocked.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        memo: str,
        demo_trading: bool = True,
        timeout_sec: float = 10.0,
        max_retries: int = 3,
        retry_base_delay_sec: float = 0.5,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._memo = memo
        self.base_url = DEMO_BASE_URL if demo_trading else PROD_BASE_URL
        self.demo_trading = demo_trading
        self._timeout = timeout_sec
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay_sec
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Low-level request plumbing
    # ------------------------------------------------------------------

    def _sign(self, timestamp_ms: str, body_str: str) -> str:
        payload = f"{timestamp_ms}#{self._memo}#{body_str}"
        return hmac.new(self._api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    def _do_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: str = "none",
    ) -> dict:
        url = f"{self.base_url}{path}"
        headers = {"User-Agent": USER_AGENT}
        body_str = json.dumps(body, separators=(",", ":")) if body is not None else None

        if auth in ("keyed", "signed"):
            headers["X-BM-KEY"] = self._api_key
        if auth == "signed":
            timestamp_ms = str(int(time.time() * 1000))
            headers["X-BM-TIMESTAMP"] = timestamp_ms
            headers["X-BM-SIGN"] = self._sign(timestamp_ms, body_str or "{}")
            headers["Content-Type"] = "application/json"

        resp = self._session.request(
            method,
            url,
            params=params,
            data=body_str,
            headers=headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        code = data.get("code")
        if code != 1000:
            raise BitMartAPIError(f"{path} failed: code={code} message={data.get('message')}", code=code, payload=data)
        return data.get("data")

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: str = "none",
    ) -> dict:
        last_exc: Optional[Exception] = None
        delay = self._retry_base_delay
        for attempt in range(1, self._max_retries + 1):
            try:
                return await asyncio.to_thread(self._do_request, method, path, params, body, auth)
            except BitMartAPIError as exc:
                # Application-level errors (bad params, insufficient leverage,
                # etc.) are not retried since retrying would repeat the same
                # mistake; the caller should decide what to do next.
                log.error(f"[bitmart-api] {method} {path} rejected: {exc}")
                raise
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                log.warning(
                    f"[bitmart-api] {method} {path} attempt {attempt}/{self._max_retries} failed: "
                    f"{type(exc).__name__}: {str(exc)[:200]}"
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        raise BitMartAPIError(f"{method} {path} failed after {self._max_retries} attempts: {last_exc}")

    # ------------------------------------------------------------------
    # Market data (NONE auth)
    # ------------------------------------------------------------------

    async def get_contract_details(self, symbol: str) -> dict:
        """Returns the first (and only) contract entry for `symbol`, including
        `max_leverage`, `contract_size`, `price_precision`, `vol_precision`,
        and `min_volume` — all required before sizing/leveraging an order."""
        data = await self._request("GET", "/contract/public/details", params={"symbol": symbol})
        symbols = data.get("symbols") or []
        if not symbols:
            raise BitMartAPIError(f"No contract details returned for {symbol}")
        return symbols[0]

    # ------------------------------------------------------------------
    # Account / private GET (KEYED auth)
    # ------------------------------------------------------------------

    async def get_trade_fee_rate(self, symbol: str) -> dict:
        return await self._request(
            "GET", "/contract/private/trade-fee-rate", params={"symbol": symbol}, auth="keyed"
        )

    async def get_position(self, symbol: Optional[str] = None) -> List[dict]:
        params = {"symbol": symbol} if symbol else None
        data = await self._request("GET", "/contract/private/position-v2", params=params, auth="keyed")
        return data or []

    async def get_order(self, symbol: str, order_id: str) -> dict:
        return await self._request(
            "GET", "/contract/private/order", params={"symbol": symbol, "order_id": str(order_id)}, auth="keyed"
        )

    async def get_trades(
        self, symbol: Optional[str] = None, order_id: Optional[str] = None
    ) -> List[dict]:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["order_id"] = order_id
        data = await self._request("GET", "/contract/private/trades", params=params, auth="keyed")
        return data or []

    async def get_transaction_history(
        self, symbol: Optional[str] = None, flow_type: Optional[int] = None, page_size: int = 20
    ) -> List[dict]:
        params: Dict[str, Any] = {"page_size": page_size}
        if symbol:
            params["symbol"] = symbol
        if flow_type is not None:
            params["flow_type"] = flow_type
        data = await self._request("GET", "/contract/private/transaction-history", params=params, auth="keyed")
        return data or []

    async def get_order_history(
        self, symbol: str, start_time: Optional[int] = None, end_time: Optional[int] = None
    ) -> List[dict]:
        """Returns filled/cancelled orders for `symbol`. `start_time`/`end_time`
        are Unix seconds (not ms — this endpoint differs from get_trades).
        Used to resolve a triggered TP/SL plan order's real execution
        order_id: BitMart assigns that execution a brand-new order_id, and
        the only link back to the plan order is `client_order_id`, which is
        formatted as `PLAN_{original_plan_order_id}`."""
        params: Dict[str, Any] = {"symbol": symbol}
        if start_time is not None:
            params["start_time"] = int(start_time)
        if end_time is not None:
            params["end_time"] = int(end_time)
        data = await self._request("GET", "/contract/private/order-history", params=params, auth="keyed")
        return data or []

    # ------------------------------------------------------------------
    # Trading (SIGNED auth)
    # ------------------------------------------------------------------

    async def submit_leverage(self, symbol: str, leverage: int, open_type: str) -> dict:
        body = {"symbol": symbol, "leverage": str(leverage), "open_type": open_type}
        return await self._request("POST", "/contract/private/submit-leverage", body=body, auth="signed")

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
        """side: 1=buy_open_long, 2=buy_close_short, 3=sell_close_long, 4=sell_open_short"""
        body: Dict[str, Any] = {"symbol": symbol, "side": side, "type": order_type, "size": size}
        if order_type == "limit" and price is not None:
            body["price"] = price
        if leverage is not None:
            body["leverage"] = str(leverage)
        if open_type is not None:
            body["open_type"] = open_type
        if mode is not None:
            body["mode"] = mode
        if client_order_id is not None:
            body["client_order_id"] = client_order_id
        return await self._request("POST", "/contract/private/submit-order", body=body, auth="signed")

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
        """order_type: 'take_profit' or 'stop_loss'.
        side: 2=close short, 3=close long (hedge mode)."""
        body: Dict[str, Any] = {
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "trigger_price": trigger_price,
            "executive_price": executive_price,
            "price_type": price_type,
            "plan_category": plan_category,
            "category": category,
        }
        if size is not None:
            body["size"] = size
        return await self._request("POST", "/contract/private/submit-tp-sl-order", body=body, auth="signed")

    async def cancel_order(self, symbol: str, order_id: Optional[str] = None) -> dict:
        body: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            body["order_id"] = order_id
        return await self._request("POST", "/contract/private/cancel-order", body=body, auth="signed")
