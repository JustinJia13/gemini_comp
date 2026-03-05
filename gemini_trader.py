"""
Gemini Prediction Markets — authenticated order placer.

Handles HMAC-SHA384 request signing, contract-symbol resolution, and order
placement/cancellation.  Used by live_trading_sim.py to place real $20 bets
when the simulation signals an entry.

Authentication:
  Same HMAC-signed-payload scheme as the main Gemini spot exchange.
  Requires an API key with *NewOrder* and *CancelOrder* permissions.
  Generate keys at: https://exchange.gemini.com/settings/api

Key design decision — instrumentSymbol cache:
  The Gemini order API identifies contracts by their ``instrumentSymbol``
  (e.g. ``GEMI-BTC2603042200-HI69000``), but the simulation tracks contracts
  by their numeric ``contract_id`` (e.g. ``3925-31234``).

  GeminiTrader maintains a TTL-based cache:  contract_id → instrumentSymbol.
  The cache is built by hitting ``/v1/prediction-markets/events`` and is
  auto-refreshed every ``cache_ttl_sec`` seconds (default 300 s / 5 min).
  A cache miss triggers an *immediate* refresh — so new hourly contracts that
  appear each hour are picked up automatically on the first trade attempt.

Usage:
    trader = GeminiTrader(api_key="YOUR_KEY", api_secret="YOUR_SECRET")
    result = trader.place_order(
        contract_id="3925-31234",
        side="NO",           # "YES" or "NO"
        ask_price=0.09,      # from the prediction CSV (ask_no / ask_yes)
    )
    # result → {"order_id": 12345, "n_contracts": 222, "filled": 210, ...}

Sandbox mode (no real money):
    trader = GeminiTrader(..., sandbox=True)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time
from typing import Optional

import requests
from requests import HTTPError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_PROD    = "https://api.gemini.com"
_BASE_SANDBOX = "https://api.sandbox.gemini.com"

_EVENTS_PATH    = "/v1/prediction-markets/events"
_ORDER_PATH     = "/v1/prediction-markets/order"
_CANCEL_PATH    = "/v1/prediction-markets/order/cancel"
_POSITIONS_PATH = "/v1/prediction-markets/positions"


# ---------------------------------------------------------------------------
# GeminiTrader
# ---------------------------------------------------------------------------

class GeminiTrader:
    """Authenticated Gemini Prediction Markets order placer.

    Parameters
    ----------
    api_key:
        Public API key string.
    api_secret:
        Secret API key string (plain text; stored as bytes internally).
    sandbox:
        If True, hit the sandbox exchange (no real money).
    bet_dollars:
        Default notional per bet; overrideable per call.
    cache_ttl_sec:
        How long (seconds) the contract_id → instrumentSymbol cache is
        considered fresh before a background refresh.  New contracts that
        appear mid-session cause an immediate forced refresh on cache miss.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        sandbox:       bool  = False,
        bet_dollars:   float = 20.0,
        cache_ttl_sec: float = 300.0,
    ) -> None:
        self.api_key    = api_key
        self._secret    = api_secret.encode()
        self.base_url   = _BASE_SANDBOX if sandbox else _BASE_PROD
        self.bet_dollars   = bet_dollars
        self.cache_ttl_sec = cache_ttl_sec
        self.session       = requests.Session()

        # contract_id (e.g. "3925-31234") → instrumentSymbol (e.g. "GEMI-BTC...")
        self._symbol_cache: dict[str, str] = {}
        self._cache_ts: float = 0.0         # epoch seconds of last full refresh

        # Monotonically-increasing nonce in EPOCH SECONDS.
        # Prediction-market private endpoints reject nonces that are not within
        # ~30 seconds of server time, so millisecond nonces are too large.
        self._nonce_lock = threading.Lock()
        self._last_nonce: int = 0

    # ── HMAC Auth ─────────────────────────────────────────────────────────────

    def _next_nonce(self) -> int:
        """Return a strictly-increasing nonce in epoch seconds.

        Gemini prediction-market private APIs validate nonce against server
        wall-clock seconds (must be close to current time), so we keep nonce
        in seconds and monotonic. If burst traffic pushes nonce too far ahead,
        wait briefly to stay inside the acceptance window.
        """
        with self._nonce_lock:
            while True:
                now_sec = int(time.time())
                candidate = now_sec if now_sec > self._last_nonce else self._last_nonce + 1
                # Keep some headroom under the server's ±30s guard.
                if candidate <= now_sec + 25:
                    self._last_nonce = candidate
                    return candidate
                time.sleep(1.0)

    def _signed_headers(self, endpoint: str, params: dict) -> dict:
        """Return the three X-GEMINI-* headers for a private POST request."""
        nonce   = self._next_nonce()
        payload = {"request": endpoint, "nonce": nonce, **params}
        encoded_bytes = base64.b64encode(json.dumps(payload).encode())
        encoded_str   = encoded_bytes.decode()          # header must be a string
        sig = hmac.new(self._secret, encoded_bytes, hashlib.sha384).hexdigest()
        return {
            "Content-Type":       "text/plain",
            "Content-Length":     "0",
            "Cache-Control":      "no-cache",
            "X-GEMINI-APIKEY":    self.api_key,
            "X-GEMINI-PAYLOAD":   encoded_str,
            "X-GEMINI-SIGNATURE": sig,
        }

    def _post(self, endpoint: str, params: dict | None = None) -> dict:
        headers = self._signed_headers(endpoint, params or {})
        r = self.session.post(self.base_url + endpoint, headers=headers, timeout=15)
        try:
            r.raise_for_status()
        except HTTPError as exc:
            try:
                body = r.json()
            except Exception:
                body = r.text
            raise RuntimeError(
                f"Gemini API POST {endpoint} failed ({r.status_code}): {body}"
            ) from exc
        return r.json()

    # ── Symbol cache (contract_id → instrumentSymbol) ─────────────────────────

    def refresh_symbol_cache(self, force: bool = False) -> None:
        """Fetch /v1/prediction-markets/events and rebuild the cache.

        Called automatically:
          - On any cache miss (force=True)
          - Every cache_ttl_sec seconds from within resolve_symbol()
        """
        now = time.time()
        if not force and (now - self._cache_ts) < self.cache_ttl_sec and self._symbol_cache:
            return

        try:
            r = self.session.get(self.base_url + _EVENTS_PATH, timeout=15)
            r.raise_for_status()
            data   = r.json()
            events = data.get("data", data) if isinstance(data, dict) else data
            for event in events:
                for contract in event.get("contracts", []):
                    cid = contract.get("id", "")
                    sym = contract.get("instrumentSymbol", "")
                    if cid and sym:
                        self._symbol_cache[cid] = sym
            self._cache_ts = now
        except Exception as exc:
            print(f"  [trader] symbol cache refresh failed: {exc}")

    def resolve_symbol(self, contract_id: str) -> str:
        """Return instrumentSymbol for a contract_id, refreshing cache if needed.

        New hourly contracts that weren't present at startup trigger an immediate
        forced refresh so that the first trade on a new contract succeeds.
        """
        # Try TTL-based refresh first (no-op if fresh)
        self.refresh_symbol_cache(force=False)

        if contract_id not in self._symbol_cache:
            # Cache miss → force refresh (new contract appeared this hour)
            self.refresh_symbol_cache(force=True)

        return self._symbol_cache.get(contract_id, "")

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_order(
        self,
        contract_id: str,
        side:        str,          # "YES" or "NO"
        ask_price:   float,        # e.g. 0.09  (9 cents per contract)
        bet_dollars: Optional[float] = None,
        limit_price: Optional[float] = None,  # optional aggressive limit >= ask
    ) -> dict:
        """Place a limit IOC buy order for ~bet_dollars worth of YES/NO contracts.

        Uses ``immediate-or-cancel`` so the order fills at or better than
        ``limit_price`` (defaults to ``ask_price``) and any unfilled remainder
        is cancelled automatically —
        no dangling GTC resting orders.

        Parameters
        ----------
        contract_id:
            Numeric contract ID from the prediction CSV (e.g. ``"3925-31234"``).
        side:
            ``"YES"`` or ``"NO"`` — which outcome to buy.
        ask_price:
            Current best ask for the chosen side (0.0–1.0 scale, e.g. 0.09).
        limit_price:
            Price sent to Gemini as IOC limit price.  If None, uses ask_price.
            Use a small premium above ask_price when you want higher fill odds.
        bet_dollars:
            Target notional.  Defaults to ``self.bet_dollars`` (20.0).

        Returns
        -------
        dict with keys:
            order_id    — Gemini order ID (int or None if parse fails)
            n_contracts — number of contracts requested
            bet_dollars — actual notional = n_contracts × ask_price
            filled      — contracts filled (from response)
            status      — order status string from API
            symbol      — instrumentSymbol used
            raw         — full API response dict
        """
        dollars = bet_dollars if bet_dollars is not None else self.bet_dollars
        if not ask_price or ask_price <= 0:
            raise ValueError(
                f"[trader] Invalid ask_price={ask_price!r}: must be a positive float."
            )
        px = float(limit_price if limit_price is not None else ask_price)
        if px <= 0 or px > 1:
            raise ValueError(
                f"[trader] Invalid limit_price={px!r}: must be in (0, 1]."
            )
        # Size off limit price so we don't exceed target spend in worst-case fill.
        n_contracts = max(1, int(dollars / px))

        symbol = self.resolve_symbol(contract_id)
        if not symbol:
            raise ValueError(
                f"[trader] Cannot resolve instrumentSymbol for contract_id={contract_id!r}. "
                "Is the events API reachable and the contract still live?"
            )

        params = {
            "symbol":      symbol,
            "orderType":   "limit",
            "side":        "buy",
            "outcome":     side.lower(),              # "yes" or "no"
            "price":       str(round(px, 4)),
            "quantity":    str(n_contracts),
            "timeInForce": "immediate-or-cancel",     # fill & kill; no GTC remainder
        }

        resp     = self._post(_ORDER_PATH, params)
        order_id = resp.get("orderId") or resp.get("order_id")
        filled   = int(float(
            resp.get("filledQuantity") or resp.get("filled_quantity") or 0
        ))
        status   = resp.get("status", "unknown")

        # Try to extract the actual average execution price from the response.
        # Field names vary by exchange version; fall back to None if absent.
        avg_price: Optional[float] = None
        for field in ("avgExecutionPrice", "avg_execution_price",
                      "executionPrice",    "execution_price"):
            raw_price = resp.get(field)
            if raw_price is not None:
                try:
                    avg_price = float(raw_price)
                    break
                except (TypeError, ValueError):
                    pass

        fill_price = avg_price if avg_price is not None else px

        return {
            "order_id":    order_id,
            "n_contracts": n_contracts,
            "bet_dollars": round(n_contracts * fill_price, 4),
            "filled":      filled,
            "avg_price":   avg_price,   # None if API doesn't return it
            "limit_price": round(px, 4),
            "status":      status,
            "symbol":      symbol,
            "raw":         resp,
        }

    def sell_order(
        self,
        contract_id:  str,
        side:         str,    # "YES" or "NO" — the outcome we're selling
        bid_price:    float,  # current best bid (what the market pays us)
        n_contracts:  int,    # exactly how many contracts to sell
    ) -> dict:
        """Place a limit IOC sell order to exit an open position.

        Mirror of place_order() but with ``"side": "sell"``.  Using IOC ensures
        the sell fills immediately at ``bid_price`` or better; any unfilled
        remainder is cancelled — no dangling resting sell orders.

        Parameters
        ----------
        contract_id:
            Same numeric ID that was used to enter the position.
        side:
            The outcome that was purchased (``"YES"`` or ``"NO"``).
        bid_price:
            Current best bid for that outcome (0.0–1.0 scale, e.g. 0.72).
        n_contracts:
            Exact number of contracts held (should equal the buy ``filled``
            count, not the originally requested quantity).

        Returns
        -------
        dict with keys: order_id, n_contracts, filled, status, symbol, raw
        """
        symbol = self.resolve_symbol(contract_id)
        if not symbol:
            raise ValueError(
                f"[trader] Cannot resolve instrumentSymbol for contract_id={contract_id!r}."
            )

        params = {
            "symbol":      symbol,
            "orderType":   "limit",
            "side":        "sell",
            "outcome":     side.lower(),               # "yes" or "no"
            "price":       str(round(bid_price, 4)),
            "quantity":    str(n_contracts),
            "timeInForce": "immediate-or-cancel",
        }

        resp     = self._post(_ORDER_PATH, params)
        order_id = resp.get("orderId") or resp.get("order_id")
        filled   = int(float(
            resp.get("filledQuantity") or resp.get("filled_quantity") or 0
        ))
        status   = resp.get("status", "unknown")

        return {
            "order_id":    order_id,
            "n_contracts": n_contracts,
            "filled":      filled,
            "status":      status,
            "symbol":      symbol,
            "raw":         resp,
        }

    def cancel_order(self, order_id: int) -> dict:
        """Cancel an open order by orderId."""
        return self._post(_CANCEL_PATH, {"orderId": order_id})

    def get_order_status(self, order_id: int) -> dict:
        """Best-effort order status snapshot.

        Gemini prediction-markets currently does not expose
        `/v1/prediction-markets/order/status` (returns EndpointNotFound).
        As a fallback, return a positions snapshot so callers can verify
        whether exposure exists after submitting an IOC order.
        """
        return {
            "order_id": order_id,
            "status_supported": False,
            "message": "Prediction-markets order status endpoint is unavailable; using positions snapshot.",
            "positions": self.get_positions(),
        }

    def get_positions(self) -> list[dict]:
        """Return a list of open prediction-market positions."""
        resp = self._post(_POSITIONS_PATH)
        if isinstance(resp, list):
            return resp
        return resp.get("positions", [])

    def get_balances(self) -> list[dict]:
        """Return account balances (USD cash + any crypto held).

        Response is a list of dicts with keys: currency, amount, available.
        """
        return self._post("/v1/account/balances")
