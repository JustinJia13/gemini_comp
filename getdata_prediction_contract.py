"""
Download/log Gemini *hourly* crypto prediction markets data (BTC/ETH/SOL only)
and store under: .data/gemini/prediction_data/

What it captures
1) Event/contract metadata (including resolution timestamp when available)
2) Live market data for each contract instrumentSymbol:
   - ticker (/v2/ticker/{symbol})
   - order book (/v1/book/{symbol})
   - trades (/v1/trades/{symbol})

It targets the “BTC price today at 1am EST” style markets (hourly).
It will:
- Backfill metadata for settled/active events (no LOB history—just metadata/outcome)
- Then continuously discover ACTIVE hourly events and log live data until you stop it.

Run:
  python gemini_hourly_predictions_logger.py

Optional:
  python gemini_hourly_predictions_logger.py --poll-sec 1 --refresh-sec 30 --book-levels 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


BASE = "https://api.gemini.com"
OUT_ROOT = os.path.join(".data", "gemini", "prediction_data")

UNDERLYINGS = ("BTC", "ETH", "SOL")

# These hourly markets on Gemini typically look like:
#   "BTC price today at 1am EST"
#   "ETH price today at 4pm EST"
# More flexible "hourly" detection: match "today at <hour>(:mm)? (am/pm) EST"
HOURLY_ANY_RE = re.compile(
    r"\b(BTC|ETH|SOL)\b.*\bprice\b.*\btoday\b.*\bat\b.*\b(\d{1,2})(:\d{2})?\s*(am|pm)\s*est\b",
    re.IGNORECASE,
)
TIME_KEYWORDS = [
    "1AM","2AM","3AM","4AM","5AM","6AM","7AM","8AM","9AM","10AM","11AM","12AM",
    "1PM","2PM","3PM","4PM","5PM","6PM","7PM","8PM","9PM","10PM","11PM","12PM",
    "EST","EDT","ET"
]

def is_hourly_text(text: str) -> bool:
    up = text.upper()
    return any(k in up for k in TIME_KEYWORDS) and ("PRICE" in up)

def all_strings(x: Any) -> List[str]:
    """Collect all string values inside nested dict/list JSON."""
    out: List[str] = []
    if isinstance(x, dict):
        for v in x.values():
            out.extend(all_strings(v))
    elif isinstance(x, list):
        for v in x:
            out.extend(all_strings(v))
    elif isinstance(x, str):
        out.append(x)
    return out

def extract_text_anywhere(event: Dict[str, Any]) -> str:
    """Join every string field so we don't depend on 'title' existing."""
    return " ".join(all_strings(event)).strip()

def is_candidate_crypto_event(event: Dict[str, Any]) -> bool:
    text = extract_text_anywhere(event)
    if not text:
        return False
    up = text.upper()
    return any(u in up for u in UNDERLYINGS)  # only BTC/ETH/SOL candidate

def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    mkdirp(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_json(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> Any:
    r = requests.get(BASE + path, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_events(status: str) -> List[Dict[str, Any]]:
    # Docs: /v1/prediction-markets/events
    # We pass status=active or status=settled where supported.
    data = get_json("/v1/prediction-markets/events", params={"status": status})
    return data if isinstance(data, list) else []


def extract_contracts(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Common key is "contracts"; fallback to others just in case.
    c = event.get("contracts") or event.get("outcomes") or event.get("markets") or []
    return c if isinstance(c, list) else []


def contract_instrument_symbol(contract: Dict[str, Any]) -> Optional[str]:
    sym = contract.get("instrumentSymbol") or contract.get("instrument_symbol") or contract.get("symbol")
    if sym is None:
        return None
    sym = str(sym).strip()
    return sym or None
def find_instrument_symbols_anywhere(x: Any) -> List[str]:
    syms: List[str] = []

    def walk(v: Any):
        if isinstance(v, dict):
            for k, vv in v.items():
                if k in ("instrumentSymbol", "instrument_symbol") and isinstance(vv, str) and vv.strip():
                    syms.append(vv.strip())
                walk(vv)
        elif isinstance(v, list):
            for vv in v:
                walk(vv)

    walk(x)
    return sorted(set(syms))
def fetch_event_detail(ticker: str) -> Dict[str, Any]:
    # Some deployments include contracts only in the per-event detail.
    return get_json(f"/v1/prediction-markets/events/{ticker}")

def underlying_from_text(text: str) -> Optional[str]:
    up = text.upper()
    for u in UNDERLYINGS:
        if u in up:
            return u
    return None


def fetch_ticker(symbol: str) -> Dict[str, Any]:
    return get_json(f"/v2/ticker/{symbol}")


def fetch_book(symbol: str, book_levels: int) -> Dict[str, Any]:
    # Try limiting params; if not supported, fall back.
    try:
        return get_json(f"/v1/book/{symbol}", params={"limit_bids": book_levels, "limit_asks": book_levels})
    except requests.HTTPError:
        return get_json(f"/v1/book/{symbol}")


def fetch_trades(symbol: str, since_tid: Optional[int]) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {}
    if since_tid is not None:
        params["since_tid"] = since_tid
    data = get_json(f"/v1/trades/{symbol}", params=params)
    return data if isinstance(data, list) else []


def dump_event_metadata(event: Dict[str, Any], status: str) -> None:
    # Store all raw event JSON in a single metadata stream.
    path = os.path.join(OUT_ROOT, "events", f"events_{status}.jsonl")
    jsonl_append(path, {"ts": utc_now_iso(), "status": status, "event": event})


def backfill_metadata() -> None:
    # Backfill active + settled metadata so you can build a label dataset of outcomes.
    for status in ("active", "settled"):
        try:
            events = fetch_events(status=status)
        except Exception as e:
            print(f"[warn] backfill fetch_events(status={status}) failed: {e}")
            continue

        kept = 0
        for ev in events:
            if isinstance(ev, dict) and is_hourly_crypto_event(ev):
                dump_event_metadata(ev, status=status)
                kept += 1
        print(f"[backfill] status={status}: kept {kept} hourly BTC/ETH/SOL events")


def build_active_universe() -> Dict[str, Dict[str, Any]]:
    universe: Dict[str, Dict[str, Any]] = {}
    events = fetch_events(status="active")

    candidates = 0
    hourly_hits = 0

    for ev in events:
        if not isinstance(ev, dict):
            continue
        if not is_candidate_crypto_event(ev):
            continue

        candidates += 1
        ticker = ev.get("ticker")
        if not ticker:
            continue

        # Always pull detail; list payload may be missing key strings/contracts
        try:
            detail = fetch_event_detail(str(ticker))
        except Exception:
            continue

        detail_text = extract_text_anywhere(detail)
        up = detail_text.upper()

        # Must be BTC/ETH/SOL
        if not any(u in up for u in UNDERLYINGS):
            continue

        # Must look "hourly" via explicit time keywords (1am/2am/... + EST/ET) and "price"
        if not is_hourly_text(detail_text):
            continue

        hourly_hits += 1
        u = underlying_from_text(detail_text) or "UNKNOWN"

        # Try normal contracts extraction first
        contracts = extract_contracts(detail)
        if contracts:
            for c in contracts:
                if not isinstance(c, dict):
                    continue
                sym = contract_instrument_symbol(c)
                if not sym:
                    continue
                universe[sym] = {
                    "underlying": u,
                    "event_text": detail_text,
                    "event": detail,
                    "contract": c,
                }
        else:
            # Fallback: search entire JSON for instrumentSymbol keys
            syms = find_instrument_symbols_anywhere(detail)
            for sym in syms:
                universe[sym] = {
                    "underlying": u,
                    "event_text": detail_text,
                    "event": detail,
                    "contract": {},
                }

    print(f"[debug] active events={len(events)} candidates(BTC/ETH/SOL)={candidates} hourly_hits={hourly_hits}")
    return universe

def write_universe_snapshot(universe: Dict[str, Dict[str, Any]]) -> None:
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_ROOT, "universe", f"hourly_active_universe_{ts}.json")
    mkdirp(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: {"underlying": v["underlying"], "event_text": v["event_text"]} for k, v in universe.items()},
                  f, ensure_ascii=False, indent=2)


def live_log_loop(poll_sec: float, refresh_sec: float, book_levels: int) -> None:
    last_refresh = 0.0
    universe: Dict[str, Dict[str, Any]] = {}
    last_tid: Dict[str, int] = {}

    print(f"[start] writing under {OUT_ROOT}")
    while True:
        now = time.time()

        if (now - last_refresh) >= refresh_sec or not universe:
            try:
                universe = build_active_universe()
                write_universe_snapshot(universe)
                last_refresh = now
                print(f"[universe] active hourly instruments: {len(universe)}")
            except Exception as e:
                print(f"[warn] universe refresh failed: {e}")

        if not universe:
            time.sleep(5)
            continue

        for sym, meta in list(universe.items()):
            ts_iso = utc_now_iso()
            day = dt.datetime.utcnow().strftime("%Y%m%d")
            base = os.path.join(OUT_ROOT, "hourly", meta["underlying"], sym, day)
            ticker_path = os.path.join(base, "ticker.jsonl")
            book_path = os.path.join(base, "book.jsonl")
            trades_path = os.path.join(base, "trades.jsonl")
            meta_path = os.path.join(base, "meta.jsonl")

            # meta (event/contract raw, helpful for labeling later)
            jsonl_append(meta_path, {"ts": ts_iso, "symbol": sym, "meta": meta})

            # ticker
            try:
                tick = fetch_ticker(sym)
                jsonl_append(ticker_path, {"ts": ts_iso, "symbol": sym, "ticker": tick})
            except Exception as e:
                jsonl_append(ticker_path, {"ts": ts_iso, "symbol": sym, "error": f"ticker: {e}"})

            # book
            try:
                book = fetch_book(sym, book_levels=book_levels)
                jsonl_append(book_path, {"ts": ts_iso, "symbol": sym, "book": book})
            except Exception as e:
                jsonl_append(book_path, {"ts": ts_iso, "symbol": sym, "error": f"book: {e}"})

            # trades (incremental)
            try:
                since = last_tid.get(sym)
                prints = fetch_trades(sym, since_tid=since)

                # normalize + sort by tid ascending
                good = [p for p in prints if isinstance(p, dict) and p.get("tid") is not None]
                good.sort(key=lambda p: int(p["tid"]))

                for p in good:
                    jsonl_append(trades_path, {"ts": ts_iso, "symbol": sym, "trade": p})

                if good:
                    last_tid[sym] = int(good[-1]["tid"])
            except Exception as e:
                jsonl_append(trades_path, {"ts": ts_iso, "symbol": sym, "error": f"trades: {e}"})

            # small spacing to avoid bursty requests
            time.sleep(0.05)

        time.sleep(poll_sec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-sec", type=float, default=2.0, help="poll interval for each loop")
    ap.add_argument("--refresh-sec", type=float, default=60.0, help="how often to refresh active hourly universe")
    ap.add_argument("--book-levels", type=int, default=10, help="depth levels for /v1/book")
    ap.add_argument("--no-backfill", action="store_true", help="skip metadata backfill step")
    args = ap.parse_args()

    mkdirp(OUT_ROOT)

    if not args.no_backfill:
        backfill_metadata()

    live_log_loop(poll_sec=args.poll_sec, refresh_sec=args.refresh_sec, book_levels=args.book_levels)


if __name__ == "__main__":
    main()