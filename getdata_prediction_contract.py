"""
Gemini prediction market data collector.

Polls /v1/prediction-markets/events every POLL_SEC seconds and records
live bid/ask prices for BTC/ETH/SOL prediction contracts.

Two collection modes (controlled by config.toml → data_collector.hourly_only):

  hourly_only = true  (default)
    Tracks only the NEXT HOURLY EST contract:
      - At 7:20 PM EST → 8 PM EST contract only.
      - At 8:01 PM EST → 9 PM EST contract only.
    Gemini typically stops listing hourly contracts after ~8 PM EST.

  hourly_only = false
    Collects ALL BTC/ETH/SOL contracts settling within max_hours_to_collect
    hours.  Use this after hourly contracts are no longer listed (e.g. late
    evening / overnight) to capture daily/multi-day contracts.

Saved to: .data/gemini/prediction_data/<YYYYMMDD>.csv  (one file per day,
rows appended on each poll so the process can be restarted without data loss).

Run:
    python getdata_prediction_contract.py
    python getdata_prediction_contract.py --poll-sec 30
    python getdata_prediction_contract.py --all-contracts          # hourly_only=false
    python getdata_prediction_contract.py --all-contracts --max-hours 12
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from btc_hourly_model import parse_event_ticker
import config_loader as cfg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE    = "https://api.gemini.com"
EST_TZ  = ZoneInfo("America/New_York")
OUT_ROOT = Path(".data/gemini/prediction_data")

# Regex to parse strike from contract label, e.g. "BTC > $69,000"
_LABEL_RE = re.compile(r"[\$]?([\d,]+(?:\.\d+)?)\s*$")

CSV_FIELDS = [
    "timestamp_utc",
    "timestamp_est",
    "event_ticker",
    "event_title",
    "contract_id",
    "contract_label",
    "asset",
    "strike",
    "direction",
    "settle_time_utc",
    "hours_to_settle",
    # market prices
    "bid_yes",
    "ask_yes",
    "bid_no",
    "ask_no",
    "mid_yes",
    "last_trade_price",
    # pure market arb signal (buy-only exchange — shorting impossible)
    # arb_long = ask_yes + ask_no - 1.0
    #   < 0 → buy both YES + NO for < $1, guaranteed $1 payout → pure arb
    #   > 0 → buying both costs > $1 → guaranteed loss (NOT arb)
    "arb_long",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_json(session: requests.Session, path: str, max_retries: int = 4) -> dict | list:
    url = BASE + path
    backoff = 1.0
    for _ in range(max_retries):
        try:
            r = session.get(url, timeout=15)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
    raise RuntimeError(f"Failed GET {url}")


def _parse_strike(label: str) -> float | None:
    """Extract numeric strike from label like 'BTC > $69,000'."""
    m = _LABEL_RE.search(label)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_direction(label: str) -> str:
    return "above" if ">" in label else "below"


def _safe_float(val: str | None) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _next_target_settle_utc(now_utc: datetime) -> datetime:
    """Return the next whole UTC hour strictly after now.

    Examples (EST = UTC-5):
      7:20 PM EST  →  00:20 UTC  →  target = 01:00 UTC  (8 PM EST)
      8:01 PM EST  →  01:01 UTC  →  target = 02:00 UTC  (9 PM EST)
      8:00:00 EST  →  01:00 UTC  →  target = 02:00 UTC  (already settled)
    """
    base = now_utc.replace(minute=0, second=0, microsecond=0)
    return base + timedelta(hours=1)


def _output_path() -> Path:
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUT_ROOT / f"{today}.csv"


def _ensure_header(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def _append_rows(path: Path, rows: list[dict]) -> None:
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Core poll loop
# ---------------------------------------------------------------------------

def poll_once(
    session: requests.Session,
    *,
    hourly_only: bool = True,
    max_hours_to_collect: float = 24.0,
) -> list[dict]:
    """Fetch active events; return rows for matching BTC/ETH/SOL contracts.

    hourly_only=True  (default)
        Targets exactly the contract settling at the next whole UTC hour
        (= next whole EST hour) ± 2 minutes, AND the event title must contain
        "EST".  This excludes all multi-day/weekly contracts.

    hourly_only=False
        Collects any BTC/ETH/SOL contract settling within max_hours_to_collect
        hours from now, regardless of title.  Use this when hourly contracts are
        no longer listed (typically after ~8 PM EST).
    """
    now_utc = datetime.now(tz=timezone.utc)
    now_est = now_utc.astimezone(EST_TZ)

    # Only needed in hourly_only mode, but compute once up-front.
    target_settle = _next_target_settle_utc(now_utc)

    data   = _get_json(session, "/v1/prediction-markets/events")
    events = data.get("data", data) if isinstance(data, dict) else data

    rows = []
    for event in events:
        ticker       = event.get("ticker", "")
        asset, settle_utc = parse_event_ticker(ticker)
        if asset is None:
            continue  # not a BTC/ETH/SOL price event

        hours_to_settle = (settle_utc - now_utc).total_seconds() / 3600.0
        if hours_to_settle < 0:
            continue  # already settled

        title = event.get("title", "")

        if hourly_only:
            # Accept only contracts that settle within 2 minutes of the target hour.
            diff_sec = abs((settle_utc - target_settle).total_seconds())
            if diff_sec > 120:
                continue
            # Only keep hourly EST contracts: "BTC price today at 8pm EST".
            # Daily/weekly events ("BTC price on March 6") never contain "EST".
            if "EST" not in title:
                continue
        else:
            # All-contracts mode: apply horizon guard only.
            if hours_to_settle > max_hours_to_collect:
                continue

        for contract in event.get("contracts", []):
            label  = contract.get("label", "")
            strike = _parse_strike(label)
            if strike is None:
                continue

            direction = _parse_direction(label)
            prices    = contract.get("prices", {})

            bid_yes = _safe_float(prices.get("sell", {}).get("yes"))
            ask_yes = _safe_float(prices.get("buy",  {}).get("yes"))
            bid_no  = _safe_float(prices.get("sell", {}).get("no"))
            ask_no  = _safe_float(prices.get("buy",  {}).get("no"))
            last    = _safe_float(prices.get("lastTradePrice"))

            mid_yes  = ((bid_yes or 0) + (ask_yes or 0)) / 2 if (bid_yes and ask_yes) \
                       else (bid_yes or ask_yes)
            arb_long = round(ask_yes + ask_no - 1.0, 4) \
                       if (ask_yes is not None and ask_no is not None) else None

            rows.append({
                "timestamp_utc":    now_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "timestamp_est":    now_est.strftime("%Y-%m-%d %H:%M:%S%z"),
                "event_ticker":     ticker,
                "event_title":      title,
                "contract_id":      contract.get("id", ""),
                "contract_label":   label,
                "asset":            asset,
                "strike":           strike,
                "direction":        direction,
                "settle_time_utc":  settle_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "hours_to_settle":  round(hours_to_settle, 4),
                "bid_yes":          bid_yes,
                "ask_yes":          ask_yes,
                "bid_no":           bid_no,
                "ask_no":           ask_no,
                "mid_yes":          mid_yes,
                "last_trade_price": last,
                "arb_long":         arb_long,
            })

    return rows


def run(
    poll_sec: int = 60,
    hourly_only: bool = True,
    max_hours_to_collect: float = 24.0,
) -> None:
    """Main loop: poll → save, indefinitely."""
    if hourly_only:
        mode_str = "next 1-hour EST contract only"
    else:
        mode_str = f"all contracts  max_hours={max_hours_to_collect}h"

    print(f"Starting prediction market collector  poll={poll_sec}s  ({mode_str})")
    print(f"Output → {OUT_ROOT.resolve()}")

    with requests.Session() as session:
        while True:
            t0      = time.time()
            now_utc = datetime.now(tz=timezone.utc)

            if hourly_only:
                target     = _next_target_settle_utc(now_utc)
                target_str = target.astimezone(EST_TZ).strftime("%I:%M %p %Z")
                ctx_str    = f"target={target_str}"
            else:
                ctx_str    = f"max={max_hours_to_collect}h"

            try:
                rows = poll_once(
                    session,
                    hourly_only=hourly_only,
                    max_hours_to_collect=max_hours_to_collect,
                )
            except Exception as e:
                print(f"[{datetime.now(tz=EST_TZ).strftime('%H:%M:%S')}] poll error: {e}")
                rows = []

            if rows:
                out = _output_path()
                _ensure_header(out)
                _append_rows(out, rows)
                n_arb = sum(1 for r in rows if r.get("arb_long") is not None and r["arb_long"] < 0)
                print(
                    f"[{datetime.now(tz=EST_TZ).strftime('%H:%M:%S')}]  "
                    f"{ctx_str}  rows={len(rows)}  arb_signals={n_arb}  → {out.name}"
                )
            else:
                print(
                    f"[{datetime.now(tz=EST_TZ).strftime('%H:%M:%S')}]  "
                    f"{ctx_str}  no contracts found"
                )

            elapsed = time.time() - t0
            time.sleep(max(0.0, poll_sec - elapsed))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _hourly_only_default        = cfg.get("data_collector", "hourly_only", True)
    _max_hours_default          = cfg.get("data_collector", "max_hours_to_collect", 24.0)

    parser = argparse.ArgumentParser(
        description="Gemini prediction market data collector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--poll-sec", type=int,
        default=cfg.get("data_collector", "poll_sec", 60),
        help="Seconds between polls",
    )
    parser.add_argument(
        "--all-contracts", action="store_true",
        default=not _hourly_only_default,
        help="Collect all contracts within --max-hours (overrides hourly_only=true in config)",
    )
    parser.add_argument(
        "--max-hours", type=float,
        default=_max_hours_default,
        help="Max hours to settle; used only when --all-contracts is active",
    )
    args = parser.parse_args()

    run(
        poll_sec=args.poll_sec,
        hourly_only=not args.all_contracts,
        max_hours_to_collect=args.max_hours,
    )
