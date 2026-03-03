"""
Gemini 1-minute OHLCV updater — runs continuously alongside live_trading_sim.py.

Fetches the latest 1-minute candles for BTC/ETH/SOL from the Gemini public API
and appends new bars to the rolling data files in .data/gemini/ohlcv_1m_7d/.

IMPORTANT API LIMITATION:
  Gemini's /v2/candles/{symbol}/1m endpoint returns at most 1440 candles per call
  (the most recent 24 hours).  The timestamp parameter is ignored — it always
  returns the latest 1440 bars.  If this script is not run for > 24 hours, the
  gap cannot be backfilled.  Run at least once every 24 hours (recommend: every
  60 seconds alongside the simulation).

Output files (appended in-place, never truncated):
  .data/gemini/ohlcv_1m_7d/{symbol}.data        — raw UTC ms timestamps
  .data/gemini/ohlcv_1m_7d/{symbol}_est.data    — EST sidecar (read by sim)

Run:
    python getdata_underlying.py                  # default 60s poll
    python getdata_underlying.py --poll-sec 30
    python getdata_underlying.py --symbols btcusd ethusd   # subset
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL  = "https://api.gemini.com"
EST_TZ    = "America/New_York"
OUT_DIR   = Path(".data/gemini/ohlcv_1m_7d")
SYMBOLS   = ["btcusd", "ethusd", "solusd"]

# Gemini returns at most 1440 bars per call for 1m timeframe (24 hours).
CANDLES_PER_CALL = 1440
TF_MS = {"1m": 60_000}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_json(session: requests.Session, path: str, max_retries: int = 6):
    url = BASE_URL + path
    backoff = 0.5
    for _ in range(max_retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2, 10.0)
    raise RuntimeError(f"Failed GET {BASE_URL + path}")


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _last_timestamp_ms(file_path: Path) -> int:
    """Return the last timestamp_ms recorded in the file, or -1 if empty/missing."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return -1

    # Fast tail-read (avoids loading the whole file).
    with file_path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        chunk = min(size, 8192)
        f.seek(-chunk, 2)
        data = f.read(chunk)

    for line in reversed(data.splitlines()):
        line = line.strip()
        if not line or line.startswith(b"timestamp_ms"):
            continue
        parts = line.split(b",")
        try:
            return int(parts[0])
        except Exception:
            continue

    # Fallback: full scan.
    last_ts = -1
    with file_path.open("r", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] == "timestamp_ms":
                continue
            try:
                last_ts = int(row[0])
            except Exception:
                pass
    return last_ts


def _latest_closed_ts_ms() -> int:
    """Unix ms of the most recently fully-closed 1-minute candle."""
    now_ms = int(time.time() * 1000)
    return (now_ms // 60_000) * 60_000 - 60_000


def _append_rows(file_path: Path, rows: list) -> None:
    is_new = not file_path.exists() or file_path.stat().st_size == 0
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp_ms", "open", "high", "low", "close", "volume"])
        w.writerows(rows)


def _write_est_sidecar(source: Path) -> Path:
    """Rewrite the full EST sidecar from the source file."""
    est_path = source.with_name(f"{source.stem}_est.data")
    tz = ZoneInfo(EST_TZ)
    with source.open("r", newline="") as f_in, \
         est_path.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fields = ["timestamp_ms", "timestamp_est", "open", "high", "low", "close", "volume"]
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            ts = int(row["timestamp_ms"])
            dt_est = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).astimezone(tz)
            writer.writerow({
                "timestamp_ms":  ts,
                "timestamp_est": dt_est.strftime("%Y-%m-%d %H:%M:%S%z"),
                "open":   row["open"],
                "high":   row["high"],
                "low":    row["low"],
                "close":  row["close"],
                "volume": row["volume"],
            })
    return est_path


# ---------------------------------------------------------------------------
# Per-symbol update
# ---------------------------------------------------------------------------

def _update_symbol(session: requests.Session, symbol: str) -> int:
    """Fetch new 1-min candles for symbol; return number of new bars appended."""
    raw_path  = OUT_DIR / f"{symbol}.data"
    last_ts   = _last_timestamp_ms(raw_path)
    closed_ts = _latest_closed_ts_ms()

    if last_ts >= closed_ts:
        return 0   # already up-to-date

    # Warn if the gap is wider than the API window.
    if last_ts > 0:
        gap_min = (closed_ts - last_ts) / 60_000
        if gap_min > CANDLES_PER_CALL:
            print(f"  WARNING [{symbol}]: gap {gap_min:.0f} min exceeds API window "
                  f"({CANDLES_PER_CALL} min) — {gap_min - CANDLES_PER_CALL:.0f} min of data lost")

    candles = _get_json(session, f"/v2/candles/{symbol}/1m")

    new_rows = []
    for row in candles:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        ts = int(row[0])
        if ts > last_ts and ts <= closed_ts:
            new_rows.append([ts, float(row[1]), float(row[2]),
                             float(row[3]), float(row[4]), float(row[5])])
    new_rows.sort(key=lambda x: x[0])

    if new_rows:
        _append_rows(raw_path, new_rows)
        _write_est_sidecar(raw_path)

    return len(new_rows)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(symbols: List[str] = SYMBOLS, poll_sec: int = 60) -> None:
    print(f"Starting OHLCV updater  symbols={symbols}  poll={poll_sec}s")
    print(f"Output → {OUT_DIR.resolve()}")
    print()

    with requests.Session() as session:
        while True:
            t0  = time.time()
            now = datetime.now(tz=ZoneInfo(EST_TZ)).strftime("%H:%M:%S")
            parts = []
            for sym in symbols:
                try:
                    n = _update_symbol(session, sym)
                    parts.append(f"{sym}=+{n}")
                except Exception as e:
                    parts.append(f"{sym}=ERR({e})")
            print(f"[{now}]  {',  '.join(parts)}")
            elapsed = time.time() - t0
            time.sleep(max(0.0, poll_sec - elapsed))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gemini 1-minute OHLCV updater",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--poll-sec", type=int, default=60,
                        help="Seconds between update cycles")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS,
                        help="Gemini symbols to track")
    args = parser.parse_args()
    run(symbols=args.symbols, poll_sec=args.poll_sec)
