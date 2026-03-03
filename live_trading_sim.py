"""
Live prediction market trading simulation.

Reads:
  .data/gemini/ohlcv_1m_7d/{sym}_est.data        — live underlying OHLCV
  .data/gemini/prediction_data/{YYYYMMDD}.csv     — live contract prices

For each contract seen for the first time:
  - Calibrates GBM and Student's t on live minute returns (no look-ahead)
  - Simulates a buy trade when edge exceeds --min-edge threshold

On every subsequent poll for an open position, checks early-exit conditions:

  1. profit_lock  — bid_now - ask_entry >= --profit-lock  (e.g. 0.05 = 5¢ locked in)
                    Sell early to capture realised gain before convergence reverses.

  2. stop_loss    — ask_entry - bid_now >= --stop-loss    (e.g. 0.10 = 10¢ loss)
                    Sell to cap downside; don't ride to zero.

  3. p_drop       — model p(our side) fell by >= --p-drop from entry p_fair.
                    New data has made us less confident; exit before the market
                    catches up and the bid falls further.

  4. edge_closed  — current edge for our side has collapsed to near zero (edge < 0).
                    The market has repriced to where we thought; take the spread.

After settle_time passes: final settlement via OHLCV spot lookup.

Output:
  .data/gemini/sim_trades/trades_{YYYYMMDD}.csv        — one row per closed position
  .data/gemini/sim_trades/edge_log_{YYYYMMDD}.csv      — edge log at entry (all contracts)
  .data/gemini/sim_trades/performance_ledger.csv       — cumulative model accuracy / P&L
                                                           (updated after every trade closes,
                                                            aggregated from all historical files)

Run:
    python live_trading_sim.py
    python live_trading_sim.py --poll-sec 60 --min-edge 0.03 --profit-lock 0.05 \\
                                --stop-loss 0.10 --p-drop 0.05
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import config_loader as cfg
from btc_hourly_model import (
    StudentTParams,
    _minute_log_returns_before,
    calibrate_gbm_from_log_returns,
    calibrate_student_t_from_log_returns,
    gbm_binary_prob,
    load_gemini_ohlcv,
    student_t_binary_prob,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EST_TZ        = ZoneInfo("America/New_York")
OHLCV_ROOT    = Path(".data/gemini/ohlcv_1m_7d")
CONTRACT_ROOT = Path(".data/gemini/prediction_data")
SIM_ROOT      = Path(".data/gemini/sim_trades")

ASSET_SYMBOLS = {"BTC": "btcusd", "ETH": "ethusd", "SOL": "solusd"}

TRADE_FIELDS = [
    "entry_time_utc",
    "contract_id",
    "event_ticker",
    "asset",
    "strike",
    "direction",
    "settle_time_utc",
    "hours_to_settle_at_entry",
    "side",           # YES or NO
    "model",          # gbm | student_t
    "p_fair",         # model prob for our side at entry
    "ask_price",      # cost to enter (e.g. 0.62)
    "edge",           # p_fair - ask_price at entry
    # Closing fields:
    "exit_time_utc",
    "exit_reason",    # settlement | profit_lock | stop_loss | p_drop | edge_closed
    "exit_bid",       # bid price when we sold (early exit only; None for settlement)
    "spot_at_settle", # underlying spot at settle_time (settlement only)
    "outcome",        # YES_WIN | NO_WIN (settlement only)
    "pnl",            # realised P&L per $1 notional
    "status",         # open | settled | exited_early
]

PERF_LEDGER_FIELDS = [
    "model",
    "n_trades",          # total closed trades (settled + early exits)
    "n_profitable",      # trades where pnl > 0
    "n_loss",            # trades where pnl <= 0
    "n_settled",         # closed via settlement (not early exit)
    "n_early_exit",      # closed via early exit signal
    "win_rate_pct",      # 100 * n_profitable / n_trades
    "avg_pnl",           # mean pnl per trade
    "total_pnl",         # sum of all pnl
    "avg_edge_at_entry", # mean model edge at entry time
    "as_of",             # UTC timestamp of last update
]

EDGE_FIELDS = [
    "timestamp_utc",
    "contract_id",
    "asset",
    "strike",
    "direction",
    "hours_to_settle",
    "spot",
    "ask_yes",
    "ask_no",
    "arb_long",
    "sigma_annual_gbm",
    "p_fair_gbm",
    "edge_yes_gbm",
    "edge_no_gbm",
    "p_fair_stud",
    "edge_yes_stud",
    "edge_no_stud",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _today_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d")


def _trade_out_path() -> Path:
    SIM_ROOT.mkdir(parents=True, exist_ok=True)
    return SIM_ROOT / f"trades_{_today_str()}.csv"


def _edge_out_path() -> Path:
    SIM_ROOT.mkdir(parents=True, exist_ok=True)
    return SIM_ROOT / f"edge_log_{_today_str()}.csv"


def _ensure_header(path: Path, fields: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _append_csv(path: Path, row: dict, fields: list[str]) -> None:
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields, extrasaction="ignore").writerow(row)


def _safe_float(val) -> float | None:
    if not val:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _load_minute_df(asset: str) -> pd.DataFrame | None:
    sym = ASSET_SYMBOLS.get(asset)
    if not sym:
        return None
    fp = OHLCV_ROOT / f"{sym}_est.data"
    if not fp.exists():
        return None
    try:
        return load_gemini_ohlcv(fp)
    except Exception:
        return None


def _read_contract_csv(date_str: str) -> list[dict]:
    fp = CONTRACT_ROOT / f"{date_str}.csv"
    if not fp.exists():
        return []
    try:
        with fp.open(newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _latest_by_contract(rows: list[dict]) -> dict[str, dict]:
    """Return the most recent CSV row per contract_id (rows are in time order)."""
    latest: dict[str, dict] = {}
    for row in rows:
        cid = row.get("contract_id", "")
        if cid:
            latest[cid] = row  # overwrite keeps latest
    return latest


# ---------------------------------------------------------------------------
# Model calibration
# ---------------------------------------------------------------------------

def _compute_edges(
    contract_id: str,
    asset: str,
    strike: float,
    direction: str,
    ask_yes: float | None,
    ask_no: float | None,
    settle_time_utc: datetime,
    eval_time_utc: datetime,
    minute_df: pd.DataFrame,
    lookback_hours: float,
) -> dict | None:
    """Calibrate GBM + Student's t and return edge signals.

    Returns None when OHLCV data is insufficient (< 20 bars before eval_time).
    Edge signals (buy-only exchange):
      edge_yes_* = p_fair - ask_yes        → buy YES when > 0
      edge_no_*  = (1 - p_fair) - ask_no   → buy NO  when > 0
    """
    eval_ts   = pd.Timestamp(eval_time_utc).tz_convert(minute_df.index.tz)
    settle_ts = pd.Timestamp(settle_time_utc).tz_convert(minute_df.index.tz)

    horizon_hours = (settle_ts - eval_ts).total_seconds() / 3600.0
    if horizon_hours <= 0:
        return None

    spot = float(minute_df["close"].iloc[-1])

    min_lr = _minute_log_returns_before(minute_df, eval_ts, lookback_hours)
    if len(min_lr) < 20:
        return None

    # GBM
    gbm      = calibrate_gbm_from_log_returns(min_lr, dt_hours=1 / 60)
    p_gbm    = gbm_binary_prob(spot, strike, gbm, horizon_hours=horizon_hours, direction=direction)
    sigma_an = gbm.sigma_per_sqrt_hour * sqrt(8_760)

    # Student's t — calibrate on 1-min returns, scale to contract horizon
    tpar_1m = calibrate_student_t_from_log_returns(min_lr)
    n       = max(1, round(horizon_hours * 60))
    tpar    = StudentTParams(loc=tpar_1m.loc * n, scale=tpar_1m.scale * sqrt(n), nu=tpar_1m.nu)
    seed    = abs(hash(contract_id)) % (2 ** 31)
    p_stud  = student_t_binary_prob(spot, strike, tpar, direction=direction, n_sims=20_000, seed=seed)

    result: dict = {
        "spot":              spot,
        "horizon_hours":     round(horizon_hours, 4),
        # GBM params
        "mu_per_hour":       gbm.mu_per_hour,
        "sigma_annual_gbm":  round(sigma_an, 4),
        # Student's t params (1-minute calibration)
        "nu_stud":           round(tpar_1m.nu, 2),
        "sigma_annual_stud": round(tpar_1m.scale * sqrt(525_600), 4),  # annualised scale
        # Fair values
        "p_fair_gbm":        round(p_gbm,  4),
        "p_fair_stud":       round(p_stud, 4),
    }

    if ask_yes is not None:
        result["edge_yes_gbm"]  = round(p_gbm  - ask_yes, 4)
        result["edge_yes_stud"] = round(p_stud - ask_yes, 4)
    else:
        result["edge_yes_gbm"] = result["edge_yes_stud"] = None

    if ask_no is not None:
        result["edge_no_gbm"]  = round((1.0 - p_gbm)  - ask_no, 4)
        result["edge_no_stud"] = round((1.0 - p_stud) - ask_no, 4)
    else:
        result["edge_no_gbm"] = result["edge_no_stud"] = None

    return result


# ---------------------------------------------------------------------------
# Position lifecycle
# ---------------------------------------------------------------------------

def _create_position(
    row: dict,
    model: str,
    side: str,
    ask_price: float,
    p_fair: float,
    edge: float,
    market_mid: float | None,
    edges: dict,
) -> dict:
    return {
        "entry_time_utc":           row["timestamp_utc"],
        "contract_id":              row.get("contract_id", ""),
        "contract_label":           row.get("contract_label", ""),
        "event_title":              row.get("event_title", ""),
        "event_ticker":             row.get("event_ticker", ""),
        "asset":                    row.get("asset", ""),
        "strike":                   row.get("strike", ""),
        "direction":                row.get("direction", ""),
        "settle_time_utc":          row["settle_time_utc"],
        "hours_to_settle_at_entry": row.get("hours_to_settle", ""),
        "side":                     side,
        "model":                    model,
        "p_fair":                   round(p_fair, 4),
        "ask_price":                round(ask_price, 4),
        "edge":                     round(edge, 4),
        "market_mid":               round(market_mid, 4) if market_mid is not None else None,
        # Model parameters at entry (for display and audit)
        "mu_per_hour":              edges.get("mu_per_hour"),
        "sigma_annual_gbm":         edges.get("sigma_annual_gbm"),
        "nu_stud":                  edges.get("nu_stud"),
        "sigma_annual_stud":        edges.get("sigma_annual_stud"),
        "exit_time_utc":            None,
        "exit_reason":              None,
        "exit_bid":                 None,
        "spot_at_settle":           None,
        "outcome":                  None,
        "pnl":                      None,
        "status":                   "open",
    }


def _check_early_exit(
    pos: dict,
    latest_row: dict | None,
    minute_df: pd.DataFrame | None,
    lookback_hours: float,
    now_utc: datetime,
    profit_lock: float,
    stop_loss: float,
    p_drop: float,
) -> dict | None:
    """Check whether an open position should be closed before settlement.

    Checks four conditions in priority order:
      1. profit_lock  — bid has moved far enough in our favour → lock in gain.
      2. stop_loss    — bid has moved far enough against us → cap the loss.
      3. p_drop       — model probability for our side has fallen → information exit.
      4. edge_closed  — current edge has gone negative → market has corrected.

    Returns a dict with exit_reason/exit_bid/exit_pnl on trigger, else None.
    """
    if latest_row is None:
        return None

    side      = pos["side"]
    ask_entry = float(pos["ask_price"])

    # Current bid for our side — what the market pays us to sell now.
    bid_now = _safe_float(latest_row.get("bid_yes" if side == "YES" else "bid_no"))
    if bid_now is None:
        return None

    realised_pnl = round(bid_now - ask_entry, 4)

    # --- 1. Profit lock-in ---
    if realised_pnl >= profit_lock:
        return {"exit_reason": "profit_lock", "exit_bid": bid_now, "exit_pnl": realised_pnl}

    # --- 2. Stop loss ---
    if -realised_pnl >= stop_loss:
        return {"exit_reason": "stop_loss", "exit_bid": bid_now, "exit_pnl": realised_pnl}

    # --- 3 & 4. Model-based exits (require fresh calibration) ---
    if minute_df is not None:
        try:
            strike          = float(pos["strike"])
            direction       = pos["direction"]
            settle_time_utc = datetime.fromisoformat(pos["settle_time_utc"])
            ask_yes         = _safe_float(latest_row.get("ask_yes"))
            ask_no          = _safe_float(latest_row.get("ask_no"))

            edges = _compute_edges(
                contract_id     = pos["contract_id"],
                asset           = pos["asset"],
                strike          = strike,
                direction       = direction,
                ask_yes         = ask_yes,
                ask_no          = ask_no,
                settle_time_utc = settle_time_utc,
                eval_time_utc   = now_utc,
                minute_df       = minute_df,
                lookback_hours  = lookback_hours,
            )

            if edges is not None:
                p_key = "gbm" if pos["model"] == "gbm" else "stud"
                p_fair_now = edges[f"p_fair_{p_key}"]

                # Probability for the side we hold (YES → p_fair, NO → 1 - p_fair)
                p_now   = p_fair_now        if side == "YES" else (1.0 - p_fair_now)
                p_entry = float(pos["p_fair"])

                # 3. Probability has dropped significantly from entry.
                if p_entry - p_now >= p_drop:
                    return {
                        "exit_reason": f"p_drop({p_entry:.3f}→{p_now:.3f})",
                        "exit_bid":    bid_now,
                        "exit_pnl":    realised_pnl,
                    }

                # 4. Edge has completely closed (market repriced to model).
                edge_key = f"edge_yes_{p_key}" if side == "YES" else f"edge_no_{p_key}"
                current_edge = edges.get(edge_key)
                if current_edge is not None and current_edge < 0:
                    return {
                        "exit_reason": f"edge_closed({current_edge:.3f})",
                        "exit_bid":    bid_now,
                        "exit_pnl":    realised_pnl,
                    }
        except Exception:
            pass

    return None  # Hold


def _apply_early_exit(pos: dict, exit_info: dict, now_utc: datetime) -> None:
    """Mutate pos with early-exit closing fields."""
    pos.update({
        "exit_time_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "exit_reason":   exit_info["exit_reason"],
        "exit_bid":      exit_info["exit_bid"],
        "pnl":           exit_info["exit_pnl"],
        "status":        "exited_early",
    })


def _try_settle(pos: dict, minute_df: pd.DataFrame | None, now_utc: datetime) -> bool:
    """Settle via OHLCV spot lookup after contract expiry.

    Returns True only when OHLCV data covers the settle timestamp.
    """
    if minute_df is None:
        return False

    settle_utc = datetime.fromisoformat(pos["settle_time_utc"])
    settle_ts  = pd.Timestamp(settle_utc).tz_convert(minute_df.index.tz)

    if minute_df.index[-1] < settle_ts:
        return False  # OHLCV not yet updated past settlement

    bars = minute_df.loc[minute_df.index <= settle_ts]
    if bars.empty:
        return False

    spot_at_settle = float(bars["close"].iloc[-1])
    direction      = pos["direction"]
    strike         = float(pos["strike"])

    outcome = ("YES_WIN" if spot_at_settle > strike else "NO_WIN") if direction == "above" \
         else ("YES_WIN" if spot_at_settle <= strike else "NO_WIN")

    ask_price = float(pos["ask_price"])
    won       = (pos["side"] == "YES" and outcome == "YES_WIN") or \
                (pos["side"] == "NO"  and outcome == "NO_WIN")
    pnl       = round((1.0 - ask_price) if won else (-ask_price), 4)

    pos.update({
        "exit_time_utc":  now_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "exit_reason":    "settlement",
        "spot_at_settle": spot_at_settle,
        "outcome":        outcome,
        "pnl":            pnl,
        "status":         "settled",
    })
    return True


# ---------------------------------------------------------------------------
# Performance ledger
# ---------------------------------------------------------------------------

def _update_performance_ledger() -> None:
    """Recompute and overwrite performance_ledger.csv from ALL historical trade files.

    Reads every trades_*.csv in SIM_ROOT so the ledger accumulates across
    days and model versions.  Old trade rows are never deleted — this is
    purely a derived aggregate view of the raw per-day files.

    Ledger schema: see PERF_LEDGER_FIELDS.
    Profitable = pnl > 0  (covers both settlement wins and early-exit gains).
    """
    all_rows: list[dict] = []
    for fp in sorted(SIM_ROOT.glob("trades_*.csv")):
        try:
            with fp.open(newline="") as f:
                all_rows.extend(list(csv.DictReader(f)))
        except Exception:
            pass

    # Only closed positions contribute to the ledger.
    closed = [r for r in all_rows if r.get("status") in ("settled", "exited_early")]

    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")
    out_rows = []

    for model in ("gbm", "student_t"):
        trades = [r for r in closed if r.get("model") == model]

        if not trades:
            out_rows.append({k: ("—" if k not in ("model", "as_of") else
                                 (model if k == "model" else now_str))
                             for k in PERF_LEDGER_FIELDS})
            continue

        pnls: list[float] = []
        for r in trades:
            try:
                pnls.append(float(r["pnl"]))
            except (ValueError, KeyError):
                pnls.append(0.0)

        edges: list[float] = []
        for r in trades:
            try:
                edges.append(float(r["edge"]))
            except (ValueError, KeyError):
                pass

        n            = len(trades)
        n_profitable = sum(1 for p in pnls if p > 0)
        n_settled    = sum(1 for r in trades if r.get("status") == "settled")
        n_early      = sum(1 for r in trades if r.get("status") == "exited_early")

        out_rows.append({
            "model":             model,
            "n_trades":          n,
            "n_profitable":      n_profitable,
            "n_loss":            n - n_profitable,
            "n_settled":         n_settled,
            "n_early_exit":      n_early,
            "win_rate_pct":      round(100.0 * n_profitable / n, 1),
            "avg_pnl":           round(sum(pnls) / n, 4),
            "total_pnl":         round(sum(pnls), 4),
            "avg_edge_at_entry": round(sum(edges) / len(edges), 4) if edges else "—",
            "as_of":             now_str,
        })

    ledger_path = SIM_ROOT / "performance_ledger.csv"
    with ledger_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PERF_LEDGER_FIELDS)
        w.writeheader()
        w.writerows(out_rows)


# ---------------------------------------------------------------------------
# Console logging
# ---------------------------------------------------------------------------

def _fmt_label(pos: dict) -> str:
    """Short readable contract label, e.g. 'BTC > $85,000 ↑'."""
    label = pos.get("contract_label") or pos.get("contract_id", "?")
    arrow = "↑" if pos.get("direction") == "above" else "↓"
    return f"{label} {arrow}"


def _fmt_settle(settle_time_utc_str: str, ref_utc: datetime | None = None) -> str:
    """Return e.g. '@8PM EST (T-50min)' from a UTC ISO string."""
    try:
        settle_utc = datetime.fromisoformat(settle_time_utc_str)
        settle_est = settle_utc.astimezone(EST_TZ)
        hour_str   = settle_est.strftime("%I%p").lstrip("0")   # "8PM"
        if ref_utc is not None:
            t_min = round((settle_utc - ref_utc).total_seconds() / 60)
            return f"@{hour_str} EST (T-{t_min}min)"
        return f"@{hour_str} EST"
    except Exception:
        return ""


def _log_enter(pos: dict) -> None:
    label     = _fmt_label(pos)
    side      = pos["side"]
    model     = pos["model"]
    mkt       = pos["market_mid"]
    fair      = float(pos["p_fair"])
    ask       = float(pos["ask_price"])
    edge      = float(pos["edge"])
    t_str     = _fmt_settle(pos["settle_time_utc"])   # "@8PM EST"

    mkt_str  = f"mkt={mkt*100:.1f}%  " if mkt is not None else ""

    if model == "gbm":
        mu   = pos.get("mu_per_hour") or 0.0
        s_an = (pos.get("sigma_annual_gbm") or 0.0) * 100
        params = f"drift={mu*100:+.4f}%/h  σ={s_an:.1f}%"
    else:
        nu   = pos.get("nu_stud") or 0.0
        s_an = (pos.get("sigma_annual_stud") or 0.0) * 100
        params = f"ν={nu:.1f}  σ={s_an:.1f}%"

    print(
        f"  [ENTER]  {label:<22}  {side}  "
        f"{mkt_str}fair={fair*100:.1f}%  ask={ask*100:.1f}%  edge={edge*100:+.1f}%  "
        f"T={t_str}   {{{model}: {params}}}"
    )


def _log_exit(pos: dict, exit_info: dict) -> None:
    label     = _fmt_label(pos)
    side      = pos["side"]
    model     = pos["model"]
    entry_pct = float(pos["ask_price"]) * 100
    bid_pct   = exit_info["exit_bid"] * 100
    pnl       = exit_info["exit_pnl"]
    reason    = exit_info["exit_reason"]
    print(
        f"  [EXIT]   {label:<22}  {side}  {reason:<28}  "
        f"entry={entry_pct:.1f}%  bid={bid_pct:.1f}%  pnl={pnl:+.2f}   [{model}]"
    )


def _log_settle(pos: dict) -> None:
    label  = _fmt_label(pos)
    side   = pos["side"]
    model  = pos["model"]
    pnl    = float(pos["pnl"])
    spot   = pos.get("spot_at_settle")
    strike = float(pos["strike"])
    outcome = pos.get("outcome", "?")
    won    = (side == "YES" and outcome == "YES_WIN") or (side == "NO" and outcome == "NO_WIN")
    result = "WIN " if won else "LOSS"
    spot_str = f"${spot:>10,.2f}" if isinstance(spot, (int, float)) else str(spot)
    print(
        f"  [SETTLE] {label:<22}  {side}  {result}  "
        f"spot={spot_str}  strike=${strike:,.0f}  pnl={pnl:+.2f}   [{model}]"
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(now_utc: datetime, open_pos: list[dict], closed_pos: list[dict]) -> None:
    ts = now_utc.astimezone(EST_TZ).strftime("%H:%M:%S")

    # Build a live "T-Xmin" display per unique settle slot for open positions.
    if open_pos:
        seen_slots: dict[str, int] = {}   # "8PM" → remaining minutes
        for p in open_pos:
            try:
                settle_utc = datetime.fromisoformat(p["settle_time_utc"])
                settle_est = settle_utc.astimezone(EST_TZ)
                slot_key   = settle_est.strftime("%I%p").lstrip("0")
                t_min      = max(0, round((settle_utc - now_utc).total_seconds() / 60))
                seen_slots[slot_key] = t_min
            except Exception:
                pass
        slot_str = "  ".join(f"{k}→T-{v}min" for k, v in sorted(seen_slots.items()))
        open_str = f"open={len(open_pos)} [{slot_str}]" if slot_str else f"open={len(open_pos)}"
    else:
        open_str = "open=0"

    def fmt(model: str) -> str:
        trades = [p for p in closed_pos if p["model"] == model]
        if not trades:
            return "pnl=+0.00 (0/0)"
        pnl  = sum(float(p["pnl"]) for p in trades)
        wins = sum(1 for p in trades if float(p["pnl"]) > 0)
        early = sum(1 for p in trades if p["status"] == "exited_early")
        return f"pnl={pnl:+.2f} ({wins}/{len(trades)}, {early} early)"

    print(
        f"[{ts}]  {open_str}  closed={len(closed_pos)}  "
        f"GBM: {fmt('gbm')}  StudT: {fmt('student_t')}"
    )


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run(
    poll_sec:           int   = 60,
    lookback_hours:     float = 48.0,
    min_edge:           float = 0.03,
    max_hours_to_settle:float = 1.5,
    profit_lock:        float = 0.05,
    stop_loss:          float = 0.10,
    p_drop:             float = 0.05,
) -> None:
    """Poll contract data, calibrate models, simulate trades, manage exits."""
    print(
        f"Starting live trading simulation\n"
        f"  poll={poll_sec}s  lookback={lookback_hours}h  "
        f"min_edge={min_edge:.1%}  max_horizon={max_hours_to_settle}h\n"
        f"  exit: profit_lock={profit_lock:.2f}  stop_loss={stop_loss:.2f}  p_drop={p_drop:.2f}"
    )
    print(f"Output → {SIM_ROOT.resolve()}")

    trade_out = _trade_out_path()
    edge_out  = _edge_out_path()
    _ensure_header(trade_out, TRADE_FIELDS)
    _ensure_header(edge_out,  EDGE_FIELDS)

    seen_gbm:  set[str] = set()
    seen_stud: set[str] = set()
    open_positions:   list[dict] = []
    closed_positions: list[dict] = []

    while True:
        t0      = time.time()
        now_utc = datetime.now(tz=timezone.utc)
        today   = now_utc.strftime("%Y%m%d")

        minute_dfs: dict[str, pd.DataFrame | None] = {
            asset: _load_minute_df(asset) for asset in ASSET_SYMBOLS
        }

        rows    = _read_contract_csv(today)
        latest  = _latest_by_contract(rows)  # {contract_id: most-recent-row}

        # ── 1. Early-exit checks for all open positions ─────────────────────
        still_open: list[dict] = []
        for pos in open_positions:
            cid     = pos["contract_id"]
            mdf     = minute_dfs.get(pos["asset"])
            lat_row = latest.get(cid)

            exit_info = _check_early_exit(
                pos, lat_row, mdf, lookback_hours, now_utc,
                profit_lock, stop_loss, p_drop,
            )

            if exit_info:
                _apply_early_exit(pos, exit_info, now_utc)
                closed_positions.append(pos)
                _append_csv(trade_out, pos, TRADE_FIELDS)
                _log_exit(pos, exit_info)
                _update_performance_ledger()
            else:
                still_open.append(pos)
        open_positions = still_open

        # ── 2. Settlement checks for remaining open positions ────────────────
        still_open = []
        for pos in open_positions:
            mdf = minute_dfs.get(pos["asset"])
            if _try_settle(pos, mdf, now_utc):
                closed_positions.append(pos)
                _append_csv(trade_out, pos, TRADE_FIELDS)
                _log_settle(pos)
                _update_performance_ledger()
            else:
                still_open.append(pos)
        open_positions = still_open

        # ── 3. Entry decisions for new contracts ─────────────────────────────
        for row in rows:
            contract_id = row.get("contract_id", "")
            if not contract_id:
                continue

            asset     = row.get("asset", "")
            minute_df = minute_dfs.get(asset)
            if minute_df is None:
                continue

            ask_yes = _safe_float(row.get("ask_yes"))
            ask_no  = _safe_float(row.get("ask_no"))
            try:
                strike          = float(row["strike"])
                direction       = row["direction"]
                settle_time_utc = datetime.fromisoformat(row["settle_time_utc"])
                eval_time_utc   = datetime.fromisoformat(row["timestamp_utc"])
                hours_to_settle = float(row["hours_to_settle"])
            except (KeyError, ValueError):
                continue

            # Skip contracts that have already settled or are too far out.
            current_hrs = (settle_time_utc - now_utc).total_seconds() / 3600.0
            if current_hrs < 0 or current_hrs > max_hours_to_settle:
                continue

            need_gbm  = contract_id not in seen_gbm
            need_stud = contract_id not in seen_stud
            if not (need_gbm or need_stud):
                continue

            edges = _compute_edges(
                contract_id     = contract_id,
                asset           = asset,
                strike          = strike,
                direction       = direction,
                ask_yes         = ask_yes,
                ask_no          = ask_no,
                settle_time_utc = settle_time_utc,
                eval_time_utc   = eval_time_utc,
                minute_df       = minute_df,
                lookback_hours  = lookback_hours,
            )

            if need_gbm:  # log edge signals once per contract
                arb_long = round(ask_yes + ask_no - 1.0, 4) if (ask_yes and ask_no) else None
                _append_csv(edge_out, {
                    "timestamp_utc":    row["timestamp_utc"],
                    "contract_id":      contract_id,
                    "asset":            asset,
                    "strike":           strike,
                    "direction":        direction,
                    "hours_to_settle":  hours_to_settle,
                    "spot":             edges["spot"]             if edges else None,
                    "ask_yes":          ask_yes,
                    "ask_no":           ask_no,
                    "arb_long":         arb_long,
                    "sigma_annual_gbm": edges["sigma_annual_gbm"] if edges else None,
                    "p_fair_gbm":       edges["p_fair_gbm"]       if edges else None,
                    "edge_yes_gbm":     edges.get("edge_yes_gbm") if edges else None,
                    "edge_no_gbm":      edges.get("edge_no_gbm")  if edges else None,
                    "p_fair_stud":      edges["p_fair_stud"]      if edges else None,
                    "edge_yes_stud":    edges.get("edge_yes_stud") if edges else None,
                    "edge_no_stud":     edges.get("edge_no_stud")  if edges else None,
                }, EDGE_FIELDS)

            for model_name, p_key, seen_set, needed in [
                ("gbm",       "gbm",  seen_gbm,  need_gbm),
                ("student_t", "stud", seen_stud, need_stud),
            ]:
                if not needed or edges is None:
                    seen_set.add(contract_id)
                    continue

                seen_set.add(contract_id)

                p_fair   = edges[f"p_fair_{p_key}"]
                edge_yes = edges.get(f"edge_yes_{p_key}")
                edge_no  = edges.get(f"edge_no_{p_key}")

                side = edge_val = ask_price = p_side = None

                if (edge_yes is not None and edge_yes > min_edge and ask_yes is not None and
                        (edge_no is None or edge_yes >= edge_no)):
                    side, edge_val, ask_price, p_side = "YES", edge_yes, ask_yes, p_fair
                elif edge_no is not None and edge_no > min_edge and ask_no is not None:
                    side, edge_val, ask_price, p_side = "NO", edge_no, ask_no, 1.0 - p_fair

                if side is None:
                    continue

                # Market mid for the side we're trading
                mid_yes = _safe_float(row.get("mid_yes"))
                market_mid = mid_yes if side == "YES" else \
                             (round(1.0 - mid_yes, 4) if mid_yes is not None else None)

                pos = _create_position(row, model_name, side, ask_price, p_side,
                                       edge_val, market_mid, edges)
                open_positions.append(pos)
                _log_enter(pos)

        _print_summary(now_utc, open_positions, closed_positions)

        elapsed = time.time() - t0
        time.sleep(max(0.0, poll_sec - elapsed))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _s  = cfg.section("simulation")
    _ex = cfg.section("simulation.exit")

    parser = argparse.ArgumentParser(
        description="Live prediction market trading simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--poll-sec",       type=int,   default=_s.get("poll_sec",       60),
                        help="Seconds between polls")
    parser.add_argument("--lookback-hours", type=float, default=_s.get("lookback_hours", 48.0),
                        help="Vol calibration lookback (hours)")
    parser.add_argument("--min-edge",           type=float, default=_s.get("min_edge",            0.03),
                        help="Min model edge required to enter a trade")
    parser.add_argument("--max-hours-to-settle",type=float, default=_s.get("max_hours_to_settle", 1.5),
                        help="Skip contracts settling more than this many hours away")
    parser.add_argument("--profit-lock",    type=float, default=_ex.get("profit_lock",   0.05),
                        help="Early exit when bid_now - ask_entry >= this")
    parser.add_argument("--stop-loss",      type=float, default=_ex.get("stop_loss",     0.10),
                        help="Early exit when ask_entry - bid_now >= this")
    parser.add_argument("--p-drop",         type=float, default=_ex.get("p_drop",        0.05),
                        help="Early exit when model p(side) drops by this from entry")
    args = parser.parse_args()
    run(
        poll_sec            = args.poll_sec,
        lookback_hours      = args.lookback_hours,
        min_edge            = args.min_edge,
        max_hours_to_settle = args.max_hours_to_settle,
        profit_lock         = args.profit_lock,
        stop_loss           = args.stop_loss,
        p_drop              = args.p_drop,
    )
