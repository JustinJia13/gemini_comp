"""
Live prediction market trading simulation.

Reads:
  .data/gemini/ohlcv_1m_7d/{sym}_est.data        — live underlying OHLCV
  .data/gemini/prediction_data/{YYYYMMDD}.csv     — live contract prices

Data collectors are started automatically as subprocesses on launch:
  getdata_underlying.py          — fetches 1-min OHLCV from Gemini public API
  getdata_prediction_contract.py — fetches Kalshi contract prices

Pass --no-collectors to suppress auto-start (e.g. if you are running those
scripts separately in other terminals).

For each contract seen for the first time:
  - Calibrates 6 volatility models on live minute returns (no look-ahead):
      GBM (24 h)  |  EWMA (48 h)  |  GARCH (72 h)
      Student-t (48 h)  |  Skewed-t (48 h)  |  Heston (96 h)
    Lookback windows are per-model hyperparameters (see config.toml
    [simulation.lookback] or --lb-* CLI flags).
  - Simulates a buy trade when edge exceeds --min-edge threshold.

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

Run (most common):
    python live_trading_sim.py

Tune exit thresholds:
    python live_trading_sim.py --min-edge 0.05 --profit-lock 0.04 \\
                                --stop-loss 0.08 --p-drop 0.04

Override per-model lookback windows:
    python live_trading_sim.py --lb-gbm 12 --lb-heston 120

Run without auto-starting the data collectors (manage them separately):
    python live_trading_sim.py --no-collectors

All defaults live in config.toml under [simulation], [simulation.exit],
and [simulation.lookback].
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
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
    calibrate_ewma_from_log_returns,
    calibrate_garch_from_log_returns,
    calibrate_gbm_from_log_returns,
    calibrate_heston_from_log_returns,
    calibrate_skewed_t_from_log_returns,
    calibrate_student_t_from_log_returns,
    estimate_price_vol_rho,
    ewma_binary_prob,
    garch_binary_prob,
    gbm_binary_prob,
    heston_binary_prob,
    load_gemini_ohlcv,
    skewed_t_binary_prob,
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
    "model",          # gbm | ewma | garch | student_t | skewed_t | heston
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
    # GBM (rolling-window sigma)
    "sigma_annual_gbm",
    "p_fair_gbm",   "edge_yes_gbm",   "edge_no_gbm",
    # EWMA vol + GBM closed form
    "sigma_annual_ewma",
    "p_fair_ewma",  "edge_yes_ewma",  "edge_no_ewma",
    # GARCH(1,1) vol + GBM closed form
    "sigma_annual_garch", "alpha_garch", "beta_garch",
    "p_fair_garch", "edge_yes_garch", "edge_no_garch",
    # Symmetric Student's t (Monte Carlo)
    "nu_stud", "sigma_annual_stud",
    "p_fair_stud",  "edge_yes_stud",  "edge_no_stud",
    # Skewed Student's t (Monte Carlo, Fernández-Steel)
    "nu_skt", "gamma_skt", "rho_est",
    "p_fair_skt",   "edge_yes_skt",   "edge_no_skt",
    # Heston stochastic vol (Monte Carlo, full-truncation Euler)
    "v0_heston", "kappa_heston", "theta_heston", "xi_heston", "rho_heston",
    "p_fair_heston", "edge_yes_heston", "edge_no_heston",
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
    """Write header if missing; rewrite in-place if the header is stale."""
    expected = ",".join(fields)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
        return
    with path.open("r", newline="") as f:
        first = f.readline().rstrip("\r\n")
    if first == expected:
        return  # header is current — nothing to do
    # Header is stale: rewrite with correct header, keep all data rows.
    with path.open("r", newline="") as f:
        lines = f.readlines()
    lines[0] = expected + "\n"
    with path.open("w", newline="") as f:
        f.writelines(lines)
    print(f"  [header] Updated stale header in {path.name}")


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
    lookbacks: dict[str, float],
    ewma_lambda: float = 0.94,
    rho: float = -0.5,
) -> dict | None:
    """Calibrate all 6 models and return fair values + edge signals.

    Models: gbm | ewma | garch | student_t | skewed_t | heston

    Each model uses its own calibration window (lookbacks dict) so shorter-
    window models stay reactive while longer-window models gain MLE stability.
    Arrays for identical window sizes are computed once and reused.

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

    # Per-model log-return arrays — cached by window size to avoid re-reading.
    _lr_cache: dict[float, object] = {}
    def _get_lr(lb: float):
        if lb not in _lr_cache:
            _lr_cache[lb] = _minute_log_returns_before(minute_df, eval_ts, lb)
        return _lr_cache[lb]

    lb_gbm    = lookbacks.get("gbm",    24.0)
    lb_ewma   = lookbacks.get("ewma",   48.0)
    lb_garch  = lookbacks.get("garch",  72.0)
    lb_stud   = lookbacks.get("stud",   48.0)
    lb_skt    = lookbacks.get("skt",    48.0)
    lb_heston = lookbacks.get("heston", 96.0)

    min_lr_gbm = _get_lr(lb_gbm)
    if len(min_lr_gbm) < 20:
        return None

    seed = abs(hash(contract_id)) % (2 ** 31)
    n    = max(1, round(horizon_hours * 60))   # 1-min periods in horizon

    # ── Model 1: GBM (rolling-window sigma) ───────────────────────────────
    gbm     = calibrate_gbm_from_log_returns(min_lr_gbm, dt_hours=1 / 60)
    p_gbm   = gbm_binary_prob(spot, strike, gbm, horizon_hours=horizon_hours, direction=direction)

    # ── Model 2: EWMA vol + GBM closed form ───────────────────────────────
    try:
        min_lr_ewma = _get_lr(lb_ewma)
        ewma_par  = calibrate_ewma_from_log_returns(min_lr_ewma, lambda_=ewma_lambda)
        p_ewma    = ewma_binary_prob(spot, strike, ewma_par, horizon_hours=horizon_hours, direction=direction)
        sigma_ewma = ewma_par.sigma_per_sqrt_hour * sqrt(8_760)
    except Exception:
        p_ewma = p_gbm;  sigma_ewma = gbm.sigma_per_sqrt_hour * sqrt(8_760)
        ewma_par = None

    # ── Model 3: GARCH(1,1) vol + GBM closed form ─────────────────────────
    try:
        min_lr_garch = _get_lr(lb_garch)
        garch_par  = calibrate_garch_from_log_returns(min_lr_garch, dt_hours=1 / 60)
        p_garch    = garch_binary_prob(spot, strike, garch_par, horizon_hours=horizon_hours, direction=direction)
        sigma_garch = garch_par.sigma_per_sqrt_hour * sqrt(8_760)
    except Exception:
        p_garch = p_gbm;  sigma_garch = gbm.sigma_per_sqrt_hour * sqrt(8_760)
        garch_par = None

    # ── Model 4: Symmetric Student's t (Monte Carlo) ───────────────────────
    min_lr_stud = _get_lr(lb_stud)
    tpar_1m = calibrate_student_t_from_log_returns(min_lr_stud)
    tpar    = StudentTParams(loc=tpar_1m.loc * n, scale=tpar_1m.scale * sqrt(n), nu=tpar_1m.nu)
    p_stud  = student_t_binary_prob(spot, strike, tpar, direction=direction, n_sims=20_000, seed=seed)

    # ── Model 5: Skewed Student's t (Monte Carlo, Fernández-Steel) ─────────
    # Estimate rho from data; blend with the config prior (50/50)
    min_lr_skt = _get_lr(lb_skt)
    try:
        rho_est = estimate_price_vol_rho(min_lr_skt)
        rho_use = 0.5 * rho + 0.5 * rho_est      # blend prior and data
    except Exception:
        rho_est = rho;  rho_use = rho
    try:
        skt_par = calibrate_skewed_t_from_log_returns(min_lr_skt, rho=rho_use)
        skt_h   = type(skt_par)(
            loc   = skt_par.loc   * n,
            scale = skt_par.scale * sqrt(n),
            nu    = skt_par.nu,
            gamma = skt_par.gamma,
        )
        p_skt = skewed_t_binary_prob(spot, strike, skt_h, direction=direction,
                                     n_sims=20_000, seed=seed + 1)
    except Exception:
        p_skt   = p_stud
        skt_par = None
        rho_est = rho

    # ── Model 6: Heston stochastic vol (Monte Carlo) ────────────────────────
    try:
        min_lr_heston = _get_lr(lb_heston)
        heston_par = calibrate_heston_from_log_returns(min_lr_heston, rho=rho_use, dt_hours=1 / 60)
        p_heston   = heston_binary_prob(spot, strike, heston_par,
                                        horizon_hours=horizon_hours,
                                        direction=direction,
                                        n_sims=20_000, seed=seed + 2)
    except Exception:
        p_heston   = p_gbm
        heston_par = None

    # ── Pack results ──────────────────────────────────────────────────────
    result: dict = {
        "spot":              spot,
        "horizon_hours":     round(horizon_hours, 4),
        # GBM
        "mu_per_hour":       gbm.mu_per_hour,
        "sigma_annual_gbm":  round(gbm.sigma_per_sqrt_hour * sqrt(8_760), 4),
        "p_fair_gbm":        round(p_gbm,   4),
        # EWMA
        "sigma_annual_ewma": round(sigma_ewma, 4),
        "p_fair_ewma":       round(p_ewma,  4),
        # GARCH
        "sigma_annual_garch": round(sigma_garch, 4),
        "alpha_garch":        round(garch_par.alpha, 4) if garch_par else None,
        "beta_garch":         round(garch_par.beta,  4) if garch_par else None,
        "p_fair_garch":       round(p_garch,  4),
        # Student's t
        "nu_stud":           round(tpar_1m.nu,  2),
        "sigma_annual_stud": round(tpar_1m.scale * sqrt(525_600), 4),
        "p_fair_stud":       round(p_stud,  4),
        # Skewed t
        "nu_skt":    round(skt_par.nu,    2) if skt_par else None,
        "gamma_skt": round(skt_par.gamma, 4) if skt_par else None,
        "rho_est":   round(rho_est,       4),
        "p_fair_skt":        round(p_skt,   4),
        # Heston
        "v0_heston":    round(heston_par.v0,    6) if heston_par else None,
        "kappa_heston": round(heston_par.kappa, 4) if heston_par else None,
        "theta_heston": round(heston_par.theta, 6) if heston_par else None,
        "xi_heston":    round(heston_par.xi,    4) if heston_par else None,
        "rho_heston":   round(heston_par.rho,   4) if heston_par else None,
        "p_fair_heston":     round(p_heston, 4),
    }

    # ── Edge signals for each model ────────────────────────────────────────
    for key, p in [("gbm", p_gbm), ("ewma", p_ewma), ("garch", p_garch),
                   ("stud", p_stud), ("skt", p_skt), ("heston", p_heston)]:
        result[f"edge_yes_{key}"] = round(p - ask_yes, 4)        if ask_yes is not None else None
        result[f"edge_no_{key}"]  = round((1.0 - p) - ask_no, 4) if ask_no  is not None else None

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
        "sigma_annual_ewma":        edges.get("sigma_annual_ewma"),
        "sigma_annual_garch":       edges.get("sigma_annual_garch"),
        "alpha_garch":              edges.get("alpha_garch"),
        "beta_garch":               edges.get("beta_garch"),
        "nu_stud":                  edges.get("nu_stud"),
        "sigma_annual_stud":        edges.get("sigma_annual_stud"),
        "nu_skt":                   edges.get("nu_skt"),
        "gamma_skt":                edges.get("gamma_skt"),
        "v0_heston":                edges.get("v0_heston"),
        "kappa_heston":             edges.get("kappa_heston"),
        "theta_heston":             edges.get("theta_heston"),
        "xi_heston":                edges.get("xi_heston"),
        "rho_heston":               edges.get("rho_heston"),
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
    lookbacks: dict[str, float],
    now_utc: datetime,
    profit_lock: float,
    stop_loss: float,
    p_drop: float,
    ewma_lambda: float = 0.94,
    rho: float = -0.5,
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
                lookbacks       = lookbacks,
                ewma_lambda     = ewma_lambda,
                rho             = rho,
            )

            if edges is not None:
                _p_key_map = {
                    "gbm":       "gbm",
                    "ewma":      "ewma",
                    "garch":     "garch",
                    "student_t": "stud",
                    "skewed_t":  "skt",
                    "heston":    "heston",
                }
                p_key = _p_key_map.get(pos["model"], "gbm")
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

def _update_performance_ledger() -> dict[str, dict]:
    """Recompute and overwrite performance_ledger.csv from ALL historical trade files.

    Reads every trades_*.csv in SIM_ROOT so the ledger accumulates across
    days and model versions.  Old trade rows are never deleted — this is
    purely a derived aggregate view of the raw per-day files.

    Ledger schema: see PERF_LEDGER_FIELDS.
    Profitable = pnl > 0  (covers both settlement wins and early-exit gains).

    Returns a dict keyed by model name for in-memory use by _print_summary.
    """
    SIM_ROOT.mkdir(parents=True, exist_ok=True)
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

    for model in ("gbm", "ewma", "garch", "student_t", "skewed_t", "heston"):
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

    return {r["model"]: r for r in out_rows}


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
    elif model == "ewma":
        s_an = (pos.get("sigma_annual_ewma") or 0.0) * 100
        params = f"σ_ewma={s_an:.1f}%"
    elif model == "garch":
        s_an = (pos.get("sigma_annual_garch") or 0.0) * 100
        a    = pos.get("alpha_garch") or 0.0
        b    = pos.get("beta_garch")  or 0.0
        params = f"σ_garch={s_an:.1f}%  α={a:.3f}  β={b:.3f}"
    elif model == "skewed_t":
        nu    = pos.get("nu_skt")    or 0.0
        gamma = pos.get("gamma_skt") or 1.0
        s_an  = (pos.get("sigma_annual_stud") or 0.0) * 100
        params = f"ν={nu:.1f}  γ={gamma:.3f}  σ={s_an:.1f}%"
    elif model == "heston":
        kappa = pos.get("kappa_heston") or 0.0
        xi    = pos.get("xi_heston")    or 0.0
        rho_h = pos.get("rho_heston")   or 0.0
        v0    = pos.get("v0_heston")    or 0.0
        s_an  = sqrt(v0 * 8_760) * 100 if v0 > 0 else 0.0
        params = f"κ={kappa:.1f}  ξ={xi:.3f}  ρ={rho_h:.3f}  σ₀={s_an:.1f}%"
    else:  # student_t
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

def _print_summary(
    now_utc: datetime,
    open_pos: list[dict],
    session_closed: int,
    ledger: dict[str, dict],
) -> None:
    """Print one-line status: open positions, session closed count, cumulative model P&L.

    Model stats come from the ledger (cumulative across all sessions) so scores
    are visible from the first poll even when no trades have closed this session.
    Format per model: TAG: total_pnl (wins/total, early_exits, win_rate%)
    """
    ts = now_utc.astimezone(EST_TZ).strftime("%H:%M:%S")

    # Build a live "T-Xmin" display per unique settle slot for open positions.
    if open_pos:
        seen_slots: dict[str, int] = {}
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

    def fmt(model: str, tag: str) -> str:
        row = ledger.get(model, {})
        n = row.get("n_trades", "—")
        if n == "—" or n == 0:
            return f"{tag}: –"
        n     = int(n)
        wins  = int(row.get("n_profitable", 0))
        early = int(row.get("n_early_exit", 0))
        pnl   = float(row.get("total_pnl", 0))
        wr    = float(row.get("win_rate_pct", 0))
        return f"{tag}: {pnl:+.2f} ({wins}/{n}, {early}e, {wr:.0f}%)"

    parts = "  ".join([
        fmt("gbm",       "GBM"),
        fmt("ewma",      "EWMA"),
        fmt("garch",     "GARCH"),
        fmt("student_t", "StudT"),
        fmt("skewed_t",  "SktT"),
        fmt("heston",    "Heston"),
    ])
    print(f"[{ts}]  {open_str}  session_closed={session_closed}  {parts}")


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run(
    poll_sec:           int   = 60,
    lookbacks:          dict[str, float] | None = None,
    min_edge:           float = 0.03,
    max_hours_to_settle:float = 1.5,
    profit_lock:        float = 0.05,
    stop_loss:          float = 0.10,
    p_drop:             float = 0.05,
    ewma_lambda:        float = 0.94,
    rho:                float = -0.5,
    no_collectors:      bool  = False,
) -> None:
    """Poll contract data, calibrate all 6 models, simulate trades, manage exits."""
    _lb_defaults = {"gbm": 24.0, "ewma": 48.0, "garch": 72.0,
                    "stud": 48.0, "skt": 48.0, "heston": 96.0}
    if lookbacks is None:
        lookbacks = _lb_defaults
    else:
        # fill in any missing keys with defaults
        lookbacks = {**_lb_defaults, **lookbacks}

    lb_str = "  ".join(f"{k}={v:.0f}h" for k, v in lookbacks.items())
    print(
        f"Starting live trading simulation\n"
        f"  poll={poll_sec}s  min_edge={min_edge:.1%}  max_horizon={max_hours_to_settle}h\n"
        f"  lookbacks: {lb_str}\n"
        f"  exit: profit_lock={profit_lock:.2f}  stop_loss={stop_loss:.2f}  p_drop={p_drop:.2f}\n"
        f"  vol models: ewma_lambda={ewma_lambda}  rho_prior={rho}"
    )
    print(f"Output → {SIM_ROOT.resolve()}")

    trade_out = _trade_out_path()
    edge_out  = _edge_out_path()
    _ensure_header(trade_out, TRADE_FIELDS)
    _ensure_header(edge_out,  EDGE_FIELDS)

    # Load cumulative ledger at startup so scores show historical data immediately.
    ledger_stats: dict[str, dict] = _update_performance_ledger()

    # ── Data-collector subprocesses ────────────────────────────────────────
    _HERE = Path(__file__).parent
    _COLLECTOR_SCRIPTS = [
        _HERE / "getdata_underlying.py",
        _HERE / "getdata_prediction_contract.py",
    ]
    collector_procs: list[subprocess.Popen | None] = [None, None]

    def _start_collector(script: Path) -> subprocess.Popen:
        p = subprocess.Popen([sys.executable, str(script)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  [collector] Started {script.name}  pid={p.pid}")
        return p

    if not no_collectors:
        for i, script in enumerate(_COLLECTOR_SCRIPTS):
            if script.exists():
                collector_procs[i] = _start_collector(script)
            else:
                print(f"  [collector] WARNING: {script.name} not found — skipping")

    seen_gbm:    set[str] = set()
    seen_ewma:   set[str] = set()
    seen_garch:  set[str] = set()
    seen_stud:   set[str] = set()
    seen_skt:    set[str] = set()
    seen_heston: set[str] = set()
    open_positions:   list[dict] = []
    closed_positions: list[dict] = []

    # Terminate collectors when the sim exits for any reason.
    import atexit
    def _cleanup_collectors():
        for proc in collector_procs:
            if proc and proc.poll() is None:
                proc.terminate()
    atexit.register(_cleanup_collectors)

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
                pos, lat_row, mdf, lookbacks, now_utc,
                profit_lock, stop_loss, p_drop,
                ewma_lambda=ewma_lambda, rho=rho,
            )

            if exit_info:
                _apply_early_exit(pos, exit_info, now_utc)
                closed_positions.append(pos)
                _append_csv(trade_out, pos, TRADE_FIELDS)
                _log_exit(pos, exit_info)
                ledger_stats = _update_performance_ledger()
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
                ledger_stats = _update_performance_ledger()
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

            need_gbm    = contract_id not in seen_gbm
            need_ewma   = contract_id not in seen_ewma
            need_garch  = contract_id not in seen_garch
            need_stud   = contract_id not in seen_stud
            need_skt    = contract_id not in seen_skt
            need_heston = contract_id not in seen_heston
            if not (need_gbm or need_ewma or need_garch or need_stud or need_skt or need_heston):
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
                lookbacks       = lookbacks,
                ewma_lambda     = ewma_lambda,
                rho             = rho,
            )

            if need_gbm:  # log edge signals once per contract (all models in one row)
                arb_long = round(ask_yes + ask_no - 1.0, 4) if (ask_yes and ask_no) else None
                _append_csv(edge_out, {
                    "timestamp_utc":      row["timestamp_utc"],
                    "contract_id":        contract_id,
                    "asset":              asset,
                    "strike":             strike,
                    "direction":          direction,
                    "hours_to_settle":    hours_to_settle,
                    "spot":               edges["spot"]               if edges else None,
                    "ask_yes":            ask_yes,
                    "ask_no":             ask_no,
                    "arb_long":           arb_long,
                    # GBM
                    "sigma_annual_gbm":   edges["sigma_annual_gbm"]   if edges else None,
                    "p_fair_gbm":         edges["p_fair_gbm"]         if edges else None,
                    "edge_yes_gbm":       edges.get("edge_yes_gbm")   if edges else None,
                    "edge_no_gbm":        edges.get("edge_no_gbm")    if edges else None,
                    # EWMA
                    "sigma_annual_ewma":  edges["sigma_annual_ewma"]  if edges else None,
                    "p_fair_ewma":        edges["p_fair_ewma"]        if edges else None,
                    "edge_yes_ewma":      edges.get("edge_yes_ewma")  if edges else None,
                    "edge_no_ewma":       edges.get("edge_no_ewma")   if edges else None,
                    # GARCH
                    "sigma_annual_garch": edges["sigma_annual_garch"] if edges else None,
                    "alpha_garch":        edges.get("alpha_garch")    if edges else None,
                    "beta_garch":         edges.get("beta_garch")     if edges else None,
                    "p_fair_garch":       edges["p_fair_garch"]       if edges else None,
                    "edge_yes_garch":     edges.get("edge_yes_garch") if edges else None,
                    "edge_no_garch":      edges.get("edge_no_garch")  if edges else None,
                    # Student's t
                    "nu_stud":            edges.get("nu_stud")        if edges else None,
                    "sigma_annual_stud":  edges.get("sigma_annual_stud") if edges else None,
                    "p_fair_stud":        edges["p_fair_stud"]        if edges else None,
                    "edge_yes_stud":      edges.get("edge_yes_stud")  if edges else None,
                    "edge_no_stud":       edges.get("edge_no_stud")   if edges else None,
                    # Skewed t
                    "nu_skt":             edges.get("nu_skt")         if edges else None,
                    "gamma_skt":          edges.get("gamma_skt")      if edges else None,
                    "rho_est":            edges.get("rho_est")        if edges else None,
                    "p_fair_skt":         edges["p_fair_skt"]         if edges else None,
                    "edge_yes_skt":       edges.get("edge_yes_skt")   if edges else None,
                    "edge_no_skt":        edges.get("edge_no_skt")    if edges else None,
                    # Heston
                    "v0_heston":          edges.get("v0_heston")      if edges else None,
                    "kappa_heston":       edges.get("kappa_heston")   if edges else None,
                    "theta_heston":       edges.get("theta_heston")   if edges else None,
                    "xi_heston":          edges.get("xi_heston")      if edges else None,
                    "rho_heston":         edges.get("rho_heston")     if edges else None,
                    "p_fair_heston":      edges["p_fair_heston"]      if edges else None,
                    "edge_yes_heston":    edges.get("edge_yes_heston") if edges else None,
                    "edge_no_heston":     edges.get("edge_no_heston")  if edges else None,
                }, EDGE_FIELDS)

            for model_name, p_key, seen_set, needed in [
                ("gbm",       "gbm",    seen_gbm,    need_gbm),
                ("ewma",      "ewma",   seen_ewma,   need_ewma),
                ("garch",     "garch",  seen_garch,  need_garch),
                ("student_t", "stud",   seen_stud,   need_stud),
                ("skewed_t",  "skt",    seen_skt,    need_skt),
                ("heston",    "heston", seen_heston, need_heston),
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

        _print_summary(now_utc, open_positions, len(closed_positions), ledger_stats)

        # ── Collector health-check: restart any dead subprocesses ───────────
        if not no_collectors:
            for i, (proc, script) in enumerate(zip(collector_procs, _COLLECTOR_SCRIPTS)):
                if proc is not None and proc.poll() is not None:
                    print(f"  [collector] {script.name} died (rc={proc.returncode}), restarting...")
                    collector_procs[i] = _start_collector(script)

        elapsed = time.time() - t0
        time.sleep(max(0.0, poll_sec - elapsed))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _s  = cfg.section("simulation")
    _ex = cfg.section("simulation.exit")
    _lb = cfg.section("simulation.lookback")

    # Per-model lookback defaults from config
    _lb_defaults = {
        "gbm":    _lb.get("gbm",    24.0),
        "ewma":   _lb.get("ewma",   48.0),
        "garch":  _lb.get("garch",  72.0),
        "stud":   _lb.get("stud",   48.0),
        "skt":    _lb.get("skt",    48.0),
        "heston": _lb.get("heston", 96.0),
    }

    parser = argparse.ArgumentParser(
        description="Live prediction market trading simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--poll-sec",            type=int,   default=_s.get("poll_sec",            60),
                        help="Seconds between polls")
    parser.add_argument("--min-edge",            type=float, default=_s.get("min_edge",            0.03),
                        help="Min model edge required to enter a trade")
    parser.add_argument("--max-hours-to-settle", type=float, default=_s.get("max_hours_to_settle", 1.5),
                        help="Skip contracts settling more than this many hours away")
    parser.add_argument("--profit-lock",         type=float, default=_ex.get("profit_lock",        0.05),
                        help="Early exit when bid_now - ask_entry >= this")
    parser.add_argument("--stop-loss",           type=float, default=_ex.get("stop_loss",          0.10),
                        help="Early exit when ask_entry - bid_now >= this")
    parser.add_argument("--p-drop",              type=float, default=_ex.get("p_drop",             0.05),
                        help="Early exit when model p(side) drops by this from entry")
    parser.add_argument("--ewma-lambda",         type=float, default=_s.get("ewma_lambda",         0.94),
                        help="EWMA decay factor λ (RiskMetrics default=0.94)")
    parser.add_argument("--rho",                 type=float, default=_s.get("rho",                 -0.5),
                        help="Prior price-vol correlation for skewed-t γ initialisation")
    # Per-model lookback windows
    parser.add_argument("--lb-gbm",    type=float, default=_lb_defaults["gbm"],
                        help="GBM calibration lookback (hours)")
    parser.add_argument("--lb-ewma",   type=float, default=_lb_defaults["ewma"],
                        help="EWMA calibration lookback (hours)")
    parser.add_argument("--lb-garch",  type=float, default=_lb_defaults["garch"],
                        help="GARCH calibration lookback (hours)")
    parser.add_argument("--lb-stud",   type=float, default=_lb_defaults["stud"],
                        help="Student-t calibration lookback (hours)")
    parser.add_argument("--lb-skt",    type=float, default=_lb_defaults["skt"],
                        help="Skewed-t calibration lookback (hours)")
    parser.add_argument("--lb-heston", type=float, default=_lb_defaults["heston"],
                        help="Heston calibration lookback (hours)")
    parser.add_argument("--no-collectors", action="store_true",
                        help="Do not auto-start getdata_underlying / getdata_prediction_contract")
    args = parser.parse_args()

    run(
        poll_sec            = args.poll_sec,
        lookbacks           = {
            "gbm":    args.lb_gbm,
            "ewma":   args.lb_ewma,
            "garch":  args.lb_garch,
            "stud":   args.lb_stud,
            "skt":    args.lb_skt,
            "heston": args.lb_heston,
        },
        min_edge            = args.min_edge,
        max_hours_to_settle = args.max_hours_to_settle,
        profit_lock         = args.profit_lock,
        stop_loss           = args.stop_loss,
        p_drop              = args.p_drop,
        ewma_lambda         = args.ewma_lambda,
        rho                 = args.rho,
        no_collectors       = args.no_collectors,
    )
