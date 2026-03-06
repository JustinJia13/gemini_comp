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
                    Skipped when < 30 min to settlement — let winner run to expiry.

  2. stop_loss    — (ask_entry - bid_now) / ask_entry >= --stop-loss  (relative, e.g.
                    0.50 = exit when bid has dropped 50% from entry price).
                    A 19¢ entry stops at 9.5¢; a 64¢ entry stops at 32¢.

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
import numpy as np
import pandas as pd
import requests

import config_loader as cfg

try:
    from gemini_trader import GeminiTrader
except ImportError:
    GeminiTrader = None  # type: ignore[assignment,misc]

from btc_hourly_model import (
    StudentTParams,
    _minute_closes_before,
    _minute_log_returns_before,
    calibrate_ewma_from_log_returns,
    calibrate_garch_from_log_returns,
    calibrate_gbm_from_log_returns,
    calibrate_heston_ewma_from_log_returns,
    calibrate_heston_from_log_returns,
    calibrate_hybrid_t_from_log_returns,
    calibrate_ou_from_closes,
    calibrate_skewed_t_from_log_returns,
    calibrate_student_t_from_log_returns,
    estimate_price_vol_rho,
    ewma_binary_prob,
    garch_binary_prob,
    gbm_binary_prob,
    heston_binary_prob,
    hybrid_t_binary_prob,
    load_gemini_ohlcv,
    ou_binary_prob,
    skewed_t_binary_prob,
    student_t_binary_prob,
    calibrate_jumps_from_log_returns,
    merton_binary_prob,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EST_TZ        = ZoneInfo("America/New_York")
OHLCV_ROOT    = Path(".data/gemini/ohlcv_1m_7d")
CONTRACT_ROOT = Path(".data/gemini/prediction_data")
SIM_ROOT      = Path(".data/gemini/sim_trades")
REAL_ROOT     = Path(".data/gemini/real_trades")

ASSET_SYMBOLS   = {"BTC": "btcusd", "ETH": "ethusd", "SOL": "solusd"}
GEMINI_BASE_URL = "https://api.gemini.com"

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
    "model",          # gbm | ewma | garch | student_t | skewed_t | heston | hybrid_t
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
    "gemini_order_id",      # buy order ID (empty when simulation)
    "n_contracts_filled",   # contracts actually filled by IOC buy (0 when simulation)
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

REAL_LEDGER_FIELDS = PERF_LEDGER_FIELDS + [
    "total_real_dollars",         # sum of pnl × n_contracts_filled (real money P&L)
    "avg_real_dollars_per_trade", # total_real_dollars / n_trades
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
    # Hybrid EWMA-σ + Student's t  (regime-aware fat tails)
    "nu_hybrid", "sigma_annual_hybrid",
    "p_fair_hybrid", "edge_yes_hybrid", "edge_no_hybrid",
    # OU (Ornstein-Uhlenbeck on log-price)
    "kappa_ou", "sigma_annual_ou",
    "p_fair_ou", "edge_yes_ou", "edge_no_ou",
    # Heston EWMA-calibrated variance
    "v0_heston_ewma", "theta_heston_ewma",
    "p_fair_heston_ewma", "edge_yes_heston_ewma", "edge_no_heston_ewma",
    # Jump diagnostics (shared across jump models — calibrated from GBM lookback)
    "lambda_j", "mu_j", "sigma_j",
    # Merton Jump-Diffusion variants (p_fair + edges)
    "p_fair_gbm_jump",    "edge_yes_gbm_jump",    "edge_no_gbm_jump",
    "p_fair_ewma_jump",   "edge_yes_ewma_jump",   "edge_no_ewma_jump",
    "p_fair_garch_jump",  "edge_yes_garch_jump",  "edge_no_garch_jump",
    "p_fair_stud_jump",   "edge_yes_stud_jump",   "edge_no_stud_jump",
    "p_fair_hybrid_jump", "edge_yes_hybrid_jump", "edge_no_hybrid_jump",
]


# Maps model_name (as used in trades CSV / active_models list) → p_key used
# in _compute_edges result dict (e.g. "p_fair_{p_key}").  Referenced in the
# entry loop (confidence computation) and in _check_early_exit.
_MODEL_TO_PKEY: dict[str, str] = {
    "gbm":            "gbm",
    "ewma":           "ewma",
    "garch":          "garch",
    "student_t":      "stud",
    "skewed_t":       "skt",
    "heston":         "heston",
    "hybrid_t":       "hybrid",
    "ou":             "ou",
    "heston_ewma":    "heston_ewma",
    "gbm_jump":       "gbm_jump",
    "ewma_jump":      "ewma_jump",
    "garch_jump":     "garch_jump",
    "student_t_jump": "stud_jump",
    "hybrid_t_jump":  "hybrid_jump",
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _today_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d")


def _real_trade_out_path() -> Path:
    REAL_ROOT.mkdir(parents=True, exist_ok=True)
    return REAL_ROOT / f"trades_{_today_str()}.csv"


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
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _fetch_live_contract_quote(
    session: requests.Session,
    event_ticker: str,
    contract_id: str,
) -> dict | None:
    """Fetch live bid/ask for a specific Gemini prediction contract.

    Called at the moment of entry or early exit to get the true current
    execution price rather than the up-to-60s-stale CSV value.

    Returns {bid_yes, ask_yes, bid_no, ask_no} or None on any failure.
    """
    try:
        r = session.get(
            f"{GEMINI_BASE_URL}/v1/prediction-markets/events",
            timeout=10,
        )
        r.raise_for_status()
        data   = r.json()
        events = data.get("data", data) if isinstance(data, dict) else data
        for event in events:
            if event.get("ticker") != event_ticker:
                continue
            for contract in event.get("contracts", []):
                if str(contract.get("id", "")) != str(contract_id):
                    continue
                prices = contract.get("prices", {})
                return {
                    "bid_yes": _safe_float(prices.get("sell", {}).get("yes")),
                    "ask_yes": _safe_float(prices.get("buy",  {}).get("yes")),
                    "bid_no":  _safe_float(prices.get("sell", {}).get("no")),
                    "ask_no":  _safe_float(prices.get("buy",  {}).get("no")),
                }
    except Exception:
        pass
    return None


def _fetch_live_spots(session: requests.Session) -> dict[str, float]:
    """Fetch current bid/ask midpoint for each asset from the Gemini public ticker.

    OHLCV files only contain *closed* candles (by construction in
    getdata_underlying.py), so the last bar is always 1-2 minutes old.
    The OHLCV file is a calibration store, not a live price feed — this
    function provides the actual current spot for the pricing formula.

    Falls back gracefully per-asset: if the API call fails, that asset's
    entry is absent and the caller falls back to the last OHLCV close.
    """
    spots: dict[str, float] = {}
    for asset, sym in ASSET_SYMBOLS.items():
        try:
            r = session.get(f"{GEMINI_BASE_URL}/v1/pubticker/{sym}", timeout=5)
            r.raise_for_status()
            data = r.json()
            spots[asset] = (float(data["bid"]) + float(data["ask"])) / 2.0
        except Exception:
            pass  # caller falls back to OHLCV last close
    return spots


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
    spot: float,
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
    vol_veto_mult: float = 2.0,
) -> dict | None:
    """Calibrate all 14 models and return fair values + edge signals.

    Models: gbm | ewma | garch | student_t | skewed_t | heston | hybrid_t | ou | heston_ewma
            + gbm_jump | ewma_jump | garch_jump | student_t_jump | hybrid_t_jump

    `spot` is the current underlying price, fetched live from the Gemini
    ticker by the caller — NOT read from the OHLCV file.  The OHLCV file
    only contains closed candles (always 1-2 min old), so it is used solely
    for vol calibration.  Separating live spot from historical calibration
    data removes the need for an OHLCV staleness guard.

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

    # Per-model log-return and close-price arrays — cached by window size.
    _lr_cache:     dict[float, object] = {}
    _close_cache:  dict[float, object] = {}
    def _get_lr(lb: float):
        if lb not in _lr_cache:
            _lr_cache[lb] = _minute_log_returns_before(minute_df, eval_ts, lb)
        return _lr_cache[lb]
    def _get_closes(lb: float):
        if lb not in _close_cache:
            _close_cache[lb] = _minute_closes_before(minute_df, eval_ts, lb)
        return _close_cache[lb]

    lb_gbm        = lookbacks.get("gbm",        24.0)
    lb_ewma       = lookbacks.get("ewma",       48.0)
    lb_garch      = lookbacks.get("garch",      72.0)
    lb_stud       = lookbacks.get("stud",       48.0)
    lb_skt        = lookbacks.get("skt",        48.0)
    lb_heston     = lookbacks.get("heston",     96.0)
    lb_hybrid     = lookbacks.get("hybrid",     48.0)
    lb_ou         = lookbacks.get("ou",         12.0)
    lb_heston_ewma= lookbacks.get("heston_ewma",96.0)

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

    # ── Vol veto: block entry when short-term vol spikes above EWMA baseline ──
    # Uses last 10 one-min bars (≈ 10 minutes) for the realised vol estimate.
    # If that spike is > vol_veto_mult × EWMA vol, models are unreliable and
    # we skip pricing entirely — returning None stops all entry signals.
    if vol_veto_mult > 0:
        _recent_lr = _get_lr(10 / 60)  # last 10 min
        if len(_recent_lr) >= 5:
            _sigma_recent_hourly = float(np.std(_recent_lr)) * sqrt(60)
            _sigma_ewma_hourly   = (ewma_par.sigma_per_sqrt_hour
                                    if ewma_par is not None
                                    else gbm.sigma_per_sqrt_hour)
            if _sigma_recent_hourly > vol_veto_mult * _sigma_ewma_hourly:
                return None  # high-vol regime veto — skip all model signals

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

    # ── Model 7: Hybrid EWMA-σ + Student's t (regime-aware fat tails) ───────
    # Uses EWMA conditional vol (responsive to current moves) as σ, and fits ν
    # from the kurtosis of EWMA-normalised residuals (tail shape only).
    # Falls back to student_t result if calibration fails.
    try:
        min_lr_hybrid = _get_lr(lb_hybrid)
        ewma_for_hybrid = ewma_par if ewma_par is not None else calibrate_ewma_from_log_returns(
            min_lr_hybrid, lambda_=ewma_lambda)
        hybrid_par = calibrate_hybrid_t_from_log_returns(min_lr_hybrid, ewma_for_hybrid)
        # Scale to horizon (same as student_t: loc scales by n bars, scale by sqrt(n))
        nh = max(1, round(horizon_hours * 60))
        from btc_hourly_model import HybridTParams as _HybridTParams
        hybrid_h = _HybridTParams(
            sigma_per_sqrt_hour = hybrid_par.sigma_per_sqrt_hour,
            nu                  = hybrid_par.nu,
            mu_per_hour         = hybrid_par.mu_per_hour,
        )
        p_hybrid = hybrid_t_binary_prob(spot, strike, hybrid_h,
                                        horizon_hours=horizon_hours,
                                        direction=direction,
                                        n_sims=20_000, seed=seed + 3)
    except Exception:
        p_hybrid   = p_stud
        hybrid_par = None

    # ── Model 8: OU (Ornstein-Uhlenbeck on log-price) ────────────────────────
    # Closed-form pricing; captures short-term mean reversion.
    # ln_s0 is overridden to log(live spot) so the starting point is exact.
    try:
        closes_ou = _get_closes(lb_ou)
        ou_par    = calibrate_ou_from_closes(closes_ou)
        ou_par    = type(ou_par)(
            kappa = ou_par.kappa,
            mu_ln = ou_par.mu_ln,
            sigma = ou_par.sigma,
            ln_s0 = float(np.log(spot)),   # use live spot, not last close
        )
        p_ou = ou_binary_prob(spot, strike, ou_par,
                              horizon_hours=horizon_hours, direction=direction)
    except Exception:
        p_ou   = p_gbm
        ou_par = None

    # ── Model 9: Heston with EWMA-calibrated variance ────────────────────────
    # theta is EWMA of hourly realised variances (lambda_h=0.9 / hourly bar).
    # Regime-aware: vol spikes lift theta, preventing over-optimistic reversion.
    try:
        min_lr_heston_ewma = _get_lr(lb_heston_ewma)
        heston_ewma_par = calibrate_heston_ewma_from_log_returns(
            min_lr_heston_ewma, rho=rho_use, dt_hours=1 / 60
        )
        p_heston_ewma = heston_binary_prob(spot, strike, heston_ewma_par,
                                           horizon_hours=horizon_hours,
                                           direction=direction,
                                           n_sims=20_000, seed=seed + 4)
    except Exception:
        p_heston_ewma   = p_heston
        heston_ewma_par = None

    # ── Models 10-14: Merton Jump-Diffusion variants ──────────────────────
    # Each variant reuses its base model's sigma+mu and overlays jump pricing.
    # Jump calibration uses the same lookback as the base model.
    lb_jump = lookbacks.get("gbm_jump",      lb_gbm)
    lb_jump_ewma    = lookbacks.get("ewma_jump",     lb_ewma)
    lb_jump_garch   = lookbacks.get("garch_jump",    lb_garch)
    lb_jump_stud    = lookbacks.get("student_t_jump",lb_stud)
    lb_jump_hybrid  = lookbacks.get("hybrid_t_jump", lb_hybrid)

    def _merton(mu_h, lr_for_jumps):
        """Helper: calibrate jumps from lr_for_jumps, then call merton_binary_prob.

        calibrate_jumps_from_log_returns returns jp.sigma_diff = diffusion-only
        vol (after stripping identified jump returns). We use jp.sigma_diff as
        sigma_diff — NOT the base model's total sigma — to avoid double-counting
        jump variance. Merton: σ²_total = σ²_diff + λ*(μ_J² + σ_J²).
        """
        try:
            jp = calibrate_jumps_from_log_returns(lr_for_jumps, dt_hours=1 / 60)
            if jp is None:
                return None, None
            return merton_binary_prob(
                spot, strike, direction, horizon_hours,
                sigma_diff=jp.sigma_diff, mu_hourly=mu_h, jump_params=jp,
            ), jp
        except Exception:
            return None, None

    # GBM+Jump
    _p_gbm_j, _jp_gbm = _merton(gbm.mu_per_hour, _get_lr(lb_jump))
    p_gbm_jump = _p_gbm_j if _p_gbm_j is not None else p_gbm

    # EWMA+Jump
    if ewma_par is not None:
        _p_ewma_j, _jp_ewma = _merton(ewma_par.mu_per_hour, _get_lr(lb_jump_ewma))
    else:
        _p_ewma_j, _jp_ewma = None, None
    p_ewma_jump = _p_ewma_j if _p_ewma_j is not None else p_ewma

    # GARCH+Jump — use σ from GARCH forward vol (σ_per_sqrt_hour) if available
    if garch_par is not None:
        _p_garch_j, _jp_garch = _merton(garch_par.mu_per_hour, _get_lr(lb_jump_garch))
    else:
        _p_garch_j, _jp_garch = None, None
    p_garch_jump = _p_garch_j if _p_garch_j is not None else p_garch

    # StudentT+Jump — drift from t-calibration; diffusion-only σ from jump calibration
    _stud_mu_h = tpar_1m.loc * 60   # 1-min loc to per-hour
    _p_stud_j, _jp_stud = _merton(_stud_mu_h, _get_lr(lb_jump_stud))
    p_stud_jump = _p_stud_j if _p_stud_j is not None else p_stud

    # HybridT+Jump — EWMA drift + jump overlay; diffusion-only σ from jump calibration
    if hybrid_par is not None:
        _p_hyb_j, _jp_hyb = _merton(hybrid_par.mu_per_hour, _get_lr(lb_jump_hybrid))
    else:
        _p_hyb_j, _jp_hyb = None, None
    p_hybrid_jump = _p_hyb_j if _p_hyb_j is not None else p_hybrid

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
        # Hybrid EWMA-σ + Student's t
        "nu_hybrid":          round(hybrid_par.nu, 2) if hybrid_par else None,
        "sigma_annual_hybrid": round(hybrid_par.sigma_per_sqrt_hour * sqrt(8_760), 4) if hybrid_par else None,
        "p_fair_hybrid":      round(p_hybrid, 4),
        # OU
        "kappa_ou":           round(ou_par.kappa, 4) if ou_par else None,
        "sigma_annual_ou":    round(ou_par.sigma * sqrt(8_760), 4) if ou_par else None,
        "p_fair_ou":          round(p_ou, 4),
        # Heston EWMA
        "v0_heston_ewma":     round(heston_ewma_par.v0,    6) if heston_ewma_par else None,
        "theta_heston_ewma":  round(heston_ewma_par.theta, 6) if heston_ewma_par else None,
        "p_fair_heston_ewma": round(p_heston_ewma, 4),
        # Jump models — p_fair only (jump params logged via base model)
        "p_fair_gbm_jump":      round(p_gbm_jump,   4),
        "p_fair_ewma_jump":     round(p_ewma_jump,  4),
        "p_fair_garch_jump":    round(p_garch_jump, 4),
        "p_fair_stud_jump":     round(p_stud_jump,  4),
        "p_fair_hybrid_jump":   round(p_hybrid_jump,4),
        # Jump calibration diagnostics (from GBM lookback, representative)
        "lambda_j":    round(_jp_gbm.lambda_j, 4) if _jp_gbm else None,
        "mu_j":        round(_jp_gbm.mu_j,     6) if _jp_gbm else None,
        "sigma_j":     round(_jp_gbm.sigma_j,  6) if _jp_gbm else None,
    }

    # ── Edge signals for each model ────────────────────────────────────────
    for key, p in [("gbm", p_gbm), ("ewma", p_ewma), ("garch", p_garch),
                   ("stud", p_stud), ("skt", p_skt), ("heston", p_heston),
                   ("hybrid", p_hybrid), ("ou", p_ou), ("heston_ewma", p_heston_ewma),
                   ("gbm_jump", p_gbm_jump), ("ewma_jump", p_ewma_jump),
                   ("garch_jump", p_garch_jump), ("stud_jump", p_stud_jump),
                   ("hybrid_jump", p_hybrid_jump)]:
        result[f"edge_yes_{key}"] = round(p - ask_yes, 4)        if ask_yes is not None else None
        result[f"edge_no_{key}"]  = round((1.0 - p) - ask_no, 4) if ask_no  is not None else None

    # ── Data confidence: fraction of the lookback window actually filled ──────
    # 1.0 = window fully populated; < 1.0 = OHLCV gap reduces reliability.
    # Penalises missing data, NOT short lookback windows (that's a deliberate
    # design choice for regime adaptation, not a data-quality problem).
    result["data_conf_gbm"]          = min(1.0, len(min_lr_gbm)               / max(1, lb_gbm         * 60))
    result["data_conf_ewma"]         = min(1.0, len(_get_lr(lb_ewma))         / max(1, lb_ewma        * 60))
    result["data_conf_garch"]        = min(1.0, len(_get_lr(lb_garch))        / max(1, lb_garch       * 60))
    result["data_conf_stud"]         = min(1.0, len(_get_lr(lb_stud))         / max(1, lb_stud        * 60))
    result["data_conf_skt"]          = min(1.0, len(_get_lr(lb_skt))          / max(1, lb_skt         * 60))
    result["data_conf_heston"]       = min(1.0, len(_get_lr(lb_heston))       / max(1, lb_heston      * 60))
    result["data_conf_hybrid"]       = min(1.0, len(_get_lr(lb_hybrid))       / max(1, lb_hybrid      * 60))
    result["data_conf_ou"]           = min(1.0, len(_get_lr(lb_ou))           / max(1, lb_ou          * 60))
    result["data_conf_heston_ewma"]  = min(1.0, len(_get_lr(lb_heston_ewma))  / max(1, lb_heston_ewma * 60))
    result["data_conf_gbm_jump"]     = min(1.0, len(_get_lr(lb_jump))         / max(1, lb_jump        * 60))
    result["data_conf_ewma_jump"]    = min(1.0, len(_get_lr(lb_jump_ewma))    / max(1, lb_jump_ewma   * 60))
    result["data_conf_garch_jump"]   = min(1.0, len(_get_lr(lb_jump_garch))   / max(1, lb_jump_garch  * 60))
    result["data_conf_stud_jump"]    = min(1.0, len(_get_lr(lb_jump_stud))    / max(1, lb_jump_stud   * 60))
    result["data_conf_hybrid_jump"]  = min(1.0, len(_get_lr(lb_jump_hybrid))  / max(1, lb_jump_hybrid * 60))

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
        "nu_hybrid":                edges.get("nu_hybrid"),
        "sigma_annual_hybrid":      edges.get("sigma_annual_hybrid"),
        "kappa_ou":                 edges.get("kappa_ou"),
        "sigma_annual_ou":          edges.get("sigma_annual_ou"),
        "v0_heston_ewma":           edges.get("v0_heston_ewma"),
        "theta_heston_ewma":        edges.get("theta_heston_ewma"),
        "exit_time_utc":            None,
        "exit_reason":              None,
        "exit_bid":                 None,
        "spot_at_settle":           None,
        "outcome":                  None,
        "pnl":                      None,
        "status":                   "open",
        "gemini_order_id":          "",    # filled when placing real orders
        "n_contracts_filled":       0,     # contracts still held (decrements on partial sell)
        "n_contracts_original":     0,     # original fill count; never changes after BUY
        "partial_realized_pnl":     0.0,   # accumulated per-unit PnL from partial sells
    }


def _check_early_exit(
    pos: dict,
    latest_row: dict | None,
    minute_df: pd.DataFrame | None,
    spot: float | None,
    lookbacks: dict[str, float],
    now_utc: datetime,
    profit_lock: float,
    stop_loss: float,
    p_drop: float,
    ewma_lambda: float = 0.94,
    rho: float = -0.5,
    edge_neg_thresh: float = 0.0,
    vol_veto_mult: float = 2.0,
    precomputed_edges: dict | None = None,
    conf_active: list[str] | None = None,  # for ensemble p_fair recomputation at exit
) -> dict | None:
    """Check whether an open position should be closed before settlement.

    Checks four conditions in priority order:
      1. profit_lock  — bid has moved far enough in our favour → lock in gain.
                        Skipped when < 30 min to settlement (let it run to expiry).
      2. stop_loss    — (bid_entry - bid_now) / ask_entry >= stop_loss (relative).
                        50% of entry price means a 19¢ entry stops at 9.5¢.
      3. p_drop       — model probability for our side has fallen → information exit.
      4. edge_closed  — current edge < -edge_neg_thresh (negative by more than the
                        hysteresis band).  0 = any negative edge triggers; 0.02 =
                        edge must be at least −2¢ before we exit.

    Returns a dict with exit_reason/exit_bid/exit_pnl on trigger, else None.
    """
    if latest_row is None:
        return None

    # If the contract has already passed its settlement time, do NOT fire
    # early-exit signals — let _try_settle() handle it once OHLCV updates.
    # Without this guard, profit_lock / stop_loss can close the position via
    # the stale bid (e.g. bid→1.0 on a won contract) before settlement runs,
    # recording a sub-optimal exit_bid instead of the true settlement value.
    try:
        settle_time_utc = datetime.fromisoformat(pos["settle_time_utc"])
        if now_utc >= settle_time_utc:
            return None
    except (KeyError, ValueError):
        pass

    side      = pos["side"]
    ask_entry = float(pos["ask_price"])

    # Current bid for our side — what the market pays us to sell now.
    bid_now = _safe_float(latest_row.get("bid_yes" if side == "YES" else "bid_no"))
    if bid_now is None:
        return None

    realised_pnl = round(bid_now - ask_entry, 4)

    # --- 1. Profit lock-in (skip when < 30 min to settlement — let it expire) ---
    try:
        mins_to_settle = (datetime.fromisoformat(pos["settle_time_utc"]) - now_utc).total_seconds() / 60
    except (KeyError, ValueError):
        mins_to_settle = 999.0
    if realised_pnl >= profit_lock and mins_to_settle >= 30.0:
        return {"exit_reason": "profit_lock", "exit_bid": bid_now, "exit_pnl": realised_pnl}

    # --- 2. Stop loss (relative: loss as fraction of entry price) ---
    # stop_loss = 0.50 means exit when bid drops to < 50% of ask_entry.
    # e.g. entered at 19¢ → stop at 9.5¢; entered at 64¢ → stop at 32¢.
    if ask_entry > 0 and -realised_pnl / ask_entry >= stop_loss:
        return {"exit_reason": "stop_loss", "exit_bid": bid_now, "exit_pnl": realised_pnl}

    # --- 3 & 4. Model-based exits (require fresh calibration + live spot) ---
    if minute_df is not None and spot is not None:
        try:
            strike          = float(pos["strike"])
            direction       = pos["direction"]
            settle_time_utc = datetime.fromisoformat(pos["settle_time_utc"])
            ask_yes         = _safe_float(latest_row.get("ask_yes"))
            ask_no          = _safe_float(latest_row.get("ask_no"))

            if precomputed_edges is not None:
                edges = precomputed_edges
            else:
                edges = _compute_edges(
                    contract_id     = pos["contract_id"],
                    spot            = spot,
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
                    vol_veto_mult   = vol_veto_mult,
                )

            if edges is not None:
                _p_key_map = {
                    "gbm":           "gbm",
                    "ewma":          "ewma",
                    "garch":         "garch",
                    "student_t":     "stud",
                    "skewed_t":      "skt",
                    "heston":        "heston",
                    "hybrid_t":      "hybrid",
                    "ou":            "ou",
                    "heston_ewma":   "heston_ewma",
                    "gbm_jump":      "gbm_jump",
                    "ewma_jump":     "ewma_jump",
                    "garch_jump":    "garch_jump",
                    "student_t_jump":"stud_jump",
                    "hybrid_t_jump": "hybrid_jump",
                }
                if pos["model"] == "ensemble":
                    # Recompute true ensemble p_fair from conf_active models.
                    # All 14 model p_fairs are available in edges; take the mean
                    # of whichever conf_active models have a result.
                    _eff = conf_active if conf_active else list(_MODEL_TO_PKEY.keys())
                    _exit_pfairs = [
                        float(edges[f"p_fair_{_MODEL_TO_PKEY[mn]}"])
                        for mn in _eff
                        if _MODEL_TO_PKEY.get(mn) and edges.get(f"p_fair_{_MODEL_TO_PKEY[mn]}") is not None
                    ]
                    p_fair_now = float(np.mean(_exit_pfairs)) if len(_exit_pfairs) >= 2 else edges["p_fair_heston_ewma"]
                else:
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

                # 4. Edge has closed past hysteresis band.
                # Computed directly from p_fair_now (works for all models including
                # ensemble where p_key is not set).
                # edge_neg_thresh = 0.0 → any negative edge triggers.
                # edge_neg_thresh = 0.02 → edge must be ≤ −0.02 to trigger.
                _ask_now = ask_yes if side == "YES" else ask_no
                if _ask_now is not None:
                    current_edge = round(
                        (p_fair_now - _ask_now) if side == "YES"
                        else ((1.0 - p_fair_now) - _ask_now), 4
                    )
                else:
                    current_edge = None
                if current_edge is not None and current_edge <= -edge_neg_thresh:
                    return {
                        "exit_reason": f"edge_closed({current_edge:.3f})",
                        "exit_bid":    bid_now,
                        "exit_pnl":    realised_pnl,
                    }
        except Exception:
            pass

    return None  # Hold


def _apply_early_exit(pos: dict, exit_info: dict, now_utc: datetime) -> None:
    """Mutate pos with early-exit closing fields.

    Blends partial-sell PnL (from any prior IOC partial fills) with the
    current exit PnL so the recorded figure reflects the true blended
    per-unit return across both sell legs.
    """
    n_original = pos.get("n_contracts_original") or pos.get("n_contracts_filled") or 1
    n_remaining = pos.get("n_contracts_filled", n_original)
    partial_pnl = pos.get("partial_realized_pnl", 0.0)

    exit_pnl = exit_info["exit_pnl"]
    if partial_pnl and n_original > 0:
        # Blend: (per-unit-gain-on-remaining * n_remaining + cumulative-partial) / n_original
        blended = round((exit_pnl * n_remaining + partial_pnl) / n_original, 4)
    else:
        blended = exit_pnl

    pos.update({
        "exit_time_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "exit_reason":   exit_info["exit_reason"],
        "exit_bid":      exit_info["exit_bid"],
        "pnl":           blended,
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

    # Use bars strictly BEFORE settle_ts: the last such bar's close is the
    # price at exactly settle_ts (candle timestamps mark the open, so the
    # 13:59 bar's close = price at 14:00 = correct settlement reference).
    # Using <= would include the 14:00 bar whose close is the price at 14:01.
    bars = minute_df.loc[minute_df.index < settle_ts]
    if bars.empty:
        return False

    spot_at_settle = float(bars["close"].iloc[-1])
    direction      = pos["direction"]
    strike         = float(pos["strike"])

    outcome = ("YES_WIN" if spot_at_settle > strike else "NO_WIN") if direction == "above" \
         else ("YES_WIN" if spot_at_settle <= strike else "NO_WIN")

    ask_price   = float(pos["ask_price"])
    won         = (pos["side"] == "YES" and outcome == "YES_WIN") or \
                  (pos["side"] == "NO"  and outcome == "NO_WIN")
    settle_pnl  = (1.0 - ask_price) if won else (-ask_price)

    # Blend with any accumulated partial-sell PnL from prior IOC partial fills.
    n_original  = pos.get("n_contracts_original") or pos.get("n_contracts_filled") or 1
    n_remaining = pos.get("n_contracts_filled", n_original)
    partial_pnl = pos.get("partial_realized_pnl", 0.0)
    if partial_pnl and n_original > 0:
        pnl = round((settle_pnl * n_remaining + partial_pnl) / n_original, 4)
    else:
        pnl = round(settle_pnl, 4)

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

def _update_performance_ledger(sim_root: Path = SIM_ROOT) -> dict[str, dict]:
    """Recompute and overwrite performance_ledger.csv from ALL historical trade files.

    Reads every trades_*.csv in sim_root so the ledger accumulates across
    days and model versions.  Old trade rows are never deleted — this is
    purely a derived aggregate view of the raw per-day files.

    Ledger schema: see PERF_LEDGER_FIELDS.
    Profitable = pnl > 0  (covers both settlement wins and early-exit gains).

    Returns a dict keyed by model name for in-memory use by _print_summary.
    """
    sim_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for fp in sorted(sim_root.glob("trades_*.csv")):
        try:
            with fp.open(newline="") as f:
                all_rows.extend(list(csv.DictReader(f)))
        except Exception:
            pass

    # Only closed positions contribute to the ledger.
    closed = [r for r in all_rows if r.get("status") in ("settled", "exited_early")]

    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")
    out_rows = []

    _ALL_MODELS = ("gbm", "ewma", "garch", "student_t", "skewed_t", "heston",
                   "hybrid_t", "ou", "heston_ewma",
                   "gbm_jump", "ewma_jump", "garch_jump", "student_t_jump", "hybrid_t_jump",
                   "ensemble")

    for model in _ALL_MODELS:
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

    ledger_path = sim_root / "performance_ledger.csv"
    with ledger_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PERF_LEDGER_FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    return {r["model"]: r for r in out_rows}


def _update_real_ledger() -> None:
    """Recompute and overwrite real_trades/performance_ledger.csv from real trade files.

    Same structure as the sim ledger but includes real-dollar columns
    (total_real_dollars, avg_real_dollars_per_trade) derived from
    pnl × n_contracts_filled.  Only written when live_trader.py is running.
    """
    REAL_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for fp in sorted(REAL_ROOT.glob("trades_*.csv")):
        try:
            with fp.open(newline="") as f:
                all_rows.extend(list(csv.DictReader(f)))
        except Exception:
            pass

    closed = [r for r in all_rows if r.get("status") in ("settled", "exited_early")]

    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")
    out_rows = []

    _ALL_MODELS = ("gbm", "ewma", "garch", "student_t", "skewed_t", "heston",
                   "hybrid_t", "ou", "heston_ewma",
                   "gbm_jump", "ewma_jump", "garch_jump", "student_t_jump", "hybrid_t_jump",
                   "ensemble")

    for model in _ALL_MODELS:
        trades = [r for r in closed if r.get("model") == model]

        if not trades:
            out_rows.append({k: ("—" if k not in ("model", "as_of") else
                                 (model if k == "model" else now_str))
                             for k in REAL_LEDGER_FIELDS})
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

        real_dollars: list[float] = []
        for r, p in zip(trades, pnls):
            try:
                n = int(float(r.get("n_contracts_filled") or 0))
                real_dollars.append(p * n)
            except (ValueError, TypeError):
                real_dollars.append(0.0)

        n            = len(trades)
        n_profitable = sum(1 for p in pnls if p > 0)
        n_settled    = sum(1 for r in trades if r.get("status") == "settled")
        n_early      = sum(1 for r in trades if r.get("status") == "exited_early")
        total_real   = round(sum(real_dollars), 4)

        out_rows.append({
            "model":                    model,
            "n_trades":                 n,
            "n_profitable":             n_profitable,
            "n_loss":                   n - n_profitable,
            "n_settled":                n_settled,
            "n_early_exit":             n_early,
            "win_rate_pct":             round(100.0 * n_profitable / n, 1),
            "avg_pnl":                  round(sum(pnls) / n, 4),
            "total_pnl":                round(sum(pnls), 4),
            "avg_edge_at_entry":        round(sum(edges) / len(edges), 4) if edges else "—",
            "as_of":                    now_str,
            "total_real_dollars":       total_real,
            "avg_real_dollars_per_trade": round(total_real / n, 4),
        })

    ledger_path = REAL_ROOT / "performance_ledger.csv"
    with ledger_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REAL_LEDGER_FIELDS)
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


def _fmt_settle(settle_time_utc_str: str) -> str:
    """Return e.g. '@8PM EST' from a UTC ISO string."""
    try:
        settle_utc = datetime.fromisoformat(settle_time_utc_str)
        settle_est = settle_utc.astimezone(EST_TZ)
        hour_str   = settle_est.strftime("%I%p").lstrip("0")   # "8PM"
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
    elif model == "hybrid_t":
        nu   = pos.get("nu_hybrid") or 0.0
        s_an = (pos.get("sigma_annual_hybrid") or 0.0) * 100
        params = f"ν={nu:.1f}  σ_ewma={s_an:.1f}%"
    elif model == "ou":
        kappa = pos.get("kappa_ou") or 0.0
        s_an  = (pos.get("sigma_annual_ou") or 0.0) * 100
        params = f"κ={kappa:.2f}/h  σ={s_an:.1f}%"
    elif model == "heston_ewma":
        v0    = pos.get("v0_heston_ewma")    or 0.0
        theta = pos.get("theta_heston_ewma") or 0.0
        s_an  = sqrt(v0 * 8_760) * 100 if v0 > 0 else 0.0
        params = f"v0_σ={s_an:.1f}%  θ={theta:.6f}"
    elif model in ("gbm_jump", "ewma_jump", "garch_jump", "hybrid_t_jump"):
        base = model.replace("_jump", "")
        s_key = {"gbm": "sigma_annual_gbm", "ewma": "sigma_annual_ewma",
                 "garch": "sigma_annual_garch", "hybrid_t": "sigma_annual_hybrid"}.get(base, "sigma_annual_gbm")
        s_an = (pos.get(s_key) or 0.0) * 100
        params = f"σ_diff={s_an:.1f}%  +jump"
    elif model == "student_t_jump":
        s_an = (pos.get("sigma_annual_stud") or 0.0) * 100
        nu   = pos.get("nu_stud") or 0.0
        params = f"ν={nu:.1f}  σ={s_an:.1f}%  +jump"
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
    n         = pos.get("n_contracts_filled")
    dollar_str = f"  real=${pnl * n:+.2f}" if n else ""
    print(
        f"  [EXIT]   {label:<22}  {side}  {reason:<28}  "
        f"entry={entry_pct:.1f}%  bid={bid_pct:.1f}%  pnl={pnl:+.2f}{dollar_str}   [{model}]"
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
    n         = pos.get("n_contracts_filled")
    dollar_str = f"  real=${pnl * n:+.2f}" if n else ""
    print(
        f"  [SETTLE] {label:<22}  {side}  {result}  "
        f"spot={spot_str}  strike=${strike:,.0f}  pnl={pnl:+.2f}{dollar_str}   [{model}]"
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(
    now_utc: datetime,
    open_pos: list[dict],
    session_closed: int,
    ledger: dict[str, dict],
    session_pnl: float = 0.0,   # real dollar P&L this session (contracts × per-unit pnl)
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
        fmt("gbm",           "GBM"),
        fmt("ewma",          "EWMA"),
        fmt("garch",         "GARCH"),
        fmt("student_t",     "StudT"),
        fmt("skewed_t",      "SktT"),
        fmt("heston",        "Heston"),
        fmt("hybrid_t",      "HybT"),
        fmt("ou",            "OU"),
        fmt("heston_ewma",   "HestEW"),
        fmt("gbm_jump",      "GBM+J"),
        fmt("ewma_jump",     "EWMA+J"),
        fmt("garch_jump",    "GARCH+J"),
        fmt("student_t_jump","StudT+J"),
        fmt("hybrid_t_jump", "HybT+J"),
        fmt("ensemble",      "ENS"),
    ])
    pnl_str = f"  session_pnl=${session_pnl:+.2f}" if session_pnl != 0.0 else ""
    print(f"[{ts}]  {open_str}  session_closed={session_closed}{pnl_str}  {parts}")


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run(
    poll_sec:           int   = 60,
    lookbacks:          dict[str, float] | None = None,
    min_edge:           float = 0.03,
    max_hours_to_settle:float = 1.5,
    min_hours_to_settle:float = 0.0,   # skip contracts with less time than this (illiquid near-expiry)
    profit_lock:        float = 0.05,
    stop_loss:          float = 0.50,
    p_drop:             float = 0.05,
    ewma_lambda:        float = 0.94,
    rho:                float = -0.5,
    edge_neg_thresh:    float = 0.02,
    vol_veto_mult:      float = 2.0,
    no_collectors:      bool  = False,
    trader=None,        # GeminiTrader | None  — None = simulation only
    active_models:      set | None = None,  # restrict entry to these models; None = all 14
    sim_root:           Path = SIM_ROOT,    # output directory for trades/ledger
    # ── Confidence system ─────────────────────────────────────────────────
    # Adjusts raw_edge → edge_adj = raw_edge × total_conf before comparing to
    # min_edge.  total_conf = pred_conf^λ1 × data_conf^λ2 × ens_conf^λ3.
    conf_pred_w:        float = 0.55,   # λ1 weight for model-disagreement component
    conf_data_w:        float = 0.10,   # λ2 weight for OHLCV data-gap component
    conf_ens_w:         float = 0.35,   # λ3 weight for directional-agreement component
    conf_k:             float = 3.0,    # pred_conf harshness: conf = max(0, 1 - k×std)
    conf_active:        list[str] | None = None,  # models for confidence; None = all 14
    # ── Sim friction model (ignored when trader is set) ───────────────────
    zero_fill_prob:     float = 0.15,   # prob of simulating an IOC zero-fill
    entry_slip_max:     float = 0.02,   # max extra ¢ added to entry ask (uniform)
    exit_slip_max:      float = 0.01,   # max ¢ subtracted from exit bid (uniform)
) -> None:
    """Poll contract data, calibrate all 14 models, simulate trades, manage exits."""
    _lb_defaults = {"gbm": 12.0, "ewma": 24.0, "garch": 48.0,
                    "stud": 24.0, "skt": 24.0, "heston": 72.0,
                    "hybrid": 24.0, "ou": 12.0, "heston_ewma": 72.0,
                    "gbm_jump": 24.0, "ewma_jump": 36.0, "garch_jump": 48.0,
                    "student_t_jump": 36.0, "hybrid_t_jump": 36.0}
    if lookbacks is None:
        lookbacks = _lb_defaults
    else:
        # fill in any missing keys with defaults
        lookbacks = {**_lb_defaults, **lookbacks}

    lb_str = "  ".join(f"{k}={v:.0f}h" for k, v in lookbacks.items())
    _conf_active_str = ",".join(conf_active) if conf_active else "all"
    print(
        f"Starting live trading simulation\n"
        f"  poll={poll_sec}s  min_edge={min_edge:.1%}  max_horizon={max_hours_to_settle}h\n"
        f"  lookbacks: {lb_str}\n"
        f"  exit: profit_lock={profit_lock:.2f}  stop_loss={stop_loss:.0%}(rel)  "
        f"p_drop={p_drop:.2f}  edge_neg_thresh={edge_neg_thresh:.3f}\n"
        f"  vol models: ewma_lambda={ewma_lambda}  rho_prior={rho}  vol_veto_mult={vol_veto_mult}x\n"
        f"  confidence: λ=(pred={conf_pred_w},data={conf_data_w},ens={conf_ens_w})  "
        f"k={conf_k}  active=[{_conf_active_str}]\n"
        f"  sim friction: zero_fill_prob={zero_fill_prob:.0%}  "
        f"entry_slip=[0,{entry_slip_max:.2f}]  exit_slip=[0,{exit_slip_max:.2f}]"
        + ("  [DISABLED — live trader]" if trader is not None else "")
    )
    # Random number generator for sim friction (unseeded = different each run).
    _rng = np.random.default_rng()
    sim_root.mkdir(parents=True, exist_ok=True)
    print(f"Output → {sim_root.resolve()}")

    trade_out = sim_root / f"trades_{_today_str()}.csv"
    edge_out  = sim_root / f"edge_log_{_today_str()}.csv"
    _ensure_header(trade_out, TRADE_FIELDS)
    _ensure_header(edge_out,  EDGE_FIELDS)

    # Real-money trade log — only created when a live trader is injected AND
    # sim_root differs from REAL_ROOT.  When they are the same path, trade_out
    # IS the real trade log so writing again would produce duplicate rows.
    real_trade_out: Path | None = None
    if trader is not None:
        _rt = _real_trade_out_path()
        if _rt.resolve() != trade_out.resolve():
            real_trade_out = _rt
            _ensure_header(real_trade_out, TRADE_FIELDS)
        print(f"Real trades → {REAL_ROOT.resolve()}")

    # Load cumulative ledger at startup so scores show historical data immediately.
    ledger_stats: dict[str, dict] = _update_performance_ledger(sim_root)

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

    seen_gbm:         set[str] = set()
    seen_ewma:        set[str] = set()
    seen_garch:       set[str] = set()
    seen_stud:        set[str] = set()
    seen_skt:          set[str] = set()
    seen_heston:       set[str] = set()
    seen_hybrid:       set[str] = set()
    seen_ou:           set[str] = set()
    seen_heston_ewma:  set[str] = set()
    seen_gbm_jump:     set[str] = set()
    seen_ewma_jump:    set[str] = set()
    seen_garch_jump:   set[str] = set()
    seen_stud_jump:    set[str] = set()
    seen_hybrid_jump:  set[str] = set()
    seen_ensemble:     set[str] = set()   # confidence-weighted mean of conf_active models
    # Contracts that have had their edge signal logged (once per contract per session).
    # Independent of model selection so edge log is written regardless of active_models.
    edge_seen: set[str] = set()
    # Maps model name → its seen_set so closed positions can be removed,
    # allowing re-entry if a new edge appears on the same contract later.
    seen_by_model: dict[str, set[str]] = {
        "gbm":           seen_gbm,
        "ewma":          seen_ewma,
        "garch":         seen_garch,
        "student_t":     seen_stud,
        "skewed_t":      seen_skt,
        "heston":        seen_heston,
        "hybrid_t":      seen_hybrid,
        "ou":            seen_ou,
        "heston_ewma":   seen_heston_ewma,
        "gbm_jump":      seen_gbm_jump,
        "ewma_jump":     seen_ewma_jump,
        "garch_jump":    seen_garch_jump,
        "student_t_jump":seen_stud_jump,
        "hybrid_t_jump": seen_hybrid_jump,
        "ensemble":      seen_ensemble,
    }
    # Entry delay: (sim only) holds pending entries for one poll before executing.
    # Key: model_name. Value: dict[contract_id → entry metadata].
    # On next poll, entries here are executed at the *current* ask price (simulates
    # the 1-poll retry delay seen in live IOC order flow).
    pending_by_model: dict[str, dict] = {
        m: {} for m in seen_by_model
    }
    # Contracts penalised after stop_loss or p_drop — require 2× min_edge to re-enter.
    # Key: f"{model}:{contract_id}".
    reentry_penalised: set[str] = set()
    open_positions:   list[dict] = []
    closed_positions: list[dict] = []
    session_pnl:      float      = 0.0   # real dollar P&L this session

    # Terminate collectors when the sim exits for any reason.
    import atexit
    def _cleanup_collectors():
        for proc in collector_procs:
            if proc and proc.poll() is None:
                proc.terminate()
    atexit.register(_cleanup_collectors)

    session = requests.Session()
    atexit.register(session.close)

    while True:
        t0      = time.time()
        now_utc = datetime.now(tz=timezone.utc)
        today   = now_utc.strftime("%Y%m%d")

        minute_dfs: dict[str, pd.DataFrame | None] = {
            asset: _load_minute_df(asset) for asset in ASSET_SYMBOLS
        }

        rows   = _read_contract_csv(today)
        latest = _latest_by_contract(rows)  # {contract_id: most-recent-row}

        # ── Live spot prices — primary source for pricing formula ────────────
        # OHLCV only contains closed candles (1-2 min lag by design); the ticker
        # gives the true current market price.  Fall back per-asset to the last
        # OHLCV close only if the API call fails.
        live_spots: dict[str, float] = _fetch_live_spots(session)
        for asset, mdf in minute_dfs.items():
            if asset not in live_spots and mdf is not None:
                live_spots[asset] = float(mdf["close"].iloc[-1])

        # ── 1. Early-exit checks for all open positions ─────────────────────
        still_open: list[dict] = []
        # Per-poll caches to avoid redundant work across open positions.
        # _exit_quote_cache: one HTTP call per contract_id per poll.
        # _exit_edges_cache: one full model calibration per
        #   (asset, strike, direction, settle_time) per poll — multiple model
        #   positions on the same contract reuse the same edges dict.
        _exit_quote_cache: dict[str, dict | None] = {}
        _exit_edges_cache: dict[tuple, dict | None] = {}
        for pos in open_positions:
            cid     = pos["contract_id"]
            mdf     = minute_dfs.get(pos["asset"])
            lat_row = latest.get(cid)

            # Pre-fetch live quote so exit DECISIONS use fresh bid/ask, not the
            # CSV value which can be up to poll_sec seconds stale.
            # Cache result so multiple model-positions on the same contract
            # share one HTTP round-trip per poll.
            if cid not in _exit_quote_cache:
                _exit_quote_cache[cid] = _fetch_live_contract_quote(
                    session, pos.get("event_ticker", ""), cid
                )
            live_q = _exit_quote_cache[cid]
            if live_q is not None:
                # Merge fresh bid/ask into lat_row without mutating the cache.
                if lat_row is not None:
                    lat_row = {**lat_row, **{k: v for k, v in live_q.items() if v is not None}}
                else:
                    lat_row = live_q

            # Pre-compute edges once per unique contract (same asset/strike/
            # direction/settle_time) so multiple model-positions on the same
            # contract don't each trigger 14-model Monte Carlo calibration.
            edges_key = (
                pos.get("asset"), pos.get("strike"),
                pos.get("direction"), pos.get("settle_time_utc"),
            )
            if edges_key not in _exit_edges_cache:
                spot_for_exit = live_spots.get(pos["asset"])
                if mdf is not None and spot_for_exit is not None:
                    try:
                        _exit_edges_cache[edges_key] = _compute_edges(
                            contract_id     = cid,
                            spot            = spot_for_exit,
                            strike          = float(pos["strike"]),
                            direction       = pos["direction"],
                            ask_yes         = _safe_float((lat_row or {}).get("ask_yes")),
                            ask_no          = _safe_float((lat_row or {}).get("ask_no")),
                            settle_time_utc = datetime.fromisoformat(pos["settle_time_utc"]),
                            eval_time_utc   = now_utc,
                            minute_df       = mdf,
                            lookbacks       = lookbacks,
                            ewma_lambda     = ewma_lambda,
                            rho             = rho,
                            vol_veto_mult   = vol_veto_mult,
                        )
                    except Exception:
                        _exit_edges_cache[edges_key] = None
                else:
                    _exit_edges_cache[edges_key] = None

            exit_info = _check_early_exit(
                pos, lat_row, mdf,
                spot             = live_spots.get(pos["asset"]),
                lookbacks        = lookbacks,
                now_utc          = now_utc,
                profit_lock      = profit_lock,
                stop_loss        = stop_loss,
                p_drop           = p_drop,
                ewma_lambda      = ewma_lambda,
                rho              = rho,
                edge_neg_thresh  = edge_neg_thresh,
                vol_veto_mult    = vol_veto_mult,
                precomputed_edges = _exit_edges_cache[edges_key],
                conf_active      = conf_active,
            )

            if exit_info:
                # Live quote already fetched above — refresh exit_bid if we got one.
                if live_q is not None:
                    side = pos["side"]
                    fresh_bid = live_q.get("bid_yes" if side == "YES" else "bid_no")
                    if fresh_bid is not None:
                        exit_info["exit_bid"] = fresh_bid
                        exit_info["exit_pnl"] = round(
                            fresh_bid - float(pos["ask_price"]), 4
                        )
                # ── Live sell order (only when we actually hold contracts) ────
                if trader is not None:
                    n_held = pos.get("n_contracts_filled", 0)
                    if n_held > 0:
                        # Use exit_bid directly — it is always set by _check_early_exit
                        # (and possibly refreshed by live quote above).  Do NOT use
                        # "or float(pos['ask_price'])" — that silently replaces a
                        # legitimate 0.0 bid with the entry ask.
                        sell_bid = exit_info["exit_bid"]
                        try:
                            sell = trader.sell_order(
                                contract_id = pos["contract_id"],
                                side        = pos["side"],
                                bid_price   = sell_bid,
                                n_contracts = n_held,
                            )
                            if sell["filled"] < n_held:
                                # Partial IOC fill — unsold contracts remain open.
                                # Accumulate per-unit realized PnL for the sold portion
                                # so the final close can blend both parts correctly.
                                remaining = n_held - sell["filled"]
                                per_unit_gain = sell_bid - float(pos["ask_price"])
                                pos["partial_realized_pnl"] = (
                                    pos.get("partial_realized_pnl", 0.0)
                                    + per_unit_gain * sell["filled"]
                                )
                                pos["n_contracts_filled"] = remaining
                                print(
                                    f"    [SELL]  PARTIAL FILL — sold {sell['filled']}"
                                    f"/{n_held} contracts. "
                                    f"{remaining} remain open; retrying next poll."
                                )
                                still_open.append(pos)
                                continue  # skip _apply_early_exit — position not fully closed
                            else:
                                print(
                                    f"    [SELL]  order_id={sell['order_id']}  "
                                    f"filled={sell['filled']}/{sell['n_contracts']}  "
                                    f"status={sell['status']}"
                                )
                        except Exception as exc:
                            exc_str = str(exc)
                            _no_position = (
                                "InsufficientFunds" in exc_str
                                or "InsufficientPosition" in exc_str
                                # Gemini ValidationError: "Cannot sell YES/NO contracts.
                                # No YES/NO position found." — same root cause.
                                or "No YES position found" in exc_str
                                or "No NO position found" in exc_str
                                or "position found" in exc_str  # catch any future wording
                            )
                            if _no_position:
                                # Exchange says we don't hold these contracts — already
                                # settled or sold externally.  Try to infer settlement
                                # value from spot vs strike if the contract has expired;
                                # otherwise fall back to $0 (external sale, unknown price).
                                _spot_now  = live_spots.get(pos.get("asset", ""))
                                _strike    = _safe_float(pos.get("strike"))
                                _settle_str = pos.get("settle_time_utc", "")
                                _expired   = False
                                try:
                                    _settle_dt = datetime.fromisoformat(
                                        _settle_str.replace("Z", "+00:00")
                                    )
                                    _expired = now_utc >= _settle_dt
                                except Exception:
                                    pass
                                if _expired and _spot_now and _strike:
                                    _direction = pos.get("direction", "HI")
                                    _side_p    = pos.get("side", "NO")
                                    _yes_wins  = (
                                        (_direction == "HI" and _spot_now > _strike)
                                        or (_direction == "LO" and _spot_now <= _strike)
                                    )
                                    _we_win = (
                                        (_side_p == "YES" and _yes_wins)
                                        or (_side_p == "NO" and not _yes_wins)
                                    )
                                    _sv = 1.0 if _we_win else 0.0
                                    print(
                                        f"    [SELL]  Position gone — settled at ${_sv:.2f} "
                                        f"(spot={_spot_now:.0f} vs strike={_strike:.0f}, "
                                        f"dir={_direction}, side={_side_p}) "
                                        f"({exc_str[:60]})"
                                    )
                                else:
                                    _sv = 0.0
                                    print(
                                        f"    [SELL]  Position gone on exchange — "
                                        f"force-closing at $0 ({exc_str[:80]})"
                                    )
                                exit_info["exit_bid"] = _sv
                                exit_info["exit_pnl"] = round(_sv - float(pos["ask_price"]), 4)
                                pos["n_contracts_filled"] = 0  # nothing left to sell
                                # fall through to _apply_early_exit below
                            else:
                                # Other error — keep open and retry next poll
                                print(
                                    f"    [SELL]  FAILED: {exc} "
                                    f"— keeping position open for retry"
                                )
                                still_open.append(pos)
                                continue
                # ── Sim exit slippage ──────────────────────────────────────
                # Models bid deterioration when we hit the bid to exit early.
                # Not applied in live trading (trader is not None handles real fills).
                if trader is None and exit_slip_max > 0:
                    _slip = float(_rng.uniform(0, exit_slip_max))
                    _slipped_bid = max(0.0, round(exit_info["exit_bid"] - _slip, 4))
                    exit_info["exit_bid"] = _slipped_bid
                    exit_info["exit_pnl"] = round(_slipped_bid - float(pos["ask_price"]), 4)
                _apply_early_exit(pos, exit_info, now_utc)
                closed_positions.append(pos)
                _append_csv(trade_out, pos, TRADE_FIELDS)
                if real_trade_out is not None:
                    _append_csv(real_trade_out, pos, TRADE_FIELDS)
                    _update_real_ledger()
                _log_exit(pos, exit_info)
                # Use blended per-unit pnl × original contracts for session display.
                # n_contracts_filled may be 0 after force-close, so use n_original.
                _n_orig = pos.get("n_contracts_original") or pos.get("n_contracts_filled") or 0
                session_pnl += float(pos.get("pnl") or 0.0) * _n_orig
                ledger_stats = _update_performance_ledger(sim_root)
                exit_reason = exit_info.get("exit_reason", "")
                if exit_reason in ("stop_loss", "p_drop"):
                    # Market or model flagged this trade as wrong.  Penalise re-entry:
                    # contract stays in seen_set AND requires 2× edge to re-enter.
                    pen_key = f"{pos['model']}:{pos['contract_id']}"
                    reentry_penalised.add(pen_key)
                    # Keep in seen_set so standard min_edge re-entry is blocked.
                    # The 2× threshold in the entry loop still allows high-conviction
                    # re-entries while filtering opportunistic ones.
                else:
                    # profit_lock / edge_closed — allow normal re-entry.
                    seen_by_model.get(pos["model"], set()).discard(pos["contract_id"])
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
                if real_trade_out is not None:
                    _append_csv(real_trade_out, pos, TRADE_FIELDS)
                    _update_real_ledger()
                _log_settle(pos)
                n = pos.get("n_contracts_filled") or 0
                session_pnl += float(pos.get("pnl") or 0) * n
                ledger_stats = _update_performance_ledger(sim_root)
                # Contract is expired — no re-entry possible, but discard for consistency.
                seen_by_model.get(pos["model"], set()).discard(pos["contract_id"])
            else:
                still_open.append(pos)
        open_positions = still_open

        # ── 3a. Process delayed sim entries from previous poll ────────────────
        # For simulation-only (trader=None): contracts queued in pending_by_model
        # are entered NOW at the CURRENT ask — one poll after the edge was first seen.
        # This mimics the one-poll IOC retry lag in live trading.
        if trader is None:
            for _model_name, _pending in list(pending_by_model.items()):
                _seen_set = seen_by_model.get(_model_name, set())
                for _cid, _pdata in list(_pending.items()):
                    _lat = latest.get(_cid)
                    if _lat is None:
                        _pending.pop(_cid, None)
                        continue
                    _side  = _pdata["side"]
                    _p_sid = _pdata["p_side"]
                    _cur_ask = _safe_float(_lat.get("ask_yes" if _side == "YES" else "ask_no"))
                    if _cur_ask is None:
                        _pending.pop(_cid, None)
                        continue
                    _cur_edge = round(_p_sid - _cur_ask, 4)
                    if _cur_edge <= min_edge:
                        # Edge gone at execution — cancel, allow re-detection
                        _pending.pop(_cid, None)
                        continue
                    # ── Sim friction: IOC zero-fill + entry slippage ─────────
                    # zero_fill_prob: models orders that return unfilled because
                    # the market moved between signal detection and submission.
                    # Re-adds to pending on next poll so re-detection can occur.
                    if _rng.random() < zero_fill_prob:
                        _pending.pop(_cid, None)
                        continue  # silently cancelled — allow re-evaluation
                    # entry_slip: models stale CSV ask + IOC execution friction.
                    # Adds a random [0, entry_slip_max] cost to the execution price.
                    if entry_slip_max > 0:
                        _slip = float(_rng.uniform(0, entry_slip_max))
                        _cur_ask = min(round(_cur_ask + _slip, 4), 0.99)
                        _cur_edge = round(_p_sid - _cur_ask, 4)
                        if _cur_edge <= min_edge:
                            _pending.pop(_cid, None)
                            continue  # slippage killed the edge — cancel
                    # Enter at current (next-poll) ask price — realistic entry
                    _pos = _create_position(
                        _pdata["row"], _model_name, _side, _cur_ask,
                        _p_sid, _cur_edge, _pdata["market_mid"], _pdata["edges"]
                    )
                    _seen_set.add(_cid)
                    open_positions.append(_pos)
                    _log_enter(_pos)
                    _pending.pop(_cid, None)

        # ── 3b. Entry decisions for new contracts ─────────────────────────────
        # Iterate the *latest* row per contract — not the full history.
        # Per-poll cache: at most one HTTP call per contract_id across all 14
        # models. Without caching, every (contract × model) that finds edge
        # fires a separate full event-list request — up to 14 calls per contract.
        _entry_quote_cache: dict[str, dict | None] = {}

        # Using latest.values() ensures we always price against the freshest
        # bid/ask quote; iterating all rows would enter on the oldest quote
        # for any contract first seen at startup.
        for row in latest.values():
            contract_id = row.get("contract_id", "")
            if not contract_id:
                continue

            asset     = row.get("asset", "")
            minute_df = minute_dfs.get(asset)
            if minute_df is None:
                continue

            spot = live_spots.get(asset)
            if spot is None:
                continue  # live spot unavailable for this asset (API + OHLCV both failed)

            ask_yes = _safe_float(row.get("ask_yes"))
            ask_no  = _safe_float(row.get("ask_no"))
            try:
                strike          = float(row["strike"])
                direction       = row["direction"]
                settle_time_utc = datetime.fromisoformat(row["settle_time_utc"])
                hours_to_settle = float(row["hours_to_settle"])
            except (KeyError, ValueError):
                continue

            # Skip contracts that have already settled, are too far out, or too close
            # to settlement (near-expiry contracts are illiquid — no sellers).
            current_hrs = (settle_time_utc - now_utc).total_seconds() / 3600.0
            if current_hrs < min_hours_to_settle or current_hrs > max_hours_to_settle:
                continue

            # need_X is False for inactive models so they don't prevent the outer
            # skip guard from firing and don't force _compute_edges on every poll.
            _am = active_models
            need_gbm         = (contract_id not in seen_gbm)         and (_am is None or "gbm"            in _am)
            need_ewma        = (contract_id not in seen_ewma)        and (_am is None or "ewma"           in _am)
            need_garch       = (contract_id not in seen_garch)       and (_am is None or "garch"          in _am)
            need_stud        = (contract_id not in seen_stud)        and (_am is None or "student_t"      in _am)
            need_skt         = (contract_id not in seen_skt)         and (_am is None or "skewed_t"       in _am)
            need_heston      = (contract_id not in seen_heston)      and (_am is None or "heston"         in _am)
            need_hybrid      = (contract_id not in seen_hybrid)      and (_am is None or "hybrid_t"       in _am)
            need_ou          = (contract_id not in seen_ou)          and (_am is None or "ou"             in _am)
            need_heston_ewma = (contract_id not in seen_heston_ewma) and (_am is None or "heston_ewma"    in _am)
            need_gbm_jump    = (contract_id not in seen_gbm_jump)    and (_am is None or "gbm_jump"       in _am)
            need_ewma_jump   = (contract_id not in seen_ewma_jump)   and (_am is None or "ewma_jump"      in _am)
            need_garch_jump  = (contract_id not in seen_garch_jump)  and (_am is None or "garch_jump"     in _am)
            need_stud_jump   = (contract_id not in seen_stud_jump)   and (_am is None or "student_t_jump" in _am)
            need_hybrid_jump = (contract_id not in seen_hybrid_jump) and (_am is None or "hybrid_t_jump"  in _am)
            # Ensemble always runs when conf_active models have data (not gated by active_models)
            need_ensemble    = contract_id not in seen_ensemble
            need_edge        = contract_id not in edge_seen  # edge log independent of model filter
            if not (need_gbm or need_ewma or need_garch or need_stud or need_skt or need_heston
                    or need_hybrid or need_ou or need_heston_ewma
                    or need_gbm_jump or need_ewma_jump or need_garch_jump
                    or need_stud_jump or need_hybrid_jump or need_ensemble or need_edge):
                continue

            edges = _compute_edges(
                contract_id     = contract_id,
                spot            = spot,
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
                vol_veto_mult   = vol_veto_mult,
            )

            # ── Ensemble confidence: gather p_fairs from conf_active models ──────
            # Done once per contract; shared by all model iterations below.
            # _active_pfairs_yes: p_fair(YES) from each active conf model
            # _n_agree_yes/no:    how many active models favour each side
            _active_pfairs_yes: list[float] = []
            _n_agree_yes = 0
            _n_agree_no  = 0
            if edges is not None:
                _eff_conf_models = conf_active if conf_active else list(_MODEL_TO_PKEY.keys())
                for _cmn in _eff_conf_models:
                    _cpk = _MODEL_TO_PKEY.get(_cmn)
                    if _cpk is None:
                        continue
                    _pf = edges.get(f"p_fair_{_cpk}")
                    if _pf is None:
                        continue
                    _active_pfairs_yes.append(float(_pf))
                    _ey = edges.get(f"edge_yes_{_cpk}")
                    _en = edges.get(f"edge_no_{_cpk}")
                    if _ey is not None and _en is not None:
                        if _ey >= _en:
                            _n_agree_yes += 1
                        else:
                            _n_agree_no += 1
                    elif _ey is not None and _ey > 0:
                        _n_agree_yes += 1
                    elif _en is not None and _en > 0:
                        _n_agree_no += 1
            _n_conf_active = len(_active_pfairs_yes)
            # NO-side p_fairs are 1 − YES-side p_fairs (same distribution, flipped)
            _active_pfairs_no = [1.0 - p for p in _active_pfairs_yes]

            # ── Patch ensemble p_fair into edges so the model loop can treat
            # "ensemble" like any other model (p_key = "ensemble").
            # p_fair_ensemble = mean of conf_active model p_fairs.
            # data_conf_ensemble = mean of their data_conf values.
            if edges is not None and _n_conf_active >= 2:
                _pf_ens = float(np.mean(_active_pfairs_yes))
                _eff_conf = conf_active if conf_active else list(_MODEL_TO_PKEY.keys())
                _dc_vals = [
                    float(edges.get(f"data_conf_{_MODEL_TO_PKEY[mn]}", 1.0))
                    for mn in _eff_conf
                    if _MODEL_TO_PKEY.get(mn) and edges.get(f"p_fair_{_MODEL_TO_PKEY[mn]}") is not None
                ]
                edges["p_fair_ensemble"]    = round(_pf_ens, 4)
                edges["edge_yes_ensemble"]  = round(_pf_ens - ask_yes, 4) if ask_yes is not None else None
                edges["edge_no_ensemble"]   = round((1.0 - _pf_ens) - ask_no, 4) if ask_no is not None else None
                edges["data_conf_ensemble"] = float(np.mean(_dc_vals)) if _dc_vals else 1.0

            if need_edge:  # log edge signals once per contract (all models in one row)
                edge_seen.add(contract_id)
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
                    # Hybrid EWMA-σ + Student's t
                    "nu_hybrid":          edges.get("nu_hybrid")       if edges else None,
                    "sigma_annual_hybrid":edges.get("sigma_annual_hybrid") if edges else None,
                    "p_fair_hybrid":      edges["p_fair_hybrid"]       if edges else None,
                    "edge_yes_hybrid":    edges.get("edge_yes_hybrid") if edges else None,
                    "edge_no_hybrid":     edges.get("edge_no_hybrid")  if edges else None,
                    # OU
                    "kappa_ou":           edges.get("kappa_ou")            if edges else None,
                    "sigma_annual_ou":    edges.get("sigma_annual_ou")     if edges else None,
                    "p_fair_ou":          edges.get("p_fair_ou")           if edges else None,
                    "edge_yes_ou":        edges.get("edge_yes_ou")         if edges else None,
                    "edge_no_ou":         edges.get("edge_no_ou")          if edges else None,
                    # Heston EWMA
                    "v0_heston_ewma":     edges.get("v0_heston_ewma")      if edges else None,
                    "theta_heston_ewma":  edges.get("theta_heston_ewma")   if edges else None,
                    "p_fair_heston_ewma": edges.get("p_fair_heston_ewma")  if edges else None,
                    "edge_yes_heston_ewma":edges.get("edge_yes_heston_ewma") if edges else None,
                    "edge_no_heston_ewma": edges.get("edge_no_heston_ewma")  if edges else None,
                    # Jump diagnostics
                    "lambda_j": edges.get("lambda_j") if edges else None,
                    "mu_j":     edges.get("mu_j")     if edges else None,
                    "sigma_j":  edges.get("sigma_j")  if edges else None,
                    # Jump models
                    "p_fair_gbm_jump":    edges.get("p_fair_gbm_jump")    if edges else None,
                    "edge_yes_gbm_jump":  edges.get("edge_yes_gbm_jump")  if edges else None,
                    "edge_no_gbm_jump":   edges.get("edge_no_gbm_jump")   if edges else None,
                    "p_fair_ewma_jump":   edges.get("p_fair_ewma_jump")   if edges else None,
                    "edge_yes_ewma_jump": edges.get("edge_yes_ewma_jump") if edges else None,
                    "edge_no_ewma_jump":  edges.get("edge_no_ewma_jump")  if edges else None,
                    "p_fair_garch_jump":  edges.get("p_fair_garch_jump")  if edges else None,
                    "edge_yes_garch_jump":edges.get("edge_yes_garch_jump")if edges else None,
                    "edge_no_garch_jump": edges.get("edge_no_garch_jump") if edges else None,
                    "p_fair_stud_jump":   edges.get("p_fair_stud_jump")   if edges else None,
                    "edge_yes_stud_jump": edges.get("edge_yes_stud_jump") if edges else None,
                    "edge_no_stud_jump":  edges.get("edge_no_stud_jump")  if edges else None,
                    "p_fair_hybrid_jump": edges.get("p_fair_hybrid_jump") if edges else None,
                    "edge_yes_hybrid_jump":edges.get("edge_yes_hybrid_jump") if edges else None,
                    "edge_no_hybrid_jump": edges.get("edge_no_hybrid_jump")  if edges else None,
                }, EDGE_FIELDS)

            for model_name, p_key, seen_set, needed in [
                ("gbm",           "gbm",        seen_gbm,         need_gbm),
                ("ewma",          "ewma",       seen_ewma,        need_ewma),
                ("garch",         "garch",      seen_garch,       need_garch),
                ("student_t",     "stud",       seen_stud,        need_stud),
                ("skewed_t",      "skt",        seen_skt,         need_skt),
                ("heston",        "heston",     seen_heston,      need_heston),
                ("hybrid_t",      "hybrid",     seen_hybrid,      need_hybrid),
                ("ou",            "ou",         seen_ou,          need_ou),
                ("heston_ewma",   "heston_ewma",seen_heston_ewma, need_heston_ewma),
                ("gbm_jump",      "gbm_jump",   seen_gbm_jump,    need_gbm_jump),
                ("ewma_jump",     "ewma_jump",  seen_ewma_jump,   need_ewma_jump),
                ("garch_jump",    "garch_jump", seen_garch_jump,  need_garch_jump),
                ("student_t_jump","stud_jump",  seen_stud_jump,   need_stud_jump),
                ("hybrid_t_jump", "hybrid_jump",seen_hybrid_jump, need_hybrid_jump),
                # Ensemble: mean p_fair of conf_active models, confidence-adjusted.
                # p_fair_ensemble / edge_*_ensemble patched into edges dict above.
                ("ensemble",      "ensemble",   seen_ensemble,    need_ensemble),
            ]:
                if not needed:
                    continue  # already evaluated for this model
                if model_name != "ensemble" and active_models is not None and model_name not in active_models:
                    continue  # model not selected for this run (ensemble always allowed)
                if edges is None:
                    continue  # OHLCV stale or insufficient — retry next poll
                # Ensemble requires at least 2 conf_active models with data
                if model_name == "ensemble" and _n_conf_active < 2:
                    continue

                # NOTE: seen_set.add() is intentionally deferred until after a
                # confirmed entry below.  Adding here would permanently blacklist
                # contracts where no edge is found or where the live quote shows
                # the edge has closed — preventing re-evaluation if the market
                # reprices back into edge territory later in the session.

                p_fair   = edges[f"p_fair_{p_key}"]
                edge_yes = edges.get(f"edge_yes_{p_key}")
                edge_no  = edges.get(f"edge_no_{p_key}")

                side = edge_val = ask_price = p_side = None

                # After a stop_loss or p_drop exit on this contract+model combo, require
                # 2× the normal edge before re-entering (soft re-entry gate).
                pen_key   = f"{model_name}:{contract_id}"
                entry_thr = min_edge * 2.0 if pen_key in reentry_penalised else min_edge
                # Near-expiry scaling: require proportionally more edge as T → 0.
                # At ≥30 min: scale=1.0; at 15 min: scale=1.5; at 0 min: scale=2.0.
                if current_hrs < 0.5:
                    entry_thr *= 1.0 + (0.5 - current_hrs) / 0.5

                if (edge_yes is not None and edge_yes > entry_thr and ask_yes is not None and
                        (edge_no is None or edge_yes >= edge_no)):
                    side, edge_val, ask_price, p_side = "YES", edge_yes, ask_yes, p_fair
                elif edge_no is not None and edge_no > entry_thr and ask_no is not None:
                    side, edge_val, ask_price, p_side = "NO", edge_no, ask_no, 1.0 - p_fair

                if side is None:
                    continue  # no edge — do NOT add to seen_set; re-evaluate next poll

                # ── Live quote: re-validate edge at true execution price ──────
                # The CSV ask may be up to 60s stale. Fetch the live order book
                # now; if the ask has moved and edge is gone, skip the trade.
                # Cache result: all models on the same contract share one HTTP call.
                if contract_id not in _entry_quote_cache:
                    _entry_quote_cache[contract_id] = _fetch_live_contract_quote(
                        session, row.get("event_ticker", ""), contract_id
                    )
                live_q = _entry_quote_cache[contract_id]
                if live_q is not None:
                    fresh_ask = live_q["ask_yes"] if side == "YES" else live_q["ask_no"]
                    if fresh_ask is not None:
                        fresh_edge = round(p_side - fresh_ask, 4)
                        if fresh_edge <= min_edge:
                            continue  # edge gone — do NOT add to seen_set; re-evaluate next poll
                        ask_price = fresh_ask
                        edge_val  = fresh_edge
                elif trader is not None:
                    # Live trading mode: skip if we can't confirm the execution price.
                    # Simulation mode (trader is None): fall through using CSV ask.
                    continue

                # ── Confidence-adjusted edge (ensemble model only) ─────────────
                # Individual models use their raw edges unchanged.
                # Only the "ensemble" model applies total_conf scaling:
                #   total_conf = pred_conf^λ1 × data_conf^λ2 × ens_conf^λ3
                # edge_val for ensemble already equals mean_p_fair − ask (raw);
                # multiplying by total_conf shrinks it toward 0 based on
                # model agreement and data quality.
                if model_name == "ensemble" and _n_conf_active >= 2:
                    _pfairs = _active_pfairs_yes if side == "YES" else _active_pfairs_no
                    _pred_conf = max(0.0, 1.0 - conf_k * float(np.std(_pfairs)))
                    _data_conf = float(edges.get("data_conf_ensemble", 1.0))
                    _n_agree   = _n_agree_yes if side == "YES" else _n_agree_no
                    _ens_conf  = 0.5 + 0.5 * (_n_agree / _n_conf_active)
                    _total_conf = (
                        (_pred_conf ** conf_pred_w if conf_pred_w > 0 else 1.0)
                        * (_data_conf ** conf_data_w if conf_data_w > 0 else 1.0)
                        * (_ens_conf  ** conf_ens_w  if conf_ens_w  > 0 else 1.0)
                    )
                    edge_val = round(edge_val * _total_conf, 4)
                    if edge_val <= min_edge:
                        continue  # confidence too low — allow re-evaluation

                # Market mid for the side we're trading
                mid_yes = _safe_float(row.get("mid_yes"))
                market_mid = mid_yes if side == "YES" else \
                             (round(1.0 - mid_yes, 4) if mid_yes is not None else None)

                # ── Simulation entry delay ─────────────────────────────────────
                # In simulation (trader=None): on the FIRST poll where edge is found,
                # place the contract in `pending_by_model` instead of entering
                # immediately.  On the NEXT poll it is picked up and entered at
                # that poll's current ask — mimicking the one-poll IOC retry lag
                # that live trading incurs.  Live trading (trader is not None) uses
                # the real IOC mechanism instead and skips this path.
                if trader is None and contract_id not in pending_by_model.get(model_name, {}):
                    pending_by_model.setdefault(model_name, {})[contract_id] = {
                        "row": row, "side": side, "p_side": p_side,
                        "market_mid": market_mid, "edges": edges,
                    }
                    # Will be processed from pending_by_model at start of next poll
                    continue

                pos = _create_position(row, model_name, side, ask_price, p_side,
                                       edge_val, market_mid, edges)

                # ── Live order placement (only when trader is configured) ──────
                # IMPORTANT: only add to open_positions AFTER a confirmed fill.
                # A failed or zero-fill BUY means we hold nothing at the exchange.
                if trader is not None:
                    try:
                        # Submit 3 cents above quoted ask to absorb staleness.
                        # IOC fills at actual market price (≤ limit), not the
                        # inflated limit, so edge calculation remains valid.
                        buffered_limit = min(round(ask_price + 0.04, 4), 0.99)
                        order = trader.place_order(
                            contract_id = contract_id,
                            side        = side,
                            ask_price   = ask_price,
                            limit_price = buffered_limit,
                        )
                        if order["filled"] == 0:
                            # IOC with zero fill — no real position, allow retry next poll
                            print(
                                f"    [BUY]  IOC filled 0 contracts "
                                f"(order_id={order['order_id']}) — skipping"
                            )
                            seen_set.discard(contract_id)
                            continue
                        pos["gemini_order_id"]      = str(order.get("order_id") or "")
                        pos["n_contracts_filled"]   = order["filled"]
                        pos["n_contracts_original"] = order["filled"]  # immutable reference count
                        # Update entry basis to actual fill price if the API returns it.
                        # This ensures profit_lock / stop_loss thresholds are anchored
                        # to real executed cost, not the quoted ask.
                        if order.get("avg_price") is not None:
                            pos["ask_price"] = str(round(order["avg_price"], 4))
                        print(
                            f"    [BUY]  order_id={order['order_id']}  "
                            f"requested={order['n_contracts']}  "
                            f"filled={order['filled']}  "
                            f"fill_price={pos['ask_price']}  status={order['status']}"
                        )
                    except Exception as exc:
                        # Order failed entirely — discard so we can retry next poll
                        print(f"    [BUY]  FAILED: {exc} — position not opened")
                        seen_set.discard(contract_id)
                        continue

                # Mark as seen only after a confirmed entry so that contracts
                # with no edge (or edge gone by execution) stay eligible for
                # re-evaluation on future polls when the market reprices.
                seen_set.add(contract_id)
                open_positions.append(pos)
                _log_enter(pos)

        _print_summary(now_utc, open_positions, len(closed_positions), ledger_stats, session_pnl)

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
    _s   = cfg.section("simulation")
    _ex  = cfg.section("simulation.exit")
    _lb  = cfg.section("simulation.lookback")
    _cf  = cfg.section("simulation.confidence")
    # active_models for confidence: list from TOML or None (all models)
    _cf_active_default: list[str] | None = _cf.get("active_models", None)

    # Per-model lookback defaults from config
    _lb_defaults = {
        "gbm":         _lb.get("gbm",         24.0),
        "ewma":        _lb.get("ewma",        48.0),
        "garch":       _lb.get("garch",       72.0),
        "stud":        _lb.get("stud",        48.0),
        "skt":         _lb.get("skt",         48.0),
        "heston":      _lb.get("heston",      96.0),
        "hybrid":      _lb.get("hybrid",      48.0),
        "ou":          _lb.get("ou",          12.0),
        "heston_ewma": _lb.get("heston_ewma", 96.0),
        "gbm_jump":    _lb.get("gbm_jump",    24.0),
        "ewma_jump":   _lb.get("ewma_jump",   48.0),
        "garch_jump":  _lb.get("garch_jump",  72.0),
        "student_t_jump": _lb.get("student_t_jump", 48.0),
        "hybrid_t_jump":  _lb.get("hybrid_t_jump",  48.0),
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
    parser.add_argument("--min-hours-to-settle", type=float, default=_s.get("min_hours_to_settle", 0.0),
                        help="Skip contracts settling less than this many hours away (avoids near-expiry illiquidity)")
    parser.add_argument("--profit-lock",         type=float, default=_ex.get("profit_lock",        0.05),
                        help="Early exit when bid_now - ask_entry >= this (skipped within 30 min of settlement)")
    parser.add_argument("--stop-loss",           type=float, default=_ex.get("stop_loss",          0.50),
                        help="Early exit when (ask_entry - bid_now) / ask_entry >= this fraction (e.g. 0.50 = 50%% drawdown)")
    parser.add_argument("--p-drop",              type=float, default=_ex.get("p_drop",             0.05),
                        help="Early exit when model p(side) drops by this from entry")
    parser.add_argument("--edge-neg-thresh",     type=float, default=_ex.get("edge_neg_thresh",    0.02),
                        help="edge_closed only fires when edge <= -this (hysteresis; 0 = any negative edge)")
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
    parser.add_argument("--lb-hybrid", type=float, default=_lb_defaults["hybrid"],
                        help="Hybrid EWMA-t calibration lookback (hours)")
    parser.add_argument("--lb-ou", type=float, default=_lb_defaults["ou"],
                        help="OU log-price mean-reversion lookback (hours)")
    parser.add_argument("--lb-heston-ewma", type=float, default=_lb_defaults["heston_ewma"],
                        help="Heston-EWMA calibration lookback (hours)")
    parser.add_argument("--lb-gbm-jump",      type=float, default=_lb_defaults["gbm_jump"],
                        help="GBM+Jump calibration lookback (hours)")
    parser.add_argument("--lb-ewma-jump",     type=float, default=_lb_defaults["ewma_jump"],
                        help="EWMA+Jump calibration lookback (hours)")
    parser.add_argument("--lb-garch-jump",    type=float, default=_lb_defaults["garch_jump"],
                        help="GARCH+Jump calibration lookback (hours)")
    parser.add_argument("--lb-stud-jump",     type=float, default=_lb_defaults["student_t_jump"],
                        help="StudentT+Jump calibration lookback (hours)")
    parser.add_argument("--lb-hybrid-jump",   type=float, default=_lb_defaults["hybrid_t_jump"],
                        help="HybridT+Jump calibration lookback (hours)")
    parser.add_argument("--vol-veto-mult",    type=float, default=_s.get("vol_veto_mult", 2.0),
                        help="Block entry when 10-min realized vol > this × EWMA vol (0=off)")
    parser.add_argument("--no-collectors", action="store_true",
                        help="Do not auto-start getdata_underlying / getdata_prediction_contract")
    parser.add_argument("--trades-dir", type=str, default=None,
                        help="Output directory for trades/ledger CSV files "
                             "(default: .data/gemini/sim_trades). Use a different "
                             "path to run a parallel experiment without mixing results.")
    # Confidence system
    parser.add_argument("--conf-pred-w",  type=float, default=_cf.get("pred_conf_weight",     0.55),
                        help="λ1 weight for pred_conf (model disagreement) in geometric mean")
    parser.add_argument("--conf-data-w",  type=float, default=_cf.get("data_conf_weight",     0.10),
                        help="λ2 weight for data_conf (OHLCV gap ratio) in geometric mean")
    parser.add_argument("--conf-ens-w",   type=float, default=_cf.get("ensemble_conf_weight", 0.35),
                        help="λ3 weight for ensemble_conf (directional agreement) in geometric mean")
    parser.add_argument("--conf-k",       type=float, default=_cf.get("pred_conf_k",          3.0),
                        help="Harshness of pred_conf: conf = max(0, 1 - k × std(p_fairs))")
    parser.add_argument("--conf-active",  type=str,   default=None,
                        help="Comma-separated model names for confidence computation "
                             "(default: from config.toml active_models). "
                             "Example: heston_ewma,garch,student_t")
    # Sim friction (slippage / IOC zero-fill)
    _sl = cfg.section("simulation.slippage")
    parser.add_argument("--zero-fill-prob",  type=float, default=_sl.get("zero_fill_prob", 0.15),
                        help="Sim-only: probability of simulating an IOC zero-fill at entry")
    parser.add_argument("--entry-slip-max",  type=float, default=_sl.get("entry_slip_max", 0.02),
                        help="Sim-only: max extra ¢ added to entry ask (uniform [0, max])")
    parser.add_argument("--exit-slip-max",   type=float, default=_sl.get("exit_slip_max",  0.01),
                        help="Sim-only: max ¢ subtracted from exit bid at early exit (uniform [0, max])")
    args = parser.parse_args()

    run(
        poll_sec            = args.poll_sec,
        lookbacks           = {
            "gbm":           args.lb_gbm,
            "ewma":          args.lb_ewma,
            "garch":         args.lb_garch,
            "stud":          args.lb_stud,
            "skt":           args.lb_skt,
            "heston":        args.lb_heston,
            "hybrid":        args.lb_hybrid,
            "ou":            args.lb_ou,
            "heston_ewma":   args.lb_heston_ewma,
            "gbm_jump":      args.lb_gbm_jump,
            "ewma_jump":     args.lb_ewma_jump,
            "garch_jump":    args.lb_garch_jump,
            "student_t_jump":args.lb_stud_jump,
            "hybrid_t_jump": args.lb_hybrid_jump,
        },
        min_edge            = args.min_edge,
        max_hours_to_settle = args.max_hours_to_settle,
        min_hours_to_settle = args.min_hours_to_settle,
        profit_lock         = args.profit_lock,
        stop_loss           = args.stop_loss,
        p_drop              = args.p_drop,
        ewma_lambda         = args.ewma_lambda,
        rho                 = args.rho,
        edge_neg_thresh     = args.edge_neg_thresh,
        vol_veto_mult       = args.vol_veto_mult,
        no_collectors       = args.no_collectors,
        sim_root            = Path(args.trades_dir) if args.trades_dir else SIM_ROOT,
        conf_pred_w         = args.conf_pred_w,
        conf_data_w         = args.conf_data_w,
        conf_ens_w          = args.conf_ens_w,
        conf_k              = args.conf_k,
        conf_active         = [m.strip() for m in args.conf_active.split(",") if m.strip()]
                              if args.conf_active else _cf_active_default,
        zero_fill_prob      = args.zero_fill_prob,
        entry_slip_max      = args.entry_slip_max,
        exit_slip_max       = args.exit_slip_max,
    )
