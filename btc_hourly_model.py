"""Hourly BTC/ETH/SOL contract modeling with GBM and Student's t fat-tail returns.

This module provides calibration and fair-value functions consumed by the live
trading simulation (live_trading_sim.py) and the data collector
(getdata_prediction_contract.py).

Model family 1 (traditional SDE / GBM):
    dS_t / S_t = mu dt + sigma dW_t

Model family 2 (fat-tail extension):
    log-return ~ Student's t with calibrated degrees of freedom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from math import erf, sqrt
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd


Direction = Literal["above", "below"]


@dataclass
class GBMParams:
    """Parameters for GBM over one hour increments.

    Attributes:
        mu_per_hour: Drift term in `dS/S = mu dt + sigma dW`, with dt in hours.
        sigma_per_sqrt_hour: Diffusion term in per-sqrt-hour units.
    """

    mu_per_hour: float
    sigma_per_sqrt_hour: float


@dataclass
class StudentTParams:
    """Parameters for Student's t model of 1-hour log returns.

    We model 1-hour log return r as:
        r = loc + scale * z,
    where z has unit variance and Student's t tails.
    """

    loc: float
    scale: float
    nu: float


def load_gemini_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load Gemini `.data` minute OHLCV. Index is the EST-aware timestamp."""
    df = pd.read_csv(path, parse_dates=["timestamp_est"])
    required = {"timestamp_est", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.sort_values("timestamp_est").set_index("timestamp_est")[["open", "high", "low", "close", "volume"]]


def calibrate_gbm_from_log_returns(log_returns: pd.Series | np.ndarray, dt_hours: float = 1.0) -> GBMParams:
    """Estimate GBM (mu, sigma) from log returns over fixed dt.

    For GBM:
        r = log(S_{t+dt}/S_t) ~ Normal((mu - 0.5*sigma^2)*dt, sigma^2*dt)

    If sample mean is m and variance is v:
        sigma = sqrt(v/dt)
        mu = m/dt + 0.5*sigma^2
    """
    r = np.asarray(log_returns, dtype=float)
    if r.size < 2:
        raise ValueError("Need at least 2 returns to calibrate GBM")

    m = float(np.mean(r))
    v = float(np.var(r, ddof=1))
    sigma = sqrt(max(v / dt_hours, 1e-18))
    mu = m / dt_hours + 0.5 * sigma * sigma
    return GBMParams(mu_per_hour=mu, sigma_per_sqrt_hour=sigma)


def _sample_excess_kurtosis(x: np.ndarray) -> float:
    """Return sample excess kurtosis (moment estimator, not bias-corrected)."""
    m = float(np.mean(x))
    c = x - m
    m2 = float(np.mean(c * c))
    if m2 <= 0:
        return 0.0
    m4 = float(np.mean(c**4))
    return m4 / (m2 * m2) - 3.0


def calibrate_student_t_from_log_returns(log_returns: pd.Series | np.ndarray) -> StudentTParams:
    """Estimate Student's t parameters for 1-hour log returns.

    Strategy:
    - `loc` = sample mean.
    - `scale` = sample std.
    - `nu` from excess kurtosis relation for Student's t:
          excess_kurtosis = 6 / (nu - 4), nu > 4.
      So nu = 6/g2 + 4 when g2 > 0, else large nu (near normal).
    """
    r = np.asarray(log_returns, dtype=float)
    if r.size < 5:
        raise ValueError("Need at least 5 returns to calibrate Student's t")

    loc = float(np.mean(r))
    scale = float(np.std(r, ddof=1))
    g2 = _sample_excess_kurtosis(r)

    if g2 > 1e-6:
        nu = 6.0 / g2 + 4.0
        nu = float(np.clip(nu, 4.1, 200.0))
    else:
        nu = 200.0

    return StudentTParams(loc=loc, scale=max(scale, 1e-9), nu=nu)


def _normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF without scipy."""
    return 0.5 * (1.0 + erf(np.asarray(x) / sqrt(2.0)))


def gbm_binary_prob(
    trade_price: float,
    strike_price: float,
    params: GBMParams,
    horizon_hours: float = 1.0,
    direction: Direction = "above",
) -> float:
    """Price binary contract under GBM: P(S_T > K) or P(S_T < K)."""
    if trade_price <= 0 or strike_price <= 0:
        raise ValueError("Prices must be positive")

    mu = params.mu_per_hour
    sigma = max(params.sigma_per_sqrt_hour, 1e-12)
    t = horizon_hours
    mean_lr = (mu - 0.5 * sigma * sigma) * t
    std_lr = sigma * sqrt(t)

    x = (np.log(strike_price / trade_price) - mean_lr) / std_lr
    p_above = float(1.0 - _normal_cdf(x))
    return p_above if direction == "above" else float(1.0 - p_above)


def student_t_binary_prob(
    trade_price: float,
    strike_price: float,
    params: StudentTParams,
    direction: Direction = "above",
    n_sims: int = 20_000,
    seed: Optional[int] = 7,
) -> float:
    """Price binary contract via Monte Carlo under Student's t log-return model."""
    if trade_price <= 0 or strike_price <= 0:
        raise ValueError("Prices must be positive")

    rng = np.random.default_rng(seed)
    # np.random.standard_t has variance nu/(nu-2). We rescale to unit variance.
    raw = rng.standard_t(df=params.nu, size=n_sims)
    z = raw * sqrt((params.nu - 2.0) / params.nu)

    lr = params.loc + params.scale * z
    terminal = trade_price * np.exp(lr)
    p_above = float(np.mean(terminal > strike_price))
    return p_above if direction == "above" else float(1.0 - p_above)


def _minute_log_returns_before(
    minute_df: pd.DataFrame,
    before_time,
    lookback_hours: float,
) -> np.ndarray:
    """1-minute log-returns in the window (before_time - lookback_hours, before_time)."""
    start = before_time - pd.Timedelta(hours=lookback_hours)
    closes = minute_df.loc[
        (minute_df.index > start) & (minute_df.index < before_time), "close"
    ].dropna()
    if len(closes) < 2:
        return np.array([])
    return np.log(closes.values[1:] / closes.values[:-1])


# ---------------------------------------------------------------------------
# Event-based fair value (Gemini prediction market — multi-hour/day contracts)
# ---------------------------------------------------------------------------

_EVENT_TICKER_RE = re.compile(
    r"^(?P<asset>BTC|ETH|SOL)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<hh>\d{2})(?P<mn>\d{2})$"
)


def parse_event_ticker(ticker: str) -> tuple[str, datetime] | tuple[None, None]:
    """Parse a Gemini prediction market ticker into (asset, settle_time_utc).

    Ticker format: {ASSET}{YY}{MM}{DD}{HH}{MN}
    Example: BTC2603022300 → ('BTC', datetime(2026,3,2,23,0, tzinfo=UTC))
    Returns (None, None) if the ticker doesn't match.
    """
    m = _EVENT_TICKER_RE.match(ticker)
    if not m:
        return None, None
    g = m.groupdict()
    settle = datetime(
        2000 + int(g["yy"]), int(g["mm"]), int(g["dd"]),
        int(g["hh"]), int(g["mn"]), tzinfo=timezone.utc,
    )
    return g["asset"], settle


def compute_event_fair_value(
    spot: float,
    strike: float,
    settle_time_utc: datetime,
    minute_df: pd.DataFrame,
    eval_time: pd.Timestamp | None = None,
    lookback_hours: float = 48.0,
    direction: Direction = "above",
) -> dict | None:
    """GBM fair value for a live Gemini prediction market contract.

    Can be called at any point during the contract's life, not just at entry.

    Parameters
    ----------
    spot            : Current underlying price S_t.
    strike          : Contract strike K.
    settle_time_utc : Contract settlement datetime (UTC-aware).
    minute_df       : 1-minute OHLCV with EST-aware DatetimeIndex.
    eval_time       : Evaluation timestamp (defaults to latest bar in minute_df).
    lookback_hours  : Window for vol calibration (no look-ahead; only bars < eval_time).
    direction       : "above" → P(S_T > K); "below" → P(S_T < K).

    Returns dict with fair value and diagnostics, or None if insufficient data.
    """
    if eval_time is None:
        eval_time = minute_df.index[-1]

    # Convert settle_time to same tz as minute_df index
    settle_est = pd.Timestamp(settle_time_utc).tz_convert(minute_df.index.tz)
    horizon_hours = (settle_est - eval_time).total_seconds() / 3600.0

    if horizon_hours <= 0:
        # Contract already settled
        realized = float(spot > strike) if direction == "above" else float(spot <= strike)
        return {"p_fair_gbm": realized, "horizon_hours": 0.0, "sigma_annual": None}

    min_lr = _minute_log_returns_before(minute_df, eval_time, lookback_hours)
    if len(min_lr) < 20:
        return None

    gbm = calibrate_gbm_from_log_returns(min_lr, dt_hours=1 / 60)
    p_fair = gbm_binary_prob(spot, strike, gbm, horizon_hours=horizon_hours, direction=direction)
    sigma_annual = gbm.sigma_per_sqrt_hour * sqrt(8_760)  # annualised vol

    return {
        "eval_time": eval_time,
        "settle_time_est": settle_est,
        "horizon_hours": horizon_hours,
        "S_t": spot,
        "strike": strike,
        "direction": direction,
        "sigma_annual": sigma_annual,
        "p_fair_gbm": p_fair,
    }


def compute_fair_value_path(
    minute_df: pd.DataFrame,
    strike: float,
    settle_time_utc: datetime,
    lookback_hours: float = 48.0,
    contract_life_hours: float = 24.0,
    direction: Direction = "above",
) -> pd.DataFrame:
    """Compute GBM fair value at every minute during a contract's life.

    Used for backtesting: compare p_fair_gbm column against p_market (from
    getdata_prediction_contract.py recordings) to measure and validate edge.

    Parameters
    ----------
    minute_df           : 1-minute OHLCV (EST-aware index).
    strike              : Contract strike K.
    settle_time_utc     : Settlement time (UTC-aware datetime).
    lookback_hours      : Rolling calibration window (strictly before each bar).
    contract_life_hours : How far back from settle_time the contract was listed.
    direction           : "above" or "below".

    Returns
    -------
    DataFrame with one row per minute. Key columns:
        t, horizon_hours, S_t, strike, p_fair_gbm, realized_above, sigma_annual
    """
    settle_est = pd.Timestamp(settle_time_utc).tz_convert(minute_df.index.tz)
    start_est = settle_est - pd.Timedelta(hours=contract_life_hours)

    life_bars = minute_df.loc[
        (minute_df.index >= start_est) & (minute_df.index < settle_est)
    ]
    if life_bars.empty:
        return pd.DataFrame()

    # Terminal price for realized outcome (last bar at or before settle)
    bars_at_or_before_settle = minute_df.loc[minute_df.index <= settle_est]
    sT = float(bars_at_or_before_settle["close"].iloc[-1]) if not bars_at_or_before_settle.empty else float("nan")
    realized = float(sT > strike) if direction == "above" else float(sT <= strike)

    rows = []
    for t, bar in life_bars.iterrows():
        result = compute_event_fair_value(
            spot=float(bar["close"]),
            strike=strike,
            settle_time_utc=settle_time_utc,
            minute_df=minute_df,
            eval_time=t,
            lookback_hours=lookback_hours,
            direction=direction,
        )
        if result is None:
            continue
        result["sT"] = sT
        result["realized"] = realized
        rows.append(result)

    return pd.DataFrame(rows)
