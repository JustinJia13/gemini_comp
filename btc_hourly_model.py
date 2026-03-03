"""Hourly BTC/ETH/SOL contract modeling — multiple vol and distribution models.

This module provides calibration and fair-value functions consumed by the live
trading simulation (live_trading_sim.py) and the data collector
(getdata_prediction_contract.py).

Model family 1 — GBM with rolling-window vol (baseline):
    dS_t / S_t = mu dt + sigma dW_t   (sigma = rolling std of 1-min returns)

Model family 2 — GBM with EWMA vol:
    Same GBM closed form but sigma estimated via EWMA recursion.
    Captures vol clustering without optimisation.

Model family 3 — GBM with GARCH(1,1) vol:
    sigma²_t = omega + alpha * eps²_{t-1} + beta * sigma²_{t-1}
    MLE calibrated on 1-min returns; gives the best conditional vol estimate.

Model family 4 — Symmetric Student's t (fat tails):
    log-return ~ t(nu, loc, scale); priced via Monte Carlo.

Model family 5 — Fernández-Steel skewed Student's t:
    Two-piece t with asymmetric tails; gamma < 1 gives left skew (crash risk).
    Skewness informed by price-vol correlation rho.  Priced via Monte Carlo.

Model family 6 — Heston stochastic volatility:
    dS_t / S_t = sqrt(V_t) dW₁
    dV_t        = kappa (theta - V_t) dt + xi sqrt(V_t) dW₂
    dW₁ dW₂    = rho dt
    V₀, kappa, theta, xi calibrated from 1-min returns; rho from elastic-net
    regression of Δvol on returns.  Priced via vectorised Monte Carlo
    (full-truncation Euler scheme, 1-min steps).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from math import erf, sqrt, log, pi
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


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


@dataclass
class EWMAParams:
    """EWMA conditional volatility at the current bar.

    Attributes:
        sigma_per_sqrt_hour: Current conditional vol scaled to per-sqrt-hour.
        lambda_: Decay factor used (typically 0.94, RiskMetrics standard).
        mu_per_hour: GBM-consistent drift (per hour), estimated from the
            sample mean of 1-min log returns in the calibration window:
            mu = mean(r) / dt + 0.5 * sigma^2.
    """

    sigma_per_sqrt_hour: float
    lambda_: float
    mu_per_hour: float


@dataclass
class GARCHParams:
    """GARCH(1,1) parameters + current conditional vol.

    Model: sigma²_t = omega + alpha * eps²_{t-1} + beta * sigma²_{t-1}

    Attributes:
        omega, alpha, beta: GARCH coefficients (MLE estimated).
        sigma_per_sqrt_hour: Current conditional vol (terminal value of the
            GARCH recursion), scaled to per-sqrt-hour.
        mu_per_hour: GBM-consistent drift (per hour), estimated from the
            sample mean of 1-min log returns in the calibration window:
            mu = mean(r) / dt + 0.5 * sigma^2.
    """

    omega: float
    alpha: float
    beta: float
    sigma_per_sqrt_hour: float
    mu_per_hour: float


@dataclass
class SkewedTParams:
    """Fernández-Steel two-piece skewed Student's t parameters.

    Two-piece distribution around `loc`:
      - Right tail (r >= loc): scale = scale * gamma
      - Left  tail (r <  loc): scale = scale / gamma

    gamma < 1  →  compressed right tail, heavy left tail  →  LEFT-SKEWED
    gamma = 1  →  symmetric (reduces to standard Student's t)
    gamma > 1  →  right-skewed

    For crypto, gamma is typically 0.6–0.9 (negative leverage effect).
    """

    loc: float
    scale: float
    nu: float
    gamma: float   # skewness: < 1 = left-skewed, 1 = symmetric, > 1 = right-skewed


@dataclass
class HestonParams:
    """Heston stochastic-vol model parameters (all in per-hour units).

    Model:
        dS / S  = sqrt(V) dW₁
        dV      = kappa (theta - V) dt + xi sqrt(V) dW₂
        dW₁ dW₂ = rho dt

    All variance quantities are in (log-return)²/hour.

    Attributes:
        v0    : Current instantaneous variance (per hour).
        kappa : Mean-reversion speed (per hour).  Typical range 1–30.
        theta : Long-run mean variance (per hour).
        xi    : Vol-of-vol.  Feller condition: 2·kappa·theta > xi².
        rho   : Price-vol correlation.  Typically −0.7 to −0.2 for crypto.
    """

    v0:    float
    kappa: float
    theta: float
    xi:    float
    rho:   float


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


# ---------------------------------------------------------------------------
# EWMA vol calibration + pricing
# ---------------------------------------------------------------------------

def calibrate_ewma_from_log_returns(
    log_returns: np.ndarray,
    lambda_: float = 0.94,
    dt_hours: float = 1 / 60,
) -> EWMAParams:
    """Estimate current conditional vol via EWMA recursion on 1-min returns.

    sigma²_t = lambda * sigma²_{t-1} + (1 - lambda) * r²_{t-1}

    The recursion is run forward over the full return series; the terminal
    value is the most-recent conditional variance.  lambda=0.94 is the
    J.P. Morgan RiskMetrics standard for daily data; 0.97 is common for
    intraday 1-minute data.

    Returns sigma scaled to per-sqrt-hour (same units as GBMParams).
    """
    r = np.asarray(log_returns, dtype=float)
    if r.size < 2:
        raise ValueError("Need at least 2 returns for EWMA")

    var = float(np.var(r))          # initialise with sample variance
    for ret in r:
        var = lambda_ * var + (1.0 - lambda_) * ret ** 2
    var = max(var, 1e-18)

    sigma = sqrt(var / dt_hours)
    # GBM-consistent drift: same formula as calibrate_gbm_from_log_returns
    mu_per_hour = float(np.mean(r)) / dt_hours + 0.5 * sigma * sigma

    return EWMAParams(
        sigma_per_sqrt_hour=sigma,
        lambda_=lambda_,
        mu_per_hour=mu_per_hour,
    )


def ewma_binary_prob(
    trade_price: float,
    strike_price: float,
    params: EWMAParams,
    horizon_hours: float = 1.0,
    direction: Direction = "above",
) -> float:
    """Binary contract probability under GBM with EWMA conditional vol.

    Identical closed-form to gbm_binary_prob but uses the EWMA conditional
    sigma rather than the rolling-window estimate.  Drift mu is calibrated
    from the sample mean of the return window (see EWMAParams.mu_per_hour).
    """
    if trade_price <= 0 or strike_price <= 0:
        raise ValueError("Prices must be positive")

    sigma   = max(params.sigma_per_sqrt_hour, 1e-12)
    t       = horizon_hours
    mean_lr = (params.mu_per_hour - 0.5 * sigma * sigma) * t
    std_lr  = sigma * sqrt(t)
    x       = (np.log(strike_price / trade_price) - mean_lr) / std_lr
    p_above = float(1.0 - _normal_cdf(x))
    return p_above if direction == "above" else float(1.0 - p_above)


# ---------------------------------------------------------------------------
# GARCH(1,1) vol calibration + pricing
# ---------------------------------------------------------------------------

def _garch_variance_series(
    r: np.ndarray, omega: float, alpha: float, beta: float, var_init: float
) -> np.ndarray:
    """Run GARCH(1,1) recursion; return conditional variance at each bar."""
    n   = len(r)
    var = np.empty(n)
    var[0] = var_init
    for i in range(1, n):
        var[i] = omega + alpha * r[i - 1] ** 2 + beta * var[i - 1]
        var[i] = max(var[i], 1e-18)
    return var


def calibrate_garch_from_log_returns(
    log_returns: np.ndarray,
    dt_hours: float = 1 / 60,
) -> GARCHParams:
    """Fit GARCH(1,1) via quasi-MLE on 1-min log returns.

    Optimises omega, alpha, beta subject to:
      omega > 0,  alpha >= 0,  beta >= 0,  alpha + beta < 1.

    Returns the fitted parameters AND the current (terminal) conditional vol
    scaled to per-sqrt-hour units for direct use in binary pricing.
    """
    r        = np.asarray(log_returns, dtype=float)
    if r.size < 20:
        raise ValueError("Need at least 20 returns for GARCH")

    var_init = float(np.var(r))
    long_run = var_init

    def neg_ll(params: np.ndarray) -> float:
        omega, alpha, beta = float(params[0]), float(params[1]), float(params[2])
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e12
        v  = _garch_variance_series(r, omega, alpha, beta, long_run)
        ll = -0.5 * np.sum(np.log(v) + r ** 2 / v)
        return -ll

    x0     = np.array([long_run * 0.05, 0.10, 0.85])
    bounds = [(1e-12, None), (1e-6, 0.5), (1e-6, 0.9998)]
    res    = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 100, "ftol": 1e-9})
    omega, alpha, beta = float(res.x[0]), float(res.x[1]), float(res.x[2])

    # Current conditional variance = terminal value of the recursion
    var_t = long_run
    for ret in r:
        var_t = omega + alpha * ret ** 2 + beta * var_t
        var_t = max(var_t, 1e-18)

    sigma = sqrt(var_t / dt_hours)
    # GBM-consistent drift: same formula as calibrate_gbm_from_log_returns
    mu_per_hour = float(np.mean(r)) / dt_hours + 0.5 * sigma * sigma

    return GARCHParams(
        omega=omega,
        alpha=alpha,
        beta=beta,
        sigma_per_sqrt_hour=sigma,
        mu_per_hour=mu_per_hour,
    )


def garch_binary_prob(
    trade_price: float,
    strike_price: float,
    params: GARCHParams,
    horizon_hours: float = 1.0,
    direction: Direction = "above",
) -> float:
    """Binary contract probability under GBM with GARCH(1,1) conditional vol.

    Uses the current GARCH conditional sigma in the GBM closed form.
    Drift mu is calibrated from the sample mean of the return window (see
    GARCHParams.mu_per_hour).
    """
    if trade_price <= 0 or strike_price <= 0:
        raise ValueError("Prices must be positive")

    sigma   = max(params.sigma_per_sqrt_hour, 1e-12)
    t       = horizon_hours
    mean_lr = (params.mu_per_hour - 0.5 * sigma * sigma) * t
    std_lr  = sigma * sqrt(t)
    x       = (np.log(strike_price / trade_price) - mean_lr) / std_lr
    p_above = float(1.0 - _normal_cdf(x))
    return p_above if direction == "above" else float(1.0 - p_above)


# ---------------------------------------------------------------------------
# Skewed Student's t — Fernández-Steel (1998) — calibration + pricing
# ---------------------------------------------------------------------------

def _skewed_t_logpdf(
    r: np.ndarray,
    loc: float,
    scale: float,
    nu: float,
    gamma: float,
) -> np.ndarray:
    """Vectorised Fernández-Steel skewed-t log-PDF.

    Two-piece t around loc:
      x >= loc: z = (x - loc) / (scale * gamma)   → right tail compressed/expanded by gamma
      x <  loc: z = (x - loc) * gamma / scale      → left tail expanded/compressed by 1/gamma

    Normalising constant c = 2 / (gamma + 1/gamma).
    """
    c          = 2.0 / (gamma + 1.0 / gamma)
    z_std      = (r - loc) / scale                       # unscaled standardised residual
    z_right    = z_std / gamma                           # right-piece rescaling
    z_left     = z_std * gamma                           # left-piece rescaling
    z_scaled   = np.where(r >= loc, z_right, z_left)

    log_norm   = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * log(nu * pi)
    log_kernel = -(nu + 1) / 2 * np.log(1.0 + z_scaled ** 2 / nu)
    return log(c) - log(scale) + log_norm + log_kernel


def calibrate_skewed_t_from_log_returns(
    log_returns: np.ndarray,
    rho: Optional[float] = None,
) -> SkewedTParams:
    """Fit Fernández-Steel skewed-t via MLE on 1-min log returns.

    When rho (price-vol correlation, typically negative for crypto) is
    provided, the initial gamma is anchored to exp(rho) so that negative
    rho → gamma < 1 (left skew).  MLE then refines all parameters.

    Parameters
    ----------
    log_returns : 1-minute log returns from the lookback window.
    rho         : Optional price-vol correlation prior in [-1, 0).
                  If None, gamma is freely fit from data.
    """
    r = np.asarray(log_returns, dtype=float)
    if r.size < 10:
        raise ValueError("Need at least 10 returns for skewed t")

    loc0   = float(np.mean(r))
    scale0 = float(np.std(r, ddof=1))
    g2     = _sample_excess_kurtosis(r)
    nu0    = float(np.clip(6.0 / g2 + 4.0 if g2 > 1e-6 else 30.0, 4.1, 100.0))
    # Use rho as prior for gamma initialisation; rho < 0 → gamma < 1 (left skew)
    gamma0 = float(np.clip(np.exp(rho), 0.3, 2.0)) if rho is not None else 0.85

    # Optimise in unconstrained space: log-transform scale, nu-2, gamma
    def neg_ll(p: np.ndarray) -> float:
        loc, log_sc, log_nu2, log_gam = p[0], p[1], p[2], p[3]
        scale = np.exp(log_sc)
        nu    = np.exp(log_nu2) + 2.0
        gamma = np.exp(log_gam)
        ll    = _skewed_t_logpdf(r, loc, scale, nu, gamma)
        return -float(np.sum(ll))

    x0  = np.array([loc0, log(scale0), log(nu0 - 2.0), log(gamma0)])
    res = minimize(neg_ll, x0, method="Nelder-Mead",
                   options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-6})

    loc, log_sc, log_nu2, log_gam = res.x
    return SkewedTParams(
        loc=float(loc),
        scale=float(max(np.exp(log_sc), 1e-9)),
        nu=float(np.clip(np.exp(log_nu2) + 2.0, 2.01, 200.0)),
        gamma=float(np.clip(np.exp(log_gam), 0.1, 5.0)),
    )


def skewed_t_binary_prob(
    trade_price: float,
    strike_price: float,
    params: SkewedTParams,
    direction: Direction = "above",
    n_sims: int = 20_000,
    seed: Optional[int] = 7,
) -> float:
    """Binary contract price via Monte Carlo under the skewed Student's t.

    Sampling from Fernández-Steel skewed t:
      - P(right piece) = gamma² / (1 + gamma²)
      - Right piece sample: +|Z| * gamma   (Z ~ t_nu)
      - Left  piece sample: -|Z| / gamma
    """
    if trade_price <= 0 or strike_price <= 0:
        raise ValueError("Prices must be positive")

    rng     = np.random.default_rng(seed)
    gamma   = params.gamma
    p_right = gamma ** 2 / (1.0 + gamma ** 2)

    # Half-t samples (always positive); rescale to approximately unit variance
    z = np.abs(rng.standard_t(df=params.nu, size=n_sims))

    u       = rng.uniform(size=n_sims)
    samples = np.where(u < p_right, z * gamma, -z / gamma)

    lr       = params.loc + params.scale * samples
    terminal = trade_price * np.exp(lr)
    p_above  = float(np.mean(terminal > strike_price))
    return p_above if direction == "above" else float(1.0 - p_above)


# ---------------------------------------------------------------------------
# Price-vol correlation (rho) estimation via elastic net
# ---------------------------------------------------------------------------

def estimate_price_vol_rho(
    log_returns: np.ndarray,
    window: int = 60,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
) -> float:
    """Estimate price-vol correlation via elastic net regression.

    Regresses changes in rolling realised vol on lagged 1-minute returns:
        delta_sigma_t = beta * r_{t-1} + intercept

    The normalised coefficient is the price-vol correlation rho.  For crypto
    this is typically in [-0.7, -0.2] — negative because price drops cause
    vol to spike (leverage / liquidation effect).

    Parameters
    ----------
    log_returns : 1-minute log returns.
    window      : Rolling window for realised vol estimate (bars).
    alpha       : Elastic net regularisation strength.
    l1_ratio    : Mix of L1 (lasso) and L2 (ridge); 0 = pure ridge, 1 = lasso.

    Returns rho clipped to [-0.99, 0.99].
    """
    r = np.asarray(log_returns, dtype=float)
    if len(r) < window + 5:
        return -0.5   # not enough data; return a sensible crypto default

    # Rolling realised vol (standard deviation over `window` bars)
    rv = np.array([np.std(r[max(0, i - window): i]) for i in range(window, len(r) + 1)])
    delta_rv  = np.diff(rv)
    ret_lagged = r[window: window + len(delta_rv)]

    if len(ret_lagged) < 5:
        return -0.5

    X = ret_lagged
    y = delta_rv

    # Centre
    xm, ym = np.mean(X), np.mean(y)
    Xc, yc = X - xm, y - ym

    # Elastic net for single predictor (coordinate descent closed form):
    #   beta* = sign(cov) * max(|cov| - l1_reg/2, 0) / (var_x + l2_reg)
    cov   = float(np.mean(Xc * yc))
    var_x = float(np.mean(Xc ** 2))
    l1_reg = alpha * l1_ratio
    l2_reg = alpha * (1.0 - l1_ratio)

    if var_x < 1e-18:
        return -0.5

    beta_raw = np.sign(cov) * max(abs(cov) - l1_reg / 2.0, 0.0) / (var_x + l2_reg)

    # Normalise to correlation scale: rho = beta * std(X) / std(y)
    std_x = sqrt(var_x)
    std_y = float(np.std(yc))
    if std_y < 1e-18:
        return -0.5

    rho = float(np.clip(beta_raw * std_x / std_y, -0.99, 0.99))
    return rho


# ---------------------------------------------------------------------------
# Heston stochastic-vol — calibration + Monte Carlo pricing
# ---------------------------------------------------------------------------

def calibrate_heston_from_log_returns(
    log_returns: np.ndarray,
    rho: Optional[float] = None,
    dt_hours: float = 1 / 60,
) -> HestonParams:
    """Calibrate Heston parameters from 1-minute log returns.

    Strategy (all in per-hour variance units):
      v0    — EWMA terminal conditional variance.
      theta — Mean of non-overlapping 1-hour realised variances.
      kappa — Derived from AR(1) on hourly variances:
                  h_var[t] = a + b·h_var[t-1] + ε
                  kappa = -ln(b_clipped) per hour.
      xi    — Vol-of-vol derived from AR(1) residuals:
                  xi ≈ std(ε) / sqrt(theta).
              Enforces the Feller condition: xi ≤ sqrt(2·kappa·theta)·0.99.
      rho   — From `estimate_price_vol_rho` when not supplied.

    Requires at least 5 complete 60-bar windows (≥ 300 returns).
    """
    r = np.asarray(log_returns, dtype=float)
    n_windows = len(r) // 60
    if n_windows < 5:
        raise ValueError(
            f"Need at least 300 log-returns (5 hourly windows) for Heston; got {len(r)}"
        )

    # ── Hourly realised variances (per-hour units) ──────────────────────────
    h_var = np.array([
        np.var(r[i * 60: (i + 1) * 60]) / dt_hours   # = var_per_minute * 60
        for i in range(n_windows)
    ])

    theta = max(float(np.mean(h_var)), 1e-12)

    # ── v0: EWMA terminal variance (per-hour) ────────────────────────────────
    var_ewma = float(np.var(r))
    lam = 0.94
    for ret in r:
        var_ewma = lam * var_ewma + (1.0 - lam) * ret ** 2
    v0 = max(var_ewma / dt_hours, 1e-12)

    # ── kappa + xi from AR(1) on h_var ──────────────────────────────────────
    x_ar = h_var[:-1]
    y_ar = h_var[1:]
    xm, ym = float(np.mean(x_ar)), float(np.mean(y_ar))
    cov_xy = float(np.sum((x_ar - xm) * (y_ar - ym)))
    var_x  = float(np.sum((x_ar - xm) ** 2))

    if var_x > 1e-20:
        b = float(np.clip(cov_xy / var_x, 0.01, 0.9999))
    else:
        b = 0.85  # fallback: moderate persistence

    a         = ym - b * xm
    residuals = y_ar - (a + b * x_ar)
    kappa     = max(float(-np.log(b)), 0.5)    # floor: at least 0.5/h reversion
    xi        = float(np.std(residuals)) / max(sqrt(theta), 1e-9)
    xi        = max(xi, 0.01)

    # ── Feller condition: 2κθ ≥ ξ²  (guarantees V_t > 0) ───────────────────
    feller_xi_max = sqrt(2.0 * kappa * theta) * 0.99
    xi = min(xi, feller_xi_max)

    # ── rho ─────────────────────────────────────────────────────────────────
    if rho is None:
        rho = estimate_price_vol_rho(r)

    return HestonParams(
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=float(np.clip(rho, -0.99, 0.99)),
    )


def heston_binary_prob(
    trade_price: float,
    strike_price: float,
    params: HestonParams,
    horizon_hours: float = 1.0,
    direction: Direction = "above",
    n_sims: int = 20_000,
    seed: Optional[int] = 7,
) -> float:
    """Price binary contract via Monte Carlo under the Heston SV model.

    Uses the full-truncation Euler scheme (Broadie & Kaya 2006):
        V⁺_t = max(V_t, 0)
        V_{t+Δt} = V⁺ + κ(θ - V⁺)Δt + ξ·√(V⁺·Δt)·Z₂
        ln S_{t+Δt} = ln S_t − ½V⁺·Δt + √(V⁺·Δt)·Z₁
    where Z₁, Z₂ are correlated: Z₂ = ρ·Z₁ + √(1−ρ²)·Z_indep.

    Steps are in 1-minute increments for accuracy at sub-hourly horizons.
    """
    if trade_price <= 0 or strike_price <= 0:
        raise ValueError("Prices must be positive")

    rng      = np.random.default_rng(seed)
    n_steps  = max(1, round(horizon_hours * 60))
    dt       = horizon_hours / n_steps          # hours per step
    rho_h    = float(np.clip(params.rho, -0.99, 0.99))
    sqrt_1r2 = sqrt(1.0 - rho_h ** 2)

    V    = np.full(n_sims, params.v0)
    ln_S = np.full(n_sims, np.log(trade_price))

    for _ in range(n_steps):
        V_pos    = np.maximum(V, 0.0)
        sqrt_Vdt = np.sqrt(V_pos * dt)

        Z1 = rng.standard_normal(n_sims)
        Z2 = rho_h * Z1 + sqrt_1r2 * rng.standard_normal(n_sims)

        ln_S += -0.5 * V_pos * dt + sqrt_Vdt * Z1
        V     = V_pos + params.kappa * (params.theta - V_pos) * dt + params.xi * sqrt_Vdt * Z2

    S_T     = np.exp(ln_S)
    p_above = float(np.mean(S_T > strike_price))
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

