"""Hourly BTC contract modeling with GBM and Student's t fat-tail returns.

This module is built for Gemini 1-minute OHLCV `.data` files. It creates
hourly prediction contracts where you trade 2 minutes before each hour and
settle 60 minutes later.

Model family 1 (traditional SDE / GBM):
    dS_t / S_t = mu dt + sigma dW_t

Model family 2 (fat-tail extension):
    log-return ~ Student's t with calibrated degrees of freedom.

The code intentionally keeps dependencies minimal (`numpy`, `pandas`) so it
can run in lightweight notebook environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from contract_pricing import three_contract_strikes_from_anchor


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
class DistLinearModel:
    """Two-head linear model for mean and variance of 1-hour log returns.

    mean_head:
        E[r | x] = x @ beta_mean
    variance_head:
        log Var[r | x] = x @ beta_log_var
    """

    feature_cols: list[str]
    beta_mean: np.ndarray
    beta_log_var: np.ndarray


def load_gemini_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load Gemini `.data` minute OHLCV and return UTC-indexed DataFrame."""
    df = pd.read_csv(path)
    required = {"timestamp_ms", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]


def build_hourly_contracts(
    minute_df: pd.DataFrame,
    trade_minute_est: int = 58,
    horizon_minutes: int = 2,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Construct hourly contracts from 1-minute candles.

    Contract convention:
    - Trade every hour at EST minute `trade_minute_est` (default :58).
    - Settle at the next whole hour mark: trade_time + horizon_minutes (default 2).
      Example: bet placed at 9:58 settles at 10:00.
    - anchor_price: close at exactly 1h before settlement (e.g. 9:00 close for
      a 10:00 contract). Used to derive a "nice" strike price.
    - contract_return_log: log(settle_price / trade_price).

    Returns a DataFrame with one row per valid contract.
    """
    if minute_df.index.tz is None:
        raise ValueError("minute_df index must be timezone-aware")

    work = minute_df.copy()
    work["ts_est"] = work.index.tz_convert(timezone)
    work["minute_est"] = work["ts_est"].dt.minute

    entries = work[work["minute_est"] == trade_minute_est][["close", "ts_est"]].copy()
    entries = entries.rename(columns={"close": "trade_price", "ts_est": "trade_time_est"})
    entries["settle_time_utc"] = entries.index + pd.Timedelta(minutes=horizon_minutes)

    close_px = work[["close"]]
    settle_px = close_px.rename(columns={"close": "settle_price"})
    merged = entries.merge(settle_px, left_on="settle_time_utc", right_index=True, how="inner")

    # Anchor price: close at 1h before settlement (for "nice" strike derivation).
    anchor_time = merged["settle_time_utc"] - pd.Timedelta(hours=1)
    anchor_px = close_px.rename(columns={"close": "anchor_price"})
    merged = merged.merge(anchor_px, left_on=anchor_time, right_index=True, how="left")
    merged["anchor_price"] = merged["anchor_price"].fillna(merged["trade_price"])

    merged["contract_return_log"] = np.log(merged["settle_price"] / merged["trade_price"])
    merged["settle_time_est"] = merged["settle_time_utc"].dt.tz_convert(timezone)
    # Label the contract by its settlement hour (e.g. 10:00 for the 9:58 bet).
    merged["contract_hour_est"] = merged["settle_time_est"].dt.floor("h")

    out_cols = [
        "trade_time_est",
        "settle_time_est",
        "contract_hour_est",
        "anchor_price",
        "trade_price",
        "settle_price",
        "contract_return_log",
    ]
    return merged[out_cols].reset_index(drop=True)


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


def _contract_horizon_hours(contracts: pd.DataFrame) -> float:
    """Infer contract duration in hours from the first data row."""
    row = contracts.iloc[0]
    return (row["settle_time_est"] - row["trade_time_est"]).total_seconds() / 3600.0


def walkforward_hourly_probabilities(
    contracts: pd.DataFrame,
    lookback_contracts: int = 72,
    strike_mode: Literal["atm", "pct"] = "atm",
    strike_pct: float = 0.0,
    n_sims_t: int = 20_000,
) -> pd.DataFrame:
    """Run rolling calibration and produce hourly probabilities.

    - Each row is a tradeable hourly contract.
    - Uses prior `lookback_contracts` realized contracts to calibrate models.
    - `strike_mode='atm'`: strike = trade_price.
    - `strike_mode='pct'`: strike = trade_price * (1 + strike_pct).

    dt_hours is inferred from the contract settle/trade times so GBMParams
    carry correct hourly units (e.g. sigma_per_sqrt_hour ≈ 0.39% for BTC,
    not the mislabeled 0.07% that results from using dt=1 on 2-min returns).
    """
    dt_hours = _contract_horizon_hours(contracts)
    df = contracts.copy().reset_index(drop=True)
    out_rows = []

    for i in range(lookback_contracts, len(df)):
        hist = df.iloc[i - lookback_contracts : i]
        now = df.iloc[i]

        gbm = calibrate_gbm_from_log_returns(hist["contract_return_log"].values, dt_hours=dt_hours)
        tpar = calibrate_student_t_from_log_returns(hist["contract_return_log"].values)

        trade_px = float(now["trade_price"])
        if strike_mode == "atm":
            strike = trade_px
        else:
            strike = trade_px * (1.0 + strike_pct)

        p_gbm = gbm_binary_prob(trade_px, strike, gbm, horizon_hours=dt_hours, direction="above")
        p_t = student_t_binary_prob(
            trade_px,
            strike,
            tpar,
            direction="above",
            n_sims=n_sims_t,
            seed=17 + i,
        )

        realized = float(now["settle_price"] > strike)
        out_rows.append(
            {
                "trade_time_est": now["trade_time_est"],
                "settle_time_est": now["settle_time_est"],
                "trade_price": trade_px,
                "strike": strike,
                "settle_price": float(now["settle_price"]),
                "realized_above": realized,
                "p_above_gbm": p_gbm,
                "p_above_student_t": p_t,
                "mu_per_hour": gbm.mu_per_hour,
                "sigma_per_sqrt_hour": gbm.sigma_per_sqrt_hour,
                "t_loc": tpar.loc,
                "t_scale": tpar.scale,
                "t_nu": tpar.nu,
            }
        )

    return pd.DataFrame(out_rows)


def model_summary_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Simple diagnostics for probability quality."""
    if preds.empty:
        return pd.DataFrame()

    y = preds["realized_above"].to_numpy(dtype=float)
    p_g = np.clip(preds["p_above_gbm"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    p_t = np.clip(preds["p_above_student_t"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)

    def brier(p: np.ndarray) -> float:
        return float(np.mean((p - y) ** 2))

    def logloss(p: np.ndarray) -> float:
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return pd.DataFrame(
        [
            {
                "model": "gbm",
                "brier": brier(p_g),
                "logloss": logloss(p_g),
                "mean_prob": float(np.mean(p_g)),
            },
            {
                "model": "student_t",
                "brier": brier(p_t),
                "logloss": logloss(p_t),
                "mean_prob": float(np.mean(p_t)),
            },
        ]
    )


def evaluate_latest_three_strikes(
    contracts: pd.DataFrame,
    lookback_contracts: int = 24,
    spacing: float = 250.0,
    n_sims_t: int = 20_000,
) -> pd.DataFrame:
    """Evaluate 3 strike contracts for the latest tradeable hour."""
    if len(contracts) < lookback_contracts + 1:
        raise ValueError("Not enough contracts for requested lookback")

    dt_hours = _contract_horizon_hours(contracts)
    hist = contracts.iloc[-(lookback_contracts + 1) : -1]
    now = contracts.iloc[-1]

    gbm = calibrate_gbm_from_log_returns(hist["contract_return_log"].values, dt_hours=dt_hours)
    tpar = calibrate_student_t_from_log_returns(hist["contract_return_log"].values)

    trade_price = float(now["trade_price"])
    settle_price = float(now["settle_price"])
    # Anchor strikes on the close price 1h before settlement (stored in anchor_price).
    anchor_price_1h_ago = float(now["anchor_price"])
    strikes = three_contract_strikes_from_anchor(anchor_price_1h_ago, spacing=spacing)

    rows = []
    for i, strike in enumerate(strikes):
        p_gbm = gbm_binary_prob(trade_price, strike, gbm, horizon_hours=dt_hours, direction="above")
        p_t = student_t_binary_prob(
            trade_price,
            strike,
            tpar,
            direction="above",
            n_sims=n_sims_t,
            seed=100 + i,
        )
        realized = float(settle_price > strike)
        pred_gbm = float(p_gbm >= 0.5)
        pred_t = float(p_t >= 0.5)
        rows.append(
            {
                "contract": f"BTC > ${strike:,.0f}",
                "trade_time_est": now["trade_time_est"],
                "settle_time_est": now["settle_time_est"],
                "trade_price": trade_price,
                "anchor_price_1h_ago": anchor_price_1h_ago,
                "settle_price": settle_price,
                "strike": strike,
                "p_above_gbm": p_gbm,
                "p_above_student_t": p_t,
                "actual_result": "YES" if realized == 1.0 else "NO",
                "gbm_prediction": "YES" if pred_gbm == 1.0 else "NO",
                "student_t_prediction": "YES" if pred_t == 1.0 else "NO",
                "gbm_correct": bool(pred_gbm == realized),
                "student_t_correct": bool(pred_t == realized),
            }
        )

    return pd.DataFrame(rows)


def walkforward_three_strikes(
    contracts: pd.DataFrame,
    lookback_contracts: int = 72,
    spacing: float = 250.0,
    n_sims_t: int = 20_000,
) -> pd.DataFrame:
    """Walk-forward over ALL contracts, pricing the 3 anchor-based strikes each hour.

    Unlike evaluate_latest_three_strikes (which only looks at the last row),
    this iterates every contract after the warmup period so accuracy can be
    measured across the full history.

    Returns one row per (contract × strike):  3 rows per hourly slot.
    """
    dt_hours = _contract_horizon_hours(contracts)
    df = contracts.copy().reset_index(drop=True)
    out_rows = []

    for i in range(lookback_contracts, len(df)):
        hist = df.iloc[i - lookback_contracts : i]
        now = df.iloc[i]

        gbm = calibrate_gbm_from_log_returns(hist["contract_return_log"].values, dt_hours=dt_hours)
        tpar = calibrate_student_t_from_log_returns(hist["contract_return_log"].values)

        trade_px = float(now["trade_price"])
        settle_px = float(now["settle_price"])
        anchor_px = float(now["anchor_price"])
        strikes = three_contract_strikes_from_anchor(anchor_px, spacing=spacing)

        for rank, strike in enumerate(strikes):  # 0=high, 1=mid, 2=low
            p_gbm = gbm_binary_prob(trade_px, strike, gbm, horizon_hours=dt_hours, direction="above")
            p_t = student_t_binary_prob(
                trade_px, strike, tpar, direction="above",
                n_sims=n_sims_t, seed=i * 3 + rank,
            )
            realized = float(settle_px > strike)
            pred_gbm = float(p_gbm >= 0.5)
            pred_t = float(p_t >= 0.5)
            out_rows.append({
                "trade_time_est": now["trade_time_est"],
                "settle_time_est": now["settle_time_est"],
                "contract_hour_est": now["contract_hour_est"],
                "strike_rank": rank,
                "anchor_price": anchor_px,
                "strike": strike,
                "trade_price": trade_px,
                "settle_price": settle_px,
                "realized_above": realized,
                "p_above_gbm": p_gbm,
                "p_above_student_t": p_t,
                "gbm_correct": bool(pred_gbm == realized),
                "student_t_correct": bool(pred_t == realized),
            })

    return pd.DataFrame(out_rows)


def three_strike_accuracy_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Accuracy breakdown by strike level (high/mid/low), overall, and contested.

    'contested' filters to predictions where GBM gives 0.2 < p < 0.8 — cases
    where the model is genuinely uncertain. High/low accuracy is almost always
    inflated by deep OTM strikes that are trivially predictable.
    """
    if preds.empty:
        return pd.DataFrame()

    rank_labels = {0: "high", 1: "mid", 2: "low"}
    rows = []
    for rank, label in [(-1, "overall")] + list(rank_labels.items()):
        sub = preds if rank == -1 else preds[preds["strike_rank"] == rank]
        if sub.empty:
            continue
        n_hours = len(sub) // 3 if label == "overall" else len(sub)
        for model, col in [("gbm", "gbm_correct"), ("student_t", "student_t_correct")]:
            rows.append({
                "strike_level": label,
                "model": model,
                "accuracy": float(sub[col].mean()),
                "n": n_hours,
            })

    # Contested: where GBM assigns genuine uncertainty (not near 0 or 1).
    contested = preds[(preds["p_above_gbm"] > 0.2) & (preds["p_above_gbm"] < 0.8)]
    if not contested.empty:
        for model, col in [("gbm", "gbm_correct"), ("student_t", "student_t_correct")]:
            rows.append({
                "strike_level": "contested(0.2<p<0.8)",
                "model": model,
                "accuracy": float(contested[col].mean()),
                "n": len(contested),
            })

    return pd.DataFrame(rows)


def classification_accuracy_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Return overall directional accuracy for each method."""
    if preds.empty:
        return pd.DataFrame()

    y = preds["realized_above"].to_numpy(dtype=float)
    gbm_pred = (preds["p_above_gbm"].to_numpy(dtype=float) >= 0.5).astype(float)
    t_pred = (preds["p_above_student_t"].to_numpy(dtype=float) >= 0.5).astype(float)

    return pd.DataFrame(
        [
            {
                "method": "gbm",
                "accuracy": float(np.mean(gbm_pred == y)),
                "n_predictions": int(len(y)),
            },
            {
                "method": "student_t",
                "accuracy": float(np.mean(t_pred == y)),
                "n_predictions": int(len(y)),
            },
        ]
    )


def build_ml_feature_table(contracts: pd.DataFrame, n_lags: int = 6) -> pd.DataFrame:
    """Create a supervised table for distributional ML on hourly contracts.

    Features:
    - Lagged hourly log returns (1..n_lags)
    - Lagged hourly volume z-score
    - Hour-of-day cyclical encoding (EST)
    """
    df = contracts.copy().sort_values("trade_time_est").reset_index(drop=True)
    df["hour_est"] = df["trade_time_est"].dt.hour

    for lag in range(1, n_lags + 1):
        df[f"ret_lag_{lag}"] = df["contract_return_log"].shift(lag)

    vol = np.log(df["trade_price"]).diff().abs().fillna(0.0)
    vol_rolling = vol.rolling(24, min_periods=8).std()
    df["vol_z"] = (vol - vol.rolling(24, min_periods=8).mean()) / vol_rolling.replace(0, np.nan)
    df["vol_z"] = df["vol_z"].fillna(0.0)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_est"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_est"] / 24.0)

    return df


def _fit_ols(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Stable closed-form linear regression with small ridge regularization."""
    xtx = X.T @ X
    n = xtx.shape[0]
    return np.linalg.solve(xtx + ridge * np.eye(n), X.T @ y)


def fit_distributional_linear_model(train_df: pd.DataFrame, n_lags: int = 6) -> DistLinearModel:
    """Fit mean and variance linear heads for conditional return distribution."""
    feature_cols = [f"ret_lag_{i}" for i in range(1, n_lags + 1)] + ["vol_z", "hour_sin", "hour_cos"]
    work = train_df.dropna(subset=feature_cols + ["contract_return_log"]).copy()
    if len(work) < 20:
        raise ValueError("Need at least 20 rows to fit distributional linear model")

    X = work[feature_cols].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    y = work["contract_return_log"].to_numpy(dtype=float)

    beta_mean = _fit_ols(X, y)
    resid = y - X @ beta_mean
    # Fit log variance on squared residuals.
    y_var = np.log(np.maximum(resid * resid, 1e-12))
    beta_log_var = _fit_ols(X, y_var)

    return DistLinearModel(feature_cols=feature_cols, beta_mean=beta_mean, beta_log_var=beta_log_var)


def predict_distributional_linear(model: DistLinearModel, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Predict conditional mean and std for hourly log returns."""
    work = feature_df.copy()
    X = work[model.feature_cols].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])

    mu = X @ model.beta_mean
    log_var = X @ model.beta_log_var
    sigma = np.sqrt(np.maximum(np.exp(log_var), 1e-12))

    out = pd.DataFrame(index=work.index)
    out["mu_logret"] = mu
    out["sigma_logret"] = sigma
    return out


def run_example(
    data_path: str | Path = ".data/gemini/ohlcv_1m_7d/btcusd.data",
    lookback_contracts: int = 24,
    spacing: float = 250.0,
    output_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience entrypoint for notebook/CLI use.

    Returns (contracts, atm_preds, summary, three_strike_preds).
    If output_path is given, three_strike_preds is saved as CSV there.
    """
    minute = load_gemini_ohlcv(data_path)
    contracts = build_hourly_contracts(minute)
    if len(contracts) <= lookback_contracts:
        lookback_contracts = max(10, len(contracts) // 2)
    preds = walkforward_hourly_probabilities(contracts, lookback_contracts=lookback_contracts)
    summary = model_summary_table(preds)
    preds_3s = walkforward_three_strikes(contracts, lookback_contracts=lookback_contracts, spacing=spacing)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        preds_3s.to_csv(out, index=False)
        print(f"  Saved → {out}")
    return contracts, preds, summary, preds_3s


# Spacings are set to roughly 1 standard deviation of each asset's 2-minute
# price move, so strikes are genuinely contested for model evaluation:
#   BTC ~1σ_2min = $80  → $100
#   ETH ~1σ_2min = $2.63 → $3
#   SOL ~1σ_2min = $0.13 → $0.10
# (The actual Gemini market grid is $250 / $20 / $1; use those for live trading.)
_SYMBOL_SPACING: dict[str, float] = {
    "btcusd": 100.0,
    "ethusd": 3.0,
    "solusd": 0.10,
}


if __name__ == "__main__":
    DATA_ROOT = Path(".data/gemini/ohlcv_1m_7d")
    OUTPUT_ROOT = Path("./output")
    LOOKBACK = 72  # ~3 days of hourly contracts

    for symbol, spacing in _SYMBOL_SPACING.items():
        data_path = DATA_ROOT / f"{symbol}.data"
        if not data_path.exists():
            print(f"[{symbol}] data file not found, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {symbol.upper()}  (strike spacing ±{spacing})")
        print(f"{'=' * 60}")

        out_path = OUTPUT_ROOT / f"{symbol}_predictions.csv"
        contracts_df, preds_df, summary_df, preds_3s = run_example(
            data_path=data_path,
            lookback_contracts=LOOKBACK,
            spacing=spacing,
            output_path=out_path,
        )

        print(f"  Contracts available: {len(contracts_df)}")
        print(f"  Walk-forward rows  : {len(preds_3s)}  ({len(preds_3s) // 3} hours × 3 strikes)")

        if not preds_3s.empty:
            print("\nAccuracy by strike level:")
            print(three_strike_accuracy_table(preds_3s).to_string(index=False))

        if not preds_df.empty:
            print("\nLatest 5 ATM predictions:")
            atm_cols = [
                "trade_time_est", "trade_price", "strike",
                "p_above_gbm", "p_above_student_t", "realized_above",
            ]
            print(preds_df[atm_cols].tail(5).to_string(index=False))
