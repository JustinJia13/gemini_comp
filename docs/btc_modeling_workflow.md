# BTC 1h Contract Modeling Workflow

This workflow is implemented in [`btc_hourly_model.py`](../btc_hourly_model.py).

## What it does

1. Loads Gemini minute OHLCV from `.data` files.
2. Calibrates six models from rolling 1-minute log-return windows:
   - **GBM** — rolling σ, closed-form Black-Scholes binary price
   - **EWMA** — exponentially-weighted σ (λ = 0.94), same closed form
   - **GARCH(1,1)** — MLE-fit α, β, ω; longer lookback for stability
   - **Student-t** — fat-tail MC (20 k paths), fits ν from return series
   - **Skewed-t** — Fernández-Steel; γ (skew) from elastic-net on vol-return regression
   - **Heston SV** — full-truncation Euler MC; κ, ξ from AR(1) on hourly realised variance
3. Produces `P(settle_price > strike)` for each model independently.
4. All calibration uses data strictly before the evaluation timestamp — no look-ahead.

## Quick start

```python
from btc_hourly_model import run_example

contracts, preds, summary = run_example(
    data_path=".data/gemini/ohlcv_1m_7d/btcusd.data",
    lookback_contracts=24,
)

print(summary)
print(preds.tail())
```

## Notebook usage

```python
from btc_hourly_model import (
    load_gemini_ohlcv,
    build_hourly_contracts,
    walkforward_hourly_probabilities,
    model_summary_table,
)

minute = load_gemini_ohlcv(".data/gemini/ohlcv_1m_7d/btcusd.data")
contracts = build_hourly_contracts(minute, trade_minute_est=58, horizon_minutes=60)
preds = walkforward_hourly_probabilities(contracts, lookback_contracts=24)
summary = model_summary_table(preds)
```

## Should you use finer-than-1-minute data?

Short answer: yes, if execution and microstructure matter.

- Keep 1-minute data for this first SDE and distributional baseline.
- Move to 5-second / tick data when you start trading real size or modeling fill/slippage around `hh:58`.
- A practical stack is: fit the directional distribution model on 1-minute or 5-minute engineered features, then overlay a separate execution model on higher-frequency data.

## Next upgrade path

1. Replace linear vol estimates with nonlinear models (e.g., boosted trees on engineered features).
2. Predict full distribution directly (quantiles or parametric outputs like `mu`, `sigma`, `nu`).
3. Optimize against contract payoff (expected value after spread), not only Brier/logloss.
4. Use tick or 5-second data for execution-time models once trading real size.
