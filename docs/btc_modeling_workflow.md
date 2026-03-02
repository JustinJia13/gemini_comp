# BTC 1h Contract Modeling Workflow

This workflow is implemented in [`btc_hourly_model.py`](../btc_hourly_model.py).

## What it does

1. Loads Gemini minute OHLCV from `.data` files.
2. Builds hourly contracts in `America/New_York` (EST/EDT):
   - trade at `hh:58`
   - settle at `trade_time + 60 minutes`
3. Calibrates:
   - GBM (`mu`, `sigma`) from hourly log returns
   - Student's t (`loc`, `scale`, `nu`) for fat tails
4. Produces walk-forward hourly probabilities `P(settle_price > strike)`.
5. Includes a simple distributional-ML baseline:
   - linear mean head for return
   - linear variance head for log-variance

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

1. Replace linear mean/variance heads with nonlinear models (e.g., boosted trees or neural nets).
2. Predict full distribution directly (quantiles or parametric outputs like `mu`, `sigma`, `nu`).
3. Optimize against your contract payoff (expected value after fees/slippage), not only Brier/logloss.
