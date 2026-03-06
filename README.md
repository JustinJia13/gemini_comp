# gemini_comp

Automated trading system for **Gemini Prediction Markets**. Runs 14 volatility
models + an ensemble to price binary BTC/ETH/SOL contracts and place real or
simulated IOC orders.

## Quick start

```bash
# 1. Set up environment
bash setup.sh
source .venv/bin/activate

# 2. Run the simulation (no real money)
python live_trading_sim.py

# 3. Run the live trader (real orders)
export GEMINI_API_KEY=your_key
export GEMINI_API_SECRET=your_secret
python live_trader.py               # ensemble only, $20/bet
python live_trader.py --sandbox     # paper trading
python live_trader.py --bet-dollars 50 --min-edge 0.09
```

Data collectors start automatically. Stop with `Ctrl-C`.

## Project structure

```
live_trading_sim.py              — simulation engine + entry/exit logic (all models)
live_trader.py                   — live trading entry point (real Gemini orders)
gemini_trader.py                 — GeminiTrader: HMAC auth, order placement, symbol cache
btc_hourly_model.py              — all 14 model implementations
getdata_underlying.py            — continuous 1-min OHLCV updater
getdata_prediction_contract.py   — continuous contract price updater
config.toml                      — all hyperparameters
config_loader.py                 — TOML loader

docs/
  simulation_overview.md         — models, confidence system, entry/exit logic
  live_trading_sim.md            — CLI reference for all flags
  btc_modeling_workflow.md       — model calibration details

.data/gemini/
  ohlcv_1m_7d/{sym}_est.data    — 1-min OHLCV (rolling 7-day)
  prediction_data/{YYYYMMDD}.csv — contract quotes
  sim_trades/                    — simulation trade logs + ledger
  real_trades/                   — live trade logs + ledger
```

## Models

14 models run in parallel, all calibrated from 1-minute OHLCV log returns:

| # | Model | Lookback |
|---|---|---|
| 1 | GBM (rolling σ) | 12 h |
| 2 | EWMA (λ=0.94) | 24 h |
| 3 | GARCH(1,1) | 48 h |
| 4 | Student-t | 24 h |
| 5 | Skewed-t (Fernández-Steel) | 24 h |
| 6 | Heston SV | 72 h |
| 7 | Hybrid-t (EWMA σ + t-tail) | 24 h |
| 8 | OU (Ornstein-Uhlenbeck) | 12 h |
| 9 | Heston-EWMA | 72 h |
| 10–14 | Jump-diffusion overlays (GBM/EWMA/GARCH/Student-t/Hybrid-t + Merton) | 24–48 h |

Plus a 15th **ensemble** model (mean of a configurable subset) that is the
default and only order-placer in live trading.

## Confidence-adjusted edge

Entries require `edge_adj = raw_edge × total_conf > min_edge` (default 7¢):

```
total_conf = pred_conf^0.55 × data_conf^0.10 × ensemble_conf^0.35

pred_conf    = max(0, 1 − 3 × std(p_fairs))   # penalise model disagreement
data_conf    = actual_bars / expected_bars       # penalise OHLCV gaps
ensemble_conf = 0.5 + 0.5 × (n_agree / n_models) # penalise directional splits
```

## Key CLI flags

See [`docs/live_trading_sim.md`](docs/live_trading_sim.md) for the full reference.

```bash
# Common overrides
python live_trading_sim.py --min-edge 0.09 --stop-loss 0.40
python live_trading_sim.py --conf-k 2.0 --conf-active heston_ewma,garch,garch_jump
python live_trading_sim.py --trades-dir .data/gemini/experiment_v2

# Live trader extras
python live_trader.py --bet-dollars 30 --min-hours-to-settle 0.1
python live_trader.py --model skewed_t heston_ewma   # add individual models
```

## Configuration

All defaults in [`config.toml`](config.toml). Key sections:
`[simulation]`, `[simulation.exit]`, `[simulation.lookback]`, `[simulation.confidence]`, `[simulation.slippage]`.
