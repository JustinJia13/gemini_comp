# Live Trading Simulation — Usage Guide

The pipeline has three scripts that work together. In normal operation you only
need to start one — the simulation auto-launches the other two.

```
live_trading_sim.py              ← start this
  └─ getdata_underlying.py       ← auto-started (1-min OHLCV updater)
  └─ getdata_prediction_contract.py  ← auto-started (contract prices)
```

---

## Quick start

```bash
python live_trading_sim.py
```

Both data-collector subprocesses start automatically, print their PIDs, and are
terminated cleanly when the sim exits (Ctrl-C or otherwise).

---

## All CLI flags

### Core simulation

| Flag | Default | Description |
|---|---|---|
| `--poll-sec` | 60 | Seconds between update cycles |
| `--min-edge` | 0.07 | Minimum confidence-adjusted edge to enter (e.g. 0.07 = 7¢) |
| `--max-hours-to-settle` | 1.5 | Skip contracts settling more than this many hours away |
| `--min-hours-to-settle` | 0.0 | Skip contracts with fewer than this many hours to settle (0 = off; avoids near-expiry illiquidity) |
| `--ewma-lambda` | 0.94 | EWMA decay factor λ (RiskMetrics default) |
| `--rho` | −0.5 | Prior price-vol correlation for skewed-t / Heston |
| `--vol-veto-mult` | 2.0 | Block all entries when 10-min realised vol > this × EWMA vol (0 = off) |
| `--no-collectors` | off | Skip auto-starting the two data-collector scripts |
| `--trades-dir` | `.data/gemini/sim_trades` | Output directory for trades/ledger CSV files |

### Exit conditions

| Flag | Default | Description |
|---|---|---|
| `--profit-lock` | 0.10 | Exit when `bid_now − ask_entry ≥ this` (10¢ absolute gain). Skipped within 30 min of settlement. |
| `--stop-loss` | 0.50 | Exit when `(ask_entry − bid_now) / ask_entry ≥ this` (50% relative drawdown). 19¢ entry stops at 9.5¢. |
| `--p-drop` | 0.10 | Exit when model p(side) drops ≥ this from entry p_fair |
| `--edge-neg-thresh` | 0.02 | `edge_closed` fires only when edge ≤ −this (hysteresis; 0 = any negative edge) |

### Per-model lookback windows (hours)

| Flag | Default | Model |
|---|---|---|
| `--lb-gbm` | 12 | GBM rolling σ |
| `--lb-ewma` | 24 | EWMA vol |
| `--lb-garch` | 48 | GARCH(1,1) |
| `--lb-stud` | 24 | Student-t |
| `--lb-skt` | 24 | Skewed-t |
| `--lb-heston` | 72 | Heston SV |
| `--lb-hybrid` | 24 | Hybrid-t |
| `--lb-ou` | 12 | Ornstein-Uhlenbeck |
| `--lb-heston-ewma` | 72 | Heston-EWMA |
| `--lb-gbm-jump` | 24 | GBM + Jump |
| `--lb-ewma-jump` | 36 | EWMA + Jump |
| `--lb-garch-jump` | 48 | GARCH + Jump |
| `--lb-stud-jump` | 36 | Student-t + Jump |
| `--lb-hybrid-jump` | 36 | Hybrid-t + Jump |

### Confidence system

| Flag | Default | Description |
|---|---|---|
| `--conf-pred-w` | 0.55 | λ₁: weight for model-disagreement component |
| `--conf-data-w` | 0.10 | λ₂: weight for OHLCV data-gap component |
| `--conf-ens-w` | 0.35 | λ₃: weight for directional-agreement component |
| `--conf-k` | 3.0 | Harshness: `pred_conf = max(0, 1 − k × std(p_fairs))` |
| `--conf-active` | (from config) | Comma-separated model names for confidence computation. Example: `heston_ewma,garch,student_t` |

### Simulation friction (sim-only, ignored in live trading)

| Flag | Default | Description |
|---|---|---|
| `--zero-fill-prob` | 0.15 | Probability a pending sim entry is cancelled (models IOC zero-fill) |
| `--entry-slip-max` | 0.02 | Max extra ¢ added to entry ask, drawn from U[0, max] |
| `--exit-slip-max` | 0.01 | Max ¢ subtracted from exit bid at early exit, drawn from U[0, max] |

All defaults are read from [`config.toml`](../config.toml). Edit that file to
change persistent defaults without touching the command line.

---

## Common invocations

```bash
# Default run
python live_trading_sim.py

# Run a confidence-system experiment in a separate output directory
python live_trading_sim.py --trades-dir .data/gemini/sim_trades_v2

# Raise edge threshold and tighten exits
python live_trading_sim.py --min-edge 0.09 --profit-lock 0.08 --stop-loss 0.40

# Adjust confidence harshness (less harsh = more trades)
python live_trading_sim.py --conf-k 2.0

# Override confidence active models
python live_trading_sim.py --conf-active heston_ewma,garch,garch_jump

# Disable slippage friction to replicate old sim behaviour
python live_trading_sim.py --zero-fill-prob 0 --entry-slip-max 0 --exit-slip-max 0

# Use shorter GARCH and Heston windows for faster regime tracking
python live_trading_sim.py --lb-garch 24 --lb-heston 48 --lb-heston-ewma 48

# Manage collectors manually in separate terminals
python live_trading_sim.py --no-collectors
python getdata_underlying.py        # in another terminal
python getdata_prediction_contract.py  # in another terminal
```

---

## Configuration file

All defaults live in [`config.toml`](../config.toml). Key sections:

```toml
[simulation]
poll_sec             = 60
min_edge             = 0.07
max_hours_to_settle  = 1.5
ewma_lambda          = 0.94
rho                  = -0.5
vol_veto_mult        = 2.0

[simulation.exit]
profit_lock      = 0.10   # 10¢ absolute gain
stop_loss        = 0.50   # 50% relative drawdown from entry price
p_drop           = 0.10   # 10pp model probability drop
edge_neg_thresh  = 0.02   # hysteresis band for edge_closed

[simulation.lookback]
gbm           = 12.0
ewma          = 24.0
garch         = 48.0
stud          = 24.0
skt           = 24.0
heston        = 72.0
hybrid        = 24.0
ou            = 12.0
heston_ewma   = 72.0
gbm_jump      = 24.0
ewma_jump     = 36.0
garch_jump    = 48.0
student_t_jump= 36.0
hybrid_t_jump = 36.0

[simulation.confidence]
pred_conf_weight     = 0.55
data_conf_weight     = 0.10
ensemble_conf_weight = 0.35
pred_conf_k          = 3.0
active_models        = ["heston_ewma", "garch", "student_t", "skewed_t", "garch_jump", "gbm_jump"]

[simulation.slippage]
zero_fill_prob = 0.15
entry_slip_max = 0.02
exit_slip_max  = 0.01
```

---

## Entry and exit execution

**Entry sequence:**
1. CSV edge check: `edge_yes` or `edge_no` for a model exceeds `min_edge`.
2. Confidence computation: `edge_adj = raw_edge × total_conf`. Skip if `edge_adj ≤ min_edge`.
3. Contract queued in `pending_by_model` (1-poll delay before executing).
4. Next poll: live ask re-fetched from Gemini API. If fresh edge gone, cancelled.
5. Sim friction applied: zero-fill check, then entry slippage. If slippage kills edge, cancelled.
6. Position opened at friction-adjusted ask.

**Exit conditions (first one to fire wins):**

| Condition | Trigger | Notes |
|---|---|---|
| `profit_lock` | `bid_now − ask_entry ≥ 0.10` | Skipped within 30 min of settlement |
| `stop_loss` | `(ask_entry − bid_now) / ask_entry ≥ 0.50` | Relative; 19¢ entry stops at 9.5¢ |
| `p_drop` | Model p(side) fell ≥ 0.10 from entry | Information-based exit |
| `edge_closed` | Edge ≤ −0.02 | Hysteresis prevents noise-triggered exits |
| `settlement` | Contract expired | Spot from OHLCV |

After `stop_loss` or `p_drop` exit, same `contract × model` requires **2× min_edge** to re-enter.

---

## Output files

```
.data/gemini/sim_trades/
  trades_{YYYYMMDD}.csv       — one row per closed position (21 fields)
  edge_log_{YYYYMMDD}.csv     — all 14 model edges at first contract sighting
  performance_ledger.csv      — cumulative P&L / accuracy per model
                                (aggregated from all historical files,
                                 updated after every close)
```

**Trade CSV fields (21):**
`entry_time_utc, contract_id, event_ticker, asset, strike, direction,
settle_time_utc, hours_to_settle_at_entry, side, model, p_fair, ask_price,
edge, exit_time_utc, exit_reason, exit_bid, spot_at_settle, outcome, pnl,
status, gemini_order_id`

Use `--trades-dir` for parallel experiments:
```bash
python live_trading_sim.py                                    # default dir
python live_trading_sim.py --trades-dir .data/gemini/sim_v2  # separate dir
```

---

## Running the data collectors standalone

```bash
# 1-minute OHLCV for BTC / ETH / SOL
python getdata_underlying.py
python getdata_underlying.py --poll-sec 30
python getdata_underlying.py --symbols btcusd ethusd

# Gemini prediction contract prices
python getdata_prediction_contract.py
python getdata_prediction_contract.py --poll-sec 30
```

Underlying data writes to `.data/gemini/ohlcv_1m_7d/{symbol}_est.data`.
Contract data writes to `.data/gemini/prediction_data/{YYYYMMDD}.csv`.

---

## Live trading (`live_trader.py`)

`live_trader.py` wraps `live_trading_sim.run()` with a real Gemini order layer.
It accepts all the same flags as the simulation plus these additional ones:

| Flag | Default | Description |
|---|---|---|
| `--api-key` | `$GEMINI_API_KEY` | Gemini API key |
| `--api-secret` | `$GEMINI_API_SECRET` | Gemini API secret |
| `--sandbox` | off | Paper trading — no real money |
| `--bet-dollars` | 20.0 | Target notional per trade (USD) |
| `--model` | (none) | Individual models to trade alongside the ensemble. Default: ensemble only. Example: `--model skewed_t heston_ewma` |
| `--trades-dir` | `.data/gemini/real_trades` | Output directory for real trade files |

**Default behaviour: ensemble only.** No individual model places orders unless
explicitly named with `--model`. The ensemble is always active regardless.

Real trades are written to `.data/gemini/real_trades/` (separate from the sim
directory to prevent cross-contamination):

```
.data/gemini/real_trades/
  trades_{YYYYMMDD}.csv       — real trade log (same 21-field schema as sim)
  performance_ledger.csv      — same as sim ledger + real-dollar columns
                                (total_real_dollars, avg_real_dollars_per_trade)
```

**"Position gone" settlement inference.** When the exchange returns
`InsufficientPosition` on a sell, the remaining contracts have already been
closed. If the contract's settle time has passed, the code infers payout
($1.00 or $0.00) by comparing the live BTC/ETH/SOL spot against the
contract's strike and direction. Pre-expiry "position gone" (external sale)
is conservatively recorded at $0.

**Order mechanics:**
- IOC (immediate-or-cancel) only — no resting GTC orders
- Entry: `floor(bet_dollars / ask_price)` contracts
- Entry price: `ask + 0.05` buffer to absorb stale-CSV spread (IOC fills at true market ask, not the limit)
- Partial IOC fills: unsold contracts stay open and are retried next poll; final PnL blends partial and full-exit legs
