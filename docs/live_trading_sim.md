# Live Trading Simulation — Usage Guide

The pipeline has three scripts that work together.  In normal operation you
only need to start one of them — the simulation auto-launches the other two.

```
live_trading_sim.py              ← start this
  └─ getdata_underlying.py       ← auto-started (1-min OHLCV updater)
  └─ getdata_prediction_contract.py  ← auto-started (Gemini prediction contract prices)
```

---

## Quick start

```bash
python live_trading_sim.py
```

That's it.  Both data-collector subprocesses start automatically, print their
PIDs, and are terminated cleanly when the sim exits (Ctrl-C or otherwise).

---

## All CLI flags

```
python live_trading_sim.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--poll-sec` | 60 | Seconds between update cycles |
| `--min-edge` | 0.03 | Minimum model edge to enter a trade (e.g. 0.03 = 3¢) |
| `--max-hours-to-settle` | 1.5 | Skip contracts settling more than this far away |
| `--profit-lock` | 0.05 | Early exit when `bid_now − ask_entry ≥ this` |
| `--stop-loss` | 0.10 | Early exit when `ask_entry − bid_now ≥ this` |
| `--p-drop` | 0.05 | Early exit when model p(side) drops by this from entry |
| `--ewma-lambda` | 0.94 | EWMA decay factor λ (RiskMetrics default) |
| `--rho` | −0.5 | Prior price-vol correlation for skewed-t / Heston |
| `--lb-gbm` | 24.0 | GBM calibration lookback (hours) |
| `--lb-ewma` | 48.0 | EWMA calibration lookback (hours) |
| `--lb-garch` | 72.0 | GARCH calibration lookback (hours) |
| `--lb-stud` | 48.0 | Student-t calibration lookback (hours) |
| `--lb-skt` | 48.0 | Skewed-t calibration lookback (hours) |
| `--lb-heston` | 96.0 | Heston calibration lookback (hours) |
| `--no-collectors` | off | Skip auto-starting the two data-collector scripts |

All defaults are read from [`config.toml`](../config.toml) — edit that file to
change the persistent defaults without touching the command line.

---

## Common invocations

```bash
# Default run (recommended — collectors start automatically)
python live_trading_sim.py

# Raise the edge bar and tighten exits
python live_trading_sim.py --min-edge 0.05 --profit-lock 0.04 --stop-loss 0.08

# Use a shorter GBM window (more reactive) and longer Heston window
python live_trading_sim.py --lb-gbm 12 --lb-heston 120

# Poll every 30 s instead of 60 s
python live_trading_sim.py --poll-sec 30

# Run collectors manually in separate terminals, skip auto-start
python live_trading_sim.py --no-collectors
# (in another terminal) python getdata_underlying.py
# (in another terminal) python getdata_prediction_contract.py
```

---

## Configuration file

All defaults live in [`config.toml`](../config.toml).  The relevant sections:

```toml
[simulation]
poll_sec            = 60
min_edge            = 0.03
max_hours_to_settle = 1.5
ewma_lambda         = 0.94
rho                 = -0.5

[simulation.exit]
profit_lock = 0.05
stop_loss   = 0.10
p_drop      = 0.05

[simulation.lookback]
# Per-model calibration windows (hours)
gbm    = 24.0   # short — tracks intraday regime shifts
ewma   = 48.0   # EWMA down-weights old data anyway
garch  = 72.0   # MLE needs more bars for tight α, β
stud   = 48.0   # Student-t tail estimation
skt    = 48.0   # Skewed-t (same as student_t)
heston = 96.0   # AR(1) on hourly realised vars needs many windows
```

---

## Volatility models

Each contract is evaluated by six models independently.  A trade fires only
when one model's edge exceeds `--min-edge`.

| # | Model | Key params | Notes |
|---|---|---|---|
| 1 | **GBM** | rolling σ | Closed-form Black-Scholes binary price |
| 2 | **EWMA** | λ, σ_EWMA | Exponentially-weighted σ, same closed form |
| 3 | **GARCH(1,1)** | α, β, ω | MLE-fit; uses longer lookback for stability |
| 4 | **Student-t** | ν, scale | Fat-tail MC (20 k paths) |
| 5 | **Skewed-t** | ν, γ, rho | Fernández-Steel; γ from elastic-net regression |
| 6 | **Heston SV** | κ, θ, ξ, ρ, v₀ | Full-truncation Euler MC; κ/ξ from AR(1) on hourly realised variance |

Per-model lookback windows are independent — GARCH and Heston use longer
histories while GBM stays reactive to intraday moves.  Windows with the same
length share a single pre-computed return array (no redundant reads).

---

## Entry and exit execution

The contract CSV (written by `getdata_prediction_contract.py`) has up to 60s
lag.  To avoid entering at a stale price or exiting at a stale bid, the sim
makes a direct Gemini API call at the moment of execution:

- **Entry:** after the CSV edge check passes, a live quote is fetched.  If the
  ask has moved and the edge is gone, the trade is skipped.  Otherwise the
  fresh ask is used as the recorded entry price.

- **Early exit:** after a profit-lock / stop-loss / p-drop / edge-closed
  signal fires, a live bid is fetched and used as `exit_bid` so the recorded
  P&L is accurate.

Settlement uses OHLCV data (not market prices), so no live quote is needed
there.  The settlement price is the close of the last 1-minute bar whose open
is strictly before the settle timestamp — the correct look-ahead-free reference.

**Exit conditions (first one to fire wins):**

| Condition | Trigger |
|---|---|
| `profit_lock` | `bid_now − ask_entry ≥ profit_lock` (default 5¢) |
| `stop_loss` | `ask_entry − bid_now ≥ stop_loss` (default 10¢) |
| `p_drop` | model p(side) fell by ≥ `p_drop` from entry (default 5 pp) |
| `edge_closed` | current edge for our side turned negative |
| settlement | contract expired; spot looked up from OHLCV |

After any exit, the contract becomes eligible for re-entry on the next poll
if a new edge appears (no session-long blacklist).

---

## Output files

```
.data/gemini/sim_trades/
  trades_{YYYYMMDD}.csv          — one row per closed position
  edge_log_{YYYYMMDD}.csv        — all model edges at first contract sighting
  performance_ledger.csv         — cumulative P&L / accuracy per model
                                   (aggregated from all historical trade files,
                                    updated after every close)
```

---

## Running the data collectors standalone

If you prefer to manage the collectors yourself (e.g. on a different machine or
in a tmux pane), launch them independently and pass `--no-collectors` to the sim.

```bash
# 1-minute OHLCV for BTC / ETH / SOL
python getdata_underlying.py                    # default 60 s poll
python getdata_underlying.py --poll-sec 30
python getdata_underlying.py --symbols btcusd ethusd

# Gemini prediction contract prices
python getdata_prediction_contract.py           # default 60 s poll
python getdata_prediction_contract.py --poll-sec 30
```

The underlying updater writes to `.data/gemini/ohlcv_1m_7d/{symbol}_est.data`,
which is the file the simulation reads on every poll.
