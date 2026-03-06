# Simulation & Model Overview

*For teammates familiar with probability and basic finance.*

---

## What we're doing and why

### The market

We trade on **Gemini** prediction contracts. Each contract pays exactly $1 if a
condition is true at expiry and $0 otherwise. Our contracts ask:
*"Will BTC / ETH / SOL close above price X at time T?"*  These are **binary
options** — payout is binary, but the contract price trades continuously between
$0 and $1 and can be read directly as a probability.

For example, if the market prices "BTC above $95,000 at 3pm EST" at $0.62, the
crowd believes there is roughly a 62% chance that happens.

### The edge

Our job is to estimate that probability using models. If our model says 68% but
the market price is 62%, we have a **6-cent edge** on a $1 contract. We buy when
edge exceeds a threshold (currently 7¢). This is the same logic a poker player
uses — you don't need certainty, you just need to be right *more often than the
price implies*.

### The data

We pull **1-minute OHLCV** (open/high/low/close/volume) bars for BTC, ETH, and
SOL from Gemini's public API, updated every 60 seconds. From those bars we
compute **log returns** (percentage price moves, log-scaled so they're additive).
Everything the model sees is from *before* the evaluation time — no look-ahead.
The OHLCV file is used solely for model calibration; the live spot price at
entry/exit is fetched directly from the Gemini ticker API.

---

## The 14 models

Rather than picking one model, we run 14 in parallel and track which perform best
over time. Each model adds one more real-world feature the previous one ignores.
The first 9 are standalone volatility models; Models 10–14 overlay Merton
jump-diffusion on top of the base model.

### Standalone volatility models

| # | Model | Key params | Lookback | Notes |
|---|---|---|---|---|
| 1 | **GBM** | rolling σ | 12 h | Closed-form Black-Scholes binary price |
| 2 | **EWMA** | λ=0.94, σ | 24 h | Exp-weighted σ; same closed form as GBM |
| 3 | **GARCH(1,1)** | α, β, ω | 48 h | MLE-fit; captures volatility clustering |
| 4 | **Student-t** | ν, σ | 24 h | Fat-tail Monte Carlo (20k paths) |
| 5 | **Skewed-t** | ν, γ, ρ | 24 h | Fernández-Steel; asymmetric tails |
| 6 | **Heston SV** | κ, θ, ξ, ρ, v₀ | 72 h | Stochastic vol; MC with full-truncation Euler |
| 7 | **Hybrid-t** | ν, σ_EWMA | 24 h | EWMA conditional σ + Student-t tail shape |
| 8 | **OU** | κ, μ, σ | 12 h | Ornstein-Uhlenbeck on log-price; mean-reversion |
| 9 | **Heston-EWMA** | κ, θ_EWMA, ξ, ρ, v₀ | 72 h | Heston with EWMA-calibrated long-run variance |

### Merton jump-diffusion overlays (Models 10–14)

Each jump model takes its base model's drift and diffusion-only σ, then overlays
a Poisson jump process calibrated from the same data. Jump parameters — arrival
rate λ_J, mean jump log-size μ_J, and jump size dispersion σ_J — are estimated
by filtering large-return outliers from the return series.

| # | Model | Base | Lookback |
|---|---|---|---|
| 10 | **GBM + Jump** | GBM | 24 h |
| 11 | **EWMA + Jump** | EWMA | 36 h |
| 12 | **GARCH + Jump** | GARCH | 48 h |
| 13 | **Student-t + Jump** | Student-t | 36 h |
| 14 | **Hybrid-t + Jump** | Hybrid-t | 36 h |

---

## Why shorter lookback windows?

We deliberately chose **shorter-than-textbook** calibration windows (e.g. 12–48 h
instead of the conventional 72–168 h). This is a **bias-variance tradeoff**
chosen for our specific context:

- **Shorter = lower bias, higher variance.** A 12-hour window captures the current
  vol regime more accurately but is noisier on any given estimate.
- **Longer = lower variance, higher bias.** A 72-hour window averages over multiple
  vol regimes. For 1-hour binary contracts, yesterday's regime shift has already
  fully repriced the market — using a 3-day window bakes in information that is
  no longer relevant to the current trade.
- **Regime switches are fast in crypto.** A vol spike from a macro event or
  liquidation cascade typically resolves within 1–4 hours. A 48-hour window is
  already measuring partially irrelevant data; a 12-hour window captures only the
  current regime.
- **The confidence system compensates for noise.** Noisy short-window estimates
  produce high disagreement across models. Our `pred_conf` component (see below)
  penalises high cross-model variance — so when estimates are unreliable, the
  confidence-adjusted edge shrinks and we skip the trade rather than entering on
  a noisy signal.

**In short:** we want models that track the current market, not models that
"remember" yesterday's regime. Short windows + confidence gating is the
principled way to achieve both.

---

## Confidence-adjusted edge

Before comparing a model's raw edge to `min_edge`, we scale it down by a
composite confidence score. This prevents entering on signals that are
technically above threshold but where the underlying probability estimate is
unreliable.

### Formula

```
edge_adj = raw_edge × total_conf

total_conf = pred_conf^λ₁ × data_conf^λ₂ × ensemble_conf^λ₃
```

The weights `(λ₁, λ₂, λ₃)` sum to 1. This is a **weighted geometric mean** —
each component is a multiplier in `[0, 1]`, and `λ=0` makes a component neutral
(`x^0 = 1`). Setting any λ to zero disables that component entirely without
affecting the others.

An entry fires only if `edge_adj > min_edge`.

### Component 1 — pred_conf (model disagreement)

```
pred_conf = max(0, 1 − k × std(p_fair across active conf models))
```

We compute the standard deviation of the YES probability estimated by each model
in the `active_models` list. High std = models disagree = the estimate is
unreliable. The harshness parameter `k` (default 3.0) controls how quickly
disagreement kills confidence:

- std = 0.00 → pred_conf = 1.00 (complete agreement)
- std = 0.10 → pred_conf = 0.70 (modest disagreement, 30% penalty)
- std = 0.33 → pred_conf = 0.00 (full disagreement, veto)

This also acts as an automatic ATM filter: contracts near 50¢ produce naturally
high model disagreement (the pricing formula is most sensitive near the money),
so they receive lower confidence even when all models agree on direction.

### Component 2 — data_conf (OHLCV completeness)

```
data_conf = min(1, n_actual_bars / n_expected_bars)
```

`n_expected_bars = lookback_hours × 60` (minutes). If the OHLCV file has gaps
(network outage, API failure), the model is calibrated on less data than it
expects. `data_conf` penalises this in proportion to how many bars are missing.

**This does NOT penalise short lookback windows.** Choosing a 12-hour window
over a 48-hour window is deliberate. `data_conf` only fires when the data
available is *less* than what the chosen window demands.

### Component 3 — ensemble_conf (directional agreement)

```
n_agree       = number of active conf models with edge > 0 on our side
ensemble_conf = 0.5 + 0.5 × (n_agree / n_active_conf_models)
```

Range is `[0.5, 1.0]`: 0.5 means half the models disagree; 1.0 means unanimous.
When models split evenly, `ensemble_conf = 0.5`, which scales the edge down by
50% — a meaningful but non-zero penalty.

### Default weights and active models

| Parameter | Default | Effect |
|---|---|---|
| λ₁ (`pred_conf_weight`) | 0.55 | Dominant factor — model disagreement |
| λ₂ (`data_conf_weight`) | 0.10 | Minor factor — data quality guard |
| λ₃ (`ensemble_conf_weight`) | 0.35 | Secondary factor — directional agreement |
| `pred_conf_k` | 3.0 | Harshness: std of 0.10 → conf = 0.70 |
| `active_models` | heston_ewma, garch, student_t, skewed_t, garch_jump, gbm_jump | High-accuracy, diverse model families |

All configurable in `config.toml [simulation.confidence]` or via `--conf-*` CLI flags.

---

## Simulation friction model

To make the simulation closer to live IOC trading, we apply three friction
parameters. These are applied only in simulation mode (`trader=None`) — live
trading handles real fills from the Gemini API directly.

### Zero-fill probability (`zero_fill_prob = 0.15`)

With probability 15%, a pending sim entry is silently cancelled. This models the
reality that IOC orders frequently return 0 fills because the market has moved
between signal detection and order submission. The contract remains eligible for
re-evaluation on the next poll.

### Entry slippage (`entry_slip_max = 0.02`)

At entry execution, a random cost `U[0, entry_slip_max]` is added to the quoted
ask price. Average extra cost: **+1¢**. This models:
- Stale CSV ask (up to 60s old at the time we see it)
- IOC limit execution at or above the quoted price
- The 3¢ buffer added in live trading to absorb staleness

If slippage causes the edge to fall below `min_edge`, the entry is cancelled.

### Exit slippage (`exit_slip_max = 0.01`)

At early exit, a random penalty `U[0, exit_slip_max]` is subtracted from the
quoted bid. Average reduction: **-0.5¢**. This models bid deterioration when we
hit the market to sell — the bid we see is not always the bid we get.

---

## Ensemble model (15th, synthetic)

Beyond the 14 calibrated models the system runs an **ensemble** that averages
the YES probability across a configurable subset of models (`conf_active` list,
default: `heston_ewma, garch, student_t, skewed_t, garch_jump, gbm_jump`).

```
p_fair_ensemble = mean(p_fair_m  for m in conf_active)
edge_adj        = (p_fair_ensemble − ask) × total_conf
```

The confidence components are computed identically to the individual models,
so the ensemble inherits all three penalties (disagreement, data gaps, directional
agreement). In live trading via `live_trader.py`, the ensemble is the **default
and only order-placing agent**. Individual models can be added back with
`--model <name> ...`.

**Exit logic uses the true ensemble mean.** `p_drop` and `edge_closed` both
compute `p_fair_now` as the mean of conf_active p_fairs at the time of the exit
check — not a proxy single model.

---

## Entry and exit execution

**Entry sequence:**
1. Model computes raw edge from CSV ask price.
2. Raw edge passes `min_edge` check.
3. Confidence-adjusted edge computed → `edge_adj = raw_edge × total_conf`.
4. `edge_adj > min_edge` check passes.
5. `min_hours_to_settle` guard: contracts with fewer than this many hours
   remaining are skipped (avoids near-expiry illiquidity; default 0.0 = off).
6. Contract queued in `pending_by_model` (1-poll delay before execution).
7. On the next poll: live ask re-fetched. If edge gone, cancelled. Otherwise
   sim friction applied (zero-fill check, then entry slippage).
8. Position opened at the friction-adjusted ask price.

**Exit conditions (first to fire wins):**

| Condition | Trigger |
|---|---|
| `profit_lock` | `bid_now − ask_entry ≥ profit_lock` (absolute gain). Skipped within 30 min of settlement. |
| `stop_loss` | `(ask_entry − bid_now) / ask_entry ≥ stop_loss` (relative drawdown from entry). |
| `p_drop` | Model/ensemble p(side) fell ≥ 0.10 from entry p_fair. |
| `edge_closed` | Current edge ≤ −`edge_neg_thresh`. Market has corrected. |
| `settlement` | Contract expired. Spot from OHLCV; no live quote needed. |

**Exit bid for decisions:** a fresh live quote is fetched *before* each exit
check so the trigger uses the true current market bid, not a stale CSV value.

**"Position gone" handling (live trading only):** if a sell order returns
`InsufficientPosition` or a similar error, the exchange has already closed the
contract. If the contract's settle time has passed, the code infers the
settlement value by comparing the live spot against the strike + direction:
`$1.00` if we won, `$0.00` if we lost. If the settle time has not yet passed
(external sale), the position is force-closed at `$0` as a conservative estimate.

**Soft re-entry gate:** after a `stop_loss` or `p_drop` exit, the same
`contract × model` combination requires **2× min_edge** to re-enter. This
prevents cascade re-entry into a falling market.

**Settlement correctness:** the settlement price is the OHLCV close of the
last 1-minute bar that *opens strictly before* the settle timestamp. Because
Gemini candles are stamped at their open time, the bar at `T−1m` closes at
exactly `T`.

---

## Output files

```
.data/gemini/sim_trades/
  trades_{YYYYMMDD}.csv       — one row per closed position
  edge_log_{YYYYMMDD}.csv     — all 14 model edges at first contract sighting
  performance_ledger.csv      — cumulative P&L / accuracy per model
                                (aggregated from all historical files,
                                 updated after every trade closes)
```

Use `--trades-dir` to write to a different directory for parallel experiments:

```bash
# Default experiment
python live_trading_sim.py

# Confidence-system experiment in a separate directory
python live_trading_sim.py --trades-dir .data/gemini/sim_trades_v2
```

---

## Per-model calibration windows

| Model | Window | Rationale |
|---|---|---|
| GBM | 12 h | Short = tracks intraday regime; 720 bars stable for rolling σ |
| EWMA | 24 h | Exp decay already down-weights old data; 1440 bars sufficient |
| GARCH | 48 h | MLE needs more bars for stable α, β estimates |
| Student-t | 24 h | ν identifiable from 1440 bars |
| Skewed-t | 24 h | Same as Student-t |
| Heston | 72 h | AR(1) on hourly realised variance needs many windows for κ, ξ |
| Hybrid-t | 24 h | EWMA vol (responsive) + t-tail from EWMA residuals |
| OU | 12 h | Short window for local mean-reversion; AR(1) on 1-min closes |
| Heston-EWMA | 72 h | Same as Heston; EWMA-calibrated long-run variance |
| GBM + Jump | 24 h | Needs ≥2–3 jumps visible for λ_J estimate |
| EWMA + Jump | 36 h | More history needed to identify jump frequency reliably |
| GARCH + Jump | 48 h | Same as GARCH base |
| Student-t + Jump | 36 h | Matches EWMA + Jump |
| Hybrid-t + Jump | 36 h | Matches EWMA + Jump |

All windows configurable via `config.toml [simulation.lookback]` or `--lb-*` CLI flags.
