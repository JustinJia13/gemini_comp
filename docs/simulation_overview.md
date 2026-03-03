# Simulation & Model Overview

*For teammates familiar with probability and basic finance.*

---

## What we're doing and why

### The market

We're trading on **Kalshi**, a regulated prediction market where contracts pay
exactly $1 if a condition is true at expiry and $0 if not.  Our contracts ask:
*"Will BTC / ETH / SOL close above price X at time T?"*  These are called
**binary options** — the payout is binary, but the price trades continuously
between $0 and $1 and can be read directly as a probability.

For example, if the market is pricing a "BTC above $95,000 at 3pm EST" contract
at $0.62, the crowd believes there's roughly a 62% chance that happens.

### The edge

Our job is to estimate that probability ourselves using a model.  If our model
says 68% but the market price is 62%, we have a **6-cent edge** on a $1
contract.  We buy when our edge exceeds a threshold (currently 3¢).  This is
the same logic a poker player uses — you don't need certainty, you just need to
be right *more often than the price implies*.

### The data

We pull **1-minute OHLCV** (open/high/low/close/volume) bars for BTC, ETH, and
SOL from Gemini's public API, updated every 60 seconds.  From those bars we
compute **log returns** (percentage price moves, log-scaled so they're
additive).  Everything the model sees is from *before* the evaluation time —
no look-ahead.

---

## The six models — a progression from simple to complex

Rather than picking one model and hoping it's right, we run six in parallel and
track which ones perform best over time.  Each model adds one more real-world
feature that the previous one ignores.

---

### 1. GBM — Geometric Brownian Motion *(the textbook baseline)*

The assumption: price moves are independent, normally distributed, and have
constant volatility.  We estimate σ (volatility) as the rolling standard
deviation of the last 24 hours of 1-minute returns, then use the Black-Scholes
formula to price the binary.

*What it gets wrong:* volatility is not constant — it clusters.  Quiet days are
followed by quiet days; chaotic days by more chaos.  And real returns have
fatter tails than a normal distribution.

---

### 2. EWMA — Exponentially Weighted Moving Average vol

Same structure as GBM, but instead of equal-weighting the last 24 hours, we
weight recent minutes more heavily using an exponential decay (λ = 0.94, the
RiskMetrics standard).  This means yesterday's volatility matters more than
last week's.

*What it adds:* adapts faster to volatility spikes than a simple rolling window.

---

### 3. GARCH(1,1) — Generalized Autoregressive Conditional Heteroskedasticity

GARCH explicitly models volatility as a process that mean-reverts over time.
Today's variance is a weighted mix of: a long-run average, yesterday's surprise
(squared return), and yesterday's estimated variance.  Parameters α, β, ω are
fit by maximum likelihood.

*What it adds:* captures **volatility clustering** — the well-documented
tendency for high-vol periods to persist.  GARCH needs more data to fit
reliably so we use a 72-hour window.

---

### 4. Student's t *(fat tails)*

Back to a simple rolling σ estimate, but we drop the normality assumption.  We
fit a **Student's t distribution** to the return series.  This distribution has
an extra parameter ν (degrees of freedom) that controls how fat the tails are.
Small ν = fat tails; as ν → ∞ it converges to a normal.

Crypto returns routinely show ν in the range of 3–6, meaning extreme moves are
far more common than any normal distribution would predict.  The binary
probability is computed via Monte Carlo (20,000 simulated paths).

*What it adds:* properly accounts for the higher probability of large sudden
moves.

---

### 5. Skewed Student's t *(fat tails + asymmetry)*

The regular Student's t is symmetric — it treats crashes and rallies as equally
likely for a given σ.  In practice, crypto tends to crash faster than it rallies
(negative skew).  The **Fernández-Steel skewed-t** adds a parameter γ that
tilts the distribution.

We estimate the tilt by regressing changes in rolling volatility against lagged
returns (via elastic net).  The intuition: when price fell in the last hour, did
volatility rise?  If yes (it usually does), the distribution skews left.  We
blend this estimate 50/50 with a −0.5 prior.

*What it adds:* asymmetric tail risk — the model understands that down-moves
and up-moves don't behave identically.

---

### 6. Heston Stochastic Volatility *(volatility itself is random)*

All models above treat volatility as something we estimate once and then hold
fixed for the horizon.  The Heston model says volatility is itself a random
process that mean-reverts:

```
dS/S = √V · dW₁          (price)
dV   = κ(θ − V)dt + ξ√V · dW₂   (variance)
dW₁ · dW₂ = ρ · dt       (price and vol are correlated)
```

Where:
- **θ** — long-run average variance (where vol mean-reverts to)
- **κ** — speed of mean-reversion (how fast it snaps back)
- **ξ** — vol-of-vol (how much variance itself jumps around)
- **ρ** — correlation between price moves and vol moves (typically negative —
  the "leverage effect")

We calibrate these from historical 1-minute returns using hourly realized
variance windows (AR(1) for κ and ξ).  Pricing is done via Monte Carlo with a
numerically stable discretization scheme.  Because Heston needs many hourly
windows to calibrate accurately, it uses a 96-hour lookback.

*What it adds:* the possibility that volatility will be very different at expiry
than it is right now — important for contracts more than 30 minutes out.

---

## Entry and exit logic

**Entry:** we buy when any model shows edge > 3¢.  Each model trades
independently (up to one open position per contract per model at a time).

**Exit — whichever fires first:**

| Condition | Trigger | Rationale |
|---|---|---|
| **Profit lock** | bid rose 5¢ from entry | Lock in gain before the market reverses |
| **Stop loss** | bid fell 10¢ from entry | Cap downside; don't ride to zero |
| **P-drop** | model p(side) fell 5 pp from entry | New data made the bet worse; get out early |
| **Natural settlement** | contract expires | Look up final spot price and book result |

---

## Why six models instead of one?

1. **No single model is always right.** Markets cycle between regimes where
   different assumptions hold.
2. **Diversification of signal.** GBM might enter a trade that Heston avoids
   (because Heston sees elevated vol-of-vol risk) — that's useful information.
3. **Built-in model comparison.** The performance ledger gives live,
   out-of-sample evidence of which model's probability estimates are best
   calibrated on Kalshi specifically.  Over time we can weight toward winners
   or cut underperformers.

---

## Per-model calibration windows

Each model uses a different amount of historical return data:

| Model | Window | Why |
|---|---|---|
| GBM | 24 h | Short = reactive to intraday regime shifts |
| EWMA | 48 h | Old data already down-weighted by λ |
| GARCH | 72 h | MLE needs more bars for stable α, β |
| Student-t | 48 h | Tail shape stabilises around 2,880 bars |
| Skewed-t | 48 h | Same as Student-t |
| Heston | 96 h | AR(1) on hourly realised vars needs many windows for κ, ξ |

Windows are configurable in `config.toml` under `[simulation.lookback]` or via
`--lb-*` CLI flags.

---

## Output files

```
.data/gemini/sim_trades/
  trades_{YYYYMMDD}.csv      — one row per closed position
  edge_log_{YYYYMMDD}.csv    — all model edges at first contract sighting
  performance_ledger.csv     — cumulative P&L / accuracy per model
                               (updated after every trade closes,
                                aggregated from all historical files)
```
