"""
Live Prediction Market Trader — real money on Gemini Prediction Markets.

Uses the exact same 6-model entry/exit logic as live_trading_sim.py, but
places real orders via the Gemini trading API instead of just simulating.

Each bet: $20 per trade (configurable via --bet-dollars or GEMINI_BET_DOLLARS env var).
Orders use ``immediate-or-cancel`` — no dangling resting orders.

Usage:
    # Set credentials once (or export as env vars)
    export GEMINI_API_KEY="your_key_here"
    export GEMINI_API_SECRET="your_secret_here"

    # Run live trader (same flags as live_trading_sim.py)
    python live_trader.py
    python live_trader.py --bet-dollars 20 --min-edge 0.05
    python live_trader.py --sandbox      # paper trading, no real money

    # Run without auto-starting data collectors (if running them separately)
    python live_trader.py --no-collectors

All model/exit hyperparameters accept the same --lb-*, --profit-lock, etc.
flags as live_trading_sim.py.  Defaults come from config.toml.

IMPORTANT: Your Gemini API key must have NewOrder + CancelOrder permissions.
Generate keys at: https://exchange.gemini.com/settings/api
You must also accept the Prediction Markets Terms of Service in the web UI.
"""

from __future__ import annotations

import argparse
import os
import sys

import config_loader as cfg
from gemini_trader import GeminiTrader
from live_trading_sim import run


def main() -> None:
    _s  = cfg.section("simulation")
    _ex = cfg.section("simulation.exit")
    _lb = cfg.section("simulation.lookback")

    _lb_defaults = {
        "gbm":           _lb.get("gbm",           24.0),
        "ewma":          _lb.get("ewma",          48.0),
        "garch":         _lb.get("garch",         72.0),
        "stud":          _lb.get("stud",          48.0),
        "skt":           _lb.get("skt",           48.0),
        "heston":        _lb.get("heston",        96.0),
        "hybrid":        _lb.get("hybrid",        48.0),
        "ou":            _lb.get("ou",            12.0),
        "heston_ewma":   _lb.get("heston_ewma",   96.0),
        "gbm_jump":      _lb.get("gbm_jump",      24.0),
        "ewma_jump":     _lb.get("ewma_jump",     48.0),
        "garch_jump":    _lb.get("garch_jump",    72.0),
        "student_t_jump":_lb.get("student_t_jump",48.0),
        "hybrid_t_jump": _lb.get("hybrid_t_jump", 48.0),
    }

    parser = argparse.ArgumentParser(
        description="Live Gemini Prediction Markets trader (real orders)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Credentials ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--api-key",
        default=os.getenv("GEMINI_API_KEY", ""),
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--api-secret",
        default=os.getenv("GEMINI_API_SECRET", ""),
        help="Gemini API secret (or set GEMINI_API_SECRET env var)",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use sandbox exchange — paper trading, no real money",
    )
    parser.add_argument(
        "--bet-dollars",
        type=float,
        default=float(os.getenv("GEMINI_BET_DOLLARS", "20.0")),
        help="Target notional per trade (USD)",
    )

    # ── Model selection ────────────────────────────────────────────────────────
    _VALID_MODELS = {
        "gbm", "ewma", "garch", "student_t", "skewed_t", "heston",
        "hybrid_t", "ou", "heston_ewma",
        "gbm_jump", "ewma_jump", "garch_jump", "student_t_jump", "hybrid_t_jump",
    }
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=(
            "Restrict entry to one or more models. "
            f"Choices: {sorted(_VALID_MODELS)}. "
            "Default: all 14 models. "
            "Example: --model hybrid_t  or  --model hybrid_t hybrid_t_jump"
        ),
    )

    # ── Simulation hyperparameters (same as live_trading_sim.py) ─────────────
    parser.add_argument("--poll-sec",            type=int,   default=_s.get("poll_sec",            60))
    parser.add_argument("--min-edge",            type=float, default=_s.get("min_edge",            0.03))
    parser.add_argument("--max-hours-to-settle", type=float, default=_s.get("max_hours_to_settle", 1.5))
    parser.add_argument("--profit-lock",         type=float, default=_ex.get("profit_lock",        0.05))
    parser.add_argument("--stop-loss",           type=float, default=_ex.get("stop_loss",          0.50))
    parser.add_argument("--p-drop",              type=float, default=_ex.get("p_drop",             0.05))
    parser.add_argument("--edge-neg-thresh",     type=float, default=_ex.get("edge_neg_thresh",    0.02))
    parser.add_argument("--ewma-lambda",         type=float, default=_s.get("ewma_lambda",         0.94))
    parser.add_argument("--rho",                 type=float, default=_s.get("rho",                 -0.5))
    parser.add_argument("--lb-gbm",    type=float, default=_lb_defaults["gbm"])
    parser.add_argument("--lb-ewma",   type=float, default=_lb_defaults["ewma"])
    parser.add_argument("--lb-garch",  type=float, default=_lb_defaults["garch"])
    parser.add_argument("--lb-stud",   type=float, default=_lb_defaults["stud"])
    parser.add_argument("--lb-skt",    type=float, default=_lb_defaults["skt"])
    parser.add_argument("--lb-heston", type=float, default=_lb_defaults["heston"])
    parser.add_argument("--lb-hybrid",      type=float, default=_lb_defaults["hybrid"])
    parser.add_argument("--lb-ou",          type=float, default=_lb_defaults["ou"])
    parser.add_argument("--lb-heston-ewma", type=float, default=_lb_defaults["heston_ewma"])
    parser.add_argument("--lb-gbm-jump",    type=float, default=_lb_defaults["gbm_jump"])
    parser.add_argument("--lb-ewma-jump",   type=float, default=_lb_defaults["ewma_jump"])
    parser.add_argument("--lb-garch-jump",  type=float, default=_lb_defaults["garch_jump"])
    parser.add_argument("--lb-stud-jump",   type=float, default=_lb_defaults["student_t_jump"])
    parser.add_argument("--lb-hybrid-jump", type=float, default=_lb_defaults["hybrid_t_jump"])
    parser.add_argument("--vol-veto-mult",  type=float, default=_s.get("vol_veto_mult", 2.0))
    parser.add_argument("--no-collectors", action="store_true",
                        help="Do not auto-start data collectors")

    args = parser.parse_args()

    # ── Validate model selection ───────────────────────────────────────────────
    active_models = None
    if args.model is not None:
        bad = set(args.model) - _VALID_MODELS
        if bad:
            print(
                f"ERROR: Unknown model(s): {sorted(bad)}\n"
                f"  Valid choices: {sorted(_VALID_MODELS)}",
                file=sys.stderr,
            )
            sys.exit(1)
        active_models = set(args.model)

    # ── Validate credentials ───────────────────────────────────────────────────
    if not args.api_key or not args.api_secret:
        print(
            "ERROR: Gemini API credentials are required.\n"
            "  Set --api-key and --api-secret, or export environment variables:\n"
            "    export GEMINI_API_KEY=your_key\n"
            "    export GEMINI_API_SECRET=your_secret",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Build trader ───────────────────────────────────────────────────────────
    trader = GeminiTrader(
        api_key      = args.api_key,
        api_secret   = args.api_secret,
        sandbox      = args.sandbox,
        bet_dollars  = args.bet_dollars,
        cache_ttl_sec= 300,   # refresh symbol cache every 5 min; cache miss = instant refresh
    )

    mode        = "SANDBOX (paper trading)" if args.sandbox else "LIVE (real money)"
    model_str   = ", ".join(sorted(active_models)) if active_models else "all 9"
    print(
        f"\n{'='*60}\n"
        f"  Gemini Prediction Markets — LIVE TRADER\n"
        f"  Mode:    {mode}\n"
        f"  Models:  {model_str}\n"
        f"  Bet size: ${args.bet_dollars:.2f} per trade\n"
        f"{'='*60}\n"
    )

    # Pre-warm the symbol cache before the first poll
    trader.refresh_symbol_cache(force=True)
    print(f"  Symbol cache loaded: {len(trader._symbol_cache)} contracts\n")

    # ── Delegate to the simulation run loop with trader injected ──────────────
    run(
        poll_sec            = args.poll_sec,
        lookbacks           = {
            "gbm":           args.lb_gbm,
            "ewma":          args.lb_ewma,
            "garch":         args.lb_garch,
            "stud":          args.lb_stud,
            "skt":           args.lb_skt,
            "heston":        args.lb_heston,
            "hybrid":        args.lb_hybrid,
            "ou":            args.lb_ou,
            "heston_ewma":   args.lb_heston_ewma,
            "gbm_jump":      args.lb_gbm_jump,
            "ewma_jump":     args.lb_ewma_jump,
            "garch_jump":    args.lb_garch_jump,
            "student_t_jump":args.lb_stud_jump,
            "hybrid_t_jump": args.lb_hybrid_jump,
        },
        min_edge            = args.min_edge,
        max_hours_to_settle = args.max_hours_to_settle,
        profit_lock         = args.profit_lock,
        stop_loss           = args.stop_loss,
        p_drop              = args.p_drop,
        ewma_lambda         = args.ewma_lambda,
        rho                 = args.rho,
        edge_neg_thresh     = args.edge_neg_thresh,
        vol_veto_mult       = args.vol_veto_mult,
        no_collectors       = args.no_collectors,
        trader              = trader,
        active_models       = active_models,
    )


if __name__ == "__main__":
    main()
