"""
Pipeline configuration loader.

Reads config.toml from the same directory as this file.
Falls back gracefully if the file is missing or a key is absent.

Usage:
    import config_loader as cfg

    poll_sec = cfg.get("data_collector", "poll_sec", default=60)
    p_drop   = cfg.get("simulation.exit", "p_drop",  default=0.05)
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib          # Python 3.11+
except ImportError:
    import tomli as tomllib  # pip install tomli  (Python < 3.11)

_CONFIG_PATH = Path(__file__).parent / "config.toml"

# Cache so we only read the file once per process.
_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if not _CONFIG_PATH.exists():
        _cache = {}
        return _cache
    with _CONFIG_PATH.open("rb") as f:
        _cache = tomllib.load(f)
    return _cache


def get(section: str, key: str, default):
    """Return config[section][key], falling back to default.

    section may use dot-notation for nested tables, e.g. "simulation.exit".
    """
    data = _load()
    for part in section.split("."):
        if not isinstance(data, dict):
            return default
        data = data.get(part, {})
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


def section(name: str) -> dict:
    """Return a full section as a flat dict (dot-notation supported)."""
    data = _load()
    for part in name.split("."):
        if not isinstance(data, dict):
            return {}
        data = data.get(part, {})
    return data if isinstance(data, dict) else {}
