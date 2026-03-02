"""Simple strike construction helpers for hourly BTC contracts."""

from __future__ import annotations


def nearest_nice_number(price: float, spacing: float = 250.0) -> float:
    """Round to nearest strike grid (e.g., 250 USD).

    The result is clamped so that the lowest strike (base - spacing) remains
    strictly positive. This matters for low-price assets (e.g. SOL ~$85) where
    naive rounding gives base=0.
    """
    if price <= 0:
        raise ValueError("price must be positive")
    base = round(price / spacing) * spacing
    # Ensure low strike (base - spacing) > 0.
    return max(base, 2 * spacing)


def three_contract_strikes_from_anchor(anchor_price: float, spacing: float = 250.0) -> list[float]:
    """Build 3 strikes centered on nearest nice number, highest first.

    Example:
    anchor=66,620 -> base=66,500 -> [66,750, 66,500, 66,250]
    """
    base = nearest_nice_number(anchor_price, spacing=spacing)
    return [base + spacing, base, base - spacing]
