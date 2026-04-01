"""
Polygon.io market data helpers.

Fetches current price + daily change (snapshot) and market cap (reference)
for a given ticker. Both calls are made lazily and cached per ticker.
"""

import time
from typing import Optional

import requests

_BASE = "https://api.polygon.io"
_DELAY = 0.12   # ~8 req/s; safe for all paid tiers and free tier


def get_market_data(ticker: str, api_key: str) -> dict:
    """
    Return a dict with:
        price       float | None   last trade / latest close
        change_pct  float | None   today's % change vs previous close
        market_cap  float | None   latest market cap in USD

    Returns an empty dict on complete failure. Individual keys may be None
    if the data is unavailable (e.g. non-US or non-equity ticker).
    """
    result: dict = {}

    # ── Snapshot: price + daily change ───────────────────────────────────
    try:
        resp = requests.get(
            f"{_BASE}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}",
            params={"apiKey": api_key},
            timeout=10,
        )
        time.sleep(_DELAY)
        if resp.ok:
            t = resp.json().get("ticker", {})
            # lastTrade available on paid plans; fall back to day close
            lt = t.get("lastTrade") or {}
            day = t.get("day") or {}
            result["price"]      = lt.get("p") or day.get("c")
            result["change_pct"] = t.get("todaysChangePerc")
    except Exception:
        pass

    # ── Reference: market cap ─────────────────────────────────────────────
    try:
        resp = requests.get(
            f"{_BASE}/v3/reference/tickers/{ticker}",
            params={"apiKey": api_key},
            timeout=10,
        )
        time.sleep(_DELAY)
        if resp.ok:
            ref = resp.json().get("results", {})
            result["market_cap"]  = ref.get("market_cap")
            result["description"] = ref.get("description") or ""
    except Exception:
        pass

    return result


# ── Formatting helpers ────────────────────────────────────────────────────

def fmt_price(price: Optional[float], change_pct: Optional[float]) -> str:
    """e.g. '$85.50 +0.8%' or '$85.50' if change unknown."""
    if price is None:
        return ""
    s = f"${price:,.2f}"
    if change_pct is not None:
        sign = "+" if change_pct >= 0 else ""
        s += f" {sign}{change_pct:.1f}%"
    return s


def fmt_mktcap(market_cap: Optional[float]) -> str:
    """e.g. '$2.85T', '$450B', '$12M'."""
    if not market_cap:
        return ""
    if market_cap >= 1e12:
        return f"${market_cap / 1e12:.2f}T"
    if market_cap >= 1e9:
        return f"${market_cap / 1e9:.1f}B"
    if market_cap >= 1e6:
        return f"${market_cap / 1e6:.0f}M"
    return f"${market_cap:,.0f}"


def change_color(change_pct: Optional[float]) -> str:
    """Green / red / grey depending on direction."""
    if change_pct is None:
        return "#6b7280"
    return "#16a34a" if change_pct >= 0 else "#dc2626"
