from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

import requests

from providers.symbols import to_provider_symbol


@dataclass(frozen=True)
class UniverseConfig:
    cache_dir: str = "cache"
    ttl_seconds: int = 24 * 60 * 60


def fetch_us_equity_symbols(cfg: UniverseConfig) -> List[str]:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "us_symbols.txt"

    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing Alpaca credentials for assets-based universe fetch")

    paper = str(os.getenv("ALPACA_PAPER") or "").strip().lower() in {"1", "true", "yes", "on"}
    url = ("https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets") + "/v2/assets"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "accept": "application/json",
        "User-Agent": os.environ.get("ORB_USER_AGENT", "orb-scanner/11.0 (+https://localhost)"),
    }
    params = {
        "status": "active",
        "asset_class": "us_equity",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Alpaca assets response shape: {type(payload).__name__}")

    symbols: Set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").strip().lower() != "active":
            continue
        if not bool(row.get("tradable")):
            continue
        if str(row.get("exchange") or "").strip().upper() == "OTC":
            continue
        sym = to_provider_symbol(str(row.get("symbol") or ""))
        if sym:
            symbols.add(sym)

    out = sorted(symbols)
    if not out:
        raise RuntimeError("Alpaca assets universe returned 0 active tradable symbols")

    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text("\n".join(out), encoding="utf-8")
    tmp.replace(cache_file)
    return out


# Backwards-compatible aliases (older scripts/tests)
def nasdaq_symbols(*, include_non_common: bool = False, cache_dir: str = "cache", ttl_seconds: int = 24*60*60) -> List[str]:
    """Return a live Alpaca assets-based US equity universe.

    The function name is kept for backwards compatibility with older project code,
    but the data source is Alpaca assets metadata so the scanner only sees tradable
    symbols supported by the broker/data provider.
    """
    cfg = UniverseConfig(cache_dir=cache_dir, ttl_seconds=ttl_seconds)
    return fetch_us_equity_symbols(cfg)

def get_nasdaq_symbols(*, include_non_common: bool = False, cache_dir: str = "cache", ttl_seconds: int = 24*60*60) -> List[str]:
    """Alias for nasdaq_symbols()."""
    return nasdaq_symbols(include_non_common=include_non_common, cache_dir=cache_dir, ttl_seconds=ttl_seconds)

__all__ = [
    "fetch_us_equity_symbols",
    "nasdaq_symbols",
    "get_nasdaq_symbols",
]
