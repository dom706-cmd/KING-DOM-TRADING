from __future__ import annotations

import csv
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


NASDAQ_LISTING_URLS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
]


def _fetch_text(url: str, timeout: int = 30) -> str:
    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": os.environ.get("ORB_USER_AGENT", "orb-scanner/11.0 (+https://localhost)")},
    )
    resp.raise_for_status()
    return resp.text


def _is_primary_common_equity(symbol: str) -> bool:
    # Keep the live universe clean. Pro scanners usually exclude warrants, rights,
    # units, preferreds, acquisition vehicles, and other low-signal junk from a
    # default common-stock scan universe.
    if not symbol:
        return False
    bad_suffixes = (
        ".U", ".W", ".WS", ".R", ".RT", ".P", ".PR", ".WD", ".WT",
    )
    return not any(symbol.endswith(sfx) for sfx in bad_suffixes)


def fetch_us_equity_symbols(cfg: UniverseConfig) -> List[str]:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "us_symbols.txt"

    symbols: Set[str] = set()

    for url in NASDAQ_LISTING_URLS:
        payload = _fetch_text(url)
        rows = csv.DictReader(payload.splitlines(), delimiter="|")
        for row in rows:
            raw_symbol = str(row.get("Symbol") or row.get("ACT Symbol") or "").strip().upper()
            test_issue = str(row.get("Test Issue") or "").strip().upper()
            etf_flag = str(row.get("ETF") or "").strip().upper()
            if not raw_symbol or test_issue == "Y":
                continue
            sym = to_provider_symbol(raw_symbol)
            if not sym or not _is_primary_common_equity(sym):
                continue
            # keep ETFs; top scanners include them by default
            _ = etf_flag
            symbols.add(sym)

    out = sorted(symbols)
    if not out:
        raise RuntimeError("NASDAQ symbol directory returned 0 symbols")

    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text("\n".join(out), encoding="utf-8")
    tmp.replace(cache_file)
    return out


# Backwards-compatible aliases (older scripts/tests)
def nasdaq_symbols(*, include_non_common: bool = False, cache_dir: str = "cache", ttl_seconds: int = 24*60*60) -> List[str]:
    cfg = UniverseConfig(cache_dir=cache_dir, ttl_seconds=ttl_seconds)
    syms = fetch_us_equity_symbols(cfg)
    if include_non_common:
        return syms
    return [s for s in syms if _is_primary_common_equity(s)]


def get_nasdaq_symbols(*, include_non_common: bool = False, cache_dir: str = "cache", ttl_seconds: int = 24*60*60) -> List[str]:
    return nasdaq_symbols(include_non_common=include_non_common, cache_dir=cache_dir, ttl_seconds=ttl_seconds)


__all__ = [
    "fetch_us_equity_symbols",
    "nasdaq_symbols",
    "get_nasdaq_symbols",
]
