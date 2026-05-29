from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Set
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

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
    headers = {"User-Agent": os.environ.get("ORB_USER_AGENT", "orb-scanner/11.0 (+https://localhost)")}
    if requests is not None:
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return resp.text

    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _load_cached_json(path: Path, *, ttl_seconds: int) -> Any | None:
    try:
        if not path.exists():
            return None
        if ttl_seconds > 0 and (time.time() - path.stat().st_mtime) > ttl_seconds:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_cached_symbols(path: Path) -> list[str]:
    try:
        if not path.exists():
            return []
        rows = [
            to_provider_symbol(line.strip().upper())
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        return [sym for sym in rows if sym and _is_primary_common_equity(sym)]
    except Exception:
        return []


def _write_cached_json(path: Path, payload: Any) -> None:
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


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


def _fetch_alpaca_asset_metadata(*, timeout: int = 20) -> dict[str, dict[str, Any]]:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        return {}

    base_url = "https://paper-api.alpaca.markets" if str(os.getenv("ALPACA_PAPER") or "").strip().lower() in {"1", "true", "yes"} else "https://api.alpaca.markets"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "User-Agent": os.environ.get("ORB_USER_AGENT", "orb-scanner/11.0 (+https://localhost)"),
    }
    params = {"status": "active", "asset_class": "us_equity"}
    if requests is not None:
        resp = requests.get(
            f"{base_url}/v2/assets",
            timeout=timeout,
            headers=headers,
            params=params,
        )
        resp.raise_for_status()
        payload = resp.json()
    else:  # pragma: no cover
        req = Request(f"{base_url}/v2/assets?{urlencode(params)}", headers=headers)
        try:
            with urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"asset metadata request failed: {exc.code}") from exc
    if not isinstance(payload, list):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        symbol = to_provider_symbol(str(row.get("symbol") or "").strip().upper())
        if not symbol:
            continue
        out[symbol] = row
    return out


def _symbol_order_score(symbol: str, *, listing: dict[str, Any] | None, asset: dict[str, Any] | None) -> float:
    score = 0.0
    sym = str(symbol or "").upper()
    info = listing or {}
    meta = asset or {}

    exchange = str(meta.get("exchange") or info.get("exchange") or "").strip().upper()
    if exchange == "NASDAQ":
        score += 4.0
    elif exchange in {"NYSE", "ARCA"}:
        score += 3.0
    elif exchange:
        score += 1.5

    market_category = str(info.get("market_category") or "").strip().upper()
    if market_category == "Q":
        score += 3.0
    elif market_category in {"G", "S"}:
        score += 2.0

    if str(info.get("financial_status") or "").strip().upper() in {"", "N"}:
        score += 1.0
    if str(info.get("etf") or "").strip().upper() == "Y":
        score += 0.5

    if bool(meta.get("tradable")):
        score += 4.0
    if bool(meta.get("easy_to_borrow")):
        score += 3.5
    if bool(meta.get("shortable")):
        score += 2.0
    if bool(meta.get("fractionable")):
        score += 2.0

    attrs = {str(x).strip().lower() for x in (meta.get("attributes") or []) if str(x).strip()}
    if "has_options" in attrs:
        score += 2.0
    if "fractional_eh_enabled" in attrs:
        score += 0.5
    if "overnight_halted" in attrs:
        score -= 1.0

    try:
        mmr = float(meta.get("maintenance_margin_requirement"))
    except Exception:
        mmr = None
    if mmr is not None:
        if mmr <= 30.0:
            score += 1.5
        elif mmr >= 100.0:
            score -= 2.0

    if "." in sym or "-" in sym or "/" in sym or "$" in sym:
        score -= 4.0
    if len(sym) <= 4:
        score += 2.5
    elif len(sym) == 5:
        score += 1.0
    elif len(sym) >= 6:
        score -= 1.0
    if sym.endswith(("W", "WS", "U", "R", "RT")):
        score -= 6.0
    return score


def _rank_symbols(
    symbols: list[str],
    *,
    listing_meta: dict[str, dict[str, Any]] | None = None,
    asset_meta: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    listings = listing_meta or {}
    assets = asset_meta or {}

    # If Alpaca metadata is available, require tradable=True to drop ADRs,
    # shells, OTC names, and anything Alpaca can't stream intraday data for.
    if assets:
        symbols = [
            sym for sym in symbols
            if assets.get(sym, {}).get("tradable", True)  # pass-through if no metadata
        ]

    return sorted(
        symbols,
        key=lambda sym: (
            -_symbol_order_score(sym, listing=listings.get(sym), asset=assets.get(sym)),
            sym,
        ),
    )


def fetch_us_equity_symbols(cfg: UniverseConfig) -> List[str]:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "us_symbols.txt"
    asset_cache_file = cache_dir / "us_symbol_asset_meta.json"
    cached_symbols = _load_cached_symbols(cache_file)

    symbols: Set[str] = set()
    listing_meta: dict[str, dict[str, Any]] = {}

    try:
        for url in NASDAQ_LISTING_URLS:
            payload = _fetch_text(url)
            rows = csv.DictReader(payload.splitlines(), delimiter="|")
            for row in rows:
                raw_symbol = str(row.get("Symbol") or row.get("ACT Symbol") or "").strip().upper()
                test_issue = str(row.get("Test Issue") or "").strip().upper()
                etf_flag = str(row.get("ETF") or "").strip().upper()
                if not raw_symbol or test_issue == "Y":
                    continue
                if etf_flag == "Y":
                    continue
                sym = to_provider_symbol(raw_symbol)
                if not sym or not _is_primary_common_equity(sym):
                    continue
                listing_meta[sym] = {
                    "exchange": str(row.get("Exchange") or row.get("Listing Exchange") or "").strip().upper(),
                    "market_category": str(row.get("Market Category") or "").strip().upper(),
                    "financial_status": str(row.get("Financial Status") or "").strip().upper(),
                    "etf": etf_flag,
                }
                symbols.add(sym)
    except Exception:
        if cached_symbols:
            return cached_symbols
        raise

    asset_meta = _load_cached_json(asset_cache_file, ttl_seconds=cfg.ttl_seconds)
    if not isinstance(asset_meta, dict):
        asset_meta = {}
    if not asset_meta:
        try:
            asset_meta = _fetch_alpaca_asset_metadata()
        except Exception:
            asset_meta = {}
        if asset_meta:
            _write_cached_json(asset_cache_file, asset_meta)

    out = _rank_symbols(sorted(symbols), listing_meta=listing_meta, asset_meta=asset_meta)
    if not out:
        if cached_symbols:
            return cached_symbols
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
