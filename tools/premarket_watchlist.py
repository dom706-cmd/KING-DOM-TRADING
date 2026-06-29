#!/usr/bin/env python3
"""
Kingdom Pre-Market Watchlist Builder
======================================
Aggregates pre-market movers from multiple sources, enriches with Alpaca
snapshot data (price, gap %, PM volume, spread), scores by signal strength,
and optionally auto-populates the Kingdom desk watchlist.

Sources:
  alpaca      — snapshot scan of the full Kingdom universe (gap %, PM vol)
  finviz      — Finviz pre-market screener (gappers, news flags)
  stocktwits  — StockTwits trending symbols (community buzz / momentum)

A symbol appearing in multiple sources gets a higher composite score and
rises in the ranking. This surfaces names that have both technical and
community momentum simultaneously.

WHEN TO RUN: 8:00–9:15 AM ET. After pre-market volume has built up but
before you start paper_trade_monitor.py. Reload after 8:30am if there's
a major macro event (FOMC, CPI, NFP) — the tape can shift fast.

Usage:
    cd ~/kingdom
    .venv/bin/python tools/premarket_watchlist.py
    .venv/bin/python tools/premarket_watchlist.py --add
    .venv/bin/python tools/premarket_watchlist.py --sources finviz,stocktwits --add
    .venv/bin/python tools/premarket_watchlist.py --sources alpaca --min-gap 4.0
    .venv/bin/python tools/premarket_watchlist.py --symbols NVDA,TSLA,AMD --add

Required packages (install once):
    .venv/bin/pip install alpaca-py finvizfinance requests

Env vars:
    ALPACA_API_KEY / APCA_API_KEY_ID
    ALPACA_SECRET_KEY / APCA_API_SECRET_KEY
    KINGDOM_PORT    Kingdom app port (default 8050)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

KINGDOM_DIR = Path(__file__).resolve().parents[1]
KINGDOM_PORT = int(os.getenv("KINGDOM_PORT", "8050"))
KINGDOM_BASE = f"http://127.0.0.1:{KINGDOM_PORT}"

DEFAULT_SOURCES = "alpaca,finviz,stocktwits,whitehouse"


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: Alpaca snapshots — our universe, gap %, PM volume, spread
# ═══════════════════════════════════════════════════════════════════════════════

def _alpaca_client():
    try:
        from alpaca.data import StockHistoricalDataClient
    except ImportError:
        print("  [alpaca] ERROR: alpaca-py not installed. Run: .venv/bin/pip install alpaca-py")
        return None
    key = (os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
           or os.getenv("ALPACA_KEY_ID") or "")
    secret = (os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
              or os.getenv("ALPACA_API_SECRET") or "")
    if not key or not secret:
        print("  [alpaca] No credentials — set ALPACA_API_KEY + ALPACA_SECRET_KEY.")
        return None
    return StockHistoricalDataClient(key, secret)


def _fetch_alpaca_snapshots(symbols: list[str]) -> dict[str, Any]:
    try:
        from alpaca.data.requests import StockSnapshotRequest
    except ImportError:
        return {}

    client = _alpaca_client()
    if client is None:
        return {}

    results: dict[str, Any] = {}
    batch_size = 1000
    total = len(symbols)
    for i in range(0, total, batch_size):
        batch = symbols[i : i + batch_size]
        pct = int((i / total) * 100) if total > 0 else 100
        print(f"  [alpaca] Fetching snapshots {pct}%  ({i}/{total})…", end="\r")
        try:
            req = StockSnapshotRequest(symbol_or_symbols=batch)
            snaps = client.get_stock_snapshot(req)
            for sym, snap in snaps.items():
                results[sym] = snap
        except Exception as e:
            print(f"\n  [alpaca] batch {i}–{i+batch_size} error: {e}")
    print(f"  [alpaca] Snapshots done. Got {len(results):,}." + " " * 30)
    return results


def _fetch_prev_closes_batch(symbols: list[str]) -> dict[str, float]:
    """
    Fetch the most recent prior-day close for a list of symbols.
    Uses AlpacaProvider.download_daily_batch — the same code path as Generate Orders,
    so gap% in the scan will always agree with gap% in Generate Orders.
    Returns {symbol: prev_close}.
    """
    import datetime

    # Strip crypto / non-equity tickers
    clean = [s for s in symbols if s and '.' not in s and s.isalpha()]
    if not clean:
        return {}

    try:
        sys.path.insert(0, str(KINGDOM_DIR))
        from providers.alpaca_provider import AlpacaProvider
        provider = AlpacaProvider()
    except Exception:
        return {}

    try:
        import pytz
        ET = pytz.timezone('America/New_York')
    except ImportError:
        ET = datetime.timezone(datetime.timedelta(hours=-4))  # EDT fallback

    today = datetime.datetime.now(ET).date() if hasattr(ET, 'localize') else datetime.date.today()
    today_start = datetime.datetime.combine(today, datetime.time.min)
    if hasattr(ET, 'localize'):
        today_start = ET.localize(today_start)

    daily_map = provider.download_daily_batch(clean, period='5d')

    result: dict[str, float] = {}
    for sym, df in daily_map.items():
        try:
            if df is None or df.empty:
                continue
            idx = df.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                idx_et = idx.tz_convert(ET)
            else:
                idx_et = idx.tz_localize('UTC').tz_convert(ET)
            prior = df[idx_et < today_start]
            if prior.empty:
                continue
            close_col = 'Close' if 'Close' in prior.columns else 'close'
            result[sym.upper()] = float(prior[close_col].iloc[-1])
        except Exception:
            pass
    return result


def _alpaca_mover(sym: str, snap: Any, *, prev_close: float,
                  min_price: float, max_price: float,
                  min_gap_pct: float, min_vol: int) -> dict | None:
    try:
        if prev_close <= 0:
            return None

        latest_trade = getattr(snap, "latest_trade", None)
        latest_quote = getattr(snap, "latest_quote", None)
        daily_bar    = getattr(snap, "daily_bar", None)

        # Alpaca snapshot daily_bar.close during premarket = yesterday's close, same as
        # prev_close. Using it produces gap=0% for every symbol. Prefer latest_trade.price
        # (the actual premarket print) so the gap reflects real PM movement.
        if latest_trade and getattr(latest_trade, "price", None):
            current_price = float(latest_trade.price)
        elif latest_quote:
            ask = float(getattr(latest_quote, "ask_price", 0) or 0)
            bid = float(getattr(latest_quote, "bid_price", 0) or 0)
            if ask > 0 and bid > 0:
                current_price = (ask + bid) / 2.0
            elif ask > 0:
                current_price = ask
            else:
                return None
        else:
            return None

        if current_price < min_price or current_price > max_price:
            return None

        gap_pct = (current_price - prev_close) / prev_close * 100.0
        if abs(gap_pct) < min_gap_pct:
            return None

        # Use daily_bar.volume = cumulative session volume (true PM vol).
        # minute_bar.volume is just one 1-min bar and wildly understates activity.
        pm_vol = int(getattr(daily_bar, "volume", 0) or 0) if daily_bar else 0
        if pm_vol < min_vol:
            return None

        spread_pct = 0.0
        if latest_quote:
            ask = float(getattr(latest_quote, "ask_price", 0) or 0)
            bid = float(getattr(latest_quote, "bid_price", 0) or 0)
            if ask > 0 and bid > 0 and current_price > 0:
                spread_pct = (ask - bid) / current_price * 100.0

        return {
            "symbol": sym,
            "current_price": current_price,
            "prev_close": prev_close,
            "gap_pct": gap_pct,
            "pm_vol": pm_vol,
            "spread_pct": spread_pct,
            "sources": {"alpaca"},
        }
    except Exception:
        return None


def _run_alpaca_source(symbols_override: list[str], *,
                       min_price: float, max_price: float,
                       min_gap_pct: float, min_vol: int) -> dict[str, dict]:
    """Returns {symbol: mover_dict} for all symbols that pass alpaca filters."""
    if symbols_override:
        symbols = symbols_override
    else:
        print("  [alpaca] Loading Kingdom universe…")
        sys.path.insert(0, str(KINGDOM_DIR))
        try:
            from universe.nasdaq_symbols import fetch_us_equity_symbols, UniverseConfig
            symbols = fetch_us_equity_symbols(UniverseConfig())
            print(f"  [alpaca] Universe: {len(symbols):,} symbols")
        except Exception as e:
            print(f"  [alpaca] Could not load universe: {e}")
            return {}

    snaps = _fetch_alpaca_snapshots(symbols)

    # Pre-filter by price and PM volume before the expensive daily bars fetch.
    # This trims 12K symbols down to a small set of candidates.
    candidates: dict[str, Any] = {}
    for sym, snap in snaps.items():
        lt = getattr(snap, "latest_trade", None)
        lq = getattr(snap, "latest_quote", None)
        db = getattr(snap, "daily_bar", None)
        # Price: latest_trade.price = actual PM print; daily_bar.close in premarket = prev close
        if lt and getattr(lt, "price", None):
            price = float(lt.price)
        elif db and getattr(db, "close", None):
            price = float(db.close)
        elif lq:
            ask = float(getattr(lq, "ask_price", 0) or 0)
            bid = float(getattr(lq, "bid_price", 0) or 0)
            price = (ask + bid) / 2.0 if (ask > 0 and bid > 0) else 0.0
        else:
            continue
        if price < min_price or price > max_price:
            continue
        # Volume: daily_bar.volume = cumulative PM vol; minute_bar is one bar only
        pm_vol = int(getattr(db, "volume", 0) or 0) if db else 0
        if pm_vol < min_vol:
            continue
        candidates[sym] = snap

    # Fetch real prev_close via daily bars — only for the candidates that passed.
    prev_closes = _fetch_prev_closes_batch(list(candidates.keys())) if candidates else {}

    out: dict[str, dict] = {}
    for sym, snap in candidates.items():
        pc = prev_closes.get(sym, 0.0)
        m = _alpaca_mover(sym, snap, prev_close=pc,
                          min_price=min_price, max_price=max_price,
                          min_gap_pct=min_gap_pct, min_vol=min_vol)
        if m:
            out[sym] = m
    print(f"  [alpaca] {len(out)} movers passed filters.")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: Finviz — pre-market screener (gappers + news)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_finviz_source(*, min_gap_pct: float, direction: str = "both") -> set[str]:
    """
    Returns set of symbols that Finviz screener flags as pre-market gappers.
    Uses finvizfinance if available, falls back to direct screener scrape.

    direction: "both" (default) | "up" (longs only) | "down" (shorts only)
    """
    label = {"up": "gap-up", "down": "gap-down", "both": "gappers"}[direction]
    print(f"  [finviz] Pulling pre-market {label}…")
    symbols: set[str] = set()

    # Determine gap filter label
    # Finviz gap filters: Up 1%, Up 2%, Up 3%, Up 4%, Up 5%+, Down 1%, etc.
    if min_gap_pct >= 5.0:
        gap_up_filter = "ta_gap_u5"
        gap_dn_filter = "ta_gap_d5"
    elif min_gap_pct >= 4.0:
        gap_up_filter = "ta_gap_u4"
        gap_dn_filter = "ta_gap_d4"
    elif min_gap_pct >= 3.0:
        gap_up_filter = "ta_gap_u3"
        gap_dn_filter = "ta_gap_d3"
    elif min_gap_pct >= 2.0:
        gap_up_filter = "ta_gap_u2"
        gap_dn_filter = "ta_gap_d2"
    else:
        gap_up_filter = "ta_gap_u1"
        gap_dn_filter = "ta_gap_d1"

    filters_to_run = []
    if direction in ("both", "up"):
        filters_to_run.append(gap_up_filter)
    if direction in ("both", "down"):
        filters_to_run.append(gap_dn_filter)

    try:
        from finvizfinance.screener.overview import Overview

        for gap_filter in filters_to_run:
            try:
                screen = Overview()
                screen.set_filter(filters_dict={"Gap": _finviz_gap_label(gap_filter)})
                df = screen.screener_view(verbose=0)
                if df is not None and not df.empty:
                    for sym in df["Ticker"].dropna():
                        symbols.add(str(sym).upper().strip())
            except Exception as e:
                print(f"  [finviz] screener pass ({gap_filter}) error: {e}")

    except ImportError:
        # Fall back to raw screener URL scrape
        up_f = gap_up_filter if direction in ("both", "up") else None
        dn_f = gap_dn_filter if direction in ("both", "down") else None
        symbols = _finviz_scrape_fallback(up_f, dn_f)

    print(f"  [finviz] {len(symbols)} symbols flagged.")
    return symbols


def _finviz_gap_label(filter_code: str) -> str:
    """Map internal filter code to finvizfinance label."""
    mapping = {
        "ta_gap_u1": "Up 1%", "ta_gap_u2": "Up 2%", "ta_gap_u3": "Up 3%",
        "ta_gap_u4": "Up 4%", "ta_gap_u5": "Up 5%",
        "ta_gap_d1": "Down 1%", "ta_gap_d2": "Down 2%", "ta_gap_d3": "Down 3%",
        "ta_gap_d4": "Down 4%", "ta_gap_d5": "Down 5%",
    }
    return mapping.get(filter_code, "Up 3%")


def _finviz_scrape_fallback(gap_up: str | None, gap_dn: str | None) -> set[str]:
    """Scrape Finviz screener HTML directly (no finvizfinance required)."""
    symbols: set[str] = set()
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }

    for gf in filter(None, (gap_up, gap_dn)):
        url = f"https://finviz.com/screener.ashx?v=111&f={gf}&o=-gap&ft=4"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r:
                html = r.read().decode("utf-8", errors="replace")
            # Parse ticker cells: look for screener table ticker links
            import re
            # Finviz ticker links look like: /quote.ashx?t=AAPL
            tickers = re.findall(r'/quote\.ashx\?t=([A-Z]{1,5})"', html)
            for t in tickers:
                symbols.add(t.upper())
        except Exception as e:
            print(f"  [finviz] scrape fallback error for {gf}: {e}")
        time.sleep(0.5)  # polite

    return symbols


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: StockTwits — trending symbols (community buzz)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_stocktwits_source() -> set[str]:
    """
    Returns set of currently trending symbols on StockTwits.
    No API key required for the public trending endpoint.
    """
    print("  [stocktwits] Pulling trending symbols…")
    url = "https://api.stocktwits.com/api/2/trending/symbols.json"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    symbols: set[str] = set()
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        for item in (data.get("symbols") or []):
            sym = str(item.get("symbol") or "").upper().strip()
            if sym:
                symbols.add(sym)
        print(f"  [stocktwits] {len(symbols)} trending symbols.")
    except Exception as e:
        print(f"  [stocktwits] error: {e}")
    return symbols


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 4: White House — policy themes → affected symbols
# ═══════════════════════════════════════════════════════════════════════════════

# Keyword → sector → symbols map.
# Any keyword match in a WH headline flags all symbols in that theme.
# Keep keywords specific enough to avoid false positives.
_WH_THEME_MAP = [
    {
        "sector": "Tariffs/Trade",
        "keywords": ["tariff", "tariffs", "import duty", "trade war", "trade deal",
                     "trade deficit", "trade surplus", "reciprocal tax"],
        "symbols": ["XRT", "AMZN", "WMT", "TGT", "COST", "NKE", "X", "NUE", "STLD", "CLF", "AA"],
    },
    {
        "sector": "China/Trade",
        "keywords": ["china", "chinese", "beijing", "prc", "hong kong", "taiwan",
                     "huawei", "tiktok", "bytedance", "de-coupling"],
        "symbols": ["BABA", "JD", "PDD", "BIDU", "KWEB", "TSM", "NVDA", "QCOM", "INTC", "AMAT"],
    },
    {
        "sector": "Oil & Gas",
        "keywords": ["oil", "petroleum", "pipeline", "lng", "drilling", "fracking",
                     "offshore", "refinery", "opec", "crude"],
        "symbols": ["XOM", "CVX", "COP", "MPC", "VLO", "PSX", "OXY", "DVN", "HES", "XLE"],
    },
    {
        "sector": "Energy Policy",
        "keywords": ["energy policy", "energy production", "domestic energy",
                     "energy independence", "coal", "fossil fuel", "natural gas"],
        "symbols": ["XOM", "CVX", "COP", "BTU", "ARCH", "XLE", "AM", "KMI", "WMB"],
    },
    {
        "sector": "Renewables/EV",
        "keywords": ["solar", "wind power", "renewable", "clean energy",
                     "electric vehicle", " ev ", "battery storage", "green new"],
        "symbols": ["NEE", "FSLR", "ENPH", "SEDG", "RUN", "PLUG", "BE", "TSLA", "RIVN", "LCID"],
    },
    {
        "sector": "Defense",
        "keywords": ["defense spending", "military", "pentagon", "nato", "weapons",
                     "missile", "aircraft carrier", "cybersecurity", "national security",
                     "armed forces", "ukraine", "israel", "military aid"],
        "symbols": ["LMT", "RTX", "NOC", "GD", "BA", "LHX", "LDOS", "SAIC", "CACI", "PLTR"],
    },
    {
        "sector": "Healthcare/Pharma",
        "keywords": ["drug pricing", "pharmaceutical", "medicare", "medicaid",
                     "health insurance", "fda", "prescription drug", "biosimilar",
                     "affordable care", "opioid"],
        "symbols": ["UNH", "CVS", "CI", "HUM", "ELV", "PFE", "MRK", "JNJ",
                    "ABBV", "BMY", "LLY", "AMGN", "GILD", "MRNA"],
    },
    {
        "sector": "Semiconductors",
        "keywords": ["semiconductor", "chips act", "chip manufacturing", "microchip",
                     "silicon", "foundry", "advanced manufacturing"],
        "symbols": ["NVDA", "AMD", "INTC", "QCOM", "TSM", "AMAT", "KLAC", "LRCX", "MRVL", "AVGO", "MU"],
    },
    {
        "sector": "Tech/Antitrust",
        "keywords": ["antitrust", "big tech", "data privacy", "social media regulation",
                     "section 230", "tech regulation", "artificial intelligence regulation"],
        "symbols": ["MSFT", "GOOGL", "META", "AMZN", "AAPL", "SNAP", "PINS"],
    },
    {
        "sector": "Banking/Finance",
        "keywords": ["bank regulation", "financial regulation", "interest rate",
                     "federal reserve", "mortgage", "fdic", "consumer finance",
                     "dodd-frank", "capital requirement"],
        "symbols": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "KRE", "XLF"],
    },
    {
        "sector": "Crypto/Digital Assets",
        "keywords": ["crypto", "cryptocurrency", "bitcoin", "digital asset",
                     "blockchain", "stablecoin", "cbdc", "digital dollar"],
        "symbols": ["COIN", "MSTR", "RIOT", "MARA", "HUT", "CLSK", "IREN"],
    },
    {
        "sector": "Agriculture",
        "keywords": ["agriculture", "farm bill", "food security", "crop insurance",
                     "corn", "soybean", "wheat", "ethanol", "fertilizer", "usda"],
        "symbols": ["ADM", "BG", "MOS", "CF", "NTR", "DE", "AGCO"],
    },
    {
        "sector": "Housing/Infrastructure",
        "keywords": ["housing", "infrastructure", "construction", "lumber",
                     "mortgage rate", "affordable housing", "infrastructure bill"],
        "symbols": ["DHI", "LEN", "PHM", "TOL", "NVR", "HD", "LOW", "BLDR", "WY"],
    },
    {
        "sector": "Steel/Materials",
        "keywords": ["steel", "aluminum tariff", "metal", "mining", "iron ore",
                     "domestic steel", "section 232"],
        "symbols": ["X", "NUE", "STLD", "CLF", "AA", "FCX", "XLB"],
    },
    {
        "sector": "Airlines/Transport",
        "keywords": ["airline", "aviation", "air traffic", "shipping", "port",
                     "freight", "supply chain", "railroad"],
        "symbols": ["UAL", "DAL", "AAL", "LUV", "FDX", "UPS", "CSX", "NSC"],
    },
    {
        "sector": "Sanctions",
        "keywords": ["sanction", "sanctions", "embargo", "export ban",
                     "entity list", "blacklist", "restrict export"],
        "symbols": ["XOM", "CVX", "BA", "LMT", "RTX", "NVDA", "INTC"],
    },
]


def _run_whitehouse_source(*, max_age_hours: float = 48.0) -> dict[str, str]:
    """
    Fetch White House briefing room RSS feed, match headlines against the
    policy theme map, and return {symbol: 'SECTOR: headline'} for every
    affected symbol found in the last max_age_hours.

    No API key required — uses the public WH RSS feed.
    Best used the night before or early morning for overnight policy news.
    """
    import xml.etree.ElementTree as ET
    import email.utils
    from datetime import datetime, timezone, timedelta

    print(f"  [whitehouse] Fetching briefing room RSS (last {int(max_age_hours)}h)…")

    url = "https://www.whitehouse.gov/briefing-room/feed/"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; KingdomScanner/1.0)",
        "Accept": "application/rss+xml, application/xml, text/xml",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            xml_data = r.read()
    except Exception as e:
        print(f"  [whitehouse] RSS fetch failed: {e}")
        return {}

    try:
        root = ET.fromstring(xml_data)
    except Exception as e:
        print(f"  [whitehouse] XML parse error: {e}")
        return {}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    # Collect recent items
    recent_items: list[dict] = []
    for item in root.iter("item"):
        title_el = item.find("title")
        desc_el  = item.find("description")
        date_el  = item.find("pubDate")

        title = (title_el.text or "").strip() if title_el is not None else ""
        desc  = (desc_el.text  or "").strip() if desc_el  is not None else ""

        pub_date = None
        if date_el is not None and date_el.text:
            try:
                pub_date = email.utils.parsedate_to_datetime(date_el.text.strip())
            except Exception:
                pass

        if pub_date and pub_date < cutoff:
            continue

        recent_items.append({"title": title, "desc": desc})

    if not recent_items:
        print("  [whitehouse] No recent items in feed.")
        return {}

    print(f"  [whitehouse] {len(recent_items)} headline(s) within window.")

    # Match keywords → symbols, track which themes fired
    matched: dict[str, str] = {}   # symbol → "SECTOR: headline"
    fired_themes: list[str] = []

    for item in recent_items:
        text = (item["title"] + " " + item["desc"]).lower()
        for theme in _WH_THEME_MAP:
            if any(kw in text for kw in theme["keywords"]):
                note = f"WH/{theme['sector']}: {item['title'][:75]}"
                for sym in theme["symbols"]:
                    if sym not in matched:
                        matched[sym] = note
                fired_themes.append(f"    [{theme['sector']}] {item['title'][:70]}")

    if fired_themes:
        for line in dict.fromkeys(fired_themes):   # deduplicate, preserve order
            print(line)

    print(f"  [whitehouse] {len(matched)} symbol(s) flagged.")
    return matched


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 5: Twitter/X (placeholder — requires paid API access)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_twitter_source() -> set[str]:
    """
    Placeholder — Twitter/X requires paid API access ($100/mo+ basic tier).
    To enable: set TWITTER_BEARER_TOKEN in your environment and replace this stub
    with a call to the v2 search/recent endpoint filtered by $TICKER cashtags.
    """
    bearer = os.getenv("TWITTER_BEARER_TOKEN", "")
    if not bearer:
        print("  [twitter] Skipped — no TWITTER_BEARER_TOKEN set.")
        return set()

    print("  [twitter] Querying cashtag mentions…")
    # TODO: implement v2 search when you have a bearer token
    # GET https://api.twitter.com/2/tweets/search/recent?query=%24SPY&max_results=100
    print("  [twitter] Twitter integration not yet implemented — add bearer token logic here.")
    return set()


# ═══════════════════════════════════════════════════════════════════════════════
# Enrichment — fill in price data for symbols found by non-Alpaca sources
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_with_alpaca(symbols: list[str], *,
                        existing: dict[str, dict]) -> dict[str, dict]:
    """
    For symbols not already in the alpaca mover dict, fetch snapshots and add
    basic price data. Used to enrich Finviz/StockTwits symbols.
    """
    need = [s for s in symbols if s not in existing]
    if not need:
        return {}

    print(f"  [enrich] Fetching Alpaca data for {len(need)} external symbols…")
    snaps = _fetch_alpaca_snapshots(need)
    prev_closes = _fetch_prev_closes_batch(list(snaps.keys()))
    enriched: dict[str, dict] = {}
    for sym, snap in snaps.items():
        try:
            prev_close = prev_closes.get(sym, 0.0)
            latest_trade = getattr(snap, "latest_trade", None)
            latest_quote = getattr(snap, "latest_quote", None)
            daily_bar    = getattr(snap, "daily_bar", None)

            # In premarket, daily_bar.close is yesterday's close — same as prev_close.
            # Prefer latest_trade.price (the actual premarket print) to get a real gap.
            if latest_trade and getattr(latest_trade, "price", None):
                current_price = float(latest_trade.price)
            elif latest_quote:
                ask = float(getattr(latest_quote, "ask_price", 0) or 0)
                bid = float(getattr(latest_quote, "bid_price", 0) or 0)
                current_price = (ask + bid) / 2.0 if (ask > 0 and bid > 0) else 0.0
            else:
                current_price = 0.0

            gap_pct = ((current_price - prev_close) / prev_close * 100.0
                       if prev_close > 0 else 0.0)

            pm_vol = int(getattr(daily_bar, "volume", 0) or 0) if daily_bar else 0

            spread_pct = 0.0
            if latest_quote and current_price > 0:
                ask = float(getattr(latest_quote, "ask_price", 0) or 0)
                bid = float(getattr(latest_quote, "bid_price", 0) or 0)
                if ask > 0 and bid > 0:
                    spread_pct = (ask - bid) / current_price * 100.0

            enriched[sym] = {
                "symbol": sym,
                "current_price": current_price,
                "prev_close": prev_close,
                "gap_pct": gap_pct,
                "pm_vol": pm_vol,
                "spread_pct": spread_pct,
                "sources": set(),
            }
        except Exception:
            pass
    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring and ranking
# ═══════════════════════════════════════════════════════════════════════════════

def _score(mover: dict) -> float:
    """
    Composite score tuned for ORB setups.

    Priority:
      1. Multi-source signals — each source adds +100 (strongest signal)
      2. Gap strength — capped at 30 pts.  A 30 % gap is exceptional;
         a 4000 % gap is a low-float pump and adds no extra ORB edge.
      3. Quality bonus — $5–$50 price with 5–30 % gap is the ORB sweet
         spot: PM range is well-defined and entry/stop levels are clean.
      4. Volume quality — up to +5 pts for deep PM liquidity (≥500K).
      5. Spread penalty — wide bid/ask (> 1 %) signals thin, illiquid
         tape where fills and stops are unreliable.
      6. Pump penalty — gap > 100 % almost always means a low-float
         overnight catalyst with no tradeable PM range at the open.
    """
    source_weight = len(mover.get("sources") or []) * 100.0

    gap_pct = abs(mover.get("gap_pct") or 0.0)
    price   = mover.get("current_price") or 0.0
    pm_vol  = mover.get("pm_vol") or 0
    spread  = mover.get("spread_pct") or 0.0

    # Cap gap contribution — extreme gappers shouldn't dominate ranking
    gap_weight = min(gap_pct, 30.0)

    # Pump penalty: gap > 100 % = likely low-float overnight explosion
    if gap_pct > 200:
        gap_weight -= 20.0
    elif gap_pct > 100:
        gap_weight -= 10.0

    # ORB quality bonus: moderate gap, mid-cap price range
    quality_bonus = 10.0 if (5.0 <= price <= 50.0 and 5.0 <= gap_pct <= 30.0) else 0.0

    # Volume quality: up to +5 pts for 500K+ PM volume
    vol_bonus = min(pm_vol / 100_000, 5.0)

    # Spread penalty: > 1 % spread signals illiquid/pump tape
    spread_penalty = min(spread * 2.0, 15.0) if spread > 1.0 else 0.0

    return source_weight + gap_weight + quality_bonus + vol_bonus - spread_penalty


def _score_short(mover: dict) -> float:
    """
    Scoring function optimised for short ORB setups (gap-down stocks).

    Rewards
    -------
    - Multiple-source confirmation (100 pts / source — same as long)
    - Gap-down in the ORB sweet spot: −5 % to −20 %
    - Price ≥ $10 (borrowable, liquid enough to short intraday)
    - High PM volume (real distribution, not just thin tape)
    - Tight spread (need clean fills to cover)

    Penalises
    ---------
    - Gap-down > 25 %: borrow cost spikes; news-driven; stop is too far
    - Price < $10: hard-to-borrow territory; skip
    - Price < $5: almost certainly un-borrowable at retail; disqualify
    - Wide spread (> 0.5 %): bad for short entries and covers
    """
    source_weight = len(mover.get("sources") or []) * 100.0

    gap_pct = mover.get("gap_pct") or 0.0
    price   = mover.get("current_price") or 0.0
    pm_vol  = mover.get("pm_vol") or 0
    spread  = mover.get("spread_pct") or 0.0

    # Disqualify gap-up stocks from the short ranking
    if gap_pct >= 0:
        return -999.0

    abs_gap = abs(gap_pct)

    # Cap gap contribution at 25 pts — extreme gap-downs are traps, not setups
    gap_weight = min(abs_gap, 25.0)

    # Penalty for very large gap-downs (borrow unavailable / halted opens)
    if abs_gap > 30:
        gap_weight -= 15.0
    elif abs_gap > 20:
        gap_weight -= 8.0

    # ORB quality bonus: moderate gap, borrowable price range
    quality_bonus = 10.0 if (price >= 10.0 and 5.0 <= abs_gap <= 20.0) else 0.0

    # Volume quality: up to +5 pts for 500 K+ PM volume
    vol_bonus = min(pm_vol / 100_000, 5.0)

    # Spread penalty — tighter threshold for shorts (covers are expensive on wide tape)
    spread_penalty = min(spread * 3.0, 20.0) if spread > 0.5 else 0.0

    # Penny-stock penalty: hard-to-borrow / un-borrowable at retail
    if price < 5.0:
        penny_penalty = 30.0
    elif price < 10.0:
        penny_penalty = 15.0
    else:
        penny_penalty = 0.0

    return source_weight + gap_weight + quality_bonus + vol_bonus - spread_penalty - penny_penalty


# ═══════════════════════════════════════════════════════════════════════════════
# Kingdom desk watchlist API
# ═══════════════════════════════════════════════════════════════════════════════

def _post(path: str, payload: dict) -> dict | None:
    url = KINGDOM_BASE + path
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  [watchlist] POST {path} failed: {e}")
        return None


def _add_to_desk_watchlist(mover: dict) -> bool:
    sym = mover["symbol"]
    gap = mover.get("gap_pct") or 0.0
    vol = mover.get("pm_vol") or 0
    spread = mover.get("spread_pct") or 0.0
    sources = sorted(mover.get("sources") or [])
    side = "long" if gap > 0 else "short"
    # Normalize composite score to 0-1 for monitor seeding (raw score: 0-245 typical range)
    raw_score = _score(mover) if gap > 0 else _score_short(mover)
    scan_score_norm = round(min(max(raw_score / 245.0, 0.0), 1.0), 4)
    notes = (
        f"PM gap {gap:+.1f}%  vol={vol:,}  spread={spread:.2f}%"
        + (f"  src={','.join(sources)}" if sources else "")
        + f"  scan_score={scan_score_norm}"
    )
    # Append WH policy context if present
    wh_note = mover.get("wh_headline")
    if wh_note:
        notes += f"  |  {wh_note}"
    from datetime import datetime, timezone
    payload = {
        "symbol": sym,
        "side": side,
        "trigger_price": None,
        "stop_price": None,
        "target_price": None,
        "notes": notes,
        "session_date": datetime.now(timezone.utc).strftime('%Y-%m-%d'),
    }
    result = _post("/api/desk_watchlist/set", payload)
    return result is not None and result.get("ok") is True


# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

SOURCE_ICONS = {"alpaca": "A", "finviz": "F", "stocktwits": "S", "whitehouse": "W", "twitter": "T"}


def _source_badge(sources: set[str]) -> str:
    return "".join(SOURCE_ICONS.get(s, "?") for s in sorted(sources))


def _print_table(movers: list[dict], *, top_n: int, will_add: bool, mode: str = "both") -> None:
    if not movers:
        print("\n  No pre-market movers found matching your filters.\n")
        return

    header = (f"{'#':>3}  {'SYM':<7}  {'PRICE':>7}  {'GAP%':>7}  "
              f"{'PM VOL':>10}  {'SPREAD':>7}  {'SRC':<4}  {'SIDE':<5}")
    divider = "─" * len(header)
    print()
    print("=" * len(header))
    if mode == "shorts":
        print("  PRE-MARKET SHORT CANDIDATES  [A=Alpaca  F=Finviz  S=StockTwits  W=WhiteHouse]")
        print("  Ranked by: multi-source signals › short quality ($10+, -5% to -20%) › spread")
    elif mode == "longs":
        print("  PRE-MARKET LONG CANDIDATES  [A=Alpaca  F=Finviz  S=StockTwits  W=WhiteHouse]")
        print("  Ranked by: multi-source signals › ORB quality › gap% (capped 30%) › volume")
    else:
        print("  PRE-MARKET MOVERS  [A=Alpaca  F=Finviz  S=StockTwits  W=WhiteHouse]")
        print("  Ranked by: multi-source signals › ORB quality › gap% (capped 30%) › volume")
    print("=" * len(header))
    print(header)
    print(divider)

    prev_src_count = None
    for i, m in enumerate(movers, 1):
        src_count = len(m.get("sources") or [])
        if prev_src_count is not None and src_count < prev_src_count:
            print(divider)  # visual break between signal tiers
        prev_src_count = src_count

        side = "LONG" if (m.get("gap_pct") or 0) > 0 else "SHORT"
        flag = " ◀" if will_add and i <= top_n else ""
        badge = _source_badge(m.get("sources") or set())
        price = m.get("current_price") or 0.0
        gap = m.get("gap_pct") or 0.0
        vol = m.get("pm_vol") or 0
        spread = m.get("spread_pct") or 0.0
        print(
            f"{i:>3}  {m['symbol']:<7}  {price:>7.2f}  {gap:>+7.2f}%  "
            f"{vol:>10,}  {spread:>6.2f}%  {badge:<4}  {side:<5}{flag}"
        )

    print("=" * len(header))
    print()
    if will_add:
        print(f"  ◀ marks the top {min(top_n, len(movers))} symbols being added to desk watchlist.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Kingdom pre-market watchlist builder")
    parser.add_argument("--sources",   type=str,   default=DEFAULT_SOURCES,
                        help=f"Comma-separated sources: alpaca,finviz,stocktwits,whitehouse,twitter (default: {DEFAULT_SOURCES})")
    parser.add_argument("--min-price", type=float, default=2.0,    help="Min stock price (default 2.0)")
    parser.add_argument("--max-price", type=float, default=100.0,  help="Max stock price (default 100.0)")
    parser.add_argument("--min-gap",   type=float, default=2.0,    help="Min abs gap %% (default 2.0)")
    parser.add_argument("--min-vol",   type=int,   default=50_000, help="Min PM volume for Alpaca filter (default 50000)")
    parser.add_argument("--top",       type=int,   default=15,     help="How many to add to desk watchlist (default 15)")
    parser.add_argument("--add",       action="store_true",        help="Auto-add top movers to Kingdom desk watchlist")
    parser.add_argument("--symbols",   type=str,   default="",     help="Comma-separated symbol override (skips universe fetch)")
    parser.add_argument("--mode",      type=str,   default="both",
                        choices=["both", "longs", "shorts"],
                        help="Filter direction: both (default), longs (gap-up only), shorts (gap-down only)")
    args = parser.parse_args()

    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    symbols_override = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    mode = args.mode

    # Shorts mode: raise min price floor to $10 unless user explicitly set it higher
    if mode == "shorts" and args.min_price < 10.0:
        args.min_price = 10.0

    print("\n[premarket-watchlist] Kingdom Pre-Market Watchlist Builder")
    print(f"  Mode     : {mode.upper()}")
    print(f"  Sources  : {', '.join(sorted(sources))}")
    print(f"  Filters  : price ${args.min_price}–${args.max_price} | gap ≥{args.min_gap}% | vol ≥{args.min_vol:,}")
    print(f"  Auto-add : {'YES — top ' + str(args.top) if args.add else 'no'}")
    print()

    # ── Run each source ────────────────────────────────────────────────────────
    movers: dict[str, dict] = {}  # symbol → mover dict

    if "alpaca" in sources:
        alpaca_movers = _run_alpaca_source(
            symbols_override,
            min_price=args.min_price, max_price=args.max_price,
            min_gap_pct=args.min_gap, min_vol=args.min_vol,
        )
        for sym, m in alpaca_movers.items():
            movers[sym] = m

    if "finviz" in sources:
        fv_direction = {"longs": "up", "shorts": "down", "both": "both"}[mode]
        fv_syms = _run_finviz_source(min_gap_pct=args.min_gap, direction=fv_direction)
        # Enrich any Finviz symbols we don't already have Alpaca data for
        enriched = _enrich_with_alpaca(list(fv_syms), existing=movers)
        for sym in fv_syms:
            if sym in movers:
                movers[sym]["sources"].add("finviz")
            elif sym in enriched:
                enriched[sym]["sources"].add("finviz")
                movers[sym] = enriched[sym]
            else:
                # No Alpaca data — add stub so it still shows up
                movers[sym] = {"symbol": sym, "current_price": 0.0, "prev_close": 0.0,
                               "gap_pct": 0.0, "pm_vol": 0, "spread_pct": 0.0,
                               "sources": {"finviz"}}

    if "stocktwits" in sources:
        st_syms = _run_stocktwits_source()
        enriched = _enrich_with_alpaca(list(st_syms), existing=movers)
        for sym in st_syms:
            if sym in movers:
                movers[sym]["sources"].add("stocktwits")
            elif sym in enriched:
                enriched[sym]["sources"].add("stocktwits")
                movers[sym] = enriched[sym]
            else:
                movers[sym] = {"symbol": sym, "current_price": 0.0, "prev_close": 0.0,
                               "gap_pct": 0.0, "pm_vol": 0, "spread_pct": 0.0,
                               "sources": {"stocktwits"}}

    if "whitehouse" in sources:
        wh_matched = _run_whitehouse_source()
        enriched = _enrich_with_alpaca(list(wh_matched.keys()), existing=movers)
        for sym, headline in wh_matched.items():
            if sym in movers:
                movers[sym]["sources"].add("whitehouse")
                movers[sym].setdefault("wh_headline", headline)
            elif sym in enriched:
                enriched[sym]["sources"].add("whitehouse")
                enriched[sym]["wh_headline"] = headline
                movers[sym] = enriched[sym]
            else:
                movers[sym] = {"symbol": sym, "current_price": 0.0, "prev_close": 0.0,
                               "gap_pct": 0.0, "pm_vol": 0, "spread_pct": 0.0,
                               "sources": {"whitehouse"}, "wh_headline": headline}

    if "twitter" in sources:
        tw_syms = _run_twitter_source()
        for sym in tw_syms:
            if sym in movers:
                movers[sym]["sources"].add("twitter")
            else:
                movers[sym] = {"symbol": sym, "current_price": 0.0, "prev_close": 0.0,
                               "gap_pct": 0.0, "pm_vol": 0, "spread_pct": 0.0,
                               "sources": {"twitter"}}

    # ── Filter by direction, then score and sort ───────────────────────────────
    candidates = list(movers.values())
    if mode == "longs":
        candidates = [m for m in candidates if (m.get("gap_pct") or 0.0) > 0]
        ranked = sorted(candidates, key=_score, reverse=True)
    elif mode == "shorts":
        candidates = [m for m in candidates if (m.get("gap_pct") or 0.0) < 0]
        ranked = sorted(candidates, key=_score_short, reverse=True)
    else:
        ranked = sorted(candidates, key=_score, reverse=True)

    # ── Display ────────────────────────────────────────────────────────────────
    _print_table(ranked, top_n=args.top, will_add=args.add, mode=mode)

    if not ranked:
        return

    # ── Auto-add to Kingdom desk watchlist ────────────────────────────────────
    if args.add:
        top = ranked[: args.top]
        print(f"  Adding top {len(top)} to Kingdom desk watchlist at {KINGDOM_BASE}…")
        added = 0
        for m in top:
            ok = _add_to_desk_watchlist(m)
            status = "✓" if ok else "✗  (is Kingdom running?)"
            print(f"    {status}  {m['symbol']}  {(m.get('gap_pct') or 0):+.1f}%  "
                  f"src={_source_badge(m.get('sources') or set())}")
            if ok:
                added += 1
        print(f"\n  Done. {added}/{len(top)} added.")
        print("  Open Kingdom in browser to fill in trigger/stop/target levels.\n")
    else:
        print("  Tip: run with --add to auto-populate the Kingdom desk watchlist.")
        print()


if __name__ == "__main__":
    main()
