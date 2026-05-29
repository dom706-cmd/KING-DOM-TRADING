"""
Morning prep data module.
Aggregates: earnings calendar, economic events, 52-week high breakouts, FDA newsflow.
All functions return plain dicts/lists — no side effects.
"""

from __future__ import annotations
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytz
_ET = pytz.timezone("America/New_York")

log = logging.getLogger(__name__)

# ── FOMC 2026 dates (statement day = second day of each meeting) ──────────────
_FOMC_2026 = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 10), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 11, 4), date(2026, 12, 16),
]

# Approximate CPI release dates 2026 (BLS ~2nd week of month)
_CPI_2026 = [
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 10), date(2026, 5, 13), date(2026, 6, 10),
    date(2026, 7,  8), date(2026, 8, 12), date(2026, 9,  9),
    date(2026, 10, 7), date(2026, 11, 12), date(2026, 12, 9),
]

# Approximate NFP/Jobs release dates 2026 (BLS first Friday of month)
_NFP_2026 = [
    date(2026, 1,  9), date(2026, 2,  6), date(2026, 3,  6),
    date(2026, 4,  3), date(2026, 5,  1), date(2026, 6,  5),
    date(2026, 7,  2), date(2026, 8,  7), date(2026, 9,  4),
    date(2026, 10, 2), date(2026, 11, 6), date(2026, 12, 4),
]


# ── Earnings calendar ─────────────────────────────────────────────────────────

def fetch_earnings_calendar(
    lookback_days: int = 0,
    forward_days: int = 3,
    max_price: float = 30.0,
) -> list[dict[str, Any]]:
    """
    Fetch upcoming earnings from NASDAQ public API.
    Returns list of dicts sorted by date ascending.
    max_price=0 means no price filter.
    """
    import requests

    today_et = datetime.now(_ET).date()
    results: list[dict] = []
    seen: set[str] = set()

    for delta in range(-lookback_days, forward_days + 1):
        target = today_et + timedelta(days=delta)
        url = f"https://api.nasdaq.com/api/calendar/earnings?date={target.isoformat()}"
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            rows = (data.get("data") or {}).get("rows") or []
            for row in rows:
                sym = (row.get("symbol") or "").strip().upper()
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                mktcap_raw = row.get("marketCap") or ""
                time_raw = (row.get("time") or "").lower()
                eps = row.get("epsForecast") or ""
                results.append({
                    "symbol":        sym,
                    "name":          row.get("name") or "",
                    "date":          target.isoformat(),
                    "days_away":     delta,
                    "time":          "pre-market" if "pre" in time_raw else ("after-close" if "after" in time_raw else "unknown"),
                    "eps_forecast":  eps,
                    "market_cap":    mktcap_raw,
                })
        except Exception as e:
            log.warning("earnings_calendar_fail date=%s: %s", target, e)

    # Price-filter: fetch snapshots for all symbols and filter by last price
    if max_price > 0 and results:
        try:
            from providers.alpaca_provider import AlpacaProvider
            provider = AlpacaProvider()
            syms = [r["symbol"] for r in results]
            # Batch snapshots
            BATCH = 500
            price_map: dict[str, float] = {}
            prev_close_map: dict[str, float] = {}
            for i in range(0, len(syms), BATCH):
                batch = syms[i:i + BATCH]
                try:
                    snaps = provider.get_snapshots(batch, feed="sip", timeout_s=12.0)
                    for s in batch:
                        snap = snaps.get(s) or {}
                        trade = snap.get("latest_trade") or {}
                        p = trade.get("price") or (snap.get("daily_bar") or {}).get("close") or 0
                        if p:
                            price_map[s] = float(p)
                        pb = snap.get("prev_daily_bar") or snap.get("previous_daily_bar") or {}
                        pc = pb.get("close") or 0
                        if pc:
                            prev_close_map[s] = float(pc)
                except Exception:
                    pass
            for r in results:
                r["last_price"] = price_map.get(r["symbol"])
                price = r.get("last_price") or 0
                prev_c = prev_close_map.get(r["symbol"]) or 0
                if price and prev_c:
                    gap_pct = (price - prev_c) / prev_c * 100.0
                    r["gap_pct"] = round(gap_pct, 2)
                    r["side"] = "long" if gap_pct >= 0 else "short"
                else:
                    r["gap_pct"] = None
                    r["side"] = None
            if max_price > 0:
                results = [r for r in results if r.get("last_price") is None or r["last_price"] <= max_price]
        except Exception as e:
            log.warning("earnings_price_filter_fail: %s", e)

    results.sort(key=lambda r: (r["days_away"], r["symbol"]))
    return results


# ── Economic events ───────────────────────────────────────────────────────────

def fetch_economic_events(forward_days: int = 14) -> list[dict[str, Any]]:
    """Return upcoming high-impact economic events within forward_days."""
    today = datetime.now(_ET).date()
    cutoff = today + timedelta(days=forward_days)
    events: list[dict] = []

    for d in _FOMC_2026:
        if today <= d <= cutoff:
            days = (d - today).days
            events.append({"date": d.isoformat(), "days_away": days, "event": "FOMC Rate Decision", "impact": "HIGH", "category": "fed"})

    for d in _CPI_2026:
        if today <= d <= cutoff:
            days = (d - today).days
            events.append({"date": d.isoformat(), "days_away": days, "event": "CPI Release", "impact": "HIGH", "category": "inflation"})

    for d in _NFP_2026:
        if today <= d <= cutoff:
            days = (d - today).days
            events.append({"date": d.isoformat(), "days_away": days, "event": "Non-Farm Payrolls", "impact": "HIGH", "category": "jobs"})

    events.sort(key=lambda e: e["days_away"])
    return events


# ── 52-week high breakouts ────────────────────────────────────────────────────

def scan_52wk_highs(
    provider=None,
    min_rvol: float = 1.5,
    max_price: float = 30.0,
    min_price: float = 1.0,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """
    Find stocks trading near their 52-week high with above-average volume.
    Uses Alpaca most-actives + market-movers as universe, yfinance for 52wk data.
    """
    import yfinance as yf
    from scanner.orb import _screener_most_actives, _screener_market_movers

    if provider is None:
        from providers.alpaca_provider import AlpacaProvider
        provider = AlpacaProvider()

    # Build universe from Alpaca screeners
    actives = _screener_most_actives(provider, top=50)
    gainers, losers = _screener_market_movers(provider, top=50)
    seen: set[str] = set()
    universe: list[str] = []
    for s in actives + gainers + losers:
        if s not in seen:
            universe.append(s)
            seen.add(s)

    if not universe:
        return []

    # Get live snapshots for price + RVOL
    snap_map: dict[str, dict] = {}
    try:
        BATCH = 500
        for i in range(0, len(universe), BATCH):
            snaps = provider.get_snapshots(universe[i:i + BATCH], feed="sip", timeout_s=15.0)
            snap_map.update(snaps or {})
    except Exception as e:
        log.warning("52wk_snapshots_fail: %s", e)

    # Pre-filter by price
    candidates = []
    for sym in universe:
        snap = snap_map.get(sym) or {}
        trade = snap.get("latest_trade") or {}
        daily = snap.get("daily_bar") or {}
        price = float(trade.get("price") or daily.get("close") or 0)
        if price <= 0 or not (min_price <= price <= max_price):
            continue
        candidates.append(sym)

    if not candidates:
        return []

    # Fetch 1-year daily bars via yfinance (batch)
    try:
        raw = yf.download(
            " ".join(candidates),
            period="1y",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        log.warning("52wk_yfinance_fail: %s", e)
        return []

    results: list[dict] = []
    for sym in candidates:
        try:
            snap = snap_map.get(sym) or {}
            trade = snap.get("latest_trade") or {}
            daily = snap.get("daily_bar") or {}
            prev  = snap.get("prev_daily_bar") or {}

            price = float(trade.get("price") or daily.get("close") or 0)
            if price <= 0:
                continue

            # Extract 52-week high from yfinance
            if len(candidates) == 1:
                hist_high = raw["High"].dropna()
            else:
                try:
                    hist_high = raw["High"][sym].dropna()
                except Exception:
                    continue
            if hist_high.empty:
                continue
            wk52_high = float(hist_high.max())
            wk52_low  = float(raw["Low"][sym].dropna().min()) if len(candidates) > 1 else float(raw["Low"].dropna().min())

            pct_from_high = (price - wk52_high) / wk52_high * 100.0

            # Only care about stocks within 5% of 52-week high
            if pct_from_high < -5.0:
                continue

            # RVOL from snapshot
            today_vol = float(daily.get("volume") or 0)
            avg_vol_raw = snap.get("avg_daily_volume") or 0
            if avg_vol_raw:
                rvol = today_vol / float(avg_vol_raw) if float(avg_vol_raw) > 0 else 0.0
            else:
                # Estimate from yfinance 20-day avg
                if len(candidates) > 1:
                    hist_vol = raw["Volume"][sym].dropna()
                else:
                    hist_vol = raw["Volume"].dropna()
                avg20 = float(hist_vol.tail(20).mean()) if len(hist_vol) >= 20 else float(hist_vol.mean())
                rvol = today_vol / avg20 if avg20 > 0 else 0.0

            if rvol < min_rvol:
                continue

            day_chg_pct = 0.0
            prev_c = float(prev.get("close") or 0)
            if prev_c > 0:
                day_chg_pct = (price - prev_c) / prev_c * 100.0

            results.append({
                "symbol":         sym,
                "price":          round(price, 2),
                "wk52_high":      round(wk52_high, 2),
                "wk52_low":       round(wk52_low, 2),
                "pct_from_high":  round(pct_from_high, 2),
                "at_new_high":    pct_from_high >= -1.0,
                "rvol":           round(rvol, 2),
                "day_chg_pct":    round(day_chg_pct, 2),
            })
        except Exception as e:
            log.debug("52wk_sym_fail %s: %s", sym, e)

    results.sort(key=lambda r: (r["pct_from_high"] >= -1.0, -r["rvol"]), reverse=True)
    return results[:top_n]


# ── FDA / catalyst newsflow ───────────────────────────────────────────────────

_FDA_KEYWORDS = [
    "PDUFA", "FDA approval", "NDA", "BLA", "IND ", "Phase 3", "Phase 2",
    "clinical trial", "FDA approved", "FDA granted", "accelerated approval",
    "fast track", "breakthrough therapy",
]

def scan_fda_events(
    provider=None,
    lookback_hours: int = 72,
    max_results: int = 15,
) -> list[dict[str, Any]]:
    """
    Scan Alpaca news for FDA/biotech catalyst events in the last lookback_hours.
    Returns list of {symbol, headline, age_hours, url}.
    """
    if provider is None:
        from providers.alpaca_provider import AlpacaProvider
        provider = AlpacaProvider()

    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest
        from scanner.orb import _get_alpaca_creds

        key, sec = _get_alpaca_creds(provider)
        client = NewsClient(api_key=key, secret_key=sec)

        start = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
        req = NewsRequest(limit=200, start=start)
        news_resp = client.get_news(req)
        articles = getattr(news_resp, "news", []) or []

        seen_sym: set[str] = set()
        results: list[dict] = []

        for art in articles:
            headline = getattr(art, "headline", "") or ""
            summary  = getattr(art, "summary", "")  or ""
            text = headline + " " + summary

            if not any(kw.lower() in text.lower() for kw in _FDA_KEYWORDS):
                continue

            syms = list(getattr(art, "symbols", []) or [])
            pub = getattr(art, "created_at", None)
            age_hours = 0.0
            if pub:
                try:
                    if isinstance(pub, str):
                        pub = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    age_hours = (datetime.now(timezone.utc) - pub).total_seconds() / 3600
                except Exception:
                    pass

            for sym in (syms or ["—"]):
                sym = sym.upper()
                if sym in seen_sym:
                    continue
                seen_sym.add(sym)
                results.append({
                    "symbol":     sym,
                    "headline":   headline[:120],
                    "age_hours":  round(age_hours, 1),
                    "url":        getattr(art, "url", "") or "",
                    "symbols":    syms,
                })
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        return results
    except Exception as e:
        log.warning("fda_scan_fail: %s", e)
        return []


# ── Master aggregator ─────────────────────────────────────────────────────────

def fetch_morning_prep(
    max_price: float = 30.0,
    min_rvol: float = 1.5,
) -> dict[str, Any]:
    """Run all four prep modules and return a single aggregated dict."""
    import concurrent.futures

    provider = None
    try:
        from providers.alpaca_provider import AlpacaProvider
        provider = AlpacaProvider()
    except Exception:
        pass

    results: dict[str, Any] = {
        "earnings":   [],
        "economic":   [],
        "wk52_highs": [],
        "fda":        [],
        "errors":     {},
        "generated_at": datetime.now(_ET).isoformat(),
    }

    def _run_earnings():
        try:
            return "earnings", fetch_earnings_calendar(lookback_days=0, forward_days=3, max_price=max_price)
        except Exception as e:
            return "earnings_err", str(e)

    def _run_economic():
        try:
            return "economic", fetch_economic_events(forward_days=30)
        except Exception as e:
            return "economic_err", str(e)

    def _run_52wk():
        try:
            return "wk52_highs", scan_52wk_highs(provider=provider, min_rvol=min_rvol, max_price=max_price)
        except Exception as e:
            return "wk52_err", str(e)

    def _run_fda():
        try:
            return "fda", scan_fda_events(provider=provider, lookback_hours=72)
        except Exception as e:
            return "fda_err", str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(f) for f in (_run_earnings, _run_economic, _run_52wk, _run_fda)]
        for fut in concurrent.futures.as_completed(futs):
            key, val = fut.result()
            if key.endswith("_err"):
                results["errors"][key] = val
            else:
                results[key] = val

    return results
