"""
Know the Trade — per-symbol conviction engine.
Aggregates float, live RVOL, spread, catalyst quality, PM trend,
ML score and emits a letter grade + position sizing recommendation.
"""
from __future__ import annotations
from typing import Any


# ── Catalyst quality detection ────────────────────────────────────────────────

_EARNINGS_KW   = ["earnings","eps","revenue","beat","quarterly","q1","q2","q3","q4","fiscal","profit","loss","guidance raised"]
_FDA_KW        = ["fda","pdufa","approval","nda","bla","sba","phase 2","phase 3","clinical","biologics","510k","ind "]
_DEAL_KW       = ["acqui","merger","buyout","takeover","deal","offer","bid for"]
_GUIDANCE_KW   = ["guidance","outlook","forecast","raised","lowered","reiterate","target price","price target"]
_CONTRACT_KW   = ["contract","partnership","agreement","collaboration","licensing"]

_BEARISH_KW = [
    "downgrade", "cut", "miss", "below estimate", "lowered guidance", "recall",
    "fda reject", "complete response letter", "crl ", "warning letter",
    "investigation", "fraud", "lawsuit", "bankruptcy", "going concern",
    "disappointing", "shortfall", "loss wider", "revenue miss",
]

def _catalyst_sentiment(headline: str) -> str:
    """Return 'bullish', 'bearish', or 'neutral' from headline text."""
    hl = headline.lower()
    if any(kw in hl for kw in _BEARISH_KW):
        return "bearish"
    if any(kw in hl for kw in _EARNINGS_KW + _FDA_KW + _DEAL_KW + _GUIDANCE_KW + _CONTRACT_KW):
        return "bullish"
    return "neutral"

def _grade_catalyst(headline: str | None, age_hours: float | None, side: str = "long") -> dict:
    if not headline:
        return {"tier": "none", "label": "Technical move — no news catalyst", "tier_pts": 8, "fresh_pts": 0, "total_pts": 8, "pts": 8, "warn": False}

    hl   = headline.lower()
    age  = age_hours if age_hours is not None else 999.0

    if any(kw in hl for kw in _EARNINGS_KW):
        tier, label, tier_pts = "earnings", "Earnings", 22
    elif any(kw in hl for kw in _FDA_KW):
        tier, label, tier_pts = "fda", "FDA/Clinical", 22
    elif any(kw in hl for kw in _DEAL_KW):
        tier, label, tier_pts = "deal", "M&A / Deal", 20
    elif any(kw in hl for kw in _GUIDANCE_KW):
        tier, label, tier_pts = "guidance", "Guidance Update", 12
    elif any(kw in hl for kw in _CONTRACT_KW):
        tier, label, tier_pts = "contract", "Contract / Partnership", 10
    else:
        tier, label, tier_pts = "news", "News / Other", 5

    if   age < 2:    fresh_pts = 18
    elif age < 12:   fresh_pts = 12
    elif age < 48:   fresh_pts = 5
    else:            fresh_pts = 0

    # Direction alignment: penalise catalyst that contradicts trade side
    sentiment = _catalyst_sentiment(headline)
    direction_warn = False
    if side == "short" and sentiment == "bullish":
        # Positive catalyst on a short — halve the score, flag it
        tier_pts  = max(0, tier_pts  // 2)
        fresh_pts = max(0, fresh_pts // 2)
        label     = f"{label} ⚠ bullish catalyst vs short"
        direction_warn = True
    elif side == "long" and sentiment == "bearish":
        tier_pts  = max(0, tier_pts  // 2)
        fresh_pts = max(0, fresh_pts // 2)
        label     = f"{label} ⚠ bearish catalyst vs long"
        direction_warn = True

    return {
        "tier":           tier,
        "label":          label,
        "tier_pts":       tier_pts,
        "fresh_pts":      fresh_pts,
        "total_pts":      tier_pts + fresh_pts,
        "pts":            tier_pts + fresh_pts,
        "age_hours":      round(age, 1) if age < 999 else None,
        "warn":           direction_warn,
        "direction_warn": direction_warn,
        "sentiment":      sentiment,
    }


# ── Float classification ──────────────────────────────────────────────────────

def _float_tier(float_shares: float | None) -> dict:
    if float_shares is None or float_shares <= 0:
        return {"pts": 5, "label": "Unknown float", "tier": "unknown"}
    m = float_shares / 1_000_000
    if   m < 5:    return {"pts": 28, "label": f"Micro float ({m:.1f}M)", "tier": "micro"}
    elif m < 20:   return {"pts": 20, "label": f"Small float ({m:.1f}M)", "tier": "small"}
    elif m < 50:   return {"pts": 10, "label": f"Mid float ({m:.1f}M)",   "tier": "mid"}
    elif m < 200:  return {"pts": 4,  "label": f"Large float ({m:.0f}M)", "tier": "large"}
    else:          return {"pts": 0,  "label": f"Mega float ({m:.0f}M)",  "tier": "mega"}


# ── RVOL scoring ──────────────────────────────────────────────────────────────

def _rvol_pts(rvol: float | None) -> dict:
    if rvol is None:
        return {"pts": 3, "label": "RVOL unknown"}
    if   rvol >= 10: return {"pts": 25, "label": f"RVOL {rvol:.1f}x — extreme"}
    elif rvol >= 5:  return {"pts": 22, "label": f"RVOL {rvol:.1f}x — very high"}
    elif rvol >= 3:  return {"pts": 17, "label": f"RVOL {rvol:.1f}x — elevated"}
    elif rvol >= 2:  return {"pts": 11, "label": f"RVOL {rvol:.1f}x — above avg"}
    elif rvol >= 1:  return {"pts": 5,  "label": f"RVOL {rvol:.1f}x — average"}
    else:            return {"pts": 0,  "label": f"RVOL {rvol:.1f}x — below avg"}


# ── Spread scoring ────────────────────────────────────────────────────────────

def _spread_pts(bid: float | None, ask: float | None) -> dict:
    if not bid or not ask or bid <= 0 or ask <= 0:
        return {"pts": 3, "label": "Spread unknown", "spread_pct": None}
    spread = ask - bid
    mid    = (bid + ask) / 2.0
    pct    = spread / mid * 100.0 if mid > 0 else 0.0
    if   pct < 0.1: pts, grade = 12, "Tight"
    elif pct < 0.3: pts, grade = 10, "Good"
    elif pct < 0.7: pts, grade = 6,  "Acceptable"
    elif pct < 1.5: pts, grade = 2,  "Wide"
    else:           pts, grade = 0,  "Very wide — dangerous fills"
    return {"pts": pts, "label": f"{grade} ({pct:.2f}%)", "spread_pct": round(pct, 3),
            "bid": bid, "ask": ask, "spread": round(spread, 4)}


# ── ML bonus ─────────────────────────────────────────────────────────────────

def _ml_pts(ml_score: float | None) -> dict:
    if ml_score is None:
        return {"pts": 0, "label": "ML not scored"}
    if   ml_score >= 0.75: return {"pts": 13, "label": f"ML {ml_score:.0%} — strong signal"}
    elif ml_score >= 0.60: return {"pts": 8,  "label": f"ML {ml_score:.0%} — above threshold"}
    elif ml_score >= 0.45: return {"pts": 4,  "label": f"ML {ml_score:.0%} — borderline"}
    else:                  return {"pts": 0,  "label": f"ML {ml_score:.0%} — weak signal"}


# ── PM trend quality ──────────────────────────────────────────────────────────

def _pm_trend(pm_last: float | None, pm_high: float | None,
              pm_low: float | None, pm_move_pct: float | None,
              side: str = "long") -> dict:
    if pm_last is None or pm_high is None or pm_low is None:
        return {"pts": 0, "label": "PM data missing"}
    if side == "long":
        hold_ratio = pm_last / pm_high if pm_high > 0 else 0
        if   hold_ratio >= 0.97: label, pts = "Holding highs — still running", 8
        elif hold_ratio >= 0.90: label, pts = "Near highs — consolidating",    5
        elif hold_ratio >= 0.80: label, pts = "Off highs — pulling back",       2
        else:                    label, pts = "Faded significantly — caution",  0
    else:
        # pm_low is always <= pm_last, so ratio is always <= 1.0 — mirror long logic
        hold_ratio = pm_low / pm_last if pm_last > 0 else 0
        if   hold_ratio >= 0.97: label, pts = "Holding lows — still fading",   8
        elif hold_ratio >= 0.90: label, pts = "Near lows — consolidating",     5
        elif hold_ratio >= 0.80: label, pts = "Off lows — bouncing",           2
        else:                    label, pts = "Recovered significantly — risk", 0
    return {"pts": pts, "label": label, "hold_ratio": round(hold_ratio, 3)}


# ── Grade from total score ────────────────────────────────────────────────────

def _letter_grade(score: int) -> dict:
    if   score >= 78: return {"grade": "A", "color": "#4ade80", "size_mult": 1.00,
                               "advice": "Full conviction — lead with this name. Prioritize your max R."}
    elif score >= 56: return {"grade": "B", "color": "#fbbf24", "size_mult": 0.50,
                               "advice": "Solid setup — half size. Can add on confirmation."}
    elif score >= 35: return {"grade": "C", "color": "#f97316", "size_mult": 0.25,
                               "advice": "Mixed signals — quarter size or watch-only."}
    else:             return {"grade": "D", "color": "#f87171", "size_mult": 0.00,
                               "advice": "Skip. Trade your A and B setups today."}


# ── Position sizing ───────────────────────────────────────────────────────────

def _position_sizing(entry: float | None, stop: float | None,
                     grade_mult: float) -> dict:
    if not entry or not stop or entry <= 0 or abs(entry - stop) < 0.01:
        return {"tiers": [], "note": "Entry/stop needed for sizing"}
    risk = abs(entry - stop)
    tiers = []
    for label, dollars in (("Conservative $250 1R", 250), ("Standard $500 1R", 500), ("Aggressive $1000 1R", 1000)):
        full_shares    = int(dollars / risk)
        graded_shares  = max(0, int(full_shares * grade_mult))
        tiers.append({
            "label":         label,
            "risk_dollars":  dollars,
            "full_shares":   full_shares,
            "graded_shares": graded_shares,
            "risk_per_share": round(risk, 4),
        })
    return {"tiers": tiers, "risk_per_share": round(risk, 4)}


# ── Master function ───────────────────────────────────────────────────────────

def analyze(
    symbol: str,
    provider=None,
    entry: float | None = None,
    stop:  float | None = None,
    target: float | None = None,
    side: str = "long",
    pm_last: float | None = None,
    pm_high: float | None = None,
    pm_low:  float | None = None,
    pm_vol:  int | None   = None,
    pm_move_pct: float | None = None,
    ml_score: float | None = None,
    catalyst_headline: str | None = None,
    catalyst_age_hours: float | None = None,
    rvol_hint: float | None = None,
    **kwargs,
) -> dict[str, Any]:

    from datetime import datetime, timezone, timedelta

    if provider is None:
        from providers.alpaca_provider import AlpacaProvider
        provider = AlpacaProvider()

    sym = symbol.upper()
    breakdown: dict[str, Any] = {}

    # ── Live quote (spread + live RVOL) ──────────────────────────────────────
    bid = ask = live_rvol = None
    try:
        q = provider.get_latest_quote(sym) or {}
        bid = float(q.get("bid_price") or q.get("bid") or 0) or None
        ask = float(q.get("ask_price") or q.get("ask") or 0) or None
    except Exception:
        pass

    # Live RVOL from snapshot — fall back to PM volume estimate if daily bar not yet formed
    avg_daily_vol = None
    try:
        snaps = provider.get_snapshots([sym], feed="sip", timeout_s=8.0) or {}
        snap  = snaps.get(sym) or {}
        daily = snap.get("daily_bar") or {}
        today_vol   = float(daily.get("volume") or 0)
        avg_daily_vol = float(snap.get("avg_daily_volume") or 0)
        if today_vol > 0 and avg_daily_vol > 0:
            live_rvol = round(today_vol / avg_daily_vol, 2)
    except Exception:
        pass

    # Use rvol_hint from scanner if live data unavailable (scanner already computed it)
    if live_rvol is None and rvol_hint and rvol_hint > 0:
        live_rvol = rvol_hint

    # Pre-market fallback: estimate RVOL from PM volume if daily bar hasn't formed yet
    if live_rvol is None and pm_vol and pm_vol > 0:
        try:
            if avg_daily_vol is None or avg_daily_vol <= 0:
                # Try fetching avg vol separately if snapshot didn't return it
                snaps2 = provider.get_snapshots([sym], feed="sip", timeout_s=6.0) or {}
                avg_daily_vol = float((snaps2.get(sym) or {}).get("avg_daily_volume") or 0)
            if avg_daily_vol and avg_daily_vol > 0:
                # PM vol / (ADV * 0.25): pre-market typically ~25% of regular session vol
                # so pm_vol equal to 25% of ADV = roughly 1x RVOL equivalent
                pm_rvol_est = round(pm_vol / (avg_daily_vol * 0.25), 2)
                live_rvol = pm_rvol_est
        except Exception:
            pass

    # ── Float ────────────────────────────────────────────────────────────────
    float_shares = None
    float_turnover_pct = None
    try:
        from scanner.orb import _fetch_float_shares
        store = getattr(provider, "_store", None)
        float_shares = _fetch_float_shares(sym, store)
        if float_shares and pm_vol and float_shares > 0:
            float_turnover_pct = round(pm_vol / float_shares * 100.0, 2)
    except Exception:
        pass

    # ── Score components ──────────────────────────────────────────────────────
    cat   = _grade_catalyst(catalyst_headline, catalyst_age_hours, side=side)
    fl    = _float_tier(float_shares)
    rv    = _rvol_pts(live_rvol)
    # Flag if RVOL was estimated from PM volume rather than live daily bar
    if live_rvol is not None and rv.get("label") and pm_vol and "unknown" not in rv["label"].lower():
        from datetime import datetime
        import pytz
        _et = pytz.timezone("America/New_York")
        _now_et = datetime.now(_et)
        _is_premarket = _now_et.hour < 9 or (_now_et.hour == 9 and _now_et.minute < 30)
        if _is_premarket:
            rv = {**rv, "label": rv["label"] + " (PM est.)"}
    sp    = _spread_pts(bid, ask)
    ml    = _ml_pts(ml_score)
    pm    = _pm_trend(pm_last, pm_high, pm_low, pm_move_pct, side)

    total = cat["total_pts"] + fl["pts"] + rv["pts"] + sp["pts"] + ml["pts"] + pm["pts"]
    total = min(100, total)

    grade_info = _letter_grade(total)
    sizing     = _position_sizing(entry, stop, grade_info["size_mult"])

    breakdown = {
        "catalyst":  {**cat,  "component": "Catalyst quality + freshness"},
        "float":     {**fl,   "component": "Float size"},
        "rvol":      {**rv,   "component": "Relative volume (live)"},
        "spread":    {**sp,   "component": "Bid/ask spread"},
        "ml":        {**ml,   "component": "ML model score"},
        "pm_trend":  {**pm,   "component": "PM trend quality"},
    }

    return {
        "symbol":              sym,
        "grade":               grade_info["grade"],
        "grade_color":         grade_info["color"],
        "grade_advice":        grade_info["advice"],
        "score":               total,
        "breakdown":           breakdown,
        "sizing":              sizing,
        # raw data for display
        "float_shares":        float_shares,
        "float_turnover_pct":  float_turnover_pct,
        "live_rvol":           live_rvol,
        "bid":                 bid,
        "ask":                 ask,
        "spread_pct":          sp.get("spread_pct"),
        "entry":               entry,
        "stop":                stop,
        "target":              target,
        "side":                side,
        "pm_last":             pm_last,
        "pm_high":             pm_high,
        "pm_low":              pm_low,
        "pm_vol":              pm_vol,
        "pm_move_pct":         pm_move_pct,
        "ml_score":            ml_score,
        "catalyst_headline":   catalyst_headline,
        "catalyst_age_hours":  catalyst_age_hours,
    }
