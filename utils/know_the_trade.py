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
_GUIDANCE_KW   = ["guidance","outlook","forecast","raised","lowered","target price","price target"]
_CONTRACT_KW   = ["contract","partnership","agreement","collaboration","licensing"]

_BEARISH_KW = [
    "downgrade", "cut", "miss", "below estimate", "lowered guidance", "recall",
    "fda reject", "complete response letter", "crl ", "warning letter",
    "investigation", "fraud", "lawsuit", "bankruptcy", "going concern",
    "disappointing", "shortfall", "loss wider", "revenue miss",
]

def _catalyst_sentiment(headline: str) -> str:
    hl = headline.lower()
    if any(kw in hl for kw in _BEARISH_KW):
        return "bearish"
    if any(kw in hl for kw in _EARNINGS_KW + _FDA_KW + _DEAL_KW + _GUIDANCE_KW + _CONTRACT_KW):
        return "bullish"
    return "neutral"

def _grade_catalyst(headline: str | None, age_hours: float | None, side: str = "long") -> dict:
    # No catalyst = -8pts: technical-only move has no thesis, cannot be A grade
    if not headline:
        return {"tier": "none", "label": "No catalyst — technical move, thesis unconfirmed",
                "tier_pts": -8, "fresh_pts": 0, "total_pts": -8, "pts": -8, "warn": False}

    hl  = headline.lower()
    age = age_hours if age_hours is not None else 999.0

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

    # Fix #13: tighter freshness curve — >12h is stale for intraday
    if   age < 2:    fresh_pts = 18
    elif age < 6:    fresh_pts = 8
    elif age < 12:   fresh_pts = 3
    else:            fresh_pts = 0

    sentiment      = _catalyst_sentiment(headline)
    direction_warn = False
    if side == "short" and sentiment == "bullish":
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

def _float_tier(float_shares: float | None, float_turnover_pct: float | None = None) -> dict:
    if float_shares is None or float_shares <= 0:
        return {"pts": 5, "label": "Unknown float", "tier": "unknown",
                "float_turnover_pct": float_turnover_pct}
    m = float_shares / 1_000_000
    if   m < 5:    pts, label, tier = 28, f"Micro float ({m:.1f}M)", "micro"
    elif m < 20:   pts, label, tier = 20, f"Small float ({m:.1f}M)", "small"
    elif m < 50:   pts, label, tier = 10, f"Mid float ({m:.1f}M)",   "mid"
    elif m < 200:  pts, label, tier =  4, f"Large float ({m:.0f}M)", "large"
    else:          pts, label, tier =  0, f"Mega float ({m:.0f}M)",  "mega"

    # Fix #7: distribution threshold at 500% (not 2000%)
    if float_turnover_pct is not None and float_turnover_pct > 0:
        if float_turnover_pct > 500:
            label += f" ⚠ DISTRIBUTED ({float_turnover_pct:.0f}% float traded — move is over)"
            pts    = 0
        elif float_turnover_pct > 300:
            label += f" ⚠ Heavy churn ({float_turnover_pct:.0f}% — late stage)"
            pts    = max(0, pts - int(pts * 0.50))
        elif float_turnover_pct >= 20:
            label += f" ✓ {float_turnover_pct:.0f}% float traded PM"

    return {"pts": pts, "label": label, "tier": tier, "float_turnover_pct": float_turnover_pct}


# ── RVOL scoring ──────────────────────────────────────────────────────────────

def _rvol_pts(rvol: float | None, pm_move_pct: float | None = None,
              float_tier: str = "unknown") -> dict:
    if rvol is None:
        return {"pts": 3, "label": "RVOL unknown", "blow_off_risk": False}

    blow_off_risk = False
    if (rvol >= 10 and pm_move_pct is not None and abs(pm_move_pct) >= 30
            and float_tier in ("micro", "small")):
        blow_off_risk = True

    # Fix #10: below-average RVOL is a negative signal for momentum plays
    if   rvol >= 10:  pts, lbl = 25, f"RVOL {rvol:.1f}x — extreme"
    elif rvol >= 5:   pts, lbl = 22, f"RVOL {rvol:.1f}x — very high"
    elif rvol >= 3:   pts, lbl = 17, f"RVOL {rvol:.1f}x — elevated"
    elif rvol >= 2:   pts, lbl = 11, f"RVOL {rvol:.1f}x — above avg"
    elif rvol >= 1.5: pts, lbl =  8, f"RVOL {rvol:.1f}x — building"
    elif rvol >= 1.0: pts, lbl =  5, f"RVOL {rvol:.1f}x — average"
    elif rvol >= 0.5: pts, lbl = -3, f"RVOL {rvol:.1f}x — below avg (weak confirmation)"
    else:             pts, lbl = -8, f"RVOL {rvol:.1f}x — very low (no institutional interest)"

    if blow_off_risk:
        pts  = max(0, pts - 8)
        lbl += " ⚠ BLOW-OFF RISK — extreme RVOL + extended move on small float"

    return {"pts": pts, "label": lbl, "blow_off_risk": blow_off_risk}


# ── R:R ratio scoring ─────────────────────────────────────────────────────────

def _rr_score(entry: float | None, stop: float | None, target: float | None,
              side: str = "long") -> dict:
    if not entry or not stop or not target or entry <= 0:
        return {"pts": 0, "label": "R:R unknown — target not set", "rr": None}
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    if risk < 0.001:
        return {"pts": 0, "label": "Stop too tight — risk undefined", "rr": None}
    rr = reward / risk
    if   rr >= 3.0: pts, lbl = 12, f"{rr:.1f}:1 R:R — excellent"
    elif rr >= 2.5: pts, lbl =  9, f"{rr:.1f}:1 R:R — strong"
    elif rr >= 2.0: pts, lbl =  5, f"{rr:.1f}:1 R:R — minimum pro standard"
    elif rr >= 1.5: pts, lbl = -5, f"{rr:.1f}:1 R:R — below pro standard, do not size up"
    else:           pts, lbl =-12, f"{rr:.1f}:1 R:R — unacceptable, avoid"
    return {"pts": pts, "label": lbl, "rr": round(rr, 2)}


# ── Spread scoring ────────────────────────────────────────────────────────────

def _spread_pts(bid: float | None, ask: float | None) -> dict:
    if not bid or not ask or bid <= 0 or ask <= 0:
        return {"pts": 3, "label": "Spread unknown", "spread_pct": None}
    spread = ask - bid
    mid    = (bid + ask) / 2.0
    pct    = spread / mid * 100.0 if mid > 0 else 0.0
    if   pct < 0.1: pts, grade = 12, "Tight"
    elif pct < 0.3: pts, grade = 10, "Good"
    elif pct < 0.7: pts, grade =  6, "Acceptable"
    elif pct < 1.5: pts, grade =  2, "Wide"
    else:           pts, grade =  0, "Very wide — dangerous fills"
    return {"pts": pts, "label": f"{grade} ({pct:.2f}%)", "spread_pct": round(pct, 3),
            "bid": bid, "ask": ask, "spread": round(spread, 4)}


# ── ML bonus ─────────────────────────────────────────────────────────────────

def _ml_pts(ml_score: float | None) -> dict:
    if ml_score is None:
        return {"pts": -8, "label": "ML not scored — no model conviction"}
    if   ml_score >= 0.75: return {"pts": 13, "label": f"ML {ml_score:.0%} — strong signal"}
    elif ml_score >= 0.60: return {"pts": 8,  "label": f"ML {ml_score:.0%} — above threshold"}
    elif ml_score >= 0.50: return {"pts": 0,  "label": f"ML {ml_score:.0%} — neutral / borderline"}
    elif ml_score >= 0.40: return {"pts": -6, "label": f"ML {ml_score:.0%} — below threshold, model cautious"}
    elif ml_score >= 0.30: return {"pts": -12,"label": f"ML {ml_score:.0%} — model says likely fail"}
    else:                  return {"pts": -18,"label": f"ML {ml_score:.0%} — model strongly against setup"}


# ── PM trend quality ──────────────────────────────────────────────────────────

def _pm_trend(pm_last: float | None, pm_high: float | None,
              pm_low: float | None, pm_move_pct: float | None,
              side: str = "long") -> dict:
    if pm_last is None or pm_high is None or pm_low is None:
        return {"pts": 0, "label": "PM data missing"}
    if side == "long":
        hold_ratio = pm_last / pm_high if pm_high > 0 else 0
        if   hold_ratio >= 0.97: label, pts = "Holding highs — still running",           8
        elif hold_ratio >= 0.90: label, pts = "Near highs — consolidating",               5
        elif hold_ratio >= 0.80: label, pts = "Off highs — pulling back",                 2
        elif hold_ratio >= 0.70: label, pts = "Faded significantly — caution",            0
        elif hold_ratio >= 0.60: label, pts = "Heavy distribution from highs",           -8
        else:                    label, pts = "Collapsed from highs — move is over",    -16
    else:
        hold_ratio = pm_low / pm_last if pm_last > 0 else 0
        if   hold_ratio <= 1.03: label, pts = "Holding lows — still fading",             8
        elif hold_ratio <= 1.10: label, pts = "Near lows — consolidating",               5
        elif hold_ratio <= 1.20: label, pts = "Off lows — bouncing",                     2
        elif hold_ratio <= 1.30: label, pts = "Bouncing significantly — caution",         0
        elif hold_ratio <= 1.45: label, pts = "Heavy recovery from lows",               -8
        else:                    label, pts = "Recovered from lows — move is over",    -16
    return {"pts": pts, "label": label, "hold_ratio": round(hold_ratio, 3)}


# ── VWAP position scoring ─────────────────────────────────────────────────────

def _vwap_pts(pm_last: float | None, vwap: float | None, side: str = "long") -> dict:
    if pm_last is None or vwap is None or vwap <= 0:
        return {"pts": 0, "label": "VWAP unknown"}
    delta_pct = (pm_last - vwap) / vwap * 100.0
    if side == "long":
        if   delta_pct >  2.0: pts, lbl = 5,  f"Price {delta_pct:.1f}% above VWAP — long confirmed"
        elif delta_pct >= 0:   pts, lbl = 3,  f"Price at/near VWAP ({delta_pct:.1f}%) — neutral for longs"
        elif delta_pct >= -2:  pts, lbl = -3, f"Price {abs(delta_pct):.1f}% below VWAP — long against trend"
        else:                  pts, lbl = -6, f"Price {abs(delta_pct):.1f}% below VWAP — VWAP resistance overhead"
    else:
        if   delta_pct < -2.0: pts, lbl = 5,  f"Price {abs(delta_pct):.1f}% below VWAP — short confirmed"
        elif delta_pct <= 0:   pts, lbl = 3,  f"Price at/near VWAP ({delta_pct:.1f}%) — neutral for shorts"
        elif delta_pct <= 2:   pts, lbl = -3, f"Price {delta_pct:.1f}% above VWAP — short against trend"
        else:                  pts, lbl = -6, f"Price {delta_pct:.1f}% above VWAP — VWAP support below"
    return {"pts": pts, "label": lbl, "vwap": round(vwap, 4), "delta_pct": round(delta_pct, 2)}


# ── Time decay ────────────────────────────────────────────────────────────────

def _time_decay(setup_age_hours: float | None) -> dict:
    if setup_age_hours is None or setup_age_hours <= 0:
        return {"pts": 0, "label": "Setup fresh / EOD review — no decay"}
    if   setup_age_hours <= 0.5: pts, lbl = 0,   "Fresh — window open"
    elif setup_age_hours <= 1.0: pts, lbl = -10, "1h — window narrowing fast"
    elif setup_age_hours <= 2.0: pts, lbl = -22, "2h — momentum setups rarely work past here"
    elif setup_age_hours <= 4.0: pts, lbl = -32, "4h — stale, avoid unless new catalyst"
    elif setup_age_hours <= 6.0: pts, lbl = -40, "6h — do not trade this"
    else:                        pts, lbl = -48, f"{setup_age_hours:.1f}h — setup is dead"
    return {"pts": pts, "label": lbl, "age_hours": round(setup_age_hours, 2)}


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

def _position_sizing(entry: float | None, natural_stop: float | None,
                     grade_mult: float, stop_cap: float | None = None) -> dict:
    if not entry or not natural_stop or entry <= 0 or abs(entry - natural_stop) < 0.01:
        return {"tiers": [], "note": "Entry/stop needed for sizing"}
    # Always size on the ACTUAL stop distance — never inflate shares using an artificial cap.
    # A wide stop means fewer shares, which is correct risk management.
    risk_per_share = abs(entry - natural_stop)
    risk_pct = risk_per_share / entry * 100.0
    tiers = []
    for label, dollars in (("Conservative $250 1R", 250), ("Standard $500 1R", 500), ("Aggressive $1000 1R", 1000)):
        full_shares   = int(dollars / risk_per_share)
        graded_shares = max(0, int(full_shares * grade_mult))
        tiers.append({
            "label":          label,
            "risk_dollars":   dollars,
            "full_shares":    full_shares,
            "graded_shares":  graded_shares,
            "risk_per_share": round(risk_per_share, 4),
        })
    note = f"Wide stop: {risk_pct:.1f}% risk/share — size is already reduced accordingly" if risk_pct > 8.0 else ""
    return {"tiers": tiers, "risk_per_share": round(risk_per_share, 4), "note": note}


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
    halt_count: int = 0,
    ssr_active: bool = False,
    stop_capped: bool = False,
    stop_cap: float | None = None,
    setup_age_hours: float | None = None,
    setup_status: str | None = None,
    gap_fill_rate: float | None = None,
    vwap: float | None = None,
    # sniper_context: when True, caps max pre-penalty score at 70 (no levels = max B grade)
    sniper_context: bool = False,
) -> dict[str, Any]:

    from datetime import datetime, timezone, timedelta

    if provider is None:
        from providers.alpaca_provider import AlpacaProvider
        provider = AlpacaProvider()

    sym = symbol.upper()

    import pytz as _pytz
    _ET = _pytz.timezone("America/New_York")

    # ── Live quote ────────────────────────────────────────────────────────────
    bid = ask = live_rvol = None
    try:
        q = provider.get_latest_quote(sym) or {}
        bid = float(q.get("bid_price") or q.get("bid") or 0) or None
        ask = float(q.get("ask_price") or q.get("ask") or 0) or None
    except Exception:
        pass

    avg_daily_vol = None
    today_vol = None
    prev_close = None
    snap = {}
    try:
        snaps = provider.get_snapshots([sym], feed="sip", timeout_s=8.0) or {}
        snap  = snaps.get(sym) or {}
        daily = snap.get("daily_bar") or {}
        prev_daily = snap.get("prev_daily_bar") or {}
        today_vol     = float(daily.get("volume") or 0)
        avg_daily_vol = float(snap.get("avg_daily_volume") or 0)
        if today_vol > 0 and avg_daily_vol > 0:
            live_rvol = round(today_vol / avg_daily_vol, 2)
        prev_close = float(prev_daily.get("close") or 0) or None
        # Use daily_bar VWAP from snapshot when caller didn't provide one
        if vwap is None:
            vwap = float(daily.get("vwap") or 0) or None
    except Exception:
        pass

    # Auto-detect SSR: stock down ≥10% from prior close = Reg SHO restriction active
    if not ssr_active and prev_close and prev_close > 0:
        try:
            _ref = pm_last or (bid and ask and (bid + ask) / 2)
            if _ref and _ref > 0:
                ssr_active = ((_ref - prev_close) / prev_close * 100.0) <= -10.0
        except Exception:
            pass

    # Auto-detect pm_move_pct from snapshot when not provided
    if pm_move_pct is None and prev_close and prev_close > 0:
        try:
            _ref = pm_last or (bid and ask and (bid + ask) / 2)
            if _ref and _ref > 0:
                pm_move_pct = round((_ref - prev_close) / prev_close * 100.0, 2)
        except Exception:
            pass

    if live_rvol is None and rvol_hint and rvol_hint > 0:
        live_rvol = rvol_hint

    # Fetch live pre-market bars when PM data not provided by caller
    _need_pm = pm_high is None or pm_low is None or pm_vol is None
    if _need_pm and provider is not None:
        try:
            from datetime import date as _date
            _now_et = datetime.now(_ET)
            _today = _now_et.date()
            _rth_cutoff = _now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            # Only fetch PM bars before RTH; during RTH use what we have
            _fetch_end = min(_now_et, _rth_cutoff)
            if _now_et.hour < 13:  # don't fetch PM bars in afternoon for stale data
                pm_bars_df = provider.get_bars_range(
                    symbol=sym,
                    interval="1m",
                    from_d=_today,
                    to_d=_today,
                    include_prepost=True,
                    feed="sip",
                    timeout_s=10,
                )
                if pm_bars_df is not None and not pm_bars_df.empty:
                    # Slice to pre-market only (before 9:30 ET)
                    if hasattr(pm_bars_df.index, 'tz') and pm_bars_df.index.tz is not None:
                        pm_bars_df = pm_bars_df.tz_convert(_ET)
                    _pm_mask = pm_bars_df.index.hour < 9
                    _pm_df = pm_bars_df[_pm_mask] if _pm_mask.any() else pm_bars_df
                    if not _pm_df.empty:
                        if pm_high is None and "High" in _pm_df.columns:
                            pm_high = float(_pm_df["High"].max())
                        if pm_low is None and "Low" in _pm_df.columns:
                            pm_low = float(_pm_df["Low"].min())
                        if pm_vol is None and "Volume" in _pm_df.columns:
                            pm_vol = int(_pm_df["Volume"].sum())
                        if pm_last is None and "Close" in _pm_df.columns:
                            pm_last = float(_pm_df["Close"].iloc[-1])
        except Exception:
            pass

    # Projected RVOL from PM volume when live snapshot didn't have today's volume yet
    if live_rvol is None and pm_vol and pm_vol > 0:
        try:
            if avg_daily_vol is None or avg_daily_vol <= 0:
                snaps2 = provider.get_snapshots([sym], feed="sip", timeout_s=6.0) or {}
                avg_daily_vol = float((snaps2.get(sym) or {}).get("avg_daily_volume") or 0)
            if avg_daily_vol and avg_daily_vol > 0:
                _now_et2 = datetime.now(_ET)
                _pm_open_h = 4.0  # 4am ET
                _elapsed_h = max(0.1, min(5.5, (_now_et2.hour + _now_et2.minute / 60.0) - _pm_open_h))
                _projected_day_vol = (pm_vol / _elapsed_h) * 6.5
                live_rvol = min(round(_projected_day_vol / avg_daily_vol, 2), 15.0)
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
    cat  = _grade_catalyst(catalyst_headline, catalyst_age_hours, side=side)
    fl   = _float_tier(float_shares, float_turnover_pct)
    rv   = _rvol_pts(live_rvol, pm_move_pct=pm_move_pct, float_tier=fl.get("tier", "unknown"))
    rr   = _rr_score(entry, stop, target, side=side)
    pm   = _pm_trend(pm_last, pm_high, pm_low, pm_move_pct, side)
    sp   = _spread_pts(bid, ask)
    ml   = _ml_pts(ml_score)
    td   = _time_decay(setup_age_hours)
    vw   = _vwap_pts(pm_last, vwap, side)

    if live_rvol is not None and pm_vol and "unknown" not in rv.get("label", "").lower():
        import pytz
        _et = pytz.timezone("America/New_York")
        _now_et = datetime.now(_et)
        if _now_et.hour < 9 or (_now_et.hour == 9 and _now_et.minute < 30):
            rv = {**rv, "label": rv["label"] + " (PM est.)"}

    total = cat["pts"] + fl["pts"] + rv["pts"] + rr["pts"] + pm["pts"] + sp["pts"] + ml["pts"] + vw["pts"]

    # Fix #6: sniper context (no entry/stop/pm levels) caps raw score at 70 → max grade B
    if sniper_context:
        total = min(70, total)

    total = max(0, min(100, total))

    # Time decay (RTH only — pre-market = 0)
    total = max(0, total + td["pts"])

    # Halt penalty
    halt_penalty, halt_label = 0, ""
    if halt_count >= 5:
        halt_penalty = 25; halt_label = f"DANGER: {halt_count} halts today — do not trade"
    elif halt_count >= 3:
        halt_penalty = 15; halt_label = f"{halt_count} halts — high risk, reduce size"
    elif halt_count == 2:
        halt_penalty = 7;  halt_label = "2 halts today — elevated volatility"
    elif halt_count == 1:
        halt_penalty = 3;  halt_label = "1 halt today — monitor closely"
    total = max(0, total - halt_penalty)

    # SSR penalty
    ssr_penalty, ssr_label = 0, ""
    if ssr_active and side == "short":
        ssr_penalty = 18
        ssr_label   = "SSR active — shorts restricted to uptick-only entries. Reduce size or avoid."
        total = max(0, total - ssr_penalty)
    elif ssr_active and side == "long":
        ssr_penalty = 12
        ssr_label   = "SSR active (down ≥10%) — bounce carries heavy gap-fill risk, prior day sellers still in control"
        total = max(0, total - ssr_penalty)

    # Fix #5: gap fill rate penalty — all four directional scenarios
    fill_penalty, fill_label = 0, ""
    if gap_fill_rate is not None and pm_move_pct is not None:
        gap_up = pm_move_pct > 0
        if side == "long" and gap_up:
            # Long on gap-up: high fill rate = gap tends to fade = bad
            if   gap_fill_rate >= 0.80: fill_penalty = 12; fill_label = f"Gap fill rate {gap_fill_rate:.0%} — gap almost always fades. Long is counter-trend."
            elif gap_fill_rate >= 0.65: fill_penalty = 7;  fill_label = f"Gap fill rate {gap_fill_rate:.0%} — gap frequently fades. High reversal risk."
        elif side == "short" and gap_up:
            # Short on gap-up: high fill rate validates short thesis
            if   gap_fill_rate >= 0.65: fill_penalty = -6; fill_label = f"Gap fill rate {gap_fill_rate:.0%} — history confirms this gap fades. Short thesis validated."
            elif gap_fill_rate < 0.35:  fill_penalty = 6;  fill_label = f"Gap fill rate {gap_fill_rate:.0%} — gap rarely fills. Short is fighting strong follow-through."
        elif side == "short" and not gap_up:
            # Short on gap-down: low fill rate = gap continues = short confirmed
            if   gap_fill_rate < 0.35:  fill_penalty = -6; fill_label = f"Gap fill rate {gap_fill_rate:.0%} — gap rarely fills. Short continuation confirmed."
            elif gap_fill_rate >= 0.65: fill_penalty = 8;  fill_label = f"Gap fill rate {gap_fill_rate:.0%} — gap often fills. Bounce risk for shorts."
        elif side == "long" and not gap_up:
            # Long on gap-down bounce: high fill rate supports bounce thesis
            if   gap_fill_rate >= 0.65: fill_penalty = -5; fill_label = f"Gap fill rate {gap_fill_rate:.0%} — gap often fills. Long bounce thesis supported."
            elif gap_fill_rate < 0.35:  fill_penalty = 6;  fill_label = f"Gap fill rate {gap_fill_rate:.0%} — gap rarely fills. Long fights continuation."
        if fill_penalty > 0:
            total = max(0, total - fill_penalty)
        elif fill_penalty < 0:
            total = min(100, total + abs(fill_penalty))

    # Entry distance penalty — price far from trigger = not today's trade
    entry_dist_penalty, entry_dist_label = 0, ""
    if entry and pm_last and entry > 0 and pm_last > 0:
        if side == "long" and pm_last < entry:
            dist_pct = (entry - pm_last) / pm_last * 100.0
        elif side == "short" and pm_last > entry:
            dist_pct = (pm_last - entry) / pm_last * 100.0
        else:
            dist_pct = 0.0
        if   dist_pct > 40: entry_dist_penalty = 30; entry_dist_label = f"Entry {dist_pct:.0f}% away — not today's trade"
        elif dist_pct > 25: entry_dist_penalty = 22; entry_dist_label = f"Entry {dist_pct:.0f}% away — unlikely to trigger today"
        elif dist_pct > 15: entry_dist_penalty = 15; entry_dist_label = f"Entry {dist_pct:.0f}% away — needs significant push"
        elif dist_pct > 8:  entry_dist_penalty = 10; entry_dist_label = f"Entry {dist_pct:.0f}% away — meaningful distance to trigger"
        elif dist_pct > 4:  entry_dist_penalty = 4;  entry_dist_label = f"Entry {dist_pct:.0f}% away — watch for approach"
        total = max(0, total - entry_dist_penalty)

    # Triggered penalty — entry window already breached, chasing is dangerous
    triggered_penalty, triggered_label = 0, ""
    if setup_status == "triggered":
        triggered_penalty = 25
        triggered_label   = "Setup already triggered — entry window passed. Chasing carries gap-fill risk."
        total = max(0, total - triggered_penalty)

    grade_info = _letter_grade(total)
    sizing     = _position_sizing(entry, stop, grade_info["size_mult"],
                                  stop_cap=stop_cap if stop_capped else None)

    breakdown = {
        "catalyst":   {**cat, "component": "Catalyst quality + freshness"},
        "float":      {**fl,  "component": "Float size"},
        "rvol":       {**rv,  "component": "Relative volume (live)"},
        "rr":         {**rr,  "component": "Risk / Reward ratio"},
        "pm_trend":   {**pm,  "component": "PM trend quality"},
        "spread":     {**sp,  "component": "Bid/ask spread"},
        "ml":         {**ml,  "component": "ML model score"},
        "vwap":       {**vw,  "component": "VWAP position"},
        "time_decay": {**td,  "component": "Setup freshness (time since open)"},
        "halts":      {"component": "Halt count today", "count": halt_count,
                       "penalty": halt_penalty, "pts": -halt_penalty,
                       "label": halt_label or f"{halt_count} halts today"},
        "ssr":        {"component": "Reg SHO Short Sale Restriction", "active": ssr_active,
                       "penalty": ssr_penalty, "pts": -ssr_penalty,
                       "label": ssr_label or ("SSR inactive" if not ssr_active else "SSR active")},
        "fill_rate":  {"component": "Historical gap fill rate", "rate": gap_fill_rate,
                       "pts": -fill_penalty if fill_penalty > 0 else abs(fill_penalty),
                       "label": fill_label or "Gap fill rate not available"},
        "entry_dist":  {"component": "Distance to entry trigger", "penalty": entry_dist_penalty,
                        "pts": -entry_dist_penalty,
                        "label": entry_dist_label or "Entry reachable — within 8% of trigger"},
        "triggered":  {"component": "Entry already triggered", "active": setup_status == "triggered",
                       "penalty": triggered_penalty, "pts": -triggered_penalty,
                       "label": triggered_label or "Entry not yet triggered"},
    }

    return {
        "symbol":             sym,
        "grade":              grade_info["grade"],
        "grade_color":        grade_info["color"],
        "grade_advice":       grade_info["advice"],
        "score":              total,
        "breakdown":          breakdown,
        "sizing":             sizing,
        "float_shares":       float_shares,
        "float_turnover_pct": float_turnover_pct,
        "live_rvol":          live_rvol,
        "bid":                bid,
        "ask":                ask,
        "spread_pct":         sp.get("spread_pct"),
        "entry":              entry,
        "stop":               stop,
        "target":             target,
        "side":               side,
        "pm_last":            pm_last,
        "pm_high":            pm_high,
        "pm_low":             pm_low,
        "pm_vol":             pm_vol,
        "pm_move_pct":        pm_move_pct,
        "ml_score":           ml_score,
        "catalyst_headline":  catalyst_headline,
        "catalyst_age_hours": catalyst_age_hours,
        "ssr_active":         ssr_active,
        "stop_capped":        stop_capped,
        "stop_cap":           stop_cap,
        "setup_age_hours":    setup_age_hours,
        "sniper_context":     sniper_context,
    }
