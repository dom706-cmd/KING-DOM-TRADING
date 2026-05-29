"""
Pre-Market Intelligence Engine
================================
Surfaces specific trade recommendations before 9:30 AM with:
  - Gap fade probability (rule-based, tuned on known trading statistics)
  - Halt prediction score (LULD band proximity + acceleration)
  - Float rotation tracker (volume / float)
  - Distribution detection (T&S uptick/downtick ratio)
  - Catalyst decay timer (category-based sustain windows)

All functions return plain dicts — no side effects.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

log = logging.getLogger(__name__)

# ── Catalyst sustain windows (minutes) based on real trading patterns ──────────
# How long each catalyst type historically keeps price elevated before fade begins
_CATALYST_SUSTAIN = {
    "earnings_beat_small":    {"avg": 47,  "range": (20, 90),  "label": "Earnings beat (small cap)"},
    "earnings_beat_mid":      {"avg": 120, "range": (60, 240), "label": "Earnings beat (mid cap)"},
    "earnings_beat_large":    {"avg": 360, "range": (120, 720),"label": "Earnings beat (large cap)"},
    "fda_approval":           {"avg": 180, "range": (60, 480), "label": "FDA approval"},
    "fda_trial_positive":     {"avg": 90,  "range": (30, 240), "label": "FDA trial positive"},
    "partnership":            {"avg": 60,  "range": (20, 180), "label": "Partnership/deal"},
    "acquisition":            {"avg": 240, "range": (120, 480),"label": "Acquisition news"},
    "short_squeeze":          {"avg": 30,  "range": (10, 90),  "label": "Short squeeze (no fundamental)"},
    "halt_resume":            {"avg": 25,  "range": (5, 60),   "label": "Halt/resume momentum"},
    "unknown":                {"avg": 35,  "range": (10, 90),  "label": "Unknown catalyst"},
}

# ── LULD band thresholds by price tier (SEC Rule 201 / NMS Plan) ───────────────
_LULD_BANDS = [
    (0.00,  1.00,  0.75),   # sub $1: ±75%
    (1.00,  5.00,  0.40),   # $1-5: ±40% (Tier 2 NMS)
    (5.00, 25.00,  0.10),   # $5-25: ±10%
    (25.00, 50.00, 0.05),   # $25-50: ±5%
    (50.00, float("inf"), 0.03),  # $50+: ±3% (Tier 1 NMS)
]


def _luld_band_pct(price: float) -> float:
    for lo, hi, band in _LULD_BANDS:
        if lo <= price < hi:
            return band
    return 0.05


# ── Gap Fade Probability ───────────────────────────────────────────────────────

def gap_fade_probability(
    gap_pct: float,
    catalyst_type: str = "unknown",
    float_shares: float | None = None,
    rvol: float | None = None,
    above_vwap: bool | None = None,
    pm_bars: int | None = None,
) -> dict[str, Any]:
    """
    Rule-based gap fade probability.
    Returns probability (0-1) that the gap fades within the session,
    plus a conviction label and key factors.

    Based on statistical patterns from real small-cap gap behavior:
    - Larger gaps = higher fade probability (distribution into retail)
    - Earnings beats on small floats fade most reliably
    - High RVOL in PM suggests real momentum (lower fade prob)
    - Below VWAP in PM = distribution already happening
    """
    gap = abs(float(gap_pct or 0.0))
    prob = 0.40  # base: 40% fade probability on any gap

    factors: list[str] = []

    # Gap size contribution
    if gap >= 150:
        prob += 0.35
        factors.append(f"+35% gap >{150}% (extreme distribution zone)")
    elif gap >= 100:
        prob += 0.28
        factors.append(f"+28% gap >{100}% (heavy distribution risk)")
    elif gap >= 75:
        prob += 0.22
        factors.append(f"+22% gap >{75}%")
    elif gap >= 50:
        prob += 0.15
        factors.append(f"+15% gap >{50}%")
    elif gap >= 30:
        prob += 0.08
        factors.append(f"+8% gap >{30}%")
    elif gap >= 15:
        prob += 0.03
        factors.append(f"+3% gap >{15}%")
    else:
        prob -= 0.05
        factors.append("-5% tight gap <15% (continuation more likely)")

    # Catalyst type adjustment
    cat = str(catalyst_type or "unknown").lower()
    if "earnings_beat" in cat and "large" in cat:
        prob -= 0.15
        factors.append("-15% large-cap earnings beat (institutional support)")
    elif "earnings_beat" in cat and "mid" in cat:
        prob -= 0.08
        factors.append("-8% mid-cap earnings beat")
    elif "earnings_beat" in cat and "small" in cat:
        prob += 0.05
        factors.append("+5% small-cap earnings beat (thin support, fade risk)")
    elif "fda_approval" in cat:
        prob -= 0.12
        factors.append("-12% FDA approval (binary catalyst, tends to hold)")
    elif "short_squeeze" in cat or "unknown" in cat:
        prob += 0.12
        factors.append("+12% no fundamental catalyst (pure momentum, fades fast)")
    elif "halt_resume" in cat:
        prob += 0.08
        factors.append("+8% halt/resume (erratic, fade common after second halt)")

    # Float size
    if float_shares is not None:
        if float_shares < 2_000_000:
            prob += 0.15
            factors.append(f"+15% tiny float <2M (fast rotation, sharp fades)")
        elif float_shares < 5_000_000:
            prob += 0.10
            factors.append(f"+10% small float <5M (high fade risk)")
        elif float_shares < 10_000_000:
            prob += 0.05
            factors.append(f"+5% float <10M")
        elif float_shares > 50_000_000:
            prob -= 0.08
            factors.append(f"-8% large float >50M (more institutional support)")

    # RVOL — high real volume = momentum, lowers fade prob
    rv = float(rvol or 0.0)
    if rv >= 20:
        prob -= 0.12
        factors.append(f"-12% RVOL {rv:.0f}x (massive volume confirms move)")
    elif rv >= 10:
        prob -= 0.07
        factors.append(f"-7% RVOL {rv:.0f}x (strong volume)")
    elif rv >= 5:
        prob -= 0.03
        factors.append(f"-3% RVOL {rv:.0f}x")
    elif rv < 2:
        prob += 0.08
        factors.append(f"+8% low RVOL {rv:.1f}x (weak volume, move unsupported)")

    # VWAP position in PM
    if above_vwap is False:
        prob += 0.10
        factors.append("+10% below PM VWAP (distribution already happening)")
    elif above_vwap is True:
        prob -= 0.05
        factors.append("-5% above PM VWAP (buyers in control)")

    # PM bar count — more bars = more established move
    if pm_bars is not None:
        if pm_bars < 5:
            prob += 0.05
            factors.append(f"+5% only {pm_bars} PM bars (move very early, unstable)")
        elif pm_bars > 30:
            prob -= 0.05
            factors.append(f"-5% {pm_bars} PM bars (sustained PM action)")

    prob = min(0.95, max(0.05, prob))

    if prob >= 0.75:
        label = "HIGH fade risk — short bias or avoid long"
        color = "red"
    elif prob >= 0.55:
        label = "MODERATE fade risk — tight stop mandatory"
        color = "orange"
    elif prob >= 0.40:
        label = "NEUTRAL — confirm direction at open"
        color = "yellow"
    else:
        label = "LOW fade risk — continuation more likely"
        color = "green"

    return {
        "fade_probability": round(prob, 2),
        "fade_pct": round(prob * 100, 0),
        "label": label,
        "color": color,
        "factors": factors,
    }


# ── Halt Prediction Score ──────────────────────────────────────────────────────

def halt_prediction_score(
    price: float,
    prev_close: float,
    pm_high: float | None = None,
    pm_low: float | None = None,
    pm_bars: list[dict] | None = None,
    rvol: float | None = None,
    float_shares: float | None = None,
) -> dict[str, Any]:
    """
    Scores probability of a halt in next 5 minutes (0-100).
    Based on LULD band proximity, acceleration, and float turnover.
    """
    if not price or not prev_close or prev_close <= 0:
        return {"score": 0, "risk": "unknown", "distance_to_band_pct": None, "factors": []}

    band_pct = _luld_band_pct(price)
    current_move_pct = (price - prev_close) / prev_close
    luld_up = prev_close * (1 + band_pct)
    distance_to_upper_band = (luld_up - price) / price if price < luld_up else 0.0

    score = 0.0
    factors: list[str] = []

    # Distance to LULD upper band
    dist_pct = distance_to_upper_band * 100
    if dist_pct < 2:
        score += 40
        factors.append(f"CRITICAL: {dist_pct:.1f}% from LULD halt band")
    elif dist_pct < 5:
        score += 25
        factors.append(f"WARNING: {dist_pct:.1f}% from LULD halt band")
    elif dist_pct < 10:
        score += 12
        factors.append(f"WATCH: {dist_pct:.1f}% from LULD halt band")

    # Acceleration — if we have PM bars, check last 3 vs prior 3
    if pm_bars and len(pm_bars) >= 6:
        try:
            recent = [float(b.get("close", 0) or 0) for b in pm_bars[-3:]]
            prior  = [float(b.get("close", 0) or 0) for b in pm_bars[-6:-3]]
            recent_move = (recent[-1] - recent[0]) / max(recent[0], 0.001)
            prior_move  = (prior[-1]  - prior[0])  / max(prior[0],  0.001)
            if prior_move > 0 and recent_move > prior_move * 2:
                score += 20
                factors.append(f"Acceleration: recent 3-bar move {recent_move*100:.1f}% vs prior {prior_move*100:.1f}%")
            elif recent_move > 0.05:
                score += 10
                factors.append(f"Strong recent momentum: +{recent_move*100:.1f}% last 3 bars")
        except Exception:
            pass

    # RVOL — extremely high RVOL near halt band is danger zone
    rv = float(rvol or 0.0)
    if rv >= 20 and dist_pct < 15:
        score += 20
        factors.append(f"RVOL {rv:.0f}x near halt band — explosive conditions")
    elif rv >= 10:
        score += 10
        factors.append(f"RVOL {rv:.0f}x — elevated volume pressure")

    # Float rotation — small float + high volume = fast halt
    if float_shares and float_shares > 0:
        # Use estimated today's volume from RVOL × avg volume (rough)
        # If float < 5M and RVOL > 5x, rotation is happening fast
        if float_shares < 5_000_000 and rv >= 5:
            score += 15
            factors.append(f"Small float {float_shares/1e6:.1f}M + RVOL {rv:.0f}x = rapid float rotation")

    # Current move magnitude
    move_pct = current_move_pct * 100
    if move_pct >= band_pct * 80 * 100:
        score += 15
        factors.append(f"Move {move_pct:.1f}% approaching LULD threshold {band_pct*100:.0f}%")

    score = min(100, max(0, score))

    if score >= 70:
        risk = "IMMINENT"
        advice = "Reduce size or set tight exit — halt likely within minutes"
    elif score >= 45:
        risk = "HIGH"
        advice = "Set sell alert at LULD band, be ready to exit fast"
    elif score >= 25:
        risk = "MODERATE"
        advice = "Monitor closely, have exit plan ready"
    else:
        risk = "LOW"
        advice = "Normal trading conditions"

    return {
        "score": round(score, 0),
        "risk": risk,
        "advice": advice,
        "luld_upper": round(luld_up, 3),
        "distance_to_band_pct": round(dist_pct, 2),
        "current_move_pct": round(move_pct, 2),
        "factors": factors,
    }


# ── Float Rotation Tracker ─────────────────────────────────────────────────────

def float_rotation(
    today_volume: float,
    float_shares: float | None,
    avg20_volume: float | None = None,
) -> dict[str, Any]:
    """
    How many times has today's volume turned over the float?
    Parabolic signal: >3x rotation in a session = explosive momentum.
    """
    if not float_shares or float_shares <= 0:
        return {"rotations": None, "signal": "unknown", "float_shares": None}

    rotations = today_volume / float_shares

    if rotations >= 5:
        signal = "EXTREME"
        label = f"{rotations:.1f}x float rotation — parabolic, halt risk very high"
        color = "red"
    elif rotations >= 3:
        signal = "PARABOLIC"
        label = f"{rotations:.1f}x float rotation — strong momentum, watch for exhaustion"
        color = "orange"
    elif rotations >= 1.5:
        signal = "ELEVATED"
        label = f"{rotations:.1f}x float rotation — above average, momentum play"
        color = "yellow"
    elif rotations >= 0.5:
        signal = "NORMAL"
        label = f"{rotations:.1f}x float rotation — normal activity"
        color = "green"
    else:
        signal = "LOW"
        label = f"{rotations:.2f}x float rotation — thin, low conviction"
        color = "gray"

    return {
        "rotations": round(rotations, 2),
        "signal": signal,
        "label": label,
        "color": color,
        "float_shares": float_shares,
        "today_volume": today_volume,
    }


# ── Distribution Detection ─────────────────────────────────────────────────────

def distribution_detection(
    trades: list[dict],
    bid: float | None = None,
    ask: float | None = None,
) -> dict[str, Any]:
    """
    Detects institutional selling into retail FOMO from T&S data.
    Analyzes uptick/downtick ratio, ask-side vs bid-side volume,
    and large print direction.

    trades: list of trade dicts with keys: price, size, tick (up/down/flat)
    """
    if not trades:
        return {"signal": "no_data", "score": 0, "label": "No T&S data"}

    total = len(trades)
    upticks   = sum(1 for t in trades if t.get("tick") == "up")
    downticks = sum(1 for t in trades if t.get("tick") == "down")

    up_vol   = sum(float(t.get("size", 0) or 0) for t in trades if t.get("tick") == "up")
    down_vol = sum(float(t.get("size", 0) or 0) for t in trades if t.get("tick") == "down")
    total_vol = up_vol + down_vol

    uptick_ratio = upticks / total if total > 0 else 0.5
    up_vol_ratio = up_vol / total_vol if total_vol > 0 else 0.5

    # Large prints (top 20% by size)
    sizes = sorted([float(t.get("size", 0) or 0) for t in trades], reverse=True)
    large_threshold = sizes[max(0, len(sizes) // 5)] if sizes else 0
    large_prints = [t for t in trades if float(t.get("size", 0) or 0) >= large_threshold and large_threshold > 0]
    large_up   = sum(1 for t in large_prints if t.get("tick") == "up")
    large_down = sum(1 for t in large_prints if t.get("tick") == "down")

    # Ask-side vs bid-side (if we have bid/ask)
    ask_side_vol = 0.0
    bid_side_vol = 0.0
    if bid and ask:
        mid = (bid + ask) / 2
        for t in trades:
            p = float(t.get("price", 0) or 0)
            sz = float(t.get("size", 0) or 0)
            if p >= mid:
                ask_side_vol += sz
            else:
                bid_side_vol += sz

    # Distribution score (0-100, higher = more distribution)
    score = 50.0  # neutral

    # Uptick ratio below 0.4 = more prints on downticks = selling
    if uptick_ratio < 0.35:
        score += 25
    elif uptick_ratio < 0.45:
        score += 12
    elif uptick_ratio > 0.65:
        score -= 20
    elif uptick_ratio > 0.55:
        score -= 10

    # Volume on downticks > upticks = distribution
    if up_vol_ratio < 0.35:
        score += 20
    elif up_vol_ratio < 0.45:
        score += 10
    elif up_vol_ratio > 0.65:
        score -= 15
    elif up_vol_ratio > 0.55:
        score -= 8

    # Large prints going down = institutions selling
    if large_prints:
        large_down_ratio = large_down / len(large_prints)
        if large_down_ratio > 0.7:
            score += 15
            signal_detail = "Large prints predominantly SELLING"
        elif large_down_ratio < 0.3:
            score -= 12
            signal_detail = "Large prints predominantly BUYING"
        else:
            signal_detail = "Large prints mixed"
    else:
        signal_detail = "No large prints detected"

    # Ask/bid side volume
    total_ab = ask_side_vol + bid_side_vol
    if total_ab > 0:
        ask_ratio = ask_side_vol / total_ab
        if ask_ratio < 0.35:
            score += 10
        elif ask_ratio > 0.65:
            score -= 10

    score = min(100, max(0, score))

    if score >= 70:
        signal = "DISTRIBUTING"
        label = "Institutions selling into retail — avoid long, consider short"
        color = "red"
    elif score >= 55:
        signal = "CAUTION"
        label = "Mixed tape, slight sell pressure — tight stop on longs"
        color = "orange"
    elif score <= 30:
        signal = "ACCUMULATING"
        label = "Strong buy-side pressure — long bias supported"
        color = "green"
    elif score <= 45:
        signal = "NEUTRAL_BULLISH"
        label = "Slight buy pressure — long bias ok"
        color = "yellow"
    else:
        signal = "NEUTRAL"
        label = "Balanced tape"
        color = "gray"

    return {
        "score": round(score, 0),
        "signal": signal,
        "label": label,
        "color": color,
        "uptick_ratio": round(uptick_ratio, 2),
        "up_vol_ratio": round(up_vol_ratio, 2),
        "large_print_detail": signal_detail,
        "total_trades": total,
    }


# ── Catalyst Decay Timer ───────────────────────────────────────────────────────

def catalyst_decay(
    catalyst_type: str,
    news_age_hours: float | None,
) -> dict[str, Any]:
    """
    How much of the catalyst's sustain window has elapsed?
    Returns time remaining and urgency level.
    """
    cat = str(catalyst_type or "unknown").lower()
    profile = _CATALYST_SUSTAIN.get(cat) or _CATALYST_SUSTAIN["unknown"]

    avg_mins   = profile["avg"]
    lo, hi     = profile["range"]
    label      = profile["label"]

    age_mins = (float(news_age_hours or 0) * 60.0)

    elapsed_pct = min(1.0, age_mins / avg_mins) if avg_mins > 0 else 1.0
    remaining_mins = max(0, avg_mins - age_mins)

    if elapsed_pct >= 1.0:
        urgency = "EXPIRED"
        advice = f"Catalyst window ({avg_mins}min avg) likely exhausted — fade risk elevated"
        color = "red"
    elif elapsed_pct >= 0.75:
        urgency = "FADING"
        advice = f"~{remaining_mins:.0f}min left in typical catalyst window — begin trailing stop"
        color = "orange"
    elif elapsed_pct >= 0.50:
        urgency = "MID"
        advice = f"~{remaining_mins:.0f}min remaining — watch for momentum shift"
        color = "yellow"
    else:
        urgency = "FRESH"
        advice = f"~{remaining_mins:.0f}min remaining in typical window — momentum should hold"
        color = "green"

    return {
        "catalyst_type": cat,
        "catalyst_label": label,
        "avg_sustain_mins": avg_mins,
        "sustain_range": profile["range"],
        "age_mins": round(age_mins, 0),
        "elapsed_pct": round(elapsed_pct * 100, 0),
        "remaining_mins": round(remaining_mins, 0),
        "urgency": urgency,
        "advice": advice,
        "color": color,
    }


# ── Pre-Market Trade Plan ──────────────────────────────────────────────────────

def build_premarket_trade_plan(
    symbol: str,
    price: float,
    prev_close: float,
    gap_pct: float,
    side: str,
    rvol: float | None = None,
    float_shares: float | None = None,
    above_vwap: bool | None = None,
    avg20_dollar_vol: float | None = None,
    today_volume: float | None = None,
    catalyst_type: str = "unknown",
    news_age_hours: float | None = None,
    pm_bars: list[dict] | None = None,
    trades: list[dict] | None = None,
    bid: float | None = None,
    ask: float | None = None,
    risk_dollars: float = 100.0,
) -> dict[str, Any]:
    """
    Full pre-market trade plan for a single symbol.
    Combines all intelligence signals into a concrete recommendation.
    """
    fade = gap_fade_probability(
        gap_pct=gap_pct,
        catalyst_type=catalyst_type,
        float_shares=float_shares,
        rvol=rvol,
        above_vwap=above_vwap,
        pm_bars=len(pm_bars) if pm_bars else None,
    )

    halt = halt_prediction_score(
        price=price,
        prev_close=prev_close,
        pm_bars=pm_bars,
        rvol=rvol,
        float_shares=float_shares,
    )

    rotation = float_rotation(
        today_volume=float(today_volume or 0),
        float_shares=float_shares,
    )

    dist = distribution_detection(
        trades=trades or [],
        bid=bid,
        ask=ask,
    )

    decay = catalyst_decay(
        catalyst_type=catalyst_type,
        news_age_hours=news_age_hours,
    )

    # Entry / stop / target based on side and PM levels
    if price and prev_close:
        if side == "long":
            entry  = round(price * 1.001, 3)   # just above current PM price
            stop   = round(max(prev_close, price * 0.93), 3)  # 7% or prev close
            risk   = max(entry - stop, price * 0.03)
            shares = max(1, int(risk_dollars / risk)) if risk > 0 else 0
            target_2r = round(entry + 2 * risk, 3)
            target_3r = round(entry + 3 * risk, 3)
        else:
            entry  = round(price * 0.999, 3)
            stop   = round(min(price * 1.07, price + price * 0.07), 3)
            risk   = max(stop - entry, price * 0.03)
            shares = max(1, int(risk_dollars / risk)) if risk > 0 else 0
            target_2r = round(entry - 2 * risk, 3)
            target_3r = round(entry - 3 * risk, 3)
    else:
        entry = stop = target_2r = target_3r = None
        shares = 0
        risk = 0

    # Overall trade quality score (0-100)
    quality = 50.0
    if side == "long":
        quality -= (fade["fade_probability"] - 0.5) * 60   # high fade = lower quality
    else:
        quality += (fade["fade_probability"] - 0.5) * 60   # high fade on short = better

    if halt["score"] > 50:
        quality -= 15  # halt risk hurts both sides during entry
    if rotation["rotations"] and rotation["rotations"] > 3:
        quality += 10 if side == "short" else -5
    if dist["signal"] == "DISTRIBUTING" and side == "long":
        quality -= 20
    elif dist["signal"] == "ACCUMULATING" and side == "long":
        quality += 15
    if decay["urgency"] == "EXPIRED":
        quality -= 20
    elif decay["urgency"] == "FRESH":
        quality += 10

    quality = min(100, max(0, quality))

    if quality >= 75:
        rec = "STRONG"
        rec_color = "green"
    elif quality >= 55:
        rec = "TAKE"
        rec_color = "yellow"
    elif quality >= 40:
        rec = "WATCH"
        rec_color = "orange"
    else:
        rec = "SKIP"
        rec_color = "red"

    return {
        "symbol":        symbol,
        "price":         price,
        "prev_close":    prev_close,
        "gap_pct":       round(gap_pct, 2),
        "side":          side,
        "recommendation": rec,
        "rec_color":     rec_color,
        "quality_score": round(quality, 0),
        "entry":         entry,
        "stop":          stop,
        "target_2r":     target_2r,
        "target_3r":     target_3r,
        "shares":        shares,
        "risk_per_share": round(risk, 3) if risk else None,
        "fade":          fade,
        "halt":          halt,
        "rotation":      rotation,
        "distribution":  dist,
        "catalyst_decay": decay,
    }
