"""
Plan Confidence Scoring — rule-based v1.

Scores a watchlist-mode MonitorSymbolState against the trader's
pre-defined plan (entry, stop, target) rather than ORB trigger logic.

Returns a 0–100 "Plan Readiness" score + a breakdown dict.

As labeled plan_snapshot data accumulates (see RuntimeStateStore.log_plan_snapshot
and label_plan_outcome), this rule-based scorer will be replaced by an ML model
trained on real outcome labels (target_reached / stopped_out / neither).
The feature schema here intentionally matches the plan_snapshots table columns
so the trained model ingests the same inputs.
"""
from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from monitor.live_monitor import MonitorSymbolState

ET = ZoneInfo("America/New_York")

# Ticker → sector ETF mapping for concentration penalty
TICKER_SECTOR: dict[str, str] = {
    # XLK — Technology
    **{s: "XLK" for s in ["AAPL","MSFT","NVDA","AMD","INTC","QCOM","AVGO","MU","AMAT","LRCX","KLAC",
                           "MRVL","MCHP","TXN","ADI","NXPI","ON","SMCI","PLTR","SNOW","CRWD","ZS",
                           "PANW","FTNT","NET","DDOG","MDB","TEAM","NOW","CRM","ORCL","IBM","INTU",
                           "ADBE","APP","COIN","MSTR","TSM","SOXL","SOXS","SMH","SOXX"]},
    # XLC — Communication Services
    **{s: "XLC" for s in ["META","GOOGL","GOOG","NFLX","CMCSA","DIS","T","VZ","TMUS","EA","TTWO",
                           "RBLX","SNAP","PINS","CHTR","PARA","FOX","FOXA","WBD","SPOT"]},
    # XLY — Consumer Discretionary
    **{s: "XLY" for s in ["AMZN","TSLA","HD","LOW","NKE","MCD","SBUX","TGT","COST","BKNG","MAR",
                           "HLT","F","GM","RIVN","LCID","SHOP","W","ETSY","UBER","LYFT","ABNB","BBY",
                           "DRI","YUM","CMG","DECK","LULU","RH","PTON"]},
    # XLP — Consumer Staples
    **{s: "XLP" for s in ["WMT","PG","KO","PEP","PM","MO","MDLZ","CL","KMB","CHD","EL","GIS",
                           "K","SJM","CAG","MKC","TSN","HRL"]},
    # XLE — Energy
    **{s: "XLE" for s in ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","HAL","DVN","FANG",
                           "APA","EQT","AR","RRC","CHK","SM","CTRA","MRO","HES","PXD","BKR","NOV",
                           "DRIP","GUSH","ERX","ERY","UCO","SCO"]},
    # XLF — Financials
    **{s: "XLF" for s in ["JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","V","MA","COF","DFS",
                           "SYF","USB","PNC","TFC","MET","PRU","AFL","BRK.B","BRKB","FRC","SVB",
                           "PYPL","SQ","XYZ","AFRM","SOFI","UPST","LC","NU","HOOD","IBKR","IBKRK"]},
    # XLV — Health Care
    **{s: "XLV" for s in ["JNJ","UNH","LLY","PFE","ABBV","MRK","BMY","AMGN","GILD","REGN","BIIB",
                           "MRNA","NVAX","MDT","ABT","SYK","BSX","ISRG","HUM","CVS","CI","ELV",
                           "ZBH","BAX","BDX","IQV","DXCM","PODD","TDOC","HIMS","ACMR","NVAX"]},
    # XLI — Industrials
    **{s: "XLI" for s in ["BA","CAT","DE","GE","HON","MMM","UPS","FDX","LMT","RTX","NOC","GD","EMR",
                           "ETN","ITW","PH","ROK","CARR","OTIS","XYL","RKLB","LUNR","RCM","WAB",
                           "GWW","FAST","AME","GNRC","TDG","AXON"]},
    # XLB — Materials
    **{s: "XLB" for s in ["LIN","APD","ECL","SHW","FCX","NUE","STLD","AA","ALB","MP","CTRA","CF",
                           "MOS","NEM","GOLD","AEM","WPM","RGLD","AG","HL","CDE","PAAS"]},
    # XLRE — Real Estate
    **{s: "XLRE" for s in ["AMT","PLD","EQIX","CCI","SPG","O","VICI","WPC","EXR","AVB","EQR","ARE",
                            "BXP","SLG","KIM","REG","FRT","MAC","IIPR"]},
    # XLU — Utilities
    **{s: "XLU" for s in ["NEE","DUK","SO","D","AEP","XEL","PCG","EIX","ES","ETR","ED","EXC","PPL",
                           "AWK","CMS","CNP","WEC","AES"]},
}

# Session archetype → per-component weight multipliers (renormalized after application)
_ARCHETYPE_WEIGHT_MODS: dict[str, dict[str, float]] = {
    # Big gap day: momentum, tape speed, and order flow dominate; time-of-day and spread matter less
    "gap_and_go": {"velocity": 1.5, "tape": 1.3, "order_flow": 1.4, "proximity": 1.1, "time": 0.7, "spread": 0.8},
    # Trend day: VWAP alignment, macro context, and sustained order flow are the edge
    "trend_day":  {"vwap": 1.4, "context": 1.3, "order_flow": 1.2, "time": 1.2, "velocity": 1.1, "spread": 0.9},
    # Choppy day: precision fills matter most; order flow is noisy — don't chase
    "chop":       {"spread": 1.5, "proximity": 1.3, "time": 1.3, "velocity": 0.5, "order_flow": 0.6, "tape": 0.8},
    # Mixed/unknown: no adjustments
    "mixed":      {},
}

# Same-sector open position concentration penalty multipliers
_SECTOR_CONCENTRATION_PENALTY: dict[int, float] = {
    0: 1.00,   # no other same-sector positions — full score
    1: 0.92,   # 1 other same-sector — mild discount
    2: 0.82,   # 2 others — meaningful discount
}
_SECTOR_CONCENTRATION_PENALTY_3_PLUS = 0.70


# Time-bucket quality weights — open impulse is the premium window
_TIME_BUCKET_QUALITY: dict[str, float] = {
    "pre_market":       0.55,
    "open_impulse":     1.00,
    "morning_trend":    0.85,
    "mid_day":          0.40,
    "afternoon_trend":  0.65,
    "late_day":         0.55,
    "after_hours":      0.30,
    "unknown":          0.50,
}

# Component weights — must sum to 1.0
# Tape leads: live order flow is the only real-time confirmation that the setup is active.
# Proximity matters but alone means nothing if tape is dead.
_W_PROXIMITY   = 0.18
_W_TAPE        = 0.22
_W_SPREAD      = 0.10
_W_TIME        = 0.10
_W_VWAP        = 0.08
_W_CATALYST    = 0.07
_W_CONTEXT     = 0.06  # macro risk-on/off alignment
_W_VELOCITY    = 0.09  # price approach velocity toward entry
_W_ORDER_FLOW  = 0.10  # L2 imbalance + trade aggressor bias


def compute_plan_readiness(st: "MonitorSymbolState") -> tuple[float, dict[str, Any]]:
    """
    Returns (score_0_to_100, breakdown_dict).
    Only meaningful when playbook == 'watchlist_plan'.
    """
    breakdown: dict[str, Any] = {}
    components: dict[str, float] = {}

    entry = st.entry
    stop  = st.stop_loss
    price = st.price
    side  = str(st.best_side or "long").lower()

    if entry is None or stop is None or price is None:
        return 0.0, {"error": "missing_plan_levels"}

    risk = abs(entry - stop)
    if risk < 1e-6:
        return 0.0, {"error": "zero_risk"}

    # ── 1. Entry Proximity ────────────────────────────────────────────────────
    # entry_distance_r: signed units of R from price to entry
    #   positive = price approaching entry (long: below entry, short: above)
    #   negative = price already past entry
    if side == "long":
        entry_distance_r = (entry - price) / risk
    else:
        entry_distance_r = (price - entry) / risk

    if entry_distance_r >= 0:
        # approaching — peak when distance ≈ 0
        proximity_raw = max(0.0, 1.0 - (entry_distance_r ** 0.7) * 0.65)
    else:
        # past entry — still valid, decays quickly
        proximity_raw = max(0.0, 0.90 - abs(entry_distance_r) * 0.45)

    components["proximity"] = proximity_raw
    breakdown["entry_distance_r"] = round(entry_distance_r, 3)
    breakdown["proximity_score"]  = round(proximity_raw, 3)

    # ── 2. Tape Alignment ─────────────────────────────────────────────────────
    tape_raw = 0.50
    if st.tape_live:
        tape_raw += 0.25
    vwap_delta = st.vwap_delta_pct_live
    if vwap_delta is not None:
        aligned = (side == "long" and vwap_delta > 0) or (side == "short" and vwap_delta < 0)
        strong  = abs(vwap_delta) > 0.2
        if aligned and strong:
            tape_raw += 0.20
        elif aligned:
            tape_raw += 0.10
        else:
            tape_raw -= 0.15
    tape_raw = max(0.0, min(1.0, tape_raw))
    components["tape"] = tape_raw
    breakdown["tape_score"] = round(tape_raw, 3)
    breakdown["tape_live"]  = st.tape_live

    # ── 3. Spread Health ──────────────────────────────────────────────────────
    spread_pct  = float(st.spread_pct or 0.0)  # expected in percent (e.g. 0.5 = 0.5%), not decimal
    risk_pct    = (risk / entry) * 100.0 if entry > 0 else 1.0
    spread_vs_r = spread_pct / max(risk_pct, 0.01)
    if   spread_vs_r <= 0.05: spread_raw = 1.00
    elif spread_vs_r <= 0.15: spread_raw = 0.85
    elif spread_vs_r <= 0.30: spread_raw = 0.60
    elif spread_vs_r <= 0.50: spread_raw = 0.35
    else:                     spread_raw = 0.10
    components["spread"] = spread_raw
    breakdown["spread_vs_risk"] = round(spread_vs_r, 3)
    breakdown["spread_score"]   = round(spread_raw, 3)

    # ── 4. Time-of-Day Quality ────────────────────────────────────────────────
    time_raw = _TIME_BUCKET_QUALITY.get(str(st.time_of_day_bucket or "unknown"), 0.5)
    components["time"] = time_raw
    breakdown["time_bucket"] = st.time_of_day_bucket
    breakdown["time_score"]  = round(time_raw, 3)

    # ── 5. VWAP Alignment ─────────────────────────────────────────────────────
    above_vwap = st.above_vwap_live
    if above_vwap is None:
        vwap_raw = 0.50
    elif (side == "long" and above_vwap) or (side == "short" and not above_vwap):
        vwap_raw = 1.00
    else:
        vwap_raw = 0.25
    components["vwap"] = vwap_raw
    breakdown["above_vwap"] = above_vwap
    breakdown["vwap_score"] = round(vwap_raw, 3)

    # ── 6. Catalyst ───────────────────────────────────────────────────────────
    cat_score = float(st.catalyst_score or 0.0)
    cat_fresh = st.catalyst_freshness_hours
    if cat_score > 0:
        if   cat_fresh is None:  freshness_factor = 0.80
        elif cat_fresh <= 2:     freshness_factor = 1.00
        elif cat_fresh <= 6:     freshness_factor = 0.80
        elif cat_fresh <= 12:    freshness_factor = 0.55
        elif cat_fresh <= 24:    freshness_factor = 0.30
        else:                    freshness_factor = 0.10
        catalyst_raw = min(1.0, cat_score * freshness_factor)
    else:
        catalyst_raw = 0.40  # no catalyst is neutral — not a dealbreaker for watchlist names

    components["catalyst"] = catalyst_raw
    breakdown["catalyst_score"] = round(cat_score, 3)
    breakdown["catalyst_raw"]   = round(catalyst_raw, 3)

    # ── 7. Macro Context (risk_on_score) ──────────────────────────────────────
    # risk_on_score is an average of SPY/QQQ daily moves + breadth, range ~-1.5 to +1.5
    # For longs: positive score = tailwind; for shorts: negative score = tailwind
    ctx = float(st.context_score or 0.0)
    if side == "long":
        ctx_aligned = ctx
    else:
        ctx_aligned = -ctx  # flip: negative risk_on is tailwind for shorts
    if   ctx_aligned >= 0.50: context_raw = 1.00
    elif ctx_aligned >= 0.20: context_raw = 0.80
    elif ctx_aligned >= 0.00: context_raw = 0.65
    elif ctx_aligned >= -0.20: context_raw = 0.45
    else:                      context_raw = 0.20
    components["context"] = context_raw
    breakdown["risk_on_score"]  = round(ctx, 3)
    breakdown["context_score"]  = round(context_raw, 3)

    # ── 8. Entry Velocity ─────────────────────────────────────────────────────
    # Measures how fast price is approaching the entry level (R per minute).
    # Positive = approaching entry; negative = moving away.
    velocity_raw = 0.55  # neutral default (no history yet)
    ph = getattr(st, "price_history", None)
    now_ts = time.time()
    if ph and len(ph) >= 2 and risk > 1e-6:
        # Find the oldest sample within 60 seconds; ph must be list of (price, unix_seconds)
        cutoff = now_ts - 60.0
        old_samples = [(p, t) for p, t in ph if isinstance(p, (int, float)) and isinstance(t, (int, float)) and t >= cutoff]
        if len(old_samples) >= 2:
            p_old, t_old = old_samples[0]
            p_new, t_new = old_samples[-1]
            if t_new > t_old:  # guard against unsorted or same-timestamp samples
                elapsed_min = (t_new - t_old) / 60.0
                raw_move_per_min = (p_new - p_old) / elapsed_min  # $/min
                # Approach velocity: positive when price moves toward entry
                if side == "long":
                    approach_v = raw_move_per_min / risk   # rising price = approaching for long
                else:
                    approach_v = -raw_move_per_min / risk  # falling price = approaching for short
                if   approach_v >= 1.5: velocity_raw = 1.00  # strong surge toward entry
                elif approach_v >= 0.5: velocity_raw = 0.85  # solid approach
                elif approach_v >= 0.1: velocity_raw = 0.70  # gradual approach — good
                elif approach_v >= -0.1: velocity_raw = 0.55 # flat — neutral, waiting
                elif approach_v >= -0.5: velocity_raw = 0.35 # drifting away — caution
                else:                    velocity_raw = 0.15  # retreating — unfavorable
                breakdown["approach_velocity_r_per_min"] = round(approach_v, 3)
    components["velocity"] = velocity_raw
    breakdown["velocity_score"] = round(velocity_raw, 3)

    # ── 9. Order Flow Imbalance ───────────────────────────────────────────────
    # Two signals combined:
    #   L2 imbalance (60%): (bid_size - ask_size) / total — who has more size at top of book
    #   Aggressor bias (40%): fraction of aggressive prints that are buys vs sells (last 60s)
    # For longs: buy pressure at bid + buy aggressors = high score
    # For shorts: sell pressure at ask + sell aggressors = high score

    # L2 imbalance component
    l2_imbalance = getattr(st, "bid_ask_imbalance", None)
    if l2_imbalance is not None:
        aligned_l2 = l2_imbalance if side == "long" else -l2_imbalance
        if   aligned_l2 >= 0.40: l2_raw = 1.00
        elif aligned_l2 >= 0.15: l2_raw = 0.80
        elif aligned_l2 >= -0.10: l2_raw = 0.60
        elif aligned_l2 >= -0.30: l2_raw = 0.35
        else:                      l2_raw = 0.15
    else:
        l2_raw = 0.50  # no quote data yet — neutral

    # Rolling aggressor bias component
    agg_hist = getattr(st, "aggressor_history", None)
    agg_raw = 0.50  # neutral default
    aggressor_buy_ratio = None
    if agg_hist and len(agg_hist) >= 3:
        _agg_cutoff = time.time() - 60.0
        recent = [v for v, t in agg_hist if t >= _agg_cutoff]
        if len(recent) >= 3:
            buys = sum(1 for v in recent if v > 0)
            buy_ratio = buys / len(recent)
            aligned_ratio = buy_ratio if side == "long" else (1.0 - buy_ratio)
            if   aligned_ratio >= 0.70: agg_raw = 1.00
            elif aligned_ratio >= 0.55: agg_raw = 0.80
            elif aligned_ratio >= 0.45: agg_raw = 0.60
            elif aligned_ratio >= 0.35: agg_raw = 0.40
            else:                        agg_raw = 0.20
            aggressor_buy_ratio = round(buy_ratio, 3)

    order_flow_raw = round(0.60 * l2_raw + 0.40 * agg_raw, 3)
    components["order_flow"] = order_flow_raw
    breakdown["bid_ask_imbalance"]   = round(l2_imbalance, 3) if l2_imbalance is not None else None
    breakdown["aggressor_buy_ratio"] = aggressor_buy_ratio
    breakdown["order_flow_score"]    = order_flow_raw

    # ── Archetype-adjusted weights ────────────────────────────────────────────
    archetype = str(getattr(st, "session_archetype", "mixed") or "mixed")
    base_weights = {
        "proximity": _W_PROXIMITY, "tape": _W_TAPE, "spread": _W_SPREAD,
        "time": _W_TIME, "vwap": _W_VWAP, "catalyst": _W_CATALYST,
        "context": _W_CONTEXT, "velocity": _W_VELOCITY, "order_flow": _W_ORDER_FLOW,
    }
    mods = _ARCHETYPE_WEIGHT_MODS.get(archetype, {})
    if mods:
        adjusted = {k: v * mods.get(k, 1.0) for k, v in base_weights.items()}
        total = sum(adjusted.values())
        weights = {k: v / total for k, v in adjusted.items()}  # renormalize to sum=1
    else:
        weights = base_weights
    breakdown["session_archetype"] = archetype
    breakdown["weights_used"] = {k: round(v, 4) for k, v in weights.items()}

    # ── Weighted sum → 0–100 ──────────────────────────────────────────────────
    raw = sum(components[k] * weights[k] for k in components)
    score = round(raw * 100.0, 1)

    # ── Sector concentration penalty ──────────────────────────────────────────
    same_sector_count = int(getattr(st, "same_sector_count", 0) or 0)
    concentration_mult = _SECTOR_CONCENTRATION_PENALTY.get(
        same_sector_count, _SECTOR_CONCENTRATION_PENALTY_3_PLUS
    )
    if concentration_mult < 1.0:
        score = round(score * concentration_mult, 1)
        breakdown["sector_concentration_count"] = same_sector_count
        breakdown["sector_concentration_penalty"] = round(1.0 - concentration_mult, 2)

    breakdown["components"]  = {k: round(v, 3) for k, v in components.items()}
    breakdown["final_score"] = score
    return score, breakdown


def plan_readiness_grade(score: float) -> str:
    if score >= 75: return "A"
    if score >= 60: return "B"
    if score >= 45: return "C"
    return "D"


def extract_plan_snapshot_features(
    st: "MonitorSymbolState",
    *,
    session_date: str | None = None,
    snapshot_trigger: str = "periodic",
) -> dict[str, Any]:
    """
    Extract the full feature vector for a plan_snapshot DB record.
    Stored at snapshot time; outcome (target_reached/stopped_out/neither) labeled later.
    This is the training dataset for the future ML model.
    """
    entry = st.entry
    stop  = st.stop_loss
    price = st.price
    side  = str(st.best_side or "long").lower()
    risk  = abs((entry or 0.0) - (stop or 0.0)) if entry is not None and stop is not None else None

    entry_distance_r   = None
    entry_distance_pct = None
    if entry is not None and stop is not None and price is not None and risk and risk > 1e-6:
        if side == "long":
            entry_distance_r = (entry - price) / risk
        else:
            entry_distance_r = (price - entry) / risk
        entry_distance_pct = abs(price - entry) / entry * 100.0 if entry > 0 else None

    spread_pct      = st.spread_pct
    risk_pct        = (risk / entry * 100.0) if risk and entry else None
    spread_vs_risk  = (spread_pct / max(risk_pct, 0.01)) if spread_pct is not None and risk_pct else None

    plan_readiness, breakdown = compute_plan_readiness(st)

    if session_date is None:
        session_date = datetime.now(tz=ET).date().isoformat()

    return {
        "session_date":             session_date,
        "symbol":                   st.symbol,
        "snapshot_trigger":         snapshot_trigger,
        "plan_side":                side,
        "plan_entry":               entry,
        "plan_stop":                stop,
        "plan_target":              st.target_2r,
        "plan_risk_per_share":      risk,
        "plan_risk_pct":            risk_pct,
        "current_price":            price,
        "entry_distance_r":         round(entry_distance_r, 4) if entry_distance_r is not None else None,
        "entry_distance_pct":       round(entry_distance_pct, 4) if entry_distance_pct is not None else None,
        "spread_pct":               spread_pct,
        "spread_vs_risk_pct":       round(spread_vs_risk, 4) if spread_vs_risk is not None else None,
        "tape_live":                int(bool(st.tape_live)),
        "above_vwap":               (1 if st.above_vwap_live else 0) if st.above_vwap_live is not None else None,
        "vwap_delta_pct":           st.vwap_delta_pct_live,
        "time_bucket":              st.time_of_day_bucket,
        "catalyst_score":           st.catalyst_score,
        "catalyst_freshness_hours": st.catalyst_freshness_hours,
        "context_score":            st.context_score,
        "bid_ask_imbalance":        getattr(st, "bid_ask_imbalance", None),
        "order_flow_score":         getattr(st, "order_flow_score", None),
        "entry_velocity_r_per_min": breakdown.get("approach_velocity_r_per_min"),
        "ml_score":                 st.ml_score,
        "combined_score":           st.combined_score,
        "plan_readiness_score":     plan_readiness,
        "plan_readiness_breakdown": breakdown,
    }
