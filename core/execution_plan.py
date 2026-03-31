from __future__ import annotations

from typing import Any


def build_plan_state(
    *,
    side: str,
    last_price: float | None,
    stop: float | None,
    target_2r: float | None,
    p_2r_30m: float | None,
    chase_r: float | None,
    vwap_delta_pct: float | None,
    trend_state: str | None,
    go_hint_fn: Any = None,
) -> tuple[str, list[str], str | None]:
    risk_per_share = (abs(last_price - stop) if (last_price is not None and stop is not None) else None)
    risk_pct = None
    two_r_pct = None
    if risk_per_share is not None and last_price is not None and float(last_price) > 0:
        risk_pct = (float(risk_per_share) / float(last_price)) * 100.0
        if target_2r is not None:
            two_r_pct = (abs(float(target_2r) - float(last_price)) / float(last_price)) * 100.0

    notes: list[str] = []
    state = "WAIT"

    if last_price is not None and float(last_price) >= 30.0:
        state = "PASS"
        notes.append("price>=30")
    if risk_pct is not None and risk_pct > 12.0:
        state = "PASS"
        notes.append("risk_pct>12%")
    if two_r_pct is not None and two_r_pct > 25.0:
        state = "PASS"
        notes.append("2R_move>25%")

    if state != "PASS":
        if p_2r_30m is None:
            state = "WAIT"
            notes.append("ml_unavailable")
        else:
            if float(p_2r_30m) < 0.20:
                state = "PASS"
                notes.append("p<0.20")
            elif float(p_2r_30m) >= 0.24:
                state = "GO"
                notes.append("p>=0.24")
            else:
                state = "WAIT"
                notes.append("0.20<=p<0.24")

    if state == "GO":
        if chase_r is not None and float(chase_r) > 1.25:
            state = "WAIT"
            notes.append("chaseR>1.25")

        if vwap_delta_pct is not None:
            try:
                if abs(float(vwap_delta_pct)) > 4.0:
                    state = "WAIT"
                    notes.append("vwap_ext>4%")
            except Exception:
                pass

        if trend_state is not None:
            ts = str(trend_state)
            if ts in {"chop", "—", "none"}:
                state = "WAIT"
                notes.append("trend_chop")
            if side == "long" and ts in {"down", "lost_vwap"}:
                state = "WAIT"
                notes.append("trend_against")
            if side == "short" and ts in {"up", "reclaim_vwap"}:
                state = "WAIT"
                notes.append("trend_against")

    hint_line = None
    if go_hint_fn is not None:
        try:
            if state != "GO" and stop is not None and last_price is not None:
                hint_line = go_hint_fn(side, float(last_price), float(stop), max_risk_pct=12.0)
        except Exception:
            hint_line = None

    return state, notes, hint_line
