from __future__ import annotations

from typing import Any


def candidate_sort_score(row: dict[str, Any] | None) -> float:
    if not isinstance(row, dict):
        return 0.0
    for key in ("combined_score", "score", "ml_score", "confidence_score", "today_dollar_vol"):
        try:
            value = row.get(key)
            if value is not None:
                return float(value)
        except Exception:
            continue
    return 0.0


def sort_candidate_rows(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out = [row for row in (rows or []) if isinstance(row, dict)]
    try:
        out.sort(key=candidate_sort_score, reverse=True)
    except Exception:
        return out
    return out


def select_primary_candidates(result: dict[str, Any] | None, *, limit: int | None = None) -> tuple[list[dict[str, Any]], str, str | None]:
    payload = result or {}
    strict_rows = sort_candidate_rows(list(payload.get("candidates") or []))
    seed_rows = sort_candidate_rows(list(payload.get("seed_candidates") or []))

    if strict_rows:
        rows = strict_rows
        if limit is not None:
            rows = rows[: max(0, int(limit))]
        return rows, "trade_ready", None

    if seed_rows:
        rows = seed_rows
        if limit is not None:
            rows = rows[: max(0, int(limit))]
        return rows, "monitor_watch", "No trade-ready names passed. Showing the best monitor/watch seeds instead."

    return [], "empty", "No trade-ready or monitor-worthy names surfaced in this run."


def build_zero_result_diagnostics(result: dict[str, Any] | None, *, limit: int = 5) -> dict[str, Any]:
    payload = result or {}
    reject_counts = payload.get("reject_counts") or {}
    top_rejection_reasons: list[tuple[str, int]] = []
    if isinstance(reject_counts, dict):
        try:
            top_rejection_reasons = sorted(
                (
                    (str(key), int(value))
                    for key, value in reject_counts.items()
                    if value is not None and int(value) > 0
                ),
                key=lambda item: item[1],
                reverse=True,
            )[: max(1, int(limit))]
        except Exception:
            top_rejection_reasons = []

    primary_rows, primary_mode, primary_message = select_primary_candidates(payload, limit=limit)
    return {
        "trade_ready_count": int(payload.get("candidates_total") or payload.get("trade_ready_total") or 0),
        "monitor_seed_count": int(payload.get("seed_candidates_total") or 0),
        "rejected_count": int(payload.get("rejected_total") or 0),
        "shortlisted_count": int(payload.get("shortlisted") or 0),
        "primary_mode": primary_mode,
        "primary_message": primary_message,
        "monitor_fallback_active": primary_mode == "monitor_watch",
        "trade_ready_empty": int(payload.get("candidates_total") or payload.get("trade_ready_total") or 0) == 0,
        "prefilter_counts": dict(payload.get("prefilter_counts") or {}),
        "top_rejection_reasons": top_rejection_reasons,
        "primary_symbols": [str(row.get("symbol") or "").upper() for row in primary_rows if str(row.get("symbol") or "").strip()],
    }


def build_primary_fallback_view(result: dict[str, Any] | None, *, limit: int | None = None) -> dict[str, Any]:
    payload = result or {}
    trade_ready_rows = sort_candidate_rows(list(payload.get("candidates") or []))
    fallback_rows = sort_candidate_rows(list(payload.get("seed_candidates") or []))
    primary_rows, primary_mode, primary_message = select_primary_candidates(payload, limit=limit)
    diagnostics = build_zero_result_diagnostics(payload, limit=(limit or 5))
    return {
        "primary_candidates": primary_rows,
        "primary_mode": primary_mode,
        "primary_message": primary_message,
        "trade_ready_candidates": trade_ready_rows[: max(0, int(limit))] if limit is not None else trade_ready_rows,
        "trade_ready_candidate_count": int(
            payload.get("tradable_now_total")
            or payload.get("trade_ready_total")
            or payload.get("candidates_total")
            or 0
        ),
        "fallback_candidates": fallback_rows[: max(0, int(limit))] if limit is not None else fallback_rows,
        "fallback_candidate_count": int(payload.get("seed_candidates_total") or 0),
        "zero_result_diagnostics": diagnostics,
    }
