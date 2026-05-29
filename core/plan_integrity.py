"""
Plan integrity validator.

Every plan that Kingdom surfaces to the user must pass these checks.
If ANY check fails, the plan is structurally corrupt and must never show GO.

Call validate_plan() before displaying or acting on any plan.
Call validate_bars_input() before using bar data to assign direction.
Call audit_plan_snapshots() to scan the DB for historical corruption.
"""
from __future__ import annotations

import pandas as _pd
from dataclasses import dataclass, field
from typing import Any


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class IntegrityResult:
    valid: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


# ── Bar-input guard ───────────────────────────────────────────────────────────

def validate_bars_input(
    bars: "_pd.DataFrame",
    *,
    symbol: str = "?",
    require_sorted: bool = True,
    min_bars: int = 1,
) -> "IntegrityResult":
    """
    Validate a bar DataFrame before it is used to assign trade direction.

    This catches the upstream input bugs that plan-level geometry checks CANNOT
    catch: an unsorted frame makes iloc[-1] return the wrong bar → wrong gap_pct
    → wrong side.  The resulting plan looks geometrically valid (stop is below
    entry for the assigned long, above for the assigned short) but is semantically
    inverted.

    Checks:
      1. DataFrame is not None / empty
      2. Minimum bar count satisfied
      3. Index is chronologically ascending (sorted)
    """
    violations: list[str] = []
    warnings: list[str] = []

    if bars is None or (hasattr(bars, "empty") and bars.empty):
        violations.append(f"[{symbol}] bars DataFrame is None or empty")
        return IntegrityResult(valid=False, violations=violations, warnings=warnings)

    if len(bars) < min_bars:
        violations.append(
            f"[{symbol}] insufficient bars: got {len(bars)}, need ≥ {min_bars}"
        )

    if require_sorted and len(bars) >= 2:
        if not bars.index.is_monotonic_increasing:
            violations.append(
                f"[{symbol}] bars index is NOT chronologically ascending — "
                f"call sort_index() before using iloc[-1] for direction"
            )

    return IntegrityResult(valid=len(violations) == 0, violations=violations, warnings=warnings)


# ── Core validator ─────────────────────────────────────────────────────────────

def validate_plan(
    *,
    side: str | None,
    entry: float | None,
    stop: float | None,
    target: float | None,
    current_price: float | None,
    chase_r: float | None = None,
    risk_pct: float | None = None,
    symbol: str = "?",
) -> IntegrityResult:
    """
    Validate a single plan for structural integrity.

    Checks (any failure → valid=False, plan must PASS):
      1. Required fields present
      2. Side is 'long' or 'short'
      3. Stop and target are on the correct side of entry
      4. Current price has not already blown through the stop
      5. Entry/stop/target are all positive
      6. Risk is not degenerate (stop == entry)
      7. Chase distance is not excessive
      8. Risk% is within bounds
    """
    violations: list[str] = []
    warnings: list[str] = []

    # 1. Required fields
    if side is None:
        violations.append(f"[{symbol}] side is None")
    if entry is None:
        violations.append(f"[{symbol}] entry is None")
    if stop is None:
        violations.append(f"[{symbol}] stop is None")

    if violations:
        return IntegrityResult(valid=False, violations=violations, warnings=warnings)

    side = str(side).strip().lower()
    entry = float(entry)
    stop = float(stop)

    # 2. Side must be long or short
    if side not in ("long", "short"):
        violations.append(f"[{symbol}] invalid side '{side}' — must be 'long' or 'short'")
        return IntegrityResult(valid=False, violations=violations, warnings=warnings)

    # 3. Stop on correct side of entry (most critical check — catches the inversion bug)
    if side == "long" and stop >= entry:
        violations.append(
            f"[{symbol}] INVERTED LONG: stop ({stop:.4f}) >= entry ({entry:.4f}) "
            f"— stop must be BELOW entry for a long"
        )
    if side == "short" and stop <= entry:
        violations.append(
            f"[{symbol}] INVERTED SHORT: stop ({stop:.4f}) <= entry ({entry:.4f}) "
            f"— stop must be ABOVE entry for a short"
        )

    # Target on correct side of entry
    if target is not None:
        t = float(target)
        if side == "long" and t <= entry:
            violations.append(
                f"[{symbol}] INVERTED LONG TARGET: target ({t:.4f}) <= entry ({entry:.4f}) "
                f"— target must be ABOVE entry for a long"
            )
        if side == "short" and t >= entry:
            violations.append(
                f"[{symbol}] INVERTED SHORT TARGET: target ({t:.4f}) >= entry ({entry:.4f}) "
                f"— target must be BELOW entry for a short"
            )

    # 4. Price already past stop
    if current_price is not None:
        px = float(current_price)
        if side == "long" and px <= stop:
            violations.append(
                f"[{symbol}] PRICE BELOW STOP: price ({px:.4f}) <= stop ({stop:.4f}) — setup is dead"
            )
        if side == "short" and px >= stop:
            violations.append(
                f"[{symbol}] PRICE ABOVE STOP: price ({px:.4f}) >= stop ({stop:.4f}) — setup is dead"
            )

    # 5. All prices positive
    for label, val in [("entry", entry), ("stop", stop)]:
        if val <= 0:
            violations.append(f"[{symbol}] {label} ({val:.4f}) is not positive")
    if target is not None and float(target) <= 0:
        violations.append(f"[{symbol}] target ({float(target):.4f}) is not positive")

    # 6. Risk is not degenerate
    risk = abs(entry - stop)
    if risk < 0.01:
        violations.append(
            f"[{symbol}] degenerate risk: entry ({entry:.4f}) ≈ stop ({stop:.4f}), "
            f"risk={risk:.4f} < $0.01"
        )

    # 7. Chase distance
    if chase_r is not None:
        cr = float(chase_r)
        if cr > 0.75:
            violations.append(
                f"[{symbol}] CHASED: price is {cr:.2f}R past entry — setup is extended/plateaued"
            )
        elif cr > 0.40:
            warnings.append(f"[{symbol}] approaching chase limit: {cr:.2f}R past entry")

    # 8. Risk %
    if risk_pct is not None:
        rp = float(risk_pct)
        if rp > 12.0:
            violations.append(f"[{symbol}] risk_pct ({rp:.1f}%) > 12% — too wide")
        elif rp > 8.0:
            warnings.append(f"[{symbol}] risk_pct ({rp:.1f}%) is elevated")

    return IntegrityResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ── DB audit ───────────────────────────────────────────────────────────────────

def audit_plan_snapshots(store: Any, session_date: str | None = None) -> dict[str, Any]:
    """
    Scan plan_snapshots for integrity violations.
    Returns a summary dict with per-violation counts and examples.
    """
    try:
        rows = store.plan_snapshots_for_training(limit=5000, labeled_only=False)
    except Exception as e:
        return {"error": str(e), "total": 0}

    if session_date:
        rows = [r for r in rows if r.get("session_date") == session_date]

    total = len(rows)
    corrupt = 0
    violation_counts: dict[str, int] = {}
    examples: list[dict] = []

    for r in rows:
        result = validate_plan(
            side=r.get("plan_side"),
            entry=r.get("plan_entry"),
            stop=r.get("plan_stop"),
            target=r.get("plan_target"),
            current_price=r.get("current_price"),
            risk_pct=r.get("plan_risk_pct"),
            symbol=str(r.get("symbol") or "?"),
        )
        if not result.valid:
            corrupt += 1
            for v in result.violations:
                key = v.split(":")[0].split("]")[-1].strip()
                violation_counts[key] = violation_counts.get(key, 0) + 1
            if len(examples) < 20:
                examples.append({
                    "symbol": r.get("symbol"),
                    "session_date": r.get("session_date"),
                    "side": r.get("plan_side"),
                    "entry": r.get("plan_entry"),
                    "stop": r.get("plan_stop"),
                    "target": r.get("plan_target"),
                    "current_price": r.get("current_price"),
                    "violations": result.violations,
                })

    return {
        "session_date_filter": session_date,
        "total_snapshots": total,
        "corrupt_snapshots": corrupt,
        "corruption_rate_pct": round(corrupt / max(total, 1) * 100, 1),
        "violation_counts": violation_counts,
        "examples": examples,
    }


# ── Startup check ──────────────────────────────────────────────────────────────

def run_startup_integrity_check(store: Any) -> None:
    """
    Run on app startup. Logs warnings for today's corrupt plans.
    Does NOT raise — never blocks the app from starting.
    """
    import logging
    from datetime import date

    log = logging.getLogger("plan_integrity")
    today = date.today().isoformat()

    try:
        result = audit_plan_snapshots(store, session_date=today)
        total = result.get("total_snapshots", 0)
        corrupt = result.get("corrupt_snapshots", 0)

        if total == 0:
            log.info("plan_integrity: no snapshots for today yet")
            return

        if corrupt == 0:
            log.info(f"plan_integrity: ✓ all {total} snapshots for {today} are valid")
            return

        log.warning(
            f"plan_integrity: ⚠ {corrupt}/{total} snapshots for {today} are CORRUPT "
            f"({result['corruption_rate_pct']}%). Violations: {result['violation_counts']}"
        )
        for ex in result.get("examples", [])[:5]:
            log.warning(f"  {ex['symbol']} side={ex['side']} entry={ex['entry']} "
                        f"stop={ex['stop']} → {ex['violations']}")
    except Exception as e:
        log.error(f"plan_integrity startup check failed: {e}")
