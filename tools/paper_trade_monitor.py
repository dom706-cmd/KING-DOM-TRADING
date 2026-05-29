#!/usr/bin/env python3
"""
Kingdom Paper Trade Monitor
============================
Runs alongside the live Kingdom app and produces a timestamped paper trade log.

Usage:
    .venv/bin/python tools/paper_trade_monitor.py

Options (env vars):
    KINGDOM_PORT      Kingdom app port (default: 8050)
    PT_POLL_SEC       Poll interval in seconds (default: 10)
    PT_LOG_DIR        Where to write log files (default: /tmp)

What it does:
  - Polls /api/monitor_status every PT_POLL_SEC seconds
  - Auto-discovers the active monitor_id from /api/debug_last_scan or the runtime DB
  - Logs every symbol state snapshot to a JSONL file
  - Records a paper trade entry the first time a symbol reaches arming/confirmed/triggered
  - Records the exit when a symbol reaches failed/extended or at session close
  - Writes a readable summary .md file at exit

Output files (in PT_LOG_DIR):
  kingdom_pt_YYYYMMDD.jsonl       — one JSON line per poll event
  kingdom_pt_YYYYMMDD_summary.md  — human-readable postmortem written at exit

No fake data. All values come directly from the running Kingdom app.
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

PORT = int(os.getenv("KINGDOM_PORT", "8050"))
POLL_SEC = max(10, int(os.getenv("PT_POLL_SEC", "10")))
LOG_DIR = Path(os.getenv("PT_LOG_DIR", "/tmp"))
BASE_URL = f"http://127.0.0.1:{PORT}"

DB_PATH = Path(__file__).resolve().parents[1] / "runtime" / "runtime_state.db"

# States that indicate a trade is actionable
ACTIONABLE_STATES = {"arming", "confirmed", "triggered"}
# States that close a paper trade
CLOSING_STATES = {"failed", "extended"}

# ── Watchlist-focus mode ────────────────────────────────────────────────────────
# When enabled: only open paper trades for symbols on the desk watchlist.
# Watchlist symbols are printed at every poll even if they're still in "watch".
PT_WATCHLIST_ONLY = os.getenv("PT_WATCHLIST_ONLY", "0").strip().lower() in {"1", "true", "yes"}

# ── Entry gate config (all overridable via env vars) ───────────────────────────
# Gate 1: minimum ML score to open a trade.
# The monitor's state machine (tape_live, triggered, confirmed) is the primary
# quality filter. ML scores on watchlist symbols are often low (0.05–0.20) because
# the model was trained on scanner-universe stats, not hand-picked setups.
# Keep threshold low so the state machine — not ML — decides trade quality.
PT_MIN_ML_SCORE   = float(os.getenv("PT_MIN_ML_SCORE",  "0.05"))
# Gate 2: maximum spread % at entry (separate thresholds by price tier).
# Real momentum stocks trade with 0.2–0.8% spreads in the first hour — especially
# sub-$10 names. 0.20/0.15 are institutional-large-cap thresholds, not small/mid cap.
PT_MAX_SPREAD_SUB10  = float(os.getenv("PT_MAX_SPREAD_SUB10",  "0.75"))  # <$10 stocks
PT_MAX_SPREAD_ABOVE10 = float(os.getenv("PT_MAX_SPREAD_ABOVE10", "0.40")) # $10+ stocks
# Gate 3: risk per share — must be between min and max
PT_MIN_RISK_PER_SHARE = float(os.getenv("PT_MIN_RISK_PER_SHARE", "0.03"))
PT_MAX_RISK_PER_SHARE = float(os.getenv("PT_MAX_RISK_PER_SHARE", "2.00"))
# Gate 4: minimum risk_on score from market context to allow long trades
PT_REGIME_MIN_RISK_ON = float(os.getenv("PT_REGIME_MIN_RISK_ON", "0.20"))
# ──────────────────────────────────────────────────────────────────────────────


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def _get(path: str, timeout: int = 30) -> dict[str, Any] | None:
    url = BASE_URL + path
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as e:
        print(f"[pt-monitor] _get error ({type(e).__name__}): {e}")
        return None


# ─── Monitor ID discovery ──────────────────────────────────────────────────────

def _monitor_id_from_db() -> str | None:
    """Read the most recent running session started today from the runtime SQLite DB."""
    if not DB_PATH.exists():
        return None
    try:
        con = sqlite3.connect(str(DB_PATH), timeout=3)
        con.row_factory = sqlite3.Row
        # Only accept sessions started today (unix ts > today midnight ET)
        today_start = datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        cur = con.execute(
            "SELECT monitor_id FROM watch_sessions WHERE running=1 AND started_at >= ? "
            "ORDER BY started_at DESC LIMIT 1",
            (today_start,)
        )
        row = cur.fetchone()
        con.close()
        return row["monitor_id"] if row else None
    except Exception:
        return None


def _monitor_id_from_scan_api() -> str | None:
    """Pull the monitor_id from the last completed scan via /api/debug_last_scan."""
    data = _get("/api/debug_last_scan")
    if not data:
        return None
    # The result key contains monitor_id at top level or nested
    result = data.get("result") or data
    mid = result.get("monitor_id")
    if mid:
        return str(mid).strip() or None
    # Also check jobs list
    jobs = data.get("jobs") or []
    for job in jobs:
        r = (job.get("result") or {})
        mid = r.get("monitor_id")
        if mid:
            return str(mid).strip() or None
    return None


def _validate_monitor_id(mid: str) -> bool:
    """Return True only if the live app responds OK for this monitor_id."""
    data = _get(f"/api/monitor_status?monitor_id={mid}&no_refresh=1")
    return bool(data and data.get("ok"))


def discover_monitor_id(max_wait_s: int = 300) -> str:
    """Block until a live, reachable monitor session is found, or raise after max_wait_s."""
    deadline = time.time() + max_wait_s
    attempt = 0
    while time.time() < deadline:
        # API first (reflects today's session immediately after scan completes)
        # DB second (also filtered to today-only sessions)
        for mid in filter(None, [_monitor_id_from_scan_api(), _monitor_id_from_db()]):
            if _validate_monitor_id(mid):
                return mid
        attempt += 1
        if attempt == 1:
            print(f"[pt-monitor] Waiting for an active Kingdom monitor session "
                  f"(up to {max_wait_s}s)…  Launch Kingdom and run a scan.")
        time.sleep(10)
    raise RuntimeError(
        "No active monitor session found after waiting. "
        "Make sure Kingdom is running and a scan has completed."
    )


# ─── Paper trade book ─────────────────────────────────────────────────────────

class PaperTradeBook:
    def __init__(self) -> None:
        self.open_trades: dict[str, dict[str, Any]] = {}   # symbol → trade
        self.closed_trades: list[dict[str, Any]] = []

    def open(self, sym: str, sym_state: dict[str, Any], now_et: datetime) -> dict[str, Any] | None:
        """Record a paper trade entry. Returns the trade dict if new, None if already open."""
        if sym in self.open_trades:
            return None
        entry = sym_state.get("entry")
        stop  = sym_state.get("stop_loss")
        t2r   = sym_state.get("target_2r")
        t3r   = sym_state.get("target_3r")
        price = sym_state.get("price")
        side  = sym_state.get("best_side") or "long"

        if entry is None or stop is None:
            return None  # no plan — don't fake it

        risk = round(abs(entry - stop), 4) if entry is not None and stop is not None else None

        trade = {
            "symbol": sym,
            "side": side,
            "playbook": sym_state.get("playbook"),
            "trigger_state": sym_state.get("monitor_state"),
            "entry_ref": round(float(entry), 4),
            "stop_ref": round(float(stop), 4),
            "target_2r": round(float(t2r), 4) if t2r is not None else None,
            "target_3r": round(float(t3r), 4) if t3r is not None else None,
            "risk_per_share": round(float(risk), 4) if risk is not None else None,
            "price_at_trigger": round(float(price), 4) if price is not None else None,
            "opened_at_et": now_et.isoformat(),
            "opened_at_ts": time.time(),
            "p_2r_30m": sym_state.get("p_2r_30m"),
            "probability_source": sym_state.get("probability_source"),
            "ml_score": sym_state.get("ml_score"),
            "catalyst_score": sym_state.get("catalyst_score"),
            "catalyst_tags": sym_state.get("catalyst_tags") or [],
            "spread_pct_at_trigger": sym_state.get("spread_pct"),
            "tape_live_at_trigger": sym_state.get("tape_live"),
            "decision_at_trigger": sym_state.get("decision"),
            "exit_price": None,
            "exit_state": None,
            "exit_at_et": None,
            "result_r": None,
            "result_label": None,
        }
        self.open_trades[sym] = trade
        return trade

    def close(self, sym: str, sym_state: dict[str, Any], now_et: datetime, reason: str) -> dict[str, Any] | None:
        """Close a paper trade. Returns the closed trade dict."""
        trade = self.open_trades.pop(sym, None)
        if trade is None:
            return None
        price = sym_state.get("price")
        trade["exit_price"] = round(float(price), 4) if price is not None else None
        trade["exit_state"] = reason
        trade["exit_at_et"] = now_et.isoformat()
        if trade["exit_price"] is not None and trade["risk_per_share"]:
            rps = trade["risk_per_share"]
            if trade["side"] == "long":
                result_r = (trade["exit_price"] - trade["entry_ref"]) / rps
            else:
                result_r = (trade["entry_ref"] - trade["exit_price"]) / rps
            trade["result_r"] = round(result_r, 3)
            if result_r >= 1.8:
                trade["result_label"] = "win_2R+"
            elif result_r >= 0.9:
                trade["result_label"] = "win_partial"
            elif result_r >= -0.15:
                trade["result_label"] = "scratch"
            else:
                trade["result_label"] = "loss"
        self.closed_trades.append(trade)
        return trade

    def close_all_open(self, sym_states: dict[str, dict[str, Any]], now_et: datetime) -> None:
        for sym in list(self.open_trades.keys()):
            st = sym_states.get(sym) or {}
            self.close(sym, st, now_et, "session_close")


# ─── Gap Order Tracker ────────────────────────────────────────────────────────

class GapOrderTracker:
    """Tracks pre-market gap order plans and fires paper trades when price crosses entry.

    Execution rule (mirrors KingDom gap-plan methodology):
      - Wait until the first 1-min candle closes (9:31 AM ET or later).
      - LONG: trigger when price >= entry (PM high + $0.01).
      - SHORT: trigger when price <= entry (PM low - $0.01).
      - Skip if price opened MORE than 2% through the entry (already chasing).
      - Use PM low as stop for longs, PM high as stop for shorts (as planned).
    """

    CHASE_SKIP_PCT = 0.02  # skip if already 2% past entry at time of check

    def __init__(self) -> None:
        self.plans: dict[str, dict[str, Any]] = {}   # symbol → plan dict
        self.triggered: set[str] = set()             # symbols already paper-traded
        self.loaded = False
        self.load_attempted_at: float = 0.0

    def try_load(self) -> None:
        """Fetch gap order plans from Kingdom. Only attempts once per session."""
        if self.loaded:
            return
        now_et = datetime.now(ET)
        # Only try during pre-market or first 15 min of session
        if now_et.hour > 9 or (now_et.hour == 9 and now_et.minute > 45):
            return
        if time.time() - self.load_attempted_at < 60:
            return
        self.load_attempted_at = time.time()
        data = _get("/api/premarket_gap_orders?risk_dollars=100")
        if not data or not data.get("ok"):
            return
        plans = data.get("plans") or []
        for p in plans:
            sym = str(p.get("symbol") or "").strip().upper()
            if not sym:
                continue
            self.plans[sym] = p
        if self.plans:
            print(f"[pt-gap] Loaded {len(self.plans)} gap order plans: {', '.join(sorted(self.plans.keys()))}")
        self.loaded = True

    def check_triggers(self, sym_map: dict[str, dict[str, Any]], now_et: datetime,
                       book: "PaperTradeBook", log_fn: Any,
                       entry_gate_fn: Any = None, context: dict[str, Any] | None = None) -> None:
        """Check if any gap plan has been triggered by current monitor prices."""
        if not self.plans:
            return
        # Only check after first 1-min candle closes (9:31 AM or later)
        if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 31):
            return

        for sym, plan in self.plans.items():
            if sym in self.triggered or sym in book.open_trades:
                continue
            st = sym_map.get(sym)
            if not st:
                continue  # symbol not in monitor — can't get live price
            price = st.get("price")
            if not price:
                continue
            price = float(price)
            entry   = float(plan.get("entry") or 0)
            stop    = float(plan.get("stop") or 0)
            side    = str(plan.get("side") or "long").lower()
            target  = float(plan.get("target_2r") or 0)
            r_share = abs(entry - stop) if entry and stop else None

            if not entry or not stop or not r_share:
                continue

            triggered = False
            if side == "long" and price >= entry:
                chase_pct = (price - entry) / entry
                if chase_pct <= self.CHASE_SKIP_PCT:
                    triggered = True
                else:
                    print(f"  [pt-gap] SKIP {sym} — already {chase_pct*100:.1f}% past long entry (chasing)")
            elif side == "short" and price <= entry:
                chase_pct = (entry - price) / entry
                if chase_pct <= self.CHASE_SKIP_PCT:
                    triggered = True
                else:
                    print(f"  [pt-gap] SKIP {sym} — already {chase_pct*100:.1f}% past short entry (chasing)")

            if triggered:
                trade_state = {
                    **st,
                    "entry": entry,
                    "stop_loss": stop,
                    "target_2r": target,
                    "best_side": side,
                    "monitor_state": "gap_triggered",
                    "playbook": "gap_order",
                }

                # Apply the same entry gates as state-machine trades
                if entry_gate_fn is not None:
                    gate_fail = entry_gate_fn(sym, trade_state, now_et, context or {})
                    if gate_fail:
                        print(f"  🚫 GAP GATE BLOCKED: {sym}  reason={gate_fail}")
                        log_fn("entry_gate_blocked", {
                            "symbol": sym, "reason": gate_fail, "source": "gap_order",
                            "state": "gap_triggered", "price": price,
                            "ml_score": st.get("ml_score"), "spread_pct": st.get("spread_pct"),
                        })
                        continue

                self.triggered.add(sym)
                trade = book.open(sym, trade_state, now_et)
                if trade:
                    print(f"  📋 GAP TRADE OPENED: {sym}  side={side}  "
                          f"entry={entry}  stop={stop}  2R={target}  price@trigger={price}")
                    log_fn("gap_trade_open", {**trade, "gap_entry_plan": plan})


# ─── Main loop ────────────────────────────────────────────────────────────────

def _next_session_paths(log_dir: Path, today: str) -> tuple[Path, Path]:
    """Return (log_path, summary_path) that don't already exist for today.

    First session:  kingdom_pt_20260402.jsonl / _summary.md
    Second session: kingdom_pt_20260402_2.jsonl / _2_summary.md
    Third session:  kingdom_pt_20260402_3.jsonl / _3_summary.md  … etc.
    """
    base = log_dir / f"kingdom_pt_{today}.jsonl"
    if not base.exists():
        return base, log_dir / f"kingdom_pt_{today}_summary.md"
    n = 2
    while True:
        candidate = log_dir / f"kingdom_pt_{today}_{n}.jsonl"
        if not candidate.exists():
            return candidate, log_dir / f"kingdom_pt_{today}_{n}_summary.md"
        n += 1


class PaperTradeMonitor:
    def __init__(self) -> None:
        today = datetime.now(ET).strftime("%Y%m%d")
        self.log_path, self.summary_path = _next_session_paths(LOG_DIR, today)
        self.book = PaperTradeBook()
        self.gap_tracker = GapOrderTracker()
        self.prev_states: dict[str, str] = {}  # symbol → last known monitor_state
        self.poll_count = 0
        self.monitor_id: str | None = None
        self._running = True
        self._session_diag_baseline: dict[str, int] = {}
        self._watchlist_syms: set[str] = set()   # populated in run()
        self._watchlist_meta: dict[str, dict] = {}  # symbol → watchlist entry
        print(f"[pt-monitor] Log → {self.log_path}")
        print(f"[pt-monitor] Summary → {self.summary_path}")
        print(f"[pt-monitor] Entry gates:")
        print(f"             Min ML score  : {PT_MIN_ML_SCORE}  (PT_MIN_ML_SCORE)")
        print(f"             Max spread    : <$10 → {PT_MAX_SPREAD_SUB10}%  ≥$10 → {PT_MAX_SPREAD_ABOVE10}%  (PT_MAX_SPREAD_*)")
        print(f"             Risk/share    : ${PT_MIN_RISK_PER_SHARE} – ${PT_MAX_RISK_PER_SHARE}  (PT_MIN/MAX_RISK_PER_SHARE)")
        print(f"             Regime gate   : risk_on ≥ {PT_REGIME_MIN_RISK_ON} for long trades  (PT_REGIME_MIN_RISK_ON)")

    def _entry_gate(self, sym: str, st: dict[str, Any], now_et: datetime,
                    context: dict[str, Any]) -> str | None:
        """Return a reason string if trade should be blocked, None if it passes all gates."""

        # Gate 1: ML score
        ml = st.get("ml_score")
        if ml is not None:
            try:
                if float(ml) < PT_MIN_ML_SCORE:
                    return f"ml_too_low ({float(ml):.3f} < {PT_MIN_ML_SCORE})"
            except Exception:
                pass

        # Gate 2: spread
        spread = st.get("spread_pct")
        price  = st.get("price")
        if spread is not None and price is not None:
            try:
                spread_f = float(spread)
                price_f  = float(price)
                if spread_f <= 0.0:
                    # 0.0% spread is impossible on a real stock — missing bid/ask data.
                    # Fail-safe: block the trade rather than silently pass a bad quote.
                    return f"spread_missing (spread={spread_f}% — no quote data at trigger)"
                max_spread = PT_MAX_SPREAD_SUB10 if price_f < 10.0 else PT_MAX_SPREAD_ABOVE10
                if spread_f > max_spread:
                    return f"spread_too_wide ({spread_f:.3f}% > {max_spread}%)"
            except Exception:
                pass

        # Gate 3: risk per share — must be within min/max band
        entry = st.get("entry")
        stop  = st.get("stop_loss")
        if entry is not None and stop is not None:
            try:
                risk = abs(float(entry) - float(stop))
                if risk < PT_MIN_RISK_PER_SHARE:
                    return f"risk_too_small (${risk:.4f} < ${PT_MIN_RISK_PER_SHARE})"
                if risk > PT_MAX_RISK_PER_SHARE:
                    return f"risk_too_large (${risk:.4f} > ${PT_MAX_RISK_PER_SHARE})"
            except Exception:
                pass

        # Gate 4: market regime — block long trades in risk-off market
        # API returns "risk_on_score" from the context engine snapshot.
        side = str(st.get("best_side") or "long").lower()
        if side == "long":
            risk_on = context.get("risk_on_score") or context.get("risk_on")
            if risk_on is not None:
                try:
                    if float(risk_on) < PT_REGIME_MIN_RISK_ON:
                        return f"regime_risk_off (risk_on={float(risk_on):.3f} < {PT_REGIME_MIN_RISK_ON})"
                except Exception:
                    pass

        # Gate 5: monitor decision must be GO (not PASS or WAIT)
        decision = str(st.get("decision") or st.get("plan_state") or "").upper()
        if decision and decision != "GO":
            return f"decision_not_go ({decision})"

        return None  # all gates passed

    def _log(self, event_type: str, payload: dict[str, Any]) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ts_et": datetime.now(ET).isoformat(),
            "event": event_type,
            **payload,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _load_watchlist(self) -> None:
        """Fetch the current desk watchlist from Kingdom and cache it."""
        data = _get("/api/desk_watchlist")
        if data and data.get("ok"):
            items = data.get("items") or []
            self._watchlist_syms = {str(i["symbol"]).upper() for i in items if i.get("symbol")}
            self._watchlist_meta = {str(i["symbol"]).upper(): i for i in items if i.get("symbol")}
        else:
            self._watchlist_syms = set()
            self._watchlist_meta = {}

    def _check_health(self) -> bool:
        h = _get("/api/health")
        if not h or not h.get("ok"):
            return False
        alpaca_ok = (h.get("alpaca") or {}).get("ok")
        stream_ok = (h.get("stream") or {}).get("ok")
        return bool(alpaca_ok or stream_ok)  # at least one data path live

    def _poll_once(self) -> None:
        if not self.monitor_id:
            return
        data = _get(f"/api/monitor_status?monitor_id={self.monitor_id}&no_refresh=1")
        if not data or not data.get("ok"):
            error = (data or {}).get("error") or "no_response"
            print(f"[pt-monitor] Poll failed: {error}")
            self._log("poll_error", {"monitor_id": self.monitor_id, "error": error})
            # Monitor session was replaced (new scan or manual restart) — re-discover.
            if error == "unknown_monitor_id":
                print("[pt-monitor] Monitor session gone — waiting for a new active session…")
                try:
                    self.monitor_id = discover_monitor_id(max_wait_s=300)
                    print(f"[pt-monitor] Re-attached to monitor session: {self.monitor_id}")
                    self._log("monitor_reattached", {"monitor_id": self.monitor_id})
                except RuntimeError as e:
                    print(f"[pt-monitor] Could not re-attach: {e}")
            return

        now_et = datetime.now(ET)
        self.poll_count += 1
        summary = data.get("summary") or {}
        diag = summary.get("diagnostics") or {}
        context = data.get("context") or {}
        sym_list: list[dict[str, Any]] = data.get("symbols") or []
        sym_map = {s["symbol"]: s for s in sym_list if s.get("symbol")}

        # Compute diag deltas since first poll
        if not self._session_diag_baseline:
            self._session_diag_baseline = dict(diag)
        diag_delta = {k: int(diag.get(k, 0)) - int(self._session_diag_baseline.get(k, 0))
                      for k in diag}

        # Log snapshot
        self._log("snapshot", {
            "monitor_id": self.monitor_id,
            "poll_count": self.poll_count,
            "running": data.get("running"),
            "refresh_count": data.get("refresh_count"),
            "symbol_count": len(sym_list),
            "state_counts": summary.get("state_counts") or {},
            "tape_counts": summary.get("tape") or {},
            "ready_count": summary.get("ready_count", 0),
            "triggered_count": summary.get("live_triggered_count", 0),
            "diagnostics": diag,
            "diagnostics_delta": diag_delta,
            "context": {
                "spy_trend": context.get("spy_trend_state"),
                "qqq_trend": context.get("qqq_trend_state"),
                "risk_on": context.get("risk_on_score"),
                "breadth": context.get("breadth_score"),
            },
        })

        # Print live status
        state_counts = summary.get("state_counts") or {}
        tape_counts = summary.get("tape") or {}
        triggered = state_counts.get("triggered", 0)
        arming = state_counts.get("arming", 0)
        confirmed = state_counts.get("confirmed", 0)
        tape_live = tape_counts.get("live", 0)
        rej_spread = diag.get("rejected_spread", 0)
        rej_stale = diag.get("rejected_stale_timing", 0)
        wl_tag = f" | watchlist={len(self._watchlist_syms)}" if PT_WATCHLIST_ONLY else ""
        print(
            f"[{now_et.strftime('%H:%M:%S')} ET] "
            f"poll#{self.poll_count} | "
            f"triggered={triggered} arming={arming} confirmed={confirmed} | "
            f"tape_live={tape_live}/{len(sym_list)} | "
            f"rej_spread={rej_spread} rej_stale={rej_stale}{wl_tag}"
        )

        # In watchlist-focus mode: print current state of every watchlist symbol each poll
        if PT_WATCHLIST_ONLY and self._watchlist_syms:
            for wsym in sorted(self._watchlist_syms):
                wst = sym_map.get(wsym)
                if wst:
                    wstate = str(wst.get("monitor_state") or "watch")
                    wtape = "live" if wst.get("tape_live") else "STALE"
                    wprice = wst.get("price") or "?"
                    wdec = wst.get("decision") or "?"
                    print(f"  👁  {wsym:<6} [{wstate:<10}] price={wprice}  tape={wtape}  decision={wdec}")
                else:
                    print(f"  👁  {wsym:<6} [not in monitor]")

        # Try to load gap order plans (no-op if already loaded or too late in day)
        self.gap_tracker.try_load()
        # Check if any gap orders have been triggered by current prices
        self.gap_tracker.check_triggers(sym_map, now_et, self.book, self._log,
                                        entry_gate_fn=self._entry_gate, context=context)

        # Process state transitions for each symbol
        for sym, st in sym_map.items():
            new_state = str(st.get("monitor_state") or "watch")
            old_state = self.prev_states.get(sym, "watch")

            if new_state != old_state:
                print(f"  ▶ {sym}: {old_state} → {new_state}  "
                      f"[tape={'live' if st.get('tape_live') else 'STALE'}  "
                      f"spread={st.get('spread_pct') or '?'}%  "
                      f"price={st.get('price')}]")
                self._log("state_transition", {
                    "symbol": sym,
                    "from_state": old_state,
                    "to_state": new_state,
                    "price": st.get("price"),
                    "entry": st.get("entry"),
                    "stop_loss": st.get("stop_loss"),
                    "target_2r": st.get("target_2r"),
                    "spread_pct": st.get("spread_pct"),
                    "tape_live": st.get("tape_live"),
                    "tape_live_reason": st.get("tape_live_reason"),
                    "decision": st.get("decision"),
                    "rejection_reasons": st.get("rejection_reasons"),
                })

                # Open paper trade on first actionable state — subject to entry gates
                if new_state in ACTIONABLE_STATES and old_state not in ACTIONABLE_STATES:
                    # Watchlist-focus gate: skip symbols not on the desk watchlist
                    if PT_WATCHLIST_ONLY and sym not in self._watchlist_syms:
                        self._log("entry_gate_blocked", {"symbol": sym, "reason": "not_on_watchlist",
                                                          "state": new_state, "price": st.get("price")})
                        continue

                    gate_fail = self._entry_gate(sym, st, now_et, context)
                    if gate_fail:
                        print(f"  🚫 GATE BLOCKED: {sym}  reason={gate_fail}")
                        self._log("entry_gate_blocked", {"symbol": sym, "reason": gate_fail,
                                                          "state": new_state, "price": st.get("price"),
                                                          "ml_score": st.get("ml_score"),
                                                          "spread_pct": st.get("spread_pct")})
                    else:
                        trade = self.book.open(sym, st, now_et)
                        if trade:
                            print(f"  📋 PAPER TRADE OPENED: {sym}  "
                                  f"side={trade['side']}  "
                                  f"entry={trade['entry_ref']}  "
                                  f"stop={trade['stop_ref']}  "
                                  f"2R={trade['target_2r']}")
                            self._log("paper_trade_open", trade)

                # Close paper trade on exit state
                if new_state in CLOSING_STATES and sym in self.book.open_trades:
                    closed = self.book.close(sym, st, now_et, new_state)
                    if closed:
                        label = closed.get("result_label") or "unknown"
                        r = closed.get("result_r")
                        print(f"  🔚 PAPER TRADE CLOSED: {sym}  "
                              f"exit={closed['exit_price']}  "
                              f"result={r}R  [{label}]")
                        self._log("paper_trade_close", closed)

            self.prev_states[sym] = new_state

    def _write_summary(self) -> None:
        now_et = datetime.now(ET)

        # Close any still-open trades
        sym_states: dict[str, dict[str, Any]] = {}
        if self.monitor_id:
            data = _get(f"/api/monitor_status?monitor_id={self.monitor_id}&no_refresh=1")
            if data and data.get("symbols"):
                sym_states = {s["symbol"]: s for s in data["symbols"] if s.get("symbol")}
        self.book.close_all_open(sym_states, now_et)

        closed = self.book.closed_trades
        wins  = [t for t in closed if (t.get("result_r") or 0) >= 0.9]
        losses = [t for t in closed if (t.get("result_r") or 0) < -0.15]
        scratches = [t for t in closed if t not in wins and t not in losses]

        total_r = round(sum(t.get("result_r") or 0.0 for t in closed), 2)

        # Count gate blocks from log
        gate_blocks: dict[str, int] = {}
        if self.log_path.exists():
            with self.log_path.open() as _f:
                for _line in _f:
                    try:
                        _r = json.loads(_line)
                        if _r.get("event") == "entry_gate_blocked":
                            reason = str(_r.get("reason") or "unknown").split("(")[0].strip()
                            gate_blocks[reason] = gate_blocks.get(reason, 0) + 1
                    except Exception:
                        continue

        wl_line = (f"- Watchlist-focus: ON — {len(self._watchlist_syms)} symbols: "
                   f"{', '.join(sorted(self._watchlist_syms))}"
                   if PT_WATCHLIST_ONLY else "- Watchlist-focus: OFF (all symbols eligible)")

        lines = [
            f"# Kingdom Paper Trade Session — {now_et.strftime('%Y-%m-%d')}",
            f"",
            f"## Session summary",
            f"- Generated: {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}",
            f"- Monitor ID: `{self.monitor_id or 'unknown'}`",
            f"- Total polls: {self.poll_count}",
            wl_line,
            f"- Paper trades taken: {len(closed)}",
            f"- Wins (≥0.9R): {len(wins)}",
            f"- Losses (< -0.15R): {len(losses)}",
            f"- Scratches: {len(scratches)}",
            f"- Total R: {total_r}R",
            f"",
            f"## Entry gates (active this session)",
            f"- Min ML score: {PT_MIN_ML_SCORE}",
            f"- Max spread: <$10 → {PT_MAX_SPREAD_SUB10}%  ≥$10 → {PT_MAX_SPREAD_ABOVE10}%",
            f"- Risk/share band: ${PT_MIN_RISK_PER_SHARE} – ${PT_MAX_RISK_PER_SHARE}",
            f"- Regime gate: risk_on ≥ {PT_REGIME_MIN_RISK_ON} for long trades",
            f"",
            f"## Gate blocks this session",
        ] + ([f"- {reason}: {count}" for reason, count in sorted(gate_blocks.items(), key=lambda x: -x[1])]
             if gate_blocks else ["- None"]) + [
            f"",
            f"## Diagnostic highlights",
        ]

        # Read final diagnostics from log
        if self.log_path.exists():
            last_snap: dict[str, Any] = {}
            with self.log_path.open() as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if row.get("event") == "snapshot":
                            last_snap = row
                    except Exception:
                        continue
            diag = last_snap.get("diagnostics") or {}
            ctx = last_snap.get("context") or {}
            state_counts = last_snap.get("state_counts") or {}
            tape_counts = last_snap.get("tape_counts") or {}
            lines += [
                f"- rejected_spread: {diag.get('rejected_spread', 0)}",
                f"- rejected_stale_timing: {diag.get('rejected_stale_timing', 0)}",
                f"- rejected_no_live_confirmation: {diag.get('rejected_no_live_confirmation', 0)}",
                f"- promoted_by_monitor_transition: {diag.get('promoted_by_monitor_transition', 0)}",
                f"- promoted_by_news: {diag.get('promoted_by_news', 0)}",
                f"- tape_live (last poll): {tape_counts.get('live', '?')}",
                f"- state_counts (last poll): {json.dumps(state_counts)}",
                f"- context: spy={ctx.get('spy_trend')} qqq={ctx.get('qqq_trend')} "
                  f"risk_on={ctx.get('risk_on')} breadth={ctx.get('breadth')}",
            ]

        lines += ["", "## Paper trade blotter"]
        if not closed:
            lines.append("No paper trades triggered this session.")
        else:
            for i, t in enumerate(closed, 1):
                r_str = f"{t['result_r']}R" if t.get("result_r") is not None else "?"
                lines += [
                    f"",
                    f"### PT-{i:02d} — {t['symbol']} — {(t.get('side') or 'long').upper()} "
                    f"({t.get('playbook') or '?'})",
                    f"- Opened: {t.get('opened_at_et', '?')[:19]}",
                    f"- Closed: {t.get('exit_at_et', '?')[:19]} via `{t.get('exit_state')}`",
                    f"- Entry ref: {t.get('entry_ref')}  Stop: {t.get('stop_ref')}  "
                    f"Target 2R: {t.get('target_2r')}",
                    f"- Price at trigger: {t.get('price_at_trigger')}",
                    f"- Exit price: {t.get('exit_price')}",
                    f"- Result: **{r_str}** [{t.get('result_label', '?')}]",
                    f"- ML score: {t.get('ml_score')}  "
                    f"p_2r_30m: {t.get('p_2r_30m')} ({t.get('probability_source')})",
                    f"- Catalyst: {t.get('catalyst_score')}  tags: {t.get('catalyst_tags')}",
                    f"- Spread at trigger: {t.get('spread_pct_at_trigger')}%  "
                    f"tape_live: {t.get('tape_live_at_trigger')}",
                    f"- Decision: `{t.get('decision_at_trigger')}`",
                ]

        lines += [
            "",
            "## Validation questions",
            "- Did symbols that reached arming/triggered produce real moves?",
            "- Were there false arming states (near-entry but no follow-through)?",
            "- What was the primary tape rejection reason today (spread vs stale timing)?",
            "- Did the spread fix (≥$5: 1.5%, <$5: 2.0%) allow more names to advance?",
        ]
        lines += ["", f"## Log file", f"- `{self.log_path}`"]

        self.summary_path.write_text("\n".join(lines) + "\n")
        print(f"\n[pt-monitor] Summary written → {self.summary_path}")

    def run(self) -> None:
        # Health check
        print(f"[pt-monitor] Checking Kingdom health at {BASE_URL}…")
        if not self._check_health():
            print("[pt-monitor] WARNING: Kingdom health check failed. "
                  "App may not be fully live yet. Continuing anyway.")

        # Discover monitor
        print("[pt-monitor] Discovering active monitor session…")
        try:
            self.monitor_id = discover_monitor_id(max_wait_s=300)
        except RuntimeError as e:
            print(f"[pt-monitor] ERROR: {e}")
            sys.exit(1)

        print(f"[pt-monitor] Monitoring session: {self.monitor_id}")

        # Load watchlist (always — used for 👁 display even outside watchlist-only mode)
        self._load_watchlist()
        if PT_WATCHLIST_ONLY:
            if self._watchlist_syms:
                print(f"[pt-monitor] Watchlist-focus mode ON — tracking {len(self._watchlist_syms)} symbols: "
                      f"{', '.join(sorted(self._watchlist_syms))}")
                print(f"[pt-monitor] Trades will ONLY open for watchlist symbols.")
            else:
                print(f"[pt-monitor] WARNING: Watchlist-focus mode ON but desk watchlist is empty.")
                print(f"[pt-monitor]   Run the Pre-Market Scan in Kingdom UI first, or add symbols manually.")
        else:
            if self._watchlist_syms:
                print(f"[pt-monitor] Watchlist ({len(self._watchlist_syms)} symbols) will be highlighted each poll.")

        print(f"[pt-monitor] Polling every {POLL_SEC}s. Press Ctrl+C to stop and write summary.\n")
        self._log("session_start", {"monitor_id": self.monitor_id, "poll_interval_s": POLL_SEC,
                                    "watchlist_only": PT_WATCHLIST_ONLY,
                                    "watchlist_symbols": sorted(self._watchlist_syms)})

        def _handle_stop(sig, frame):
            print("\n[pt-monitor] Stopping…")
            self._running = False

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        while self._running:
            try:
                self._poll_once()
            except Exception as e:
                print(f"[pt-monitor] Unexpected error: {type(e).__name__}: {e}")
                self._log("error", {"error": str(e), "type": type(e).__name__})
            if self._running:
                time.sleep(POLL_SEC)

        self._write_summary()
        print(f"[pt-monitor] Done. {len(self.book.closed_trades)} paper trades recorded.")


if __name__ == "__main__":
    import argparse as _argparse
    _ap = _argparse.ArgumentParser(description="Kingdom paper trade monitor")
    _ap.add_argument("--watchlist-only", action="store_true",
                     help="Only open paper trades for symbols on the desk watchlist "
                          "(also sets PT_WATCHLIST_ONLY=1)")
    _args = _ap.parse_args()
    if _args.watchlist_only:
        os.environ["PT_WATCHLIST_ONLY"] = "1"
        # Re-read the constant so the runtime value is correct
        import importlib as _il, sys as _sys_mod
        _mod = _sys_mod.modules[__name__]
        _mod.PT_WATCHLIST_ONLY = True  # type: ignore[attr-defined]
    PaperTradeMonitor().run()
