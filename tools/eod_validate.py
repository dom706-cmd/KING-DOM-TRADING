#!/usr/bin/env python3
"""
Kingdom End-of-Day Paper Trade Validator
=========================================
Reads today's paper trade JSONL log, fetches 1-minute intraday bars for each
traded symbol, and produces a validation report showing:

  - Was the entry price actually achievable?
  - Did price hit stop or 2R target first?
  - Max adverse excursion (MAE) and max favorable excursion (MFE)
  - State transition timeline for each symbol
  - Spread and tape quality at entry
  - ML score vs actual outcome
  - Session-level diagnostics and context

Usage:
    cd ~/kingdom
    .venv/bin/python tools/eod_validate.py
    .venv/bin/python tools/eod_validate.py --date 20260402
    .venv/bin/python tools/eod_validate.py --log /tmp/kingdom_pt_20260402.jsonl

Output:
    Prints to terminal.
    Writes ~/Desktop/kingdom_eod_YYYYMMDD.md

Env vars (same as app):
    ALPACA_API_KEY / APCA_API_KEY_ID
    ALPACA_SECRET_KEY / APCA_API_SECRET_KEY
    KINGDOM_PORT (default 8050)
    PT_LOG_DIR   (default /tmp)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
KINGDOM_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = Path(os.getenv("PT_LOG_DIR", "/tmp"))
DESKTOP = Path.home() / "Desktop"


# ── JSONL log reader ──────────────────────────────────────────────────────────

def load_log(log_path: Path) -> list[dict[str, Any]]:
    rows = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_trades(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return {symbol: merged_trade_dict} combining open+close events."""
    trades: dict[str, dict[str, Any]] = {}
    for row in rows:
        ev = row.get("event")
        sym = row.get("symbol")
        if not sym:
            continue
        if ev == "paper_trade_open":
            trades[sym] = dict(row)
            trades[sym]["transitions"] = []
        elif ev == "paper_trade_close" and sym in trades:
            trades[sym].update({
                "exit_price":   row.get("exit_price"),
                "exit_state":   row.get("exit_state"),
                "exit_at_et":   row.get("exit_at_et"),
                "result_r":     row.get("result_r"),
                "result_label": row.get("result_label"),
            })
    # Attach state transition timeline to each trade
    for row in rows:
        if row.get("event") == "state_transition":
            sym = row.get("symbol")
            if sym in trades:
                trades[sym]["transitions"].append(row)
    return trades


def extract_session_diag(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull first + last snapshot diagnostics and context."""
    snaps = [r for r in rows if r.get("event") == "snapshot"]
    if not snaps:
        return {}
    first, last = snaps[0], snaps[-1]
    return {
        "poll_count":    last.get("poll_count", 0),
        "symbol_count":  last.get("symbol_count", 0),
        "diag_first":    first.get("diagnostics") or {},
        "diag_last":     last.get("diagnostics") or {},
        "context_first": first.get("context") or {},
        "context_last":  last.get("context") or {},
        "state_counts":  last.get("state_counts") or {},
        "tape_counts":   last.get("tape_counts") or {},
    }


# ── Alpaca 1-min bars ─────────────────────────────────────────────────────────

def _alpaca_client():
    """Return an alpaca-py StockHistoricalDataClient using env keys."""
    try:
        from alpaca.data import StockHistoricalDataClient
    except ImportError:
        print("ERROR: alpaca-py not installed. Run: pip install alpaca-py")
        sys.exit(1)
    key = (os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
           or os.getenv("ALPACA_KEY_ID") or "")
    secret = (os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
              or os.getenv("ALPACA_API_SECRET") or "")
    if not key or not secret:
        print("ERROR: Alpaca API key/secret not set in environment.")
        print("  Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
        sys.exit(1)
    return StockHistoricalDataClient(key, secret)


def fetch_bars(client, symbol: str, session_date: date) -> list[dict]:
    """Fetch 1-min bars for symbol on session_date (full regular session)."""
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        return []

    # Regular session: 9:30am–4:00pm ET
    start = datetime(session_date.year, session_date.month, session_date.day,
                     9, 30, 0, tzinfo=ET)
    end   = datetime(session_date.year, session_date.month, session_date.day,
                     16, 0, 0, tzinfo=ET)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start,
            end=end,
            feed="sip",
        )
        resp = client.get_stock_bars(req)
        df = resp.df
        if df is None or df.empty:
            return []
        df = df.reset_index()
        bars = []
        for _, row in df.iterrows():
            ts = row.get("timestamp")
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            bars.append({
                "ts":    ts,
                "open":  float(row["open"]),
                "high":  float(row["high"]),
                "low":   float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
                "vwap":  float(row["vwap"]) if "vwap" in row else None,
            })
        return bars
    except Exception as e:
        print(f"  [bars] {symbol}: fetch error — {type(e).__name__}: {e}")
        return []


# ── Price-level validation logic ──────────────────────────────────────────────

def _parse_et(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str[:26], fmt[:len(ts_str[:26])])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ET)
            return dt.astimezone(ET)
        except ValueError:
            continue
    return None


def validate_trade_levels(trade: dict, bars: list[dict]) -> dict:
    """
    Using 1-min bars from entry time onwards, compute:
      - entry_achievable: was entry_ref traded at or through?
      - first_exit: did stop or 2R get hit first? and when?
      - mae: max adverse excursion from entry (in R)
      - mfe: max favorable excursion from entry (in R)
      - bars_to_stop / bars_to_2r
    """
    entry   = trade.get("entry_ref")
    stop    = trade.get("stop_ref")
    t2r     = trade.get("target_2r")
    side    = (trade.get("side") or "long").lower()
    opened  = _parse_et(trade.get("opened_at_et"))

    result: dict[str, Any] = {
        "entry_achievable": None,
        "first_exit":       "not_reached",
        "first_exit_time":  None,
        "bars_to_stop":     None,
        "bars_to_2r":       None,
        "mae_r":            None,
        "mfe_r":            None,
        "price_at_open_bar": None,
        "high_after_entry": None,
        "low_after_entry":  None,
        "bars_analyzed":    0,
        "note":             "",
    }

    if entry is None or stop is None or not bars:
        result["note"] = "missing entry/stop or no bars"
        return result

    risk = abs(entry - stop)
    if risk == 0:
        result["note"] = "zero risk — cannot compute R"
        return result

    # Filter bars to those at/after entry time
    post_bars = bars
    if opened:
        post_bars = [b for b in bars if b["ts"] >= opened]
    if not post_bars:
        result["note"] = "no bars after entry time"
        return result

    result["bars_analyzed"]    = len(post_bars)
    result["price_at_open_bar"] = post_bars[0]["open"]

    # Was the entry price ever traded?
    if side == "long":
        result["entry_achievable"] = any(b["low"] <= entry for b in post_bars)
    else:
        result["entry_achievable"] = any(b["high"] >= entry for b in post_bars)

    lows  = [b["low"]  for b in post_bars]
    highs = [b["high"] for b in post_bars]
    result["low_after_entry"]  = min(lows)
    result["high_after_entry"] = max(highs)

    # MAE / MFE (in R, from entry perspective)
    if side == "long":
        mae_price = min(lows)
        mfe_price = max(highs)
        result["mae_r"] = round((entry - mae_price) / risk, 2)  # positive = adverse
        result["mfe_r"] = round((mfe_price - entry) / risk, 2)
    else:
        mae_price = max(highs)
        mfe_price = min(lows)
        result["mae_r"] = round((mae_price - entry) / risk, 2)
        result["mfe_r"] = round((entry - mfe_price) / risk, 2)

    # Walk bar by bar to find which exits first: stop or 2R
    stop_hit_at = None
    t2r_hit_at  = None
    for i, b in enumerate(post_bars):
        if side == "long":
            if stop_hit_at is None and b["low"] <= stop:
                stop_hit_at = (i, b["ts"])
            if t2r is not None and t2r_hit_at is None and b["high"] >= t2r:
                t2r_hit_at = (i, b["ts"])
        else:
            if stop_hit_at is None and b["high"] >= stop:
                stop_hit_at = (i, b["ts"])
            if t2r is not None and t2r_hit_at is None and b["low"] <= t2r:
                t2r_hit_at = (i, b["ts"])

    if stop_hit_at and t2r_hit_at:
        if stop_hit_at[0] < t2r_hit_at[0]:
            result["first_exit"] = "stop"
            result["first_exit_time"] = stop_hit_at[1].strftime("%H:%M ET")
            result["bars_to_stop"] = stop_hit_at[0] + 1
        else:
            result["first_exit"] = "2R"
            result["first_exit_time"] = t2r_hit_at[1].strftime("%H:%M ET")
            result["bars_to_2r"] = t2r_hit_at[0] + 1
    elif stop_hit_at:
        result["first_exit"] = "stop"
        result["first_exit_time"] = stop_hit_at[1].strftime("%H:%M ET")
        result["bars_to_stop"] = stop_hit_at[0] + 1
    elif t2r_hit_at:
        result["first_exit"] = "2R"
        result["first_exit_time"] = t2r_hit_at[1].strftime("%H:%M ET")
        result["bars_to_2r"] = t2r_hit_at[0] + 1

    return result


# ── Report rendering ──────────────────────────────────────────────────────────

def _r_label(r_val) -> str:
    if r_val is None:
        return "?"
    r = float(r_val)
    if r >= 1.8:   return f"+{r:.2f}R  [WIN 2R+]"
    if r >= 0.9:   return f"+{r:.2f}R  [WIN partial]"
    if r >= -0.15: return f"{r:.2f}R   [SCRATCH]"
    return f"{r:.2f}R   [LOSS]"


def _exit_icon(first_exit: str) -> str:
    return {"2R": "✅", "stop": "🔴", "not_reached": "⏸", "session_close": "🔔"}.get(first_exit, "?")


def fmt_time(ts_str: str | None) -> str:
    dt = _parse_et(ts_str)
    return dt.strftime("%H:%M:%S ET") if dt else "?"


def build_report(trades: dict, session_diag: dict, validations: dict,
                 session_date: date, log_path: Path) -> str:
    lines = []
    now_et = datetime.now(ET)
    total_r = sum(float(t.get("result_r") or 0) for t in trades.values())
    wins      = [t for t in trades.values() if (t.get("result_r") or 0) >= 0.9]
    losses    = [t for t in trades.values() if (t.get("result_r") or 0) < -0.15]
    scratches = [t for t in trades.values() if t not in wins and t not in losses]

    # ── Header
    lines += [
        f"# Kingdom EOD Paper Trade Validation — {session_date}",
        f"",
        f"Generated: {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}",
        f"Log: `{log_path}`",
        f"",
        f"---",
        f"",
        f"## Session Overview",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Session date | {session_date} |",
        f"| Total polls | {session_diag.get('poll_count', '?')} |",
        f"| Symbols monitored | {session_diag.get('symbol_count', '?')} |",
        f"| Paper trades | {len(trades)} |",
        f"| Wins (≥0.9R) | {len(wins)} |",
        f"| Losses (<−0.15R) | {len(losses)} |",
        f"| Scratches | {len(scratches)} |",
        f"| **Total R** | **{total_r:+.2f}R** |",
    ]

    # ── Diagnostics
    diag_first = session_diag.get("diag_first") or {}
    diag_last  = session_diag.get("diag_last")  or {}
    ctx_first  = session_diag.get("context_first") or {}
    ctx_last   = session_diag.get("context_last")  or {}

    def _delta(key):
        v_last  = int(diag_last.get(key)  or 0)
        v_first = int(diag_first.get(key) or 0)
        return v_last - v_first

    lines += [
        f"",
        f"## Session Diagnostics",
        f"",
        f"| Counter | Session total |",
        f"|---------|--------------|",
        f"| rejected_stale_timing | {_delta('rejected_stale_timing')} |",
        f"| rejected_spread | {_delta('rejected_spread')} |",
        f"| rejected_no_live_confirmation | {_delta('rejected_no_live_confirmation')} |",
        f"| promoted_by_monitor_transition | {_delta('promoted_by_monitor_transition')} |",
        f"| promoted_by_news | {_delta('promoted_by_news')} |",
        f"",
        f"**Tape (last poll):** "
        f"live={session_diag.get('tape_counts',{}).get('live','?')} / "
        f"{session_diag.get('symbol_count','?')} symbols",
        f"",
        f"**Market context:**",
        f"- Open:  SPY={ctx_first.get('spy_trend','?')}  QQQ={ctx_first.get('qqq_trend','?')}  "
        f"risk_on={ctx_first.get('risk_on','?')}  breadth={ctx_first.get('breadth','?')}",
        f"- Close: SPY={ctx_last.get('spy_trend','?')}   QQQ={ctx_last.get('qqq_trend','?')}   "
        f"risk_on={ctx_last.get('risk_on','?')}  breadth={ctx_last.get('breadth','?')}",
        f"",
        f"---",
        f"",
        f"## Trade-by-Trade Validation",
    ]

    if not trades:
        lines.append("")
        lines.append("No paper trades triggered this session.")
    else:
        for i, (sym, t) in enumerate(trades.items(), 1):
            v = validations.get(sym) or {}
            side      = (t.get("side") or "long").upper()
            playbook  = t.get("playbook") or "?"
            entry     = t.get("entry_ref")
            stop      = t.get("stop_ref")
            t2r       = t.get("target_2r")
            risk      = round(abs((entry or 0) - (stop or 0)), 4) if entry and stop else None
            result_r  = t.get("result_r")
            exit_px   = t.get("exit_price")
            first_ex  = v.get("first_exit", "not_reached")
            icon      = _exit_icon(first_ex)

            lines += [
                f"",
                f"### PT-{i:02d} — {sym} — {side} ({playbook})",
                f"",
                f"**Trade plan**",
                f"",
                f"| Field | Value |",
                f"|-------|-------|",
                f"| Entry ref | ${entry} |",
                f"| Stop | ${stop} |",
                f"| Target 2R | ${t2r} |",
                f"| Risk/share | ${risk} |",
                f"| Price at trigger | ${t.get('price_at_trigger')} |",
                f"",
                f"**Timing**",
                f"",
                f"| Event | Time |",
                f"|-------|------|",
                f"| Opened | {fmt_time(t.get('opened_at_et'))} |",
                f"| Closed | {fmt_time(t.get('exit_at_et'))} via `{t.get('exit_state','?')}` |",
                f"",
                f"**Conditions at entry**",
                f"",
                f"| Condition | Value |",
                f"|-----------|-------|",
                f"| Tape live | {t.get('tape_live_at_trigger')} |",
                f"| Spread | {t.get('spread_pct_at_trigger')}% |",
                f"| ML score | {t.get('ml_score')} |",
                f"| p(2R in 30m) | {t.get('p_2r_30m')} ({t.get('probability_source')}) |",
                f"| Catalyst score | {t.get('catalyst_score')} |",
                f"| Catalyst tags | {t.get('catalyst_tags')} |",
                f"| Decision | `{t.get('decision_at_trigger')}` |",
                f"",
                f"**Price-level validation (1-min bars)**",
                f"",
                f"| Check | Result |",
                f"|-------|--------|",
                f"| Entry achievable at trigger time? | {'✅ Yes' if v.get('entry_achievable') else ('❌ No — price already moved' if v.get('entry_achievable') is False else '—')} |",
                f"| Bar open at trigger | ${v.get('price_at_open_bar')} |",
                f"| Session low after entry | ${v.get('low_after_entry')} |",
                f"| Session high after entry | ${v.get('high_after_entry')} |",
                f"| {icon} First exit hit | {first_ex.upper()} at {v.get('first_exit_time') or '—'} |",
                f"| Bars to stop | {v.get('bars_to_stop') or '—'} min |",
                f"| Bars to 2R | {v.get('bars_to_2r') or '—'} min |",
                f"| MAE (max adverse, R) | {v.get('mae_r')} |",
                f"| MFE (max favorable, R) | {v.get('mfe_r')} |",
                f"| Bars analyzed | {v.get('bars_analyzed')} |",
                f"| Note | {v.get('note') or '—'} |",
                f"",
                f"**Result**",
                f"",
                f"| Reported exit | ${exit_px} |",
                f"|---|---|",
                f"| Monitor result | {_r_label(result_r)} |",
                f"| Actual first exit (bars) | {first_ex.upper()} {_exit_icon(first_ex)} |",
            ]

            # State transition timeline
            transitions = t.get("transitions") or []
            if transitions:
                lines += [f"", f"**State timeline**", f""]
                lines.append("| Time | From | To | Price | Tape | Spread |")
                lines.append("|------|------|----|-------|------|--------|")
                for tr in transitions:
                    ts_et = _parse_et(tr.get("ts_et") or tr.get("ts"))
                    ts_str = ts_et.strftime("%H:%M:%S") if ts_et else "?"
                    lines.append(
                        f"| {ts_str} | {tr.get('from_state','?')} | {tr.get('to_state','?')} "
                        f"| ${tr.get('price','?')} | {'live' if tr.get('tape_live') else 'STALE'} "
                        f"| {tr.get('spread_pct','?')}% |"
                    )

            lines.append("")
            lines.append("---")

    # ── Validation questions
    lines += [
        f"",
        f"## End-of-Day Validation Questions",
        f"",
        f"Work through these for every session:",
        f"",
        f"1. **Entry slippage** — Was entry_ref achievable at the trigger time, or had price already moved?",
        f"2. **Stop vs 2R order** — For each trade, which got hit first in the bars? Does it match result_label?",
        f"3. **Tape at entry** — Were any trades opened with STALE tape? Should those be filtered out?",
        f"4. **Spread** — Were spreads acceptable at entry? Flag anything over 1.5% (large-cap) or 2.0% (small-cap).",
        f"5. **MAE** — Trades with high MAE but that recovered — are you holding through too much heat?",
        f"6. **MFE vs exit** — Trades where MFE >> exit price — are you exiting too early?",
        f"7. **ML score correlation** — Did higher ML score names perform better today?",
        f"8. **Context alignment** — Were trades in direction of SPY/QQQ trend? How did off-trend trades perform?",
        f"9. **Stale rejections** — {_delta('rejected_stale_timing')} stale rejections today — is the 2500ms threshold right?",
        f"10. **Missed setups** — Were there names in confirmed/arming that didn't get a trade opened? Why?",
    ]

    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kingdom EOD paper trade validator")
    parser.add_argument("--date", help="YYYYMMDD date (default: today)")
    parser.add_argument("--log", help="Path to JSONL log file (overrides --date lookup)")
    parser.add_argument("--no-bars", action="store_true",
                        help="Skip Alpaca bar fetch (faster, skips price-level validation)")
    args = parser.parse_args()

    # Resolve session date
    if args.date:
        session_date = date(int(args.date[:4]), int(args.date[4:6]), int(args.date[6:8]))
    else:
        session_date = datetime.now(ET).date()

    # Resolve log path(s)
    if args.log:
        log_paths = [Path(args.log)]
    else:
        date_str = session_date.strftime('%Y%m%d')
        base = LOG_DIR / f"kingdom_pt_{date_str}.jsonl"
        # Collect all session files for this date: base + _2, _3, … up to _50
        log_paths = []
        if base.exists():
            log_paths.append(base)
        n = 2
        while n <= 50:
            extra = LOG_DIR / f"kingdom_pt_{date_str}_{n}.jsonl"
            if extra.exists():
                log_paths.append(extra)
                n += 1
            else:
                break

    if not log_paths:
        print(f"ERROR: No log files found for {session_date} in {LOG_DIR}")
        print(f"  Is the paper trade monitor running? (tools/paper_trade_monitor.py)")
        print(f"  Or specify --log <path>")
        sys.exit(1)

    # Use first file as the canonical path for report naming
    log_path = log_paths[0]

    rows: list[dict] = []
    for lp in log_paths:
        chunk = load_log(lp)
        print(f"[eod-validate] Reading {lp}… ({len(chunk)} entries)")
        rows.extend(chunk)
    print(f"[eod-validate] {len(rows)} total log entries across {len(log_paths)} session(s)")

    trades = extract_trades(rows)
    session_diag = extract_session_diag(rows)
    print(f"[eod-validate] {len(trades)} paper trade(s) found")

    # Fetch 1-min bars and validate each trade
    validations: dict[str, dict] = {}
    if trades and not args.no_bars:
        print(f"[eod-validate] Fetching 1-min bars from Alpaca…")
        sys.path.insert(0, str(KINGDOM_DIR))
        client = _alpaca_client()
        for sym in trades:
            print(f"  {sym}…", end=" ", flush=True)
            bars = fetch_bars(client, sym, session_date)
            print(f"{len(bars)} bars")
            v = validate_trade_levels(trades[sym], bars)
            validations[sym] = v
    elif args.no_bars:
        print("[eod-validate] Skipping bar fetch (--no-bars)")

    # Build and write report
    report = build_report(trades, session_diag, validations, session_date, log_path)

    out_path = DESKTOP / f"kingdom_eod_{session_date.strftime('%Y%m%d')}.md"
    out_path.write_text(report)

    # Also print to terminal
    print()
    print("=" * 72)
    print(report)
    print("=" * 72)
    print(f"[eod-validate] Report saved → {out_path}")


if __name__ == "__main__":
    main()
