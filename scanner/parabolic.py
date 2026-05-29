"""
Parabolic Planner — pre-market acceleration scanner.

Identifies stocks showing parabolic pre-market momentum BEFORE open so they
appear on the watchlist at 4-7am, not at 9am when they've already run.

Scoring keys off:
  - PM move % vs prev close (magnitude)
  - Acceleration ratio: is it still moving late-PM or did it peak early?
  - Float turnover: fraction of float already traded in PM
  - RVOL: elevated relative volume
  - Catalyst freshness: news within 24h multiplies score

Run this strategy at 4am–9am. Auto-seeds from Alpaca Market Movers (top 25
gainers) when no symbol list is provided.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from core.plan_integrity import validate_plan as _validate_plan

from scanner.parabolic_signals import (
    float_exhaustion,
    halt_probability,
    gamma_exposure,
    tape_velocity,
    social_velocity,
    sector_cohort,
)

_ET = ZoneInfo("America/New_York")


def scan_parabolic_symbols(
    symbols: list[str],
    cfg,
    limit: int = 20,
    **kwargs,
) -> dict[str, Any]:
    """
    Parabolic pre-market scan.

    Returns the same dict schema as other scan strategies so the existing
    result-view / UI pipeline requires no changes.
    """
    from scanner.orb import (
        Candidate,
        _fetch_alpaca_news_batch,
        _screener_market_movers,
        _screener_most_actives,
        _stream_confidence_grade,
        _fetch_float_shares,
        resolve_session_date,
    )
    from scanner.indicators import avg_daily_volume, vwap as _vwap
    from providers.alpaca_provider import AlpacaProvider
    from providers.symbols import to_provider_symbol
    from sentiment.catalyst import CatalystService

    # ── Parameters ────────────────────────────────────────────────────────────
    min_pm_move_pct:      float = float(kwargs.get("min_pm_move_pct",      5.0))
    max_pm_move_pct:      float = float(kwargs.get("max_pm_move_pct",      500.0))
    min_price:            float = float(kwargs.get("min_price",             2.0))
    max_price:            float = float(kwargs.get("max_price",             50.0))
    min_rvol:             float = float(kwargs.get("min_rvol",              1.5))
    min_pm_bars:          int   = int(  kwargs.get("min_pm_bars",           3))
    min_avg20_dollar_vol: float = float(kwargs.get("min_avg20_dollar_vol",  250_000))
    use_catalyst:         bool  = bool( kwargs.get("use_catalyst",          True))
    catalyst_lookback:    int   = int(  kwargs.get("catalyst_lookback_hours", 72))
    store                       = kwargs.get("store")

    base_provider = kwargs.get("provider") or AlpacaProvider()
    from datetime import time as _time
    _now_et = datetime.now(_ET)
    if _time(4, 0) <= _now_et.time() < _time(9, 30):
        session_date = _now_et.date()   # premarket: use today, not yesterday
    else:
        session_date = resolve_session_date(base_provider)

    # ── Build symbol list ─────────────────────────────────────────────────────
    # If the caller provides symbols (full universe path), use them directly.
    # Only fall back to market movers when truly no symbols are provided.
    _full_market_seeded = False
    _universe_size = 0       # total Nasdaq symbols checked by snapshot loop
    _snapshot_passed = 0     # symbols that cleared the 3% snapshot floor
    if not symbols:
        # Seed with most_actives + gainers for guaranteed coverage of known movers
        gainers, losers = _screener_market_movers(base_provider, top=50)
        actives = _screener_most_actives(base_provider, top=50)
        seen: set[str] = set()
        symbols = []
        for s in actives + gainers + losers:
            if s not in seen:
                symbols.append(s); seen.add(s)
        # Expand to full market via snapshot pre-filter on the ORB universe
        try:
            from universe.nasdaq_symbols import fetch_us_equity_symbols, UniverseConfig
            import time as _time_mod
            all_syms = [s for s in fetch_us_equity_symbols(UniverseConfig()) if s not in seen]
            _universe_size = len(all_syms)
            SNAP_BATCH = 200  # smaller batches — discard each after extracting movers
            TIME_BUDGET = 180.0  # seconds
            t0 = _time_mod.monotonic()
            move_map: dict[str, float] = {}
            _snap_checked = 0
            for i in range(0, len(all_syms), SNAP_BATCH):
                if _time_mod.monotonic() - t0 > TIME_BUDGET:
                    break
                batch = all_syms[i:i + SNAP_BATCH]
                _snap_checked += len(batch)
                try:
                    snaps = base_provider.get_snapshots(batch, feed="sip", timeout_s=10.0)
                except Exception:
                    snaps = {}
                # Extract movers then immediately discard the full snapshot dict
                for sym in batch:
                    snap = snaps.get(sym) or {}
                    if snap.get("error"):
                        continue
                    prev   = snap.get("prev_daily_bar") or {}
                    trade  = snap.get("latest_trade") or {}
                    price  = trade.get("price") or (snap.get("daily_bar") or {}).get("close")
                    prev_c = prev.get("close")
                    if not price or not prev_c or prev_c <= 0:
                        continue
                    if not (min_price <= float(price) <= max_price):
                        continue
                    move = (float(price) - float(prev_c)) / float(prev_c) * 100.0
                    if move >= 3.0:
                        move_map[sym] = move
                del snaps  # free memory immediately after each batch
            _universe_size = _snap_checked
            # All symbols that passed the snapshot floor — no cap, full Nasdaq coverage
            top_movers = sorted(move_map, key=lambda s: -move_map[s])
            _snapshot_passed = len(top_movers)
            for s in top_movers:
                if s not in seen:
                    symbols.append(s); seen.add(s)
            _full_market_seeded = True
        except Exception as _seed_exc:
            import logging as _lg2; _lg2.getLogger(__name__).warning("PARABOLIC_SEED_FAIL %s", _seed_exc)

    shortlisted: list[str] = []
    for s in symbols:
        y = to_provider_symbol(s)
        if y and "." not in y:
            shortlisted.append(y)

    # ── Snapshot pre-filter (large universe only) ─────────────────────────────
    # Skip when full-market seeding already snapshot-qualified every symbol.
    _SNAPSHOT_PREFILTER_THRESHOLD = 20
    if not _full_market_seeded and len(shortlisted) > _SNAPSHOT_PREFILTER_THRESHOLD:
        try:
            BATCH = 500
            snap_passed: list[str] = []
            _cur_t_pf = datetime.now(_ET).time()
            from datetime import time as _time3
            _is_pm_pf = _time3(4, 0) <= _cur_t_pf < _time3(9, 30)
            # pre-market: use a low floor (3%) so thin movers aren't culled before bar-level check
            rough_move_floor = 3.0 if _is_pm_pf else min_pm_move_pct * 0.5
            for i in range(0, len(shortlisted), BATCH):
                batch = shortlisted[i:i + BATCH]
                try:
                    snaps = base_provider.get_snapshots(batch, feed="sip", timeout_s=20.0)
                except Exception:
                    # If snapshots fail, keep the whole batch for bar-level filtering
                    snap_passed.extend(batch)
                    continue
                _cur_t = datetime.now(_ET).time()
                from datetime import time as _time2
                is_premarket = _time2(4, 0) <= _cur_t < _time2(9, 30)
                for sym in batch:
                    snap = snaps.get(sym) or {}
                    if snap.get("error"):
                        continue
                    daily_bar  = snap.get("daily_bar")  or {}
                    prev_bar   = snap.get("prev_daily_bar") or {}
                    cur_price  = (snap.get("latest_trade") or {}).get("price") or daily_bar.get("close")
                    prev_close = prev_bar.get("close")
                    day_vol    = daily_bar.get("volume") or 0
                    if not cur_price or not prev_close or prev_close <= 0:
                        continue
                    # pre-market: daily bar hasn't formed yet so day_vol is 0 — skip this gate
                    if not is_premarket and day_vol <= 0:
                        continue
                    if not (min_price <= float(cur_price) <= max_price):
                        continue
                    rough_move = (float(cur_price) - float(prev_close)) / float(prev_close) * 100.0
                    if rough_move < rough_move_floor:
                        continue
                    snap_passed.append(sym)
            shortlisted = snap_passed
            import logging as _lg; _lg.getLogger(__name__).warning("PARABOLIC_DEBUG snap_passed=%d", len(snap_passed))
        except Exception as _pf_exc:
            import logging as _lg; _lg.getLogger(__name__).warning("PARABOLIC_DEBUG prefilter_exception=%s", _pf_exc)
            pass  # if pre-filter blows up entirely, keep original shortlisted and let _one() filter

    # ── Time anchors ─────────────────────────────────────────────────────────
    now_et     = datetime.now(_ET)
    now_t      = now_et.time()
    from datetime import time as _time
    _PRE_OPEN  = _time(4, 0)
    _MKT_OPEN  = _time(9, 30)
    _MKT_CLOSE = _time(16, 0)

    if _PRE_OPEN <= now_t < _MKT_OPEN:
        scan_window = "optimal"          # 4 AM – 9:30 AM: designed use case
    elif _MKT_OPEN <= now_t < _MKT_CLOSE:
        scan_window = "intraday"         # 9:30 AM – 4 PM: data is from this morning's PM, still same day
    else:
        scan_window = "post_close"       # after 4 PM or before 4 AM: PM data is from today's already-closed session

    pm_open_t  = datetime.combine(session_date, datetime.min.time()).replace(
        hour=4, minute=0, tzinfo=_ET)
    pm_split_t = datetime.combine(session_date, datetime.min.time()).replace(
        hour=6, minute=30, tzinfo=_ET)
    pm_close_t = datetime.combine(session_date, datetime.min.time()).replace(
        hour=9, minute=30, tzinfo=_ET)

    candidates:    list[Candidate] = []
    data_failures: list[dict]      = []
    reject_counts: dict[str, int]  = {}

    # ── Per-symbol worker ─────────────────────────────────────────────────────
    def _one(sym: str):
        import math
        import numpy as np

        try:
            # 1. All 1m bars for the day (including premarket)
            bars = base_provider.get_bars_range(
                symbol=sym, interval="1m",
                from_d=session_date, to_d=session_date,
                include_prepost=True, timeout_s=15, feed="sip",
            )
            if bars is None or bars.empty:
                return ("fail", sym, "no_bars")

            idx = bars.index
            if hasattr(idx, "tz") and idx.tz is None:
                idx = idx.tz_localize("UTC")
            idx_et = idx.tz_convert(_ET)

            # 2. Slice to PM window
            pm_mask   = (idx_et >= pm_open_t)  & (idx_et < pm_close_t)
            pm_bars   = bars[pm_mask]
            if len(pm_bars) < min_pm_bars:
                return ("fail", sym, "filtered_insufficient_pm_bars")

            pm_last   = float(pm_bars["Close"].iloc[-1])
            pm_high   = float(pm_bars["High"].max())
            pm_low    = float(pm_bars["Low"].min())
            pm_volume = float(pm_bars["Volume"].sum())
            pm_trades_total   = int(pm_bars["Trades"].sum()) if "Trades" in pm_bars.columns else 0
            pm_avg_trade_size = pm_volume / max(1, pm_trades_total) if pm_trades_total > 0 else 0.0

            # Price gate
            if not (min_price <= pm_last <= max_price):
                return ("fail", sym, "filtered_price")

            # 3. Previous close from daily history
            try:
                daily = base_provider.get_daily_history(sym, period="90d")
            except Exception:
                return ("fail", sym, "no_daily_history")
            if daily is None or daily.empty:
                return ("fail", sym, "no_daily_history")

            daily = daily.sort_index()
            didx = daily.index
            if hasattr(didx, "tz") and didx.tz is not None:
                didx_et = didx.tz_convert(_ET)
            else:
                didx_et = didx.tz_localize("UTC").tz_convert(_ET)

            session_start = datetime.combine(session_date, datetime.min.time()).replace(tzinfo=_ET)
            prior_daily   = daily[didx_et < session_start]
            if prior_daily.empty:
                return ("fail", sym, "no_prev_session")

            prev_row   = prior_daily.iloc[-1]
            prev_close = float(prev_row["Close"])
            if prev_close <= 0:
                return ("fail", sym, "invalid_prev_close")

            # 4. PM move %
            pm_move_pct = (pm_last - prev_close) / prev_close * 100.0
            if pm_move_pct < min_pm_move_pct:
                return ("fail", sym, "filtered_pm_move_too_small")
            if pm_move_pct > max_pm_move_pct:
                return ("fail", sym, "filtered_pm_move_too_large")

            # 5. Avg20 dollar vol + RVOL
            avg20_vol = float(avg_daily_volume(prior_daily, window=20) or 0.0)
            avg20_dollar_vol = float(avg20_vol * prev_close) if avg20_vol > 0 else 0.0
            avg20_trade_count = (
                float(prior_daily["Trades"].dropna().tail(20).mean())
                if "Trades" in prior_daily.columns and not prior_daily["Trades"].dropna().empty
                else 0.0
            )
            historical_avg_trade_size = avg20_vol / max(1.0, avg20_trade_count) if avg20_trade_count > 0 else 0.0
            if avg20_dollar_vol < min_avg20_dollar_vol:
                return ("fail", sym, "filtered_avg20_dollar_vol")

            # RVOL: during intraday use ALL today's volume (PM + market hours so far)
            # so the projection isn't stuck dividing by full 5.5h of PM on a tiny volume.
            now_et_sym = datetime.now(_ET)
            if scan_window == "intraday":
                today_mask = idx_et >= pm_open_t
                today_vol  = float(bars[today_mask]["Volume"].sum())
                # 4am–4pm = 12h full window; use actual elapsed from 4am
                elapsed_total_h = max(0.1, min(12.0, (now_et_sym - pm_open_t).total_seconds() / 3600.0))
                projected_day_vol = (today_vol / elapsed_total_h) * 12.0
            else:
                pm_hours = max(0.1, (pm_close_t - pm_open_t).seconds / 3600.0)
                elapsed_h = max(0.1, min(pm_hours, (now_et_sym - pm_open_t).total_seconds() / 3600.0))
                projected_day_vol = (pm_volume / elapsed_h) * 6.5
            rvol = projected_day_vol / avg20_vol if avg20_vol > 0 else 0.0
            if rvol < min_rvol:
                return ("fail", sym, "filtered_rvol")

            # 6. Acceleration: early PM (4–6:30) vs late PM (6:30–now)
            early_mask = (idx_et >= pm_open_t)  & (idx_et < pm_split_t)
            late_mask  = (idx_et >= pm_split_t) & (idx_et < pm_close_t)
            early_bars = bars[early_mask]
            late_bars  = bars[late_mask]

            acceleration_ratio = 1.0  # neutral default
            if not early_bars.empty and not late_bars.empty:
                price_at_split = float(early_bars["Close"].iloc[-1])
                early_move = (price_at_split - prev_close) / max(0.01, prev_close) * 100.0
                late_move  = (pm_last - price_at_split) / max(0.01, price_at_split) * 100.0
                if abs(early_move) > 0.1:
                    acceleration_ratio = late_move / abs(early_move)
            elif not early_bars.empty and late_bars.empty:
                # All the move happened in early PM — it may have peaked
                acceleration_ratio = 0.5
            # else: only late bars (started scanning late) → keep 1.0 neutral

            # 7. Float turnover
            float_turnover_pct: float | None = None
            float_sh = _fetch_float_shares(sym, store)
            if float_sh and float_sh > 0:
                float_turnover_pct = (pm_volume / float_sh) * 100.0

            # 8. VWAP from PM bars (best-effort)
            vwap_last = None
            try:
                vw = _vwap(pm_bars)
                vwap_last = float(vw.iloc[-1]) if not vw.empty else None
            except Exception:
                pass

            # 9. Composite score
            accel_factor    = min(2.0, max(0.5, acceleration_ratio))
            rvol_factor     = min(5.0, rvol) / 5.0
            # Catalyst factor applied after candidate is built (batch fetch)
            float_factor    = 1.0
            if float_turnover_pct is not None:
                float_factor = 1.4 if float_turnover_pct >= 100 else (1.2 if float_turnover_pct >= 50 else 1.0)

            # Institutional footprint: large avg trade size → block/institutional prints
            institutional_factor = 1.0
            if historical_avg_trade_size > 0 and pm_avg_trade_size > 0:
                ratio = pm_avg_trade_size / historical_avg_trade_size
                institutional_factor = min(2.0, max(1.0, 1.0 + (ratio - 1.0) * 0.5))

            # Fix #20: log-normalize so scores have meaningful spread across movers
            # log2(1+5%)=2.6, log2(1+30%)=5.0, log2(1+100%)=6.7 — calibrated range
            _base = math.log2(1.0 + abs(pm_move_pct)) * 10.0
            composite_score = _base * accel_factor * rvol_factor * float_factor * institutional_factor

            # 10. New predictive signals (signals 1–5)
            _sig_exhaust = float_exhaustion(pm_bars, float_sh)
            _sig_halt    = halt_probability(
                sym, pm_move_pct, float_sh, now_et,
                kwargs.get("stream_cache"),
            )
            _sig_gamma   = gamma_exposure(sym, pm_last, base_provider, store)
            _sig_tape    = tape_velocity(pm_bars, historical_avg_trade_size)
            # news_rows sourced later in batch; pass empty for now (enriched post-scan)
            _sig_social  = social_velocity(sym, [])

            # Apply signal multipliers to composite score
            composite_score *= _sig_exhaust["factor"]
            composite_score *= _sig_halt["factor"]
            composite_score *= _sig_gamma["factor"]
            composite_score *= _sig_tape["factor"]
            composite_score *= _sig_social["factor"]

            stop_pct = max(0.03, min(0.08, pm_move_pct * 0.15 / 100.0))
            entry    = round(pm_last * 1.003, 4)
            stop     = round(pm_last * (1.0 - stop_pct), 4)
            risk     = max(0.01, abs(entry - stop))
            risk_dollars = float(getattr(cfg, "risk_dollars", 50.0))
            shares   = max(1, int(risk_dollars / risk))
            notional = shares * entry
            target_2r = round(entry + 2 * risk, 4)
            target_3r = round(entry + 3 * risk, 4)
            tradable_now = bool(pm_last >= entry * 0.998)

            c = Candidate(
                symbol=sym,
                data_date=session_date.isoformat(),
                last_price=pm_last,
                pct_change=pm_move_pct,
                rvol=rvol,
                today_dollar_vol=pm_volume * pm_last,
                avg20_dollar_vol=avg20_dollar_vol,
                or_high=pm_high,
                or_low=pm_low,
                or_range_pct=pm_move_pct,
                above_vwap=bool(pm_last >= vwap_last) if vwap_last else None,
                vwap_last=vwap_last,
                vwap_delta_pct=((pm_last - vwap_last) / vwap_last * 100.0) if vwap_last else None,
                trend_state=None,
                trend_slope_pct=None,
                best_side="long",
                entry=entry,
                stop=stop,
                target_2r=target_2r,
                target_3r=target_3r,
                risk_per_share=risk,
                shares=shares,
                notional=notional,
                long_entry=entry,
                long_stop=stop,
                long_2r=target_2r,
                long_3r=target_3r,
                long_risk_per_share=risk,
                long_shares=shares,
                long_notional=notional,
                short_entry=None,
                short_stop=None,
                short_2r=None,
                short_3r=None,
                short_risk_per_share=None,
                short_shares=None,
                short_notional=None,
                strategy="parabolic",
                exec_style="parabolic",
                tradable_now=tradable_now,
                trade_ready_passes=tradable_now,
                prev_close=prev_close,
                prev_day_high=float(prev_row["High"]) if "High" in prev_row.index else prev_close,
                dist_from_prev_close_pct=pm_move_pct,
                scan_ts=datetime.now(timezone.utc).isoformat(),
                live_price=pm_last,
                notes=(
                    f"pm_move={pm_move_pct:+.1f}%  accel={acceleration_ratio:.2f}x  "
                    f"rvol={rvol:.1f}x"
                    + (f"  float_turn={float_turnover_pct:.0f}%" if float_turnover_pct is not None else "")
                    + (f"  avg_trade_sz={pm_avg_trade_size:.0f}sh  inst={institutional_factor:.2f}x" if pm_avg_trade_size > 0 else "")
                ),
                combined_score=composite_score,
                confidence_score=min(100.0, composite_score),
                confidence_grade=None,
                gate_passes=tradable_now,
                gate_fail_reasons=[] if tradable_now else ["awaiting_trigger"],
            )
            # Attach parabolic-specific fields as extras
            c._parabolic_accel            = acceleration_ratio
            c._parabolic_float_turn       = float_turnover_pct
            c._parabolic_pm_move          = pm_move_pct
            c._parabolic_inst_factor      = institutional_factor
            c._parabolic_pm_avg_trd_sz    = pm_avg_trade_size
            c._parabolic_hist_avg_trd_sz  = historical_avg_trade_size
            c._parabolic_pm_trades        = pm_trades_total
            # Signal 1 — Float Exhaustion
            c._para_exhaust_pct           = _sig_exhaust.get("exhaust_pct")
            c._para_exhaust_eta_min       = _sig_exhaust.get("eta_min")
            c._para_exhaust_factor        = _sig_exhaust.get("factor", 1.0)
            # Signal 2 — Halt Probability
            c._para_halt_prob             = _sig_halt.get("halt_prob", 0.0)
            c._para_is_halted             = _sig_halt.get("is_halted", False)
            c._para_recently_resumed      = _sig_halt.get("recently_resumed", False)
            c._para_resume_signal         = _sig_halt.get("resume_signal", False)
            # Signal 3 — Gamma Exposure
            c._para_gamma_wall_strike     = _sig_gamma.get("gamma_wall_strike")
            c._para_gamma_wall_oi         = _sig_gamma.get("gamma_wall_oi")
            c._para_gamma_score           = _sig_gamma.get("gamma_score", 0.0)
            c._para_near_gamma_wall       = _sig_gamma.get("near_gamma_wall", False)
            c._para_iv_rank               = _sig_gamma.get("iv_rank")
            # Signal 4 — Tape Velocity
            c._para_trades_per_min        = _sig_tape.get("trades_per_min")
            c._para_size_ratio            = _sig_tape.get("size_ratio")
            c._para_velocity_score        = _sig_tape.get("velocity_score", 0.0)
            c._para_is_blow_off           = _sig_tape.get("is_blow_off", False)
            # Signal 5 — Social Velocity (enriched post-scan)
            c._para_mention_velocity      = None
            c._para_buzz_score            = 0.0
            c._para_social_signal         = "neutral"
            # Signal 6 — Sector Cohort (enriched post-scan)
            c._para_sector_name           = None
            c._para_cohort_count          = 1
            c._para_cohort_score          = 0.0

            return ("ok", sym, c)

        except Exception as exc:
            return ("fail", sym, f"exception:{type(exc).__name__}:{exc}")

    workers = min(len(shortlisted), int(os.getenv("ORB_SCAN_WORKERS", 8)))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_one, sym): sym for sym in shortlisted}
        for fut in as_completed(futs):
            status, sym, payload = fut.result()
            if status == "ok":
                candidates.append(payload)
            else:
                code = str(payload)
                reject_counts[code] = reject_counts.get(code, 0) + 1
                data_failures.append({"symbol": sym, "stage": "parabolic", "error": code})

    # ── Catalyst enrichment ───────────────────────────────────────────────────
    if use_catalyst and candidates:
        try:
            cat_svc = CatalystService(provider=base_provider)
            cat_map = cat_svc.fetch_batch(
                [c.symbol for c in candidates],
                lookback_hours=catalyst_lookback,
            ) or {}
            for c in candidates:
                rec = cat_map.get(c.symbol.upper())
                if rec is None:
                    continue
                c.catalyst_score           = float(rec.score or 0.0) or None
                c.catalyst_confidence      = float(rec.confidence or 0.0) or None
                c.catalyst_strength        = float(rec.strength or 0.0) or None
                c.catalyst_article_count   = rec.article_count
                c.catalyst_freshness_hours = rec.freshness_hours
                c.catalyst_tags            = rec.tags
                # Apply catalyst factor to combined score
                fresh_h = rec.freshness_hours
                cat_factor = 1.3 if (fresh_h is not None and fresh_h < 24) else 1.0
                c.combined_score    = float(c.combined_score or 0.0) * cat_factor
                c.confidence_score  = min(100.0, c.combined_score)
        except Exception:
            pass

    # ── Signal 5 — Social Velocity (post-scan, reuses news already fetched) ───
    if candidates:
        try:
            news_map_raw = _fetch_alpaca_news_batch([c.symbol for c in candidates], base_provider)
            for c in candidates:
                # Build news_rows list from catalyst articles if available
                news_rows: list[dict] = []
                cat_arts = getattr(c, "_catalyst_articles", None)
                if cat_arts:
                    news_rows = [{"created_at": a.created_at} for a in cat_arts if a.created_at]
                sig = social_velocity(c.symbol, news_rows)
                c._para_mention_velocity  = sig.get("mention_velocity")
                c._para_buzz_score        = sig.get("buzz_score", 0.0)
                c._para_social_signal     = sig.get("social_signal", "neutral")
                # Apply social factor to combined score
                c.combined_score   = float(c.combined_score or 0.0) * sig.get("factor", 1.0)
                c.confidence_score = min(100.0, c.combined_score)
        except Exception:
            pass

    # ── Signal 6 — Sector Cohort Amplification (post-scan) ───────────────────
    if candidates:
        try:
            cohort_map = sector_cohort([c.symbol for c in candidates], store)
            for c in candidates:
                cohort = cohort_map.get(c.symbol, {})
                c._para_sector_name  = cohort.get("sector_name")
                c._para_cohort_count = cohort.get("cohort_count", 1)
                c._para_cohort_score = cohort.get("cohort_score", 0.0)
                factor = cohort.get("factor", 1.0)
                c.combined_score   = float(c.combined_score or 0.0) * factor
                c.confidence_score = min(100.0, c.combined_score)
        except Exception:
            pass

    # Set confidence grades after all scoring adjustments
    for c in candidates:
        c.confidence_grade = _stream_confidence_grade(c.confidence_score)

    # ── News enrichment ───────────────────────────────────────────────────────
    if candidates:
        try:
            news_map = _fetch_alpaca_news_batch([c.symbol for c in candidates], base_provider)
            for c in candidates:
                if c.symbol in news_map:
                    c.news_headline, c.news_age_hours = news_map[c.symbol]
        except Exception:
            pass

    # ── Opening auction confirmation (post-9:30am only) ───────────────────────
    if candidates:
        try:
            from scanner.orb import _fetch_opening_auction
            auction_map = _fetch_opening_auction(
                [c.symbol for c in candidates], base_provider, session_date
            )
            for c in candidates:
                rec = auction_map.get(c.symbol.upper())
                if rec:
                    ap = rec["auction_price"]
                    c.auction_gap_pct    = round((ap - c.last_price) / max(0.01, c.last_price) * 100.0, 2)
                    c.auction_price      = ap
                    c.auction_volume     = rec["auction_volume"]
        except Exception:
            pass

    # ── Sort: tradable first, then by composite score ─────────────────────────
    candidates.sort(
        key=lambda c: (float(c.tradable_now or 0), float(c.combined_score or 0)),
        reverse=True,
    )
    top = candidates[:int(limit)]

    # ── Integrity gate: drop structurally corrupt candidates ──────────────────
    def _integrity_ok(c) -> bool:
        try:
            side = getattr(c, "best_side", None) or "short"
            entry = float(c.entry or 0) if getattr(c, "entry", None) else None
            stop  = float(c.stop or 0)  if getattr(c, "stop",  None) else None
            tgt   = float(c.target_2r or 0) if getattr(c, "target_2r", None) else None
            price = float(c.last_price or 0) if getattr(c, "last_price", None) else None
            return bool(_validate_plan(side=side, entry=entry, stop=stop,
                                        target=tgt, current_price=price, symbol=c.symbol))
        except Exception:
            return True  # don't drop on unexpected error
    top = [c for c in top if _integrity_ok(c)]

    def _cdict(c):
        d = asdict(c)
        d["price"] = float(c.last_price or 0.0)
        d["scan_date"] = session_date.isoformat()
        d["scan_ts"] = c.scan_ts or datetime.now(timezone.utc).isoformat()
        # Parabolic-specific extra fields (not in dataclass schema)
        _extra_attrs = (
            # original parabolic fields
            "_parabolic_accel", "_parabolic_float_turn", "_parabolic_pm_move",
            "_parabolic_inst_factor", "_parabolic_pm_avg_trd_sz", "_parabolic_hist_avg_trd_sz",
            "_parabolic_pm_trades",
            # signal 1 — float exhaustion
            "_para_exhaust_pct", "_para_exhaust_eta_min", "_para_exhaust_factor",
            # signal 2 — halt probability
            "_para_halt_prob", "_para_is_halted", "_para_recently_resumed", "_para_resume_signal",
            # signal 3 — gamma exposure
            "_para_gamma_wall_strike", "_para_gamma_wall_oi", "_para_gamma_score",
            "_para_near_gamma_wall", "_para_iv_rank",
            # signal 4 — tape velocity
            "_para_trades_per_min", "_para_size_ratio", "_para_velocity_score", "_para_is_blow_off",
            # signal 5 — social velocity
            "_para_mention_velocity", "_para_buzz_score", "_para_social_signal",
            # signal 6 — sector cohort
            "_para_sector_name", "_para_cohort_count", "_para_cohort_score",
            # auction
            "auction_gap_pct", "auction_price", "auction_volume",
        )
        for attr in _extra_attrs:
            v = getattr(c, attr, None)
            if v is not None:
                d[attr.lstrip("_")] = v
        return d

    return {
        "provider":           getattr(base_provider, "name", "alpaca"),
        "scan_date":          session_date.isoformat(),
        "scan_window":        scan_window,
        "regime":             {},
        "debug": {
            "session_date_used": session_date.isoformat(),
            "failure_samples":   data_failures[:10],
            "failure_samples_by_code": reject_counts,
        },
        "count":              len(top),
        "candidates_total":   len(candidates),
        "rejected_total":     len(data_failures),
        "tradable_now_total": sum(1 for c in candidates if c.tradable_now),
        "trade_ready_total":  sum(1 for c in candidates if c.trade_ready_passes),
        "candidates": [_cdict(c) for c in top],
        "seed_candidates_total": len(candidates),
        "seed_candidates": [_cdict(c) for c in candidates],
        "shortlisted":       len(shortlisted),
        "universe_size":     _universe_size,
        "snapshot_passed":   _snapshot_passed,
        "reject_counts":     reject_counts,
        "data_failures":     data_failures[:20],
    }
