# Monitor-First Rebuild Report

This report describes how the rebuilt project implements the requested monitor-first architecture using the final deliverable in this zip.

## Project outcome

The rebuilt app keeps the Flask dashboard on port 8050, promotes live monitoring to the center of the workflow, persists runtime state to SQLite, adds a live alert tape, introduces market-context scoring, performs news-driven promotions, supports replay, exposes SSE for browser push, includes optional Redis pub/sub hooks, and adds a second crypto page backed by real Alpaca crypto snapshots.

## Must-have implementation mapping

### 1. Convert the product from scan-first to monitor-first

Implemented by automatically handing completed scans to the live monitor and returning `monitor_id` in the scan result.

**Code**
- `app.py:2143` assigns the auto-started monitor session back into the scan result.
- `templates/index.html:2472` auto-attaches the UI to `resSummary.monitor_id`.
- `monitor/live_monitor.py:635` contains the new monitor-first manager.

```python
# app.py
if not _is_truthy(params.get("disable_auto_monitor", "0")):
    sess = _MONITOR.start_from_scan_candidates(...)
    result["monitor_id"] = sess.monitor_id
```

### 2. Automatically start a live monitor session from completed scan results

Implemented in the scan worker immediately after the result payload is assembled.

**Code**
- `app.py:2143`
- `monitor/live_monitor.py:763`

```python
sess = _MONITOR.start_from_scan_candidates(
    job_id=jid,
    candidates=top,
    top_n=monitor_top_n,
    feed=(os.getenv("ALPACA_DATA_FEED") or "sip").strip().lower() or "sip",
    provider=provider,
    stream_cache=_STREAM,
    long_only=_is_truthy(params.get("long_only", "0")),
    source="auto_scan_handoff",
    promotion_candidates=slice_syms,
)
```

### 3. Continuously recompute watched symbol state

Implemented by the background loop in the monitor manager.

**Fields recomputed**
- trigger state
- retest confirmation
- VWAP location / distance
- spread quality
- live chase-R
- tape freshness / deterioration
- market-session state

**Code**
- `monitor/live_monitor.py:681` background loop
- `monitor/live_monitor.py:1025` `_apply_update_locked`
- `monitor/live_monitor.py:1063` `_compute_tape_locked`
- `monitor/live_monitor.py:1084` `_compute_state_and_alerts_locked`

```python
def _run_loop(self) -> None:
    while not self._stop.is_set():
        ...
        self.refresh(monitor_id=mid, provider=self._provider, stream_cache=self._stream_cache, force=False)
```

### 4. Emit alerts on state transitions, not static candidate presence

Implemented by `MonitorAlertEvent` plus `_emit_alert_locked()`.

**Code**
- `monitor/live_monitor.py:145` `MonitorAlertEvent`
- `monitor/live_monitor.py:1226` `_emit_alert_locked`
- `templates/index.html:3198` alert tape rendering

```python
if st.just_transitioned:
    self._emit_alert_locked(sess, st, old_state, new_state, reasons=..., flags=..., now_ts=now_ts)
```

### 5. Promote fresh catalyst/news symbols into the watch universe before normal ranking

Implemented by `_promote_news_locked()` using real Alpaca news through `CatalystService` and then building a real seed from provider intraday bars.

**Code**
- `monitor/live_monitor.py:1327`
- `monitor/live_monitor.py:786` `_build_seed_from_provider`
- `runtime/state_store.py:243` `append_news_event`

```python
bundles = self._catalyst.fetch_batch(source_symbols, per_symbol_limit=4, lookback_hours=8)
...
st = self._build_seed_from_provider(sym, playbook="catalyst_news_ignition", seed_source="news")
st.promoted_by_news = True
sess.symbols[sym] = st
```

### 6. Split logic into distinct playbooks

Implemented with explicit playbook classes and registry.

**Playbooks**
- open-drive ORB
- retest/reclaim ORB
- midday continuation
- range reversion
- catalyst/news ignition
- sympathy continuation

**Code**
- `monitor/live_monitor.py:430` `OpenDriveORBPlaybook`
- `monitor/live_monitor.py:485` `RetestReclaimPlaybook`
- `monitor/live_monitor.py:524` `MiddayContinuationPlaybook`
- `monitor/live_monitor.py:537` `RangeReversionPlaybook`
- `monitor/live_monitor.py:567` `CatalystNewsIgnitionPlaybook`
- `monitor/live_monitor.py:585` `SympathyContinuationPlaybook`
- `monitor/live_monitor.py:625` `PLAYBOOKS`

### 7. Add live market context inputs

Implemented by `MarketContextEngine`.

**Inputs**
- SPY / QQQ direction
- sector strength / weakness
- volatility regime
- time-of-day decay bucket
- cohort tape quality

**Code**
- `macro/context_engine.py:13` sector ETF universe
- `macro/context_engine.py:59` engine class
- `macro/context_engine.py:139` snapshot computation
- `app.py:1079` `/api/context_status`
- `templates/index.html:3197` context rendering

```python
return ContextSnapshot(
    spy_trend_state=spy_state,
    qqq_trend_state=qqq_state,
    breadth_score=round(breadth_score, 4),
    risk_on_score=round(risk_on_score, 4),
    volatility_regime=volatility_regime,
    time_of_day_bucket=self._bucket_time_of_day(...),
    sector_strength_by_etf=...,
    cohort_tape_quality=...,
)
```

### 8. Add diagnostics that attribute rejections/promotions

Implemented as explicit counters and failure samples stored on the monitor session.

**Tracked diagnostics**
- rejected_stale_timing
- rejected_spread
- rejected_no_live_confirmation
- rejected_no_catalyst_freshness
- rejected_multi_timeframe_disagreement
- promoted_by_news
- promoted_by_monitor_transition
- demoted_by_decay

**Code**
- `monitor/live_monitor.py:373` session diagnostics schema
- `monitor/live_monitor.py:1069` stale/spread/no-live diagnostics
- `monitor/live_monitor.py:1093` context and catalyst diagnostics

### 9. Keep Alpaca as the primary provider and use real-time streams as source of truth

Implemented by keeping the existing Alpaca provider + stream cache for stock monitoring and reading stream cache first before REST.

**Code**
- `app.py:117` `_ALPACA_PROVIDER = AlpacaProvider()`
- `app.py:167` stream cache startup
- `monitor/live_monitor.py:978` trade/quote reads prefer stream cache
- `providers/streaming.py` remains the real stock stream cache

### 10. Keep the current Flask/Dash-style app working while internals are refactored

Implemented by retaining:
- Flask app entrypoint
- index template
- scan endpoints
- monitor endpoints

And extending them with:
- `/api/alerts_recent`
- `/api/context_status`
- `/api/alerts_stream`
- `/api/replay_session`
- `/api/watch_start`
- `/api/watch_status`

**Code**
- `app.py:1069`
- `app.py:1079`
- `app.py:1089`
- `app.py:1101`

## Should-have implementation mapping

### SQLite persistence
Implemented in `runtime/state_store.py`.

Tables:
- `watch_sessions`
- `watch_symbols`
- `alert_events`
- `news_events`
- `context_snapshots`
- `replay_sessions`

### Per-playbook cooldowns
Implemented in `monitor/live_monitor.py` via `self._cooldowns`.

### Real-time alert tape in UI
Implemented in `templates/index.html` with `renderAlertTape()` and the `Alert Tape` card.

### Scanner mode and watchlist mode
Implemented via `MonitorSession.mode` and `MonitorSession.watch_mode`, plus `start_from_scan_candidates()` and `start_from_symbols()`.

### Session-open, midday, and late-day score adjustments
Implemented with `time_of_day_bucket` and playbook scoring.

### Replay mode
Implemented with:
- `runtime/state_store.py:305`
- `app.py:1089` `/api/replay_session`

### SSE/WebSocket push to browser instead of short polling
Implemented with SSE:
- `app.py:1101` `/api/alerts_stream`
- `templates/index.html` EventSource consumer

### Redis-backed pub/sub
Implemented as optional runtime publisher:
- `runtime/pubsub.py`
- used by alert emission in `monitor/live_monitor.py:1248`

### Separate services for scanner, monitor, news router, and alerting
Added under `services/`:
- `scanner_service.py`
- `monitor_service.py`
- `news_router_service.py`
- `alert_service.py`

### Optional Level 2 / order book quality metrics
The app continues to store real `bid`, `ask`, `bid_size`, and `ask_size` in monitor symbol state, which are used as real quote-quality inputs. Full depth-of-book was not added because the current provider layer in this codebase does not yet expose a stock L2 feed.

## Crypto second page

A second page was added at `/crypto`.

**Code**
- `app.py:2946` `/crypto`
- `app.py:2950` `/api/crypto_snapshot`
- `templates/crypto.html`

This page uses the real Alpaca crypto snapshot endpoint and computes real breakout-style ticket math from the returned latest trade, quote, and minute bar fields.

## Files added or heavily changed

### Added
- `runtime/state_store.py`
- `runtime/pubsub.py`
- `macro/context_engine.py`
- `templates/crypto.html`
- `services/scanner_service.py`
- `services/monitor_service.py`
- `services/news_router_service.py`
- `services/alert_service.py`

### Rebuilt / heavily modified
- `monitor/live_monitor.py`
- `app.py`
- `templates/index.html`

## Validation performed

Syntax validation completed successfully for the main changed Python modules:
- `app.py`
- `monitor/live_monitor.py`
- `macro/context_engine.py`
- `runtime/state_store.py`
- `runtime/pubsub.py`
- `services/*.py`

## Important operational note

This deliverable is wired to your real Alpaca provider and stream interfaces. Live validation of streams, SIP freshness, and news promotion still depends on:
- your Alpaca credentials
- your market-data entitlement
- market session / symbol liquidity
- live provider availability

No placeholder data was introduced.
