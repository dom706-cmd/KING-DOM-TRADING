
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_REPLAY_SNAPSHOTS_PER_MONITOR = 50
_CONTEXT_SNAPSHOTS_MAX = 500

class RuntimeStateStore:
    """SQLite-backed runtime store for monitor sessions, alerts, news, and context snapshots.

    Rules:
    - Uses real runtime payloads only.
    - Records real failures rather than fabricating state.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS watch_sessions (
                    monitor_id TEXT PRIMARY KEY,
                    source_job_id TEXT,
                    source TEXT,
                    mode TEXT,
                    started_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    running INTEGER NOT NULL,
                    feed_requested TEXT,
                    feed_used TEXT,
                    summary_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS watch_symbols (
                    monitor_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    playbook TEXT,
                    seed_source TEXT,
                    current_state TEXT,
                    previous_state TEXT,
                    last_transition_ts REAL,
                    cooldown_until_ts REAL,
                    updated_at REAL NOT NULL,
                    live_score REAL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (monitor_id, symbol),
                    FOREIGN KEY (monitor_id) REFERENCES watch_sessions(monitor_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS alert_events (
                    event_id TEXT PRIMARY KEY,
                    monitor_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    playbook TEXT,
                    from_state TEXT,
                    to_state TEXT,
                    event_type TEXT NOT NULL,
                    event_ts REAL NOT NULL,
                    price REAL,
                    bid REAL,
                    ask REAL,
                    spread_pct REAL,
                    vwap_delta_pct REAL,
                    live_chase_r REAL,
                    catalyst_score REAL,
                    context_score REAL,
                    dedupe_key TEXT,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (monitor_id) REFERENCES watch_sessions(monitor_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_alert_events_monitor_ts ON alert_events (monitor_id, event_ts DESC);

                CREATE TABLE IF NOT EXISTS news_events (
                    news_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    source TEXT,
                    url TEXT,
                    published_at TEXT,
                    received_at REAL NOT NULL,
                    sentiment_score REAL,
                    catalyst_score REAL,
                    freshness_sec REAL,
                    tags_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_news_symbol_received ON news_events (symbol, received_at DESC);

                CREATE TABLE IF NOT EXISTS context_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS saved_scan_presets (
                    name TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    favorite INTEGER NOT NULL DEFAULT 0,
                    notes TEXT,
                    category TEXT,
                    last_used_at REAL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS replay_sessions (
                    replay_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    monitor_id TEXT NOT NULL,
                    captured_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS desk_watchlist (
                    symbol TEXT PRIMARY KEY,
                    side TEXT,
                    trigger_price REAL,
                    stop_price REAL,
                    target_price REAL,
                    notes TEXT,
                    session_date TEXT,
                    added_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS gap_plan_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    saved_at REAL NOT NULL,
                    session_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    entry REAL,
                    stop REAL,
                    target_2r REAL,
                    gap_pct REAL,
                    pm_high REAL,
                    pm_low REAL,
                    notes TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_gap_plan_log_date ON gap_plan_log (session_date DESC, symbol);

                CREATE TABLE IF NOT EXISTS plan_snapshots (
                    snapshot_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date      TEXT NOT NULL,
                    symbol            TEXT NOT NULL,
                    monitor_id        TEXT,
                    plan_side         TEXT,
                    plan_entry        REAL,
                    plan_stop         REAL,
                    plan_target       REAL,
                    plan_risk_per_share REAL,
                    plan_risk_pct     REAL,
                    snapshot_ts       REAL NOT NULL,
                    snapshot_trigger  TEXT,
                    current_price     REAL,
                    entry_distance_r  REAL,
                    entry_distance_pct REAL,
                    spread_pct        REAL,
                    spread_vs_risk_pct REAL,
                    tape_live         INTEGER,
                    above_vwap        INTEGER,
                    vwap_delta_pct    REAL,
                    time_bucket       TEXT,
                    catalyst_score    REAL,
                    catalyst_freshness_hours REAL,
                    context_score     REAL,
                    ml_score          REAL,
                    combined_score    REAL,
                    plan_readiness_score REAL,
                    outcome           TEXT,
                    outcome_ts        REAL,
                    outcome_price     REAL,
                    outcome_labeled_at REAL,
                    payload_json      TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_plan_snapshots_sym_date
                    ON plan_snapshots (symbol, session_date, snapshot_ts DESC);
                CREATE INDEX IF NOT EXISTS idx_plan_snapshots_outcome
                    ON plan_snapshots (outcome, session_date DESC);

                CREATE TABLE IF NOT EXISTS positions (
                    position_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol        TEXT NOT NULL,
                    side          TEXT NOT NULL DEFAULT 'long',
                    entry_price   REAL NOT NULL,
                    stop_price    REAL NOT NULL,
                    target_price  REAL,
                    shares        REAL,
                    notes         TEXT,
                    entry_type    TEXT DEFAULT 'limit',
                    session_date  TEXT NOT NULL,
                    entry_ts      REAL,
                    status        TEXT NOT NULL DEFAULT 'open',
                    exit_price    REAL,
                    exit_ts       REAL,
                    exit_reason   TEXT,
                    realized_pnl  REAL,
                    realized_r    REAL,
                    outcome       TEXT,
                    added_at      REAL NOT NULL,
                    updated_at    REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_positions_status
                    ON positions (status, added_at DESC);
                CREATE INDEX IF NOT EXISTS idx_positions_symbol
                    ON positions (symbol, session_date DESC);

                CREATE TABLE IF NOT EXISTS scan_jobs (
                    job_id       TEXT PRIMARY KEY,
                    created_at   REAL NOT NULL,
                    scan_date    TEXT,
                    candidates_json TEXT NOT NULL,
                    symbols_json    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_scan_jobs_created
                    ON scan_jobs (created_at DESC);

                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    outcome_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol         TEXT NOT NULL,
                    strategy       TEXT NOT NULL,
                    direction      TEXT NOT NULL,
                    entry          REAL NOT NULL,
                    stop           REAL NOT NULL,
                    target_1r      REAL NOT NULL,
                    target_2r      REAL NOT NULL,
                    target_3r      REAL NOT NULL,
                    risk_per_share REAL NOT NULL,
                    session_date   TEXT NOT NULL,
                    recorded_at    REAL NOT NULL,
                    scan_ts        TEXT,
                    ml_score       REAL,
                    combined_score REAL,
                    notes          TEXT,
                    outcome        TEXT,
                    outcome_r      REAL,
                    outcome_price  REAL,
                    resolved_at    REAL,
                    runner_stop    REAL,
                    shakeout_watch INTEGER NOT NULL DEFAULT 0,
                    shakeout_flush_low REAL
                );
                CREATE INDEX IF NOT EXISTS idx_trade_outcomes_sym_date
                    ON trade_outcomes (symbol, session_date DESC, recorded_at DESC);
                CREATE INDEX IF NOT EXISTS idx_trade_outcomes_open
                    ON trade_outcomes (outcome, session_date DESC);

                CREATE TABLE IF NOT EXISTS float_cache (
                    symbol             TEXT PRIMARY KEY,
                    float_shares       REAL,
                    shares_outstanding REAL,
                    fetched_at         REAL NOT NULL
                );
                """
            )
            self._ensure_column(conn, "news_events", "url", "TEXT")
            self._ensure_column(conn, "trade_outcomes", "runner_stop", "REAL")
            self._ensure_column(conn, "trade_outcomes", "shakeout_watch", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "trade_outcomes", "shakeout_flush_low", "REAL")
            self._ensure_column(conn, "saved_scan_presets", "favorite", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "saved_scan_presets", "notes", "TEXT")
            self._ensure_column(conn, "saved_scan_presets", "category", "TEXT")
            self._ensure_column(conn, "saved_scan_presets", "last_used_at", "REAL")
            conn.commit()

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        cols = {str(r["name"]) for r in rows}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def scan_job_save(self, job_id: str, result: dict[str, Any]) -> None:
        """Persist a completed scan job's candidates and seed symbols to SQLite."""
        candidates = result.get("seed_candidates") or result.get("candidates") or []
        symbols = result.get("seed_symbols") or result.get("slice_symbols") or []
        scan_date = str(result.get("scan_date") or result.get("session_date_used") or "")
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scan_jobs
                    (job_id, created_at, scan_date, candidates_json, symbols_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(job_id),
                    time.time(),
                    scan_date,
                    json.dumps(candidates),
                    json.dumps(symbols),
                ),
            )
            # Prune jobs older than 3 days to keep the DB lean
            cutoff = time.time() - 86400 * 3
            conn.execute("DELETE FROM scan_jobs WHERE created_at < ?", (cutoff,))
            conn.commit()

    def scan_job_get(self, job_id: str) -> dict[str, Any] | None:
        """Load a previously persisted scan job. Returns None if not found."""
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT candidates_json, symbols_json, scan_date FROM scan_jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        if row is None:
            return None
        try:
            candidates = json.loads(row["candidates_json"] or "[]")
        except Exception:
            candidates = []
        try:
            symbols = json.loads(row["symbols_json"] or "[]")
        except Exception:
            symbols = []
        return {
            "status": "done",
            "result": {
                "seed_candidates": candidates,
                "seed_symbols": symbols,
                "scan_date": row["scan_date"],
            },
        }

    def scan_jobs_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent persisted scan jobs, newest first."""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT job_id, candidates_json, symbols_json, scan_date FROM scan_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for row in rows:
            try:
                candidates = json.loads(row["candidates_json"] or "[]")
            except Exception:
                candidates = []
            try:
                symbols = json.loads(row["symbols_json"] or "[]")
            except Exception:
                symbols = []
            out.append({
                "job_id": row["job_id"],
                "status": "done",
                "result": {
                    "seed_candidates": candidates,
                    "seed_symbols": symbols,
                    "scan_date": row["scan_date"],
                },
            })
        return out

    def save_session(self, session_payload: dict[str, Any]) -> None:
        payload = dict(session_payload or {})
        monitor_id = str(payload.get("monitor_id") or "").strip()
        if not monitor_id:
            raise ValueError("save_session.monitor_id required")
        now = float(payload.get("updated_at") or time.time())
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO watch_sessions (
                    monitor_id, source_job_id, source, mode, started_at, updated_at,
                    running, feed_requested, feed_used, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(monitor_id) DO UPDATE SET
                    source_job_id=excluded.source_job_id,
                    source=excluded.source,
                    mode=excluded.mode,
                    started_at=excluded.started_at,
                    updated_at=excluded.updated_at,
                    running=excluded.running,
                    feed_requested=excluded.feed_requested,
                    feed_used=excluded.feed_used,
                    summary_json=excluded.summary_json
                """,
                (
                    monitor_id,
                    payload.get("job_id"),
                    payload.get("source"),
                    payload.get("mode"),
                    float(payload.get("started_at") or now),
                    now,
                    1 if payload.get("running", True) else 0,
                    payload.get("feed_requested"),
                    payload.get("feed_used"),
                    json.dumps(payload, separators=(",", ":"), default=str),
                ),
            )
            conn.commit()

    def save_symbol_state(self, monitor_id: str, symbol_payload: dict[str, Any]) -> None:
        payload = dict(symbol_payload or {})
        symbol = str(payload.get("symbol") or "").strip().upper()
        if not monitor_id or not symbol:
            raise ValueError("save_symbol_state monitor_id/symbol required")
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO watch_symbols (
                    monitor_id, symbol, playbook, seed_source, current_state, previous_state,
                    last_transition_ts, cooldown_until_ts, updated_at, live_score, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(monitor_id, symbol) DO UPDATE SET
                    playbook=excluded.playbook,
                    seed_source=excluded.seed_source,
                    current_state=excluded.current_state,
                    previous_state=excluded.previous_state,
                    last_transition_ts=excluded.last_transition_ts,
                    cooldown_until_ts=excluded.cooldown_until_ts,
                    updated_at=excluded.updated_at,
                    live_score=excluded.live_score,
                    payload_json=excluded.payload_json
                """,
                (
                    monitor_id,
                    symbol,
                    payload.get("playbook"),
                    payload.get("seed_source"),
                    payload.get("monitor_state"),
                    payload.get("previous_state"),
                    payload.get("last_transition_ts"),
                    payload.get("cooldown_until_ts"),
                    float(payload.get("updated_at") or time.time()),
                    payload.get("live_score"),
                    json.dumps(payload, separators=(",", ":"), default=str),
                ),
            )
            conn.commit()

    def append_alert(self, alert_payload: dict[str, Any]) -> None:
        payload = dict(alert_payload or {})
        event_id = str(payload.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("append_alert.event_id required")
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alert_events (
                    event_id, monitor_id, symbol, playbook, from_state, to_state, event_type,
                    event_ts, price, bid, ask, spread_pct, vwap_delta_pct, live_chase_r,
                    catalyst_score, context_score, dedupe_key, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    payload.get("monitor_id"),
                    payload.get("symbol"),
                    payload.get("playbook"),
                    payload.get("from_state"),
                    payload.get("to_state"),
                    payload.get("event_type"),
                    float(payload.get("event_ts") or time.time()),
                    payload.get("price"),
                    payload.get("bid"),
                    payload.get("ask"),
                    payload.get("spread_pct"),
                    payload.get("vwap_delta_pct"),
                    payload.get("live_chase_r"),
                    payload.get("catalyst_score"),
                    payload.get("context_score"),
                    payload.get("dedupe_key"),
                    json.dumps(payload, separators=(",", ":"), default=str),
                ),
            )
            conn.commit()

    def append_news_event(self, payload: dict[str, Any]) -> None:
        row = dict(payload or {})
        news_id = str(row.get("news_id") or "").strip()
        if not news_id:
            raise ValueError("append_news_event.news_id required")
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO news_events (
                    news_id, symbol, headline, source, published_at, received_at,
                    url, sentiment_score, catalyst_score, freshness_sec, tags_json, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    news_id,
                    str(row.get("symbol") or "").strip().upper(),
                    str(row.get("headline") or "").strip(),
                    row.get("source"),
                    row.get("published_at"),
                    float(row.get("received_at") or time.time()),
                    row.get("url"),
                    row.get("sentiment_score"),
                    row.get("catalyst_score"),
                    row.get("freshness_sec"),
                    json.dumps(row.get("tags") or [], separators=(",", ":"), default=str),
                    json.dumps(row, separators=(",", ":"), default=str),
                ),
            )
            conn.commit()

    def append_context_snapshot(self, payload: dict[str, Any]) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO context_snapshots (created_at, payload_json) VALUES (?, ?)",
                (float(time.time()), json.dumps(payload or {}, separators=(",", ":"), default=str)),
            )
            conn.execute(
                """
                DELETE FROM context_snapshots
                WHERE snapshot_id NOT IN (
                    SELECT snapshot_id FROM context_snapshots ORDER BY snapshot_id DESC LIMIT ?
                )
                """,
                (_CONTEXT_SNAPSHOTS_MAX,),
            )
            conn.commit()

    def save_replay_snapshot(self, monitor_id: str, payload: dict[str, Any]) -> None:
        raw = dict(payload or {})
        compact = {
            "monitor_id": monitor_id,
            "captured_at": float(time.time()),
            "started_at": raw.get("started_at"),
            "updated_at": raw.get("updated_at"),
            "running": raw.get("running"),
            "source": raw.get("source"),
            "mode": raw.get("mode"),
            "job_id": raw.get("job_id"),
            "feed_requested": raw.get("feed_requested"),
            "feed_used": raw.get("feed_used"),
            "refresh_count": raw.get("refresh_count"),
            "refresh_error_count": raw.get("refresh_error_count"),
            "symbol_count": len(raw.get("symbols") or []),
            "alert_count": len(raw.get("alerts_recent") or []),
            "symbols": [
                {
                    "symbol": item.get("symbol"),
                    "monitor_state": item.get("monitor_state"),
                    "previous_state": item.get("previous_state"),
                    "playbook": item.get("playbook"),
                    "updated_at": item.get("updated_at"),
                    "live_score": item.get("live_score"),
                    "last_price": item.get("last_price"),
                    "spread_pct": item.get("spread_pct"),
                    "live_chase_r": item.get("live_chase_r"),
                    "grade": item.get("grade"),
                    "action_flag": item.get("action_flag"),
                    "last_error": item.get("last_error"),
                }
                for item in (raw.get("symbols") or [])[:50]
                if isinstance(item, dict)
            ],
            "alerts_recent": [
                {
                    "event_id": item.get("event_id"),
                    "symbol": item.get("symbol"),
                    "event_type": item.get("event_type"),
                    "from_state": item.get("from_state"),
                    "to_state": item.get("to_state"),
                    "event_ts": item.get("event_ts"),
                    "price": item.get("price"),
                }
                for item in (raw.get("alerts_recent") or [])[:20]
                if isinstance(item, dict)
            ],
        }
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO replay_sessions (monitor_id, captured_at, payload_json) VALUES (?, ?, ?)",
                (monitor_id, compact["captured_at"], json.dumps(compact, separators=(",", ":"), default=str)),
            )
            conn.execute(
                """
                DELETE FROM replay_sessions
                WHERE monitor_id = ?
                  AND replay_id NOT IN (
                      SELECT replay_id FROM replay_sessions
                      WHERE monitor_id = ?
                      ORDER BY replay_id DESC
                      LIMIT ?
                  )
                """,
                (monitor_id, monitor_id, _REPLAY_SNAPSHOTS_PER_MONITOR),
            )
            conn.commit()

    def recent_scan_presets(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                """
                SELECT name, created_at, updated_at, favorite, notes, category, last_used_at, payload_json
                FROM saved_scan_presets
                ORDER BY favorite DESC, COALESCE(last_used_at, 0) DESC, updated_at DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                payload = json.loads(r["payload_json"])
            except Exception:
                payload = {}
            out.append({
                "name": r["name"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "favorite": bool(r["favorite"]),
                "notes": r["notes"],
                "category": r["category"],
                "last_used_at": r["last_used_at"],
                "payload": payload,
            })
        return out

    def save_scan_preset(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        favorite: bool = False,
        notes: str | None = None,
        category: str | None = None,
        last_used_at: float | None = None,
    ) -> None:
        nm = str(name or '').strip()
        if not nm:
            raise ValueError('save_scan_preset.name required')
        now = float(time.time())
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO saved_scan_presets (name, created_at, updated_at, favorite, notes, category, last_used_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    favorite=excluded.favorite,
                    notes=excluded.notes,
                    category=excluded.category,
                    last_used_at=COALESCE(excluded.last_used_at, saved_scan_presets.last_used_at),
                    payload_json=excluded.payload_json
                """,
                (
                    nm,
                    now,
                    now,
                    1 if favorite else 0,
                    (str(notes).strip() if notes is not None else None),
                    (str(category).strip() if category is not None else None),
                    float(last_used_at) if last_used_at is not None else None,
                    json.dumps(payload or {}, separators=(",", ":"), default=str),
                ),
            )
            conn.commit()

    def touch_scan_preset(self, name: str, *, last_used_at: float | None = None) -> bool:
        nm = str(name or "").strip()
        if not nm:
            return False
        used_ts = float(last_used_at or time.time())
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE saved_scan_presets SET last_used_at=?, updated_at=? WHERE name=?",
                (used_ts, used_ts, nm),
            )
            conn.commit()
            return int(cur.rowcount or 0) > 0

    def delete_scan_preset(self, name: str) -> bool:
        nm = str(name or '').strip()
        if not nm:
            return False
        with self._lock, self._conn() as conn:
            cur = conn.execute("DELETE FROM saved_scan_presets WHERE name=?", (nm,))
            conn.commit()
            return int(cur.rowcount or 0) > 0

    def recent_news(self, *, symbol: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        q = "SELECT payload_json FROM news_events"
        args: list[Any] = []
        if symbol:
            q += " WHERE symbol=?"
            args.append(str(symbol).strip().upper())
        q += " ORDER BY received_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self._lock, self._conn() as conn:
            rows = conn.execute(q, args).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                out.append(json.loads(r["payload_json"]))
            except Exception:
                continue
        return out

    def recent_alerts(self, *, monitor_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        q = "SELECT payload_json FROM alert_events"
        args: list[Any] = []
        if monitor_id:
            q += " WHERE monitor_id=?"
            args.append(monitor_id)
        q += " ORDER BY event_ts DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self._lock, self._conn() as conn:
            rows = conn.execute(q, args).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                out.append(json.loads(r["payload_json"]))
            except Exception:
                continue
        return out

    def latest_context(self) -> dict[str, Any] | None:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT payload_json FROM context_snapshots ORDER BY snapshot_id DESC LIMIT 1").fetchone()
        if not row:
            return None
        return json.loads(row["payload_json"])

    # ------------------------------------------------------------------
    # Desk watchlist (pre-market, server-backed, overnight-persistent)
    # ------------------------------------------------------------------

    def desk_watchlist_set(self, symbol: str, *, side: str | None = None,
                           trigger_price: float | None = None, stop_price: float | None = None,
                           target_price: float | None = None, notes: str | None = None,
                           session_date: str | None = None) -> None:
        sym = str(symbol or "").strip().upper()
        if not sym:
            raise ValueError("desk_watchlist_set: symbol required")
        now = float(time.time())
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO desk_watchlist (symbol, side, trigger_price, stop_price, target_price,
                    notes, session_date, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    side=COALESCE(excluded.side, desk_watchlist.side),
                    trigger_price=COALESCE(excluded.trigger_price, desk_watchlist.trigger_price),
                    stop_price=COALESCE(excluded.stop_price, desk_watchlist.stop_price),
                    target_price=COALESCE(excluded.target_price, desk_watchlist.target_price),
                    notes=COALESCE(excluded.notes, desk_watchlist.notes),
                    session_date=COALESCE(excluded.session_date, desk_watchlist.session_date),
                    updated_at=excluded.updated_at
                """,
                (
                    sym,
                    str(side).strip().lower() if side else None,
                    float(trigger_price) if trigger_price is not None else None,
                    float(stop_price) if stop_price is not None else None,
                    float(target_price) if target_price is not None else None,
                    str(notes).strip() if notes is not None else None,
                    str(session_date).strip() if session_date is not None else None,
                    now,
                    now,
                ),
            )
            conn.commit()

    def desk_watchlist_remove(self, symbol: str) -> bool:
        sym = str(symbol or "").strip().upper()
        if not sym:
            return False
        with self._lock, self._conn() as conn:
            cur = conn.execute("DELETE FROM desk_watchlist WHERE symbol=?", (sym,))
            conn.commit()
            return int(cur.rowcount or 0) > 0

    def desk_watchlist_all(self) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT symbol, side, trigger_price, stop_price, target_price, notes, session_date, added_at, updated_at "
                "FROM desk_watchlist ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def desk_watchlist_symbols(self) -> set[str]:
        with self._lock, self._conn() as conn:
            rows = conn.execute("SELECT symbol FROM desk_watchlist").fetchall()
        return {str(r["symbol"]) for r in rows}

    def desk_watchlist_clear(self) -> int:
        """Remove every symbol from the desk watchlist. Returns count deleted."""
        with self._lock, self._conn() as conn:
            cur = conn.execute("DELETE FROM desk_watchlist")
            conn.commit()
            return int(cur.rowcount or 0)

    def desk_watchlist_purge_stale(self, today: str) -> list[str]:
        """Remove entries from previous sessions (session_date set and < today).

        Leaves entries with no session_date alone — those are manually managed
        or freshly added without a date. Returns list of purged symbols.
        """
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT symbol FROM desk_watchlist "
                "WHERE session_date IS NOT NULL AND session_date < ?",
                (today,),
            ).fetchall()
            syms = [str(r["symbol"]) for r in rows]
            if syms:
                conn.execute(
                    "DELETE FROM desk_watchlist "
                    "WHERE session_date IS NOT NULL AND session_date < ?",
                    (today,),
                )
                conn.commit()
        return syms

    def desk_watchlist_purge_aged(self, max_age_days: float) -> list[str]:
        """Remove entries older than max_age_days based on added_at, regardless
        of session_date.

        Catches manually-added entries (session_date IS NULL) that were left
        behind for weeks — desk_watchlist_purge_stale deliberately ignores those.
        Returns list of purged symbols.
        """
        cutoff = float(time.time()) - float(max_age_days) * 86400.0
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT symbol FROM desk_watchlist WHERE added_at < ?",
                (cutoff,),
            ).fetchall()
            syms = [str(r["symbol"]) for r in rows]
            if syms:
                conn.execute(
                    "DELETE FROM desk_watchlist WHERE added_at < ?",
                    (cutoff,),
                )
                conn.commit()
        return syms

    def desk_watchlist_update_side(self, symbol: str, side: str) -> bool:
        """Update only the side column for an existing watchlist entry.

        Used by the gap-order generator to sync the watchlist direction to the
        current pre-market gap direction so EOD-saved entries don't contradict
        live gap plans. Does nothing if the symbol is not in the watchlist.
        Returns True if a row was updated.
        """
        sym = str(symbol or "").strip().upper()
        side = str(side or "").strip().lower()
        if not sym or side not in ('long', 'short'):
            return False
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE desk_watchlist SET side=?, updated_at=? WHERE symbol=?",
                (side, float(time.time()), sym),
            )
            conn.commit()
            return int(cur.rowcount or 0) > 0

    # ------------------------------------------------------------------
    # Gap plan log — immutable record of every saved gap order plan
    # ------------------------------------------------------------------

    def gap_plan_log_add(self, *, session_date: str, symbol: str, side: str | None = None,
                         entry: float | None = None, stop: float | None = None,
                         target_2r: float | None = None, gap_pct: float | None = None,
                         pm_high: float | None = None, pm_low: float | None = None,
                         notes: str | None = None) -> None:
        now = float(time.time())
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO gap_plan_log
                   (saved_at, session_date, symbol, side, entry, stop, target_2r, gap_pct, pm_high, pm_low, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, str(session_date), str(symbol).upper(), side,
                 entry, stop, target_2r, gap_pct, pm_high, pm_low, notes),
            )

    def gap_plan_log_recent(self, session_date: str | None = None, limit: int = 100) -> list[dict]:
        with self._lock, self._conn() as conn:
            if session_date:
                rows = conn.execute(
                    "SELECT * FROM gap_plan_log WHERE session_date=? ORDER BY saved_at DESC LIMIT ?",
                    (session_date, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM gap_plan_log ORDER BY saved_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Positions — persistent trade journal + ML training source
    # ------------------------------------------------------------------

    def positions_save(self, data: dict[str, Any]) -> int:
        """Insert or update a position. Returns position_id."""
        from datetime import datetime, timezone
        now = float(time.time())
        session_date = str(data.get("session_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        position_id = data.get("position_id")
        sym = str(data.get("symbol") or "").strip().upper()
        if not sym:
            raise ValueError("positions_save: symbol required")
        entry = float(data["entry_price"]) if data.get("entry_price") is not None else None
        stop  = float(data["stop_price"])  if data.get("stop_price")  is not None else None
        if entry is None or stop is None:
            raise ValueError("positions_save: entry_price and stop_price required")
        with self._lock, self._conn() as conn:
            if position_id:
                conn.execute(
                    """UPDATE positions SET symbol=?, side=?, entry_price=?, stop_price=?,
                       target_price=?, shares=?, notes=?, entry_type=?, session_date=?,
                       entry_ts=?, updated_at=? WHERE position_id=?""",
                    (sym, str(data.get("side") or "long").lower(),
                     entry, stop,
                     float(data["target_price"]) if data.get("target_price") is not None else None,
                     float(data["shares"]) if data.get("shares") is not None else None,
                     str(data["notes"]).strip() if data.get("notes") else None,
                     str(data.get("entry_type") or "limit"),
                     session_date,
                     float(data["entry_ts"]) if data.get("entry_ts") is not None else None,
                     now, int(position_id))
                )
                conn.commit()
                return int(position_id)
            else:
                cur = conn.execute(
                    """INSERT INTO positions (symbol, side, entry_price, stop_price,
                       target_price, shares, notes, entry_type, session_date, entry_ts,
                       status, added_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,'open',?,?)""",
                    (sym, str(data.get("side") or "long").lower(),
                     entry, stop,
                     float(data["target_price"]) if data.get("target_price") is not None else None,
                     float(data["shares"]) if data.get("shares") is not None else None,
                     str(data["notes"]).strip() if data.get("notes") else None,
                     str(data.get("entry_type") or "limit"),
                     session_date,
                     float(data["entry_ts"]) if data.get("entry_ts") is not None else None,
                     now, now)
                )
                conn.commit()
                return int(cur.lastrowid)

    def positions_close(self, position_id: int, *, exit_price: float, exit_reason: str) -> dict[str, Any]:
        """Close a position. Computes realized_r and outcome. Returns the updated row."""
        now = float(time.time())
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE position_id=?", (int(position_id),)
            ).fetchone()
            if row is None:
                raise ValueError(f"positions_close: position_id {position_id} not found")
            row = dict(row)
            entry  = float(row["entry_price"])
            stop   = float(row["stop_price"])
            shares = float(row.get("shares") or 0.0)
            side   = str(row.get("side") or "long").lower()
            exit_p = float(exit_price)
            risk   = abs(entry - stop)
            if risk < 1e-6:
                risk = 1e-6
            if side == "long":
                realized_r   = (exit_p - entry) / risk
                realized_pnl = (exit_p - entry) * shares
            else:
                realized_r   = (entry - exit_p) / risk
                realized_pnl = (entry - exit_p) * shares
            # Outcome labeling
            er = str(exit_reason or "discretionary").lower()
            if er == "target_hit" or realized_r >= 1.8:
                outcome = "target_reached"
            elif er == "stopped_out" or realized_r <= -0.85:
                outcome = "stopped_out"
            else:
                outcome = "discretionary"
            conn.execute(
                """UPDATE positions SET status='closed', exit_price=?, exit_ts=?,
                   exit_reason=?, realized_pnl=?, realized_r=?, outcome=?, updated_at=?
                   WHERE position_id=?""",
                (exit_p, now, er, round(realized_pnl, 4), round(realized_r, 4),
                 outcome, now, int(position_id))
            )
            conn.commit()
        row.update({"exit_price": exit_p, "exit_ts": now, "exit_reason": er,
                    "realized_pnl": round(realized_pnl, 4), "realized_r": round(realized_r, 4),
                    "outcome": outcome, "status": "closed"})
        return row

    def positions_open(self) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status='open' ORDER BY added_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def positions_history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status='closed' ORDER BY exit_ts DESC LIMIT ?",
                (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    def positions_remove(self, position_id: int) -> bool:
        with self._lock, self._conn() as conn:
            cur = conn.execute("DELETE FROM positions WHERE position_id=?", (int(position_id),))
            conn.commit()
        return int(cur.rowcount or 0) > 0

    def positions_daily_summary(self, session_date: str | None = None) -> dict:
        """Return today's closed-trade summary: total R, total P&L, win/loss counts."""
        from datetime import datetime, timezone
        date = session_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT realized_pnl, realized_r, outcome FROM positions "
                "WHERE status='closed' AND session_date=?", (date,)
            ).fetchall()
        total_r   = sum(float(r["realized_r"]   or 0) for r in rows)
        total_pnl = sum(float(r["realized_pnl"] or 0) for r in rows)
        wins  = sum(1 for r in rows if r["outcome"] == "target_reached")
        losses = sum(1 for r in rows if r["outcome"] == "stopped_out")
        disc  = sum(1 for r in rows if r["outcome"] == "discretionary")
        return {
            "session_date": date,
            "trade_count": len(rows),
            "wins": wins,
            "losses": losses,
            "discretionary": disc,
            "total_r": round(total_r, 3),
            "total_pnl": round(total_pnl, 2),
        }

    # ------------------------------------------------------------------
    # Plan snapshots — training data for watchlist plan confidence model
    # ------------------------------------------------------------------

    def log_plan_snapshot(self, features: dict[str, Any], *, monitor_id: str | None = None) -> int:
        """Insert one plan snapshot. Returns the new snapshot_id."""
        now = float(time.time())
        payload = dict(features)
        breakdown = payload.pop("plan_readiness_breakdown", None)
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO plan_snapshots (
                    session_date, symbol, monitor_id,
                    plan_side, plan_entry, plan_stop, plan_target,
                    plan_risk_per_share, plan_risk_pct,
                    snapshot_ts, snapshot_trigger, current_price,
                    entry_distance_r, entry_distance_pct,
                    spread_pct, spread_vs_risk_pct,
                    tape_live, above_vwap, vwap_delta_pct,
                    time_bucket, catalyst_score, catalyst_freshness_hours,
                    context_score, ml_score, combined_score,
                    plan_readiness_score, payload_json
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                (
                    payload.get("session_date") or "",
                    str(payload.get("symbol") or "").upper(),
                    str(monitor_id or "") or None,
                    payload.get("plan_side"),
                    payload.get("plan_entry"),
                    payload.get("plan_stop"),
                    payload.get("plan_target"),
                    payload.get("plan_risk_per_share"),
                    payload.get("plan_risk_pct"),
                    now,
                    payload.get("snapshot_trigger"),
                    payload.get("current_price"),
                    payload.get("entry_distance_r"),
                    payload.get("entry_distance_pct"),
                    payload.get("spread_pct"),
                    payload.get("spread_vs_risk_pct"),
                    payload.get("tape_live"),
                    payload.get("above_vwap"),
                    payload.get("vwap_delta_pct"),
                    payload.get("time_bucket"),
                    payload.get("catalyst_score"),
                    payload.get("catalyst_freshness_hours"),
                    payload.get("context_score"),
                    payload.get("ml_score"),
                    payload.get("combined_score"),
                    payload.get("plan_readiness_score"),
                    json.dumps({**payload, "plan_readiness_breakdown": breakdown}),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def label_plan_outcome(
        self,
        *,
        monitor_id: str,
        symbol: str,
        session_date: str,
        outcome: str,
        price: float | None = None,
    ) -> int:
        """
        Label the most recent unlabeled snapshot for (monitor_id, symbol, session_date).
        outcome must be one of: 'target_reached', 'stopped_out', 'neither'.
        Returns number of rows updated.
        """
        now = float(time.time())
        sym = str(symbol or "").strip().upper()
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                """
                UPDATE plan_snapshots
                SET outcome = ?, outcome_ts = ?, outcome_price = ?, outcome_labeled_at = ?
                WHERE snapshot_id = (
                    SELECT snapshot_id FROM plan_snapshots
                    WHERE monitor_id = ? AND symbol = ? AND session_date = ? AND outcome IS NULL
                    ORDER BY snapshot_ts DESC
                    LIMIT 1
                )
                """,
                (outcome, now, price, now, str(monitor_id), sym, str(session_date)),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def plan_snapshots_for_training(self, *, limit: int = 10000, labeled_only: bool = True) -> list[dict[str, Any]]:
        """
        Fetch plan snapshots for ML training.
        labeled_only=True returns only rows where outcome is set (the training set).
        """
        limit = max(1, min(int(limit), 50000))
        with self._lock, self._conn() as conn:
            if labeled_only:
                rows = conn.execute(
                    "SELECT * FROM plan_snapshots WHERE outcome IS NOT NULL ORDER BY session_date DESC, snapshot_ts DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM plan_snapshots ORDER BY session_date DESC, snapshot_ts DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def replay_session(self, monitor_id: str, limit: int = 500) -> dict[str, Any]:
        with self._lock, self._conn() as conn:
            session_row = conn.execute("SELECT summary_json FROM watch_sessions WHERE monitor_id=?", (monitor_id,)).fetchone()
            symbol_rows = conn.execute(
                "SELECT payload_json FROM watch_symbols WHERE monitor_id=? ORDER BY symbol ASC LIMIT ?",
                (monitor_id, max(1, min(int(limit), 1000))),
            ).fetchall()
        session = json.loads(session_row["summary_json"]) if session_row else None
        symbols = [json.loads(r["payload_json"]) for r in symbol_rows]
        alerts = self.recent_alerts(monitor_id=monitor_id, limit=limit)
        return {"session": session, "symbols": symbols, "alerts": alerts}

    # ------------------------------------------------------------------
    # R-Multiple Outcome Tracker
    # ------------------------------------------------------------------

    def record_trade_outcome(self, data: dict[str, Any]) -> int:
        """Insert a trade setup to track for R-multiple resolution.  Returns outcome_id."""
        now = float(time.time())
        sym = str(data.get("symbol") or "").strip().upper()
        if not sym:
            raise ValueError("symbol required")
        direction = str(data.get("direction") or data.get("best_side") or data.get("side") or "long").lower()
        entry     = float(data["entry"])
        stop_val  = float(data["stop"])
        risk      = abs(entry - stop_val)
        t1r = entry + risk if direction == "long" else entry - risk
        t2r = float(data.get("target_2r") or (entry + 2 * risk if direction == "long" else entry - 2 * risk))
        t3r = float(data.get("target_3r") or (entry + 3 * risk if direction == "long" else entry - 3 * risk))
        runner_stop_val = data.get("runner_stop")
        runner_stop_val = float(runner_stop_val) if runner_stop_val is not None else None
        session_date_str = str(data.get("session_date") or data.get("scan_date") or "")
        with self._lock, self._conn() as conn:
            # Dedup: same symbol + strategy + direction + session_date already tracked?
            strategy_str = str(data.get("strategy") or "unknown")
            existing = conn.execute(
                "SELECT outcome_id FROM trade_outcomes WHERE symbol=? AND session_date=? AND strategy=? AND direction=? LIMIT 1",
                (sym, session_date_str, strategy_str, direction),
            ).fetchone()
            if existing:
                return int(existing["outcome_id"])
            cur = conn.execute(
                """
                INSERT INTO trade_outcomes
                    (symbol, strategy, direction, entry, stop, target_1r, target_2r, target_3r,
                     risk_per_share, session_date, recorded_at, scan_ts, ml_score, combined_score, notes,
                     runner_stop)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    sym,
                    strategy_str,
                    direction,
                    entry,
                    stop_val,
                    t1r, t2r, t3r,
                    risk,
                    session_date_str,
                    now,
                    str(data.get("scan_ts") or ""),
                    data.get("ml_score"),
                    data.get("combined_score"),
                    str(data.get("notes") or ""),
                    runner_stop_val,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def get_trade_outcomes(
        self,
        *,
        session_date: str | None = None,
        open_only: bool = False,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 2000))
        with self._lock, self._conn() as conn:
            if open_only:
                rows = conn.execute(
                    "SELECT * FROM trade_outcomes WHERE outcome IS NULL ORDER BY recorded_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            elif session_date:
                rows = conn.execute(
                    "SELECT * FROM trade_outcomes WHERE session_date=? ORDER BY recorded_at DESC LIMIT ?",
                    (str(session_date), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trade_outcomes ORDER BY recorded_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_trade_outcome_stats(
        self,
        *,
        session_date: str | None = None,
    ) -> dict[str, Any]:
        """All-time (or per-session) aggregate stats over the FULL table, ignoring any display limit."""
        where = "WHERE session_date=?" if session_date else ""
        params: tuple[Any, ...] = (str(session_date),) if session_date else ()
        with self._lock, self._conn() as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*)                                                   AS total,
                    COALESCE(ROUND(SUM(COALESCE(outcome_r, 0)), 2), 0.0)        AS total_r,
                    SUM(CASE WHEN outcome LIKE 'hit_%' THEN 1 ELSE 0 END)       AS wins,
                    SUM(CASE WHEN outcome = 'stopped' THEN 1 ELSE 0 END)        AS stops,
                    SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END)            AS open_count
                FROM trade_outcomes {where}
                """,
                params,
            ).fetchone()
        d = dict(row) if row else {}
        wins = int(d.get("wins") or 0)
        stops = int(d.get("stops") or 0)
        return {
            "total": int(d.get("total") or 0),
            "total_r": round(float(d.get("total_r") or 0.0), 2),
            "wins": wins,
            "stops": stops,
            "open_count": int(d.get("open_count") or 0),
            "win_rate": round(wins / (wins + stops) * 100, 1) if (wins + stops) > 0 else None,
        }

    def export_training_data(self, *, strategy: str | None = None) -> list[dict[str, Any]]:
        """Resolved outcomes as ML training rows: label=1 for a win (hit_*), 0 for a stop.

        Only rows with a session_date and a win/stop outcome are returned (expired/manual
        are excluded — they aren't clean ground-truth for the ranker). Used by
        ml/retrain_from_outcomes.py, which re-computes features from historical market data.
        """
        sql = (
            "SELECT symbol, strategy, session_date, direction, outcome, outcome_r, "
            "ml_score, combined_score "
            "FROM trade_outcomes "
            "WHERE session_date IS NOT NULL AND session_date != '' "
            "AND (outcome LIKE 'hit_%' OR outcome = 'stopped') "
        )
        params: list[Any] = []
        if strategy:
            sql += "AND strategy = ? "
            params.append(str(strategy))
        sql += "ORDER BY session_date ASC"
        with self._lock, self._conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["label"] = 1 if str(d.get("outcome") or "").startswith("hit_") else 0
            out.append(d)
        return out

    def resolve_trade_outcome(
        self,
        outcome_id: int,
        *,
        outcome: str,
        price: float | None = None,
        outcome_r: float | None = None,
    ) -> int:
        """Mark a trade outcome.  outcome: 'hit_1r'|'hit_2r'|'hit_3r'|'stopped'|'expired'|'manual'."""
        now = float(time.time())
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE trade_outcomes SET outcome=?, outcome_price=?, outcome_r=?, resolved_at=? WHERE outcome_id=?",
                (str(outcome), price, outcome_r, now, int(outcome_id)),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def auto_resolve_trade_outcomes(self, price_map: dict[str, float]) -> list[int]:
        """Check open outcomes against current prices and resolve any that have hit a target or stop.
        price_map: {SYMBOL: current_price}.  Returns list of resolved outcome_ids."""
        resolved_ids: list[int] = []
        open_rows = self.get_trade_outcomes(open_only=True, limit=500)
        for row in open_rows:
            sym   = str(row["symbol"]).upper()
            price = price_map.get(sym)
            if price is None:
                continue
            direction = str(row["direction"] or "long").lower()
            entry  = float(row["entry"])
            stop   = float(row["stop"])
            t1r    = float(row["target_1r"])
            t2r    = float(row["target_2r"])
            t3r    = float(row["target_3r"])
            risk   = float(row["risk_per_share"]) or abs(entry - stop)
            oid    = int(row["outcome_id"])

            outcome_str:  str   | None = None
            outcome_r_val: float | None = None

            if direction == "long":
                if price <= stop:
                    outcome_str   = "stopped"
                    outcome_r_val = -1.0
                elif price >= t3r:
                    outcome_str   = "hit_3r"
                    outcome_r_val = round((price - entry) / risk, 2) if risk > 0 else None
                elif price >= t2r:
                    outcome_str   = "hit_2r"
                    outcome_r_val = round((price - entry) / risk, 2) if risk > 0 else None
                elif price >= t1r:
                    outcome_str   = "hit_1r"
                    outcome_r_val = round((price - entry) / risk, 2) if risk > 0 else None
            else:  # short
                if price >= stop:
                    outcome_str   = "stopped"
                    outcome_r_val = -1.0
                elif price <= t3r:
                    outcome_str   = "hit_3r"
                    outcome_r_val = round((entry - price) / risk, 2) if risk > 0 else None
                elif price <= t2r:
                    outcome_str   = "hit_2r"
                    outcome_r_val = round((entry - price) / risk, 2) if risk > 0 else None
                elif price <= t1r:
                    outcome_str   = "hit_1r"
                    outcome_r_val = round((entry - price) / risk, 2) if risk > 0 else None

            if outcome_str:
                self.resolve_trade_outcome(oid, outcome=outcome_str, price=price, outcome_r=outcome_r_val)
                resolved_ids.append(oid)
        return resolved_ids

    def set_runner_stop(self, outcome_id: int, runner_stop: float) -> int:
        """Set the runner-shares stop price on an open outcome."""
        runner_stop = float(runner_stop)
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT direction, stop FROM trade_outcomes WHERE outcome_id=?",
                (int(outcome_id),),
            ).fetchone()
            if row:
                direction = str(row["direction"] or "long").lower()
                main_stop = float(row["stop"] or 0)
                # Runner stop must be on the profitable side of the main stop
                if direction == "long" and runner_stop < main_stop:
                    raise ValueError(
                        f"Runner stop {runner_stop} is below main stop {main_stop} for a long position"
                    )
                if direction == "short" and runner_stop > main_stop:
                    raise ValueError(
                        f"Runner stop {runner_stop} is above main stop {main_stop} for a short position"
                    )
            cur = conn.execute(
                "UPDATE trade_outcomes SET runner_stop=? WHERE outcome_id=?",
                (runner_stop, int(outcome_id)),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def set_shakeout_watch(self, outcome_id: int, flush_low: float | None = None) -> int:
        """Flag a stopped outcome as a shakeout watch (spring/bounce re-entry candidate)."""
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE trade_outcomes SET shakeout_watch=1, shakeout_flush_low=? WHERE outcome_id=?",
                (float(flush_low) if flush_low is not None else None, int(outcome_id)),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def get_shakeout_watches(self) -> list[dict[str, Any]]:
        """Return all outcomes flagged as shakeout watches (regardless of resolution status)."""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT outcome_id, symbol, direction, entry, stop, shakeout_flush_low, session_date, recorded_at "
                "FROM trade_outcomes WHERE shakeout_watch=1 ORDER BY recorded_at DESC LIMIT 100"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Float cache — persists yfinance float data with 24h TTL
    # ------------------------------------------------------------------

    _FLOAT_TTL_SEC = 14_400.0  # 4 hours — yfinance fast_info.shares can be days stale at source; refresh more often

    def get_cached_float(self, symbol: str) -> float | None:
        sym = str(symbol or "").strip().upper()
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT float_shares, fetched_at FROM float_cache WHERE symbol=?", (sym,)
            ).fetchone()
        if row is None:
            return None
        if (time.time() - float(row["fetched_at"])) > self._FLOAT_TTL_SEC:
            return None  # stale
        v = row["float_shares"]
        return float(v) if v is not None else None

    def save_float_cache(self, symbol: str, float_shares: float | None, shares_outstanding: float | None = None) -> None:
        sym = str(symbol or "").strip().upper()
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO float_cache (symbol, float_shares, shares_outstanding, fetched_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(symbol) DO UPDATE SET
                       float_shares=excluded.float_shares,
                       shares_outstanding=excluded.shares_outstanding,
                       fetched_at=excluded.fetched_at""",
                (sym, float_shares, shares_outstanding, time.time()),
            )
            conn.commit()
