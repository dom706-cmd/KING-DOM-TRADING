
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
                """
            )
            self._ensure_column(conn, "news_events", "url", "TEXT")
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
