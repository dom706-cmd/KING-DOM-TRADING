from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List
import hashlib
import json
import logging
from collections import deque
import platform
import os
import sys
import subprocess
import atexit
import fcntl
import threading
import concurrent.futures
import time
import uuid
import signal
import traceback
from flask import Flask, render_template, request, jsonify, redirect, Response, stream_with_context

log = logging.getLogger(__name__)


from werkzeug.exceptions import HTTPException

_PROJECT_ROOT = Path(__file__).resolve().parent


def _load_local_env_files() -> None:
    """Best-effort env bootstrap before provider init.

    Accept simple KEY=VALUE lines from common local env files if they exist.
    Existing process env wins; file values only backfill missing vars.
    """
    candidates = [
        _PROJECT_ROOT / ".orb_env",
        _PROJECT_ROOT / ".env",
        _PROJECT_ROOT / ".env.local",
        _PROJECT_ROOT / ".env.alpaca",
        _PROJECT_ROOT / ".flaskenv",
    ]
    for path in candidates:
        try:
            if not path.exists() or not path.is_file():
                continue
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if not k or k in os.environ:
                    continue
                if len(v) >= 2 and ((v[0] == v[-1]) and v[0] in {'"', "'"}):
                    v = v[1:-1]
                os.environ[k] = v
        except Exception:
            # Never fail startup because an optional env file is malformed.
            continue


def _alpaca_key() -> str | None:
    return (
        os.getenv("ALPACA_API_KEY")
        or os.getenv("APCA_API_KEY_ID")
        or os.getenv("ALPACA_KEY_ID")
    )


def _alpaca_secret() -> str | None:
    return (
        os.getenv("ALPACA_SECRET_KEY")
        or os.getenv("APCA_API_SECRET_KEY")
        or os.getenv("ALPACA_API_SECRET")
    )


_load_local_env_files()

BROKER_ACTIONS_ENABLED = (os.getenv("BROKER_ACTIONS_ENABLED") or "0").strip().lower() in {"1", "true", "yes", "on"}
_SINGLE_INSTANCE_LOCK = None


class SingleInstanceRunning(RuntimeError):
    pass


def _acquire_single_instance_lock() -> None:
    global _SINGLE_INSTANCE_LOCK
    if (os.getenv("ORB_SINGLE_INSTANCE") or "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    lock_path = os.getenv("ORB_SINGLE_INSTANCE_LOCK_PATH") or "/tmp/kingdom_app.lock"
    fh = open(lock_path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            fh.seek(0)
            holder = fh.read().strip()
        except Exception:
            holder = ""
        raise SingleInstanceRunning(f"another_kingdom_instance_running lock={lock_path} holder={holder or 'unknown'}")
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} port={os.getenv('ORB_PORT', '8050')} started_at={int(time.time())}\n")
    fh.flush()
    _SINGLE_INSTANCE_LOCK = fh

    def _cleanup_lock() -> None:
        global _SINGLE_INSTANCE_LOCK
        try:
            if _SINGLE_INSTANCE_LOCK is not None:
                fcntl.flock(_SINGLE_INSTANCE_LOCK.fileno(), fcntl.LOCK_UN)
                _SINGLE_INSTANCE_LOCK.close()
        except Exception:
            pass
        _SINGLE_INSTANCE_LOCK = None

    atexit.register(_cleanup_lock)


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# Flask 3.x custom JSON provider — converts NaN/Inf to null so the browser
# never receives invalid JSON (Python's default encoder emits bare NaN).
import math as _math
from flask.json.provider import DefaultJSONProvider as _DefaultJSONProvider

class _SafeJSONProvider(_DefaultJSONProvider):
    @staticmethod
    def default(o):
        if isinstance(o, float) and (_math.isnan(o) or _math.isinf(o)):
            return None
        return _DefaultJSONProvider.default(o)

    def dumps(self, obj, **kwargs):
        # ignore_nan not available in stdlib json; walk the object instead
        obj = _sanitize_nan(obj)
        return super().dumps(obj, **kwargs)

def _sanitize_nan(obj):
    if isinstance(obj, float):
        return None if (_math.isnan(obj) or _math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nan(v) for v in obj]
    return obj

app.json_provider_class = _SafeJSONProvider
app.json = _SafeJSONProvider(app)

# --- Build fingerprint so you can PROVE what code is running ---
def _build_id() -> str:
    try:
        p = os.path.abspath(__file__)
        st = os.stat(p)
        raw = f"{p}|{st.st_mtime}|{st.st_size}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]
    except Exception:
        return "unknown"

BUILD_ID = _build_id()

# If the process receives SIGUSR1 (common "dump stacks" signal), do NOT exit.
# Instead, print thread stack traces to stderr and continue.
def _sigusr1_dump_stacks(signum, frame):  # noqa: ARG001
    try:
        sys.stderr.write("\n\n=== SIGUSR1: thread stack dump ===\n")
        for t in threading.enumerate():
            sys.stderr.write(f"\n--- Thread: {t.name} (ident={t.ident}) ---\n")
            if t.ident:
                stack = traceback.format_stack(sys._current_frames().get(t.ident))
                sys.stderr.write("".join(stack))
        sys.stderr.write("\n=== end stack dump ===\n\n")
        sys.stderr.flush()
    except Exception:
        # Never crash on debugging signal.
        pass

try:
    signal.signal(signal.SIGUSR1, _sigusr1_dump_stacks)
except Exception:
    # Not available on some platforms.
    pass

@app.errorhandler(Exception)
def handle_api_errors(e):
    """Return JSON for /api/* endpoints; let Flask handle non-API errors normally."""
    try:
        path = request.path or ""
    except Exception:
        path = ""

    if isinstance(e, HTTPException):
        if path.startswith("/api/"):
            return jsonify(ok=False, error=e.description, code=e.code or 500), (e.code or 500)
        return e

    if path.startswith("/api/"):
        return jsonify(ok=False, error=str(e), code=500), 500

    # IMPORTANT: For non-API routes, do NOT return the exception object.
    # Returning an Exception instance causes Flask to raise:
    #   TypeError: The view function did not return a valid response...
    # Instead, re-raise so Flask's default error handling/debugger can
    # render the traceback and point to the real root cause.
    raise e
from universe.nasdaq_symbols import fetch_us_equity_symbols, UniverseConfig
from scanner.orb import scan_symbols, ORBConfig
from scanner.result_view import (
    build_primary_fallback_view,
    build_zero_result_diagnostics,
    candidate_sort_score,
    select_primary_candidates,
)
from core.errors import IntradayDataFailure, TrendContextFailure, EntryNowMLFailure, failure_string
from core.execution_plan import build_plan_state
from providers.alpaca_provider import AlpacaProvider
from providers.etrade_provider import ETradeProvider
from providers.symbols import to_provider_symbol
from providers.streaming import AlpacaStreamCache
from monitor.live_monitor import LiveMonitorManager
from runtime.state_store import RuntimeStateStore
from runtime.pubsub import OptionalRedisPublisher
from runtime.stream_market_data import recent_bars_df, latest_trade_payload, latest_quote_payload
from macro.context_engine import MarketContextEngine

import inspect
from providers.base import BarsRequest
from scanner.indicators import vwap, avg_daily_volume
from sentiment.multi import SentimentService
from macro.feeds import MacroFeedClient
from macro.topics import classify_items, score_relevance_for_symbol

from dataclasses import replace



# Resolve paths relative to this file so running from any CWD works.
BASE_DIR = Path(__file__).resolve().parent

# ---- Alpaca provider + streaming cache -----------------------------------
# --- HARD LOCK: Alpaca-only market data (tenant-safe, no silent fallback) ---
_mdp = (os.getenv("ORB_MARKET_DATA_PROVIDER") or "alpaca").strip().lower()
if _mdp != "alpaca":
    raise RuntimeError(f"ORB_MARKET_DATA_PROVIDER must be 'alpaca'. Got: {_mdp}")

_feed = (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower()
if _feed not in {"sip", "iex"}:
    raise RuntimeError(f"ALPACA_DATA_FEED must be 'sip' or 'iex'. Got: {_feed}")
# --------------------------------------------------------------------------


_ALPACA_PROVIDER = None
_PROVIDER_ERROR: str | None = None
try:
    _ALPACA_PROVIDER = AlpacaProvider()
except Exception as _provider_e:
    _PROVIDER_ERROR = f"{type(_provider_e).__name__}: {_provider_e}"

_BROKER_PROVIDER = _ALPACA_PROVIDER
_BROKER_PROVIDER_NAME = "alpaca"
_BROKER_PROVIDER_ERROR: str | None = _PROVIDER_ERROR
if (os.getenv("BROKER_PROVIDER") or "alpaca").strip().lower() == "etrade":
    try:
        _BROKER_PROVIDER = ETradeProvider()
        _BROKER_PROVIDER_NAME = "etrade"
        _BROKER_PROVIDER_ERROR = None
    except Exception as _broker_e:
        _BROKER_PROVIDER = None
        _BROKER_PROVIDER_NAME = "etrade"
        _BROKER_PROVIDER_ERROR = f"{type(_broker_e).__name__}: {_broker_e}"

_MONITOR = LiveMonitorManager()
_STREAM = None
_STREAM_ERROR: str | None = None

# --- HARD LOCK: streaming requirements (tenant-safe, fail-hard when required) ---
_STREAM_REQUIRED = (os.getenv("ALPACA_STREAM_REQUIRED") or "0").strip().lower() in ("1", "true", "yes", "on")
_STREAM_ENABLED = (os.getenv("ALPACA_STREAMING") or "1").strip().lower() in ("1", "true", "yes", "on")


def _wait_for_stream_boot(stream: AlpacaStreamCache, timeout_sec: float = 8.0) -> None:
    """
    Prove the websocket boot path is healthy enough after an explicit start.
    Scanner startup must not auto-start the stream.
    """
    deadline = time.time() + max(0.5, float(timeout_sec))
    last_state = None

    while time.time() < deadline:
        st = stream.state()
        last_state = st

        if getattr(st, "error", None):
            raise RuntimeError(f"stream_state_error: {st.error}")

        if bool(getattr(st, "symbols", [])):
            return

        time.sleep(0.05)

    if last_state is None:
        raise RuntimeError("stream_failed_to_report_state")
    raise RuntimeError(
        "stream_boot_timeout: stream did not register subscriptions within "
        f"{float(timeout_sec):.1f}s"
    )


def _stream_secret() -> str | None:
    return _alpaca_secret()


def _stream_key() -> str | None:
    return _alpaca_key()


def _stream_feed() -> str:
    return (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex"


def _ensure_stream(
    *,
    start: bool = False,
    require: bool = False,
    canary: str | None = None,
    wait_for_boot: bool = False,
    timeout_sec: float | None = None,
) -> AlpacaStreamCache | None:
    """Lazily construct/start the market-data websocket only for monitor/explicit stream probes."""
    global _STREAM, _STREAM_ERROR

    if not _STREAM_ENABLED:
        if require or _STREAM_REQUIRED:
            raise RuntimeError("stream_disabled")
        return None

    key = _stream_key()
    sec = _stream_secret()
    if not key or not sec:
        if require or _STREAM_REQUIRED:
            raise RuntimeError("stream_required_but_missing_creds_or_disabled")
        return None

    if _STREAM is None:
        try:
            _STREAM = AlpacaStreamCache(key, sec, feed=_stream_feed())
            _STREAM._market_event_hook = lambda: _MARKET_EVENTS_EVENT.set()
            _STREAM_ERROR = None
            _MONITOR.configure_runtime(
                provider=_ALPACA_PROVIDER,
                stream_cache=_STREAM,
                store=_RUNTIME_STORE,
                context_engine=_CONTEXT_ENGINE,
                pubsub=_PUBSUB,
            )
        except Exception as e:
            _STREAM = None
            _STREAM_ERROR = f"{type(e).__name__}: {e}"
            if require or _STREAM_REQUIRED:
                raise
            return None

    if start:
        try:
            _STREAM.start()
            boot_symbol = (str(canary or os.getenv("ALPACA_STREAM_BOOT_SYMBOL") or "AAPL").strip().upper() or "AAPL")
            if boot_symbol:
                _STREAM.ensure_symbols([boot_symbol])
            if wait_for_boot:
                _wait_for_stream_boot(
                    _STREAM,
                    timeout_sec=float(timeout_sec or os.getenv("ALPACA_STREAM_BOOT_TIMEOUT", "8")),
                )
            _STREAM_ERROR = None
        except Exception as e:
            _STREAM_ERROR = f"{type(e).__name__}: {e}"
            if require or _STREAM_REQUIRED:
                raise
            return None

    return _STREAM


_RUNTIME_DB_PATH = BASE_DIR / "runtime" / "runtime_state.db"
_RUNTIME_STORE = RuntimeStateStore(_RUNTIME_DB_PATH)

# ── Plan integrity startup check ───────────────────────────────────────────────
try:
    from core.plan_integrity import run_startup_integrity_check
    run_startup_integrity_check(_RUNTIME_STORE)
except Exception:
    pass

# Repopulate _JOBS from persisted scan jobs so they survive app restarts.
try:
    for _sj in _RUNTIME_STORE.scan_jobs_recent(limit=50):
        _jid = _sj.get("job_id")
        if _jid and _jid not in _JOBS:
            _JOBS[_jid] = {"status": "done", "result": _sj["result"], "updated_at": time.time()}
except Exception:
    pass

_CONTEXT_ENGINE = MarketContextEngine(refresh_interval_s=float(os.getenv("ORB_CONTEXT_REFRESH_S", "10")))
_CONTEXT_ENGINE.configure(_ALPACA_PROVIDER, store=_RUNTIME_STORE)
_CONTEXT_ENGINE.start()
_PUBSUB = OptionalRedisPublisher()
_MONITOR.configure_runtime(provider=_ALPACA_PROVIDER, stream_cache=_STREAM, store=_RUNTIME_STORE, context_engine=_CONTEXT_ENGINE, pubsub=_PUBSUB)

LOG_PATH = BASE_DIR / "models" / "orb_ranker_train.log"

_ML_LOCK = threading.Lock()
_ML_STATE = {
    "status": "ready",
    "started_at": None,
    "updated_at": None,
    "error": None,
    "log_path": str(LOG_PATH),
    "pid": None,
    "orb_under_30_path": None,
    "orb_ge_30_path": None,
    "rr_path": None,
    "entry_now_pm_path": str(BASE_DIR / "models" / "entry_now_30m_pm.pkl"),
    "entry_now_rth_path": str(BASE_DIR / "models" / "entry_now_30m_rth.pkl"),
}


def _ml_set(**updates):
    with _ML_LOCK:
        _ML_STATE.update(updates)
        _ML_STATE["updated_at"] = time.time()


def _refresh_ml_state() -> None:
    from ml.model_registry import resolve_orb_bucket_path, resolve_rr_model_path

    orb_under_30 = Path(resolve_orb_bucket_path(29.99))
    orb_ge_30 = Path(resolve_orb_bucket_path(30.00))
    rr_path = Path(resolve_rr_model_path())
    entry_pm = BASE_DIR / "models" / "entry_now_30m_pm.pkl"
    entry_rth = BASE_DIR / "models" / "entry_now_30m_rth.pkl"

    missing = []
    for label, p in (
        ("orb_under_30", orb_under_30),
        ("orb_ge_30", orb_ge_30),
        ("rr", rr_path),
        ("entry_now_pm", entry_pm),
        ("entry_now_rth", entry_rth),
    ):
        if not (p.exists() and p.stat().st_size > 0):
            missing.append(f"{label}:{p}")

    _ml_set(
        orb_under_30_path=str(orb_under_30),
        orb_ge_30_path=str(orb_ge_30),
        rr_path=str(rr_path),
        entry_now_pm_path=str(entry_pm),
        entry_now_rth_path=str(entry_rth),
        status=("failed" if missing else "ready"),
        error=("missing_models: " + "; ".join(missing)) if missing else None,
    )


_refresh_ml_state()


def _filter_regular_hours(df):
    try:
        if df is None or df.empty:
            return df
        import pytz
        ET = pytz.timezone("America/New_York")
        if getattr(df.index, "tz", None) is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        return df.between_time("09:30", "16:00")
    except Exception:
        return df


def _tail_file(path: str, max_lines: int = 40) -> list[str]:
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        return lines[-max_lines:]
    except Exception:
        return []


_ET = ZoneInfo("America/New_York")

_ENTRY_NOW_CACHE: dict[str, dict] = {}
_ENTRY_NOW_CACHE_MTIME: dict[str, float] = {}


def _load_entry_now_bundle(path: str) -> dict:
    """Load Entry-Now model bundle from disk with mtime caching."""
    p = str(path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    mt = os.path.getmtime(p)
    if _ENTRY_NOW_CACHE.get(p) is not None and _ENTRY_NOW_CACHE_MTIME.get(p) == mt:
        return _ENTRY_NOW_CACHE[p]
    import joblib
    blob = joblib.load(p)
    if not isinstance(blob, dict):
        raise TypeError(f"Expected dict bundle from joblib.load({p}), got {type(blob)}")
    if "model" not in blob or "feature_names" not in blob:
        raise KeyError(f"Entry-Now bundle missing keys; found: {list(blob.keys())}")
    _ENTRY_NOW_CACHE[p] = blob
    _ENTRY_NOW_CACHE_MTIME[p] = mt
    return blob


def _select_entry_now_model_path() -> tuple[str, str]:
    """Return (model_path, bucket) where bucket is 'pm' or 'rth'. Strict: no legacy default fallback."""
    now = datetime.now(tz=_ET).time()
    is_pm = (now.hour > 4 or (now.hour == 4 and now.minute >= 0)) and (now.hour < 9 or (now.hour == 9 and now.minute < 30))
    p_pm = str(Path(__file__).resolve().parent / "models" / "entry_now_30m_pm.pkl")
    p_rth = str(Path(__file__).resolve().parent / "models" / "entry_now_30m_rth.pkl")
    if is_pm:
        if os.path.exists(p_pm):
            return p_pm, "pm"
        raise FileNotFoundError("Entry-Now PM model missing: models/entry_now_30m_pm.pkl")
    if os.path.exists(p_rth):
        return p_rth, "rth"
    raise FileNotFoundError("Entry-Now RTH model missing: models/entry_now_30m_rth.pkl")


def _go_hint(side: str, entry: float, stop: float, *, max_risk_pct: float = 12.0) -> str:
    """Hint text to adjust stop/entry so risk_pct <= max_risk_pct."""
    if entry is None or stop is None:
        return ""
    try:
        e = float(entry)
        s = float(stop)
    except Exception:
        return ""
    if e <= 0:
        return ""
    r = float(max_risk_pct) / 100.0
    sd = (side or "").lower()

    if sd == "long":
        min_stop = e * (1.0 - r)
        max_entry = s / (1.0 - r) if (1.0 - r) > 0 else None
        parts = [f"need stop ≥ ${min_stop:.2f} (risk ≤ {max_risk_pct:.0f}%)"]
        if max_entry and max_entry > 0:
            parts.append(f"or pullback entry ≤ ${max_entry:.2f} (keeping stop ${s:.2f})")
        return "To make this GO: " + " • ".join(parts)

    if sd == "short":
        max_stop = e * (1.0 + r)
        min_entry = s / (1.0 + r) if (1.0 + r) > 0 else None
        parts = [f"need stop ≤ ${max_stop:.2f} (risk ≤ {max_risk_pct:.0f}%)"]
        if min_entry and min_entry > 0:
            parts.append(f"or entry ≥ ${min_entry:.2f} (keeping stop ${s:.2f})")
        return "To make this GO: " + " • ".join(parts)

    return ""


def _start_ml_bootstrap_once():
    # Legacy training/bootstrap removed.
    # Runtime now uses strict active models resolved from ml.model_registry.
    _refresh_ml_state()


@app.get("/api/ml_status")
def api_ml_status():
    """Return active runtime model status for the UI."""
    try:
        _refresh_ml_state()
    except Exception as e:
        _ml_set(status="failed", error=f"{type(e).__name__}: {e}")

    with _ML_LOCK:
        st = dict(_ML_STATE)

    st["log_tail"] = _tail_file(st["log_path"], 40)
    return jsonify(ok=True, **st)


def _version_hash(paths: List[Path]) -> str:
    """Stable build hash (content-based) for quick verification."""
    h = hashlib.sha256()
    for p in paths:
        try:
            h.update(p.name.encode("utf-8"))
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
        except Exception:
            h.update(p.name.encode("utf-8"))
            h.update(b":missing\0")
    return h.hexdigest()[:16]



@app.get("/api/health")
def api_health():
    """Health snapshot for server/provider/stream/models (real checks only)."""
    base = Path(__file__).resolve().parent
    models_dir = base / "models"
    model_glob = sorted(models_dir.glob("*.pkl")) if models_dir.exists() else []
    key_present = bool(_alpaca_key())
    secret_present = bool(_alpaca_secret())
    feed_req = (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower()

    out: Dict[str, Any] = {
        "ok": True,
        "build_id": BUILD_ID,
        "server": {
            "ok": True,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "time_et": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "alpaca": {
            "ok": False,
            "keys_present": key_present and secret_present,
            "feed_requested": feed_req,
            "trade": None,
            "quote": None,
            "bars_1m_count": None,
            "error": None,
        },
        "stream": {
            "ok": False,
            "initialized": _STREAM is not None,
            "connected": False,
            "symbols": [],
            "last_event_at": None,
            "last_subscribe_at": None,
            "error": None,
        },
        "models": {
            "dir": str(models_dir),
            "count": len(model_glob),
            "files": [
                {"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime}
                for p in model_glob[:50]
            ],
            "orb_under_30_present": bool(_ML_STATE.get("orb_under_30_path") and Path(_ML_STATE["orb_under_30_path"]).exists()),
            "orb_ge_30_present": bool(_ML_STATE.get("orb_ge_30_path") and Path(_ML_STATE["orb_ge_30_path"]).exists()),
            "rr_present": bool(_ML_STATE.get("rr_path") and Path(_ML_STATE["rr_path"]).exists()),
            "entry_now_pm_present": bool(Path(_ML_STATE["entry_now_pm_path"]).exists()),
            "entry_now_rth_present": bool(Path(_ML_STATE["entry_now_rth_path"]).exists()),
        },
        "ml": {
            "status": _ML_STATE.get("status"),
            "model_loaded": (_ML_STATE.get("status") == "ready"),
            "error": _ML_STATE.get("error"),
        },
        "sentiment": {"ok": False, "error": None},
    }

    try:
        tr = _ALPACA_PROVIDER.get_latest_trade("AAPL") if hasattr(_ALPACA_PROVIDER, "get_latest_trade") else {"price": float(_ALPACA_PROVIDER.get_latest_trade_price("AAPL"))}
        qt = _ALPACA_PROVIDER.get_latest_quote("AAPL")
        bars = _ALPACA_PROVIDER.get_bars(BarsRequest(symbol="AAPL", interval="1m", period="1d", include_prepost=True))
        out["alpaca"]["trade"] = tr
        out["alpaca"]["quote"] = qt
        out["alpaca"]["bars_1m_count"] = int(len(getattr(bars, "index", [])))
        out["alpaca"]["ok"] = True
    except Exception as e:
        out["alpaca"]["error"] = f"{type(e).__name__}: {e}"

    try:
        if _STREAM is not None:
            st = _STREAM.state()
            out["stream"].update({
                "ok": True,
                "connected": bool(st.connected),
                "symbols": st.symbols,
                "last_event_at": st.last_event_at,
                "last_subscribe_at": getattr(st, "last_subscribe_at", None),
                "error": st.error,
            })
    except Exception as e:
        out["stream"]["error"] = f"{type(e).__name__}: {e}"

    try:
        _ = SentimentService(provider="auto")
        out["sentiment"]["ok"] = True
    except Exception as e:
        out["sentiment"]["error"] = f"{type(e).__name__}: {e}"

    out["ok"] = bool(out["server"]["ok"] and out["alpaca"]["ok"] and out["models"]["count"] > 0)
    return jsonify(out)


@app.get("/api/preflight")
def api_preflight():
    """AM readiness checklist (real checks; no placeholders)."""
    canary = str(request.args.get("symbol") or "AAPL").strip().upper() or "AAPL"
    out: Dict[str, Any] = {
        "ok": True,
        "build_id": BUILD_ID,
        "canary": canary,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "time_et": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "checks": [],
        "summary": {},
    }

    def add_check(key: str, label: str, ok: bool, severity: str = "fail", detail: Any = None):
        item = {"key": key, "label": label, "ok": bool(ok), "severity": severity}
        if detail is not None:
            item["detail"] = detail
        out["checks"].append(item)
        return item

    feed_req = (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex"
    key_present = bool(_alpaca_key())
    sec_present = bool(_alpaca_secret())
    add_check("alpaca_keys", "Alpaca API credentials present", key_present and sec_present, "fail", {"feed_requested": feed_req})

    try:
        tr = _ALPACA_PROVIDER.get_latest_trade(canary) if hasattr(_ALPACA_PROVIDER, "get_latest_trade") else {"price": _ALPACA_PROVIDER.get_latest_trade_price(canary)}
        qt = _ALPACA_PROVIDER.get_latest_quote(canary)
        bars = _ALPACA_PROVIDER.get_bars(BarsRequest(symbol=canary, interval="1m", period="1d", include_prepost=True))
        bars_n = int(len(getattr(bars, "index", [])))
        spread_pct = None
        try:
            bid = float(qt.get("bid_price")); ask = float(qt.get("ask_price")); mid = (bid + ask) / 2.0
            if mid > 0: spread_pct = (ask - bid) / mid * 100.0
        except Exception:
            pass
        add_check("alpaca_rest", f"Alpaca REST market data works ({canary})", True, "fail", {
            "trade_price": tr.get("price") if isinstance(tr, dict) else None,
            "quote": qt,
            "bars_1m_count": bars_n,
            "spread_pct": spread_pct,
        })
    except Exception as e:
        add_check("alpaca_rest", f"Alpaca REST market data works ({canary})", False, "fail", str(e))

    try:
        stream_probe = _ensure_stream(start=True, require=False, canary=canary, wait_for_boot=True, timeout_sec=8.0)
        if stream_probe is None:
            add_check("stream_init", "Live stream initialized", False, "warn", (_STREAM_ERROR or "stream_not_initialized"))
        else:
            st0 = stream_probe.state()
            add_check(
                "stream_init",
                "Live stream initialized",
                True,
                "warn",
                {
                    "connected": bool(st0.connected),
                    "symbols": st0.symbols[:20],
                    "last_subscribe_at": getattr(st0, "last_subscribe_at", None),
                    "last_run_started_at": getattr(st0, "last_run_started_at", None),
                    "last_run_exited_at": getattr(st0, "last_run_exited_at", None),
                    "run_attempt": getattr(st0, "run_attempt", None),
                    "last_callback_kind": getattr(st0, "last_callback_kind", None),
                    "last_callback_symbol": getattr(st0, "last_callback_symbol", None),
                },
            )
            try:
                _require_stream_events(timeout_sec=8.0)
                st1 = stream_probe.state()
                add_check(
                    "stream_live",
                    f"Live stream receives events ({canary})",
                    bool(st1.last_event_at),
                    "warn",
                    {
                        "connected": bool(st1.connected),
                        "last_event_at": st1.last_event_at,
                        "last_subscribe_at": getattr(st1, "last_subscribe_at", None),
                        "last_run_started_at": getattr(st1, "last_run_started_at", None),
                        "last_run_exited_at": getattr(st1, "last_run_exited_at", None),
                        "run_attempt": getattr(st1, "run_attempt", None),
                        "last_callback_kind": getattr(st1, "last_callback_kind", None),
                        "last_callback_symbol": getattr(st1, "last_callback_symbol", None),
                        "error": st1.error,
                        "symbols": st1.symbols[:20],
                    },
                )
            except Exception as e:
                st1 = stream_probe.state()
                add_check(
                    "stream_live",
                    f"Live stream receives events ({canary})",
                    False,
                    "warn",
                    {
                        "connected": bool(st1.connected),
                        "last_event_at": st1.last_event_at,
                        "last_subscribe_at": getattr(st1, "last_subscribe_at", None),
                        "last_run_started_at": getattr(st1, "last_run_started_at", None),
                        "last_run_exited_at": getattr(st1, "last_run_exited_at", None),
                        "run_attempt": getattr(st1, "run_attempt", None),
                        "last_callback_kind": getattr(st1, "last_callback_kind", None),
                        "last_callback_symbol": getattr(st1, "last_callback_symbol", None),
                        "error": st1.error,
                        "symbols": st1.symbols[:20],
                        "probe_error": f"{type(e).__name__}: {e}",
                    },
                )
    except Exception as e:
        add_check("stream_live", f"Live stream receives events ({canary})", False, "warn", str(e))

    from ml.model_registry import resolve_orb_bucket_path, resolve_rr_model_path

    models_dir = _PROJECT_ROOT / "models"
    model_checks = [
        resolve_orb_bucket_path(29.99),   # under-30 ORB model
        resolve_orb_bucket_path(30.00),   # 30+ ORB model
        resolve_rr_model_path(),          # RR model
        models_dir / "entry_now_30m_pm.pkl",
        models_dir / "entry_now_30m_rth.pkl",
    ]

    seen = set()
    for p in model_checks:
        p = Path(p)
        if str(p) in seen:
            continue
        seen.add(str(p))

        okf = p.exists() and p.stat().st_size > 0
        add_check(
            f"model:{p.name}",
            f"Model present: {p.name}",
            okf,
            "fail",
            {
                "path": str(p),
                "size": (p.stat().st_size if p.exists() else None),
            },
        )

    add_check(
    "ml_loaded",
    "Active runtime models resolved",
    _ML_STATE.get("status") == "ready",
    "warn",
    {
        "ml_state": _ML_STATE.get("status"),
        "error": _ML_STATE.get("error"),
        "orb_under_30_path": _ML_STATE.get("orb_under_30_path"),
        "orb_ge_30_path": _ML_STATE.get("orb_ge_30_path"),
        "rr_path": _ML_STATE.get("rr_path"),
        "entry_now_pm_path": _ML_STATE.get("entry_now_pm_path"),
        "entry_now_rth_path": _ML_STATE.get("entry_now_rth_path"),
    },
)

    try:
        _ = SentimentService(provider="auto")
        add_check("sentiment", "Sentiment/catalyst service initializes", True, "warn")
    except Exception as e:
        add_check("sentiment", "Sentiment/catalyst service initializes", False, "warn", str(e))

    try:
        if hasattr(_ALPACA_PROVIDER, "get_news"):
            items = _ALPACA_PROVIDER.get_news(canary, limit=2)
            add_check("alpaca_news", f"Alpaca news endpoint works ({canary})", True, "warn", {"articles": len(items or [])})
        else:
            add_check("alpaca_news", "Alpaca news endpoint works", False, "warn", "provider_missing_get_news")
    except Exception as e:
        add_check("alpaca_news", f"Alpaca news endpoint works ({canary})", False, "warn", str(e))

    fail_count = sum(1 for c in out["checks"] if (not c["ok"]) and c.get("severity") == "fail")
    warn_count = sum(1 for c in out["checks"] if (not c["ok"]) and c.get("severity") != "fail")
    out["summary"] = {
        "fail_count": fail_count,
        "warn_count": warn_count,
        "ready_for_scan": fail_count == 0,
        "ready_for_live_monitor": fail_count == 0,
        "ready_for_stream_confidence": fail_count == 0 and warn_count == 0,
        "feed_requested": feed_req,
    }
    out["ok"] = fail_count == 0
    return jsonify(out)

# In-memory job store for long scans (development only)
_JOBS = {}
_JOBS_LOCK = threading.Lock()

def _new_job():
    jid = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[jid] = {
            "status": "running",
            "started_at": time.time(),
            "updated_at": time.time(),
            "progress": {
                "scanned": 0,
                "chunks_done": 0,
                "chunks_total": 0,
                "offset": 0,
                "end_offset": 0,
            },
            "result": None,
            "partial_result": {
                "provider": getattr(_ALPACA_PROVIDER, "name", "alpaca"),
                "strategy": None,
                "scan_date": None,
                "session_date_used": None,
                "prefilter_counts": {},
                "prefilter_samples": [],
                "thresholds_used": {},
                "reject_counts": {},
                "failure_samples_by_code": {},
                "data_failures": {},
                "shortlisted": 0,
                "candidates": [],
                "seed_candidates": [],
                "primary_candidates": [],
                "primary_mode": "empty",
                "primary_message": None,
                "zero_result_diagnostics": {},
                "scanned": 0,
                "chunks": 0,
                "end_offset": 0,
                "universe_size": 0,
                "mode": None,
                "debug": {
                    "strategy": None,
                    "session_date_used": None,
                    "failure_samples": [],
                    "failure_samples_by_code": {},
                    "chunk_errors": [],
                    "chunk_timings": [],
                },
            },
            "error": None,
            "params": {},
            "thresholds_used": {},
            "provider": getattr(_ALPACA_PROVIDER, "name", "alpaca"),
        }
    return jid

def _set_job(jid, **updates):
    with _JOBS_LOCK:
        job = _JOBS.get(jid)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()

def _update_progress(jid, **p):
    with _JOBS_LOCK:
        job = _JOBS.get(jid)
        if not job:
            return
        job["progress"].update(p)
        job["updated_at"] = time.time()

def _get_job(jid):
    with _JOBS_LOCK:
        return _JOBS.get(jid)


def _find_running_job() -> tuple[str, dict[str, Any]] | tuple[None, None]:
    with _JOBS_LOCK:
        running = [
            (jid, job)
            for jid, job in _JOBS.items()
            if isinstance(job, dict) and str(job.get("status") or "").strip().lower() == "running"
        ]
    if not running:
        return None, None
    running.sort(key=lambda item: float((item[1] or {}).get("updated_at") or (item[1] or {}).get("started_at") or 0.0), reverse=True)
    return running[0]


def _is_truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _disable_auto_monitor(params: dict[str, Any] | None = None) -> bool:
    default_v = os.getenv("ORB_DISABLE_AUTO_MONITOR_DEFAULT", "1")
    if params is None:
        return _is_truthy(default_v)
    return _is_truthy((params or {}).get("disable_auto_monitor", default_v))

def _candidate_sort_score(c: dict) -> float:
    return candidate_sort_score(c)

def _get_scan_candidates_for_monitor(job_id: str) -> list[dict]:
    job = _get_job(job_id)
    if not job:
        # Not in memory — try the persistent store
        job = _RUNTIME_STORE.scan_job_get(job_id)
    if not job:
        raise KeyError("unknown_job_id")
    st = job.get("status")
    if st != "done":
        raise RuntimeError(f"scan_job_not_done:{st}")
    result = job.get("result") or {}
    cands = result.get("seed_candidates") or result.get("candidates") or []
    if not isinstance(cands, list):
        raise RuntimeError("scan_job_candidates_invalid")
    out = [c for c in cands if isinstance(c, dict)]
    out.sort(key=_candidate_sort_score, reverse=True)
    return out


def _get_scan_seed_symbols(job_id: str) -> list[str]:
    job = _get_job(job_id)
    if not job:
        # Not in memory — try the persistent store
        job = _RUNTIME_STORE.scan_job_get(job_id)
    if not job:
        raise KeyError("unknown_job_id")
    st = job.get("status")
    if st != "done":
        raise RuntimeError(f"scan_job_not_done:{st}")
    result = job.get("result") or {}
    rows = result.get("seed_symbols") or result.get("slice_symbols") or []
    if not isinstance(rows, list):
        return []
    out = []
    seen = set()
    for s in rows:
        sym = str(s or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


# STRICT_STREAM_GATE_V1
def _require_stream_events(timeout_sec: float = 8.0) -> None:
    """
    Gold rule: if streaming is enabled/required and we subscribed symbols, we must see events soon.
    No fallback / no silent failure.
    """
    import time
    global _STREAM
    stream = _ensure_stream(start=True, require=True)
    if stream is None:
        raise RuntimeError("Stream required but not initialized")
    _STREAM = stream
    deadline = time.time() + max(0.5, float(timeout_sec))
    last_err = None
    while time.time() < deadline:
        st = _STREAM.state()
        if getattr(st, "error", None):
            last_err = st.error
            break
        if getattr(st, "last_event_at", None):
            return
        time.sleep(0.05)
    if last_err:
        raise RuntimeError(f"Stream subscribe failed: {last_err}")
    raise RuntimeError("Stream subscribed but no events received (possible WS cap/429, network, or handler failure)")


@app.get("/api/stream_status")
def api_stream_status():
    """Inspect Alpaca streaming connection + subscribed symbols."""
    stream_cache = _ensure_stream(start=False, require=False)
    if stream_cache is None:
        return jsonify(ok=False, streaming=False, error=(_STREAM_ERROR or "stream_not_initialized"), lazy=True)
    st = stream_cache.state()
    return jsonify(ok=True, streaming=True, **{
        "connected": st.connected,
        "started_at": st.started_at,
        "last_event_at": st.last_event_at,
        "last_subscribe_at": getattr(st, "last_subscribe_at", None),
        "last_run_started_at": getattr(st, "last_run_started_at", None),
        "last_run_exited_at": getattr(st, "last_run_exited_at", None),
        "run_attempt": getattr(st, "run_attempt", None),
        "last_callback_kind": getattr(st, "last_callback_kind", None),
        "last_callback_symbol": getattr(st, "last_callback_symbol", None),
        "symbols": st.symbols,
        "error": st.error,
        "feed": (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex",
    })




# ---------------- Watchlist / monitor (server-backed, stream live) ----------------
_WATCHLIST: list[str] = []
_WATCHLIST_MAX = 25

def _wl_norm(symbols):
    out=[]
    seen=set()
    for s in symbols or []:
        sym=str(s).strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
        if len(out) >= _WATCHLIST_MAX:
            break
    return out

@app.post("/api/watchlist/set")
def api_watchlist_set():
    global _WATCHLIST
    body = request.get_json(silent=True) or {}
    symbols = _wl_norm(body.get("symbols") or [])
    _WATCHLIST = symbols

    try:
        stream_cache = _ensure_stream(start=True, require=True)
    except Exception as e:
        return jsonify(ok=False, max=_WATCHLIST_MAX, symbols=_WATCHLIST, streaming=False, error=f"{type(e).__name__}: {e}"), 500
    if stream_cache is None:
        return jsonify(ok=False, max=_WATCHLIST_MAX, symbols=_WATCHLIST, streaming=False, error=(_STREAM_ERROR or "stream_not_initialized")), 500

    # ensure symbols subscribed (strict: real error bubbles into stream_status)
    try:
        stream_cache.ensure_symbols(_WATCHLIST)
    except Exception as e:
        return jsonify(ok=False, max=_WATCHLIST_MAX, symbols=_WATCHLIST, streaming=True, error=f"Watchlist stream subscribe failed: {type(e).__name__}: {e}"), 500

    return jsonify(ok=True, max=_WATCHLIST_MAX, symbols=_WATCHLIST)

@app.get("/api/watchlist")
def api_watchlist():
    stream_cache = _ensure_stream(start=False, require=False)
    if stream_cache is None:
        return jsonify(ok=False, max=_WATCHLIST_MAX, symbols=_WATCHLIST, rows={}, streaming=False, error=(_STREAM_ERROR or "stream_not_initialized")), 500

    st = stream_cache.state()
    rows={}
    for sym in _WATCHLIST:
        trade = stream_cache.latest_trade(sym) or {}
        quote = stream_cache.latest_quote(sym) or {}
        source = "stream"
        if not trade and _ALPACA_PROVIDER is not None:
            try:
                trade = _ALPACA_PROVIDER.get_latest_trade(sym) or {}
                source = "rest"
            except Exception:
                pass
        if not quote and _ALPACA_PROVIDER is not None:
            try:
                quote = _ALPACA_PROVIDER.get_latest_quote(sym) or {}
                source = "rest" if source == "stream" else source
            except Exception:
                pass
        rows[sym] = {
            "trade": trade,
            "quote": quote,
            "source": source,
        }
    return jsonify(ok=True, max=_WATCHLIST_MAX, symbols=_WATCHLIST, rows=rows, streaming=bool(st.streaming), connected=bool(st.connected), last_event_at=st.last_event_at, error=st.error)


@app.get("/api/model_status")
def api_model_status():
    """Gold-standard model transparency for active runtime models only."""
    import joblib
    from ml.model_registry import (
        env_orb_strict_ml,
        env_rr_strict_ml,
        resolve_orb_bucket_path,
        resolve_rr_model_path,
        sha256_file,
        mtime_utc_iso,
    )

    load_check = str(request.args.get("load_check") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
    base = Path(__file__).resolve().parent
    entry_pm = base / "models" / "entry_now_30m_pm.pkl"
    entry_rth = base / "models" / "entry_now_30m_rth.pkl"

    def _stat(path: Path) -> dict:
        return {
            "path": str(path),
            "exists": path.exists(),
            "mtime_utc": mtime_utc_iso(path),
            "sha256": sha256_file(path),
        }

    def _loadable(path: Path) -> dict:
        if not load_check:
            return {"loadable_ok": None, "load_error": None}
        if not path.exists():
            return {"loadable_ok": False, "load_error": "missing"}
        try:
            obj = joblib.load(path)
            if not isinstance(obj, dict):
                return {"loadable_ok": False, "load_error": f"expected dict bundle; got {type(obj)}"}
            return {"loadable_ok": True, "load_error": None}
        except Exception as e:
            return {"loadable_ok": False, "load_error": f"{type(e).__name__}: {e}"}

    orb_under30 = resolve_orb_bucket_path(29.99)
    orb_liquid = resolve_orb_bucket_path(30.00)
    rr_path = resolve_rr_model_path()

    payload = {
        "ok": True,
        "strict": {
            "orb": env_orb_strict_ml(),
            "rr": env_rr_strict_ml(),
            "entry_now": True,
        },
        "active_models": {
            "orb_under_30": {**_stat(orb_under30), **_loadable(orb_under30)},
            "orb_ge_30": {**_stat(orb_liquid), **_loadable(orb_liquid)},
            "rr_gold": {**_stat(rr_path), **_loadable(rr_path)},
            "entry_now_pm": {**_stat(entry_pm), **_loadable(entry_pm)},
            "entry_now_rth": {**_stat(entry_rth), **_loadable(entry_rth)},
        },
        "retired_models": [
            "models/orb_ranker.pkl",
            "models/orb_ranker_b_under20.pkl",
            "models/orb_ranker_liquid.pkl",
            "models/orb_ranker_outlier.pkl",
            "models/orb_ranker_5k_90d.pkl",
            "models/orb_ranker_part1.pkl",
            "models/orb_ranker_part2.pkl",
            "models/entry_now_30m.pkl",
            "models/entry_now_30m_smoke.pkl",
            "models/SMOKE_liquid_time.pkl",
        ],
    }
    return jsonify(payload)


@app.post("/api/monitor_start")
def api_monitor_start():
    data = request.get_json(silent=True) or request.form.to_dict(flat=True) or {}
    for k, v in request.args.items():
        data.setdefault(k, v)

    feed = str(data.get("feed") or os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex"
    if feed not in {"iex", "sip"}:
        return jsonify(ok=False, error="invalid_alpaca_data_feed"), 400
    try:
        stream_cache = _ensure_stream(start=True, require=True)
    except Exception as e:
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}", feed=feed), 503
    if stream_cache is None:
        return jsonify(ok=False, error=(_STREAM_ERROR or "stream_not_initialized"), feed=feed), 503

    try:
        top_n = _safe_int(data.get("top_n", 10), 10)
    except Exception:
        return jsonify(ok=False, error="invalid_top_n"), 400
    top_n = max(1, min(50 if feed == "sip" else 30, top_n))

    job_id = str(data.get("job_id") or "").strip()
    symbols = data.get("symbols")
    long_only = _is_truthy(data.get("long_only", "0"))
    short_only = _is_truthy(data.get("short_only", "0"))
    symbols_order = data.get("symbols_order")
    candidates = data.get("candidates")
    if isinstance(symbols_order, str) and symbols_order.strip():
        symbols_order = [s.strip().upper() for s in symbols_order.split(",") if s.strip()]
    elif isinstance(symbols_order, list):
        symbols_order = [str(s).strip().upper() for s in symbols_order if str(s).strip()]
    else:
        symbols_order = None

    try:
        force_restart = str(data.get("force_restart") or "").strip().lower() in {"1", "true", "yes", "on"}
        if not force_restart:
            existing = _MONITOR.active_session(watch_modes={"scanner", "watchlist"})
            if existing is not None:
                return jsonify(ok=True, reused=True, monitor_id=existing.monitor_id, job_id=existing.job_id, symbols=sorted(list(existing.symbols.keys())), feed_requested=existing.feed_requested, feed_used=existing.feed_used, source=existing.source, started_at=existing.started_at, long_only=bool(existing.long_only))

        if str(data.get("source") or "").strip() == "watchlist":
            plans = _RUNTIME_STORE.desk_watchlist_all()
            if not plans:
                return jsonify(ok=False, error="desk_watchlist_empty — add symbols to the Desk Watchlist first"), 400
            sess = _MONITOR.start_from_watchlist_plans(plans=plans, feed=feed, provider=_ALPACA_PROVIDER, stream_cache=stream_cache)
        elif isinstance(symbols, str) and symbols.strip():
            syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            sess = _MONITOR.start_from_symbols(symbols=syms, feed=feed, provider=_ALPACA_PROVIDER, stream_cache=stream_cache)
        elif isinstance(symbols, list) and symbols:
            sess = _MONITOR.start_from_symbols(symbols=symbols, feed=feed, provider=_ALPACA_PROVIDER, stream_cache=stream_cache)
        else:
            if isinstance(candidates, list) and candidates:
                cands = [c for c in candidates if isinstance(c, dict) and str(c.get("symbol") or "").strip()]
            else:
                if not job_id:
                    return jsonify(ok=False, error="missing_job_id_or_symbols"), 400
                cands = _get_scan_candidates_for_monitor(job_id)
            if long_only:
                cands = [c for c in cands if str((c or {}).get("best_side") or "").strip().lower() != "short"]
            if short_only:
                cands = [c for c in cands if str((c or {}).get("best_side") or "").strip().lower() == "short"]
            try:
                _promo_syms = _get_scan_seed_symbols(job_id) if job_id else None
            except (KeyError, RuntimeError):
                _promo_syms = None
            sess = _MONITOR.start_from_scan_candidates(
                job_id=job_id,
                candidates=cands,
                top_n=top_n,
                feed=feed,
                provider=_ALPACA_PROVIDER,
                stream_cache=stream_cache,
                symbols_order=symbols_order,
                long_only=long_only,
                short_only=short_only,
                source=str(data.get("source") or ("candidates_payload" if isinstance(candidates, list) and candidates else "scan_job_top_n")),
                promotion_candidates=_promo_syms,
            )
        # Auto-start paper trader whenever the live monitor starts
        _pt_ensure_running()
        return jsonify(ok=True, reused=False, monitor_id=sess.monitor_id, job_id=sess.job_id, symbols=sorted(list(sess.symbols.keys())), feed_requested=sess.feed_requested, feed_used=sess.feed_used, source=sess.source, started_at=sess.started_at, long_only=long_only, short_only=short_only)
    except KeyError as e:
        return jsonify(ok=False, error=str(e).strip("'")), 404
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("scan_job_not_done:"):
            return jsonify(ok=False, error=msg, pending=True), 409
        return jsonify(ok=False, error=msg), 400


@app.post("/api/monitor_stop")
def api_monitor_stop():
    data = request.get_json(silent=True) or request.form.to_dict(flat=True) or {}
    monitor_id = str(data.get("monitor_id") or request.args.get("monitor_id") or "").strip()
    if not monitor_id:
        return jsonify(ok=False, error="missing_monitor_id"), 400
    if not _MONITOR.stop(monitor_id):
        return jsonify(ok=False, error="unknown_monitor_id"), 404
    return jsonify(ok=True, stopped=True, monitor_id=monitor_id)

@app.get('/favicon.ico')
def favicon():
    try:
        return send_from_directory(str((ROOT / 'static').resolve()), 'icon.png')
    except Exception:
        return ('', 204)

@app.route('/api/analyze', methods=['GET','POST'])
def api_analyze():
    symbol = (request.args.get('symbol') or '').strip().upper()
    include_prepost = str(request.args.get('include_prepost', 'false')).strip().lower() in {'1','true','yes','on'}
    if request.method == 'POST':
        payload = request.get_json(silent=True) or {}
        if not symbol:
            symbol = str(payload.get('symbol') or '').strip().upper()
    if not symbol:
        return jsonify(ok=False, error='missing_symbol'), 400

    try:
        from dataclasses import replace
        from scanner.orb import ORBConfig, scan_symbols, scan_range_reversion_symbols, resolve_session_date
        provider = _ALPACA_PROVIDER

        if provider is None:
            return jsonify(ok=False, error=(_PROVIDER_ERROR or 'provider_not_initialized'), symbol=symbol), 503
        intraday = provider.get_bars(BarsRequest(symbol=symbol, interval="1m", period="1d", include_prepost=include_prepost)).sort_index()
        if intraday is None or intraday.empty:
            return jsonify(ok=False, error='intraday_empty', symbol=symbol), 400

        live_state = _get_live_state(symbol)
        if live_state.get('last_trade_price') is None:
            return jsonify(ok=False, error=live_state.get('error') or 'live_snapshot_unavailable', symbol=symbol), 503
        last_price = float(live_state.get('last_trade_price'))

        first_close = float(_col(intraday,'Close','close').astype(float).iloc[0])
        pct_change = ((last_price - first_close) / first_close * 100.0) if first_close > 0 else None
        today_vol = float(_col(intraday,'Volume','volume').astype(float).sum())
        today_dollar_vol = today_vol * last_price

        try:
            from scanner.indicators import vwap as _vwap, trend_state_1m as _trend
            _vw = _vwap(intraday)
            _tr = _trend(intraday, vw=_vw, lookback=15)
            vwap_last = _tr.get("vwap_last")
            vwap_delta_pct = _tr.get("vwap_delta_pct")
            trend_state = _tr.get("state")
            trend_slope_pct = _tr.get("slope_pct_lookback")
            above_vwap = bool(last_price > float(vwap_last)) if vwap_last is not None else None
        except Exception:
            vwap_last = None
            vwap_delta_pct = None
            trend_state = None
            trend_slope_pct = None
            above_vwap = None

        cfg = replace(
            ORBConfig(),
            min_price=0.0,
            max_price=1e9,
            min_today_dollar_vol=0.0,
            min_avg20_dollar_vol=0.0,
            min_rvol=0.0,
            min_or_range_pct=0.0,
            max_or_range_pct=1e9,
            min_risk_per_share=0.01,
            max_risk_per_share=5.00,
            min_shares=0,
            max_notional=1e18,
        )

        orb_res = scan_symbols(
            [symbol],
            cfg,
            limit=1,
            provider=provider,
            use_ml=True,
            use_sentiment=False,
            use_catalyst=False,
            long_only=True,
            min_grade_enabled=True,
            min_grade='B',
            min_combined_enabled=True,
            min_combined_score=0.40,
            no_chop_enabled=True,
            min_vwap_enabled=True,
            min_pct_over_vwap=1.0,
        )
        rr_res = scan_range_reversion_symbols(
            [symbol],
            cfg,
            limit=1,
            provider=provider,
            use_ml=True,
            use_sentiment=False,
            use_catalyst=False,
        )

        orb_hit = orb_res.get('candidates', [{}])[0] if orb_res.get('candidates') else None
        orb_rej = orb_res.get('rejected_candidates', [{}])[0] if orb_res.get('rejected_candidates') else None
        rr_hit = rr_res.get('candidates', [{}])[0] if rr_res.get('candidates') else None
        rr_rej = rr_res.get('rejected_candidates', [{}])[0] if rr_res.get('rejected_candidates') else None

        analyses = {
            'orb': orb_hit if orb_hit and orb_hit.get('symbol') == symbol else orb_rej,
            'rr': rr_hit if rr_hit and rr_hit.get('symbol') == symbol else rr_rej,
        }
        strategy_selected = None
        admissible = False
        if analyses['orb'] and analyses['orb'].get('gate_passes'):
            strategy_selected = 'orb'
            admissible = True
        elif analyses['rr'] and analyses['rr'].get('gate_passes'):
            strategy_selected = 'range_reversion'
            admissible = True
        elif analyses['orb']:
            strategy_selected = 'orb'
        elif analyses['rr']:
            strategy_selected = 'range_reversion'

        return jsonify(
            ok=True,
            symbol=symbol,
            last_price=last_price,
            pct_change=pct_change,
            today_dollar_vol=today_dollar_vol,
            above_vwap=above_vwap,
            vwap_last=vwap_last,
            vwap_delta_pct=vwap_delta_pct,
            trend_state=trend_state,
            trend_slope_pct=trend_slope_pct,
            analyses=analyses,
            strategy_selected=strategy_selected,
            admissible=admissible,
            scan_date=resolve_session_date(provider, probe_symbol=symbol).isoformat(),
            live_state=live_state,
        )
    except Exception as e:
        return jsonify(ok=False, error=str(e), symbol=symbol), 400


def _entry_now_request_values() -> tuple[str, str, bool, bool]:
    data = request.get_json(silent=True) or {}
    symbol      = str(data.get("symbol") or request.args.get("symbol") or "").strip().upper()
    side_req    = str(data.get("side")   or request.args.get("side")   or "long").strip().lower()
    include_prepost = str(data.get("include_prepost") or request.args.get("include_prepost") or "0").lower() in ("1","true","yes")
    want_sent   = str(data.get("sentiment") or request.args.get("sentiment") or "0").lower() in ("1","true","yes")
    return symbol, side_req, include_prepost, want_sent



def _entry_now_intraday_context(provider: AlpacaProvider, symbol: str, *, include_prepost: bool) -> tuple[Any, float]:
    try:
        intraday = provider.get_bars(
            BarsRequest(symbol=symbol, interval='1m', period='1d', include_prepost=include_prepost)
        ).sort_index()
    except Exception as e:
        raise IntradayDataFailure(
            code='intraday_fetch_failed',
            message=f'{type(e).__name__}: {e}',
            stage='entry_now_intraday',
            symbol=symbol,
            context={'include_prepost': bool(include_prepost)},
            cause_type=type(e).__name__,
        ) from e

    if not include_prepost:
        intraday = _filter_regular_hours(intraday)
    if intraday is None or intraday.empty:
        raise IntradayDataFailure(
            code='intraday_empty',
            message='intraday_empty',
            stage='entry_now_intraday',
            symbol=symbol,
            context={'include_prepost': bool(include_prepost)},
        )

    # Prefer a fresh stream trade when available, but do not hard-require the stream
    # for Entry-Now. This planner can use real Alpaca snapshot/latest-trade data too.
    errors: list[str] = []

    try:
        stream_cache = _ensure_stream(start=True, require=False)
    except Exception as e:
        stream_cache = None
        errors.append(f'stream_boot_failed:{type(e).__name__}:{e}')

    if stream_cache is not None:
        try:
            stream_cache.ensure_symbols([symbol])
            trade = latest_trade_payload(stream_cache, symbol, max_age_sec=30.0)
            return intraday, float(trade["price"])
        except Exception as e:
            errors.append(f'stream_trade_unavailable:{type(e).__name__}:{e}')

    # Snapshot path (real provider data; batch/snapshot architecture)
    try:
        feed = (os.getenv('ALPACA_DATA_FEED') or 'iex').strip().lower() or 'iex'
        snap_map = provider.get_snapshots([symbol], feed=feed)
        snap = dict(snap_map.get(symbol) or {})
        latest_trade = dict(snap.get('latest_trade') or {})
        ref_price = latest_trade.get('price')
        if ref_price is None:
            ref_price = snap.get('reference_price')
        if ref_price is not None:
            return intraday, float(ref_price)
        snap_err = snap.get('error') or 'snapshot_missing_reference_price'
        errors.append(f'snapshot_unavailable:{snap_err}')
    except Exception as e:
        errors.append(f'snapshot_unavailable:{type(e).__name__}:{e}')

    # Final HTTP latest-trade fallback (still real provider data, not synthetic/fake).
    try:
        return intraday, float(provider.get_latest_trade_price(symbol))
    except Exception as e:
        errors.append(f'latest_trade_http_unavailable:{type(e).__name__}:{e}')

    raise IntradayDataFailure(
        code='latest_trade_unavailable',
        message=' | '.join(errors) if errors else 'latest_trade_unavailable',
        stage='entry_now_intraday',
        symbol=symbol,
        context={'include_prepost': bool(include_prepost)},
        cause_type='RuntimeError',
    )





def _entry_now_trend_context(intraday) -> dict[str, Any]:
    try:
        from scanner.indicators import vwap as _vwap, trend_state_1m as _trend
        _vw = _vwap(intraday)
        _tr = _trend(intraday, vw=_vw, lookback=15)
        return {
            'vwap_last': _tr.get('vwap_last'),
            'vwap_delta_pct': _tr.get('vwap_delta_pct'),
            'trend_state': _tr.get('state'),
            'trend_slope_pct': _tr.get('slope_pct_lookback'),
        }
    except Exception as e:
        raise TrendContextFailure(
            code='trend_context_failed',
            message=f'{type(e).__name__}: {e}',
            stage='entry_now_trend',
            cause_type=type(e).__name__,
        ) from e


def _entry_now_plan_state(*, side: str, last_price: float | None, stop: float | None, t2: float | None, p_2r_30m: float | None, chase_r: float | None, vwap_delta_pct: float | None, trend_state: str | None, entry: float | None = None) -> tuple[str, list[str], str | None]:
    return build_plan_state(
        side=side,
        last_price=last_price,
        entry=entry,
        stop=stop,
        target_2r=t2,
        p_2r_30m=p_2r_30m,
        chase_r=chase_r,
        vwap_delta_pct=vwap_delta_pct,
        trend_state=trend_state,
        go_hint_fn=_go_hint,
    )


def _entry_now_ml_payload(*, symbol: str, last_price: float, provider: AlpacaProvider, cand: Any, side: str, trend_state: str | None) -> dict[str, Any]:
    ml_base = None
    ml_adjusted = None
    ml_reason = None
    ml_bucket = None
    chase_r = None

    try:
        from ml.orb_model_service import score_orb_symbol

        ml_out = score_orb_symbol(symbol, last_price=float(last_price), provider=provider)
        ml_bucket = ml_out.get('bucket')
        if ml_out.get('score') is not None:
            ml_base = float(ml_out['score'])
    except Exception as e:
        raise EntryNowMLFailure(
            code='entry_now_ml_score_failed',
            message=f'{type(e).__name__}: {e}',
            stage='entry_now_ml',
            symbol=symbol,
            cause_type=type(e).__name__,
        ) from e

    try:
        entry0 = float(cand.entry)
        stop0 = float(cand.stop)
        r0 = abs(entry0 - stop0)
        if r0 > 0:
            chase_r = (last_price - entry0) / r0 if side == 'long' else (entry0 - last_price) / r0
        adj = ml_base
        if adj is not None and chase_r is not None:
            import math as _math
            adj = adj * _math.exp(-0.35 * max(0.0, float(chase_r) - 1.0))
            if trend_state in {'up', 'reclaim_vwap'} and side == 'long':
                adj *= 1.10
            if trend_state in {'down', 'lost_vwap'} and side == 'short':
                adj *= 1.10
            if trend_state in {'up', 'reclaim_vwap'} and side == 'short':
                adj *= 0.85
            if trend_state in {'down', 'lost_vwap'} and side == 'long':
                adj *= 0.85
            ml_adjusted = float(max(0.0, min(1.0, adj)))
            ml_reason = f"ml_base * exp(-0.35*max(0,chaseR-1)) with trend nudges; chaseR={chase_r:.2f}"
    except Exception as e:
        raise EntryNowMLFailure(
            code='entry_now_ml_adjust_failed',
            message=f'{type(e).__name__}: {e}',
            stage='entry_now_ml',
            symbol=symbol,
            cause_type=type(e).__name__,
        ) from e

    return {
        'ml_base': ml_base,
        'ml_adjusted': ml_adjusted,
        'ml_reason': ml_reason,
        'ml_bucket': ml_bucket,
        'chase_r': chase_r,
    }


@app.route('/api/entry_now', methods=['GET','POST'])
def api_entry_now():
    """Plan an entry *now* at the current price, with stop/targets + ML/sentiment context.

    Tenants:
    - Uses only real provider data (1m bars + daily history + trained .pkl models).
    - No placeholders. If upstream fails, returns the real error.
    """
    symbol = (request.args.get('symbol') or '').strip().upper()
    side_req = (request.args.get('side') or 'auto').strip().lower()
    include_prepost = str(request.args.get('include_prepost', 'false')).strip().lower() in {'1','true','yes','on'}
    want_sent = str(request.args.get('use_sentiment', 'false')).strip().lower() in {'1','true','yes','on'}

    if request.method == 'POST':
        payload = request.get_json(silent=True) or {}
        if not symbol:
            symbol = str(payload.get('symbol') or '').strip().upper()
        side_req = str(payload.get('side') or side_req).strip().lower()
        include_prepost = bool(payload.get('include_prepost', include_prepost))
        want_sent = bool(payload.get('use_sentiment', want_sent))

    if not symbol:
        return jsonify(ok=False, error='missing_symbol'), 400

    try:
        provider = _ALPACA_PROVIDER

        try:
            intraday, last_price = _entry_now_intraday_context(provider, symbol, include_prepost=include_prepost)
        except IntradayDataFailure as e:
            return jsonify(ok=False, error=e.message, code=e.code, symbol=symbol)

        trend_error = None
        try:
            trend_ctx = _entry_now_trend_context(intraday)
        except TrendContextFailure as e:
            trend_error = e.to_dict()
            trend_ctx = {
                'vwap_last': None,
                'vwap_delta_pct': None,
                'trend_state': None,
                'trend_slope_pct': None,
            }

        vwap_last = trend_ctx['vwap_last']
        vwap_delta_pct = trend_ctx['vwap_delta_pct']
        trend_state = trend_ctx['trend_state']
        trend_slope_pct = trend_ctx['trend_slope_pct']

        # Build a baseline ORB plan (informational; relaxed bounds)
        from dataclasses import replace as _replace
        from scanner.orb import build_orb_plan, resolve_session_date, ORBConfig, _buffer

        cfg = _replace(
            ORBConfig(),
            min_price=0.0,
            max_price=1e9,
            min_today_dollar_vol=0.0,
            min_avg20_dollar_vol=0.0,
            min_rvol=0.0,
            min_or_range_pct=0.0,
            max_or_range_pct=1e9,
            # keep real sizing bounds for sensible stops
            min_risk_per_share=0.01,
            max_risk_per_share=5.00,
            min_shares=0,
            max_notional=1e18,
        )
        session_date = resolve_session_date(provider, probe_symbol='AAPL')
        cand = build_orb_plan(provider, symbol, cfg, session_date=session_date)

        # Decide side
        side = (side_req if side_req in {'long','short'} else (cand.best_side or 'long'))
        # Gentle trend-aware override only when auto
        if side_req == 'auto' and trend_state:
            ts = str(trend_state)
            if ts in {'up','reclaim_vwap'}:
                side = 'long'
            elif ts in {'down','lost_vwap'}:
                side = 'short'

        # Structural stop candidates from real data
        tail = intraday.iloc[-15:] if len(intraday) >= 15 else intraday
        hi15 = float(_col(tail,'High','high').astype(float).max()) if not tail.empty else last_price
        lo15 = float(_col(tail,'Low','low').astype(float).min()) if not tail.empty else last_price
        buf = float(_buffer(last_price, cfg))

        if side == 'long':
            baseline_stop = cand.long_stop if getattr(cand,'long_stop',None) is not None else cand.stop
            stops = [x for x in [baseline_stop, vwap_last, lo15] if x is not None]
            structural_stop = float(min(stops)) - buf if stops else (last_price - 0.10)
            stop = structural_stop
            # Cap risk if huge
            risk = last_price - stop
            if risk > cfg.max_risk_per_share:
                stop = last_price - cfg.max_risk_per_share
            if (last_price - stop) < cfg.min_risk_per_share:
                stop = last_price - cfg.min_risk_per_share
            risk = max(0.0, last_price - stop)
            t1 = last_price + 1*risk
            t2 = last_price + 2*risk
            t3 = last_price + 3*risk
        else:
            baseline_stop = cand.short_stop if getattr(cand,'short_stop',None) is not None else cand.stop
            stops = [x for x in [baseline_stop, vwap_last, hi15] if x is not None]
            structural_stop = float(max(stops)) + buf if stops else (last_price + 0.10)
            stop = structural_stop
            risk = stop - last_price
            if risk > cfg.max_risk_per_share:
                stop = last_price + cfg.max_risk_per_share
            if (stop - last_price) < cfg.min_risk_per_share:
                stop = last_price + cfg.min_risk_per_share
            risk = max(0.0, stop - last_price)
            t1 = last_price - 1*risk
            t2 = last_price - 2*risk
            t3 = last_price - 3*risk

        # Chase distance — always compute, independent of ML model availability
        chase_r = None
        try:
            entry0 = float(cand.entry)
            stop0 = float(cand.stop)
            r0 = abs(entry0 - stop0)
            if r0 > 0:
                chase_r = (last_price - entry0) / r0 if side == 'long' else (entry0 - last_price) / r0
        except Exception:
            pass

        # ML score (base model probability) + a transparent adjustment for "chase" vs original ORB trigger
        ml_base = None
        ml_bucket = None
        ml_adjusted = None
        ml_reason = None
        ml_error = None
        try:
            from ml.orb_model_service import score_orb_symbol

            ml_out = score_orb_symbol(symbol, last_price=float(last_price), provider=provider)
            ml_bucket = ml_out.get('bucket')
            if ml_out.get('score') is not None:
                ml_base = float(ml_out['score'])

            # Transparent adjustment (NOT a new model): penalize heavy extension vs original trigger
            try:
                adj = ml_base
                if adj is not None and chase_r is not None:
                    import math as _math
                    adj = adj * _math.exp(-0.35 * max(0.0, float(chase_r) - 1.0))
                    if trend_state in {'up','reclaim_vwap'} and side == 'long':
                        adj *= 1.10
                    if trend_state in {'down','lost_vwap'} and side == 'short':
                        adj *= 1.10
                    if trend_state in {'up','reclaim_vwap'} and side == 'short':
                        adj *= 0.85
                    if trend_state in {'down','lost_vwap'} and side == 'long':
                        adj *= 0.85
                    ml_adjusted = float(max(0.0, min(1.0, adj)))
                    ml_reason = f"ml_base * exp(-0.35*max(0,chaseR-1)) with trend nudges; chaseR={chase_r:.2f}"
            except Exception:
                pass
        except Exception as e:
            ml_error = f"{type(e).__name__}: {e}"
            ml_reason = f"ml_error: {ml_error}"

        # Optional sentiment
        sentiment_score = None
        sentiment_error = None
        if want_sent:
            try:
                ys = SentimentService(provider='auto')
                sr = ys.get(symbol)
                sentiment_score = sr.score
            except Exception as e:
                sentiment_error = str(e)

        # --- Plan sanity + decision state ---------------------------------
        risk_per_share = (abs(last_price - stop) if stop is not None else None)
        risk_pct = None
        twoR_pct = None
        if risk_per_share is not None and last_price > 0:
            risk_pct = (float(risk_per_share) / float(last_price)) * 100.0
            twoR_pct = (abs(float(t2) - float(last_price)) / float(last_price)) * 100.0

        # Use the adjusted probability as the primary "P(2R/30m)" when available.
        entry_now_prob = None
        entry_now_bucket = None
        entry_now_error = None
        try:
            # Build live feature row consistent with ml.entry_now_dataset schema.
            from ml.entry_now_dataset import (
                EntryNowParams,
                _slice_rth_1m,
                _slice_pm_1m,
                _opening_range_5m_from_1m,
                _today_dollar_vol_so_far,
            )
            from scanner.indicators import avg_daily_volume
            import numpy as _np
            import pandas as _pd

            # Choose model by time (PM vs RTH)
            mp, entry_now_bucket = _select_entry_now_model_path()
            bundle = _load_entry_now_bundle(mp)
            feat_names = list(bundle.get('feature_names') or [])
            fillna_value = float(bundle.get('fillna_value', -999.0) or -999.0)
            model = bundle.get('model')

            # Daily history for avg20 dollar vol + RVOL (real provider)
            daily = provider.get_daily_history(symbol, period='180d').sort_index()

            # Session slice for feature computations
            day0 = _pd.Timestamp(intraday.index[-1]).tz_convert(_ET).normalize()
            slicer = _slice_pm_1m if entry_now_bucket == 'pm' else _slice_rth_1m
            sess_1m = slicer(intraday, day0)
            sess_1m = sess_1m.sort_index()
            if sess_1m is None or sess_1m.empty or len(sess_1m) < 20:
                raise RuntimeError('session_bars_insufficient')

            # Index i = last bar position in session
            i_now = len(sess_1m) - 1
            entry = float(sess_1m['Close'].astype(float).iloc[i_now])

            or_high, or_low = _opening_range_5m_from_1m(sess_1m)
            mid = (or_high + or_low) / 2.0
            or_range_pct = ((or_high - or_low) / max(1e-9, mid)) * 100.0
            dist_orh_pct = ((entry - or_high) / max(1e-9, or_high)) * 100.0
            dist_orl_pct = ((entry - or_low) / max(1e-9, or_low)) * 100.0

            dv_so_far = float(_today_dollar_vol_so_far(sess_1m, i_now))

            # Avoid lookahead: daily up to previous day
            day_utc = sess_1m.index[0].normalize()
            daily_prior = daily[daily.index.normalize() < day_utc]
            avg20_dollar_vol = _np.nan
            rvol_now = _np.nan
            if daily_prior is not None and len(daily_prior) >= 60:
                try:
                    avg20_vol = float(avg_daily_volume(daily_prior, window=20) or 0.0)
                    prev_close = float(daily_prior['Close'].astype(float).iloc[-1])
                    avg20_dollar_vol = float(avg20_vol * prev_close) if (avg20_vol > 0 and prev_close > 0) else _np.nan
                    today_vol_so_far = float(sess_1m['Volume'].astype(float).iloc[: i_now + 1].sum())
                    frac = max(1e-6, (i_now + 1) / float(len(sess_1m)))
                    rvol_now = (today_vol_so_far / max(1e-9, avg20_vol * frac)) if avg20_vol > 0 else _np.nan
                except Exception:
                    pass

            # Trend-state one-hot
            ts = str(trend_state or '')
            row = {
                'side_long': 1.0 if side == 'long' else 0.0,
                'entry': float(entry),
                'stop_dist_pct': (abs(float(entry) - float(stop)) / max(1e-9, float(entry))) * 100.0,
                'vwap_delta_pct': float(vwap_delta_pct) if vwap_delta_pct is not None else _np.nan,
                'trend_slope_pct': float(trend_slope_pct) if trend_slope_pct is not None else _np.nan,
                'trend_state_up': 1.0 if ts in ('up','reclaim_vwap') else 0.0,
                'trend_state_down': 1.0 if ts in ('down','lost_vwap') else 0.0,
                'trend_state_chop': 1.0 if ts == 'chop' else 0.0,
                'or_range_pct': float(or_range_pct) if or_range_pct is not None else _np.nan,
                'dist_orh_pct': float(dist_orh_pct) if dist_orh_pct is not None else _np.nan,
                'dist_orl_pct': float(dist_orl_pct) if dist_orl_pct is not None else _np.nan,
                'today_dollar_vol_so_far': float(dv_so_far) if dv_so_far is not None else _np.nan,
                'avg20_dollar_vol': float(avg20_dollar_vol),
                'rvol_now': float(rvol_now),
                'minutes_since_open': float(i_now),
            }

            X = _pd.DataFrame([row])
            # Align to model feature order
            if feat_names:
                for c in feat_names:
                    if c not in X.columns:
                        X[c] = _np.nan
                X = X[feat_names]
            X = X.replace([_np.inf, -_np.inf], _np.nan).fillna(fillna_value)

            if model is None or not hasattr(model, 'predict_proba'):
                raise RuntimeError('entry_now_model_invalid')
            entry_now_prob = float(model.predict_proba(X)[0,1])
        except Exception as _e:
            entry_now_error = f"{type(_e).__name__}: {_e}"

        
        p_2r_30m = entry_now_prob if entry_now_prob is not None else (ml_adjusted if ml_adjusted is not None else ml_base)

        _plan_entry = None
        try:
            _plan_entry = float(cand.entry) if getattr(cand, 'entry', None) is not None else None
        except Exception:
            pass

        state, notes, hint_line = _entry_now_plan_state(
            side=side,
            last_price=last_price,
            entry=_plan_entry,
            stop=stop,
            t2=t2,
            p_2r_30m=p_2r_30m,
            chase_r=chase_r,
            vwap_delta_pct=vwap_delta_pct,
            trend_state=trend_state,
        )

        return jsonify(
            ok=True,
            symbol=symbol,
            last_price=last_price,
            side=side,
            entry_now=last_price,
            stop=stop,
            risk_per_share=risk_per_share,
            risk_pct=risk_pct,
            target_1r=t1,
            target_2r=t2,
            target_3r=t3,
            twoR_pct=twoR_pct,
            vwap_last=vwap_last,
            vwap_delta_pct=vwap_delta_pct,
            trend_state=trend_state,
            trend_slope_pct=trend_slope_pct,
            structural_context={
                'hi15': hi15,
                'lo15': lo15,
                'baseline_stop': (baseline_stop if 'baseline_stop' in locals() else None),
            },
            ml_score=ml_base,
            ml_adjusted=ml_adjusted,
            model_bucket=ml_bucket,
            ml_note=ml_reason,
            ml_error=ml_error,
            chase_r=chase_r,
            p_2r_30m=p_2r_30m,
            plan_state=state,
            plan_notes=notes,
            trend_error=trend_error,
            sentiment_score=sentiment_score,
            sentiment_error=sentiment_error,
        )
    except Exception as e:
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}", symbol=symbol)


def load_symbols_from_file(path: str) -> list[str]:
    if not os.path.exists(path):
        raise RuntimeError(f"symbols file not found: {path}")
    syms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s and not s.startswith("#"):
                syms.append(s)
    if not syms:
        raise RuntimeError("symbols.txt is empty")
    return syms


def _scan_worker(jid: str):
    """Background scan runner.

    Tenants:
    - Uses real provider/universe data (no placeholders).
    - If upstream fails, we record the real failure (per-chunk) and continue where possible.
    - Must never hang forever on a single chunk: provider/network timeouts are enforced in providers layer.
    """
    try:
        job = _get_job(jid) or {}
        params = job.get("params") or {}

        universe = params.get("universe", "nasdaq")
        max_symbols = _param_int(params, "max_symbols", 400)
        limit = _safe_int(params.get("top_n", params.get("limit", 25)), 25)
        offset = _param_int(params, "offset", 0)
        mode = params.get("scan_mode", params.get("mode", "until"))
        strategy = params.get("strategy", params.get("scan_strategy", "orb"))
        # Parabolic must sweep the full Nasdaq — override any low max_symbols at the source
        if str(strategy).strip().lower() in {"parabolic", "para", "parabolic_watch"}:
            max_symbols = max(max_symbols, 5000)
        exec_style = params.get("exec_style") or params.get("execution") or params.get("execStyle") or "retest"
        range_window_min = _param_int(params, "range_window_min", 60)
        range_band_k = _param_float(params, "range_band_k", 2.0)
        rr_touch_lookback_min = _param_int(params, "rr_touch_lookback_min", 45)
        rr_stop_sigma_mult = _param_float(params, "rr_stop_sigma_mult", 0.75)
        rr_min_risk_per_share = _param_float(params, "rr_min_risk_per_share", 0.01)
        rr_max_risk_per_share = _param_float(params, "rr_max_risk_per_share", 5.00)
        rt_lookback_min     = _param_int(params,   "rt_lookback_min",     45)
        rt_min_touches      = _param_int(params,   "rt_min_touches",      2)
        rt_entry_zone_pct   = _param_float(params, "rt_entry_zone_pct",   0.35)
        rt_stop_atr_mult    = _param_float(params, "rt_stop_atr_mult",    0.50)
        rt_min_rr           = _param_float(params, "rt_min_rr",           1.5)
        rt_min_range_pct    = _param_float(params, "rt_min_range_pct",    0.50)
        rt_max_range_pct    = _param_float(params, "rt_max_range_pct",    4.0)
        rt_slope_threshold  = _param_float(params, "rt_slope_threshold",  0.05)
        rt_vwap_min_gap_pct = _param_float(params, "rt_vwap_min_gap_pct", 0.30)

        # Optional explicit symbol list (comma/space/newline separated). If present, overrides universe slicing.
        symbols_raw = params.get("symbols") or params.get("tickers") or ""
        symbols_list = []
        if isinstance(symbols_raw, str) and symbols_raw.strip():
            import re as _re
            symbols_list = [s.strip().upper() for s in _re.split(r"[\s,]+", symbols_raw.strip()) if s.strip()]

        # Filters (balanced defaults)
        min_rvol = _param_float(params, "min_rvol", 0.5)
        _raw_today = _safe_float(params.get("min_today_dollar_vol", params.get("min_today_dvol")))
        min_today_dollar_vol = 2_000_000.0 if _raw_today is None else _raw_today
        _raw_avg20 = _safe_float(params.get("min_avg20_dollar_vol", params.get("min_avg20_dvol")))
        # Parabolic plays are small-caps — their avg20 dollar vol is typically $250K-750K.
        # ORB/other strategies need $1M (liquid tape). Don't apply the ORB floor to parabolic.
        _is_para_strategy = str(strategy).strip().lower() in {"parabolic", "para", "parabolic_watch"}
        min_avg20_dollar_vol = (_raw_avg20 if _raw_avg20 is not None
                                else 250_000.0 if _is_para_strategy
                                else 1_000_000.0)
        min_price = _param_float(params, "min_price", 1.0)
        # Default max_price should match ORBConfig() default unless caller overrides
        max_price = _param_float(params, "max_price", 30.0)
        min_or_range_pct = _param_float(params, "min_or_range_pct", 0.6)
        max_or_range_pct = _param_float(params, "max_or_range_pct", 6.0)
        # Parabolic-specific filter params (also used by gap_and_go)
        min_pm_move_pct = _param_float(params, "min_pm_move_pct", 5.0)
        max_pm_move_pct = _param_float(params, "max_pm_move_pct", 500.0)
        min_pm_bars     = int(_param_float(params, "min_pm_bars", 3))

        # Optional scoring knobs
        # If caller does not specify use_ml, default ON when a real model file exists.
        use_ml_raw = params.get("use_ml", None)
        if use_ml_raw is None:
            models_dir = _PROJECT_ROOT / "models"
            has_model = any((models_dir / p).exists() for p in (
                "model_b_outlier.pkl",
                "model_a_liquid.pkl",
            ))
            use_ml = bool(has_model)
        else:
            use_ml = str(use_ml_raw).strip().lower() in {"1", "true", "yes", "on"}

        use_sentiment = str(params.get("use_sentiment", "false")).strip().lower() in {"1", "true", "yes", "on"}
        sentiment_provider = str(params.get("sentiment_provider", os.getenv("ORB_SENTIMENT_PROVIDER", "rss"))).strip().lower()
        try:
            sentiment_alpha = _safe_float(params.get("sentiment_alpha", os.getenv("ORB_SENTIMENT_ALPHA", "0.10"))) or 0.10
        except Exception:
            sentiment_alpha = 0.10
        use_catalyst = str(params.get("use_catalyst", os.getenv("ORB_USE_CATALYST", "false"))).strip().lower() in {"1", "true", "yes", "on"}
        try:
            catalyst_alpha = _safe_float(params.get("catalyst_alpha", os.getenv("ORB_CATALYST_ALPHA", "0.08"))) or 0.08
        except Exception:
            catalyst_alpha = 0.08
        try:
            catalyst_topn = _safe_int(params.get("catalyst_topn"), None) if params.get("catalyst_topn") is not None else None
        except Exception:
            catalyst_topn = None
        try:
            catalyst_lookback_hours = _safe_int(params.get("catalyst_lookback_hours", os.getenv("ORB_CATALYST_LOOKBACK_HOURS", "72")), 72)
        except Exception:
            catalyst_lookback_hours = 72
        regime_profile = str(params.get("regime_profile", os.getenv("ORB_REGIME_PROFILE", "auto"))).strip().lower() or "auto"

        # Build base config
        strategy_key = str(strategy).strip().lower()
        is_rr_scan = strategy_key in {"range_reversion", "rr", "range"}
        is_rt_scan = strategy_key in {"range_trap", "rt", "trap"}
        cfg = ORBConfig(
            min_price=min_price,
            max_price=max_price,
            min_today_dollar_vol=min_today_dollar_vol,
            min_avg20_dollar_vol=min_avg20_dollar_vol,
            min_rvol=min_rvol,
            min_or_range_pct=min_or_range_pct,
            max_or_range_pct=max_or_range_pct,
            min_risk_per_share=(rr_min_risk_per_share if (is_rr_scan or is_rt_scan) else ORBConfig.min_risk_per_share),
            max_risk_per_share=(rr_max_risk_per_share if (is_rr_scan or is_rt_scan) else ORBConfig.max_risk_per_share),
        )

        provider = _ALPACA_PROVIDER

        # Self-seeding strategies: ignore the universe entirely — the scan function
        # fetches its own symbol list from Alpaca (Market Movers / Most Actives).
        # Parabolic always self-seeds — it runs its own full-Nasdaq snapshot loop internally.
        # All other strategies respect the universe selection: if universe="nasdaq", they
        # receive the full Nasdaq symbol list and do NOT self-seed from market movers.
        _self_seeding = str(strategy).strip().lower() in {"parabolic", "para", "parabolic_watch"}
        if _self_seeding and not symbols_list:
            slice_syms = []
            universe_size = 0
            start = 0
            end = 0
            chunks_total = 1

        # Determine symbols to scan.
        elif symbols_list:
            # Explicit symbols requested (proof test / focused scan)
            slice_syms = symbols_list[:max_symbols] if max_symbols > 0 else symbols_list
            universe_size = len(symbols_list)
            start = 0
            end = len(slice_syms)
        else:
            # Universe symbols (cached by UniverseConfig)
            syms_all = fetch_us_equity_symbols(UniverseConfig())
            universe_size = len(syms_all)
            if universe_size == 0:
                raise RuntimeError(
                    "Universe returned 0 symbols. This indicates a real upstream/universe fetch failure "
                    "(blocked network, DNS, or a corrupt/empty cache file)."
                )

            # Slice for this scan run
            start = max(0, offset)
            end = min(universe_size, start + max_symbols)
            slice_syms = syms_all[start:end]

        chunk_size = _param_int(params, "chunk_size", 100)
        chunk_size = max(25, min(250, chunk_size))
        chunks_total = (len(slice_syms) + chunk_size - 1) // chunk_size
        if _self_seeding and not symbols_list:
            chunks_total = 1  # run exactly once with [] so the scan function auto-seeds

        # Per-chunk timeout so one bad/network-stuck chunk cannot hang the whole scan.
        # Tenant-compliant: on timeout we record the real failure and continue.
        chunk_timeout_s = _safe_float(params.get("chunk_timeout_s", os.getenv("ORB_CHUNK_TIMEOUT_S", "30"))) or 30.0
        if chunk_timeout_s <= 0:
            chunk_timeout_s = 30.0
        if _self_seeding and not symbols_list:
            chunk_timeout_s = max(chunk_timeout_s, 480.0)  # self-seed: full-Nasdaq snapshot loop (200s) + bar analysis

        prefilter_sum: dict = {}
        reject_sum: dict = {}
        data_fail_sum: dict = {}
        all_candidates: list = []
        all_seed_candidates: list = []
        rejected_candidates_all: list = []
        candidates_total_est = 0
        seed_candidates_total_est = 0
        tradable_now_total_est = 0
        rejected_total_est = 0
        shortlisted_total = 0
        scanned = 0
        snapshot_passed_total = 0
        scan_date_used = None
        last_failure_samples = []
        chunk_errors = []  # keep a few real exceptions for debugging
        chunk_timings = []
        thresholds_used: dict[str, Any] = {
            "min_price": float(min_price),
            "max_price": float(max_price),
            "min_today_dollar_vol": float(min_today_dollar_vol),
            "min_avg20_dollar_vol": float(min_avg20_dollar_vol),
            "min_rvol": float(min_rvol),
            "min_or_range_pct": float(min_or_range_pct),
            "max_or_range_pct": float(max_or_range_pct),
            "limit": int(limit),
            "max_symbols": int(max_symbols),
            "offset": int(offset),
            "chunk_size": int(chunk_size),
            "strategy": str(strategy),
            "exec_style": str(exec_style),
            "range_window_min": int(range_window_min),
            "range_band_k": float(range_band_k),
            "rr_touch_lookback_min": int(rr_touch_lookback_min),
            "rr_stop_sigma_mult": float(rr_stop_sigma_mult),
            "rr_min_risk_per_share": float(rr_min_risk_per_share),
            "rr_max_risk_per_share": float(rr_max_risk_per_share),
            "rt_lookback_min":       int(rt_lookback_min),
            "rt_min_touches":        int(rt_min_touches),
            "rt_entry_zone_pct":     float(rt_entry_zone_pct),
            "rt_stop_atr_mult":      float(rt_stop_atr_mult),
            "rt_min_rr":             float(rt_min_rr),
            "rt_min_range_pct":      float(rt_min_range_pct),
            "rt_max_range_pct":      float(rt_max_range_pct),
            "rt_slope_threshold":    float(rt_slope_threshold),
            "rt_vwap_min_gap_pct":   float(rt_vwap_min_gap_pct),
            "use_ml": bool(use_ml),
            "use_sentiment": bool(use_sentiment),
            "use_catalyst": bool(use_catalyst),
            "auto_monitor": not _disable_auto_monitor(params),
        }
        prefilter_samples: list[dict[str, Any]] = []
        failure_samples_by_code: dict[str, list[dict[str, Any]]] = {}

        def _sum_into(dst: dict, src):
            """Accumulate counts.

            scan_symbols() returns dict counts for prefilter/rejects; data_failures can be a list of failures.
            """
            if not src:
                return
            if isinstance(src, list):
                dst["count"] = dst.get("count", 0) + len(src)
                for it in src:
                    if isinstance(it, dict):
                        stg = it.get("stage") or "unknown"
                        key = f"stage_{stg}"
                        dst[key] = dst.get(key, 0) + 1
                return
            for k, v in (src or {}).items():
                try:
                    dst[k] = dst.get(k, 0) + int(v)
                except Exception:
                    dst[k] = v

        _set_job(
            jid,
            progress={"scanned": 0, "chunks_done": 0, "chunks_total": chunks_total, "offset": start, "end_offset": end, "stage": "prefilter"},
        )

        for ci in range(chunks_total):
            chunk = slice_syms[ci * chunk_size : (ci + 1) * chunk_size]
            t0 = time.time()

            try:
                # ask each chunk for more than final limit so we can merge across chunks
                # Run scan_symbols with a hard timeout so a single chunk cannot stall the entire scan.
                _set_job(
                    jid,
                    progress={
                        "scanned": scanned,
                        "chunks_done": ci,
                        "chunks_total": chunks_total,
                        "offset": start,
                        "end_offset": end,
                        "current_chunk": ci,
                        "current_chunk_size": len(chunk),
                        "chunk_started_at": time.time(),
                        "chunk_timeout_s": chunk_timeout_s,
                        "stage": "scan_symbols",
                    },
                    updated_at=time.time(),
                )
                                # Run scan_symbols in a daemon thread so we can enforce a timeout
                # without blocking on ThreadPoolExecutor shutdown (which would wait for completion).
                import threading
                import queue as _queue

                q: "_queue.Queue[object]" = _queue.Queue(maxsize=1)

                def _runner():
                    try:
                        q.put(scan_symbols(
                            chunk,
                            cfg,
                            limit=max(limit, 50),
                            provider=provider,
                            use_ml=use_ml,
                            use_sentiment=use_sentiment,
                            sentiment_provider=sentiment_provider,
                            sentiment_alpha=sentiment_alpha,
                            scan_mode=mode,
                            use_catalyst=use_catalyst,
                            catalyst_alpha=catalyst_alpha,
                            catalyst_topn=catalyst_topn,
                            catalyst_lookback_hours=catalyst_lookback_hours,
                            regime_profile=regime_profile,
                            strategy=strategy,
                            range_window_min=range_window_min,
                            range_band_k=range_band_k,
                            rr_touch_lookback_min=rr_touch_lookback_min,
                            rr_stop_sigma_mult=rr_stop_sigma_mult,
                            rt_lookback_min=rt_lookback_min,
                            rt_min_touches=rt_min_touches,
                            rt_entry_zone_pct=rt_entry_zone_pct,
                            rt_stop_atr_mult=rt_stop_atr_mult,
                            rt_min_rr=rt_min_rr,
                            rt_min_range_pct=rt_min_range_pct,
                            rt_max_range_pct=rt_max_range_pct,
                            rt_slope_threshold=rt_slope_threshold,
                            rt_vwap_min_gap_pct=rt_vwap_min_gap_pct,
                            exec_style=exec_style,
                            long_only=_is_truthy(params.get("long_only", "0")),
                            short_only=_is_truthy(params.get("short_only", "0")),
                            min_grade_enabled=_is_truthy(params.get("min_grade_enabled", "0")),
                            min_grade=str(params.get("min_grade", "B")).strip().upper() or "B",
                            min_combined_enabled=_is_truthy(params.get("min_combined_enabled", "0")),
                            min_combined_score=_param_float(params, "min_combined_score", 0.40),
                            no_chop_enabled=_is_truthy(params.get("no_chop_enabled", "0")),
                            min_vwap_enabled=_is_truthy(params.get("min_vwap_enabled", "0")),
                            min_pct_over_vwap=_param_float(params, "min_pct_over_vwap", 1.0),
                            expansion_min=_param_float(params, "expansion_min", 2.0),
                            min_day_move_pct=_param_float(params, "min_day_move_pct", 3.0),
                            max_day_move_pct=_param_float(params, "max_day_move_pct", 50.0),
                            min_close_vs_range=_param_float(params, "min_close_vs_range", 0.60),
                            min_price=min_price,
                            max_price=max_price,
                            min_rvol=min_rvol,
                            min_avg20_dollar_vol=min_avg20_dollar_vol,
                            min_today_dollar_vol=min_today_dollar_vol,
                            min_pm_move_pct=min_pm_move_pct,
                            max_pm_move_pct=max_pm_move_pct,
                            min_pm_bars=min_pm_bars,
                        ))
                    except Exception as _e:
                        q.put(_e)

                th = threading.Thread(target=_runner, daemon=True)
                th.start()
                th.join(timeout=chunk_timeout_s)
                if th.is_alive():
                    raise concurrent.futures.TimeoutError()
                out_obj = q.get() if not q.empty() else RuntimeError("scan_symbols returned no result")
                if isinstance(out_obj, Exception):
                    raise out_obj
                out = out_obj
                dt = time.time() - t0
                if len(chunk_timings) < 5:
                    chunk_timings.append({"chunk_index": ci, "seconds": round(dt, 3), "n_symbols": len(chunk)})
            except concurrent.futures.TimeoutError as e:
                # Record a real timeout for this chunk and continue.
                if len(chunk_errors) < 5:
                    chunk_errors.append({
                        "chunk_index": ci,
                        "symbols": chunk[:10],
                        "error": f"TimeoutError: chunk exceeded {chunk_timeout_s:.1f}s",
                    })
                out = {
                    "candidates": [],
                    "candidates_total": 0,
                    "shortlisted": 0,
                    "prefilter_counts": {},
                    "reject_counts": {},
                    "data_failures": [{"stage": "chunk", "error": f"TimeoutError: chunk exceeded {chunk_timeout_s:.1f}s", "chunk_index": ci}],
                    "debug": {"failure_samples": [{"stage": "chunk", "error": f"TimeoutError: chunk exceeded {chunk_timeout_s:.1f}s", "traceback": ""}]},
                }
                dt = time.time() - t0
            except Exception as e:
                # Record a real failure for this chunk and continue (no placeholders).
                tb = traceback.format_exc()
                if len(chunk_errors) < 5:
                    chunk_errors.append({
                        "chunk_index": ci,
                        "symbols": chunk[:10],
                        "error": f"{type(e).__name__}: {e}",
                    })
                out = {
                    "candidates": [],
                    "candidates_total": 0,
                    "shortlisted": 0,
                    "prefilter_counts": {},
                    "reject_counts": {},
                    "data_failures": [{"stage": "chunk", "error": f"{type(e).__name__}: {e}", "chunk_index": ci}],
                    "debug": {"failure_samples": [{"stage": "chunk", "error": f"{type(e).__name__}: {e}", "traceback": tb[:2000]}]},
                }

            if scan_date_used is None:
                sd = out.get("scan_date") or (out.get("debug") or {}).get("session_date_used")
                if isinstance(sd, str) and sd.strip():
                    scan_date_used = sd.strip()
            # capture a few real failures for debugging
            try:
                fs = (out.get("debug") or {}).get("failure_samples")
                if isinstance(fs, list) and fs:
                    last_failure_samples = fs[:5]
            except Exception:
                pass

            scanned += len(chunk)
            # For self-seeding strategies, chunk=[] — use universe_size (full snapshot checked)
            # if available, else fall back to candidates+rejected as a proxy.
            if _self_seeding and not symbols_list and len(chunk) == 0:
                universe_checked = int(out.get("universe_size") or 0)
                if universe_checked > 0:
                    scanned += universe_checked
                    universe_size = max(universe_size, universe_checked)  # update for final result
                else:
                    scanned += int(out.get("candidates_total") or 0) + int(out.get("rejected_total") or 0)
            snapshot_passed_total += int(out.get("snapshot_passed") or 0)
            all_candidates.extend(out.get("candidates") or [])
            all_seed_candidates.extend(out.get("seed_candidates") or [])
            for _rej in (out.get("rejected_candidates") or []):
                if isinstance(_rej, dict) and len(rejected_candidates_all) < max(100, limit * 5):
                    rejected_candidates_all.append(_rej)
            candidates_total_est += int(out.get("candidates_total") or 0)
            seed_candidates_total_est += int(out.get("seed_candidates_total") or 0)
            tradable_now_total_est += int(out.get("tradable_now_total") or 0)
            rejected_total_est += int(out.get("rejected_total") or 0)
            shortlisted_total += int(out.get("shortlisted") or 0)

            _sum_into(prefilter_sum, out.get("prefilter_counts") or {})
            _sum_into(reject_sum, out.get("reject_counts") or {})
            _sum_into(data_fail_sum, out.get("data_failures") or {})
            if not thresholds_used and isinstance(out.get("thresholds_used"), dict):
                thresholds_used = dict(out.get("thresholds_used") or {})
            for item in (out.get("prefilter_samples") or []):
                if isinstance(item, dict) and len(prefilter_samples) < 50:
                    prefilter_samples.append(item)
            dbg = out.get("debug") or {}
            fsbc = dbg.get("failure_samples_by_code") or {}
            if isinstance(fsbc, dict):
                for code, items in fsbc.items():
                    bucket = failure_samples_by_code.setdefault(code, [])
                    if isinstance(items, list):
                        for it in items:
                            if isinstance(it, dict) and len(bucket) < 5:
                                bucket.append(it)

            _set_job(
                jid,
                progress={
                    "scanned": scanned,
                    "chunks_done": ci + 1,
                    "chunks_total": chunks_total,
                    "offset": start,
                    "end_offset": end,
                },
                partial_result={
                    "provider": getattr(provider, "name", "alpaca"),
                    "strategy": params.get("strategy"),
                    "scan_date": scan_date_used,
                    "session_date_used": scan_date_used,
                    "prefilter_counts": prefilter_sum,
                    "prefilter_samples": prefilter_samples,
                    "thresholds_used": thresholds_used,
                    "reject_counts": reject_sum,
                    "failure_samples_by_code": failure_samples_by_code,
                    "data_failures": data_fail_sum,
                    "shortlisted": shortlisted_total,
                    "rejected_total": rejected_total_est,
                    "rejected_candidates": rejected_candidates_all[:limit],
                    "scanned": scanned,
                    "chunks": chunks_total,
                    "end_offset": end,
                    "universe_size": universe_size,
                    "mode": mode,
                    "debug": {
                        "strategy": params.get("strategy"),
                        "session_date_used": scan_date_used,
                        "failure_samples": last_failure_samples,
                        "failure_samples_by_code": failure_samples_by_code,
                        "chunk_errors": chunk_errors,
                        "chunk_timings": chunk_timings,
                    },
                },
            )

        # Global top-N across all chunks
        def _score(c):
            if isinstance(c, dict):
                for key in ("combined_score", "score", "ml_score"):
                    v = c.get(key)
                    if v is not None:
                        try:
                            return float(v)
                        except Exception:
                            pass
            return 0.0

        try:
            all_candidates.sort(key=_score, reverse=True)
        except Exception:
            # If something is non-comparable, keep original order (real data only)
            pass
        try:
            all_seed_candidates.sort(key=_score, reverse=True)
        except Exception:
            pass

        top = all_candidates[:limit]
        top_seed = all_seed_candidates[:limit]

        # Ensure candidates are JSON-serializable (Flask jsonify cannot handle dataclass objects).
        def _cand_to_dict(c):
            try:
                import dataclasses as _dc
                if _dc.is_dataclass(c):
                    d = _dc.asdict(c)
                elif isinstance(c, dict):
                    d = dict(c)
                else:
                    try:
                        d = dict(c)
                    except Exception:
                        try:
                            d = vars(c)
                        except Exception:
                            d = {"value": str(c)}
            except Exception:
                d = {"value": str(c)}
            # Alias ScanCandidate field names → Candidate field names for JS rendering
            if "best_direction" in d and "best_side" not in d:
                d["best_side"] = str(d["best_direction"] or "").lower()
            if "dollar_volume" in d and "today_dollar_vol" not in d:
                d["today_dollar_vol"] = d["dollar_volume"]
            if "or_pct" in d and "or_range_pct" not in d:
                d["or_range_pct"] = d["or_pct"]
            if "long_trigger" in d and "long_entry" not in d:
                d["long_entry"] = d["long_trigger"]
            if "long_target_2r" in d and "long_2r" not in d:
                d["long_2r"] = d["long_target_2r"]
            if "short_trigger" in d and "short_entry" not in d:
                d["short_entry"] = d["short_trigger"]
            if "short_target_2r" in d and "short_2r" not in d:
                d["short_2r"] = d["short_target_2r"]
            return d

        top = [_normalize_target_rr(_cand_to_dict(c)) for c in top]
        top_seed = [_normalize_target_rr(_cand_to_dict(c)) for c in top_seed]
        for _row in top:
            if isinstance(_row, dict):
                _row["strategy"] = _row.get("strategy") or params.get("strategy")
                _row["provider"] = _row.get("provider") or getattr(provider, "name", "alpaca")
        for _row in top_seed:
            if isinstance(_row, dict):
                _row["strategy"] = _row.get("strategy") or params.get("strategy")
                _row["provider"] = _row.get("provider") or getattr(provider, "name", "alpaca")

        # ── Post-scan KTT quality gate ────────────────────────────────────────
        # Run live KTT grades on all top candidates in parallel immediately
        # after the scan. Only B+(score≥56) survive into top; D-grade stocks
        # are demoted to rejected so the scanner only surfaces real trades.
        try:
            from utils.know_the_trade import analyze as _ktt_scan_analyze
            from concurrent.futures import ThreadPoolExecutor as _KttPool, as_completed as _ktt_as_completed
            import dateutil.parser as _ktt_dp

            _ktt_halt_map: dict[str, int] = {}
            try:
                if _STREAM is not None:
                    for _ktt_e in (_STREAM.recent_halt_resume_events(max_age_sec=86400) or []):
                        _ktt_s = str(_ktt_e.get('symbol', '')).upper()
                        if _ktt_e.get('halt_status', '').lower() in ('halted', 'trading_halt', 'h'):
                            _ktt_halt_map[_ktt_s] = _ktt_halt_map.get(_ktt_s, 0) + 1
            except Exception:
                pass

            def _ktt_grade_scan_cand(c: dict) -> dict:
                _sym = str(c.get('symbol') or '').upper()
                _side = str(c.get('best_side') or c.get('side') or 'long').lower()
                _entry  = _safe_float(c.get('long_entry')  if _side == 'long' else c.get('short_entry'))
                _stop   = _safe_float(c.get('long_stop')   if _side == 'long' else c.get('short_stop'))
                _target = _safe_float(c.get('long_2r')     if _side == 'long' else c.get('short_2r'))
                _age = None
                try:
                    _ts = c.get('scan_ts')
                    if _ts:
                        _scanned = _ktt_dp.parse(str(_ts))
                        if _scanned.tzinfo is None:
                            _scanned = _scanned.replace(tzinfo=timezone.utc)
                        _age = (datetime.now(timezone.utc) - _scanned).total_seconds() / 3600
                except Exception:
                    pass
                _status = 'triggered' if (
                    (_side == 'long' and c.get('long_triggered')) or
                    (_side == 'short' and c.get('short_triggered'))
                ) else None
                try:
                    _res = _ktt_scan_analyze(
                        symbol=_sym, provider=provider,
                        entry=_entry, stop=_stop, target=_target, side=_side,
                        ml_score=_safe_float(c.get('ml_score')),
                        catalyst_headline=c.get('news_headline') or None,
                        catalyst_age_hours=_safe_float(c.get('news_age_hours')),
                        rvol_hint=_safe_float(c.get('rvol')),
                        halt_count=_ktt_halt_map.get(_sym, 0),
                        setup_age_hours=_age,
                        setup_status=_status,
                        vwap=_safe_float(c.get('vwap_last')),
                    )
                    _g  = str(_res.get('grade') or 'D')
                    _sc = int(_res.get('score') or 0)
                    c['ktt_grade'] = _g
                    c['ktt_score'] = _sc
                    c['ktt_color'] = {'A': '#4ade80', 'B': '#fbbf24', 'C': '#f97316', 'D': '#f87171'}.get(_g, '#94a3b8')
                except Exception:
                    c.setdefault('ktt_grade', '?')
                    c.setdefault('ktt_score', 0)
                    c.setdefault('ktt_color', '#94a3b8')
                return c

            if top:
                with _KttPool(max_workers=min(len(top), 16)) as _ktt_ex:
                    _ktt_futs = {_ktt_ex.submit(_ktt_grade_scan_cand, c): c for c in top}
                    top = [_f.result() for _f in _ktt_as_completed(_ktt_futs)]
                top.sort(key=lambda _c: _c.get('ktt_score') or 0, reverse=True)
                _ktt_demoted = [_c for _c in top if (_c.get('ktt_score') or 0) < 56]
                top = [_c for _c in top if (_c.get('ktt_score') or 0) >= 56]
                rejected_candidates_all = _ktt_demoted + rejected_candidates_all
                rejected_candidates_all = rejected_candidates_all[:max(limit * 3, 100)]

                # Auto-add all surviving B+ candidates to the desk watchlist
                _today_str = datetime.now(_ET).strftime('%Y-%m-%d')
                for _wc in top:
                    try:
                        _wsym  = str(_wc.get('symbol') or '').upper()
                        _wside = str(_wc.get('best_side') or _wc.get('side') or 'long').lower()
                        _wentry = _safe_float(_wc.get('long_entry') if _wside == 'long' else _wc.get('short_entry'))
                        _wstop  = _safe_float(_wc.get('long_stop')  if _wside == 'long' else _wc.get('short_stop'))
                        _wtgt   = _safe_float(_wc.get('long_2r')    if _wside == 'long' else _wc.get('short_2r'))
                        if not _wsym:
                            continue
                        _RUNTIME_STORE.desk_watchlist_set(
                            _wsym,
                            side=_wside,
                            trigger_price=_wentry,
                            stop_price=_wstop,
                            target_price=_wtgt,
                            notes=f"auto-added by scanner — KTT {_wc.get('ktt_grade','?')} {_wc.get('ktt_score',0)}",
                            session_date=_today_str,
                        )
                    except Exception:
                        pass
        except Exception:
            pass

        primary_candidates, primary_mode, primary_message = select_primary_candidates(
            {
                "candidates": top,
                "seed_candidates": top_seed,
                "candidates_total": candidates_total_est,
                "trade_ready_total": candidates_total_est,
                "seed_candidates_total": seed_candidates_total_est,
                "rejected_total": rejected_total_est,
                "shortlisted": shortlisted_total,
                "prefilter_counts": prefilter_sum,
                "reject_counts": reject_sum,
            },
            limit=limit,
        )
        zero_result_diagnostics = build_zero_result_diagnostics(
            {
                "candidates": top,
                "seed_candidates": top_seed,
                "candidates_total": candidates_total_est,
                "trade_ready_total": candidates_total_est,
                "seed_candidates_total": seed_candidates_total_est,
                "rejected_total": rejected_total_est,
                "shortlisted": shortlisted_total,
                "prefilter_counts": prefilter_sum,
                "reject_counts": reject_sum,
            },
            limit=limit,
        )

        result = {
            "provider": getattr(provider, "name", "alpaca"),
            "strategy": params.get("strategy"),
            "count": len(top),
            "candidates_total": candidates_total_est,
            "seed_candidates_total": seed_candidates_total_est,
            "tradable_now_total": tradable_now_total_est,
            "rejected_total": rejected_total_est,
            "candidates": top,
            "seed_candidates": top_seed,
            "primary_candidates": primary_candidates,
            "primary_mode": primary_mode,
            "primary_message": primary_message,
            "zero_result_diagnostics": zero_result_diagnostics,
            "rejected_candidates": rejected_candidates_all[:limit],
            "prefilter_counts": prefilter_sum,
            "prefilter_samples": prefilter_samples,
            "thresholds_used": thresholds_used,
            "reject_counts": reject_sum,
            "failure_samples_by_code": failure_samples_by_code,
            "data_failures": data_fail_sum,
            "shortlisted": shortlisted_total,
            "snapshot_passed": snapshot_passed_total,
            "scanned": scanned,
            "chunks": chunks_total,
            "end_offset": end,
            "universe_size": universe_size,
            "mode": mode,
            "scan_date": scan_date_used,
            "session_date_used": scan_date_used,
            "seed_symbols": slice_syms,
            "debug": {"strategy": params.get("strategy"), "session_date_used": scan_date_used, "failure_samples": last_failure_samples, "failure_samples_by_code": failure_samples_by_code, "chunk_errors": chunk_errors, "chunk_timings": chunk_timings},
        }
        _set_job(
            jid,
            status="done",
            result=result,
            partial_result=result,
            progress={
                "scanned": scanned,
                "chunks_done": chunks_total,
                "chunks_total": chunks_total,
                "offset": start,
                "end_offset": end,
                "stage": "done",
            },
            updated_at=time.time(),
        )
        try:
            _RUNTIME_STORE.scan_job_save(jid, result)
        except Exception:
            pass
        if not _disable_auto_monitor(params):
            try:
                _set_job(jid, progress={"scanned": scanned, "chunks_done": chunks_total, "chunks_total": chunks_total, "offset": start, "end_offset": end, "stage": "monitor_handoff"}, updated_at=time.time())
                monitor_top_n = _param_int(params, "monitor_top_n", min(25, max(5, limit)))
                stream_cache = _ensure_stream(start=True, require=True)
                monitor_candidates = list(result.get("seed_candidates") or top)
                sess = _MONITOR.start_from_scan_candidates(
                    job_id=jid,
                    candidates=monitor_candidates,
                    top_n=monitor_top_n,
                    feed=(os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex",
                    provider=provider,
                    stream_cache=stream_cache,
                    long_only=_is_truthy(params.get("long_only", "0")),
                    short_only=_is_truthy(params.get("short_only", "0")),
                    source="auto_scan_handoff",
                    promotion_candidates=slice_syms,
                )
                result["monitor_id"] = sess.monitor_id
                _pt_ensure_running()
            except Exception as _monitor_e:
                result["monitor_error"] = f"{type(_monitor_e).__name__}: {_monitor_e}"
            _set_job(
                jid,
                status="done",
                result=result,
                partial_result=result,
                progress={
                    "scanned": scanned,
                    "chunks_done": chunks_total,
                    "chunks_total": chunks_total,
                    "offset": start,
                    "end_offset": end,
                },
                updated_at=time.time(),
            )
    except Exception as e:
        _set_job(jid, status="error", error=str(e), updated_at=time.time())


@app.post("/api/scan_start")
def api_scan_start():
    data = request.get_json(silent=True) or request.form.to_dict(flat=True) or {}
    force_restart = _is_truthy((data or {}).get("force_restart", "0"))
    for k, v in request.args.items():
        data.setdefault(k, v)

    strategy = str(data.get("strategy", data.get("scan_strategy", "orb"))).strip().lower() or "orb"
    exec_style = str(data.get("exec_style", data.get("execution", "retest"))).strip().lower() or "retest"
    offset = _param_int(data, "offset", 0)
    max_symbols = _param_int(data, "max_symbols", 400)
    limit = _safe_int(data.get("top_n", data.get("limit", 25)), 25)

    data.setdefault("disable_auto_monitor", os.getenv("ORB_DISABLE_AUTO_MONITOR_DEFAULT", "1"))

    strategy_norm = str(strategy).strip().lower()
    # Parabolic needs to sweep the full universe; bump max_symbols if user left it at default
    if strategy_norm in {"parabolic", "para", "parabolic_watch"} and max_symbols <= 400:
        max_symbols = 5000
        data["max_symbols"] = str(max_symbols)
    max_price_for_profile = _param_float(data, "max_price", 30.0)
    orb_scalp_under10 = strategy_norm == "orb" and max_price_for_profile <= 10.0
    scan_preset = str(data.get("preset", data.get("scan_preset", "auto"))).strip().lower() or "auto"

    # Benchmark-aware presets inspired by top scanner workflows.
    if scan_preset == "tradeideas":
        data.setdefault("use_ml", "1")
        data.setdefault("use_sentiment", "1")
        data.setdefault("use_catalyst", "1")
        data.setdefault("min_grade", "B")
        data.setdefault("min_combined_score", "0.46")
        data.setdefault("min_rvol", "2.0")
        data.setdefault("min_today_dollar_vol", "5000000")
        data.setdefault("orb_min_minutes_after_open", "10")
    elif scan_preset == "tradingview":
        data.setdefault("use_ml", "0")
        data.setdefault("use_sentiment", "0")
        data.setdefault("use_catalyst", "0")
        data.setdefault("min_grade", "C")
        data.setdefault("min_combined_enabled", "0")
        data.setdefault("min_rvol", "1.2")
        data.setdefault("min_today_dollar_vol", "1500000")
    elif scan_preset == "trendspider":
        data.setdefault("use_ml", "1")
        data.setdefault("use_sentiment", "1")
        data.setdefault("use_catalyst", "1")
        data.setdefault("min_grade", "B")
        data.setdefault("min_combined_score", "0.44")
        data.setdefault("no_chop_enabled", "1")
        data.setdefault("min_vwap_enabled", "1")
        data.setdefault("min_pct_over_vwap", "0.6")
    elif scan_preset == "finviz":
        data.setdefault("use_ml", "0")
        data.setdefault("use_sentiment", "0")
        data.setdefault("use_catalyst", "0")
        data.setdefault("min_grade", "C")
        data.setdefault("min_combined_enabled", "0")
        data.setdefault("min_today_dollar_vol", "2000000")
        data.setdefault("min_avg20_dollar_vol", "1500000")
    elif scan_preset == "benzinga":
        data.setdefault("use_ml", "1")
        data.setdefault("use_sentiment", "1")
        data.setdefault("use_catalyst", "1")
        data.setdefault("min_grade", "B")
        data.setdefault("min_combined_score", "0.42")
        data.setdefault("min_today_dollar_vol", "3000000")
        data.setdefault("min_rvol", "1.6")
    elif scan_preset == "shorts":
        # Short-biased ORB scan: gap-down stocks, liquid tape, borrowable price range.
        # VWAP filter disabled — shorts are below VWAP by definition.
        # Stricter dollar volume and spread requirements for clean short fills.
        data.setdefault("short_only", "1")
        data.setdefault("long_only", "0")
        data.setdefault("min_price", "10.0")       # hard-to-borrow below $10
        data.setdefault("max_price", "500.0")       # big-cap shorts are fine
        data.setdefault("min_today_dollar_vol", "5000000")   # liquid enough to cover
        data.setdefault("min_avg20_dollar_vol", "3000000")
        data.setdefault("min_rvol", "1.5")
        data.setdefault("min_grade", "B")
        data.setdefault("min_grade_enabled", "1")
        data.setdefault("min_combined_enabled", "0")   # don't require ML score for shorts
        data.setdefault("min_vwap_enabled", "0")       # shorts are below VWAP
        data.setdefault("no_chop_enabled", "1")
        data.setdefault("min_or_range_pct", "0.5")
        data.setdefault("max_or_range_pct", "8.0")
    elif scan_preset == "parabolic_watch":
        data["strategy"] = "parabolic"
        data.setdefault("min_pm_move_pct",      "5")
        data.setdefault("min_rvol",              "1.5")
        data.setdefault("min_price",             "1.0")
        data.setdefault("max_price",             "500")
        data.setdefault("min_avg20_dollar_vol",  "250000")
        data.setdefault("use_catalyst",          "1")
    elif scan_preset == "saturday_prep":
        # Weekend EOD momentum scan — generates Monday watchlist candidates from Friday's closes.
        # Uses only daily bars; works with no live market data.
        data["strategy"] = "eod_momentum"  # force — setdefault silently loses to existing key
        data.setdefault("min_day_move_pct", "3.0")
        data.setdefault("max_day_move_pct", "50.0")
        data.setdefault("min_close_vs_range", "0.60")
        data.setdefault("min_rvol", "1.5")
        data.setdefault("min_avg20_dollar_vol", "1000000")
        data.setdefault("min_price", "2.0")
        data.setdefault("max_price", "200.0")
        data.setdefault("max_symbols", "5000")
        data.setdefault("use_ml", "0")
        data.setdefault("use_sentiment", "0")
        data.setdefault("use_catalyst", "0")

    # Trade-facing scan defaults should be quality-first unless explicitly overridden.
    # These gates can still be turned off by passing 0/false, but the default live scan
    # should not run in loose discovery mode.
    data.setdefault("min_grade_enabled", "1")
    data.setdefault("min_grade", "B")
    data.setdefault("min_combined_enabled", "1")
    data.setdefault("no_chop_enabled", "1")
    data.setdefault("min_vwap_enabled", "1")
    data.setdefault("min_pct_over_vwap", "1.0")

    # Under-$10 ORB scalp profile: stricter liquidity / RVOL / chase defaults for cheap names.
    if orb_scalp_under10:
        data.setdefault("long_only", "1")
        data.setdefault("min_rvol", "2.25")
        data.setdefault("min_today_dollar_vol", "4000000")
        data.setdefault("min_avg20_dollar_vol", "2000000")
        data.setdefault("min_or_range_pct", "0.8")
        data.setdefault("max_or_range_pct", "4.5")
        data.setdefault("min_combined_score", "0.42")
        data.setdefault("orb_min_ml_score", "0.00")
        data.setdefault("orb_max_chase_r", "0.25")
        data.setdefault("orb_retest_max_chase_r", "0.15")
        data.setdefault("orb_breakout_now_max_chase_r", "0.08")
        data.setdefault("orb_breakout_now_min_ml_score", "0.50")
        data.setdefault("orb_retest_min_ml_score", "0.45")
        data.setdefault("orb_min_minutes_after_open", "12")
    else:
        data.setdefault("min_combined_score", "0.40")

    # Re-read strategy after preset may have force-set it (e.g. parabolic_watch, saturday_prep)
    strategy = str(data.get("strategy", strategy)).strip().lower() or strategy

    thresholds_used = {
        "strategy": strategy,
        "exec_style": exec_style,
        "scan_mode": "orb",
        "trade_style": exec_style,
        "top_tool_profile": True,
        "scan_preset": scan_preset,
        "offset": offset,
        "max_symbols": max_symbols,
        "limit": limit,
        "min_price": _param_float(data, "min_price", 1.0),
        "max_price": _param_float(data, "max_price", 30.0),
        "min_today_dollar_vol": _param_float(data, "min_today_dollar_vol", 2_000_000.0),
        "min_avg20_dollar_vol": _param_float(data, "min_avg20_dollar_vol", 1_000_000.0),
        "min_rvol": _param_float(data, "min_rvol", 0.5),
        "min_or_range_pct": _param_float(data, "min_or_range_pct", 0.6),
        "max_or_range_pct": _param_float(data, "max_or_range_pct", 6.0),
        "use_ml": _is_truthy(data.get("use_ml", "0")),
        "use_sentiment": _is_truthy(data.get("use_sentiment", "0")),
        "use_catalyst": _is_truthy(data.get("use_catalyst", "0")),
        "long_only": _is_truthy(data.get("long_only", "0")),
        "short_only": _is_truthy(data.get("short_only", "0")),
        "min_grade_enabled": _is_truthy(data.get("min_grade_enabled", "0")),
        "min_grade": str(data.get("min_grade", "B")).strip().upper() or "B",
        "min_combined_enabled": _is_truthy(data.get("min_combined_enabled", "0")),
        "min_combined_score": _param_float(data, "min_combined_score", 0.40),
        "no_chop_enabled": _is_truthy(data.get("no_chop_enabled", "0")),
        "min_vwap_enabled": _is_truthy(data.get("min_vwap_enabled", "0")),
        "min_pct_over_vwap": _param_float(data, "min_pct_over_vwap", 1.0),
        "orb_scalp_under10": bool(orb_scalp_under10),
        "orb_min_ml_score": _param_float(data, "orb_min_ml_score", 0.05),
        "orb_max_chase_r": _param_float(data, "orb_max_chase_r", 0.35),
        "orb_retest_max_chase_r": _param_float(data, "orb_retest_max_chase_r", 0.20),
        "orb_breakout_now_max_chase_r": _param_float(data, "orb_breakout_now_max_chase_r", 0.10),
        "orb_breakout_now_min_ml_score": _param_float(data, "orb_breakout_now_min_ml_score", 0.10),
        "orb_retest_min_ml_score": _param_float(data, "orb_retest_min_ml_score", 0.10),
        "orb_min_minutes_after_open": _param_int(data, "orb_min_minutes_after_open", 10),
        "range_window_min": _param_int(data, "range_window_min", 60),
        "range_band_k": _param_float(data, "range_band_k", 2.0),
        "rr_touch_lookback_min": _param_int(data, "rr_touch_lookback_min", 15),
        "rr_stop_sigma_mult": _param_float(data, "rr_stop_sigma_mult", 0.75),
        "auto_monitor": not _disable_auto_monitor(data),
    }

    if not force_restart:
        existing_jid, existing_job = _find_running_job()
        if existing_jid and existing_job:
            return jsonify(ok=True, reused=True, job_id=existing_jid, thresholds_used=(existing_job.get("thresholds_used") or {}), progress=(existing_job.get("progress") or {}))

    jid = _new_job()
    _set_job(
        jid,
        params=data,
        thresholds_used=thresholds_used,
        provider=getattr(_ALPACA_PROVIDER, "name", "alpaca"),
        partial_result={
            "provider": getattr(_ALPACA_PROVIDER, "name", "alpaca"),
            "strategy": strategy,
            "scan_date": None,
            "session_date_used": None,
            "prefilter_counts": {},
            "prefilter_samples": [],
            "thresholds_used": thresholds_used,
            "reject_counts": {},
            "failure_samples_by_code": {},
            "data_failures": {},
            "shortlisted": 0,
            "candidates": [],
            "seed_candidates": [],
            "primary_candidates": [],
            "primary_mode": "empty",
            "primary_message": None,
            "zero_result_diagnostics": {},
            "scanned": 0,
            "chunks": 0,
            "end_offset": offset + max_symbols,
            "universe_size": 0,
            "mode": data.get("scan_mode", data.get("mode", "until")),
            "debug": {
                "strategy": strategy,
                "session_date_used": None,
                "failure_samples": [],
                "failure_samples_by_code": {},
                "chunk_errors": [],
                "chunk_timings": [],
            },
        },
        progress={
            "scanned": 0,
            "chunks_done": 0,
            "chunks_total": 0,
            "offset": offset,
            "end_offset": offset + max_symbols,
        },
        updated_at=time.time(),
    )

    t = threading.Thread(target=_scan_worker, args=(jid,), daemon=True)
    t.start()
    return jsonify(ok=True, reused=False, job_id=jid, thresholds_used=thresholds_used)



def _et_now():
    return datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))

def _session_date_now_et():
    return _et_now().date().isoformat()

def _clean_numeric_value(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    s = str(v).strip()
    if not s:
        return None
    # Accept values copied from terminals/markdown/forms like `0, $1.25, 2_000_000.
    s = s.replace('`', '').replace(',', '').replace('$', '').replace('_', '').strip()
    if s.lower() in {'none', 'null', 'nan', 'na', 'n/a'}:
        return None
    return s


def _safe_float(v):
    try:
        cleaned = _clean_numeric_value(v)
        if cleaned is None:
            return None
        return float(cleaned)
    except Exception:
        return None


def _safe_int(v, default=0):
    try:
        cleaned = _clean_numeric_value(v)
        if cleaned is None:
            return default
        return int(float(cleaned))
    except Exception:
        return default


def _param_float(source, key, default):
    val = _safe_float(source.get(key, default))
    return float(default) if val is None else val


def _param_int(source, key, default):
    return _safe_int(source.get(key, default), int(default))


def _batch_snapshot_live_states(symbols: list[str] | None) -> dict[str, dict[str, Any]]:
    syms = [str(s or "").strip().upper() for s in (symbols or []) if str(s or "").strip()]
    if not syms:
        return {}
    if _ALPACA_PROVIDER is None:
        feed = (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex"
        err = _PROVIDER_ERROR or "provider_not_initialized"
        return {
            sym: {
                "symbol": sym,
                "ok": False,
                "provider": "alpaca",
                "feed": feed,
                "source": "snapshot",
                "last_trade_price": None,
                "last_trade_ts": None,
                "bid": None,
                "ask": None,
                "mid": None,
                "quote_ts": None,
                "minute_bar_ts": None,
                "minute_bar_close": None,
                "daily_bar_close": None,
                "prev_close": None,
                "stale_reason": err,
                "error": err,
            }
            for sym in syms
        }

    feed = (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex"
    try:
        snapshots = _ALPACA_PROVIDER.get_snapshots(syms, feed=feed)
    except Exception as e:
        err = f"snapshot_batch_failed: {type(e).__name__}: {e}"
        return {
            sym: {
                "symbol": sym,
                "ok": False,
                "provider": "alpaca",
                "feed": feed,
                "source": "snapshot",
                "last_trade_price": None,
                "last_trade_ts": None,
                "bid": None,
                "ask": None,
                "mid": None,
                "quote_ts": None,
                "minute_bar_ts": None,
                "minute_bar_close": None,
                "daily_bar_close": None,
                "prev_close": None,
                "stale_reason": err,
                "error": err,
            }
            for sym in syms
        }

    out: dict[str, dict[str, Any]] = {}
    for sym in syms:
        snap = dict(snapshots.get(sym) or {})
        latest_trade = dict(snap.get("latest_trade") or {})
        latest_quote = dict(snap.get("latest_quote") or {})
        minute_bar = dict(snap.get("minute_bar") or {})
        daily_bar = dict(snap.get("daily_bar") or {})
        prev_daily_bar = dict(snap.get("prev_daily_bar") or {})

        price = _safe_float(
            latest_trade.get("price")
            if latest_trade.get("price") is not None
            else snap.get("reference_price")
        )
        bid = _safe_float(latest_quote.get("bid_price"))
        ask = _safe_float(latest_quote.get("ask_price"))
        mid = ((bid + ask) / 2.0) if bid is not None and ask is not None else None

        err = snap.get("error")
        out[sym] = {
            "symbol": sym,
            "ok": bool(price is not None or bid is not None or ask is not None),
            "provider": "alpaca",
            "feed": feed,
            "source": "snapshot",
            "price_source": snap.get("reference_price_source"),
            "last_trade_price": price,
            "last_trade_ts": latest_trade.get("timestamp") or minute_bar.get("timestamp") or daily_bar.get("timestamp"),
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "quote_ts": latest_quote.get("timestamp"),
            "minute_bar_ts": minute_bar.get("timestamp"),
            "minute_bar_close": _safe_float(minute_bar.get("close")),
            "daily_bar_close": _safe_float(daily_bar.get("close")),
            "prev_close": _safe_float(prev_daily_bar.get("close")),
            "stale_reason": err,
            "error": err,
        }
    return out


def _get_live_state(symbol: str) -> dict[str, Any]:
    sym = (symbol or "").strip().upper()
    if not sym:
        feed = (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex"
        return {"symbol": sym, "ok": False, "provider": "alpaca", "feed": feed, "source": "snapshot", "error": "missing_symbol"}
    return _batch_snapshot_live_states([sym]).get(sym) or {
        "symbol": sym,
        "ok": False,
        "provider": "alpaca",
        "feed": (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex",
        "source": "snapshot",
        "error": "snapshot_missing",
        "stale_reason": "snapshot_missing",
    }


def _apply_live_state_to_candidate(c: dict[str, Any], live: dict[str, Any] | None = None) -> dict[str, Any]:
    row = _normalize_target_rr(dict(c or {}))
    sym = str(row.get("symbol") or "").upper()
    if not sym:
        return row
    live = dict(live or _get_live_state(sym) or {})
    row["live_state"] = live
    row["provider"] = row.get("provider") or live.get("provider") or "alpaca"
    row["live_source"] = live.get("source") or "snapshot"
    price = _safe_float(live.get("last_trade_price"))
    if price is not None:
        row["live_price"] = price
        row["price"] = price
        row["last_price"] = price
    else:
        row["live_price"] = None
        row["price"] = row.get("price")
        row["last_price"] = row.get("last_price")
    row["live_price_ts"] = live.get("last_trade_ts")
    row["live_bid"] = live.get("bid")
    row["live_ask"] = live.get("ask")
    row["live_mid"] = live.get("mid")
    row["live_feed"] = live.get("feed")
    row["live_quote_ts"] = live.get("quote_ts")
    row["live_minute_bar_ts"] = live.get("minute_bar_ts")
    row["live_minute_bar_close"] = live.get("minute_bar_close")
    row["prev_close"] = row.get("prev_close") if row.get("prev_close") is not None else live.get("prev_close")
    row["live_state_error"] = live.get("error") or live.get("stale_reason")
    if live.get("stale_reason"):
        row["stale_reason"] = live.get("stale_reason")

    touch_ts = row.get("touch_ts")
    try:
        if touch_ts:
            ts = datetime.fromisoformat(str(touch_ts).replace("Z", "+00:00"))
            ts_et = ts.astimezone(ZoneInfo("America/New_York"))
            prior = ts_et.date().isoformat() != _session_date_now_et()
            row["prior_session_touch"] = bool(prior)
            if prior and not row.get("stale_reason"):
                row["stale_reason"] = "prior_session_touch"
    except Exception:
        pass

    live_px = _safe_float(row.get("last_price"))
    entry = _safe_float(row.get("entry"))
    stop = _safe_float(row.get("stop"))
    vwap_last = _safe_float(row.get("vwap_last"))
    if live_px is not None and vwap_last not in (None, 0.0):
        row["live_vwap_delta_pct"] = ((live_px - vwap_last) / vwap_last) * 100.0
    if live_px is not None and entry is not None and stop is not None and entry > stop:
        risk = max(entry - stop, 1e-9)
        row["live_chase_r"] = (live_px - entry) / risk
        if row.get("strategy") in {"range_reversion", "rr"}:
            row["rr_chase_r"] = row["live_chase_r"]
            actionable = True
            if row.get("prior_session_touch"):
                actionable = False
            if row.get("trend_state") in {"down", "lost_vwap", "chop"}:
                actionable = False
            if row.get("live_vwap_delta_pct") is not None and float(row["live_vwap_delta_pct"]) < -2.0:
                actionable = False
            if row["live_chase_r"] is not None and float(row["live_chase_r"]) > 0.35:
                actionable = False
            row["rr_actionable_now"] = actionable
    return row


def _apply_live_state_to_candidates(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    prepared = [dict(row or {}) for row in (rows or [])]
    symbols = [str(r.get("symbol") or "").upper() for r in prepared if str(r.get("symbol") or "").strip()]
    live_map = _batch_snapshot_live_states(symbols)
    return [_apply_live_state_to_candidate(row, live=live_map.get(str(row.get("symbol") or "").upper())) for row in prepared]


def _tag_desk_watched(candidates: list[dict[str, Any]], desk_map: dict[str, dict]) -> list[dict[str, Any]]:
    """Stamp desk_watched=True and desk_watch fields onto any candidate on the pre-market desk watchlist."""
    if not desk_map:
        return candidates
    out = []
    for c in candidates:
        sym = str(c.get("symbol") or "").upper()
        entry = desk_map.get(sym)
        if entry:
            c = dict(c)
            c["desk_watched"] = True
            c["desk_watch_trigger"] = entry.get("trigger_price")
            c["desk_watch_stop"] = entry.get("stop_price")
            c["desk_watch_target"] = entry.get("target_price")
            c["desk_watch_side"] = entry.get("side")
            c["desk_watch_notes"] = entry.get("notes")
        out.append(c)
    return out


def _scan_result_view_payload(result: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(result or {})
    view_limit = max(
        1,
        _safe_int(
            (payload.get("thresholds_used") or {}).get("limit", payload.get("count", 25)),
            25,
        ),
    )
    view = build_primary_fallback_view(payload, limit=view_limit)
    payload["primary_candidates"] = _apply_live_state_to_candidates(view.get("primary_candidates") or [])
    payload["primary_mode"] = view.get("primary_mode")
    payload["primary_message"] = view.get("primary_message")
    payload["trade_ready_candidates"] = _apply_live_state_to_candidates(view.get("trade_ready_candidates") or [])
    payload["trade_ready_candidate_count"] = int(view.get("trade_ready_candidate_count") or 0)
    payload["fallback_candidates"] = _apply_live_state_to_candidates(view.get("fallback_candidates") or [])
    payload["fallback_candidate_count"] = int(view.get("fallback_candidate_count") or 0)
    payload["zero_result_diagnostics"] = view.get("zero_result_diagnostics") or build_zero_result_diagnostics(payload)
    return payload


def _normalize_target_rr(row: dict[str, Any]) -> dict[str, Any]:
    r = dict(row or {})
    target = _safe_float(r.get("target"))
    if target is None:
        for k in (
            "take_profit",
            "target_2r",
            "target_1r",
            "target_3r",
            "profit_target",
            "tp",
        ):
            target = _safe_float(r.get(k))
            if target is not None:
                r["target"] = target
                break
    entry = _safe_float(r.get("entry"))
    stop = _safe_float(r.get("stop"))
    rr = _safe_float(r.get("rr"))
    if rr is None and entry is not None and stop is not None and target is not None:
        risk = entry - stop
        reward = target - entry
        if risk > 0 and reward > 0:
            r["rr"] = float(reward / risk)
    elif rr is not None:
        r["rr"] = float(rr)
    return r


def _time_bucket_label(now_et: datetime) -> str:
    mins = now_et.hour * 60 + now_et.minute
    if mins < 9 * 60 + 30:
        return "premarket"
    if mins < 10 * 60:
        return "open_drive"
    if mins < 11 * 60 + 30:
        return "morning"
    if mins < 14 * 60:
        return "midday"
    if mins < 15 * 60 + 30:
        return "power_hour"
    if mins < 16 * 60:
        return "closing_hour"
    return "after_hours"


def _minutes_to_close(now_et: datetime) -> int | None:
    close_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    delta = int((close_dt - now_et).total_seconds() // 60)
    return delta if delta >= 0 else None


def _position_plan_payload(symbol: str, position: dict[str, Any], provider, context_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    symbol = str(symbol or "").strip().upper()
    side = str(position.get("side") or "long").strip().lower()
    entry = _safe_float(position.get("entry"))
    stop = _safe_float(position.get("stop"))
    shares = _safe_float(position.get("shares")) or 0.0
    target = _safe_float(position.get("target"))
    notes = str(position.get("notes") or "").strip()
    must_flat = bool(position.get("must_flat") or False)

    if not symbol:
        raise ValueError("missing_symbol")
    if side not in {"long", "short"}:
        raise ValueError("invalid_side")
    if entry is None or stop is None:
        raise ValueError("missing_entry_or_stop")

    intraday, live_price = _entry_now_intraday_context(provider, symbol, include_prepost=False)
    trend_ctx = _entry_now_trend_context(intraday)
    vwap_last = trend_ctx.get("vwap_last")
    vwap_delta_pct = trend_ctx.get("vwap_delta_pct")
    trend_state = trend_ctx.get("trend_state")
    trend_slope_pct = trend_ctx.get("trend_slope_pct")

    tail = intraday.iloc[-15:] if len(intraday) >= 15 else intraday
    hi15 = float(_col(tail, 'High', 'high').astype(float).max()) if not tail.empty else None
    lo15 = float(_col(tail, 'Low', 'low').astype(float).min()) if not tail.empty else None

    risk_per_share = abs(float(entry) - float(stop)) if entry is not None and stop is not None else None
    if risk_per_share is None or risk_per_share <= 0:
        raise ValueError("invalid_risk_per_share")

    if side == "long":
        current_r = (float(live_price) - float(entry)) / risk_per_share
        stop_distance_r = (float(live_price) - float(stop)) / risk_per_share
        r1 = float(entry) + risk_per_share
        r2 = float(entry) + 2.0 * risk_per_share
        r3 = float(entry) + 3.0 * risk_per_share
    else:
        current_r = (float(entry) - float(live_price)) / risk_per_share
        stop_distance_r = (float(stop) - float(live_price)) / risk_per_share
        r1 = float(entry) - risk_per_share
        r2 = float(entry) - 2.0 * risk_per_share
        r3 = float(entry) - 3.0 * risk_per_share

    target_active = target if target is not None else r2
    if side == "long":
        target_distance_r = (float(target_active) - float(live_price)) / risk_per_share
    else:
        target_distance_r = (float(live_price) - float(target_active)) / risk_per_share

    pnl_open = (float(live_price) - float(entry)) * float(shares) if side == "long" else (float(entry) - float(live_price)) * float(shares)
    spread_pct = None
    live_state = _get_live_state(symbol)
    bid = _safe_float(live_state.get("bid"))
    ask = _safe_float(live_state.get("ask"))
    if bid is not None and ask is not None and (bid + ask) > 0:
        spread_pct = ((ask - bid) / ((ask + bid) / 2.0)) * 100.0

    now_et = datetime.now(_ET)
    minutes_to_close = _minutes_to_close(now_et)
    time_bucket = _time_bucket_label(now_et)
    context_snapshot = context_snapshot or {}
    risk_on_score = _safe_float(context_snapshot.get("risk_on_score"))
    breadth_score = _safe_float(context_snapshot.get("breadth_score"))
    spy_trend = str(context_snapshot.get("spy_trend_state") or "unknown")
    qqq_trend = str(context_snapshot.get("qqq_trend_state") or "unknown")
    volatility_regime = str(context_snapshot.get("volatility_regime") or "unknown")

    against_trend = (side == "long" and trend_state in {"down", "lost_vwap"}) or (side == "short" and trend_state in {"up", "reclaim_vwap"})
    vwap_good = (side == "long" and (vwap_delta_pct is None or float(vwap_delta_pct) >= -0.15)) or (side == "short" and (vwap_delta_pct is None or float(vwap_delta_pct) <= 0.15))
    near_stop = stop_distance_r <= 0.35
    near_1r = abs(current_r - 1.0) <= 0.20
    deep_in_money = current_r >= 1.25
    weak_spread = spread_pct is not None and spread_pct >= 0.45
    late_day = minutes_to_close is not None and minutes_to_close <= 30
    very_late = minutes_to_close is not None and minutes_to_close <= 12
    risk_off_market = ((risk_on_score is not None and risk_on_score < -0.35) or (breadth_score is not None and breadth_score < -0.35) or spy_trend == "down")
    breakeven_price = float(entry)
    if side == "long":
        lock_price = float(entry) + (0.25 * risk_per_share)
        soft_trail = max(float(stop), float(entry), float(vwap_last) if vwap_last is not None else float(stop), float(lo15) if lo15 is not None else float(stop))
        hard_exit_price = max(float(stop), float(vwap_last) if vwap_last is not None else float(stop))
        trim_price = target_active if target_active is not None else r1
    else:
        lock_price = float(entry) - (0.25 * risk_per_share)
        soft_trail = min(float(stop), float(entry), float(vwap_last) if vwap_last is not None else float(stop), float(hi15) if hi15 is not None else float(stop))
        hard_exit_price = min(float(stop), float(vwap_last) if vwap_last is not None else float(stop))
        trim_price = target_active if target_active is not None else r1

    action = "HOLD"
    urgency = 40
    rationale: list[str] = []
    exit_plan: list[str] = []
    confidence = 0.5
    management_mode = "neutral"

    if stop_distance_r <= 0:
        action = "EXIT NOW"
        urgency = 100
        confidence = 0.98
        management_mode = "damage_control"
        rationale.append("stop breached")
        exit_plan.append("flat now; thesis broken")
    elif near_stop and against_trend:
        action = "EXIT NOW"
        urgency = 96
        confidence = 0.92
        management_mode = "damage_control"
        rationale.extend(["near stop", "trend against position"])
        exit_plan.append(f"hard exit if {symbol} loses {hard_exit_price:.2f}")
    elif very_late and current_r < 0.35 and (must_flat or against_trend or weak_spread):
        action = "EXIT INTO CLOSE"
        urgency = 94
        confidence = 0.90
        management_mode = "closeout"
        rationale.append("late day with weak cushion")
        if must_flat:
            rationale.append("must flat by close")
        exit_plan.append("use marketable limit near bid/ask; do not carry")
    elif late_day and near_1r and (against_trend or risk_off_market):
        action = "TRIM / PAY YOURSELF"
        urgency = 82
        confidence = 0.78
        management_mode = "de_risk"
        rationale.extend(["near 1R", "late day context not ideal"])
        exit_plan.append(f"take at least partial near {trim_price:.2f}")
        exit_plan.append(f"move stop toward {lock_price:.2f} if partial fills")
    elif deep_in_money and late_day:
        action = "TRAIL WINNER"
        urgency = 76
        confidence = 0.76
        management_mode = "protect_winner"
        rationale.extend(["late day winner", "protect gains"])
        exit_plan.append(f"trail stop near {soft_trail:.2f}")
        exit_plan.append("consider partial into strength before close")
    elif current_r >= 0.60 and vwap_good and not against_trend:
        action = "HOLD / LET IT WORK"
        urgency = 54
        confidence = 0.68
        management_mode = "trend_hold"
        rationale.extend(["trend intact", "position has cushion"])
        if current_r >= 1.0:
            exit_plan.append(f"consider stop to breakeven or better ({breakeven_price:.2f}+)")
        else:
            exit_plan.append(f"if it re-tests and holds VWAP, stay patient above {hard_exit_price:.2f}")
    elif current_r < 0.25 and not vwap_good:
        action = "REDUCE RISK"
        urgency = 74
        confidence = 0.72
        management_mode = "weak_hold"
        rationale.extend(["not getting paid yet", "VWAP relationship weak"])
        exit_plan.append(f"tighten stop or cut partial on failed bounce below/above {hard_exit_price:.2f}")
    else:
        action = "WAIT / REASSESS"
        urgency = 60
        confidence = 0.55
        management_mode = "neutral"
        rationale.append("mixed signals")
        exit_plan.append("wait for reclaim / break of local range")

    overnight = "NO HOLD"
    overnight_reasons: list[str] = []
    if must_flat:
        overnight = "NO HOLD"
        overnight_reasons.append("marked intraday only")
    elif current_r >= 1.0 and vwap_good and not against_trend and not weak_spread and not risk_off_market and minutes_to_close is not None and minutes_to_close <= 20:
        overnight = "PARTIAL HOLD OK"
        overnight_reasons.extend(["good cushion", "trend intact", "close carry acceptable"])
    elif current_r >= 0.35 and not against_trend and not weak_spread:
        overnight = "MAYBE HOLD SMALL"
        overnight_reasons.append("acceptable only with reduced size")
    else:
        overnight = "NO HOLD"
        overnight_reasons.append("insufficient edge for overnight")

    return {
        "ok": True,
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "stop": stop,
        "shares": shares,
        "target": target_active,
        "notes": notes,
        "must_flat": must_flat,
        "live_price": live_price,
        "bid": bid,
        "ask": ask,
        "spread_pct": spread_pct,
        "pnl_open": pnl_open,
        "risk_per_share": risk_per_share,
        "current_r": current_r,
        "stop_distance_r": stop_distance_r,
        "target_distance_r": target_distance_r,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "vwap_last": vwap_last,
        "vwap_delta_pct": vwap_delta_pct,
        "trend_state": trend_state,
        "trend_slope_pct": trend_slope_pct,
        "hi15": hi15,
        "lo15": lo15,
        "minutes_to_close": minutes_to_close,
        "time_bucket": time_bucket,
        "market_context": {
            "spy_trend_state": spy_trend,
            "qqq_trend_state": qqq_trend,
            "risk_on_score": risk_on_score,
            "breadth_score": breadth_score,
            "volatility_regime": volatility_regime,
        },
        "flags": {
            "near_stop": near_stop,
            "near_1r": near_1r,
            "deep_in_money": deep_in_money,
            "against_trend": against_trend,
            "vwap_good": vwap_good,
            "weak_spread": weak_spread,
            "late_day": late_day,
            "very_late": very_late,
            "risk_off_market": risk_off_market,
        },
        "action": action,
        "urgency": urgency,
        "confidence": confidence,
        "management_mode": management_mode,
        "breakeven_price": breakeven_price,
        "lock_price": lock_price,
        "soft_trail": soft_trail,
        "hard_exit_price": hard_exit_price,
        "trim_price": trim_price,
        "rationale": rationale,
        "exit_plan": exit_plan,
        "overnight": overnight,
        "overnight_reasons": overnight_reasons,
    }


@app.post("/api/position_manager")
def api_position_manager():
    payload = request.get_json(silent=True) or {}
    positions = payload.get("positions") or []
    if not isinstance(positions, list) or not positions:
        return jsonify(ok=False, error="missing_positions"), 400
    provider = _ALPACA_PROVIDER
    if provider is None:
        return jsonify(ok=False, error=(_PROVIDER_ERROR or "provider_not_initialized")), 503

    ctx = _CONTEXT_ENGINE.snapshot() if '_CONTEXT_ENGINE' in globals() else {}
    rows = []
    errors = []
    for raw in positions[:25]:
        try:
            row = _position_plan_payload(str(raw.get("symbol") or ""), raw, provider, ctx)
            rows.append(row)
        except IntradayDataFailure:
            errors.append({
                "symbol": str((raw or {}).get("symbol") or "").strip().upper(),
                "error": "market_closed",
            })
        except Exception as e:
            errors.append({
                "symbol": str((raw or {}).get("symbol") or "").strip().upper(),
                "error": f"{type(e).__name__}: {e}",
            })
    rows.sort(key=lambda r: (-int(r.get("urgency") or 0), str(r.get("symbol") or "")))
    scoreboard = []
    for r in rows:
        scoreboard.append({
            "symbol": r.get("symbol"),
            "action": r.get("action"),
            "urgency": r.get("urgency"),
            "current_r": r.get("current_r"),
            "minutes_to_close": r.get("minutes_to_close"),
            "overnight": r.get("overnight"),
            "confidence": r.get("confidence"),
        })
    return jsonify(ok=True, context_snapshot=ctx, rows=rows, scoreboard=scoreboard, errors=errors, count=len(rows))


# ── Positions Desk — persistent trade journal ──────────────────────────────

@app.get("/api/positions/open")
def api_positions_open():
    rows = _RUNTIME_STORE.positions_open()
    return jsonify(ok=True, positions=rows, count=len(rows))


@app.get("/api/positions/history")
def api_positions_history():
    limit = min(int(request.args.get("limit", 100)), 500)
    rows = _RUNTIME_STORE.positions_history(limit=limit)
    return jsonify(ok=True, positions=rows, count=len(rows))


@app.post("/api/positions/save")
def api_positions_save():
    data = request.get_json(silent=True) or {}
    try:
        position_id = _RUNTIME_STORE.positions_save(data)
        return jsonify(ok=True, position_id=position_id)
    except Exception as e:
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 400


@app.post("/api/positions/close")
def api_positions_close():
    data = request.get_json(silent=True) or {}
    position_id = data.get("position_id")
    exit_price  = _safe_float(data.get("exit_price"))
    exit_reason = str(data.get("exit_reason") or "discretionary").strip()
    if not position_id or exit_price is None:
        return jsonify(ok=False, error="position_id and exit_price required"), 400
    try:
        row = _RUNTIME_STORE.positions_close(int(position_id), exit_price=exit_price, exit_reason=exit_reason)
        return jsonify(ok=True, position=row)
    except Exception as e:
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 400


@app.delete("/api/positions/<int:position_id>")
def api_positions_remove(position_id: int):
    removed = _RUNTIME_STORE.positions_remove(position_id)
    return jsonify(ok=removed)


@app.get("/api/positions/daily_summary")
def api_positions_daily_summary():
    from datetime import datetime
    date = request.args.get("date") or datetime.now(_ET).strftime("%Y-%m-%d")
    summary = _RUNTIME_STORE.positions_daily_summary(session_date=date)
    return jsonify(ok=True, **summary)


@app.get("/api/broker_snapshot")
def api_broker_snapshot():
    provider = _BROKER_PROVIDER
    if provider is None:
        return jsonify(ok=False, degraded=True, buying_power=0, equity=0,
                       broker=(_BROKER_PROVIDER_NAME or 'alpaca'), note='broker_provider_unavailable'), 503
    if not hasattr(provider, "get_broker_snapshot"):
        return jsonify(ok=False, degraded=True, buying_power=0, equity=0,
                       broker=(_BROKER_PROVIDER_NAME or 'alpaca'), note='get_broker_snapshot_not_implemented'), 503
    try:
        snap = provider.get_broker_snapshot()
        # Validate snapshot has actual data before claiming ok=True
        eq = float(snap.get('equity') or 0)
        bp = float(snap.get('buying_power') or 0)
        if eq == 0 and bp == 0:
            return jsonify(ok=False, degraded=True, snapshot=snap,
                           broker=(_BROKER_PROVIDER_NAME or 'alpaca'),
                           note='equity_and_buying_power_zero_check_credentials'), 503
        return jsonify(ok=True, snapshot=snap)
    except Exception as e:
        return jsonify(ok=False, degraded=True, buying_power=0, equity=0,
                       broker=(_BROKER_PROVIDER_NAME or 'alpaca'),
                       note=f'broker_snapshot_failed: {e}'), 503


@app.post("/api/broker_action")
def api_broker_action():
    provider = _BROKER_PROVIDER
    if provider is None:
        return jsonify(ok=False, error=(_BROKER_PROVIDER_ERROR or "provider_not_initialized"), broker=_BROKER_PROVIDER_NAME), 503
    if not BROKER_ACTIONS_ENABLED:
        return jsonify(ok=False, error="broker_actions_disabled", broker=_BROKER_PROVIDER_NAME), 403
    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action") or "").strip().lower()
    symbol = str(payload.get("symbol") or "").strip().upper()
    qty = _safe_float(payload.get("qty"))
    pct = _safe_float(payload.get("pct"))
    limit_price = _safe_float(payload.get("limit_price"))
    confirm = bool(payload.get("confirm") or False)
    if not confirm:
        return jsonify(ok=False, error="confirmation_required"), 400
    if action not in {"trim25", "trim50", "flatten", "limit_exit"}:
        return jsonify(ok=False, error="unsupported_action"), 400
    if not hasattr(provider, "submit_exit_order"):
        return jsonify(ok=False, error="broker_action_not_supported"), 501
    try:
        if action == "trim25":
            out = provider.submit_exit_order(symbol=symbol, notional_pct=0.25)
        elif action == "trim50":
            out = provider.submit_exit_order(symbol=symbol, notional_pct=0.50)
        elif action == "flatten":
            out = provider.submit_exit_order(symbol=symbol)
        else:
            out = provider.submit_exit_order(symbol=symbol, qty=qty, limit_price=limit_price)
        return jsonify(ok=True, broker=_BROKER_PROVIDER_NAME, action=action, symbol=symbol, order=out)
    except Exception as e:
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 500


@app.post("/api/position_chart")
def api_position_chart():
    payload = request.get_json(silent=True) or {}
    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify(ok=False, error="missing_symbol"), 400
    provider = _ALPACA_PROVIDER
    if provider is None:
        return jsonify(ok=False, error=(_PROVIDER_ERROR or "provider_not_initialized")), 503

    entry = _safe_float(payload.get("entry"))
    stop = _safe_float(payload.get("stop"))
    target = _safe_float(payload.get("target"))
    side = str(payload.get("side") or "long").strip().lower()
    timeframe = str(payload.get("timeframe") or "1m").strip().lower()

    try:
        intraday, live_price = _entry_now_intraday_context(provider, symbol, include_prepost=False)
    except IntradayDataFailure as e:
        return jsonify(ok=False, error=e.code, message=e.message, symbol=symbol), 400
    import pandas as _pd
    if timeframe == "5m":
        intraday = intraday.resample("5min").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    elif timeframe == "15m":
        intraday = intraday.resample("15min").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    close = _col(intraday, 'Close', 'close').astype(float)
    high = _col(intraday, 'High', 'high').astype(float)
    low = _col(intraday, 'Low', 'low').astype(float)
    open_ = _col(intraday, 'Open', 'open').astype(float)
    vw = vwap(intraday).astype(float)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    bb_upper = bb_mid + (2.0 * bb_std)
    bb_lower = bb_mid - (2.0 * bb_std)

    bars = []
    for ts, o, h, l, c in zip(intraday.index, open_, high, low, close):
        try:
            bars.append({
                "time": int(_pd.Timestamp(ts).timestamp()),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
            })
        except Exception:
            continue

    def _line_payload(series):
        out = []
        for ts, val in series.items():
            if val is None:
                continue
            try:
                fv = float(val)
            except Exception:
                continue
            if fv != fv:
                continue
            out.append({"time": int(_pd.Timestamp(ts).timestamp()), "value": fv})
        return out

    risk_per_share = abs(float(entry) - float(stop)) if entry is not None and stop is not None else None
    levels = {}
    if entry is not None:
        levels["entry"] = entry
    if stop is not None:
        levels["stop"] = stop
    if target is not None:
        levels["target"] = target
    if risk_per_share is not None and risk_per_share > 0 and entry is not None:
        if side == "long":
            levels["r1"] = float(entry) + risk_per_share
            levels["r2"] = float(entry) + 2.0 * risk_per_share
            levels["r3"] = float(entry) + 3.0 * risk_per_share
        else:
            levels["r1"] = float(entry) - risk_per_share
            levels["r2"] = float(entry) - 2.0 * risk_per_share
            levels["r3"] = float(entry) - 3.0 * risk_per_share
    try:
        first_5 = intraday.iloc[:5]
        if len(first_5) == 5:
            idx = _pd.DatetimeIndex(first_5.index)
            if idx.tz is None:
                idx = idx.tz_localize(_ET)
            else:
                idx = idx.tz_convert(_ET)
            expected_open = idx[0].normalize() + _pd.Timedelta(hours=9, minutes=30)
            expected_window = _pd.date_range(expected_open, periods=5, freq="min", tz=_ET)
            if all(idx[i] == expected_window[i] for i in range(5)):
                levels["or_high"] = float(_col(first_5, 'High', 'high').astype(float).max())
                levels["or_low"] = float(_col(first_5, 'Low', 'low').astype(float).min())
    except Exception:
        pass

    labels = []
    if stop is not None:
        labels.append({"key": "stop", "price": stop, "text": "GET OUT HERE", "color": "#ef4444"})
    if entry is not None:
        labels.append({"key": "entry", "price": entry, "text": "YOU BOUGHT HERE", "color": "#fbbf24"})
    if target is not None:
        labels.append({"key": "target", "price": target, "text": "PAY YOURSELF", "color": "#22c55e"})
    if "r1" in levels:
        labels.append({"key": "r1", "price": levels["r1"], "text": "1R FIRST PAY WINDOW", "color": "#22c55e"})
    if "r2" in levels:
        labels.append({"key": "r2", "price": levels["r2"], "text": "2R STRONG EXIT ZONE", "color": "#16a34a"})
    if "or_high" in levels:
        labels.append({"key": "or_high", "price": levels["or_high"], "text": "ORB HIGH", "color": "#a78bfa"})
    if "or_low" in levels:
        labels.append({"key": "or_low", "price": levels["or_low"], "text": "ORB LOW", "color": "#a78bfa"})
    if len(vw) > 0:
        try:
            vwap_px = float(vw.iloc[-1])
            labels.append({"key": "vwap", "price": vwap_px, "text": "WEAK IF LOSES VWAP", "color": "#facc15"})
            if side == "long":
                labels.append({"key": "hold_above", "price": vwap_px, "text": "OKAY TO HOLD ABOVE THIS", "color": "#eab308"})
            else:
                labels.append({"key": "hold_below", "price": vwap_px, "text": "OKAY TO HOLD BELOW THIS", "color": "#eab308"})
        except Exception:
            pass
    if risk_per_share is not None and entry is not None:
        if side == "long":
            labels.append({"key": "chase", "price": float(entry) + 1.25 * risk_per_share, "text": "DON'T CHASE ABOVE THIS", "color": "#fb7185"})
            labels.append({"key": "cooked", "price": float(stop), "text": "LOSES THIS = YOU'RE COOKED", "color": "#ef4444"})
        else:
            labels.append({"key": "chase", "price": float(entry) - 1.25 * risk_per_share, "text": "DON'T CHASE BELOW THIS", "color": "#fb7185"})
            labels.append({"key": "cooked", "price": float(stop), "text": "BREAKS THIS = YOU'RE COOKED", "color": "#ef4444"})

    return jsonify(
        ok=True,
        symbol=symbol,
        timeframe=timeframe,
        live_price=live_price,
        bars=bars,
        overlays={
            "vwap": _line_payload(vw),
            "ema9": _line_payload(ema9),
            "ema20": _line_payload(ema20),
            "bb_upper": _line_payload(bb_upper),
            "bb_mid": _line_payload(bb_mid),
            "bb_lower": _line_payload(bb_lower),
        },
        levels=levels,
        labels=labels,
    )


@app.get("/api/live_quotes")
def api_live_quotes():
    raw = (request.args.get("symbols") or "").strip()
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not symbols:
        return jsonify(ok=False, error="missing_symbols"), 400
    live_map = _batch_snapshot_live_states(symbols[:100])
    out = {}
    for sym in symbols[:100]:
        live = live_map.get(sym) or _get_live_state(sym)
        out[sym] = {
            "symbol": sym,
            "provider": live.get("provider") or "alpaca",
            "feed": live.get("feed"),
            "source": live.get("source") or "snapshot",
            "price": live.get("last_trade_price"),
            "price_ts": live.get("last_trade_ts"),
            "last_trade_ts": live.get("last_trade_ts"),
            "bid": live.get("bid"),
            "ask": live.get("ask"),
            "mid": live.get("mid"),
            "quote_ts": live.get("quote_ts"),
            "minute_bar_ts": live.get("minute_bar_ts"),
            "minute_bar_close": live.get("minute_bar_close"),
            "prev_close": live.get("prev_close"),
            "error": live.get("error"),
            "stale_reason": live.get("stale_reason"),
        }
    return jsonify(
        ok=True,
        provider="alpaca",
        feed=(os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex",
        quotes=out,
    )

@app.get("/api/scan_status")
def api_scan_status():
    jid = (request.args.get("job_id") or "").strip()
    if not jid:
        return jsonify(ok=False, error="missing job_id"), 400
    job = _get_job(jid)
    if not job:
        return jsonify(ok=False, error="unknown job_id"), 404
    status = job.get("status")
    progress = job.get("progress")
    err = job.get("error")
    result = job.get("result") or job.get("partial_result") or {}
    summary = None
    next_offset = None
    if result:
        result_view = _scan_result_view_payload(result)
        try:
            end_offset = int(result_view.get("end_offset") or 0)
            universe_size = int(result_view.get("universe_size") or 0)
            next_offset = end_offset
            if universe_size > 0 and next_offset >= universe_size:
                next_offset = 0
        except Exception:
            next_offset = None

        # Return a rich result payload so the UI can display real scan counts even when zero candidates.
        reject_counts = result_view.get("reject_counts") or {}
        top_rejection_reasons = []
        try:
            if isinstance(reject_counts, dict):
                top_rejection_reasons = sorted(
                    ((str(k), int(v)) for k, v in reject_counts.items() if v is not None),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:5]
        except Exception:
            top_rejection_reasons = []
        summary = {
            "count": int(result_view.get("count") or 0),
            "scanned": int(result_view.get("scanned") or 0),
            "chunks": int(result_view.get("chunks") or (progress or {}).get("chunks_total") or 0),
            "end_offset": int(result_view.get("end_offset") or 0),
            "universe_size": int(result_view.get("universe_size") or 0),
            "shortlisted": int(result_view.get("shortlisted") or 0),
            "candidates_total": int(result_view.get("candidates_total") or 0),
            "seed_candidates_total": int(result_view.get("seed_candidates_total") or 0),
            "tradable_now_total": int(result_view.get("tradable_now_total") or 0),
            "rejected_total": int(result_view.get("rejected_total") or 0),
            "mode": result_view.get("mode"),
            "provider": result_view.get("provider"),
            "prefilter_counts": result_view.get("prefilter_counts") or {},
            "prefilter_samples": result_view.get("prefilter_samples") or [],
            "thresholds_used": result_view.get("thresholds_used") or {},
            "reject_counts": reject_counts,
            "top_rejection_reasons": top_rejection_reasons,
            "data_failures": result_view.get("data_failures") or {},
            "monitor_id": result_view.get("monitor_id"),
            "monitor_error": result_view.get("monitor_error"),
            "seed_symbols": result_view.get("seed_symbols") or [],
            # include candidates so the browser can render without a full page reload
            "candidates": _apply_live_state_to_candidates(result_view.get("candidates") or []),
            "seed_candidates": _apply_live_state_to_candidates(result_view.get("seed_candidates") or []),
            "primary_candidates": result_view.get("primary_candidates") or [],
            "primary_mode": result_view.get("primary_mode"),
            "primary_message": result_view.get("primary_message"),
            "trade_ready_candidates": result_view.get("trade_ready_candidates") or [],
            "trade_ready_candidate_count": int(result_view.get("trade_ready_candidate_count") or 0),
            "fallback_candidates": result_view.get("fallback_candidates") or [],
            "fallback_candidate_count": int(result_view.get("fallback_candidate_count") or 0),
            "zero_result_diagnostics": result_view.get("zero_result_diagnostics") or build_zero_result_diagnostics(result_view),
            "rejected_candidates": result_view.get("rejected_candidates") or [],
        }
    if summary is not None:
        # Tag any candidate that appears on the pre-market desk watchlist
        try:
            _desk_entries = _RUNTIME_STORE.desk_watchlist_all()
            _desk_map = {str(e["symbol"]).upper(): e for e in _desk_entries}
            if _desk_map:
                for _key in ("candidates", "seed_candidates", "primary_candidates",
                             "trade_ready_candidates", "fallback_candidates", "rejected_candidates"):
                    summary[_key] = _tag_desk_watched(summary.get(_key) or [], _desk_map)
        except Exception:
            pass
    if summary is not None and _is_truthy(request.args.get("debug", "")):
        summary["debug"] = result_view.get("debug") or {}
    return jsonify(ok=True, job_id=jid, status=status, progress=progress, error=err, result=summary, next_offset=next_offset)




@app.get('/api/debug_job_raw')
def api_debug_job_raw():
    jid = (request.args.get('job_id') or '').strip()
    if not jid:
        return jsonify(ok=False, error='missing_job_id'), 400
    job = _get_job(jid)
    if not job:
        return jsonify(ok=False, error='unknown_job_id', job_id=jid), 404
    result = dict(job.get('result') or job.get('partial_result') or {})
    rebuilt = _scan_result_view_payload(result)
    return jsonify(
        ok=True,
        job_id=jid,
        status=job.get('status'),
        raw_result=result,
        rebuilt_result=rebuilt,
    )


@app.get("/api/debug_last_scan")
def api_debug_last_scan():
    req_jid = (request.args.get("job_id") or "").strip()

    def _job_payload(jid: str, job: dict[str, Any]) -> dict[str, Any]:
        result = _scan_result_view_payload((job or {}).get("result") or (job or {}).get("partial_result") or {})
        params = dict((job or {}).get("params") or {})
        candidates = result.get("candidates") or result.get("results") or []
        enriched_candidates = _apply_live_state_to_candidates(candidates)
        if "candidates" in result or candidates:
            result["candidates"] = enriched_candidates

        seed_candidates = list(result.get("seed_candidates") or [])
        if seed_candidates:
            result["seed_candidates"] = _apply_live_state_to_candidates(seed_candidates)

        result["primary_candidates"] = _apply_live_state_to_candidates(result.get("primary_candidates") or [])
        result["trade_ready_candidates"] = _apply_live_state_to_candidates(result.get("trade_ready_candidates") or [])
        result["fallback_candidates"] = _apply_live_state_to_candidates(result.get("fallback_candidates") or [])
        result["zero_result_diagnostics"] = result.get("zero_result_diagnostics") or build_zero_result_diagnostics(result)

        rejected_candidates = list(result.get("rejected_candidates") or [])
        if rejected_candidates:
            result["rejected_candidates"] = rejected_candidates

        debug_block = dict(result.get("debug") or {})
        thresholds = dict((job or {}).get("thresholds_used") or result.get("thresholds_used") or {})
        progress = dict((job or {}).get("progress") or {})

        if not thresholds:
            thresholds = {
                "strategy": str(params.get("strategy", params.get("scan_strategy", ""))).strip().lower() or None,
                "exec_style": str(params.get("exec_style", params.get("execution", ""))).strip().lower() or None,
                "offset": _param_int(params, "offset", 0),
                "max_symbols": _param_int(params, "max_symbols", 0),
                "limit": _safe_int(params.get("top_n", params.get("limit", 0)), 0),
                "min_price": _param_float(params, "min_price", 0.0),
                "max_price": _param_float(params, "max_price", 0.0),
                "min_today_dollar_vol": _param_float(params, "min_today_dollar_vol", 0.0),
                "min_avg20_dollar_vol": _param_float(params, "min_avg20_dollar_vol", 0.0),
                "min_rvol": _param_float(params, "min_rvol", 0.0),
                "use_ml": _is_truthy(params.get("use_ml", "0")),
                "use_sentiment": _is_truthy(params.get("use_sentiment", "0")),
                "use_catalyst": _is_truthy(params.get("use_catalyst", "0")),
                "long_only": _is_truthy(params.get("long_only", "0")),
                "min_grade_enabled": _is_truthy(params.get("min_grade_enabled", "0")),
                "min_grade": str(params.get("min_grade", "")).strip().upper() or None,
                "min_combined_enabled": _is_truthy(params.get("min_combined_enabled", "0")),
                "min_combined_score": _param_float(params, "min_combined_score", 0.0),
                "no_chop_enabled": _is_truthy(params.get("no_chop_enabled", "0")),
                "min_vwap_enabled": _is_truthy(params.get("min_vwap_enabled", "0")),
                "min_pct_over_vwap": _param_float(params, "min_pct_over_vwap", 0.0),
            }

        strategy = (
            thresholds.get("strategy")
            or result.get("strategy")
            or debug_block.get("strategy")
        )
        session_date_used = (
            result.get("session_date_used")
            or debug_block.get("session_date_used")
            or result.get("scan_date")
        )
        provider = (
            (job or {}).get("provider")
            or result.get("provider")
            or getattr(_ALPACA_PROVIDER, "name", "alpaca")
        )

        failure_samples = list(debug_block.get("failure_samples") or [])
        failure_samples_by_code = dict(
            result.get("failure_samples_by_code")
            or debug_block.get("failure_samples_by_code")
            or {}
        )
        prefilter_samples = list(result.get("prefilter_samples") or [])

        if not failure_samples_by_code and failure_samples:
            try:
                from scanner.orb import _build_failure_samples_by_code
                failure_samples_by_code = _build_failure_samples_by_code(failure_samples, limit_per_code=5)
            except Exception:
                failure_samples_by_code = {}

        if not prefilter_samples:
            seen_symbols = set()

            for item in failure_samples:
                if not isinstance(item, dict):
                    continue
                sym = str(item.get("symbol") or "").strip().upper()
                if not sym or sym in seen_symbols:
                    continue
                seen_symbols.add(sym)
                prefilter_samples.append({
                    "symbol": sym,
                    "passed": False,
                    "reason": item.get("code") or item.get("error") or "failure",
                })
                if len(prefilter_samples) >= 5:
                    break

            if not prefilter_samples:
                for item in enriched_candidates:
                    if not isinstance(item, dict):
                        continue
                    sym = str(item.get("symbol") or "").strip().upper()
                    if not sym or sym in seen_symbols:
                        continue
                    seen_symbols.add(sym)
                    prefilter_samples.append({
                        "symbol": sym,
                        "passed": True,
                        "reason": "candidate",
                    })
                    if len(prefilter_samples) >= 5:
                        break

        result["strategy"] = strategy
        result["session_date_used"] = session_date_used
        result["provider"] = provider
        result["thresholds_used"] = thresholds
        result["prefilter_samples"] = prefilter_samples
        result["failure_samples_by_code"] = failure_samples_by_code
        result["debug"] = {
            **debug_block,
            "failure_samples": failure_samples,
            "failure_samples_by_code": failure_samples_by_code,
        }

        return {
            "job_id": jid,
            "status": (job or {}).get("status"),
            "updated_at": (job or {}).get("updated_at"),
            "error": (job or {}).get("error"),
            "strategy": strategy,
            "scan_date": result.get("scan_date"),
            "session_date_used": session_date_used,
            "provider": provider,
            "thresholds_used": thresholds,
            "progress": progress,
            "prefilter_counts": result.get("prefilter_counts") or {},
            "prefilter_samples": prefilter_samples,
            "reject_counts": result.get("reject_counts") or {},
            "failure_samples_by_code": failure_samples_by_code,
            "data_failures": result.get("data_failures") or {},
            "debug": result.get("debug") or {},
            "candidates": enriched_candidates,
            "seed_candidates": result.get("seed_candidates") or [],
            "primary_candidates": result.get("primary_candidates") or [],
            "primary_mode": result.get("primary_mode"),
            "primary_message": result.get("primary_message"),
            "trade_ready_candidates": result.get("trade_ready_candidates") or [],
            "trade_ready_candidate_count": int(result.get("trade_ready_candidate_count") or 0),
            "fallback_candidates": result.get("fallback_candidates") or [],
            "fallback_candidate_count": int(result.get("fallback_candidate_count") or 0),
            "zero_result_diagnostics": result.get("zero_result_diagnostics") or {},
            "rejected_candidates": rejected_candidates,
            "result": result,
        }

    with _JOBS_LOCK:
        jobs = dict(_JOBS)

    if req_jid:
        job = jobs.get(req_jid)
        if not job:
            return jsonify(ok=False, error="unknown_job_id", job_id=req_jid), 404
        payload = _job_payload(req_jid, job)
        payload["ok"] = True
        return jsonify(payload)

    if not jobs:
        return jsonify(ok=False, error="no_scan_jobs"), 404

    items = sorted(jobs.items(), key=lambda kv: float((kv[1] or {}).get("updated_at") or 0.0), reverse=True)
    latest_jid, latest_job = items[0]
    running = next(((jid, job) for jid, job in items if (job or {}).get("status") == "running"), None)
    completed = next(((jid, job) for jid, job in items if (job or {}).get("status") in {"done", "completed"} and ((job or {}).get("result") or {})), None)

    payload = {
        "ok": True,
        "latest": _job_payload(latest_jid, latest_job),
        "current_running_scan": _job_payload(*running) if running else None,
        "last_completed_scan": _job_payload(*completed) if completed else None,
    }
    payload.update({k: v for k, v in payload["latest"].items() if k != "result"})
    return jsonify(payload)

@app.get("/scan")
def scan_start_get():
    """Start a scan via GET (UI fallback, no-JS).

    Uses the same worker as /api/scan_start and redirects back to / with job_id.
    """
    data = request.args.to_dict(flat=True) or {}
    jid = _new_job()
    _set_job(jid, params=data, updated_at=time.time())
    t = threading.Thread(target=_scan_worker, args=(jid,), daemon=True)
    t.start()

    from urllib.parse import urlencode
    q = dict(data)
    q["job_id"] = jid
    return redirect("/?" + urlencode(q))


@app.route("/", methods=["GET"])

def index():
    # Start ML bootstrap (load existing model or train it) in the background.
    _start_ml_bootstrap_once()

    # Always render dashboard immediately. Scans run via /api/scan_start in the background.
    job_id = (request.args.get("job_id") or "").strip()
    universe = request.args.get("universe", "nasdaq")
    max_symbols = _param_int(request.args, "max_symbols", 400)
    limit = _param_int(request.args, "limit", 25)
    offset = _param_int(request.args, "offset", 0)
    mode = request.args.get("mode", "until")

    # Balanced defaults (can be overridden by query params)
    min_rvol = _param_float(request.args, "min_rvol", 0.5)
    min_today_dollar_vol = _param_float(request.args, "min_today_dollar_vol", 2_000_000.0)
    min_avg20_dollar_vol = _param_float(request.args, "min_avg20_dollar_vol", 1_000_000.0)
    min_price = _param_float(request.args, "min_price", 1.0)
    max_price = _param_float(request.args, "max_price", 30.0)
    min_or_range_pct = _param_float(request.args, "min_or_range_pct", 0.6)
    max_or_range_pct = _param_float(request.args, "max_or_range_pct", 6.0)

    candidates = []
    scan_meta = None
    job = _get_job(job_id) if job_id else None
    job_running = False
    if job and job.get("status") == "running":
        job_running = True
        p = job.get("progress") or {}
        scan_meta = {
            "provider": (job.get("result") or {}).get("provider") or "alpaca",
            "progress": p,
            "scanned": int(p.get("scanned") or 0),
            "chunks": int(p.get("chunks_total") or 0),
            "end_offset": int(p.get("end_offset") or (p.get("offset") or 0)),
        }
    if job and job.get("status") == "done":
        out = job.get("result") or {}
        candidates = out.get("candidates") or []
        scan_meta = out
    elif job and job.get("status") == "error":
        scan_meta = {"error": job.get("error"), "progress": job.get("progress")}

    # Template expects expanded scan fields even when no scan has run yet.
    out_dict = scan_meta or {}
    provider = out_dict.get('provider', 'alpaca')
    prefilter_counts = out_dict.get('prefilter_counts') or {
        'normalized_skipped': 0,
        'daily_empty': 0,
        'filtered_price': 0,
        'filtered_avg20_dollar_vol': 0,
        'daily_ok': 0,
    }
    reject_counts = out_dict.get('reject_counts') or {
        'or_range': 0,
        'today_dollar_vol': 0,
        'rvol': 0,
        'risk_per_share': 0,
        'shares': 0,
        'notional': 0,
        'intraday_empty': 0,
        'intraday_or_window': 0,
        'other_data_failures': 0,
    }
    shortlisted = out_dict.get('shortlisted', 0)
    scanned = out_dict.get('scanned', 0)
    chunks = out_dict.get('chunks', 0)
    end_offset = out_dict.get('end_offset', offset)
    universe_size = out_dict.get('universe_size', 0)
    count = len(candidates)

    with _ML_LOCK:
        ml_state = dict(_ML_STATE)
    ml_state["model_loaded"] = (ml_state.get("status") == "ready")

    broker_snapshot = None
    if _BROKER_PROVIDER is not None and hasattr(_BROKER_PROVIDER, "get_broker_snapshot"):
        try:
            broker_snapshot = _BROKER_PROVIDER.get_broker_snapshot(timeout_s=8.0)
        except Exception:
            broker_snapshot = None

    return render_template(
        "index.html",
        job_running=job_running,
        universe=universe,
        max_symbols=max_symbols,
        limit=limit,
        offset=offset,
        mode=mode,
        min_rvol=min_rvol,
        min_today_dollar_vol=int(min_today_dollar_vol),
        min_avg20_dollar_vol=int(min_avg20_dollar_vol),
        min_price=min_price,
        max_price=max_price,
        min_or_range_pct=min_or_range_pct,
        max_or_range_pct=max_or_range_pct,
        provider=provider,
        broker_provider_name=_BROKER_PROVIDER_NAME,
        broker_provider_error=_BROKER_PROVIDER_ERROR,
        broker_actions_enabled=BROKER_ACTIONS_ENABLED,
        broker_snapshot=broker_snapshot,
        context_snapshot=_CONTEXT_ENGINE.snapshot(),
        recent_alerts=_RUNTIME_STORE.recent_alerts(limit=25),
        recent_news=_RUNTIME_STORE.recent_news(limit=25),
        prefilter_counts=prefilter_counts,
        reject_counts=reject_counts,
        shortlisted=shortlisted,
        scanned=scanned,
        chunks=chunks,
        end_offset=end_offset,
        universe_size=universe_size,
        count=count,
        candidates=candidates,
        out=scan_meta,
        job_id=job_id,
        job=job,
        ml_state=ml_state,
    )



def _col(df, *names):
    """Return the first matching column in df for any of the given names (case-insensitive).

    Tenant: no fake data. If the expected column is missing, we raise KeyError.
    """
    if df is None:
        raise KeyError(names[0] if names else "column")
    cols = list(getattr(df, "columns", []))
    # Pandas Index objects have ambiguous truthiness; never use `or []` here.
    lower = {str(c).lower(): c for c in cols}
    for n in names:
        key = str(n).lower()
        if key in lower:
            return df[lower[key]]
    raise KeyError(names[0] if names else "column")




@app.get("/crypto")
def crypto_page():
    return render_template("crypto.html", provider="alpaca", feed="crypto", context_snapshot=_CONTEXT_ENGINE.snapshot())

@app.get("/api/crypto_snapshot")
def api_crypto_snapshot():
    syms_raw = str(request.args.get("symbols") or os.getenv("ORB_CRYPTO_SYMBOLS", "BTC/USD,ETH/USD,SOL/USD")).strip()
    symbols = [s.strip().upper() for s in syms_raw.split(",") if s.strip()]
    rows = []
    if not symbols:
        return jsonify(ok=False, error="missing_symbols"), 400
    try:
        prov = _ALPACA_PROVIDER
        for sym in symbols[:25]:
            trade = None
            quote = None
            error = None
            try:
                # Raw Alpaca crypto snapshot endpoint via provider credentials.
                payload = prov._http_get_json(
                    f"https://data.alpaca.markets/v1beta3/crypto/us/snapshots",
                    params={"symbols": sym},
                    timeout_s=20.0,
                )
                snap = (payload.get("snapshots") or {}).get(sym) or {}
                latest_trade = snap.get("latestTrade") or {}
                latest_quote = snap.get("latestQuote") or {}
                minute_bar = snap.get("minuteBar") or {}
                price = latest_trade.get("p")
                minute_open = minute_bar.get("o")
                minute_high = minute_bar.get("h")
                minute_low = minute_bar.get("l")
                signal_side = None
                entry = None
                stop_loss = None
                target_2r = None
                target_3r = None
                risk_per_unit = None
                if minute_high is not None and minute_low is not None and price is not None:
                    try:
                        price_f = float(price)
                        high_f = float(minute_high)
                        low_f = float(minute_low)
                        if price_f >= high_f:
                            signal_side = "long"
                            entry = high_f
                            stop_loss = low_f
                        elif price_f <= low_f:
                            signal_side = "short"
                            entry = low_f
                            stop_loss = high_f
                        else:
                            signal_side = "watch"
                            entry = high_f
                            stop_loss = low_f
                        risk_per_unit = abs(float(entry) - float(stop_loss))
                        if risk_per_unit > 0:
                            target_2r = (float(entry) + 2.0 * risk_per_unit) if signal_side == "long" else (float(entry) - 2.0 * risk_per_unit)
                            target_3r = (float(entry) + 3.0 * risk_per_unit) if signal_side == "long" else (float(entry) - 3.0 * risk_per_unit)
                    except Exception:
                        pass
                rows.append({
                    "symbol": sym,
                    "price": price,
                    "trade_ts": latest_trade.get("t"),
                    "bid": latest_quote.get("bp"),
                    "ask": latest_quote.get("ap"),
                    "spread_pct": (((float(latest_quote.get("ap")) - float(latest_quote.get("bp"))) / (((float(latest_quote.get("ap")) + float(latest_quote.get("bp"))) / 2.0))) * 100.0) if latest_quote.get("bp") is not None and latest_quote.get("ap") is not None and (float(latest_quote.get("ap")) + float(latest_quote.get("bp"))) > 0 else None,
                    "minute_open": minute_open,
                    "minute_high": minute_high,
                    "minute_low": minute_low,
                    "minute_close": minute_bar.get("c"),
                    "minute_volume": minute_bar.get("v"),
                    "signal_side": signal_side,
                    "entry": entry,
                    "stop_loss": stop_loss,
                    "target_2r": target_2r,
                    "target_3r": target_3r,
                    "risk_per_unit": risk_per_unit,
                    "error": None,
                })
            except Exception as e:
                rows.append({"symbol": sym, "error": str(e)})
        return jsonify(ok=True, symbols=symbols, rows=rows)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

@app.get("/api/provider_info")
def provider_info():
    import os
    return jsonify({
        "market_data_provider_env": os.getenv("ORB_MARKET_DATA_PROVIDER"),
        "alpaca_feed_env": os.getenv("ALPACA_DATA_FEED"),
        "provider_object": type(_ALPACA_PROVIDER).__name__,
        "provider_name": getattr(_ALPACA_PROVIDER, "name", None),
    })


def _benchmark_scorecard_payload() -> dict[str, Any]:
    features = [
        {"key": "saved_presets", "label": "Server-saved presets", "kingdom": True, "benchmark": ["Trade Ideas", "TrendSpider", "TradingView"], "notes": "SQLite-backed presets with quick-load support."},
        {"key": "preset_metadata", "label": "Preset metadata", "kingdom": True, "benchmark": ["Trade Ideas", "TrendSpider"], "notes": "Favorites, notes, categories, and usage timestamps."},
        {"key": "candidate_explain", "label": "Structured candidate explain", "kingdom": True, "benchmark": ["Trade Ideas", "TrendSpider"], "notes": "Rank, plan, catalyst, and diagnostic sections."},
        {"key": "detail_chart", "label": "Decision chart hook", "kingdom": True, "benchmark": ["TradingView", "TrendSpider"], "notes": "Real intraday chart API with OR, VWAP, EMA, and risk overlays."},
        {"key": "news_timeline", "label": "Stored catalyst timeline", "kingdom": True, "benchmark": ["Benzinga", "Trade Ideas"], "notes": "Timeline rendered from stored runtime news events."},
        {"key": "alert_routing", "label": "Alert routing tiers", "kingdom": True, "benchmark": ["Trade Ideas"], "notes": "Near-trigger, ready, and triggered separation with suppression."},
        {"key": "monitor_replay", "label": "Monitor replay snapshot", "kingdom": True, "benchmark": ["TrendSpider"], "notes": "Replay snapshots persisted for monitor sessions."},
        {"key": "scanner_categories", "label": "Scanner categories", "kingdom": True, "benchmark": ["Trade Ideas", "Finviz"], "notes": "Momentum, news, reversal, and liquidity filters in UI."},
        {"key": "broker_sync", "label": "Broker sync", "kingdom": bool(BROKER_ACTIONS_ENABLED), "benchmark": ["Trade Ideas"], "notes": "Live broker snapshot and optional order actions when enabled."},
    ]
    supported = sum(1 for item in features if item["kingdom"])
    return {
        "updated_at": time.time(),
        "score": {
            "supported": supported,
            "total": len(features),
            "coverage_pct": round((supported / max(1, len(features))) * 100.0, 1),
        },
        "features": features,
    }



@app.get('/api/scan_presets')
def api_scan_presets():
    return jsonify(ok=True, presets=_RUNTIME_STORE.recent_scan_presets(limit=200))


@app.post('/api/scan_presets/save')
def api_scan_presets_save():
    data = request.get_json(silent=True) or request.form.to_dict(flat=True) or {}
    name = str(data.get('name') or '').strip()
    payload = data.get('payload') or {}
    meta = data.get('meta') or {}
    if not name:
        return jsonify(ok=False, error='missing_name'), 400
    if not isinstance(payload, dict):
        return jsonify(ok=False, error='invalid_payload'), 400
    if meta and not isinstance(meta, dict):
        return jsonify(ok=False, error='invalid_meta'), 400
    favorite = _is_truthy(meta.get('favorite') if isinstance(meta, dict) else data.get('favorite', 0))
    notes = meta.get('notes') if isinstance(meta, dict) else data.get('notes')
    category = meta.get('category') if isinstance(meta, dict) else data.get('category')
    last_used_at = _safe_float(meta.get('last_used_at') if isinstance(meta, dict) else data.get('last_used_at'))
    _RUNTIME_STORE.save_scan_preset(
        name,
        payload,
        favorite=favorite,
        notes=(str(notes).strip() if notes is not None else None),
        category=(str(category).strip() if category is not None else None),
        last_used_at=last_used_at,
    )
    return jsonify(ok=True, name=name)


@app.post('/api/scan_presets/touch')
def api_scan_presets_touch():
    data = request.get_json(silent=True) or request.form.to_dict(flat=True) or {}
    name = str(data.get('name') or '').strip()
    if not name:
        return jsonify(ok=False, error='missing_name'), 400
    touched = _RUNTIME_STORE.touch_scan_preset(name, last_used_at=_safe_float(data.get('last_used_at')))
    return jsonify(ok=True, touched=bool(touched), name=name)


@app.post('/api/scan_presets/delete')
def api_scan_presets_delete():
    data = request.get_json(silent=True) or request.form.to_dict(flat=True) or {}
    name = str(data.get('name') or '').strip()
    if not name:
        return jsonify(ok=False, error='missing_name'), 400
    deleted = _RUNTIME_STORE.delete_scan_preset(name)
    return jsonify(ok=True, deleted=bool(deleted), name=name)


# ── Desk Watchlist (pre-market, server-backed) ────────────────────────────────

@app.get('/api/desk_watchlist')
def api_desk_watchlist():
    return jsonify(ok=True, items=_RUNTIME_STORE.desk_watchlist_all())


@app.post('/api/desk_watchlist/set')
def api_desk_watchlist_set():
    data = request.get_json(silent=True) or request.form.to_dict(flat=True) or {}
    symbol = str(data.get('symbol') or '').strip().upper()
    if not symbol:
        return jsonify(ok=False, error='missing_symbol'), 400
    session_date = str(data.get('session_date') or '').strip() or None
    try:
        _RUNTIME_STORE.desk_watchlist_set(
            symbol,
            side=data.get('side') or None,
            trigger_price=_safe_float(data.get('trigger_price')),
            stop_price=_safe_float(data.get('stop_price')),
            target_price=_safe_float(data.get('target_price')),
            notes=str(data.get('notes') or '').strip() or None,
            session_date=session_date,
        )
        # Permanently log every save — allows recovery even if watchlist is cleared
        _RUNTIME_STORE.gap_plan_log_add(
            session_date=session_date or datetime.now(_ET).strftime('%Y-%m-%d'),
            symbol=symbol,
            side=data.get('side') or None,
            entry=_safe_float(data.get('trigger_price')),
            stop=_safe_float(data.get('stop_price')),
            target_2r=_safe_float(data.get('target_price')),
            gap_pct=_safe_float(data.get('gap_pct')),
            pm_high=_safe_float(data.get('pm_high')),
            pm_low=_safe_float(data.get('pm_low')),
            notes=str(data.get('notes') or '').strip() or None,
        )
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, symbol=symbol)


@app.post('/api/desk_watchlist/remove')
def api_desk_watchlist_remove():
    data = request.get_json(silent=True) or request.form.to_dict(flat=True) or {}
    symbol = str(data.get('symbol') or '').strip().upper()
    if not symbol:
        return jsonify(ok=False, error='missing_symbol'), 400
    removed = _RUNTIME_STORE.desk_watchlist_remove(symbol)
    return jsonify(ok=True, removed=bool(removed), symbol=symbol)


@app.post('/api/desk_watchlist/clear')
def api_desk_watchlist_clear():
    count = _RUNTIME_STORE.desk_watchlist_clear()
    return jsonify(ok=True, cleared=count)


@app.post('/api/desk_watchlist/purge_stale')
def api_desk_watchlist_purge_stale():
    today = datetime.now(_ET).strftime('%Y-%m-%d')
    purged = _RUNTIME_STORE.desk_watchlist_purge_stale(today)
    return jsonify(ok=True, purged=purged, count=len(purged))


@app.post('/api/desk_watchlist/bulk_add')
def api_desk_watchlist_bulk_add():
    """Add a list of candidates (already KTT-graded) to the desk watchlist."""
    data = request.get_json(force=True, silent=True) or {}
    candidates = data.get('candidates') or []
    if not candidates:
        return jsonify(ok=False, error='no candidates'), 400
    today = datetime.now(_ET).strftime('%Y-%m-%d')
    added = []
    for c in candidates:
        sym = str(c.get('symbol') or '').strip().upper()
        if not sym:
            continue
        side = str(c.get('side') or c.get('best_side') or 'long').lower()
        entry  = _safe_float(c.get('long_entry')  if side == 'long' else c.get('short_entry'))
        stop   = _safe_float(c.get('long_stop')   if side == 'long' else c.get('short_stop'))
        target = _safe_float(c.get('long_2r')     if side == 'long' else c.get('short_2r'))
        ktt    = c.get('ktt_grade') or '?'
        score  = c.get('ktt_score') or 0
        try:
            _RUNTIME_STORE.desk_watchlist_set(
                sym, side=side,
                trigger_price=entry, stop_price=stop, target_price=target,
                notes=f"scanner B+ auto-add — KTT {ktt} {score}",
                session_date=today,
            )
            added.append(sym)
        except Exception:
            pass
    return jsonify(ok=True, added=added, count=len(added))


@app.get('/api/screener_seeds')
def api_screener_seeds():
    from scanner.orb import _screener_market_movers, _screener_most_actives
    seed_type = request.args.get('type', 'movers')
    try:
        if seed_type == 'gainers':
            gainers, _ = _screener_market_movers(_ALPACA_PROVIDER, top=50)
            symbols = gainers
        elif seed_type == 'losers':
            _, losers = _screener_market_movers(_ALPACA_PROVIDER, top=50)
            symbols = losers
        elif seed_type == 'movers':
            gainers, losers = _screener_market_movers(_ALPACA_PROVIDER, top=50)
            actives = _screener_most_actives(_ALPACA_PROVIDER, top=50)
            seen: set = set()
            symbols: list = []
            for s in actives + gainers + losers:
                if s not in seen:
                    symbols.append(s)
                    seen.add(s)
        else:
            return jsonify(ok=False, error=f'unknown type: {seed_type}'), 400
        return jsonify(ok=True, symbols=symbols, count=len(symbols))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.get('/api/know_the_trade')
def api_know_the_trade():
    sym = (request.args.get('symbol') or '').strip().upper()
    if not sym:
        return jsonify(ok=False, error='symbol required'), 400
    try:
        from utils.know_the_trade import analyze
        # Count today's halts for this symbol from the stream cache
        halt_count = 0
        try:
            if _STREAM is not None:
                evts = _STREAM.recent_halt_resume_events(max_age_sec=86400)
                halt_count = sum(1 for e in evts
                                 if e.get('symbol','').upper() == sym
                                 and e.get('halt_status','').lower() in ('halted','trading_halt','h'))
        except Exception:
            pass

        _entry  = _safe_float(request.args.get('entry'))
        _stop   = _safe_float(request.args.get('stop'))
        _target = _safe_float(request.args.get('target'))
        _side   = request.args.get('side') or 'long'

        # ── Auto-compute live RVOL from snapshot when not provided ────────────
        _rvol_hint = _safe_float(request.args.get('rvol_hint'))
        _live_price_for_ml = None
        if _rvol_hint is None and _ALPACA_PROVIDER is not None:
            try:
                _snaps = _ALPACA_PROVIDER.get_snapshots([sym], feed='sip', timeout_s=6.0) or {}
                _snap = _snaps.get(sym) or {}
                _daily = _snap.get('daily_bar') or {}
                _avg_vol = float(_snap.get('avg_daily_volume') or 0)
                _today_vol = float(_daily.get('volume') or 0)
                if _today_vol > 0 and _avg_vol > 0:
                    _rvol_hint = round(_today_vol / _avg_vol, 2)
                # grab last trade price for ML scoring
                _lt2 = _snap.get('latest_trade') or {}
                _lp2 = float(_lt2.get('price') or 0)
                if _lp2 > 0:
                    _live_price_for_ml = _lp2
            except Exception:
                pass

        # ── Auto-compute ML score when not supplied ───────────────────────────
        _ml_score = _safe_float(request.args.get('ml_score'))
        if _ml_score is None and _ALPACA_PROVIDER is not None:
            try:
                from ml.orb_model_service import score_orb_symbol as _ktt_score_ml
                _px_for_ml = _live_price_for_ml or _entry or 0
                if _px_for_ml and _px_for_ml > 0:
                    _ml_out = _ktt_score_ml(sym, last_price=float(_px_for_ml), provider=_ALPACA_PROVIDER)
                    if _ml_out.get('score') is not None:
                        _ml_score = float(_ml_out['score'])
            except Exception:
                pass

        # ── Auto-fetch catalyst from sniper buffer when not supplied ──────────
        _catalyst = request.args.get('catalyst_headline') or None
        _cat_age  = _safe_float(request.args.get('catalyst_age_hours'))
        if _catalyst is None:
            try:
                with _SNIPER_LOCK:
                    _recent = [a for a in _SNIPER_ALERTS if str(a.get('symbol','')).upper() == sym]
                if _recent:
                    _best = _recent[0]
                    _catalyst = _best.get('headline') or None
                    _fired_ts = _best.get('fired_ts')
                    if _fired_ts and _catalyst:
                        _cat_age = round((time.time() - float(_fired_ts)) / 3600, 2)
            except Exception:
                pass

        # ── Auto-infer setup_age_hours from time since RTH open ──────────────
        _setup_age = _safe_float(request.args.get('setup_age_hours'))
        if _setup_age is None:
            try:
                import pytz as _pytz_ktt
                _et_ktt = _pytz_ktt.timezone('America/New_York')
                _now_et = datetime.now(_et_ktt)
                _rth_open = _now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                if _now_et > _rth_open:
                    _setup_age = round((_now_et - _rth_open).total_seconds() / 3600, 2)
            except Exception:
                pass

        # Target-hit guard
        if _target is not None and _ALPACA_PROVIDER is not None:
            try:
                _lt = _ALPACA_PROVIDER.get_latest_trade(sym) or {}
                _live = float(_lt.get('price') or 0)
                if _live > 0:
                    target_hit = (_side == 'long' and _live >= _target) or \
                                 (_side == 'short' and _live <= _target)
                    if target_hit:
                        return jsonify(
                            ok=True,
                            symbol=sym, grade='X', grade_color='#64748b',
                            score=0,
                            grade_advice='Target already hit — setup is expired. Do not enter.',
                            breakdown={},
                            sizing={'tiers': [], 'note': 'Setup expired'},
                            target_hit=True,
                            live_price=_live,
                        )
            except Exception:
                pass

        result = analyze(
            symbol=sym,
            provider=_ALPACA_PROVIDER,
            entry=_entry,
            stop=_stop,
            target=_target,
            side=_side,
            pm_last=_safe_float(request.args.get('pm_last')),
            pm_high=_safe_float(request.args.get('pm_high')),
            pm_low=_safe_float(request.args.get('pm_low')),
            pm_vol=int(float(request.args.get('pm_vol') or 0)) or None,
            pm_move_pct=_safe_float(request.args.get('pm_move_pct')),
            ml_score=_ml_score,
            catalyst_headline=_catalyst,
            catalyst_age_hours=_cat_age,
            rvol_hint=_rvol_hint,
            halt_count=halt_count,
            ssr_active=request.args.get('ssr_active', '').lower() in ('1', 'true', 'yes'),
            stop_capped=request.args.get('stop_capped', '').lower() in ('1', 'true', 'yes'),
            stop_cap=_safe_float(request.args.get('stop_cap')),
            setup_age_hours=_setup_age,
            setup_status=request.args.get('setup_status') or None,
            gap_fill_rate=_safe_float(request.args.get('gap_fill_rate')),
            vwap=_safe_float(request.args.get('vwap')),
        )
        return jsonify(ok=True, **result)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post('/api/ktt_grades_batch')
def api_ktt_grades_batch():
    """Batch KTT grading for the scanner list view. Returns {symbol: {grade, score, color}}."""
    try:
        from utils.know_the_trade import analyze as _ktt_analyze
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

        body = request.get_json(force=True, silent=True) or {}
        candidates = body.get('candidates') or []
        if not candidates:
            # Help callers who send the wrong key
            alt_keys = [k for k in ('setups', 'symbols', 'items', 'data') if body.get(k)]
            if alt_keys:
                return jsonify(ok=False, error=f"wrong payload key '{alt_keys[0]}' — use 'candidates'"), 400
            return jsonify(ok=True, grades={})

        halt_map: dict[str, int] = {}
        try:
            if _STREAM is not None:
                evts = _STREAM.recent_halt_resume_events(max_age_sec=86400)
                for e in evts:
                    s = str(e.get('symbol', '')).upper()
                    if e.get('halt_status', '').lower() in ('halted', 'trading_halt', 'h'):
                        halt_map[s] = halt_map.get(s, 0) + 1
        except Exception:
            pass

        # ── Pre-compute RTH age once for all candidates ──────────────────────
        _batch_rth_age: float | None = None
        try:
            import pytz as _batch_pytz
            _batch_et = _batch_pytz.timezone('America/New_York')
            _batch_now = datetime.now(_batch_et)
            _batch_open = _batch_now.replace(hour=9, minute=30, second=0, microsecond=0)
            if _batch_now > _batch_open:
                _batch_rth_age = round((_batch_now - _batch_open).total_seconds() / 3600, 2)
        except Exception:
            pass

        # ── Pre-index sniper alerts for catalyst lookup ───────────────────────
        _batch_catalyst_map: dict[str, tuple[str, float]] = {}
        try:
            with _SNIPER_LOCK:
                for _a in _SNIPER_ALERTS:
                    _asym = str(_a.get('symbol', '')).upper()
                    if _asym and _asym not in _batch_catalyst_map:
                        _hl = _a.get('headline') or ''
                        _ft = float(_a.get('fired_ts') or 0)
                        if _hl and _ft:
                            _batch_catalyst_map[_asym] = (_hl, round((time.time() - _ft) / 3600, 2))
        except Exception:
            pass

        def _grade_one(c: dict) -> tuple[str, dict]:
            sym = str(c.get('symbol') or '').upper()
            if not sym:
                return sym, {}
            side = str(c.get('side') or c.get('best_side') or 'long').lower()
            entry  = _safe_float(c.get('long_entry') if side == 'long' else c.get('short_entry'))
            stop   = _safe_float(c.get('long_stop')  if side == 'long' else c.get('short_stop'))
            target = _safe_float(c.get('long_2r')    if side == 'long' else c.get('short_2r'))

            # setup_age_hours: prefer explicit → scan_ts → RTH elapsed
            setup_age_hours = _safe_float(c.get('setup_age_hours'))
            if setup_age_hours is None:
                try:
                    scan_ts_raw = c.get('scan_ts')
                    if scan_ts_raw:
                        import dateutil.parser as _dp
                        scanned_at = _dp.parse(str(scan_ts_raw))
                        if scanned_at.tzinfo is None:
                            scanned_at = scanned_at.replace(tzinfo=timezone.utc)
                        setup_age_hours = (datetime.now(timezone.utc) - scanned_at).total_seconds() / 3600
                except Exception:
                    pass
            if setup_age_hours is None:
                setup_age_hours = _batch_rth_age

            # ml_score: prefer candidate value, else auto-score
            ml_score = _safe_float(c.get('ml_score'))
            if ml_score is None and _ALPACA_PROVIDER is not None:
                try:
                    from ml.orb_model_service import score_orb_symbol as _b_score_ml
                    _px = entry or 0
                    if _px > 0:
                        _b_ml_out = _b_score_ml(sym, last_price=float(_px), provider=_ALPACA_PROVIDER)
                        if _b_ml_out.get('score') is not None:
                            ml_score = float(_b_ml_out['score'])
                except Exception:
                    pass

            # rvol: prefer candidate value, else auto-fetch
            rvol_hint = _safe_float(c.get('rvol'))
            if rvol_hint is None and _ALPACA_PROVIDER is not None:
                try:
                    _b_snaps = _ALPACA_PROVIDER.get_snapshots([sym], feed='sip', timeout_s=5.0) or {}
                    _b_snap  = _b_snaps.get(sym) or {}
                    _b_daily = _b_snap.get('daily_bar') or {}
                    _b_avg   = float(_b_snap.get('avg_daily_volume') or 0)
                    _b_vol   = float(_b_daily.get('volume') or 0)
                    if _b_vol > 0 and _b_avg > 0:
                        rvol_hint = round(_b_vol / _b_avg, 2)
                except Exception:
                    pass

            # catalyst: prefer candidate value, else sniper buffer
            catalyst_headline = c.get('news_headline') or None
            catalyst_age_hours = _safe_float(c.get('news_age_hours'))
            if catalyst_headline is None and sym in _batch_catalyst_map:
                catalyst_headline, catalyst_age_hours = _batch_catalyst_map[sym]

            setup_status = 'triggered' if (
                (side == 'long' and c.get('long_triggered')) or
                (side == 'short' and c.get('short_triggered'))
            ) else None
            try:
                result = _ktt_analyze(
                    symbol=sym, provider=_ALPACA_PROVIDER,
                    entry=entry, stop=stop, target=target, side=side,
                    ml_score=ml_score,
                    catalyst_headline=catalyst_headline,
                    catalyst_age_hours=catalyst_age_hours,
                    rvol_hint=rvol_hint,
                    halt_count=halt_map.get(sym, 0),
                    setup_age_hours=setup_age_hours,
                    setup_status=setup_status,
                    vwap=_safe_float(c.get('vwap_last')),
                )
                grade = str(result.get('grade') or '?')
                score = int(result.get('score') or 0)
                color = {'A': '#4ade80', 'B': '#fbbf24', 'C': '#f97316', 'D': '#f87171'}.get(grade, '#94a3b8')
                return sym, {'grade': grade, 'score': score, 'color': color,
                             'ml_score': ml_score, 'rvol': result.get('live_rvol')}
            except Exception as ex:
                return sym, {'grade': '?', 'score': 0, 'color': '#94a3b8', 'error': str(ex)}

        grades: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(len(candidates), 12)) as pool:
            futs = {pool.submit(_grade_one, c): c for c in candidates}
            for fut in _as_completed(futs):
                sym, data = fut.result()
                if sym:
                    grades[sym] = data

        return jsonify(ok=True, grades=grades)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.get('/api/morning_prep')
def api_morning_prep():
    max_price = float(request.args.get('max_price', 30.0))
    min_rvol  = float(request.args.get('min_rvol',  1.5))
    try:
        from utils.morning_prep import fetch_morning_prep
        data = fetch_morning_prep(max_price=max_price, min_rvol=min_rvol)
        return jsonify(ok=True, **data)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.get('/api/premarket_trade_plan')
def api_premarket_trade_plan():
    """
    Pre-market trade recommendations with gap fade probability, halt prediction,
    float rotation, distribution detection, and catalyst decay timer.
    Synthesizes morning prep earnings + gap movers into specific trade plans.
    """
    risk_dollars = float(request.args.get('risk_dollars', 100.0))
    max_price    = float(request.args.get('max_price', 30.0))
    min_gap_pct  = float(request.args.get('min_gap_pct', 5.0))
    limit        = int(request.args.get('limit', 20))

    try:
        from utils.morning_prep import fetch_earnings_calendar
        from utils.premarket_intelligence import build_premarket_trade_plan
        from providers.alpaca_provider import AlpacaProvider

        provider = AlpacaProvider()

        # Pull earnings names with gap data
        earnings = fetch_earnings_calendar(lookback_days=0, forward_days=1, max_price=max_price)
        today_pm = [e for e in earnings if e.get('days_away', 99) == 0 and e.get('gap_pct') is not None
                    and abs(e.get('gap_pct', 0)) >= min_gap_pct]

        if not today_pm:
            # Fall back to all earnings with gap data
            today_pm = [e for e in earnings if e.get('gap_pct') is not None
                        and abs(e.get('gap_pct', 0)) >= min_gap_pct]

        today_pm.sort(key=lambda e: -abs(e.get('gap_pct', 0)))
        today_pm = today_pm[:limit]

        if not today_pm:
            return jsonify(ok=True, plans=[], note="No earnings gaps meeting criteria today")

        syms = [e['symbol'] for e in today_pm]

        # Get snapshots for RVOL and volume data
        snap_map: dict[str, dict] = {}
        try:
            snaps = provider.get_snapshots(syms, feed='sip', timeout_s=15.0)
            snap_map = snaps or {}
        except Exception:
            pass

        # Get T&S from stream cache if available
        sc = _STREAM

        plans = []
        for e in today_pm:
            sym  = e['symbol']
            snap = snap_map.get(sym) or {}
            db   = snap.get('daily_bar') or {}
            pb   = snap.get('prev_daily_bar') or {}
            lt   = snap.get('latest_trade') or {}
            lq   = snap.get('latest_quote') or {}

            price      = float(lt.get('price') or db.get('close') or e.get('last_price') or 0)
            prev_close = float(pb.get('close') or 0)
            gap_pct    = float(e.get('gap_pct') or 0)
            side       = str(e.get('side') or ('long' if gap_pct >= 0 else 'short'))
            today_vol  = float(db.get('volume') or 0)
            avg20_dvol = None

            rvol = None
            if today_vol and prev_close:
                prev_vol = float((snap.get('prev_daily_bar') or {}).get('volume') or 0)
                if prev_vol > 0:
                    rvol = round(today_vol / prev_vol, 2)

            bid = float(lq.get('bid_price') or 0) or None
            ask = float(lq.get('ask_price') or 0) or None

            trades: list[dict] = []
            if sc is not None:
                try:
                    raw = sc.recent_trades(sym, limit=50)
                    prev_p = None
                    for tr in raw:
                        p = tr.get('price')
                        tick = 'up' if (prev_p and p and p > prev_p) else ('down' if (prev_p and p and p < prev_p) else 'flat')
                        trades.append({**tr, 'tick': tick})
                        if p:
                            prev_p = p
                except Exception:
                    pass

            # Determine catalyst type from earnings context
            cat_type = 'earnings_beat_small'
            mktcap = str(e.get('market_cap') or '').lower()
            if 'large' in mktcap or any(c in mktcap for c in ['b', 't']):
                cat_type = 'earnings_beat_large'
            elif 'mid' in mktcap:
                cat_type = 'earnings_beat_mid'

            if not price or not prev_close:
                continue

            plan = build_premarket_trade_plan(
                symbol=sym,
                price=price,
                prev_close=prev_close,
                gap_pct=gap_pct,
                side=side,
                rvol=rvol,
                float_shares=None,
                above_vwap=None,
                today_volume=today_vol,
                catalyst_type=cat_type,
                news_age_hours=None,
                pm_bars=None,
                trades=trades,
                bid=bid,
                ask=ask,
                risk_dollars=risk_dollars,
            )
            plan['name']        = e.get('name', '')
            plan['report_time'] = e.get('time', '')
            plan['eps_forecast'] = e.get('eps_forecast', '')
            plans.append(plan)

        plans.sort(key=lambda p: -p['quality_score'])

        return jsonify(ok=True, plans=plans, generated_at=datetime.now(timezone.utc).isoformat())

    except Exception as e:
        log.exception("premarket_trade_plan error")
        return jsonify(ok=False, error=str(e)), 500


@app.get('/api/symbol_intelligence')
def api_symbol_intelligence():
    """
    Full intelligence profile for a single symbol: fade probability,
    halt prediction, float rotation, distribution detection, catalyst decay.
    """
    sym = (request.args.get('symbol') or '').strip().upper()
    if not sym:
        return jsonify(ok=False, error='symbol required'), 400

    risk_dollars  = float(request.args.get('risk_dollars', 100.0))
    catalyst_type = request.args.get('catalyst_type', 'unknown')
    float_shares  = request.args.get('float_shares')
    float_shares  = float(float_shares) if float_shares else None
    news_age_hours = request.args.get('news_age_hours')
    news_age_hours = float(news_age_hours) if news_age_hours else None

    try:
        from utils.premarket_intelligence import build_premarket_trade_plan
        from providers.alpaca_provider import AlpacaProvider

        provider = AlpacaProvider()
        snaps = provider.get_snapshots([sym], feed='sip', timeout_s=10.0)
        snap  = (snaps or {}).get(sym) or {}
        db    = snap.get('daily_bar') or {}
        pb    = snap.get('prev_daily_bar') or {}
        lt    = snap.get('latest_trade') or {}
        lq    = snap.get('latest_quote') or {}

        price      = float(lt.get('price') or db.get('close') or 0)
        prev_close = float(pb.get('close') or 0)
        today_vol  = float(db.get('volume') or 0)
        prev_vol   = float(pb.get('volume') or 0)
        rvol       = round(today_vol / prev_vol, 2) if prev_vol > 0 else None
        gap_pct    = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0
        side       = 'long' if gap_pct >= 0 else 'short'
        bid        = float(lq.get('bid_price') or 0) or None
        ask        = float(lq.get('ask_price') or 0) or None

        trades: list[dict] = []
        sc = _STREAM
        if sc is not None:
            try:
                raw = sc.recent_trades(sym, limit=100)
                prev_p = None
                for tr in raw:
                    p = tr.get('price')
                    tick = 'up' if (prev_p and p and p > prev_p) else ('down' if (prev_p and p and p < prev_p) else 'flat')
                    trades.append({**tr, 'tick': tick})
                    if p:
                        prev_p = p
            except Exception:
                pass

        plan = build_premarket_trade_plan(
            symbol=sym, price=price, prev_close=prev_close,
            gap_pct=gap_pct, side=side, rvol=rvol,
            float_shares=float_shares, today_volume=today_vol,
            catalyst_type=catalyst_type, news_age_hours=news_age_hours,
            trades=trades, bid=bid, ask=ask, risk_dollars=risk_dollars,
        )

        return jsonify(ok=True, **plan)

    except Exception as e:
        log.exception("symbol_intelligence error")
        return jsonify(ok=False, error=str(e)), 500


@app.get('/api/gap_plan_log')
def api_gap_plan_log():
    session_date = request.args.get('session_date') or None
    limit = int(request.args.get('limit') or 200)
    rows = _RUNTIME_STORE.gap_plan_log_recent(session_date=session_date, limit=limit)
    return jsonify(ok=True, rows=rows)


# ── Pre-market scan (runs premarket_watchlist.py in background) ───────────────

import subprocess as _subprocess
import threading as _threading
import uuid as _uuid

_PM_SCAN_JOBS: dict[str, dict] = {}
_PM_SCAN_LOCK = _threading.Lock()


@app.post('/api/premarket_scan')
def api_premarket_scan_start():
    """Start a background pre-market scan job. Returns job_id."""
    data = request.get_json(silent=True) or {}
    # Prevent concurrent runs
    with _PM_SCAN_LOCK:
        for job in _PM_SCAN_JOBS.values():
            if job.get('status') == 'running':
                return jsonify(ok=True, reused=True, job_id=job['job_id'],
                               message='Scan already running.')

    min_gap  = float(data.get('min_gap', 2.0))
    min_vol  = int(data.get('min_vol', 50000))
    sources  = str(data.get('sources', 'alpaca,finviz,stocktwits'))
    top      = int(data.get('top', 15))

    job_id = _uuid.uuid4().hex
    job: dict = {'job_id': job_id, 'status': 'running', 'lines': [], 'error': None}
    with _PM_SCAN_LOCK:
        _PM_SCAN_JOBS[job_id] = job

    script = Path(__file__).resolve().parent / 'tools' / 'premarket_watchlist.py'

    def _run():
        try:
            cmd = [
                sys.executable, str(script),
                '--sources', sources,
                '--min-gap', str(min_gap),
                '--min-vol', str(min_vol),
                '--top', str(top),
                '--add',
            ]
            proc = _subprocess.Popen(
                cmd,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.STDOUT,
                text=True,
                cwd=str(Path(__file__).resolve().parent),
                env={**os.environ},
            )
            for line in proc.stdout:
                line = line.rstrip('\n')
                with _PM_SCAN_LOCK:
                    job['lines'].append(line)
            proc.wait()
            with _PM_SCAN_LOCK:
                job['status'] = 'done' if proc.returncode == 0 else 'error'
                if proc.returncode != 0:
                    job['error'] = f'exit code {proc.returncode}'
        except Exception as exc:
            with _PM_SCAN_LOCK:
                job['status'] = 'error'
                job['error'] = str(exc)

    _threading.Thread(target=_run, daemon=True).start()
    return jsonify(ok=True, job_id=job_id)


@app.get('/api/premarket_scan/status')
def api_premarket_scan_status():
    job_id = request.args.get('job_id', '')
    with _PM_SCAN_LOCK:
        job = _PM_SCAN_JOBS.get(job_id)
    if not job:
        return jsonify(ok=False, error='not_found'), 404
    return jsonify(ok=True, status=job['status'], lines=list(job['lines']),
                   error=job.get('error'))

# ─────────────────────────────────────────────────────────────────────────────

# ── Pre-market Gap Order Plans ────────────────────────────────────────────────

@app.get('/api/premarket_gap_orders')
def api_premarket_gap_orders():
    """
    For each symbol in the desk watchlist (or ?symbols=A,B,C override),
    fetch today's pre-market 1m bars, compute PM high/low, and return a
    concrete order plan (entry, stop, target 2R) based on the gap direction.
    """
    import math as _gom

    provider = _ALPACA_PROVIDER
    if provider is None:
        return jsonify(ok=False, error='provider_not_ready'), 503

    # Symbol list: query param override or desk watchlist
    sym_param = request.args.get('symbols', '').strip()
    if sym_param:
        raw_syms = [s.strip().upper() for s in sym_param.split(',') if s.strip()]
        _saved_sides: dict[str, str] = {}
    else:
        entries = _RUNTIME_STORE.desk_watchlist_all()
        raw_syms = [e['symbol'] for e in entries if e.get('symbol')]
        # Snapshot existing sides so we can detect direction reversals
        _saved_sides = {e['symbol']: (e.get('side') or 'long') for e in entries}

    if not raw_syms:
        return jsonify(ok=True, plans=[], note='desk_watchlist_empty')

    risk_dollars = float(request.args.get('risk_dollars', 50.0))

    today_et = datetime.now(_ET).date()

    # Roll back to the most recent trading day with actual bar data.
    # Two cases where today has no bars:
    #   1. Weekend (Sat/Sun) — no trading.
    #   2. Before 4:00 AM ET on a weekday — pre-market hasn't opened yet.
    # In both cases fall back to the previous trading weekday so the endpoint
    # always returns a session with real data.
    now_et_dt = datetime.now(_ET)
    trade_date = today_et
    while trade_date.weekday() >= 5:
        trade_date = trade_date - timedelta(days=1)

    pm_open_cutoff = datetime.combine(trade_date, datetime.min.time()).replace(
        hour=4, minute=0, second=0, microsecond=0, tzinfo=_ET)
    if now_et_dt < pm_open_cutoff:
        # Before 4 AM on the current trading day — roll back one more day
        trade_date = trade_date - timedelta(days=1)
        while trade_date.weekday() >= 5:
            trade_date = trade_date - timedelta(days=1)

    pm_open_t  = datetime.combine(trade_date, datetime.min.time()).replace(
        hour=4, minute=0, second=0, microsecond=0,
        tzinfo=_ET)
    pm_close_t = datetime.combine(trade_date, datetime.min.time()).replace(
        hour=9, minute=30, second=0, microsecond=0,
        tzinfo=_ET)

    plans = []
    errors = []

    def _build_plan(sym: str):
        try:
            bars = provider.get_bars_range(
                symbol=sym, interval='1m',
                from_d=trade_date, to_d=trade_date,
                include_prepost=True, timeout_s=10, feed='sip',
            )
            if bars is None or bars.empty:
                return None, {'symbol': sym, 'error': 'no_bars'}

            idx = bars.index
            if hasattr(idx, 'tz') and idx.tz is None:
                idx = idx.tz_localize('UTC')
            idx_et = idx.tz_convert(_ET)

            pm_mask = (idx_et >= pm_open_t) & (idx_et < pm_close_t)
            pm_bars = bars[pm_mask]
            if pm_bars.empty:
                # Fallback to RTH bars for symbols that skip premarket
                _rth_open_t2 = datetime.combine(trade_date, datetime.min.time()).replace(
                    hour=9, minute=30, second=0, microsecond=0, tzinfo=_ET)
                _rth_close_t2 = datetime.combine(trade_date, datetime.min.time()).replace(
                    hour=16, minute=0, second=0, microsecond=0, tzinfo=_ET)
                _rth_mask2 = (idx_et >= _rth_open_t2) & (idx_et <= _rth_close_t2)
                pm_bars = bars[_rth_mask2]
            if pm_bars.empty:
                return None, {'symbol': sym, 'error': 'no_premarket_bars'}

            prev_close = None
            try:
                daily = provider.get_daily_history(sym, period='5d')
                if daily is not None and not daily.empty:
                    didx = daily.index
                    if hasattr(didx, 'tz') and didx.tz is not None:
                        didx_et = didx.tz_convert(_ET)
                    else:
                        didx_et = didx.tz_localize('UTC').tz_convert(_ET)
                    trade_date_start = datetime.combine(trade_date, datetime.min.time()).replace(tzinfo=_ET)
                    prior = daily[didx_et < trade_date_start]
                    if not prior.empty:
                        prev_close = float(prior['Close'].iloc[-1])
            except Exception:
                pass
            if prev_close is None:
                return None, {'symbol': sym, 'error': 'cannot_determine_prev_close'}

            pm_bars = pm_bars.sort_index()
            from core.plan_integrity import validate_bars_input as _vbi
            _bar_chk = _vbi(pm_bars, symbol=sym, require_sorted=True, min_bars=1)
            if not _bar_chk.valid:
                return None, {'symbol': sym, 'error': f'bars_invalid: {_bar_chk.violations}'}

            pm_high = float(pm_bars['High'].max())
            pm_low  = float(pm_bars['Low'].min())
            pm_vol  = int(pm_bars['Volume'].sum())
            pm_last = float(pm_bars['Close'].iloc[-1])

            gap_pct = (pm_last - prev_close) / prev_close * 100.0 if prev_close > 0 else 0.0
            _saved_sym = _saved_sides.get(sym)
            if abs(gap_pct) >= 2.0:
                side = 'long' if gap_pct > 0 else 'short'
            elif _saved_sym:
                side = _saved_sym
            else:
                side = 'long' if gap_pct > 0 else 'short'

            if side == 'long':
                entry = round(pm_high + 0.01, 2)
                stop  = round(pm_low, 2)
            else:
                entry = round(pm_low - 0.01, 2)
                stop  = round(pm_high, 2)

            risk_per_share = round(abs(entry - stop), 4)
            if risk_per_share < 0.05:
                return None, {'symbol': sym, 'error': f'pm_range_too_tight ({risk_per_share:.3f})'}

            _max_risk = round(entry * 0.07, 4)
            if risk_per_share > _max_risk:
                if side == 'long':
                    stop = round(entry - _max_risk, 2)
                else:
                    stop = round(entry + _max_risk, 2)
                risk_per_share = round(abs(entry - stop), 4)

            _risk_pct = (risk_per_share / entry * 100.0) if entry > 0 else 999.0
            if _risk_pct > 12.0:
                return None, {'symbol': sym, 'error': f'risk_pct_too_wide ({_risk_pct:.1f}%)'}

            if side == 'long':
                target_2r = round(entry + 2.0 * risk_per_share, 2)
            else:
                target_2r = round(entry - 2.0 * risk_per_share, 2)

            shares = max(1, int(_gom.floor(risk_dollars / risk_per_share)))

            open_price        = None
            open_through_stop = False
            open_above_entry  = False
            open_slippage     = 0.0

            now_et = datetime.now(_ET)
            mkt_open_t = datetime.combine(today_et, datetime.min.time()).replace(
                hour=9, minute=30, second=0, microsecond=0, tzinfo=_ET)

            if now_et >= mkt_open_t:
                rth_mask = idx_et >= mkt_open_t
                rth_bars = bars[rth_mask]
                if not rth_bars.empty:
                    open_price = round(float(rth_bars['Open'].iloc[0]), 2)
                    if side == 'long':
                        open_through_stop = open_price <= stop
                        open_above_entry  = open_price > entry
                        if open_above_entry:
                            open_slippage = round(open_price - entry, 4)
                            actual_risk   = round(abs(open_price - stop), 4)
                            shares        = max(1, int(_gom.floor(risk_dollars / actual_risk)))
                    else:
                        open_through_stop = open_price >= stop
                        open_above_entry  = open_price < entry
                        if open_above_entry:
                            open_slippage = round(entry - open_price, 4)
                            actual_risk   = round(abs(open_price - stop), 4)
                            shares        = max(1, int(_gom.floor(risk_dollars / actual_risk)))

            return {
                'symbol':            sym,
                'side':              side,
                'gap_pct':           round(gap_pct, 2),
                'prev_close':        round(prev_close, 2),
                'pm_high':           round(pm_high, 2),
                'pm_low':            round(pm_low, 2),
                'pm_vol':            pm_vol,
                'pm_last':           round(pm_last, 2),
                'entry':             entry,
                'stop':              stop,
                'target_2r':         target_2r,
                'risk_per_share':    round(risk_per_share, 4),
                'shares':            shares,
                'notional':          round(entry * shares, 2),
                'open_price':        open_price,
                'open_through_stop': open_through_stop,
                'open_above_entry':  open_above_entry,
                'open_slippage':     round(open_slippage, 4),
            }, None

        except Exception as exc:
            return None, {'symbol': sym, 'error': str(exc)}

    from concurrent.futures import ThreadPoolExecutor, as_completed as _asc
    workers = min(len(raw_syms), 12)
    with ThreadPoolExecutor(max_workers=workers) as _pool:
        futs = {_pool.submit(_build_plan, sym): sym for sym in raw_syms}
        for fut in _asc(futs):
            plan, err = fut.result()
            if plan:
                plans.append(plan)
            elif err:
                errors.append(err)

    # Drop plans that opened through the stop — structurally dead at the open.
    plans = [p for p in plans if not p.get('open_through_stop')]

    # Final integrity gate: validate every plan before it reaches the UI.
    from core.plan_integrity import validate_plan as _vp
    clean_plans = []
    for _p in plans:
        _chk = _vp(
            side=_p.get('side'),
            entry=_p.get('entry'),
            stop=_p.get('stop'),
            target=_p.get('target_2r'),
            current_price=None,  # pre-market: no live price yet
        )
        if _chk.valid:
            clean_plans.append(_p)
        else:
            errors.append({'symbol': _p.get('symbol'), 'error': f'integrity: {_chk.violations}'})
    plans = clean_plans

    plans.sort(key=lambda p: abs(p.get('gap_pct', 0)), reverse=True)

    # Auto-sync desk watchlist side to match the current gap direction.
    # The watchlist may have been saved from a prior session (e.g. Friday EOD as LONG)
    # but Monday pre-market the stock gaps DOWN → update to SHORT to avoid contradiction.
    reversed_syms: list[str] = []
    for p in plans:
        prev_side = _saved_sides.get(p['symbol'])
        if prev_side and prev_side != p['side']:
            reversed_syms.append(p['symbol'])
        try:
            # Only overwrite watchlist side when the PM gap is meaningful (≥2%).
            # Trivial moves must not silently flip a strong EOD signal.
            if abs(p.get('gap_pct', 0)) >= 2.0:
                _RUNTIME_STORE.desk_watchlist_update_side(p['symbol'], p['side'])
        except Exception:
            pass

    return jsonify(ok=True, plans=plans, errors=errors, risk_dollars=risk_dollars,
                   reversed_symbols=reversed_syms)

@app.get('/api/gap_watchlist_analysis')
def api_gap_watchlist_analysis():
    """
    For each symbol (desk watchlist OR ?symbols= override), fetch pre-market data
    and return entry/stop/target plan, setup status, and EOD quality indicators.
    """
    import math as _gma

    provider = _ALPACA_PROVIDER
    if provider is None:
        return jsonify(ok=False, error='provider_not_ready'), 503

    # Caller can pass an explicit symbol list (e.g. scan candidates).
    # Format: ?symbols=AAPL,TSLA,NVDA  OR  ?symbols=AAPL%20TSLA%20NVDA
    sym_param = (request.args.get('symbols') or '').strip()
    if sym_param:
        import re as _re
        _override_syms = [s.strip().upper() for s in _re.split(r'[\s,]+', sym_param) if s.strip()]
        # Build synthetic entries with no saved levels — fresh plan will be computed
        entries = [{'symbol': s, 'side': None, 'trigger_price': None,
                    'stop_price': None, 'target_price': None} for s in _override_syms]
    else:
        entries = _RUNTIME_STORE.desk_watchlist_all()

    if not entries:
        return jsonify(ok=True, rows=[], note='desk_watchlist_empty')

    today_et  = datetime.now(_ET).date()
    now_et_dt = datetime.now(_ET)

    # Resolve the current trading day (same logic as gap orders)
    trade_date = today_et
    while trade_date.weekday() >= 5:
        trade_date = trade_date - timedelta(days=1)
    pm_open_cutoff = datetime.combine(trade_date, datetime.min.time()).replace(
        hour=4, minute=0, second=0, microsecond=0, tzinfo=_ET)
    if now_et_dt < pm_open_cutoff:
        trade_date = trade_date - timedelta(days=1)
        while trade_date.weekday() >= 5:
            trade_date = trade_date - timedelta(days=1)

    pm_open_t  = datetime.combine(trade_date, datetime.min.time()).replace(
        hour=4, minute=0, second=0, microsecond=0, tzinfo=_ET)
    # If trade_date is a past session (weekend or pre-4am rollback), show the full
    # extended session (4am–8pm) so the user sees where the stock actually closed.
    # When viewing today's live session, keep the traditional PM window (4am–9:30am).
    if trade_date < today_et:
        pm_close_t = datetime.combine(trade_date, datetime.min.time()).replace(
            hour=20, minute=0, second=0, microsecond=0, tzinfo=_ET)
    else:
        pm_close_t = datetime.combine(trade_date, datetime.min.time()).replace(
            hour=9, minute=30, second=0, microsecond=0, tzinfo=_ET)

    # Load PM ML bundle once for all symbols
    _ml_model = None
    _ml_feat_names = []
    _ml_fillna = -999.0
    try:
        _pm_path = _ML_STATE.get("entry_now_pm_path", "")
        if _pm_path:
            _b = _load_entry_now_bundle(_pm_path)
            _ml_feat_names = list(_b.get("feature_names") or [])
            _ml_fillna = float(_b.get("fillna_value", -999.0) or -999.0)
            _ml_model = _b.get("model")
    except Exception:
        pass

    rows = []
    errors = []

    # Process each symbol concurrently — 3 network calls per symbol (bars, daily, latest_trade)
    # stack linearly when serial; parallelism cuts wall time to ~1 slowest symbol
    def _analyze_one(entry):
        sym = (entry.get('symbol') or '').strip().upper()
        if not sym:
            return

        saved_side    = entry.get('side') or 'long'
        saved_entry   = _safe_float(entry.get('trigger_price'))
        saved_stop    = _safe_float(entry.get('stop_price'))
        saved_target  = _safe_float(entry.get('target_price'))

        try:
            bars = provider.get_bars_range(
                symbol=sym, interval='1m',
                from_d=trade_date, to_d=trade_date,
                include_prepost=True, timeout_s=10, feed='sip',
            )
            if bars is None or bars.empty:
                errors.append({'symbol': sym, 'error': 'no_bars'})
                return

            idx    = bars.index
            if hasattr(idx, 'tz') and idx.tz is None:
                idx = idx.tz_localize('UTC')
            idx_et = idx.tz_convert(_ET)

            pm_mask = (idx_et >= pm_open_t) & (idx_et < pm_close_t)
            pm_bars = bars[pm_mask]

            # Prev day OHLC from daily history (60d needed for avg20 RVOL)
            prev_close = prev_open = prev_high = prev_low = prev_vol = None
            daily = None
            try:
                daily = provider.get_daily_history(sym, period='60d')
                if daily is not None and not daily.empty:
                    didx = daily.index
                    if hasattr(didx, 'tz') and didx.tz is not None:
                        didx_et = didx.tz_convert(_ET)
                    else:
                        didx_et = didx.tz_localize('UTC').tz_convert(_ET)
                    trade_date_start = datetime.combine(trade_date, datetime.min.time()).replace(tzinfo=_ET)
                    prior = daily[didx_et < trade_date_start]
                    if not prior.empty:
                        prev_row   = prior.iloc[-1]
                        prev_close = float(prev_row['Close'])
                        prev_open  = float(prev_row['Open'])
                        prev_high  = float(prev_row['High'])
                        prev_low   = float(prev_row['Low'])
                        prev_vol   = int(prev_row['Volume']) if 'Volume' in prev_row.index else None
            except Exception:
                pass

            # ── EOD metrics (yesterday's session) ────────────────────────────
            eod_move_pct    = None   # how much stock moved on the day (open→close)
            eod_close_vs_rng = None  # where close landed in day's range (0=LOD, 1=HOD)
            rvol_eod        = None   # yesterday's RVOL vs avg20
            if prev_open and prev_close and prev_high and prev_low:
                if prev_open > 0:
                    eod_move_pct = round((prev_close - prev_open) / prev_open * 100.0, 2)
                rng = prev_high - prev_low
                if rng > 0:
                    eod_close_vs_rng = round((prev_close - prev_low) / rng, 3)

            # RVOL: yesterday's volume vs 20-day avg (excluding yesterday)
            try:
                if daily is not None and not daily.empty and prev_vol:
                    from scanner.indicators import avg_daily_volume as _avg_dv
                    _daily_s = daily.sort_index()
                    _didx2 = _daily_s.index
                    if hasattr(_didx2, 'tz') and _didx2.tz is not None:
                        _didx2_et = _didx2.tz_convert(_ET)
                    else:
                        _didx2_et = _didx2.tz_localize('UTC').tz_convert(_ET)
                    _tds = datetime.combine(trade_date, datetime.min.time()).replace(tzinfo=_ET)
                    _prior20 = _daily_s[_didx2_et < _tds]
                    if len(_prior20) >= 5:
                        _avg20 = float(_avg_dv(_prior20, window=20) or 0.0)
                        if _avg20 > 0:
                            rvol_eod = round(prev_vol / _avg20, 2)
            except Exception:
                pass

            # ── Float-adjusted RVOL (float turnover %) ───────────────────────
            float_shares = None
            float_turnover_pct = None
            try:
                from scanner.orb import _fetch_float_shares as _ffs
                float_shares = _ffs(sym, getattr(_ALPACA_PROVIDER, '_store', None))
                if float_shares and float_shares > 0 and pm_vol > 0:
                    float_turnover_pct = round(pm_vol / float_shares * 100.0, 2)
            except Exception:
                pass

            # ── Historical gap-fill rate (last 90 sessions, two-tier: close-fill vs wick-fill) ──
            gap_fill_rate = None
            gap_fill_rate_close = None
            gap_fill_n = 0
            try:
                if daily is not None and not daily.empty and len(prior) >= 10:
                    _gf = prior.sort_index().iloc[-90:]
                    _go = _gf['Open'].astype(float).values
                    _gc = _gf['Close'].astype(float).values
                    _gh = _gf['High'].astype(float).values
                    _gl = _gf['Low'].astype(float).values
                    _gaps_found = 0
                    _gaps_filled_wick = 0   # intraday wick touched prev close (within 0.5%)
                    _gaps_filled_close = 0  # session closed on the fill side of prev close
                    for _gi in range(1, len(_gf)):
                        _pc = _gc[_gi - 1]
                        if _pc <= 0:
                            continue
                        _g = (_go[_gi] - _pc) / _pc * 100.0
                        if abs(_g) < 1.0:
                            continue
                        _gaps_found += 1
                        if _g > 0:
                            # Gap up: filled if low wicked back to within 0.5% of prev close
                            if _gl[_gi] <= _pc * 1.005:
                                _gaps_filled_wick += 1
                            # Close fill: session closed at or below prev close
                            if _gc[_gi] <= _pc:
                                _gaps_filled_close += 1
                        else:
                            # Gap down: filled if high wicked back to within 0.5% of prev close
                            if _gh[_gi] >= _pc * 0.995:
                                _gaps_filled_wick += 1
                            # Close fill: session closed at or above prev close
                            if _gc[_gi] >= _pc:
                                _gaps_filled_close += 1
                    if _gaps_found >= 3:
                        gap_fill_rate = round(_gaps_filled_wick / _gaps_found, 3)
                        gap_fill_rate_close = round(_gaps_filled_close / _gaps_found, 3)
                        gap_fill_n = _gaps_found
            except Exception:
                pass

            # ── 5-day historical context ──────────────────────────────────────
            atr_5 = high_5d = low_5d = trend_5d = accum_score_5d = vol_trend_5d = None
            try:
                if daily is not None and not daily.empty and len(prior) >= 2:
                    _d5 = prior.iloc[-5:] if len(prior) >= 5 else prior
                    _h5 = _d5['High'].astype(float)
                    _l5 = _d5['Low'].astype(float)
                    _c5 = _d5['Close'].astype(float)
                    high_5d = round(float(_h5.max()), 2)
                    low_5d  = round(float(_l5.min()), 2)
                    # ATR(5): mean true range over last 5 sessions
                    _trs = []
                    for _i5 in range(len(_d5)):
                        _h = _h5.iloc[_i5]; _l = _l5.iloc[_i5]
                        _pc = _c5.iloc[_i5 - 1] if _i5 > 0 else _c5.iloc[_i5]
                        _trs.append(max(_h - _l, abs(_h - _pc), abs(_l - _pc)))
                    atr_5 = round(sum(_trs) / len(_trs), 3) if _trs else None
                    # Trend: count up vs down closes
                    _up = sum(1 for _i5 in range(1, len(_c5)) if _c5.iloc[_i5] > _c5.iloc[_i5 - 1])
                    _dn = (len(_c5) - 1) - _up
                    trend_5d = 'uptrend' if _up >= _dn + 2 else ('downtrend' if _dn >= _up + 2 else 'sideways')
                    # Accumulation score: avg position of close in day range (1=HOD, 0=LOD)
                    _cvr = []
                    for _i5 in range(len(_d5)):
                        _rng = _h5.iloc[_i5] - _l5.iloc[_i5]
                        if _rng > 0:
                            _cvr.append((_c5.iloc[_i5] - _l5.iloc[_i5]) / _rng)
                    accum_score_5d = round(sum(_cvr) / len(_cvr), 3) if _cvr else None
                    # Volume trend
                    if 'Volume' in _d5.columns and len(_d5) >= 4:
                        _vols = _d5['Volume'].astype(float).tolist()
                        _v1 = sum(_vols[:len(_vols)//2]); _v2 = sum(_vols[len(_vols)//2:])
                        vol_trend_5d = 'rising' if _v2 > _v1 * 1.1 else ('falling' if _v2 < _v1 * 0.9 else 'flat')
            except Exception:
                pass

            ssr_active = False  # updated after pm_last is known; closure captures by ref

            def _eod_row_base():
                return {
                    'symbol':            sym,
                    'saved_side':        saved_side,
                    'saved_entry':       saved_entry,
                    'saved_stop':        saved_stop,
                    'saved_target':      saved_target,
                    # EOD context
                    'prev_open':         round(prev_open, 2)  if prev_open  else None,
                    'prev_high':         round(prev_high, 2)  if prev_high  else None,
                    'prev_low':          round(prev_low, 2)   if prev_low   else None,
                    'prev_close':        round(prev_close, 2) if prev_close else None,
                    'prev_vol':          prev_vol,
                    'eod_move_pct':      eod_move_pct,
                    'eod_close_vs_rng':  eod_close_vs_rng,
                    # 4 setup-quality indicators
                    'close_vs_rng_pct':  round(eod_close_vs_rng * 100, 1) if eod_close_vs_rng is not None else None,
                    'rvol_eod':          rvol_eod,
                    'catalyst_age_hours': None,   # filled after loop via batch news fetch
                    'catalyst_headline':  None,
                    # Reg SHO SSR flag
                    'ssr_active':          ssr_active,
                    # Float-adjusted RVOL
                    'float_shares':        int(float_shares) if float_shares else None,
                    'float_turnover_pct':  float_turnover_pct,
                    # Historical gap-fill rate (wick=intraday touch, close=session close fill)
                    'gap_fill_rate':       gap_fill_rate,
                    'gap_fill_rate_close': gap_fill_rate_close,
                    'gap_fill_n':          gap_fill_n,
                    # 5-day historical context
                    'atr_5':             atr_5,
                    'high_5d':           high_5d,
                    'low_5d':            low_5d,
                    'trend_5d':          trend_5d,
                    'accum_score_5d':    accum_score_5d,
                    'vol_trend_5d':      vol_trend_5d,
                }

            if pm_bars.empty:
                # Fallback: use RTH bars (9:30–16:00) for stocks that don't trade premarket
                _rth_open_t = datetime.combine(trade_date, datetime.min.time()).replace(
                    hour=9, minute=30, second=0, microsecond=0, tzinfo=_ET)
                _rth_close_t = datetime.combine(trade_date, datetime.min.time()).replace(
                    hour=16, minute=0, second=0, microsecond=0, tzinfo=_ET)
                _rth_mask = (idx_et >= _rth_open_t) & (idx_et <= _rth_close_t)
                pm_bars = bars[_rth_mask]
                if pm_bars.empty:
                    row = _eod_row_base()
                    row.update({
                        'pm_last': None, 'pm_high': None, 'pm_low': None, 'pm_vol': 0,
                        'pm_move_pct': None, 'move_from_entry': None,
                        'status': 'no_pm_data', 'status_label': 'No PM data yet',
                        'ml_score': None,
                    })
                    rows.append(row)
                    return

            pm_bars  = pm_bars.sort_index()  # ensure chronological; iloc[-1] must be latest bar
            from core.plan_integrity import validate_bars_input as _vbi
            _bar_chk = _vbi(pm_bars, symbol=sym, require_sorted=True, min_bars=1)
            if not _bar_chk.valid:
                errors.append({'symbol': sym, 'error': f'bars_invalid: {_bar_chk.violations}'})
                return

            pm_high  = float(pm_bars['High'].max())
            pm_low   = float(pm_bars['Low'].min())
            pm_last  = float(pm_bars['Close'].iloc[-1])
            pm_vol   = int(pm_bars['Volume'].sum())

            # Reg SHO SSR: stock down ≥10% from prior close restricts shorting to uptick only
            try:
                if prev_close and prev_close > 0:
                    ssr_active = ((pm_last - prev_close) / prev_close * 100.0) <= -10.0
            except Exception:
                pass

            # ML score using PM bars + entry_now_30m_pm.pkl
            ml_score = None
            if _ml_model is not None and hasattr(_ml_model, 'predict_proba') and len(pm_bars) >= 20:
                try:
                    from ml.entry_now_dataset import _opening_range_5m_from_1m, _today_dollar_vol_so_far
                    from scanner.indicators import vwap as _ind_vwap, trend_state_1m as _ind_trend, avg_daily_volume as _avg_dv
                    import numpy as _np
                    import pandas as _pd

                    _sess = pm_bars.sort_index()
                    _i = len(_sess) - 1
                    _vw = _ind_vwap(_sess)
                    _tr = _ind_trend(_sess, vw=_vw, lookback=15)
                    _ts = str(_tr.get("state") or "")
                    try:
                        _orh, _orl = _opening_range_5m_from_1m(_sess)
                        _mid = (_orh + _orl) / 2.0
                        _or_range_pct = ((_orh - _orl) / max(1e-9, _mid)) * 100.0
                        _dist_orh = ((pm_last - _orh) / max(1e-9, _orh)) * 100.0
                        _dist_orl = ((pm_last - _orl) / max(1e-9, _orl)) * 100.0
                    except Exception:
                        _or_range_pct = _dist_orh = _dist_orl = _np.nan
                    _dv_so_far = float(_today_dollar_vol_so_far(_sess, _i))
                    _avg20_dv = _np.nan
                    _rvol = _np.nan
                    if daily is not None and not daily.empty:
                        try:
                            _d = daily.sort_index()
                            _day_utc = _sess.index[0].normalize()
                            _d_prior = _d[_d.index.normalize() < _day_utc]
                            if len(_d_prior) >= 60:
                                _a20v = float(_avg_dv(_d_prior, window=20) or 0.0)
                                _pc = float(_d_prior['Close'].astype(float).iloc[-1])
                                _avg20_dv = float(_a20v * _pc) if (_a20v > 0 and _pc > 0) else _np.nan
                                _tvol = float(_sess['Volume'].astype(float).iloc[:_i+1].sum())
                                _frac = max(1e-6, (_i + 1) / float(len(_sess)))
                                _rvol = (_tvol / max(1e-9, _a20v * _frac)) if _a20v > 0 else _np.nan
                        except Exception:
                            pass
                    _stop_val = float(saved_stop) if saved_stop is not None else pm_last
                    _feat = {
                        'side_long': 1.0 if saved_side == 'long' else 0.0,
                        'entry': float(pm_last),
                        'stop_dist_pct': (abs(pm_last - _stop_val) / max(1e-9, pm_last)) * 100.0,
                        'vwap_delta_pct': float(_tr.get("vwap_delta_pct")) if _tr.get("vwap_delta_pct") is not None else _np.nan,
                        'trend_slope_pct': float(_tr.get("slope_pct_lookback")) if _tr.get("slope_pct_lookback") is not None else _np.nan,
                        'trend_state_up': 1.0 if _ts in ('up', 'reclaim_vwap') else 0.0,
                        'trend_state_down': 1.0 if _ts in ('down', 'lost_vwap') else 0.0,
                        'trend_state_chop': 1.0 if _ts == 'chop' else 0.0,
                        'or_range_pct': float(_or_range_pct),
                        'dist_orh_pct': float(_dist_orh),
                        'dist_orl_pct': float(_dist_orl),
                        'today_dollar_vol_so_far': float(_dv_so_far),
                        'avg20_dollar_vol': float(_avg20_dv),
                        'rvol_now': float(_rvol),
                        'minutes_since_open': float((_sess.index[_i].astimezone(_ET).hour * 60 + _sess.index[_i].astimezone(_ET).minute) - (4 * 60)) if _i < len(_sess) else float(_i),  # Fix #18: clock minutes from 4am ET
                    }
                    _X = _pd.DataFrame([_feat])
                    if _ml_feat_names:
                        for _c in _ml_feat_names:
                            if _c not in _X.columns:
                                _X[_c] = _np.nan
                        _X = _X[_ml_feat_names]
                    _X = _X.replace([_np.inf, -_np.inf], _np.nan).fillna(_ml_fillna)
                    ml_score = round(float(_ml_model.predict_proba(_X)[0, 1]), 4)
                except Exception:
                    pass

            # PM move = close-to-close vs prev session close
            pm_move_pct = round((pm_last - prev_close) / prev_close * 100.0, 2) if prev_close and prev_close > 0 else None

            # ── Fresh gap-plan levels from today's PM bars ───────────────────
            # Same math as api_premarket_gap_orders so the analyze card is self-
            # contained — no need to also run "Generate Orders".
            import math as _gom_a
            gap_pct_fresh = round((pm_last - prev_close) / prev_close * 100.0, 2) if prev_close and prev_close > 0 else None
            # Mirror the ≥2% threshold from api_premarket_gap_orders: small gaps are noise.
            # Zero gap must NOT default to LONG — use saved side when direction is unclear.
            if gap_pct_fresh is not None and abs(gap_pct_fresh) >= 2.0:
                fresh_side = 'long' if gap_pct_fresh > 0 else 'short'
            else:
                fresh_side = saved_side  # preserve EOD conviction when PM move is trivial
            if fresh_side == 'long':
                fresh_entry = round(pm_high + 0.01, 2)
                fresh_stop  = round(pm_low, 2)   # natural stop = PM low
            else:
                fresh_entry = round(pm_low - 0.01, 2)
                fresh_stop  = round(pm_high, 2)  # natural stop = PM high
            fresh_risk = round(abs(fresh_entry - fresh_stop), 4)
            _max_risk_a = round(fresh_entry * 0.07, 4)  # unified with gap_orders cap
            stop_capped = False
            fresh_stop_cap = None  # informational display only — sizing/target use natural stop
            if fresh_risk > _max_risk_a:
                stop_capped = True
                if fresh_side == 'long':
                    fresh_stop_cap = round(fresh_entry - _max_risk_a, 2)
                else:
                    fresh_stop_cap = round(fresh_entry + _max_risk_a, 2)
                # fresh_stop and fresh_risk stay as natural — target and sizing both use real stop
            if fresh_risk >= 0.05:
                fresh_target = round(fresh_entry + 2.0 * fresh_risk, 2) if fresh_side == 'long' \
                               else round(fresh_entry - 2.0 * fresh_risk, 2)
            else:
                fresh_entry = fresh_stop = fresh_target = None

            # Gate: fresh plan must pass structural integrity before reaching the UI
            if fresh_entry is not None:
                from core.plan_integrity import validate_plan as _vp_a
                _chk_a = _vp_a(side=fresh_side, entry=fresh_entry, stop=fresh_stop,
                                target=fresh_target, current_price=None, symbol=sym)
                if not _chk_a.valid:
                    fresh_entry = fresh_stop = fresh_target = None

            # ── Rebuilt plan from last 30-min PM bars (for invalidated setups) ─
            # When the full-session PM range is blown (price past stop), rebuild from
            # the most recent 30-min consolidation so the user gets a valid setup for open.
            rebuilt_entry = rebuilt_stop = rebuilt_target = rebuilt_side = None
            try:
                if len(pm_bars) >= 5:
                    # Anchor to 9:00-9:30 AM window only.
                    # pm_close_t is 8 PM when trade_date is yesterday, so pm_mask spans the full
                    # extended session. We must clamp both ends to get the true pre-open range.
                    _rth_open_t = datetime.combine(trade_date, datetime.min.time()).replace(
                        hour=9, minute=30, second=0, microsecond=0, tzinfo=_ET)
                    _cutoff_30 = _rth_open_t - timedelta(minutes=30)
                    _recent_idx = idx_et[pm_mask]
                    _recent_mask_inner = (_recent_idx >= _cutoff_30) & (_recent_idx < _rth_open_t)
                    _recent_pm = pm_bars[_recent_mask_inner.values]
                    if len(_recent_pm) >= 3:
                        _r_high = float(_recent_pm['High'].max())
                        _r_low  = float(_recent_pm['Low'].min())
                        _r_last = float(_recent_pm['Close'].iloc[-1])
                        _r_gap  = (_r_last - prev_close) / prev_close * 100.0 if prev_close and prev_close > 0 else 0.0
                        rebuilt_side = 'long' if _r_gap > 0 else 'short'
                        if rebuilt_side == 'long':
                            rebuilt_entry = round(_r_high + 0.01, 2)
                            rebuilt_stop  = round(_r_low, 2)
                        else:
                            rebuilt_entry = round(_r_low - 0.01, 2)
                            rebuilt_stop  = round(_r_high, 2)
                        _r_risk = round(abs(rebuilt_entry - rebuilt_stop), 4)
                        _r_max  = round(rebuilt_entry * 0.07, 4)
                        if _r_risk > _r_max:
                            if rebuilt_side == 'long':
                                rebuilt_stop = round(rebuilt_entry - _r_max, 2)
                            else:
                                rebuilt_stop = round(rebuilt_entry + _r_max, 2)
                            _r_risk = round(abs(rebuilt_entry - rebuilt_stop), 4)
                        if _r_risk >= 0.05:
                            rebuilt_target = round(rebuilt_entry + 2.0 * _r_risk, 2) if rebuilt_side == 'long' \
                                             else round(rebuilt_entry - 2.0 * _r_risk, 2)
                            from core.plan_integrity import validate_plan as _vp_r
                            # Use _r_last (9:00-9:30 AM close) not live price — rebuilt plan is
                            # a historical anchor, live price is irrelevant to its structural validity
                            _r_chk = _vp_r(side=rebuilt_side, entry=rebuilt_entry, stop=rebuilt_stop,
                                           target=rebuilt_target, current_price=None, symbol=sym)
                            if not _r_chk.valid:
                                log.debug(f"rebuilt_plan [{sym}] invalid: {_r_chk.violations}")
                                rebuilt_entry = rebuilt_stop = rebuilt_target = rebuilt_side = None
                        else:
                            rebuilt_entry = rebuilt_stop = rebuilt_target = rebuilt_side = None
            except Exception as _re:
                log.warning(f"rebuilt_plan [{sym}]: {_re}")

            # After 9:30 use live trade price for status — pm_last is stale once RTH opens.
            # Fallback priority if get_latest_trade fails:
            #   1. Last 1-min bar close (at most ~1 min stale, already in memory)
            #   2. pm_last (only if bars also unavailable — hours stale, last resort)
            _now_et_s = datetime.now(_ET)
            _is_rth = (_now_et_s.hour > 9 or (_now_et_s.hour == 9 and _now_et_s.minute >= 30)) \
                      and _now_et_s.hour < 16
            _live_price = pm_last
            if _is_rth:
                # Best fallback: last close from already-fetched 1-min bars (~1 min stale max)
                try:
                    _last_bar_close = float(bars.iloc[-1]['Close'])
                    if _last_bar_close > 0:
                        _live_price = _last_bar_close
                except Exception:
                    pass
                # Override with true latest trade if available (zero latency)
                try:
                    _lt = _ALPACA_PROVIDER.get_latest_trade(sym) or {}
                    _lp = float(_lt.get('price') or 0)
                    if _lp > 0:
                        _live_price = _lp
                except Exception:
                    pass  # keep last bar close set above

            # How far has price moved toward/past the fresh entry (fresh_entry matches displayed Entry column)
            move_from_entry = None
            _ref_entry = saved_entry if saved_entry is not None else fresh_entry
            _ref_side  = saved_side or fresh_side
            if _ref_entry and _ref_entry > 0:
                if _ref_side == 'long':
                    move_from_entry = round((_live_price - _ref_entry) / _ref_entry * 100.0, 2)
                else:
                    move_from_entry = round((_ref_entry - _live_price) / _ref_entry * 100.0, 2)

            # Setup status — saved levels from the scanner are authoritative.
            # fresh_* (re-derived from PM bars) is only used when no saved levels exist.
            # This ensures a scanner entry of $9.86 isn't silently replaced by the
            # PM-recomputed $7.64 when the user added the setup from an intraday scan.
            _eval_entry  = saved_entry  if saved_entry  is not None else fresh_entry
            _eval_target = saved_target if saved_target is not None else fresh_target
            _eval_side   = saved_side or fresh_side or 'long'
            # Stop check uses saved_stop when available (the scanner's structural stop).
            # PM low/high is only a fallback when no stop was explicitly saved.
            # Using pm_low as the primary stop caused false invalidations when live
            # price dipped below PM low but was still above the scanner's actual stop.
            _pm_natural_stop = float(pm_low) if _eval_side == 'long' else float(pm_high)
            _eval_stop = saved_stop if saved_stop is not None else _pm_natural_stop
            if _eval_entry is not None:
                if _eval_side == 'long':
                    if _eval_target is not None and _live_price >= _eval_target:
                        status, label = 'target_hit', 'Target already hit — setup complete, do not chase'
                    elif _live_price <= _eval_stop:
                        status, label = 'stopped_out', 'Stop violated — setup invalidated'
                    elif _live_price >= _eval_entry:
                        status, label = 'triggered', 'At/above entry — already triggered'
                    elif _live_price >= _eval_entry * 0.995:
                        status, label = 'near_entry', 'Within 0.5% of entry — watch closely'
                    else:
                        status, label = 'intact', 'Setup intact — waiting for entry'
                else:
                    if _eval_target is not None and _live_price <= _eval_target:
                        status, label = 'target_hit', 'Target already hit — setup complete, do not chase'
                    elif _live_price >= _eval_stop:
                        status, label = 'stopped_out', 'Stop violated — setup invalidated'
                    elif _live_price <= _eval_entry:
                        status, label = 'triggered', 'At/below entry — already triggered'
                    elif _live_price <= _eval_entry * 1.005:
                        status, label = 'near_entry', 'Within 0.5% of entry — watch closely'
                    else:
                        status, label = 'intact', 'Setup intact — waiting for entry'
            else:
                status, label = 'no_levels', 'No entry/stop saved'

            # Time decay: hours since PM close, same-day RTH only
            _setup_age_hours = None
            try:
                from datetime import time as _dtime
                _now_age = datetime.now(_ET)
                _pm_close_tod = datetime.combine(_now_age.date(), _dtime(9, 30), tzinfo=_ET)
                _is_rth_now = (_now_age.hour > 9 or (_now_age.hour == 9 and _now_age.minute >= 30)) \
                              and _now_age.hour < 16
                if _is_rth_now and trade_date == _now_age.date():
                    _setup_age_hours = round((_now_age - _pm_close_tod).total_seconds() / 3600.0, 2)
            except Exception:
                pass

            row = _eod_row_base()
            row.update({
                'pm_last':         round(pm_last, 2),
                'pm_high':         round(pm_high, 2),
                'pm_low':          round(pm_low, 2),
                'pm_vol':          pm_vol,
                'pm_move_pct':     pm_move_pct,
                'gap_pct':         gap_pct_fresh,
                'fresh_side':      fresh_side,
                'fresh_entry':     fresh_entry,
                'fresh_stop':      fresh_stop,
                'fresh_target':    fresh_target,
                'rebuilt_side':    rebuilt_side,
                'rebuilt_entry':   rebuilt_entry,
                'rebuilt_stop':    rebuilt_stop,
                'rebuilt_target':  rebuilt_target,
                'move_from_entry': move_from_entry,
                'status':          status,
                'status_label':    label,
                'ml_score':        ml_score,
                'stop_capped':     stop_capped,
                'fresh_stop_cap':  fresh_stop_cap,
                'setup_age_hours': _setup_age_hours,
            })
            rows.append(row)

        except Exception as exc:
            errors.append({'symbol': sym, 'error': str(exc)})

    from concurrent.futures import ThreadPoolExecutor as _AnalysisPool
    with _AnalysisPool(max_workers=8) as _pool:
        list(_pool.map(_analyze_one, entries))

    # Batch news fetch — fills catalyst_age_hours + catalyst_headline for all rows
    if rows:
        try:
            from scanner.orb import _fetch_alpaca_news_batch
            _syms = [r['symbol'] for r in rows]
            _news = _fetch_alpaca_news_batch(_syms, provider, lookback_hours=48)
            for r in rows:
                _n = _news.get(r['symbol'])
                if _n:
                    r['catalyst_headline']  = _n[0]
                    r['catalyst_age_hours'] = round(_n[1], 1)
        except Exception:
            pass

    # Fix #16: status-priority sort (intact > near_entry > triggered > stopped_out/target_hit)
    # then by magnitude within each priority group
    _status_priority = {'intact': 0, 'near_entry': 1, 'triggered': 2,
                        'no_levels': 3, 'target_hit': 4, 'stopped_out': 5, 'no_pm_data': 6}
    rows.sort(key=lambda r: (
        _status_priority.get(r.get('status', 'no_pm_data'), 9),
        -abs(r.get('pm_move_pct') or 0),
    ))
    return jsonify(ok=True, rows=rows, errors=errors, trade_date=str(trade_date))

# ─────────────────────────────────────────────────────────────────────────────

# ── Paper Trade Monitor — live control & positions desk ───────────────────────

_PT_STATE: dict = {
    'proc': None, 'running': False, 'log_path': None,
    'start_time_iso': None, 'watchlist_only': False,
}
_PT_LOCK = threading.Lock()


def _pt_ensure_running() -> bool:
    """Launch paper_trade_monitor.py if it isn't already running. Returns True if started."""
    with _PT_LOCK:
        proc = _PT_STATE.get('proc')
        if proc and proc.poll() is None:
            return False  # already alive
        # Also check for orphaned processes from prior sessions
        try:
            import subprocess as _sp
            _check = _sp.run(['pgrep', '-f', 'paper_trade_monitor.py'],
                             capture_output=True, text=True)
            if _check.stdout.strip():
                _PT_STATE['running'] = True
                return False
        except Exception:
            pass
        script = Path(__file__).resolve().parent / 'tools' / 'paper_trade_monitor.py'
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent), env={**os.environ},
        )
        _PT_STATE.update({
            'proc': proc, 'running': True, 'log_path': None,
            'start_time_iso': datetime.now(timezone.utc).isoformat(),
            'watchlist_only': False,
        })
    return True

_PT_VAL_STATE: dict = {'running': False, 'lines': [], 'done': False, 'error': None}
_PT_VAL_LOCK = threading.Lock()


def _find_pt_log_today() -> str | None:
    """Return path to the most recently written JSONL log for today (ET).
    Kept for backwards-compat; prefer _find_all_pt_logs_today()."""
    log_dir = Path(os.getenv('PT_LOG_DIR', '/tmp'))
    today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d')
    files = sorted(log_dir.glob(f'kingdom_pt_{today}*.jsonl'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def _find_all_pt_logs_today() -> list[str]:
    """Return ALL of today's JSONL session files, oldest-first (chronological merge order)."""
    log_dir = Path(os.getenv('PT_LOG_DIR', '/tmp'))
    today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d')
    files = sorted(log_dir.glob(f'kingdom_pt_{today}*.jsonl'),
                   key=lambda p: p.stat().st_mtime)
    return [str(p) for p in files]


def _parse_pt_log(log_path: str | list[str]) -> dict:
    """Parse one or more JSONL session logs into structured data.

    Pass a list of paths (oldest-first) to merge across multiple sessions
    that ran on the same day — e.g. after a PT restart.
    """
    paths = [log_path] if isinstance(log_path, str) else log_path
    events: list[dict] = []
    for p in paths:
        try:
            with open(p) as _f:
                for _line in _f:
                    try:
                        events.append(json.loads(_line.strip()))
                    except Exception:
                        pass
        except Exception:
            pass

    open_trades: dict[str, dict] = {}
    closed_trades: list[dict] = []
    gate_blocks: dict[str, int] = {}
    feed_events: list[dict] = []
    poll_count = 0
    last_poll_et: str | None = None
    watchlist_syms: list[str] = []
    last_diag: dict = {}
    last_context: dict = {}

    for e in events:
        ev = e.get('event')
        if ev == 'paper_trade_open':
            open_trades[e.get('symbol', '')] = e
        elif ev == 'paper_trade_close':
            sym = e.get('symbol', '')
            closed_trades.append(e)
            open_trades.pop(sym, None)
        elif ev == 'entry_gate_blocked':
            reason = str(e.get('reason') or 'unknown').split('(')[0].strip()
            gate_blocks[reason] = gate_blocks.get(reason, 0) + 1
            feed_events.append(e)
        elif ev == 'snapshot':
            poll_count = e.get('poll_count', poll_count)
            last_poll_et = e.get('ts_et', '')
            last_diag = e.get('diagnostics', {})
            last_context = e.get('context', {})
        elif ev == 'session_start':
            watchlist_syms = e.get('watchlist_symbols', [])
        if ev in ('state_transition', 'paper_trade_open', 'paper_trade_close', 'entry_gate_blocked'):
            feed_events.append(e)

    return {
        'open_trades': list(open_trades.values()),
        'closed_trades': closed_trades,
        'feed_events': feed_events[-150:],
        'gate_blocks': gate_blocks,
        'poll_count': poll_count,
        'last_poll_et': last_poll_et,
        'watchlist_syms': watchlist_syms,
        'last_diag': last_diag,
        'last_context': last_context,
    }


@app.get('/paper')
def paper_trader_page():
    return render_template('paper.html')


@app.post('/api/pt/start')
def api_pt_start():
    data = request.get_json(silent=True) or {}
    watchlist_only = bool(data.get('watchlist_only', False))
    env = {**os.environ}
    if watchlist_only:
        env['PT_WATCHLIST_ONLY'] = '1'
    for key, envvar in [
        ('min_ml_score', 'PT_MIN_ML_SCORE'),
        ('min_risk_per_share', 'PT_MIN_RISK_PER_SHARE'),
        ('max_risk_per_share', 'PT_MAX_RISK_PER_SHARE'),
        ('max_spread_sub10', 'PT_MAX_SPREAD_SUB10'),
        ('max_spread_above10', 'PT_MAX_SPREAD_ABOVE10'),
        ('regime_min_risk_on', 'PT_REGIME_MIN_RISK_ON'),
    ]:
        if data.get(key) is not None:
            env[envvar] = str(data[key])

    with _PT_LOCK:
        proc = _PT_STATE.get('proc')
        if proc and proc.poll() is None:
            return jsonify(ok=True, reused=True, message='Monitor already running.')
        script = Path(__file__).resolve().parent / 'tools' / 'paper_trade_monitor.py'
        cmd = [sys.executable, str(script)]
        if watchlist_only:
            cmd.append('--watchlist-only')
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent), env=env,
        )
        _PT_STATE.update({
            'proc': proc, 'running': True, 'log_path': None,
            'start_time_iso': datetime.now(timezone.utc).isoformat(),
            'watchlist_only': watchlist_only,
        })
    return jsonify(ok=True, pid=proc.pid)


@app.post('/api/pt/stop')
def api_pt_stop():
    with _PT_LOCK:
        proc = _PT_STATE.get('proc')
        if proc and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception as exc:
                return jsonify(ok=False, error=str(exc)), 500
        _PT_STATE['running'] = False

    # Kill any orphaned paper_trade_monitor processes that survived an app
    # restart or were started outside of api_pt_start (e.g. from the CLI).
    try:
        import subprocess as _sp
        _sp.run(['pkill', '-TERM', '-f', 'paper_trade_monitor.py'],
                capture_output=True)
    except Exception:
        pass

    return jsonify(ok=True)


@app.get('/api/pt/status')
def api_pt_status():
    with _PT_LOCK:
        proc = _PT_STATE.get('proc')
        proc_alive = bool(proc and proc.poll() is None)
        # If our tracked proc is gone, check whether any paper_trade_monitor
        # process is still running (survives app restarts / orphaned procs).
        if not proc_alive:
            import subprocess as _sp
            _check = _sp.run(
                ['pgrep', '-f', 'paper_trade_monitor.py'],
                capture_output=True, text=True
            )
            proc_alive = bool(_check.stdout.strip())
        running = proc_alive
        _PT_STATE['running'] = running
        start_time = _PT_STATE.get('start_time_iso')
        watchlist_only = _PT_STATE.get('watchlist_only', False)

    # Collect ALL of today's session files (oldest → newest) so that trades
    # from earlier sessions are not lost when the PT is stopped and restarted.
    all_logs = _find_all_pt_logs_today()
    parsed = _parse_pt_log(all_logs) if all_logs else {
        'open_trades': [], 'closed_trades': [], 'feed_events': [],
        'gate_blocks': {}, 'poll_count': 0, 'last_poll_et': None,
        'watchlist_syms': [], 'last_diag': {}, 'last_context': {},
    }

    closed = parsed['closed_trades']
    total_r = round(sum(t.get('result_r') or 0.0 for t in closed), 2)
    wins    = sum(1 for t in closed if (t.get('result_r') or 0) >= 0.9)
    losses  = sum(1 for t in closed if (t.get('result_r') or 0) < -0.15)

    return jsonify(
        ok=True, running=running, log_path=all_logs[-1] if all_logs else None,
        start_time=start_time, watchlist_only=watchlist_only,
        poll_count=parsed['poll_count'], last_poll_et=parsed['last_poll_et'],
        open_trades=parsed['open_trades'], closed_trades=parsed['closed_trades'],
        feed_events=parsed['feed_events'], gate_blocks=parsed['gate_blocks'],
        watchlist_syms=parsed['watchlist_syms'],
        last_diag=parsed['last_diag'], last_context=parsed['last_context'],
        total_r=total_r, trade_count=len(closed),
        wins=wins, losses=losses, scratches=len(closed) - wins - losses,
    )


@app.post('/api/pt/validate')
def api_pt_validate():
    data = request.get_json(silent=True) or {}
    with _PT_VAL_LOCK:
        if _PT_VAL_STATE['running']:
            return jsonify(ok=False, error='Validation already running.'), 400
        _PT_VAL_STATE.update({'running': True, 'lines': [], 'done': False, 'error': None})
    script = Path(__file__).resolve().parent / 'tools' / 'eod_validate.py'
    def _run():
        try:
            cmd = [sys.executable, str(script)]
            if data.get('date'):
                cmd += ['--date', str(data['date'])]
            if data.get('no_bars'):
                cmd.append('--no-bars')
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(Path(__file__).resolve().parent),
                env={**os.environ},
            )
            for line in proc.stdout:
                with _PT_VAL_LOCK:
                    _PT_VAL_STATE['lines'].append(line.rstrip('\n'))
            proc.wait()
            with _PT_VAL_LOCK:
                _PT_VAL_STATE.update({
                    'running': False, 'done': True,
                    'error': f'exit {proc.returncode}' if proc.returncode != 0 else None,
                })
        except Exception as exc:
            with _PT_VAL_LOCK:
                _PT_VAL_STATE.update({'running': False, 'done': True, 'error': str(exc)})
    threading.Thread(target=_run, daemon=True).start()
    return jsonify(ok=True)


@app.get('/api/pt/validate/status')
def api_pt_validate_status():
    with _PT_VAL_LOCK:
        return jsonify(
            ok=True, running=_PT_VAL_STATE['running'],
            done=_PT_VAL_STATE['done'], lines=list(_PT_VAL_STATE['lines']),
            error=_PT_VAL_STATE.get('error'),
        )

# ─────────────────────────────────────────────────────────────────────────────
# ── Live News Sentiment Feed ──────────────────────────────────────────────────

_NEWS_SEEN_IDS: set[str] = set()
_NEWS_SEEN_IDS_MAX = 5000  # evict oldest half when full
_NEWS_FEED_LOCK = threading.Lock()
_NEWS_POLL_INTERVAL = 15  # seconds — fast poll; WS stream requires Algo Trader Plus
_NEWS_REFRESH_EVENT = threading.Event()  # set() to wake the news thread immediately
_MARKET_EVENTS_EVENT = threading.Event()  # set() to wake market-events SSE listeners

# ── News Sniper — fires the instant a NEW catalyst headline drops ─────────────
_SNIPER_ALERTS: deque = deque(maxlen=50)
_SNIPER_LOCK = threading.Lock()
_SNIPER_WL_COOLDOWN: dict[str, float] = {}  # sym → last watchlist-add timestamp

# Catalyst tags that warrant an immediate sniper alert
_SNIPER_CATALYST_TAGS = frozenset({
    'earnings', 'guidance', 'fda_bio', 'deal', 'analyst',
    'capital_raise', 'corporate_action', 'bankruptcy_delisting', 'legal_risk',
})
# Minimum sentiment confidence to fire (avoids noise from weak articles)
_SNIPER_MIN_CONFIDENCE = 40.0

def _emit_sniper_alert(sym: str, article: dict, bundle, price: float | None, *, plan: dict | None = None) -> None:
    """Push a sniper alert to the ring buffer when a new catalyst headline drops."""
    tags = list(getattr(bundle, 'tags', []) or [])
    _plan = plan or {}
    alert = {
        'symbol':      sym,
        'headline':    article.get('headline') or article.get('title') or '',
        'source':      article.get('source') or '',
        'url':         article.get('url') or '',
        'published_at': str(article.get('created_at') or article.get('updated_at') or ''),
        'fired_at':    datetime.now(timezone.utc).isoformat(),
        'fired_ts':    time.time(),
        'tags':        tags,
        'sentiment':   round(float(getattr(bundle, 'score', 0) or 0), 4),
        'confidence':  round(float(getattr(bundle, 'confidence', 0) or 0), 1),
        'strength':    round(float(getattr(bundle, 'strength', 0) or 0), 4),
        'price':       price,
        'direction':   'LONG' if (getattr(bundle, 'score', 0) or 0) > 0 else ('SHORT' if (getattr(bundle, 'score', 0) or 0) < 0 else 'NEUTRAL'),
        # Entry plan levels — available immediately in the banner, used by _bg_grade without watchlist lookup
        'entry':       _plan.get('entry'),
        'stop':        _plan.get('stop'),
        'target':      _plan.get('target_2r'),
        'risk_per_share': _plan.get('risk_per_share'),
    }
    with _SNIPER_LOCK:
        _SNIPER_ALERTS.appendleft(alert)
    _NEWS_REFRESH_EVENT.set()   # wake any SSE listeners

    # Auto-add to desk watchlist — 5-minute cooldown per symbol to prevent flip-flopping
    direction = alert['direction']
    if direction in ('LONG', 'SHORT') and alert['confidence'] >= _SNIPER_MIN_CONFIDENCE:
        side = direction.lower()
        _now_ts = time.time()
        _last_add = _SNIPER_WL_COOLDOWN.get(sym, 0)
        if _now_ts - _last_add >= 300:  # 5-minute gate per symbol
            _SNIPER_WL_COOLDOWN[sym] = _now_ts
            try:
                _RUNTIME_STORE.desk_watchlist_set(
                    sym,
                    side=side,
                    trigger_price=_plan.get('entry'),
                    stop_price=_plan.get('stop'),
                    target_price=_plan.get('target_2r'),
                    notes=f"\U0001f4f0 {alert['headline'][:120]}",
                )
            except Exception:
                pass

        def _bg_grade(a=alert, s=side, h=alert['headline'], px=price):
            import logging as _log
            _lg = _log.getLogger(__name__)
            try:
                live_px = px
                if not live_px and _STREAM is not None:
                    try:
                        snap = _STREAM.latest_quote(a['symbol'])
                        if snap:
                            bp = snap.get('bp') or snap.get('bid_price') or 0
                            ap = snap.get('ap') or snap.get('ask_price') or 0
                            if bp and ap:
                                live_px = (bp + ap) / 2
                            elif bp:
                                live_px = float(bp)
                            elif ap:
                                live_px = float(ap)
                    except Exception as _qe:
                        _lg.warning('_bg_grade quote fallback failed %s: %s', a.get('symbol'), _qe)
                # Fix #15: pass actual halt count from stream cache
                _halt_count = 0
                try:
                    if _STREAM is not None:
                        _halt_evts = _STREAM.recent_halt_resume_events(max_age_sec=86400)
                        _halt_count = sum(1 for _e in _halt_evts
                                          if str(_e.get('symbol', '')).upper() == a['symbol'].upper()
                                          and str(_e.get('status_code', '')).upper()
                                          in ('T1', 'T2', 'T3', 'H', 'HALTED', 'LUDP'))
                except Exception:
                    pass

                # Compute real catalyst age from article publish time
                _catalyst_age = None
                try:
                    _pub = a.get('published_at') or a.get('fired_at')
                    if _pub:
                        from datetime import datetime as _dt, timezone as _tz
                        _pub_dt = _dt.fromisoformat(str(_pub).replace('Z', '+00:00'))
                        _catalyst_age = max(0.0, (_dt.now(_tz.utc) - _pub_dt).total_seconds() / 3600.0)
                except Exception:
                    pass

                # Compute setup age from when the alert fired
                _setup_age = None
                try:
                    _fired = float(a.get('fired_ts') or 0)
                    if _fired > 0:
                        import time as _time
                        _setup_age = max(0.0, (_time.time() - _fired) / 3600.0)
                except Exception:
                    pass

                # Use entry/stop/target from the alert dict (set at sniper fire time from plan)
                # This avoids a race condition between watchlist write and _bg_grade thread start
                _entry  = a.get('entry')
                _stop   = a.get('stop')
                _target = a.get('target')
                _ml     = None

                from utils.know_the_trade import analyze as _ktt
                ktt = _ktt(
                    symbol=a['symbol'],
                    provider=_ALPACA_PROVIDER,
                    side=s,
                    pm_last=live_px,
                    entry=_entry,
                    stop=_stop,
                    target=_target,
                    ml_score=_ml,
                    catalyst_headline=h,
                    catalyst_age_hours=_catalyst_age,
                    setup_age_hours=_setup_age,
                    halt_count=_halt_count,
                    sniper_context=False,
                )
                a['ktt_grade'] = ktt.get('grade') or '?'
                a['ktt_score'] = ktt.get('score')
                a['ktt_advice'] = ktt.get('grade_advice')
                _lg.debug('_bg_grade OK %s -> %s %s', a.get('symbol'), a['ktt_grade'], a['ktt_score'])
            except Exception as exc:
                _lg.warning('_bg_grade failed %s: %s', a.get('symbol'), exc, exc_info=True)
                a['ktt_grade'] = '?'
                a['ktt_score'] = None
            finally:
                _NEWS_REFRESH_EVENT.set()   # push updated alert to SSE listeners
        threading.Thread(target=_bg_grade, daemon=True).start()


# ── Pre-Halt Spread Explosion Detector ───────────────────────────────────────
_SPREAD_ALERTS: deque = deque(maxlen=100)
_SPREAD_LOCK   = threading.Lock()
_SPREAD_SEEN: set[str] = set()   # "SYM:fired_ts_bucket" to avoid alert storms

def _spread_monitor_loop() -> None:
    """Every 3s: check all watched symbols for bid-ask spread explosion."""
    while True:
        try:
            if _STREAM is not None:
                syms: set[str] = set()
                try:
                    for e in _RUNTIME_STORE.desk_watchlist_all():
                        if e.get('symbol'):
                            syms.add(str(e['symbol']).upper())
                except Exception:
                    pass
                try:
                    sess = _MONITOR.active_session()
                    if sess:
                        syms.update(str(s).upper() for s in sess.symbols.keys())
                except Exception:
                    pass

                for sym in syms:
                    try:
                        result = _STREAM.spread_explosion_check(sym)
                        if not result.get('is_exploding'):
                            continue
                        # Deduplicate: only fire once per 60s per symbol
                        bucket = f"{sym}:{int(time.time() // 60)}"
                        with _SPREAD_LOCK:
                            if bucket in _SPREAD_SEEN:
                                continue
                            _SPREAD_SEEN.add(bucket)
                            # Trim seen set to avoid unbounded growth
                            if len(_SPREAD_SEEN) > 500:
                                oldest = sorted(_SPREAD_SEEN)[:250]
                                for k in oldest:
                                    _SPREAD_SEEN.discard(k)

                        mid_price = result.get('mid_price')
                        # Attach resume prediction so UI can show it immediately
                        resume_pred: dict = {}
                        if mid_price:
                            try:
                                resume_pred = predict_halt_resume(
                                    sym, 'T1', mid_price,
                                    halt_hour_et=datetime.now(_ET).hour,
                                )
                            except Exception:
                                pass
                        alert = {
                            'type':             'spread_explosion',
                            'symbol':           sym,
                            'fired_at':         datetime.now(timezone.utc).isoformat(),
                            'fired_ts':         time.time(),
                            'current_spread':   result.get('current_spread'),
                            'baseline_spread':  result.get('baseline_spread'),
                            'ratio':            result.get('ratio'),
                            'mid_price':        mid_price,
                            'resume_prediction':resume_pred,
                            'message':          (
                                f"Spread {result.get('current_spread', 0):.2f}% "
                                f"({result.get('ratio', 0):.1f}x baseline) — HALT IMMINENT"
                            ),
                        }
                        with _SPREAD_LOCK:
                            _SPREAD_ALERTS.appendleft(alert)
                        _NEWS_REFRESH_EVENT.set()
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(3)

def _start_spread_monitor() -> None:
    t = threading.Thread(target=_spread_monitor_loop, daemon=True, name='spread-monitor')
    t.start()

_start_spread_monitor()


# ── Halt Resume Price Predictor ───────────────────────────────────────────────
# Research-based lookup: when a halt fires, predict where it resumes and where
# to place limit orders. Buckets are derived from documented LULD/reg-halt patterns.

def predict_halt_resume(
    symbol: str,
    halt_code: str,
    halt_price: float,
    pm_move_pct: float | None = None,
    float_shares: float | None = None,
    halt_hour_et: int | None = None,
) -> dict:
    """
    Predict resume price range and suggested entry/stop for a halted stock.

    Returns:
        resume_low       – low end of expected resume price
        resume_high      – high end of expected resume price
        resume_mid       – midpoint estimate
        direction        – 'LONG' | 'SHORT' | 'UNCERTAIN'
        suggested_entry  – limit order price to place during halt
        suggested_stop   – stop loss after resume
        suggested_target – 2R target after resume
        confidence       – 'HIGH' | 'MEDIUM' | 'LOW'
        rationale        – one-line explanation
    """
    code  = str(halt_code or '').upper()
    price = float(halt_price or 0)
    if price <= 0:
        return {}

    pm_move = float(pm_move_pct or 0)
    flt     = float(float_shares or 0)
    hour    = int(halt_hour_et or 9)

    # ── Classify halt type ────────────────────────────────────────────────────
    is_luld       = code in ('T1',)          # LULD circuit breaker — most common
    is_news_pend  = code in ('T2',)          # news pending — can go either way
    is_regulatory = code in ('T3', 'H')      # regulatory / SEC — almost always bad

    # ── Base resume delta by halt type ───────────────────────────────────────
    if is_regulatory:
        # Regulatory halts almost always resume DOWN, often -20% to -50%
        return {
            'resume_low':      round(price * 0.50, 2),
            'resume_high':     round(price * 0.85, 2),
            'resume_mid':      round(price * 0.70, 2),
            'direction':       'SHORT',
            'suggested_entry': round(price * 0.82, 2),
            'suggested_stop':  round(price * 0.90, 2),
            'suggested_target':round(price * 0.65, 2),
            'confidence':      'MEDIUM',
            'rationale':       'Regulatory/SEC halt — historically resumes 15-50% below halt price.',
        }

    if is_news_pend:
        if pm_move >= 20:
            # Strong positive momentum + news pending → resume higher
            resume_up   = 0.05 + min(0.12, pm_move / 300.0)
            return {
                'resume_low':      round(price * 1.02, 2),
                'resume_high':     round(price * (1 + resume_up + 0.05), 2),
                'resume_mid':      round(price * (1 + resume_up), 2),
                'direction':       'LONG',
                'suggested_entry': round(price * 1.03, 2),
                'suggested_stop':  round(price * 0.96, 2),
                'suggested_target':round(price * (1 + resume_up * 2), 2),
                'confidence':      'MEDIUM',
                'rationale':       f'T2 news-pending halt on +{pm_move:.0f}% mover — positive catalyst likely, resume expected higher.',
            }
        elif pm_move <= -10:
            return {
                'resume_low':      round(price * 0.75, 2),
                'resume_high':     round(price * 0.98, 2),
                'resume_mid':      round(price * 0.88, 2),
                'direction':       'SHORT',
                'suggested_entry': round(price * 0.95, 2),
                'suggested_stop':  round(price * 1.03, 2),
                'suggested_target':round(price * 0.80, 2),
                'confidence':      'LOW',
                'rationale':       'T2 news-pending halt on negative mover — resume direction uncertain, lean short.',
            }
        else:
            return {
                'resume_low':      round(price * 0.92, 2),
                'resume_high':     round(price * 1.08, 2),
                'resume_mid':      price,
                'direction':       'UNCERTAIN',
                'suggested_entry': None,
                'suggested_stop':  None,
                'suggested_target':None,
                'confidence':      'LOW',
                'rationale':       'T2 news-pending halt — direction uncertain without knowing catalyst.',
            }

    # LULD T1 halt — the most common case for parabolic movers
    # Small float + early session + big PM move → high confidence bullish resume
    small_float = flt > 0 and flt < 10_000_000
    early_sess  = hour <= 10

    if pm_move >= 30:
        resume_pct = 0.05 if not small_float else 0.08
        conf = 'HIGH' if (small_float and early_sess) else 'MEDIUM'
        return {
            'resume_low':      round(price * 1.02, 2),
            'resume_high':     round(price * (1 + resume_pct + 0.06), 2),
            'resume_mid':      round(price * (1 + resume_pct), 2),
            'direction':       'LONG',
            'suggested_entry': round(price * 1.025, 2),
            'suggested_stop':  round(price * 0.96, 2),
            'suggested_target':round(price * (1 + resume_pct * 2.5), 2),
            'confidence':      conf,
            'rationale':       (
                f'LULD T1 on +{pm_move:.0f}% mover'
                + (' · small float' if small_float else '')
                + (' · early session' if early_sess else '')
                + ' — 85%+ of these resume higher. Place limit just above halt price.'
            ),
        }
    elif pm_move >= 15:
        return {
            'resume_low':      round(price * 1.00, 2),
            'resume_high':     round(price * 1.07, 2),
            'resume_mid':      round(price * 1.035, 2),
            'direction':       'LONG',
            'suggested_entry': round(price * 1.02, 2),
            'suggested_stop':  round(price * 0.97, 2),
            'suggested_target':round(price * 1.08, 2),
            'confidence':      'MEDIUM',
            'rationale':       f'LULD T1 on +{pm_move:.0f}% mover — moderate resume likelihood. Wait for price confirm.',
        }
    else:
        return {
            'resume_low':      round(price * 0.97, 2),
            'resume_high':     round(price * 1.05, 2),
            'resume_mid':      price,
            'direction':       'UNCERTAIN',
            'suggested_entry': None,
            'suggested_stop':  None,
            'suggested_target':None,
            'confidence':      'LOW',
            'rationale':       'LULD T1 on low-momentum stock — resume direction unclear.',
        }


def _news_entry_plan(sym: str, sentiment_score: float) -> dict:
    """Compute a simplified entry/stop/target plan for a news-driven signal.

    Direction: positive sentiment → LONG, negative → SHORT.
    Stop distance: 2% of price (3% for sub-$5 stocks).
    Does NOT call the full Entry-Now planner (avoids circular HTTP calls).
    Returns {} on any failure.
    """
    if _ALPACA_PROVIDER is None:
        return {}
    try:
        trade = _ALPACA_PROVIDER.get_latest_trade(sym)
        price = float(trade.get('price') or 0)
        if price <= 0:
            return {}
    except Exception:
        return {}

    side = 'long' if sentiment_score >= 0 else 'short'
    stop_pct = 0.03 if price < 5 else 0.02
    if side == 'long':
        entry  = round(price, 2)
        stop   = round(price * (1 - stop_pct), 2)
        t2r    = round(entry + 2 * (entry - stop), 2)
        t3r    = round(entry + 3 * (entry - stop), 2)
    else:
        entry  = round(price, 2)
        stop   = round(price * (1 + stop_pct), 2)
        t2r    = round(entry - 2 * (stop - entry), 2)
        t3r    = round(entry - 3 * (stop - entry), 2)

    # Try to get ML score from the orb model
    ml_score = None
    try:
        from ml.orb_model_service import score_orb_symbol
        ml_out = score_orb_symbol(sym, last_price=price, provider=_ALPACA_PROVIDER)
        if ml_out.get('score') is not None:
            ml_score = float(ml_out['score'])
    except Exception:
        pass

    # Combined score: blend ML (if available) with |sentiment|, preserve direction
    abs_sent = abs(float(sentiment_score))
    if ml_score is not None:
        combined = (0.5 * ml_score + 0.5 * abs_sent) * (1 if sentiment_score >= 0 else -1)
    else:
        combined = float(sentiment_score)

    return {
        'side': side,
        'price': price,
        'entry': entry,
        'stop': stop,
        'target_2r': t2r,
        'target_3r': t3r,
        'risk_per_share': round(abs(entry - stop), 4),
        'ml_score': ml_score,
        'combined_score': round(combined, 4),
    }


def _news_feed_loop() -> None:
    """Background thread: poll Alpaca News API every 45s, score+plan, store in DB."""
    import hashlib
    from sentiment.catalyst import CatalystService

    svc = CatalystService(_ALPACA_PROVIDER)

    while True:
        try:
            if _ALPACA_PROVIDER is None:
                _NEWS_REFRESH_EVENT.wait(timeout=_NEWS_POLL_INTERVAL)
                _NEWS_REFRESH_EVENT.clear()
                continue

            now = datetime.now(timezone.utc)
            # Outside market hours the 4h window misses the trading session.
            # Use 20h lookback so overnight runs always cover the full prior day.
            start_iso = (now - timedelta(hours=20)).isoformat()

            # Collect symbols: desk watchlist + active monitor session
            syms: set[str] = set()
            try:
                for entry in _RUNTIME_STORE.desk_watchlist_all():
                    if entry.get('symbol'):
                        syms.add(str(entry['symbol']).strip().upper())
            except Exception:
                pass
            try:
                sess = _MONITOR.active_session()
                if sess:
                    syms.update(str(s).upper() for s in sess.symbols.keys())
            except Exception:
                pass

            if not syms:
                _NEWS_REFRESH_EVENT.wait(timeout=_NEWS_POLL_INTERVAL)
                _NEWS_REFRESH_EVENT.clear()
                continue

            raw_map = _ALPACA_PROVIDER.get_news_batch(
                list(syms), limit_per_symbol=5, start=start_iso
            )

            for sym, articles in raw_map.items():
                for art in articles:
                    raw_id = str(art.get('id') or art.get('news_id') or '').strip()
                    if not raw_id:
                        key = f"{sym}:{art.get('headline', '')}:{art.get('created_at', '')}"
                        raw_id = hashlib.md5(key.encode()).hexdigest()
                    news_id = f"{sym}:{raw_id}"

                    headline = str(art.get('headline') or art.get('title') or '').strip()
                    if not headline:
                        continue

                    bundle = svc._score_symbol(sym, [art], now=now)

                    # ── News Sniper: fire on first sight of a catalyst headline ──
                    is_new_article = news_id not in _NEWS_SEEN_IDS
                    if is_new_article:
                        if len(_NEWS_SEEN_IDS) >= _NEWS_SEEN_IDS_MAX:
                            _NEWS_SEEN_IDS.clear()
                        _NEWS_SEEN_IDS.add(news_id)
                        has_catalyst_tag = bool(_SNIPER_CATALYST_TAGS & set(bundle.tags or []))
                        strong_enough    = float(bundle.confidence or 0) >= _SNIPER_MIN_CONFIDENCE
                        if has_catalyst_tag and strong_enough:
                            try:
                                snap_price: float | None = None
                                if _ALPACA_PROVIDER is not None:
                                    t = _ALPACA_PROVIDER.get_latest_trade(sym)
                                    snap_price = float(t.get('price') or 0) or None
                            except Exception:
                                snap_price = None
                            _emit_sniper_alert(sym, art, bundle, snap_price)
                    src = art.get('source')
                    source = src if isinstance(src, str) else ((src or {}).get('name') or '')

                    # For actionable sentiment, compute entry plan + ML
                    plan: dict = {}
                    if abs(bundle.score) >= 0.25:
                        try:
                            plan = _news_entry_plan(sym, bundle.score)
                        except Exception:
                            plan = {}

                    payload = {
                        'news_id': news_id,
                        'symbol': sym,
                        'headline': headline,
                        'summary': str(art.get('summary') or '')[:300] or None,
                        'source': source or None,
                        'url': art.get('url') or None,
                        'published_at': str(art.get('created_at') or art.get('updated_at') or ''),
                        'received_at': now.timestamp(),
                        'sentiment_score': bundle.score,
                        'catalyst_score': bundle.score,
                        'freshness_sec': (bundle.freshness_hours * 3600) if bundle.freshness_hours is not None else None,
                        'tags': bundle.tags,
                        'confidence': bundle.confidence,
                        'strength': bundle.strength,
                        'article_count': 1,
                        # Entry plan fields (populated for |score| >= 0.25)
                        'side': plan.get('side'),
                        'price': plan.get('price'),
                        'entry': plan.get('entry'),
                        'stop': plan.get('stop'),
                        'target_2r': plan.get('target_2r'),
                        'target_3r': plan.get('target_3r'),
                        'risk_per_share': plan.get('risk_per_share'),
                        'ml_score': plan.get('ml_score'),
                        'combined_score': plan.get('combined_score') if plan else bundle.score,
                    }
                    try:
                        _RUNTIME_STORE.append_news_event(payload)
                    except Exception:
                        pass

        except Exception:
            pass

        _NEWS_REFRESH_EVENT.wait(timeout=_NEWS_POLL_INTERVAL)
        _NEWS_REFRESH_EVENT.clear()


def _start_news_feed() -> None:
    t = threading.Thread(target=_news_feed_loop, daemon=True, name='news-feed')
    t.start()


def _process_news_article(sym: str, art: dict) -> None:
    """Shared pipeline: score one article, fire sniper if warranted, store in DB."""
    import hashlib
    from datetime import timezone as _tz
    from sentiment.catalyst import CatalystService
    try:
        svc = CatalystService(_ALPACA_PROVIDER)
        now = datetime.now(_tz.utc)
        raw_id = str(art.get('id') or art.get('news_id') or '').strip()
        if not raw_id:
            key = f"{sym}:{art.get('headline', '')}:{art.get('created_at', '')}"
            raw_id = hashlib.md5(key.encode()).hexdigest()
        news_id = f"{sym}:{raw_id}"
        headline = str(art.get('headline') or art.get('title') or '').strip()
        if not headline:
            return
        bundle = svc._score_symbol(sym, [art], now=now)
        # Compute plan before sniper check so entry/stop/target reach the watchlist immediately
        plan: dict = {}
        if abs(bundle.score) >= 0.25:
            try:
                plan = _news_entry_plan(sym, bundle.score)
            except Exception:
                plan = {}
        is_new = news_id not in _NEWS_SEEN_IDS
        if is_new:
            if len(_NEWS_SEEN_IDS) >= _NEWS_SEEN_IDS_MAX:
                _NEWS_SEEN_IDS.clear()
            _NEWS_SEEN_IDS.add(news_id)
            has_catalyst_tag = bool(_SNIPER_CATALYST_TAGS & set(bundle.tags or []))
            strong_enough    = float(bundle.confidence or 0) >= _SNIPER_MIN_CONFIDENCE
            if has_catalyst_tag and strong_enough:
                try:
                    snap_price = None
                    if _ALPACA_PROVIDER is not None:
                        t = _ALPACA_PROVIDER.get_latest_trade(sym)
                        snap_price = float(t.get('price') or 0) or None
                except Exception:
                    snap_price = None
                _emit_sniper_alert(sym, art, bundle, snap_price, plan=plan)
            elif plan and plan.get('entry'):
                # Directional news below sniper threshold but with an actionable plan
                _ns_side = plan.get('side') or ('long' if bundle.score > 0 else 'short')
                try:
                    _RUNTIME_STORE.desk_watchlist_set(
                        sym, side=_ns_side,
                        trigger_price=plan.get('entry'),
                        stop_price=plan.get('stop'),
                        target_price=plan.get('target_2r'),
                        notes=f"\U0001f4f0 {headline[:120]}",
                    )
                except Exception:
                    pass
        src = art.get('source')
        source = src if isinstance(src, str) else ((src or {}).get('name') or '')
        payload = {
            'news_id': news_id, 'symbol': sym, 'headline': headline,
            'summary': str(art.get('summary') or '')[:300] or None,
            'source': source or None, 'url': art.get('url') or None,
            'published_at': str(art.get('created_at') or art.get('updated_at') or ''),
            'received_at': now.timestamp(),
            'sentiment_score': bundle.score, 'catalyst_score': bundle.score,
            'freshness_sec': (bundle.freshness_hours * 3600) if bundle.freshness_hours is not None else None,
            'tags': bundle.tags, 'confidence': bundle.confidence, 'strength': bundle.strength,
            'article_count': 1,
            'side': plan.get('side'), 'price': plan.get('price'),
            'entry': plan.get('entry'), 'stop': plan.get('stop'),
            'target_2r': plan.get('target_2r'), 'target_3r': plan.get('target_3r'),
            'risk_per_share': plan.get('risk_per_share'),
            'ml_score': plan.get('ml_score'),
            'combined_score': plan.get('combined_score') if plan else bundle.score,
        }
        try:
            _RUNTIME_STORE.append_news_event(payload)
        except Exception:
            pass
    except Exception:
        pass


def _news_ws_loop() -> None:
    """Background thread: WebSocket news stream — fires the instant Alpaca publishes."""
    import asyncio
    import logging
    log = logging.getLogger('news-ws')

    api_key    = os.getenv('ALPACA_API_KEY', '')
    secret_key = os.getenv('ALPACA_SECRET_KEY', '')
    if not api_key or not secret_key:
        log.warning('news-ws: no Alpaca credentials — WebSocket news stream disabled')
        return

    try:
        from alpaca.data.live import NewsDataStream
        stream = NewsDataStream(api_key=api_key, secret_key=secret_key)

        async def _on_news(article) -> None:
            try:
                art = article if isinstance(article, dict) else article.__dict__
                symbols = art.get('symbols') or art.get('tickers') or []
                for sym in symbols:
                    sym = str(sym).strip().upper()
                    if sym:
                        _process_news_article(sym, art)
            except Exception:
                pass

        stream.subscribe_news(_on_news, '*')
        log.info('news-ws: connected — streaming all news in real-time')
        stream.run()

    except Exception as e:
        err = str(e).lower()
        if 'connection limit' in err:
            log.info('news-ws: connection limit reached (SIP Pro allows 1 WebSocket) — '
                     'using 15s poll fallback. Upgrade to Algo Trader Plus for real-time news stream.')
        else:
            log.warning(f'news-ws: failed to connect ({e}) — using poll fallback')


def _start_news_feed() -> None:
    t = threading.Thread(target=_news_feed_loop, daemon=True, name='news-feed')
    t.start()
    ws = threading.Thread(target=_news_ws_loop, daemon=True, name='news-ws')
    ws.start()


_start_news_feed()


def _outcome_auto_resolve_loop() -> None:
    """Background thread: auto-resolve open R-multiple outcomes every 60s during market hours."""
    import time as _time
    while True:
        try:
            now_et = _et_now()
            # Only run during market hours (9:25–16:05 ET, weekdays)
            if now_et.weekday() < 5:
                t = now_et.time()
                if dtime(9, 25) <= t <= dtime(16, 5):
                    _run_auto_resolve_outcomes()
        except Exception:
            pass
        _time.sleep(60)


threading.Thread(target=_outcome_auto_resolve_loop, daemon=True, name='outcome-auto-resolve').start()

# ─────────────────────────────────────────────────────────────────────────────


@app.get('/api/news_recent')
def api_news_recent():
    symbol = str(request.args.get('symbol') or '').strip().upper() or None
    limit = _safe_int(request.args.get('limit', 25), 25)
    return jsonify(ok=True, items=_RUNTIME_STORE.recent_news(symbol=symbol, limit=limit))


@app.get('/api/news_live')
def api_news_live():
    """Live news with per-article sentiment signal flags. Used by the Catalyst Desk."""
    symbol = str(request.args.get('symbol') or '').strip().upper() or None
    limit = min(_safe_int(request.args.get('limit', 50), 50), 200)
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(hours=16)).timestamp()
    items = _RUNTIME_STORE.recent_news(symbol=symbol, limit=limit)
    # Drop articles whose published_at is older than 16 hours (covers full trading day + overnight)
    def _pub_ts(item: dict) -> float:
        raw = item.get('published_at') or ''
        try:
            from dateutil.parser import parse as _parse
            return _parse(raw).timestamp() if raw else 0.0
        except Exception:
            return float(item.get('received_at') or 0)
    items = [i for i in items if _pub_ts(i) >= cutoff_ts]

    import math as _math
    _NOW_TS = datetime.now(timezone.utc).timestamp()

    # Catalyst timing decay: sentiment strength decays with article age.
    # ML score (model-based, edge-driven) is NOT decayed — only sentiment signal.
    # Half-life = 1.5 h → a 3h-old article runs at 25% of original strength.
    _DECAY_HALF_LIFE_H = 1.5

    out = []
    for item in items:
        raw_score = float(item.get('sentiment_score') or item.get('catalyst_score') or 0.0)
        conf      = float(item.get('confidence') or 0.0)
        ml        = float(item.get('ml_score') or 0.0)
        combined  = float(item.get('combined_score') or 0.0)

        # ── Timing decay ────────────────────────────────────────────────────
        pub_ts_val  = _pub_ts(item)
        age_hours   = max(0.0, (_NOW_TS - pub_ts_val) / 3600.0) if pub_ts_val > 0 else 0.0
        decay_factor = _math.exp(-_math.log(2) * age_hours / _DECAY_HALF_LIFE_H)
        score        = raw_score * decay_factor   # decayed sentiment score

        # ── Signal badge requires BOTH sentiment AND ML to clear their thresholds.
        # When sentiment is bullish/bearish but ML is too low (<0.25), the signal
        # is demoted to WATCH: the article is shown but NO entry plan is rendered.
        # This prevents high-sentiment / low-edge articles from looking tradeable.
        ML_MIN_BUY        = 0.25   # minimum ML for any BUY/SELL badge
        ML_MIN_STRONG_BUY = 0.35   # minimum ML for STRONG BUY/SELL badge
        CMB_MIN_STRONG    = 0.50   # minimum combined for STRONG badge

        if score >= 0.55 and conf >= 55:
            if ml >= ML_MIN_STRONG_BUY and combined >= CMB_MIN_STRONG:
                signal = 'STRONG_BUY'
            elif ml >= ML_MIN_BUY:
                signal = 'BUY'
            else:
                signal = 'WATCH'
        elif score >= 0.25:
            signal = 'BUY' if ml >= ML_MIN_BUY else 'WATCH'
        elif score <= -0.55 and conf >= 55:
            if ml >= ML_MIN_STRONG_BUY and combined >= CMB_MIN_STRONG:
                signal = 'STRONG_SELL'
            elif ml >= ML_MIN_BUY:
                signal = 'SELL'
            else:
                signal = 'WATCH'
        elif score <= -0.25:
            signal = 'SELL' if ml >= ML_MIN_BUY else 'WATCH'
        else:
            signal = 'NEUTRAL'
        out.append({
            **item,
            'signal':        signal,
            'raw_score':     round(raw_score, 4),
            'decay_factor':  round(decay_factor, 4),
            'age_hours':     round(age_hours, 2),
            'decayed_score': round(score, 4),
        })
    # Sort: newest first within each signal tier so fresh articles always lead
    out.sort(key=lambda x: (
        0 if x['signal'] == 'NEUTRAL' else 1,
        float(x.get('received_at') or 0),
    ), reverse=True)
    return jsonify(ok=True, items=out, count=len(out))


@app.post('/api/news_refresh_now')
def api_news_refresh_now():
    """Wake the news-feed background thread immediately instead of waiting 45s."""
    _NEWS_REFRESH_EVENT.set()
    return jsonify(ok=True)


@app.get('/api/spread_alerts')
def api_spread_alerts():
    """Return recent pre-halt spread explosion alerts."""
    max_age = float(request.args.get('max_age_sec', 600))
    cutoff  = time.time() - max_age
    with _SPREAD_LOCK:
        alerts = [a for a in _SPREAD_ALERTS if a.get('fired_ts', 0) >= cutoff]
    return jsonify(ok=True, alerts=alerts, count=len(alerts))


@app.get('/api/halt_resume_predict')
def api_halt_resume_predict():
    """Predict resume price range for a currently halted symbol."""
    sym   = str(request.args.get('symbol') or '').strip().upper()
    price = _safe_float(request.args.get('halt_price'))
    code  = str(request.args.get('halt_code') or 'T1').strip().upper()
    pm_move   = _safe_float(request.args.get('pm_move_pct'))
    float_sh  = _safe_float(request.args.get('float_shares'))
    halt_hour = _safe_int(request.args.get('halt_hour_et'), None)
    if not sym or not price:
        return jsonify(ok=False, error='symbol and halt_price required'), 400
    pred = predict_halt_resume(sym, code, price,
                               pm_move_pct=pm_move,
                               float_shares=float_sh,
                               halt_hour_et=halt_hour)
    return jsonify(ok=True, symbol=sym, prediction=pred)


@app.get('/api/news_sniper')
def api_news_sniper():
    """Return recent sniper alerts (new catalyst headlines on watchlist symbols)."""
    max_age = float(request.args.get('max_age_sec', 1800))
    cutoff  = time.time() - max_age
    with _SNIPER_LOCK:
        alerts = [a for a in _SNIPER_ALERTS if a.get('fired_ts', 0) >= cutoff]
    return jsonify(ok=True, alerts=alerts, count=len(alerts))


@app.get('/api/news_sniper/stream')
def api_news_sniper_stream():
    """SSE stream: pushes a 'sniper' event the instant a new catalyst headline fires."""
    last_seen_ts = float(request.args.get('since', 0))

    def _generate():
        yield 'data: {"type":"connected"}\n\n'
        nonlocal last_seen_ts
        while True:
            with _SNIPER_LOCK:
                new_alerts = [
                    a for a in _SNIPER_ALERTS
                    if a.get('fired_ts', 0) > last_seen_ts
                ]
            if new_alerts:
                for alert in reversed(new_alerts):
                    last_seen_ts = max(last_seen_ts, alert.get('fired_ts', 0))
                    yield f'event: sniper\ndata: {json.dumps(alert)}\n\n'
            # Heartbeat every 20s to keep connection alive
            _NEWS_REFRESH_EVENT.wait(timeout=20)
            _NEWS_REFRESH_EVENT.clear()
            yield ': heartbeat\n\n'

    return Response(
        stream_with_context(_generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


# ── Market Events (Halts / LULD bands) ───────────────────────────────────────

_HALT_REASON_MAP = {
    'T1':   {'label': 'News Pending',       'note': 'Bullish — news imminent'},
    'T2':   {'label': 'News Released',      'note': 'News disseminated'},
    'T5':   {'label': 'Single Stock Pause', 'note': 'Volatility pause'},
    'LUDP': {'label': 'LULD Pause',         'note': '5-min halt — watch collar'},
    'T12':  {'label': 'NASDAQ Info Req',    'note': 'Neutral'},
    'H4':   {'label': 'Non-Compliance',     'note': 'Delisting risk'},
    'H10':  {'label': 'SEC Suspension',     'note': 'Do not trade'},
    'IPO1': {'label': 'IPO Not Trading',    'note': 'Watch first print'},
    'MWC1': {'label': 'Circuit Breaker L1', 'note': '7% market halt'},
    'MWC2': {'label': 'Circuit Breaker L2', 'note': '13% market halt'},
    'MWC3': {'label': 'MARKET CLOSED',      'note': '20% — day done'},
}


@app.get('/api/market_events')
def api_market_events():
    """Return recent halts, LULD bands, and imbalances for all monitored symbols."""
    stream = _STREAM
    halts = []
    lulds = []
    imbalances = []
    try:
        raw_halts = stream.recent_halt_resume_events(max_age_sec=1800) if stream else []
        for h in raw_halts:
            entry = dict(h)
            rc = str(entry.get('reason_code') or entry.get('status_code') or '')
            info = _HALT_REASON_MAP.get(rc, {})
            entry['reason_label'] = info.get('label', rc or 'Unknown')
            entry['reason_note']  = info.get('note', '')
            halts.append(entry)
    except Exception:
        pass
    try:
        lulds = stream.recent_luld_events(max_age_sec=3600) if stream else []
    except Exception:
        pass
    try:
        imbalances = stream.recent_imbalances(max_age_sec=1800) if stream else []
    except Exception:
        pass
    return jsonify(ok=True, halts=halts, lulds=lulds, imbalances=imbalances)


@app.get('/api/market_events/stream')
def api_market_events_stream():
    """SSE stream: pushes a 'market_event' event on halt or LULD changes."""
    def _generate():
        yield 'data: {"type":"connected"}\n\n'
        last_poll = time.time()
        seen_halt_ts: float = last_poll - 30
        seen_luld_ts: float = last_poll - 30
        seen_imb_ts:  float = last_poll - 30
        while True:
            _MARKET_EVENTS_EVENT.wait(timeout=15)
            _MARKET_EVENTS_EVENT.clear()
            try:
                stream = _STREAM
                if stream:
                    new_halts = [
                        h for h in stream.recent_halt_resume_events(max_age_sec=60)
                        if float(h.get('received_at', 0)) > seen_halt_ts
                    ]
                    new_lulds = [
                        l for l in stream.recent_luld_events(max_age_sec=60)
                        if float(l.get('received_at', 0)) > seen_luld_ts
                    ]
                    new_imbs = [
                        i for i in stream.recent_imbalances(max_age_sec=60)
                        if float(i.get('received_at', 0)) > seen_imb_ts
                    ]
                    if new_halts or new_lulds or new_imbs:
                        for h in new_halts:
                            seen_halt_ts = max(seen_halt_ts, float(h.get('received_at', 0)))
                            rc = str(h.get('reason_code') or h.get('status_code') or '')
                            info = _HALT_REASON_MAP.get(rc, {})
                            h['reason_label'] = info.get('label', rc or 'Unknown')
                            h['reason_note']  = info.get('note', '')
                        for l in new_lulds:
                            seen_luld_ts = max(seen_luld_ts, float(l.get('received_at', 0)))
                        for i in new_imbs:
                            seen_imb_ts = max(seen_imb_ts, float(i.get('received_at', 0)))
                        payload = {'halts': new_halts, 'lulds': new_lulds, 'imbalances': new_imbs}
                        yield f'event: market_event\ndata: {json.dumps(payload)}\n\n'
            except Exception:
                pass
            yield ': heartbeat\n\n'

    return Response(
        stream_with_context(_generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


# ── R-Multiple Outcome Tracker ────────────────────────────────────────────────

@app.post('/api/outcomes/record')
def api_outcomes_record():
    """Record a scan candidate as a tracked trade for R-multiple outcome tracking."""
    body = request.get_json(silent=True) or {}
    required = ('symbol', 'entry', 'stop')
    missing = [k for k in required if not body.get(k)]
    if missing:
        return jsonify(ok=False, error=f"missing fields: {missing}"), 400
    try:
        oid = _RUNTIME_STORE.record_trade_outcome(body)
        return jsonify(ok=True, outcome_id=oid)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 400


@app.get('/api/outcomes')
def api_outcomes_list():
    """List tracked trade outcomes. ?open_only=1 for unresolved, ?date=YYYY-MM-DD for one session."""
    open_only    = request.args.get('open_only', '0') in ('1', 'true', 'yes')
    session_date = str(request.args.get('date') or '').strip() or None
    limit        = _safe_int(request.args.get('limit', 200), 200)
    rows = _RUNTIME_STORE.get_trade_outcomes(session_date=session_date, open_only=open_only, limit=limit)
    # Compute R for open trades using latest monitor prices if available
    price_map: dict[str, float] = {}
    try:
        sess = _MONITOR.active_session()
        if sess:
            for sym, st in sess.symbols.items():
                p = getattr(st, 'price', None) or getattr(st, 'live_price', None)
                if p:
                    price_map[str(sym).upper()] = float(p)
    except Exception:
        pass
    out = []
    for r in rows:
        d = dict(r)
        sym = str(d.get('symbol') or '').upper()
        cur_price = price_map.get(sym)
        if cur_price and d.get('outcome') is None:
            direction = str(d.get('direction') or 'long').lower()
            entry     = float(d.get('entry') or 0)
            risk      = float(d.get('risk_per_share') or 0)
            if entry > 0 and risk > 0:
                if direction == 'long':
                    d['current_r'] = round((cur_price - entry) / risk, 2)
                else:
                    d['current_r'] = round((entry - cur_price) / risk, 2)
                d['current_price'] = cur_price
        out.append(d)
    total_r = sum(float(r.get('outcome_r') or 0) for r in out if r.get('outcome_r') is not None)
    wins  = sum(1 for r in out if str(r.get('outcome') or '').startswith('hit_'))
    stops = sum(1 for r in out if r.get('outcome') == 'stopped')
    return jsonify(ok=True, outcomes=out, count=len(out), stats={
        "total_r": round(total_r, 2), "wins": wins, "stops": stops,
        "win_rate": round(wins / (wins + stops) * 100, 1) if (wins + stops) > 0 else None,
    })


@app.post('/api/outcomes/resolve')
def api_outcomes_resolve():
    """Manually resolve a trade outcome."""
    body       = request.get_json(silent=True) or {}
    outcome_id = body.get('outcome_id')
    outcome    = str(body.get('outcome') or '').strip()
    price      = body.get('price')
    if not outcome_id or not outcome:
        return jsonify(ok=False, error="outcome_id and outcome required"), 400
    valid = {'hit_1r', 'hit_2r', 'hit_3r', 'stopped', 'expired', 'manual'}
    if outcome not in valid:
        return jsonify(ok=False, error=f"outcome must be one of {sorted(valid)}"), 400
    rows = _RUNTIME_STORE.resolve_trade_outcome(
        int(outcome_id), outcome=outcome, price=float(price) if price else None,
    )
    return jsonify(ok=True, rows_updated=rows)


def _fetch_outcome_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch current prices for a list of symbols using Alpaca snapshots."""
    price_map: dict[str, float] = {}
    if not symbols or _ALPACA_PROVIDER is None:
        return price_map
    try:
        snaps = _ALPACA_PROVIDER.get_snapshots(symbols)
        for sym, snap in (snaps or {}).items():
            p = snap.get("reference_price") or snap.get("latest_trade_price")
            if p and float(p) > 0:
                price_map[sym.upper()] = float(p)
    except Exception:
        pass
    # Fall back to individual trades for any still missing
    missing = [s for s in symbols if s.upper() not in price_map]
    for sym in missing:
        try:
            trade = _ALPACA_PROVIDER.get_latest_trade(sym)
            p = float(trade.get("price") or trade.get("p") or 0)
            if p > 0:
                price_map[sym.upper()] = p
        except Exception:
            pass
    return price_map


def _run_auto_resolve_outcomes() -> list[int]:
    """Fetch live prices for all open outcomes and resolve any that hit a target or stop."""
    open_rows = _RUNTIME_STORE.get_trade_outcomes(open_only=True, limit=500)
    if not open_rows:
        return []
    price_map: dict[str, float] = {}
    # Try monitor first (no extra API call)
    try:
        sess = _MONITOR.active_session()
        if sess:
            for sym, st in sess.symbols.items():
                p = getattr(st, 'price', None) or getattr(st, 'live_price', None)
                if p:
                    price_map[str(sym).upper()] = float(p)
    except Exception:
        pass
    # Fetch from Alpaca for any symbols the monitor doesn't have
    syms_needed = [str(r["symbol"]).upper() for r in open_rows
                   if r.get("symbol") and str(r["symbol"]).upper() not in price_map]
    if syms_needed:
        price_map.update(_fetch_outcome_prices(syms_needed))
    if not price_map:
        return []
    return _RUNTIME_STORE.auto_resolve_trade_outcomes(price_map)


@app.post('/api/outcomes/resolve_auto')
def api_outcomes_resolve_auto():
    """Auto-resolve open outcomes using live Alpaca prices (monitor not required)."""
    resolved = _run_auto_resolve_outcomes()
    return jsonify(ok=True, resolved=resolved, count=len(resolved))


@app.post('/api/outcomes/runner_stop')
def api_outcomes_set_runner_stop():
    """Set runner stop price on a tracked outcome."""
    body = request.get_json(silent=True) or {}
    outcome_id  = body.get('outcome_id')
    runner_stop = body.get('runner_stop')
    if not outcome_id or runner_stop is None:
        return jsonify(ok=False, error='missing outcome_id or runner_stop'), 400
    try:
        n = _RUNTIME_STORE.set_runner_stop(int(outcome_id), float(runner_stop))
        return jsonify(ok=True, updated=n)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 400


@app.post('/api/outcomes/shakeout_watch')
def api_outcomes_set_shakeout_watch():
    """Flag a stopped outcome as a shakeout watch. Optionally supply flush_low."""
    body      = request.get_json(silent=True) or {}
    outcome_id = body.get('outcome_id')
    flush_low  = body.get('flush_low')
    if not outcome_id:
        return jsonify(ok=False, error='missing outcome_id'), 400
    try:
        n = _RUNTIME_STORE.set_shakeout_watch(
            int(outcome_id),
            float(flush_low) if flush_low is not None else None,
        )
        return jsonify(ok=True, updated=n)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 400


@app.get('/api/outcomes/shakeout_watches')
def api_outcomes_shakeout_watches():
    """Return all outcomes flagged as shakeout watches."""
    watches = _RUNTIME_STORE.get_shakeout_watches()
    # Build a symbol-keyed map for easy JS lookup
    sym_map: dict[str, dict] = {}
    for w in watches:
        sym = str(w.get('symbol') or '').upper()
        if sym and sym not in sym_map:
            sym_map[sym] = w
    return jsonify(ok=True, watches=watches, by_symbol=sym_map)


@app.get('/api/benchmark_scorecard')
def api_benchmark_scorecard():
    return jsonify(ok=True, scorecard=_benchmark_scorecard_payload())


@app.get('/api/context_status')
def api_context_status():
    try:
        ctx = _CONTEXT_ENGINE.snapshot() if '_CONTEXT_ENGINE' in globals() else {}
    except Exception:
        ctx = {}
    return jsonify(ok=True, context=ctx)


@app.get('/api/alerts_recent')
def api_alerts_recent():
    monitor_id = str(request.args.get('monitor_id') or '').strip() or None
    limit = min(int(request.args.get('limit') or 25), 100)
    try:
        alerts = _MONITOR.recent_alerts(monitor_id=monitor_id, limit=limit)
    except Exception as e:
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 500
    return jsonify(ok=True, alerts=alerts)


@app.get('/api/alerts_stream')
def api_alerts_stream():
    """SSE stream that emits 'alert' events as new monitor alerts arrive.

    Polls in-memory session alerts every 2s and emits any with event_ts > since.
    Falls back gracefully when no session is active.
    """
    import time as _time
    import json as _json_sse

    monitor_id = str(request.args.get('monitor_id') or '').strip() or None
    try:
        since = float(request.args.get('since') or 0)
    except (TypeError, ValueError):
        since = 0.0

    def _generate():
        nonlocal since
        deadline = _time.time() + 55.0  # close before gunicorn/nginx 60s timeout
        yield ': ping\n\n'
        while _time.time() < deadline:
            try:
                alerts = _MONITOR.recent_alerts(monitor_id=monitor_id, limit=50)
                new = [a for a in alerts if float(a.get('event_ts') or 0) > since]
                new.sort(key=lambda a: float(a.get('event_ts') or 0))
                for a in new:
                    since = max(since, float(a.get('event_ts') or 0))
                    yield f"event: alert\ndata: {_json_sse.dumps(a, separators=(',', ':'), default=str)}\n\n"
            except Exception:
                pass
            _time.sleep(2.0)
        yield ': close\n\n'

    return Response(
        stream_with_context(_generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.get("/api/watchlist_snapshot")
def api_watchlist_snapshot():
    """
    Manual refresh for monitor/watchlist prices using Alpaca STREAM latest trade/quote.
    No REST fallback. Returns per-symbol real failures in rows[sym].error.
    """
    syms_raw = request.args.get("symbols", "")
    symbols = [x.strip().upper() for x in syms_raw.split(",") if x.strip()]
    symbols = symbols[:25]
    if not symbols:
        return jsonify({"ok": True, "symbols": [], "rows": {}})
    try:
        stream_cache = _ensure_stream(start=True, require=True)
    except Exception as e:
        return jsonify({"ok": False, "symbols": symbols, "rows": {}, "error": f"{type(e).__name__}: {e}"}), 503
    if stream_cache is None:
        return jsonify({"ok": False, "symbols": symbols, "rows": {}, "error": (_STREAM_ERROR or "stream_not_initialized")}), 503

    try:
        stream_cache.ensure_symbols(symbols)
    except Exception as e:
        return jsonify({"ok": False, "symbols": symbols, "rows": {}, "error": f"{type(e).__name__}: {e}"}), 503

    rows = {}
    for sym in symbols:
        try:
            q = latest_quote_payload(stream_cache, sym, max_age_sec=30.0)
            t = latest_trade_payload(stream_cache, sym, max_age_sec=30.0)
            rows[sym] = {
                "quote": {
                    "bp": q.get("bid"),
                    "ap": q.get("ask"),
                    "bs": q.get("bid_size"),
                    "as": q.get("ask_size"),
                    "t": q.get("timestamp") or q.get("quote_ts"),
                },
                "trade": {
                    "p": t.get("price"),
                    "s": t.get("size"),
                    "x": t.get("exchange"),
                    "c": t.get("conditions"),
                    "t": t.get("timestamp"),
                },
                "error": None,
            }
        except Exception as e:
            rows[sym] = {"quote": {}, "trade": {}, "error": str(e)}

    return jsonify({"ok": True, "symbols": symbols, "rows": rows})

@app.get("/api/plan_integrity")
def api_plan_integrity():
    """Run a full integrity audit on today's (or a specified date's) plan snapshots."""
    from core.plan_integrity import audit_plan_snapshots
    from datetime import date as _date
    session_date = request.args.get("date") or _date.today().isoformat()
    result = audit_plan_snapshots(_RUNTIME_STORE, session_date=session_date)
    ok = result.get("corrupt_snapshots", 0) == 0
    return jsonify(ok=ok, **result)


@app.get("/api/monitor_active")
def api_monitor_active():
    sess = _MONITOR.active_session()
    if sess is None:
        return jsonify(ok=False, error="no_active_monitor"), 404
    return jsonify(ok=True, monitor_id=sess.monitor_id, symbols=sorted(sess.symbols.keys()), watch_mode=sess.watch_mode, source=sess.source)


@app.get("/api/monitor_status")
def api_monitor_status():
    monitor_id = str(request.args.get("monitor_id") or "").strip()
    if not monitor_id:
        # Auto-resolve to active session if no ID given
        sess = _MONITOR.active_session()
        if sess:
            monitor_id = sess.monitor_id
        else:
            return jsonify(ok=False, error="missing_monitor_id"), 400

    no_refresh = _is_truthy(request.args.get("no_refresh", "0"))
    try:
        stream_cache = _ensure_stream(start=not no_refresh, require=not no_refresh)
    except Exception as e:
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}", monitor_id=monitor_id), 503

    try:
        payload = _MONITOR.status(
            monitor_id,
            provider=_ALPACA_PROVIDER,
            stream_cache=stream_cache,
            refresh=(not no_refresh),
        )
        if not isinstance(payload, dict):
            return jsonify(ok=False, error="invalid_monitor_status", monitor_id=monitor_id), 500
        payload.setdefault("ok", True)
        payload.setdefault("monitor_id", monitor_id)
        return jsonify(payload)
    except KeyError:
        return jsonify(ok=False, error="unknown_monitor_id", monitor_id=monitor_id), 404
    except Exception as e:
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}", monitor_id=monitor_id), 500


# ── Tape Tracker ────────────────────────────────────────────────────────────────

_tape_bars_cache: dict[str, tuple[float, Any]] = {}
_TAPE_BARS_TTL = 60.0  # refresh bars every 60s per symbol


def _tape_compute_signal(
    sym: str,
    snap: dict,
    spy_snap: dict,
    qqq_snap: dict,
    bars_df: Any,
    entry_price: float | None = None,
    stop_price: float | None = None,
    plan_side: str | None = None,
) -> dict:
    """Compute L1/L2/L3 signals, composite bull/bear score, and pre-entry confidence for one symbol."""
    result: dict = {
        "symbol": sym,
        "price": None,
        "change_pct": None,
        "l1": {"score": 50, "imbalance": None, "bid_size": None, "ask_size": None,
               "bid_price": None, "ask_price": None, "direction": "neutral"},
        "l2": {"score": 50, "vwap": None, "above_vwap": None, "vwap_delta_pct": None,
               "structure": "neutral", "vol_quality": None, "momentum": None},
        "l3": {"score": 50, "spy_pct": None, "qqq_pct": None, "rvol": None, "news_count": 0},
        "composite": 50,
        "signal": "NEUTRAL",
        "error": snap.get("error"),
        "pre_entry": {"signal": "NONE", "score": 0, "atr_ratio": None,
                      "vol_slope": None, "dist_atr": None, "higher_lows": None,
                      "notes": [], "entry": entry_price, "stop": stop_price,
                      "side": plan_side},
    }

    qt  = snap.get("latest_quote") or {}
    db  = snap.get("daily_bar") or {}
    pb  = snap.get("prev_daily_bar") or {}

    price = snap.get("reference_price")
    if price is not None:
        result["price"] = round(price, 4)

    prev_close = pb.get("close")
    if price and prev_close and prev_close > 0:
        result["change_pct"] = round((price - prev_close) / prev_close * 100, 2)

    # ── L1: Tape (bid/ask imbalance + price direction) ──────────────────────
    bid_size  = qt.get("bid_size")  or 0
    ask_size  = qt.get("ask_size")  or 0
    bid_price = qt.get("bid_price")
    ask_price = qt.get("ask_price")

    result["l1"]["bid_size"]  = bid_size
    result["l1"]["ask_size"]  = ask_size
    result["l1"]["bid_price"] = bid_price
    result["l1"]["ask_price"] = ask_price

    l1 = 50.0

    # Alpaca returns bid/ask sizes as None after market close (no active quotes).
    # Treat missing sizes as 0 for score math, but track whether real size data exists.
    _has_size_data = (bid_size > 0 or ask_size > 0)
    total_size = bid_size + ask_size
    if total_size > 0:
        imbalance = (bid_size - ask_size) / total_size          # [-1, 1]
        result["l1"]["imbalance"] = round(imbalance, 3)
        l1 += imbalance * 25                                    # ±25 pts

    pct = result["change_pct"] or 0.0
    dir_pts = min(max(pct * 5.0, -15.0), 15.0)
    l1 += dir_pts

    # Dir = where price is actually going RIGHT NOW (last 2 bar closes).
    # Imbalance informs the score above but the arrow must match what the trader
    # sees on the chart — price lifting through a 52k ask wall is ▲, not ▼.
    # Fallback chain: recent bars → imbalance → flat (never daily change).
    _dir_set = False
    if bars_df is not None and not getattr(bars_df, "empty", True) and len(bars_df) >= 2:
        try:
            _recent = bars_df["Close"].astype(float).iloc[-3:].tolist()
            if len(_recent) >= 2:
                _net = _recent[-1] - _recent[0]
                if _net > 0.001:
                    result["l1"]["direction"] = "up"
                    _dir_set = True
                elif _net < -0.001:
                    result["l1"]["direction"] = "down"
                    _dir_set = True
        except Exception:
            pass
    if not _dir_set:
        _imb = result["l1"]["imbalance"]
        if _imb is not None and _has_size_data:
            result["l1"]["direction"] = "up" if _imb > 0.1 else ("down" if _imb < -0.1 else "flat")
        else:
            result["l1"]["direction"] = "flat"

    if bid_size > 0:
        ask_ratio = ask_size / max(bid_size, 1)
        if ask_ratio < 0.5:
            l1 += 10                                            # thin ask = bullish
        elif ask_ratio > 2.0:
            l1 -= 10                                            # fat ask = bearish

    result["l1"]["score"] = round(min(max(l1, 0), 100))

    # ── L2: Structure (VWAP, price structure, volume quality, momentum) ──────
    l2 = 50.0

    if bars_df is not None and not getattr(bars_df, "empty", True) and len(bars_df) >= 3:
        try:
            from scanner.indicators import vwap as _vwap, trend_state_1m as _trend
            vw    = _vwap(bars_df)
            tr_d  = _trend(bars_df, vw=vw, lookback=15)

            vwap_last      = tr_d.get("vwap_last")
            vwap_delta_pct = float(tr_d.get("vwap_delta_pct") or 0.0)

            result["l2"]["vwap"]           = round(float(vwap_last), 4) if vwap_last else None
            result["l2"]["vwap_delta_pct"] = round(vwap_delta_pct, 2)
            result["l2"]["above_vwap"]     = bool(price > float(vwap_last)) if (price and vwap_last) else None

            l2 += min(max(vwap_delta_pct * 8.0, -20.0), 20.0)  # ±20 pts

            closes = bars_df["Close"].astype(float).tail(10).tolist() if "Close" in bars_df.columns else []
            if len(closes) >= 3:
                up_c = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
                dn_c = len(closes) - 1 - up_c
                struct = (up_c - dn_c) / max(len(closes) - 1, 1)   # [-1, 1]
                result["l2"]["structure"] = "up" if struct > 0.2 else ("down" if struct < -0.2 else "neutral")
                l2 += struct * 12.5                                  # ±12.5 pts

            if all(c in bars_df.columns for c in ("Close", "Open", "Volume")):
                tail = bars_df.tail(10)
                cl = tail["Close"].astype(float)
                op = tail["Open"].astype(float)
                vl = tail["Volume"].astype(float)
                up_vol  = float(vl[cl > op].sum())
                dn_vol  = float(vl[cl <= op].sum())
                tot_vol = up_vol + dn_vol
                if tot_vol > 0:
                    vq = (up_vol - dn_vol) / tot_vol               # [-1, 1]
                    result["l2"]["vol_quality"] = round(vq, 2)
                    l2 += vq * 10.0                                 # ±10 pts

            if len(closes) >= 5:
                avg5 = sum(closes[-5:-1]) / 4.0
                mom  = (closes[-1] - avg5) / max(avg5, 0.01) * 100
                result["l2"]["momentum"] = round(mom, 2)
                l2 += min(max(mom * 3.0, -7.5), 7.5)              # ±7.5 pts

        except Exception:
            pass

    result["l2"]["score"] = round(min(max(l2, 0), 100))

    # ── L3: Context (SPY, QQQ, RVOL, news freshness) ────────────────────────
    l3 = 50.0

    def _ref_pct(sn: dict) -> float | None:
        p  = sn.get("reference_price")
        pb = (sn.get("prev_daily_bar") or {}).get("close")
        if p and pb and pb > 0:
            return (p - pb) / pb * 100
        return None

    spy_pct = _ref_pct(spy_snap)
    qqq_pct = _ref_pct(qqq_snap)
    result["l3"]["spy_pct"] = round(spy_pct, 2) if spy_pct is not None else None
    result["l3"]["qqq_pct"] = round(qqq_pct, 2) if qqq_pct is not None else None

    if spy_pct is not None:
        l3 += min(max(spy_pct * 8.0, -20.0), 20.0)   # ±20 pts
    if qqq_pct is not None:
        l3 += min(max(qqq_pct * 4.0, -10.0), 10.0)   # ±10 pts

    # Simple RVOL: today vol vs yesterday vol
    today_vol = db.get("volume")
    yest_vol  = pb.get("volume")
    if today_vol and yest_vol and yest_vol > 0:
        try:
            import pytz as _pytz
            _et = _pytz.timezone("America/New_York")
            _now_et = datetime.now(_et)
            _open_et = _now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            _elapsed_raw = (_now_et - _open_et).total_seconds() / 60.0
            # Only pace-adjust during market hours (0–390 min after open).
            # Outside that window (pre-open, after-close, weekend) use raw ratio —
            # dividing by near-zero pct_sess inflates RVOL by 390x+ otherwise.
            if 1.0 <= _elapsed_raw <= 390.0:
                _pct_sess = _elapsed_raw / 390.0
                rvol = (today_vol / _pct_sess) / yest_vol
            else:
                rvol = today_vol / yest_vol
            result["l3"]["rvol"] = round(rvol, 2)
            if rvol > 2.5:
                l3 += 12.5
            elif rvol > 1.5:
                l3 += 7.5
            elif rvol > 1.0:
                l3 += 2.5
            elif rvol < 0.5:
                l3 -= 10.0
        except Exception:
            pass

    # News freshness
    try:
        _cutoff_2h = (datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()
        _news = _RUNTIME_STORE.recent_news(symbol=sym, limit=20)

        def _pts(it: dict) -> float:
            raw = it.get("published_at") or ""
            try:
                from dateutil.parser import parse as _p
                return _p(raw).timestamp() if raw else 0.0
            except Exception:
                return float(it.get("received_at") or 0)

        fresh_bull = sum(
            1 for it in _news
            if _pts(it) >= _cutoff_2h and float(it.get("sentiment_score") or it.get("catalyst_score") or 0) > 0.2
        )
        result["l3"]["news_count"] = fresh_bull
        if fresh_bull >= 3:
            l3 += 7.5
        elif fresh_bull >= 1:
            l3 += 3.0
    except Exception:
        pass

    result["l3"]["score"] = round(min(max(l3, 0), 100))

    # ── Composite ────────────────────────────────────────────────────────────
    composite = (
        result["l1"]["score"] * 0.40 +
        result["l2"]["score"] * 0.40 +
        result["l3"]["score"] * 0.20
    )
    # ── L2 book pressure (from streaming orderbook if available) ─────────────
    result["book"] = {"pressure": None, "bid_wall": None, "ask_wall": None, "supported": False}
    try:
        sc = _STREAM
        if sc is not None:
            book = sc.latest_orderbook(sym, levels=10)
            result["book"] = {
                "pressure":  book.get("pressure"),
                "bid_wall":  book.get("bid_wall"),
                "ask_wall":  book.get("ask_wall"),
                "supported": book.get("supported", False),
                "bids":      book.get("bids", [])[:5],
                "asks":      book.get("asks", [])[:5],
            }
            # Book pressure nudges the composite: >0.6 is bullish, <0.4 bearish
            if book.get("pressure") is not None:
                p = float(book["pressure"])
                composite += (p - 0.5) * 12.0   # ±6 pts max
    except Exception:
        pass

    result["composite"] = round(min(max(composite, 0), 100))
    result["signal"] = "BULL" if composite >= 62 else ("BEAR" if composite <= 38 else "NEUTRAL")

    # ── Pre-Entry Confidence Score ────────────────────────────────────────────
    # Only computed when entry+stop are provided; never touches scanner code.
    pe = result["pre_entry"]
    if (entry_price is not None and stop_price is not None
            and bars_df is not None and not getattr(bars_df, "empty", True)
            and len(bars_df) >= 10):
        try:
            _closes = bars_df["Close"].astype(float)
            _highs  = bars_df["High"].astype(float)
            _lows   = bars_df["Low"].astype(float)
            _vols   = bars_df["Volume"].astype(float)
            _c      = _closes.values
            _h      = _highs.values
            _l      = _lows.values
            _v      = _vols.values

            def _atr_n(n: int) -> float | None:
                trs = []
                for i in range(1, min(n + 1, len(_c))):
                    tr = max(_h[-i] - _l[-i],
                             abs(_h[-i] - _c[-i - 1]),
                             abs(_l[-i] - _c[-i - 1]))
                    trs.append(tr)
                return float(sum(trs) / len(trs)) if trs else None

            atr5  = _atr_n(5)
            atr14 = _atr_n(14)
            pe_score = 0
            pe_notes: list[str] = []
            # Side resolution: explicit > price-based inference > entry/stop relationship
            _cur_px_pre = result["price"] or float(_c[-1])
            if plan_side in ("long", "short"):
                _side = plan_side
            elif _cur_px_pre != entry_price:
                _side = "long" if _cur_px_pre > entry_price else "short"
            else:
                _side = "long" if entry_price > stop_price else "short"
            pe["side"] = _side

            # 1. ATR compression
            if atr5 and atr14 and atr14 > 0:
                atr_ratio = atr5 / atr14
                pe["atr_ratio"] = round(atr_ratio, 3)
                if atr_ratio < 0.70:
                    pe_score += 25; pe_notes.append("ATR compressing")
                elif atr_ratio < 0.85:
                    pe_score += 12; pe_notes.append("ATR contracting")

            # 2. Volume accumulation slope (recent 5 vs prior 5)
            if len(_v) >= 10:
                rv = float(_v[-5:].mean())
                pv = float(_v[-10:-5].mean())
                if pv > 0:
                    vol_slope = (rv - pv) / pv
                    pe["vol_slope"] = round(vol_slope, 3)
                    if vol_slope > 0.25:
                        pe_score += 25; pe_notes.append("volume accumulating")
                    elif vol_slope > 0.0:
                        pe_score += 10

            # 3. Distance from current price to entry trigger (in ATR units)
            _cur_px = result["price"] or float(_c[-1])
            if atr5 and atr5 > 0:
                dist = abs(entry_price - _cur_px)
                dist_atr = dist / atr5
                pe["dist_atr"] = round(dist_atr, 2)
                if dist_atr <= 0.5:
                    pe_score += 30; pe_notes.append("price near trigger")
                elif dist_atr <= 1.0:
                    pe_score += 15; pe_notes.append("approaching trigger")
                elif dist_atr > 3.5:
                    pe_score -= 10

            # 4. Higher lows (long) / lower highs (short) in last 6 bars
            if len(_l) >= 6 and _side == "long":
                hl = sum(1 for i in range(1, 6) if _l[-i] > _l[-i - 1])
                pe["higher_lows"] = hl >= 3
                if hl >= 4:
                    pe_score += 20; pe_notes.append("higher lows forming")
                elif hl >= 3:
                    pe_score += 10
            elif len(_h) >= 6 and _side == "short":
                lh = sum(1 for i in range(1, 6) if _h[-i] < _h[-i - 1])
                pe["higher_lows"] = lh >= 3
                if lh >= 4:
                    pe_score += 20; pe_notes.append("lower highs forming")
                elif lh >= 3:
                    pe_score += 10

            pe["score"]  = round(min(max(pe_score, 0), 100))
            pe["notes"]  = pe_notes[:3]
            if pe_score >= 65:
                pe["signal"] = "EARLY"
            elif pe_score >= 35:
                pe["signal"] = "WATCH"
            else:
                pe["signal"] = "WAIT"
        except Exception:
            pass

    return result


@app.get('/tape')
def tape_page():
    return render_template('tape.html')


@app.get('/api/tape_signal')
def api_tape_signal():
    symbols_raw = (request.args.get('symbols') or '').strip().upper()
    if not symbols_raw:
        return jsonify(ok=False, error='symbols required'), 400

    symbols = [s.strip() for s in symbols_raw.split(',') if s.strip()][:20]
    if not symbols:
        return jsonify(ok=False, error='no valid symbols'), 400

    # Optional per-symbol plans: plans=SLNH:1.93:1.79:L,EOSE:7.74:7.81:L
    # Format: SYM:entry:stop[:L|S]  — side is optional, L=long S=short
    _plans: dict[str, tuple[float, float, str | None]] = {}
    plans_raw = (request.args.get('plans') or '').strip().upper()
    if plans_raw:
        for chunk in plans_raw.split(','):
            parts = chunk.strip().split(':')
            if len(parts) >= 3:
                try:
                    sym_key = parts[0].strip()
                    entry_v = float(parts[1])
                    stop_v  = float(parts[2])
                    side_v  = None
                    if len(parts) >= 4:
                        raw_side = parts[3].strip()
                        if raw_side in ('L', 'LONG'):
                            side_v = 'long'
                        elif raw_side in ('S', 'SHORT'):
                            side_v = 'short'
                    _plans[sym_key] = (entry_v, stop_v, side_v)
                except (ValueError, IndexError):
                    pass

    provider = _ALPACA_PROVIDER
    if provider is None:
        return jsonify(ok=False, error=_PROVIDER_ERROR or 'provider_not_initialized'), 503

    all_syms = list(dict.fromkeys(symbols + ['SPY', 'QQQ']))
    try:
        snapshots = provider.get_snapshots(all_syms)
    except Exception as exc:
        return jsonify(ok=False, error=f'snapshot_failed: {exc}'), 500

    spy_snap = snapshots.get('SPY', {})
    qqq_snap = snapshots.get('QQQ', {})

    now_ts = time.time()
    signals: dict = {}

    for sym in symbols:
        snap = snapshots.get(sym, {})

        # Use cached bars if fresh, else fetch
        bars_df = None
        cached = _tape_bars_cache.get(sym)
        if cached and (now_ts - cached[0]) < _TAPE_BARS_TTL:
            bars_df = cached[1]
        else:
            try:
                bars_df = provider.get_bars(
                    BarsRequest(symbol=sym, interval='1m', period='1d')
                ).sort_index()
                _tape_bars_cache[sym] = (now_ts, bars_df)
            except Exception:
                bars_df = None

        if sym in _plans:
            entry_p, stop_p, side_p = _plans[sym]
        else:
            entry_p, stop_p, side_p = None, None, None
        signals[sym] = _tape_compute_signal(sym, snap, spy_snap, qqq_snap, bars_df,
                                            entry_price=entry_p, stop_price=stop_p,
                                            plan_side=side_p)

    return jsonify(ok=True, signals=signals, ts=now_ts)


@app.get('/api/l2_book')
def api_l2_book():
    """Level 2 order book + Time & Sales for a symbol.

    Requires the Alpaca stream to be running (api_monitor_start) and subscribed
    to this symbol.  If orderbook feed is not available on the current Alpaca
    subscription tier, returns supported=false with the quote-derived L1 fallback.
    """
    sym = (request.args.get('symbol') or '').strip().upper()
    if not sym:
        return jsonify(ok=False, error='symbol required'), 400

    stream_cache = _STREAM
    if stream_cache is None:
        return jsonify(ok=False, error='stream_not_running', supported=False), 503

    # Ensure symbol is subscribed
    stream_cache.ensure_symbols([sym])

    book  = stream_cache.latest_orderbook(sym, levels=15)
    tns   = stream_cache.recent_trades(sym, limit=100)

    # Annotate time & sales with uptick/downtick
    annotated: list[dict] = []
    prev_price: float | None = None
    for tr in tns:
        p = tr.get("price")
        tick = "up" if (prev_price is not None and p is not None and p > prev_price) else (
               "down" if (prev_price is not None and p is not None and p < prev_price) else "flat")
        annotated.append({**tr, "tick": tick})
        if p is not None:
            prev_price = p

    # L1 quote fallback if L2 not supported
    l1_fallback: dict | None = None
    if not book["supported"] or (not book["bids"] and not book["asks"]):
        try:
            qt = stream_cache.latest_quote(sym)
            if qt:
                l1_fallback = {
                    "bid_price": qt.get("bid_price"),
                    "bid_size":  qt.get("bid_size"),
                    "ask_price": qt.get("ask_price"),
                    "ask_size":  qt.get("ask_size"),
                    "ts":        qt.get("timestamp"),
                }
        except Exception:
            pass

    return jsonify(
        ok=True,
        symbol=sym,
        book=book,
        tns=annotated[-100:],
        l1_fallback=l1_fallback,
        ts=time.time(),
    )


@app.get('/api/morning_validation')
def api_morning_validation():
    """SSE stream — runs 10 real checks and benchmarks each against commercial tool standards.

    Each event: {"check": str, "status": "pass"|"warn"|"fail", "actual": any, "benchmark": any, "msg": str}
    Final event: {"done": true, "score": int, "grade": "A"|"B"|"C"|"D", "summary": str}
    """
    provider = _ALPACA_PROVIDER

    def _run():
        import json as _json
        results: list[dict] = []

        def _emit(check: str, status: str, actual, benchmark, msg: str) -> str:
            row = {"check": check, "status": status, "actual": actual,
                   "benchmark": benchmark, "msg": msg}
            results.append(row)
            return f"data: {_json.dumps(row)}\n\n"

        # ── 1. Alpaca API connectivity ──────────────────────────────────────
        yield "data: {\"progress\": \"Checking Alpaca API connectivity...\"}\n\n"
        try:
            if provider is None:
                raise RuntimeError("provider not initialized")
            t0 = time.time()
            snaps = provider.get_snapshots(["SPY"], feed="sip", timeout_s=6.0)
            latency_ms = round((time.time() - t0) * 1000)
            spy_price = (snaps.get("SPY") or {}).get("reference_price")
            if spy_price and latency_ms < 2000:
                yield _emit("Alpaca API", "pass", f"{latency_ms}ms", "<2000ms",
                            f"SPY quote received in {latency_ms}ms (${spy_price:.2f})")
            elif spy_price:
                yield _emit("Alpaca API", "warn", f"{latency_ms}ms", "<2000ms",
                            f"Slow response: {latency_ms}ms — check network")
            else:
                yield _emit("Alpaca API", "fail", None, "SPY quote", "Snapshot returned no price for SPY")
        except Exception as e:
            yield _emit("Alpaca API", "fail", str(e), "connected", f"API call failed: {e}")

        # ── 2. Live multi-symbol quotes ────────────────────────────────────
        yield "data: {\"progress\": \"Spot-checking live quotes...\"}\n\n"
        try:
            CHECK_SYMS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
            snaps = provider.get_snapshots(CHECK_SYMS, feed="sip", timeout_s=8.0)
            live = [s for s in CHECK_SYMS if (snaps.get(s) or {}).get("reference_price")]
            pct = len(live) / len(CHECK_SYMS) * 100
            status = "pass" if pct == 100 else ("warn" if pct >= 60 else "fail")
            yield _emit("Live quotes", status, f"{len(live)}/{len(CHECK_SYMS)} symbols", "5/5",
                        f"Quotes live: {', '.join(live)}")
        except Exception as e:
            yield _emit("Live quotes", "fail", str(e), "5/5 symbols", f"Snapshot failed: {e}")

        # ── 3. Streaming connection ────────────────────────────────────────
        yield "data: {\"progress\": \"Checking streaming connection...\"}\n\n"
        try:
            sc = _STREAM
            if sc is None:
                yield _emit("Stream", "warn", "not started", "connected",
                            "Start monitor first to enable real-time streaming")
            else:
                state = sc.state()
                age = round(time.time() - (state.last_event_at or 0))
                if state.connected and age < 30:
                    yield _emit("Stream", "pass", f"connected, last event {age}s ago",
                                "connected + event <30s", "Live stream healthy")
                elif state.connected:
                    yield _emit("Stream", "warn", f"connected but stale ({age}s)", "<30s",
                                "Stream connected but no recent events — stale data risk")
                else:
                    yield _emit("Stream", "warn", "not connected", "connected",
                                f"Stream not connected: {state.error or 'unknown'}")
        except Exception as e:
            yield _emit("Stream", "fail", str(e), "connected", f"Stream check failed: {e}")

        # ── 4. L2 order book availability ─────────────────────────────────
        yield "data: {\"progress\": \"Checking L2 order book...\"}\n\n"
        try:
            sc = _STREAM
            if sc is None:
                yield _emit("L2 book", "warn", "stream not started", "streaming",
                            "Start monitor to enable L2 order book")
            elif not sc._orderbook_supported:
                yield _emit("L2 book", "info", "L1 NBBO only (orderbook not in alpaca-py SDK)", "streaming",
                            "alpaca-py v0.43.x does not implement subscribe_orderbooks. Your plan supports L2 — SDK upgrade needed.")
            else:
                sc.ensure_symbols(["SPY"])
                import asyncio as _asyncio; _asyncio.sleep(0)   # yield to stream thread
                time.sleep(0.5)
                book = sc.latest_orderbook("SPY", levels=5)
                if book["bids"] or book["asks"]:
                    yield _emit("L2 book", "pass",
                                f"{len(book['bids'])} bid levels, {len(book['asks'])} ask levels",
                                "10 levels each side", "L2 depth streaming")
                else:
                    yield _emit("L2 book", "warn", "no levels yet", "streaming",
                                "L2 subscription active but no book data yet — subscribe to a symbol first")
        except Exception as e:
            yield _emit("L2 book", "warn", str(e), "streaming", f"L2 check failed: {e}")

        # ── 5. News feed ───────────────────────────────────────────────────
        yield "data: {\"progress\": \"Checking news feed...\"}\n\n"
        try:
            cutoff = time.time() - 86400  # last 24h
            news = _RUNTIME_STORE.recent_news(limit=50)
            fresh = [n for n in news if float(n.get("received_at") or 0) >= cutoff]
            if len(fresh) >= 10:
                yield _emit("News feed", "pass", f"{len(fresh)} articles (24h)", "≥10/24h",
                            "News pipeline active")
            elif len(fresh) >= 1:
                yield _emit("News feed", "warn", f"{len(fresh)} articles (24h)", "≥10/24h",
                            "Low news volume — check news source connection")
            else:
                yield _emit("News feed", "fail", "0 articles (24h)", "≥10/24h",
                            "No news in last 24h — news feed may be down")
        except Exception as e:
            yield _emit("News feed", "fail", str(e), "≥10/24h", f"News check failed: {e}")

        # ── 6. ATR expansion scanner smoke test ───────────────────────────
        yield "data: {\"progress\": \"Running ATR expansion scanner smoke test...\"}\n\n"
        try:
            from scanner.orb import scan_symbols as _scan, ORBConfig as _ORBConfig
            _cfg = _ORBConfig(min_price=1.0, max_price=500.0, min_rvol=0.5,
                              min_avg20_dollar_vol=0, min_today_dollar_vol=0)
            t0 = time.time()
            result = _scan(
                symbols=[],
                cfg=_cfg,
                limit=20,
                strategy="atr_expansion",
                provider=provider,
                min_grade_enabled=False,
                min_combined_enabled=False,
                use_ml=False, use_sentiment=False, use_catalyst=False,
            )
            elapsed = round(time.time() - t0, 1)
            cands = (result or {}).get("candidates") or (result or {}).get("all") or []
            n = len(cands)

            def _sym(c):
                return c.get("symbol") if isinstance(c, dict) else getattr(c, "symbol", "?")

            if n >= 3:
                yield _emit("ATR scan", "pass", f"{n} candidates in {elapsed}s", "≥3 candidates",
                            f"Scanner operational: {', '.join(_sym(c) for c in list(cands)[:5])}")
            elif n >= 1:
                yield _emit("ATR scan", "warn", f"{n} candidates in {elapsed}s", "≥3 candidates",
                            "Low candidate count — market may be quiet or pre-market")
            else:
                yield _emit("ATR scan", "warn", f"0 candidates in {elapsed}s", "≥3 candidates",
                            "No ATR candidates — check data feed and market hours")
        except Exception as e:
            yield _emit("ATR scan", "fail", str(e), "≥3 candidates", f"Scanner error: {e}")

        # ── 7. Know the Trade grading ─────────────────────────────────────
        yield "data: {\"progress\": \"Validating Know the Trade conviction scoring...\"}\n\n"
        try:
            from utils.know_the_trade import analyze as _ktt
            result = _ktt("SPY", provider=provider, side="long")
            grade = result.get("grade")
            score = result.get("score")
            if grade and score is not None:
                yield _emit("Know the Trade", "pass", f"grade={grade} score={score}", "A-D grade",
                            f"Conviction scoring working — SPY: {grade} ({score}pts)")
            else:
                yield _emit("Know the Trade", "fail", f"grade={grade}", "A-D grade",
                            "Conviction scoring returned no grade")
        except Exception as e:
            yield _emit("Know the Trade", "fail", str(e), "A-D grade", f"KTT error: {e}")

        # ── 8. ORB grading distribution ────────────────────────────────────
        yield "data: {\"progress\": \"Checking ORB grade distribution...\"}\n\n"
        try:
            from scanner.orb import scan_symbols as _scan2, ORBConfig as _ORBConfig2
            _cfg2 = _ORBConfig2(min_price=1.0, max_price=100.0, min_rvol=0.5,
                                min_avg20_dollar_vol=0, min_today_dollar_vol=0)
            orb_result = _scan2(
                symbols=[],
                cfg=_cfg2,
                limit=50,
                strategy="orb",
                provider=provider,
                min_grade_enabled=False,
                min_combined_enabled=False,
                use_ml=False, use_sentiment=False, use_catalyst=False,
            )
            raw = orb_result if isinstance(orb_result, list) else (orb_result or {}).get("candidates", []) or (orb_result or {}).get("all", [])
            grades = {}
            for c in (raw or []):
                g = (c.get("confidence_grade") if isinstance(c, dict) else getattr(c, "confidence_grade", None)) or "F"
                grades[g] = grades.get(g, 0) + 1
            total = sum(grades.values())
            if total == 0:
                yield _emit("ORB grading", "info", "0 candidates", "grade spread A-F",
                            "No ORB candidates — expected before/after RTH. Pipeline is operational.")
            elif len(grades) >= 2:
                grade_str = ", ".join(f"{g}:{n}" for g, n in sorted(grades.items()))
                yield _emit("ORB grading", "pass", grade_str, "multiple grade tiers",
                            f"Grade distribution healthy across {total} candidates")
            else:
                grade_str = ", ".join(f"{g}:{n}" for g, n in grades.items())
                yield _emit("ORB grading", "warn", grade_str, "multiple grade tiers",
                            f"Only one grade tier — may be normal pre-market or low volume")
        except Exception as e:
            yield _emit("ORB grading", "warn", str(e), "grade spread", f"ORB grading check failed: {e}")

        # ── 9. Gap orders pipeline ─────────────────────────────────────────
        yield "data: {\"progress\": \"Validating gap orders pipeline...\"}\n\n"
        try:
            from scanner.orb import scan_symbols as _scan3, ORBConfig as _ORBConfig3
            _cfg3 = _ORBConfig3(min_price=1.0, max_price=200.0, min_rvol=0.5,
                                min_avg20_dollar_vol=0, min_today_dollar_vol=0)
            gap_result = _scan3(
                symbols=[],
                cfg=_cfg3,
                limit=20,
                strategy="gap_and_go",
                provider=provider,
                min_grade_enabled=False,
                min_combined_enabled=False,
                use_ml=False, use_sentiment=False, use_catalyst=False,
            )
            raw = gap_result if isinstance(gap_result, list) else (gap_result or {}).get("candidates", [])
            n = len(raw) if raw else 0
            if n >= 1:
                yield _emit("Gap orders", "pass", f"{n} gap candidates", "≥1 candidate",
                            "Gap orders pipeline operational")
            else:
                yield _emit("Gap orders", "info", "0 gap candidates", "≥1 candidate",
                            "No gap candidates — expected before RTH. Pipeline is operational.")
        except Exception as e:
            yield _emit("Gap orders", "warn", str(e), "≥1 candidate", f"Gap scan failed: {e}")

        # ── 10. Spread alert health ────────────────────────────────────────
        yield "data: {\"progress\": \"Checking spread alert system...\"}\n\n"
        try:
            sc2 = _STREAM
            if sc2 is None:
                yield _emit("Spread alerts", "warn", "stream not started", "running",
                            "Start monitor to enable spread explosion detection")
            else:
                state2 = sc2.state()
                n_syms = len(state2.symbols)
                if n_syms >= 1:
                    yield _emit("Spread alerts", "pass", f"monitoring {n_syms} symbols", "≥1 symbol",
                                "Spread explosion detection active")
                else:
                    yield _emit("Spread alerts", "warn", "0 symbols subscribed", "≥1 symbol",
                                "Add symbols to watchlist to enable spread alerts")
        except Exception as e:
            yield _emit("Spread alerts", "warn", str(e), "running", f"Spread check failed: {e}")

        # ── 11. Market session / time-of-day gating ───────────────────────
        yield "data: {\"progress\": \"Checking market session state...\"}\n\n"
        try:
            from datetime import datetime as _dt
            _now_et = _dt.now(_ET)
            _h, _m = _now_et.hour, _now_et.minute
            _mins = _h * 60 + _m
            _PM_OPEN  = 4  * 60       # 4:00 AM
            _RTH_OPEN = 9  * 60 + 30  # 9:30 AM
            _RTH_CLOSE= 16 * 60       # 4:00 PM
            _AH_CLOSE = 20 * 60       # 8:00 PM
            _wd = _now_et.weekday()
            if _wd >= 5:
                yield _emit("Market session", "warn", "weekend — markets closed", "RTH",
                            "Trading not available Sat/Sun — run validation Monday AM before open")
            elif _mins < _PM_OPEN:
                yield _emit("Market session", "warn", "before pre-market (before 4 AM)", "PM or RTH",
                            "Too early — pre-market opens 4 AM ET")
            elif _mins < _RTH_OPEN:
                yield _emit("Market session", "pass", f"pre-market ({_now_et.strftime('%I:%M %p')} ET)", "PM",
                            "Pre-market active — gap setups are valid, volume building")
            elif _mins < _RTH_CLOSE:
                yield _emit("Market session", "pass", f"RTH ({_now_et.strftime('%I:%M %p')} ET)", "RTH",
                            "Regular trading hours — full liquidity, all features active")
            elif _mins < _AH_CLOSE:
                yield _emit("Market session", "warn", f"after-hours ({_now_et.strftime('%I:%M %p')} ET)", "RTH",
                            "After-hours — spreads wide, thin liquidity, gap setups stale")
            else:
                yield _emit("Market session", "warn", "markets closed", "RTH",
                            "Markets closed — data from last session")
        except Exception as e:
            yield _emit("Market session", "warn", str(e), "RTH", f"Session check failed: {e}")

        # ── 12. ML model loaded + producing non-trivial scores ─────────────
        yield "data: {\"progress\": \"Validating ML model state...\"}\n\n"
        try:
            ml_status = _ML_STATE.get("status")
            ml_err    = _ML_STATE.get("error")
            has_pm    = bool(_ML_STATE.get("entry_now_pm_path") and Path(_ML_STATE["entry_now_pm_path"]).exists())
            has_rth   = bool(_ML_STATE.get("entry_now_rth_path") and Path(_ML_STATE["entry_now_rth_path"]).exists())
            if ml_status == "ready" and (has_pm or has_rth):
                models = []
                if has_pm:  models.append("PM")
                if has_rth: models.append("RTH")
                yield _emit("ML models", "pass", f"loaded: {', '.join(models)}", "PM + RTH models",
                            f"ML scoring active — {', '.join(models)} models ready")
            elif ml_status == "ready":
                yield _emit("ML models", "warn", "ready but no model files found", "PM + RTH models",
                            "ML status=ready but model files missing — scores will be absent")
            elif ml_err:
                yield _emit("ML models", "warn", f"error: {ml_err[:80]}", "PM + RTH models",
                            "ML model failed to load — scores will fall back to rule-based")
            else:
                yield _emit("ML models", "warn", f"status={ml_status}", "PM + RTH models",
                            "ML models not ready — scores may be absent. Train a model from Scan settings.")
        except Exception as e:
            yield _emit("ML models", "warn", str(e), "loaded", f"ML check failed: {e}")

        # ── 13. Saved setup R:R ratio sanity ──────────────────────────────
        yield "data: {\"progress\": \"Validating saved setup risk/reward ratios...\"}\n\n"
        try:
            _wl = _RUNTIME_STORE.desk_watchlist_all()
            _rr_ok = _rr_warn = _rr_bad = 0
            _bad_syms = []
            for _e in _wl:
                _en = _safe_float(_e.get("trigger_price"))
                _st = _safe_float(_e.get("stop_price"))
                _tg = _safe_float(_e.get("target_price"))
                _sd = (_e.get("side") or "long").lower()
                if not (_en and _st and _tg):
                    continue
                if _sd == "long":
                    _risk   = _en - _st
                    _reward = _tg - _en
                else:
                    _risk   = _st - _en
                    _reward = _en - _tg
                if _risk <= 0 or _reward <= 0:
                    _rr_bad += 1; _bad_syms.append(_e.get("symbol", "?") + "(invalid)")
                    continue
                _rr = _reward / _risk
                if _rr >= 2.0:   _rr_ok   += 1
                elif _rr >= 1.5: _rr_warn += 1
                else:            _rr_bad  += 1; _bad_syms.append(f"{_e.get('symbol','?')}({_rr:.1f}R)")
            _total_rr = _rr_ok + _rr_warn + _rr_bad
            if _total_rr == 0:
                yield _emit("Setup R:R", "warn", "no levels saved", "≥2:1 on all setups",
                            "No entry/stop/target saved on watchlist — add levels in Analyze Watchlist")
            elif _rr_bad == 0 and _rr_warn == 0:
                yield _emit("Setup R:R", "pass", f"{_rr_ok}/{_total_rr} setups ≥2:1", "≥2:1 on all setups",
                            f"All setups have clean 2:1+ R:R — edge is properly structured")
            elif _rr_bad == 0:
                yield _emit("Setup R:R", "warn", f"{_rr_ok} ok, {_rr_warn} marginal (1.5-2R)", "≥2:1 on all setups",
                            f"Some setups below 2:1 — consider adjusting target")
            else:
                yield _emit("Setup R:R", "fail", f"{_rr_bad} setups <1.5R or invalid: {', '.join(_bad_syms[:4])}", "≥2:1 on all setups",
                            "Poor R:R setups present — do not trade sub-1.5R. Recalculate levels.")
        except Exception as e:
            yield _emit("Setup R:R", "warn", str(e), "≥2:1", f"R:R check failed: {e}")

        # ── 14. Stop distance sanity (not too tight / not too wide) ────────
        yield "data: {\"progress\": \"Checking stop placement sanity...\"}\n\n"
        try:
            _wl2 = _RUNTIME_STORE.desk_watchlist_all()
            _stop_ok = _stop_tight = _stop_wide = 0
            _stop_issues = []
            for _e in _wl2:
                _en2 = _safe_float(_e.get("trigger_price"))
                _st2 = _safe_float(_e.get("stop_price"))
                # Use capped stop for distance check when stop is intentionally wide (natural PM level)
                _st2_cap = _safe_float(_e.get("fresh_stop_cap")) or _st2
                _sd2 = (_e.get("side") or "long").lower()
                if not (_en2 and _st2 and _en2 > 0):
                    continue
                _dist_pct = abs(_en2 - (_st2_cap or _st2)) / _en2 * 100.0
                _sym2 = _e.get("symbol", "?")
                if _dist_pct < 0.5:
                    _stop_tight += 1; _stop_issues.append(f"{_sym2}({_dist_pct:.1f}% too tight)")
                elif _dist_pct > 12.0:
                    _stop_wide  += 1; _stop_issues.append(f"{_sym2}({_dist_pct:.1f}% too wide)")
                else:
                    _stop_ok += 1
            _tot2 = _stop_ok + _stop_tight + _stop_wide
            if _tot2 == 0:
                yield _emit("Stop placement", "warn", "no stops saved", "0.5%–12% from entry",
                            "No stops saved — always define your stop before entering")
            elif not _stop_issues:
                yield _emit("Stop placement", "pass", f"{_stop_ok}/{_tot2} stops in range (0.5–12%)", "0.5–12% of entry",
                            "All stops reasonably placed — not too tight, not too wide")
            else:
                yield _emit("Stop placement", "warn", f"{len(_stop_issues)} issue(s): {', '.join(_stop_issues[:3])}", "0.5–12% of entry",
                            "Some stops outside 12% — verify these are natural PM levels with cap applied for sizing.")
        except Exception as e:
            yield _emit("Stop placement", "warn", str(e), "0.5–12%", f"Stop check failed: {e}")

        # ── 15. Tape / setup direction alignment ─────────────────────────
        yield "data: {\"progress\": \"Checking tape alignment with saved setups...\"}\n\n"
        try:
            _ctx = _CONTEXT_ENGINE.snapshot() if '_CONTEXT_ENGINE' in globals() else {}
            _spy_t = _ctx.get("spy_trend_state", "unknown")
            _qqq_t = _ctx.get("qqq_trend_state", "unknown")
            _wl3 = _RUNTIME_STORE.desk_watchlist_all()
            _against = []
            _aligned  = 0
            for _e3 in _wl3:
                _sd3 = (_e3.get("side") or "long").lower()
                _sym3 = _e3.get("symbol", "?")
                if _sd3 == "long" and _spy_t == "downtrend" and _qqq_t == "downtrend":
                    _against.append(f"{_sym3}(L vs ↓tape)")
                elif _sd3 == "short" and _spy_t == "uptrend" and _qqq_t == "uptrend":
                    _against.append(f"{_sym3}(S vs ↑tape)")
                else:
                    _aligned += 1
            _total3 = _aligned + len(_against)
            if _spy_t == "unknown":
                yield _emit("Tape alignment", "warn", "context engine not ready", "setups match tape",
                            "Market context unavailable — cannot check tape alignment")
            elif not _against:
                yield _emit("Tape alignment", "pass",
                            f"{_aligned}/{_total3 or 1} setups aligned with SPY({_spy_t})/QQQ({_qqq_t})",
                            "setups match tape",
                            "All setups trading with the tape — no counter-trend risk")
            else:
                yield _emit("Tape alignment", "warn",
                            f"{len(_against)} counter-tape: {', '.join(_against[:4])}",
                            "setups match tape",
                            "Counter-tape setups present. SPY/QQQ direction conflicts with setup side — needs strong catalyst to override.")
        except Exception as e:
            yield _emit("Tape alignment", "warn", str(e), "aligned", f"Tape check failed: {e}")

        # ── 16. Halt danger on watchlist symbols ──────────────────────────
        yield "data: {\"progress\": \"Checking halt history for watchlist symbols...\"}\n\n"
        try:
            _sc_h = _STREAM
            _wl4  = _RUNTIME_STORE.desk_watchlist_all()
            _wl_syms4 = {(str(_e4.get("symbol") or "")).upper() for _e4 in _wl4 if _e4.get("symbol")}
            if _sc_h is None or not _wl_syms4:
                yield _emit("Halt danger", "warn", "stream not active or no watchlist", "0 multi-halt names",
                            "Start monitor with watchlist symbols to track halt history")
            else:
                _halts = _sc_h.recent_halt_resume_events(max_age_sec=86400)
                _halt_counts: dict[str, int] = {}
                for _hev in _halts:
                    _hsym = str(_hev.get("symbol", "")).upper()
                    if _hsym in _wl_syms4:
                        _halt_counts[_hsym] = _halt_counts.get(_hsym, 0) + 1
                _danger = {s: c for s, c in _halt_counts.items() if c >= 2}
                _single = {s: c for s, c in _halt_counts.items() if c == 1}
                if _danger:
                    _d_str = ", ".join(f"{s}({c}x)" for s, c in sorted(_danger.items(), key=lambda x: -x[1]))
                    yield _emit("Halt danger", "fail", f"multi-halt today: {_d_str}", "0 multi-halt names",
                                "Symbols with ≥2 halts are erratic — extreme caution or avoid entirely")
                elif _single:
                    _s_str = ", ".join(sorted(_single.keys()))
                    yield _emit("Halt danger", "warn", f"1 halt each: {_s_str}", "0 multi-halt names",
                                "Halted once today — proceed with extreme caution, honor stops strictly")
                else:
                    yield _emit("Halt danger", "pass", f"0 halts on {len(_wl_syms4)} watchlist symbols", "0 multi-halt names",
                                "No halt events on watchlist — clean trading environment")
        except Exception as e:
            yield _emit("Halt danger", "warn", str(e), "0 halts", f"Halt check failed: {e}")

        # ── 17. Catalyst freshness for active setups ──────────────────────
        yield "data: {\"progress\": \"Checking news catalyst freshness on setups...\"}\n\n"
        try:
            _wl5 = _RUNTIME_STORE.desk_watchlist_all()
            _stale_cat = []
            _fresh_cat = []
            _no_cat    = []
            _cutoff_fresh = time.time() - 4 * 3600    # 4 hours
            _cutoff_stale = time.time() - 12 * 3600   # 12 hours
            for _e5 in _wl5:
                _sym5 = str(_e5.get("symbol") or "").upper()
                if not _sym5: continue
                _news5 = _RUNTIME_STORE.recent_news(symbol=_sym5, limit=5)
                _latest = max(
                    (float(n.get("received_at") or 0) for n in _news5),
                    default=0.0
                )
                if _latest >= _cutoff_fresh:
                    _fresh_cat.append(_sym5)
                elif _latest >= _cutoff_stale:
                    _stale_cat.append(_sym5)
                else:
                    _no_cat.append(_sym5)
            if not _wl5:
                yield _emit("Catalyst freshness", "warn", "no watchlist symbols", "≥1 fresh catalyst",
                            "Add symbols to desk watchlist to check catalyst freshness")
            elif _stale_cat or _no_cat:
                _issues5 = _stale_cat[:3] + _no_cat[:3]
                yield _emit("Catalyst freshness", "warn",
                            f"{len(_fresh_cat)} fresh, {len(_stale_cat)} stale, {len(_no_cat)} no news",
                            "all setups with fresh catalyst",
                            f"Stale/no catalyst: {', '.join(_issues5[:5])} — momentum may have faded")
            else:
                yield _emit("Catalyst freshness", "pass",
                            f"{len(_fresh_cat)}/{len(_wl5)} setups have fresh catalyst (<4h)",
                            "all setups with fresh catalyst",
                            "All active setups have a recent catalyst — edge is supported")
        except Exception as e:
            yield _emit("Catalyst freshness", "warn", str(e), "fresh", f"Catalyst check failed: {e}")

        # ── 19. Spread quality on active watchlist ─────────────────────────
        yield "data: {\"progress\": \"Checking spread quality on watchlist...\"}\n\n"
        try:
            _sc3 = _STREAM
            _wl6 = _RUNTIME_STORE.desk_watchlist_all()
            _wl_syms6 = [str(_e6.get("symbol") or "").upper() for _e6 in _wl6 if _e6.get("symbol")][:10]
            if _sc3 is None or not _wl_syms6:
                yield _emit("Spread quality", "warn", "stream inactive or no watchlist", "spread <0.5%",
                            "Start monitor with watchlist to check live spreads")
            else:
                _wide_spreads = []
                _ok_spreads   = []
                _missing      = []
                for _sym6 in _wl_syms6:
                    try:
                        _q6 = _sc3.latest_quote(_sym6)
                        if _q6 is None:
                            _missing.append(_sym6)
                            continue
                        _raw6 = _q6 if isinstance(_q6, dict) else _q6.__dict__
                        _bid6 = float(_raw6.get("bid_price") or _raw6.get("bp") or 0)
                        _ask6 = float(_raw6.get("ask_price") or _raw6.get("ap") or 0)
                        if _bid6 <= 0 or _ask6 <= 0:
                            _missing.append(_sym6)
                            continue
                        _mid6 = (_bid6 + _ask6) / 2.0
                        _spd6 = (_ask6 - _bid6) / _mid6 * 100.0
                        if _spd6 > 0.5:
                            _wide_spreads.append(f"{_sym6}({_spd6:.2f}%)")
                        else:
                            _ok_spreads.append(_sym6)
                    except Exception:
                        _missing.append(_sym6)
                if not _ok_spreads and not _wide_spreads:
                    yield _emit("Spread quality", "warn", f"no quote data for {_wl_syms6}", "spread <0.5%",
                                "No live quotes available — stream may be warming up")
                elif not _wide_spreads:
                    yield _emit("Spread quality", "pass",
                                f"{len(_ok_spreads)} symbols spread <0.5%", "spread <0.5%",
                                "All watchlist spreads tight — good execution quality")
                else:
                    yield _emit("Spread quality", "warn",
                                f"wide: {', '.join(_wide_spreads[:4])}", "spread <0.5%",
                                "Wide spreads detected — execution cost is elevated. Avoid limit orders near mid.")
        except Exception as e:
            yield _emit("Spread quality", "warn", str(e), "<0.5%", f"Spread check failed: {e}")

        # ── 20. Context engine freshness ────────────────────────────────────
        yield "data: {\"progress\": \"Checking market context engine freshness...\"}\n\n"
        try:
            _ctx2 = _CONTEXT_ENGINE.snapshot() if '_CONTEXT_ENGINE' in globals() else {}
            _spy2    = _ctx2.get("spy_trend_state", "unknown")
            _qqq2    = _ctx2.get("qqq_trend_state", "unknown")
            _regime2 = _ctx2.get("volatility_regime", "unknown")
            _spy_mv  = _ctx2.get("spy_move")
            _qqq_mv  = _ctx2.get("qqq_move")
            _gen_at  = _ctx2.get("generated_at")
            _ctx_age = None
            if _gen_at:
                try:
                    from dateutil.parser import parse as _parse_dt
                    _ctx_age = time.time() - _parse_dt(_gen_at).timestamp()
                except Exception:
                    pass
            _mv_str = ""
            if _spy_mv is not None and _qqq_mv is not None:
                _mv_str = f" | SPY {_spy_mv:+.2f}% QQQ {_qqq_mv:+.2f}%"
            if _spy2 == "unknown" and _qqq2 == "unknown":
                yield _emit("Market context", "warn", "not populated", "SPY+QQQ trend known",
                            "Context engine returned unknown states — may need monitor started")
            elif _ctx_age is not None and _ctx_age > 120:
                yield _emit("Market context", "warn", f"stale: {int(_ctx_age)}s old", "fresh <120s",
                            f"Context last updated {int(_ctx_age)}s ago — data may lag live conditions")
            else:
                _age_str = f"{int(_ctx_age)}s old" if _ctx_age is not None else "age unknown"
                yield _emit("Market context", "pass",
                            f"SPY={_spy2} QQQ={_qqq2} regime={_regime2} ({_age_str}){_mv_str}",
                            "SPY+QQQ trend + regime",
                            f"Context engine live — tape ({_spy2}/{_qqq2}), regime ({_regime2})")
        except Exception as e:
            yield _emit("Market context", "warn", str(e), "fresh", f"Context check failed: {e}")

        # ── Final score ────────────────────────────────────────────────────
        # Scoring weights: pass=100, info=95 (known/structural), warn=65 (real issue), fail=0
        # "info" is for checks that are expected non-green (plan limitations, pre-market state).
        n_pass = sum(1 for r in results if r["status"] == "pass")
        n_info = sum(1 for r in results if r["status"] == "info")
        n_warn = sum(1 for r in results if r["status"] == "warn")
        n_fail = sum(1 for r in results if r["status"] == "fail")
        score = round((n_pass * 100 + n_info * 95 + n_warn * 65) / max(len(results), 1))
        grade = "A" if score >= 90 else ("B" if score >= 75 else ("C" if score >= 55 else "D"))
        _warn_note = f", {n_warn} warn" if n_warn else ""
        _info_note = f", {n_info} info" if n_info else ""
        _fail_note = f", {n_fail} fail" if n_fail else ""
        summary = (f"{n_pass} pass{_info_note}{_warn_note}{_fail_note} — "
                   f"System readiness {score}% ({grade})")
        yield f"data: {{\"done\": true, \"score\": {score}, \"grade\": \"{grade}\", "
        yield f"\"n_pass\": {n_pass}, \"n_warn\": {n_warn}, \"n_fail\": {n_fail}, "
        yield f"\"summary\": {_json.dumps(summary)}}}\n\n"

    return Response(stream_with_context(_run()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get('/api/eod_check')
def api_eod_check():
    sym   = (request.args.get('symbol') or '').strip().upper()
    entry = _safe_float(request.args.get('entry'))
    stop  = _safe_float(request.args.get('stop'))
    if not sym:
        return jsonify(ok=False, error='symbol required'), 400

    provider = _ALPACA_PROVIDER
    if provider is None:
        return jsonify(ok=False, error=_PROVIDER_ERROR or 'provider_not_initialized'), 503

    try:
        snapshots = provider.get_snapshots([sym, 'SPY', 'QQQ'])
    except Exception as exc:
        return jsonify(ok=False, error=f'snapshot_failed: {exc}'), 500

    snap     = snapshots.get(sym, {})
    spy_snap = snapshots.get('SPY', {})

    # Reuse cached bars if available
    now_ts = time.time()
    bars_df = None
    cached  = _tape_bars_cache.get(sym)
    if cached and (now_ts - cached[0]) < _TAPE_BARS_TTL:
        bars_df = cached[1]
    else:
        try:
            bars_df = provider.get_bars(
                BarsRequest(symbol=sym, interval='1m', period='1d')
            ).sort_index()
            _tape_bars_cache[sym] = (now_ts, bars_df)
        except Exception:
            bars_df = None

    # ── compute ──────────────────────────────────────────────────────────────
    price      = snap.get('reference_price')
    pb         = snap.get('prev_daily_bar') or {}
    db         = snap.get('daily_bar') or {}
    prev_close = pb.get('close')

    result: dict = {
        'symbol'    : sym,
        'verdict'   : 'CAUTION',
        'score'     : 50,
        'price'     : round(price, 4) if price else None,
        'change_pct': round((price - prev_close) / prev_close * 100, 2) if (price and prev_close and prev_close > 0) else None,
        'position'  : None,
        'signals'   : [],
    }

    if entry and stop and price:
        rps = abs(entry - stop)
        result['position'] = {
            'entry'         : entry,
            'stop'          : stop,
            'risk_per_share': round(rps, 2),
            'current_r'     : round((price - entry) / rps, 2) if rps > 0 else 0,
        }

    score   = 50.0
    signals = []

    def _sig(label, value, bullish):
        signals.append({'label': label, 'value': value, 'bullish': bullish})

    if bars_df is not None and not getattr(bars_df, 'empty', True) and len(bars_df) >= 5:
        last30 = bars_df.tail(30)
        last5  = bars_df.tail(5)

        # 1. VWAP position
        try:
            from scanner.indicators import vwap as _vwap
            vw         = _vwap(bars_df)
            vwap_last  = float(vw.iloc[-1])
            if price:
                delta  = (price - vwap_last) / vwap_last * 100
                above  = price > vwap_last
                score += 15 if above else -15
                _sig('VWAP',
                     f'{"Above" if above else "Below"} ${vwap_last:.2f} ({delta:+.2f}%)',
                     above)
        except Exception:
            pass

        # 2. Close position in daily range
        dh = db.get('high') or (float(bars_df['High'].max()) if 'High' in bars_df.columns else None)
        dl = db.get('low')  or (float(bars_df['Low'].min())  if 'Low'  in bars_df.columns else None)
        if dh and dl and price and dh > dl:
            rng = (price - dl) / (dh - dl)
            if rng >= 0.75:
                score += 15
                _sig('Daily Range', f'Top {rng*100:.0f}% — closing strong', True)
            elif rng <= 0.25:
                score -= 15
                _sig('Daily Range', f'Bottom {rng*100:.0f}% — closing weak', False)
            else:
                _sig('Daily Range', f'Mid-range {rng*100:.0f}%', None)

        # 3. Last 30-min price structure
        if 'Close' in last30.columns and len(last30) >= 5:
            cl30  = last30['Close'].astype(float).tolist()
            up_c  = sum(1 for i in range(1, len(cl30)) if cl30[i] > cl30[i-1])
            dn_c  = len(cl30) - 1 - up_c
            struct = (up_c - dn_c) / max(len(cl30) - 1, 1)
            if struct > 0.2:
                score += 12
                _sig('Last 30m Structure', f'Higher lows — {up_c}/{len(cl30)-1} bars up', True)
            elif struct < -0.2:
                score -= 12
                _sig('Last 30m Structure', f'Lower highs — {dn_c}/{len(cl30)-1} bars down', False)
            else:
                _sig('Last 30m Structure', 'Consolidating — no clear direction', None)

        # 4. Last 30-min volume quality
        if all(c in last30.columns for c in ('Close', 'Open', 'Volume')):
            cl = last30['Close'].astype(float)
            op = last30['Open'].astype(float)
            vl = last30['Volume'].astype(float)
            uv = float(vl[cl > op].sum())
            dv = float(vl[cl <= op].sum())
            tv = uv + dv
            if tv > 0:
                vq = (uv - dv) / tv
                if vq > 0.2:
                    score += 10
                    _sig('Last 30m Volume', f'Buy volume dominant ({vq*100:.0f}% net up)', True)
                elif vq < -0.2:
                    score -= 10
                    _sig('Last 30m Volume', f'Sell volume dominant ({abs(vq)*100:.0f}% net down)', False)
                else:
                    _sig('Last 30m Volume', 'Volume mixed — no conviction', None)

        # 5. Final 5-min momentum
        if 'Close' in last5.columns and len(last5) >= 3:
            cl5 = last5['Close'].astype(float)
            mom = (float(cl5.iloc[-1]) - float(cl5.iloc[0])) / max(float(cl5.iloc[0]), 0.01) * 100
            if mom > 0.1:
                score += 8
                _sig('Final 5m Momentum', f'+{mom:.2f}% — closing on highs', True)
            elif mom < -0.1:
                score -= 8
                _sig('Final 5m Momentum', f'{mom:.2f}% — closing on lows', False)
            else:
                _sig('Final 5m Momentum', 'Flat — indecisive close', None)

    # 6. SPY direction
    spy_price = spy_snap.get('reference_price')
    spy_prev  = (spy_snap.get('prev_daily_bar') or {}).get('close')
    if spy_price and spy_prev and spy_prev > 0:
        spy_pct = (spy_price - spy_prev) / spy_prev * 100
        score  += 5 if spy_pct > 0 else -5
        _sig('SPY', f'{spy_pct:+.2f}% — {"tailwind" if spy_pct > 0 else "headwind"}', spy_pct > 0)

    # 7. RVOL vs yesterday
    today_vol = db.get('volume')
    yest_vol  = pb.get('volume')
    if today_vol and yest_vol and yest_vol > 0:
        try:
            import pytz as _ptz
            _et     = _ptz.timezone('America/New_York')
            _now_et = datetime.now(_et)
            _open_et = _now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            _elapsed = max(1.0, (_now_et - _open_et).total_seconds() / 60.0)
            rvol = (today_vol / min(1.0, _elapsed / 390.0)) / yest_vol
            if rvol > 1.5:
                score += 5
                _sig('RVOL', f'{rvol:.1f}x — elevated, someone is interested', True)
            elif rvol < 0.7:
                score -= 5
                _sig('RVOL', f'{rvol:.1f}x — low interest', False)
            else:
                _sig('RVOL', f'{rvol:.1f}x — normal activity', None)
        except Exception:
            pass

    result['score']   = round(min(max(score, 0), 100))
    result['signals'] = signals
    result['verdict'] = 'HOLD' if result['score'] >= 60 else ('EXIT' if result['score'] <= 40 else 'CAUTION')

    return jsonify(ok=True, **result)


@app.get('/api/close_watch')
def api_close_watch():
    import pytz as _ptz
    sym   = (request.args.get('symbol') or '').strip().upper()
    entry = _safe_float(request.args.get('entry'))
    stop  = _safe_float(request.args.get('stop'))

    if not sym:
        return jsonify(ok=False, error='symbol required'), 400

    provider = _ALPACA_PROVIDER
    if provider is None:
        return jsonify(ok=False, error=_PROVIDER_ERROR or 'provider_not_initialized'), 503

    # ── ET time ──────────────────────────────────────────────────────────────
    _et    = _ptz.timezone('America/New_York')
    now_et = datetime.now(_et)

    # (start_h, start_m, end_h, end_m, name, desc, urgency)
    WINDOWS = [
        ( 9, 30, 10,  0, 'Opening Range',    'High volatility — avoid exits unless stopping out',          'low'),
        (10,  0, 11, 30, 'Primary Trend',    'Best window to exit a winner or a confirmed loser',          'low'),
        (11, 30, 13,  0, 'Lunch Lull',       'Thin and choppy — exits are sloppy',                        'low'),
        (13,  0, 14, 30, 'Afternoon Trend',  'Trend resumes — good read if momentum confirms',             'low'),
        (14, 30, 15,  0, 'Late Day Read',    'Last clear read before MOC pressure — key decision window',  'medium'),
        (15,  0, 15, 30, 'Pre-Close Drift',  'Institutions showing their hand — weak stocks drift lower',  'high'),
        (15, 30, 15, 50, 'MOC Build',        'MOC order accumulation — weak stocks get hit harder',        'critical'),
        (15, 50, 16,  0, 'MOC Execute',      'MOC orders firing — spreads blown out, exit before this',    'critical'),
    ]

    h, m    = now_et.hour, now_et.minute
    now_min = h * 60 + m

    current_win = None
    next_win    = None
    for i, (sh, sm, eh, em, name, desc, urgency) in enumerate(WINDOWS):
        start_m = sh * 60 + sm
        end_m   = eh * 60 + em
        if start_m <= now_min < end_m:
            current_win = {'name': name, 'desc': desc, 'urgency': urgency}
            secs_until  = (end_m - now_min) * 60 - now_et.second
            if i + 1 < len(WINDOWS):
                nsh, nsm = WINDOWS[i+1][0], WINDOWS[i+1][1]
                next_win = {
                    'name'         : WINDOWS[i+1][4],
                    'starts_at'    : f'{nsh:02d}:{nsm:02d}',
                    'seconds_until': max(0, secs_until),
                }
            break

    if current_win is None:
        if now_min < 9 * 60 + 30:
            current_win = {'name': 'Pre-Market',  'desc': 'Market not yet open', 'urgency': 'low'}
        else:
            current_win = {'name': 'After Hours', 'desc': 'Market is closed',    'urgency': 'low'}

    # ── Bars (reuse 60s cache) ───────────────────────────────────────────────
    now_ts  = time.time()
    bars_df = None
    cached  = _tape_bars_cache.get(sym)
    if cached and (now_ts - cached[0]) < _TAPE_BARS_TTL:
        bars_df = cached[1]
    else:
        try:
            bars_df = provider.get_bars(
                BarsRequest(symbol=sym, interval='1m', period='1d')
            ).sort_index()
            _tape_bars_cache[sym] = (now_ts, bars_df)
        except Exception:
            bars_df = None

    # ── Price + VWAP ─────────────────────────────────────────────────────────
    price      = None
    vwap_last  = None
    above_vwap = None
    try:
        snap_map = provider.get_snapshots([sym])
        snap     = snap_map.get(sym, {})
        price    = snap.get('reference_price')
    except Exception:
        pass

    if bars_df is not None and price:
        try:
            from scanner.indicators import vwap as _vwap_fn
            vw        = _vwap_fn(bars_df)
            vwap_last = float(vw.iloc[-1])
            above_vwap = price > vwap_last
        except Exception:
            pass

    # ── 5-bar momentum ───────────────────────────────────────────────────────
    momentum = None
    if bars_df is not None and 'Close' in bars_df.columns and len(bars_df) >= 5:
        cl5   = bars_df['Close'].astype(float).tail(5).tolist()
        up_b  = sum(1 for i in range(1, len(cl5)) if cl5[i] > cl5[i-1])
        dn_b  = len(cl5) - 1 - up_b
        pct   = (cl5[-1] - cl5[0]) / max(cl5[0], 0.01) * 100
        if up_b > dn_b:
            direction, label = 'up',   f'Rising — {up_b}/{len(cl5)-1} bars higher'
        elif dn_b > up_b:
            direction, label = 'down', f'Dropping — {dn_b}/{len(cl5)-1} bars lower'
        else:
            direction, label = 'flat', 'Flat — indecisive'
        momentum = {'direction': direction, 'pct': round(pct, 3), 'label': label}

    # ── Volume rate vs session average ───────────────────────────────────────
    vol_rate = None
    if bars_df is not None and 'Volume' in bars_df.columns and len(bars_df) >= 10:
        try:
            avg_bar_vol = float(bars_df['Volume'].mean())
            last5_vol   = float(bars_df['Volume'].tail(5).mean())
            ratio       = last5_vol / avg_bar_vol if avg_bar_vol > 0 else 1.0
            if   ratio > 1.5: label = f'{ratio:.1f}× avg — heavy activity'
            elif ratio > 1.0: label = f'{ratio:.1f}× avg — above normal'
            elif ratio < 0.5: label = f'{ratio:.1f}× avg — drying up'
            else:             label = f'{ratio:.1f}× avg — normal'
            vol_rate = {'ratio': round(ratio, 2), 'label': label}
        except Exception:
            pass

    # ── Recommendation ────────────────────────────────────────────────────────
    urgency   = current_win.get('urgency', 'low')
    mom_dir   = momentum['direction'] if momentum else 'flat'
    vol_heavy = vol_rate is not None and vol_rate['ratio'] > 1.5
    is_weak   = mom_dir == 'down' or above_vwap is False

    nw_min    = f"{next_win['seconds_until']//60}m" if next_win else '?'

    if urgency == 'critical':
        if is_weak:
            rec = 'CUT NOW'
            reasoning = (
                f"You are in {current_win['name']} — MOC orders are active. "
                f"Price is {'below VWAP and ' if above_vwap is False else ''}"
                f"{'dropping' if mom_dir == 'down' else 'weak'}. "
                "Exit now before the spread blows out."
            )
        else:
            rec = 'WATCH'
            reasoning = (
                "MOC window is active but momentum is holding. "
                "Watch the next 2 bars — if it dips, exit immediately."
            )
    elif urgency == 'high':
        if mom_dir == 'down' and above_vwap is False:
            rec = 'CUT NOW'
            reasoning = (
                f"Pre-close drift with price below VWAP and dropping. "
                f"MOC Build starts in {nw_min}. Exit before institutional end-of-day selling hits."
            )
        elif mom_dir == 'up' and above_vwap:
            rec = 'WAIT'
            reasoning = "Momentum is up and holding above VWAP into the close. Let it run — monitor the 3:30 flip."
        else:
            rec = 'WATCH'
            reasoning = (
                f"Mixed signals in a high-urgency window. "
                f"If the next bar turns red, CUT immediately — MOC Build starts in {nw_min}."
            )
    elif urgency == 'medium':
        if mom_dir == 'down' and above_vwap is False:
            rec = 'CUT NOW'
            reasoning = (
                "Late Day Read window — this is your best exit. "
                "Price is below VWAP and dropping. After 3:00 PM exits get worse."
            )
        elif mom_dir == 'up':
            rec = 'WAIT'
            reasoning = "Momentum turning up in the Late Day Read window. Give it to 3:00 PM to confirm before acting."
        else:
            rec = 'WATCH'
            reasoning = "Late Day Read window — stay alert. This is your last clear signal before close pressure builds."
    else:
        if mom_dir == 'down' and above_vwap is False and vol_heavy:
            rec = 'CUT NOW'
            reasoning = "Strong sell signal — dropping, below VWAP, heavy sell volume. No reason to hold."
        elif mom_dir == 'up' and above_vwap:
            rec = 'WAIT'
            reasoning = "Momentum and structure are bullish. Hold and reassess at the Late Day Read window (2:30 PM)."
        else:
            rec = 'WATCH'
            reasoning = "No clear signal yet. Reassess at 2:30 PM when the Late Day Read window opens."

    return jsonify(
        ok             = True,
        et_time        = now_et.strftime('%H:%M'),
        et_second      = now_et.second,
        window         = current_win,
        next_window    = next_win,
        momentum       = momentum,
        vol_rate       = vol_rate,
        above_vwap     = above_vwap,
        vwap_price     = round(vwap_last, 2) if vwap_last else None,
        price          = round(price, 2) if price else None,
        recommendation = rec,
        reasoning      = reasoning,
    )


# ── Position Recovery Analyzer ───────────────────────────────────────────────

@app.get('/api/recovery_analysis')
def api_recovery_analysis():
    """
    Analyze how long it historically takes a stock to recover from a similar drawdown.
    Returns current price, drawdown %, ATR-based breakeven estimate, and historical comps.
    """
    import math as _math

    sym        = (request.args.get('symbol') or '').strip().upper()
    cost_basis = _safe_float(request.args.get('cost_basis'))
    shares     = _safe_float(request.args.get('shares')) or 0.0

    if not sym:
        return jsonify(ok=False, error='symbol required'), 400
    if cost_basis is None or cost_basis <= 0:
        return jsonify(ok=False, error='cost_basis required and must be > 0'), 400

    provider = _ALPACA_PROVIDER
    if provider is None:
        return jsonify(ok=False, error=_PROVIDER_ERROR or 'provider_not_initialized'), 503

    try:
        df = provider.get_daily_history(sym, period='2y', timeout_s=15)
    except Exception as exc:
        return jsonify(ok=False, error=f'failed to fetch bars: {exc}'), 502

    if df is None or df.empty:
        return jsonify(ok=False, error='no price history available for this symbol'), 404

    # Alpaca returns capitalized column names; normalize to lowercase
    df.columns = [c.lower() for c in df.columns]

    if 'close' not in df.columns:
        return jsonify(ok=False, error='no close prices in history'), 404

    closes = df['close'].dropna().tolist()
    if len(closes) < 10:
        return jsonify(ok=False, error='insufficient history (< 10 days)'), 422

    current_price = float(closes[-1])
    drawdown_pct  = ((current_price - cost_basis) / cost_basis) * 100.0
    gap_to_basis  = cost_basis - current_price   # positive = underwater
    unrealized_pnl = (current_price - cost_basis) * shares if shares else None

    # ATR (14-day) for volatility-based estimate
    highs  = df['high'].dropna().tolist()  if 'high'  in df.columns else closes
    lows   = df['low'].dropna().tolist()   if 'low'   in df.columns else closes
    n = min(len(closes), len(highs), len(lows))
    closes_t = closes[-n:]; highs_t = highs[-n:]; lows_t = lows[-n:]
    trs = []
    for i in range(1, n):
        tr = max(highs_t[i] - lows_t[i],
                 abs(highs_t[i] - closes_t[i - 1]),
                 abs(lows_t[i]  - closes_t[i - 1]))
        trs.append(tr)
    atr14 = float(sum(trs[-14:]) / len(trs[-14:])) if trs else None

    # ATR-based breakeven estimate: how many days at 1 ATR/day upward drift
    atr_days_estimate = None
    if atr14 and atr14 > 0 and gap_to_basis > 0:
        atr_days_estimate = int(_math.ceil(gap_to_basis / atr14))

    # Historical drawdown recovery analysis:
    # Find every window in history where price was >= cost_basis, then dropped
    # below, and measure how many trading days until it recovered.
    recovery_instances: list[dict] = []
    in_drawdown = False
    drawdown_start_i = None
    drawdown_start_price = None

    for i, price in enumerate(closes):
        if not in_drawdown:
            if price >= cost_basis:
                # Price was above (or at) cost basis — potential drawdown start reference
                drawdown_start_i = i
                drawdown_start_price = price
            else:
                # Price just dropped below cost basis
                if drawdown_start_i is not None:
                    in_drawdown = True
        else:
            # In drawdown — waiting for recovery above cost_basis
            if price >= cost_basis:
                days_to_recover = i - drawdown_start_i
                depth_pct = ((min(closes[drawdown_start_i:i + 1]) - drawdown_start_price) / drawdown_start_price) * 100.0
                recovery_instances.append({
                    'days_to_recover': days_to_recover,
                    'depth_pct':       round(depth_pct, 1),
                    'start_price':     round(float(drawdown_start_price), 2),
                    'recovery_price':  round(float(price), 2),
                })
                in_drawdown = False
                drawdown_start_i = i
                drawdown_start_price = price

    # Summary stats from historical recoveries
    if recovery_instances:
        all_days = [r['days_to_recover'] for r in recovery_instances]
        median_days  = int(sorted(all_days)[len(all_days) // 2])
        avg_days     = int(sum(all_days) / len(all_days))
        min_days     = min(all_days)
        max_days     = max(all_days)
        # Comparable = within 20% of current drawdown depth
        similar = [r for r in recovery_instances if abs(r['depth_pct'] - drawdown_pct) <= 20]
        similar_median = int(sorted([r['days_to_recover'] for r in similar])[len(similar) // 2]) if similar else None
    else:
        median_days = avg_days = min_days = max_days = similar_median = None
        similar = []

    # Still in a drawdown at end of history — note it
    currently_underwater = current_price < cost_basis

    return jsonify(
        ok=True,
        symbol=sym,
        current_price=round(current_price, 4),
        cost_basis=round(cost_basis, 4),
        drawdown_pct=round(drawdown_pct, 2),
        gap_to_basis=round(gap_to_basis, 4),
        unrealized_pnl=round(unrealized_pnl, 2) if unrealized_pnl is not None else None,
        shares=shares or None,
        atr14=round(atr14, 4) if atr14 else None,
        atr_days_estimate=atr_days_estimate,
        history_days=len(closes),
        currently_underwater=currently_underwater,
        recovery_summary={
            'instances':     len(recovery_instances),
            'median_days':   median_days,
            'avg_days':      avg_days,
            'min_days':      min_days,
            'max_days':      max_days,
            'similar_count': len(similar),
            'similar_median_days': similar_median,
        },
        recent_recoveries=recovery_instances[-5:],  # last 5 for reference
    )


if __name__ == "__main__":
    import os
    import sys

    try:
        _acquire_single_instance_lock()
    except SingleInstanceRunning as e:
        sys.stderr.write(f"{e}\n")
        sys.stderr.flush()
        raise SystemExit(0)

    host = os.getenv("ORB_HOST", "127.0.0.1")
    port = int(os.getenv("ORB_PORT", "8050"))

    debug = str(os.getenv("FLASK_DEBUG", "0")).lower() in ("1","true","yes","on")

    app.run(host=host, port=port, debug=debug, threaded=True)
