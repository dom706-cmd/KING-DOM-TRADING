from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List
import hashlib
import json
import platform
import os
import sys
import subprocess
import threading
import concurrent.futures
import time
import uuid
import signal
import traceback
from flask import Flask, render_template, request, jsonify, redirect, Response, stream_with_context


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

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

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
from core.errors import IntradayDataFailure, TrendContextFailure, EntryNowMLFailure, failure_string
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


def _is_truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _disable_auto_monitor(params: dict[str, Any] | None = None) -> bool:
    default_v = os.getenv("ORB_DISABLE_AUTO_MONITOR_DEFAULT", "1")
    if params is None:
        return _is_truthy(default_v)
    return _is_truthy((params or {}).get("disable_auto_monitor", default_v))

def _candidate_sort_score(c: dict) -> float:
    if not isinstance(c, dict):
        return 0.0
    for k in ("combined_score", "score", "ml_score"):
        try:
            v = c.get(k)
            if v is not None:
                return float(v)
        except Exception:
            pass
    return 0.0

def _get_scan_candidates_for_monitor(job_id: str) -> list[dict]:
    job = _get_job(job_id)
    if not job:
        raise KeyError("unknown_job_id")
    st = job.get("status")
    if st != "done":
        raise RuntimeError(f"scan_job_not_done:{st}")
    result = job.get("result") or {}
    cands = result.get("candidates") or []
    if not isinstance(cands, list):
        raise RuntimeError("scan_job_candidates_invalid")
    out = [c for c in cands if isinstance(c, dict)]
    out.sort(key=_candidate_sort_score, reverse=True)
    return out


def _get_scan_seed_symbols(job_id: str) -> list[str]:
    job = _get_job(job_id)
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
    symbols_order = data.get("symbols_order")
    if isinstance(symbols_order, str) and symbols_order.strip():
        symbols_order = [s.strip().upper() for s in symbols_order.split(",") if s.strip()]
    elif isinstance(symbols_order, list):
        symbols_order = [str(s).strip().upper() for s in symbols_order if str(s).strip()]
    else:
        symbols_order = None

    try:
        if isinstance(symbols, str) and symbols.strip():
            syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            sess = _MONITOR.start_from_symbols(symbols=syms, feed=feed, provider=_ALPACA_PROVIDER, stream_cache=stream_cache)
        elif isinstance(symbols, list) and symbols:
            sess = _MONITOR.start_from_symbols(symbols=symbols, feed=feed, provider=_ALPACA_PROVIDER, stream_cache=stream_cache)
        else:
            if not job_id:
                return jsonify(ok=False, error="missing_job_id_or_symbols"), 400
            cands = _get_scan_candidates_for_monitor(job_id)
            if long_only:
                cands = [c for c in cands if str((c or {}).get("best_side") or "").strip().lower() != "short"]
            sess = _MONITOR.start_from_scan_candidates(
                job_id=job_id,
                candidates=cands,
                top_n=top_n,
                feed=feed,
                provider=_ALPACA_PROVIDER,
                stream_cache=stream_cache,
                symbols_order=symbols_order,
                long_only=long_only,
                source=str(data.get("source") or "scan_job_top_n"),
                promotion_candidates=_get_scan_seed_symbols(job_id),
            )
        return jsonify(ok=True, monitor_id=sess.monitor_id, job_id=sess.job_id, symbols=sorted(list(sess.symbols.keys())), feed_requested=sess.feed_requested, feed_used=sess.feed_used, source=sess.source, started_at=sess.started_at, long_only=long_only)
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
    symbol, side_req, include_prepost, want_sent = _entry_now_request_values()

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


def _entry_now_plan_state(*, side: str, last_price: float | None, stop: float | None, t2: float | None, p_2r_30m: float | None, chase_r: float | None, vwap_delta_pct: float | None, trend_state: str | None) -> tuple[str, list[str], str | None]:
    risk_per_share = (abs(last_price - stop) if (last_price is not None and stop is not None) else None)
    risk_pct = None
    twoR_pct = None
    if risk_per_share is not None and last_price is not None and float(last_price) > 0:
        risk_pct = (float(risk_per_share) / float(last_price)) * 100.0
        if t2 is not None:
            twoR_pct = (abs(float(t2) - float(last_price)) / float(last_price)) * 100.0

    notes: list[str] = []
    state = 'WAIT'

    if last_price is not None and float(last_price) >= 30.0:
        state = 'PASS'
        notes.append('price>=30')
    if risk_pct is not None and risk_pct > 12.0:
        state = 'PASS'
        notes.append(f'risk_pct>{12.0:.0f}%')
    if twoR_pct is not None and twoR_pct > 25.0:
        state = 'PASS'
        notes.append(f'2R_move>{25.0:.0f}%')

    if state != 'PASS':
        if p_2r_30m is None:
            state = 'WAIT'
            notes.append('ml_unavailable')
        else:
            if float(p_2r_30m) < 0.20:
                state = 'PASS'
                notes.append('p<0.20')
            elif float(p_2r_30m) >= 0.24:
                state = 'GO'
                notes.append('p>=0.24')
            else:
                state = 'WAIT'
                notes.append('0.20<=p<0.24')

    if state == 'GO':
        if chase_r is not None and float(chase_r) > 1.25:
            state = 'WAIT'
            notes.append(f'chaseR>{1.25:.2f}')

        if vwap_delta_pct is not None:
            try:
                vdp = abs(float(vwap_delta_pct))
                if vdp > 4.0:
                    state = 'WAIT'
                    notes.append('vwap_ext>4%')
            except Exception:
                pass

        if trend_state is not None:
            ts = str(trend_state)
            if ts in {'chop', '—', 'none'}:
                state = 'WAIT'
                notes.append('trend_chop')
            if side == 'long' and ts in {'down', 'lost_vwap'}:
                state = 'WAIT'
                notes.append('trend_against')
            if side == 'short' and ts in {'up', 'reclaim_vwap'}:
                state = 'WAIT'
                notes.append('trend_against')

    hint_line = None
    try:
        if state != 'GO' and stop is not None and last_price is not None:
            hint_line = _go_hint(side, float(last_price), float(stop), max_risk_pct=12.0)
    except Exception:
        hint_line = None

    return state, notes, hint_line


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

        # ML score (base model probability) + a transparent adjustment for "chase" vs original ORB trigger
        ml_base = None
        ml_bucket = None
        ml_adjusted = None
        ml_reason = None
        ml_error = None
        chase_r = None
        try:
            from ml.orb_model_service import score_orb_symbol

            ml_out = score_orb_symbol(symbol, last_price=float(last_price), provider=provider)
            ml_bucket = ml_out.get('bucket')
            if ml_out.get('score') is not None:
                ml_base = float(ml_out['score'])

            # Transparent adjustment (NOT a new model): penalize heavy extension vs original trigger
            try:
                entry0 = float(cand.entry)
                stop0 = float(cand.stop)
                r0 = abs(entry0 - stop0)
                if r0 > 0:
                    chase_r = (last_price - entry0) / r0 if side == 'long' else (entry0 - last_price) / r0
                adj = ml_base
                if adj is not None and chase_r is not None:
                    import math as _math
                    # Penalize >1R extension from the original trigger
                    adj = adj * _math.exp(-0.35 * max(0.0, float(chase_r) - 1.0))
                    # Small trend alignment nudge
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

        state, notes, hint_line = _entry_now_plan_state(
            side=side,
            last_price=last_price,
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
        exec_style = params.get("exec_style") or params.get("execution") or params.get("execStyle") or "retest"
        range_window_min = _param_int(params, "range_window_min", 60)
        range_band_k = _param_float(params, "range_band_k", 2.0)
        rr_touch_lookback_min = _param_int(params, "rr_touch_lookback_min", 45)
        rr_stop_sigma_mult = _param_float(params, "rr_stop_sigma_mult", 0.75)
        rr_min_risk_per_share = _param_float(params, "rr_min_risk_per_share", 0.01)
        rr_max_risk_per_share = _param_float(params, "rr_max_risk_per_share", 5.00)

        # Optional explicit symbol list (comma/space/newline separated). If present, overrides universe slicing.
        symbols_raw = params.get("symbols") or params.get("tickers") or ""
        symbols_list = []
        if isinstance(symbols_raw, str) and symbols_raw.strip():
            import re as _re
            symbols_list = [s.strip().upper() for s in _re.split(r"[\s,]+", symbols_raw.strip()) if s.strip()]

        # Filters (balanced defaults)
        min_rvol = _param_float(params, "min_rvol", 1.5)
        min_today_dollar_vol = _safe_float(params.get("min_today_dollar_vol", params.get("min_today_dvol", 2_000_000))) or 2_000_000.0
        min_avg20_dollar_vol = _safe_float(params.get("min_avg20_dollar_vol", params.get("min_avg20_dvol", 1_000_000))) or 1_000_000.0
        min_price = _param_float(params, "min_price", 1.0)
        # Default max_price should match ORBConfig() default unless caller overrides
        max_price = _param_float(params, "max_price", 30.0)
        min_or_range_pct = _param_float(params, "min_or_range_pct", 0.6)
        max_or_range_pct = _param_float(params, "max_or_range_pct", 6.0)

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
        cfg = ORBConfig(
            min_price=min_price,
            max_price=max_price,
            min_today_dollar_vol=min_today_dollar_vol,
            min_avg20_dollar_vol=min_avg20_dollar_vol,
            min_rvol=min_rvol,
            min_or_range_pct=min_or_range_pct,
            max_or_range_pct=max_or_range_pct,
            min_risk_per_share=(rr_min_risk_per_share if is_rr_scan else ORBConfig.min_risk_per_share),
            max_risk_per_share=(rr_max_risk_per_share if is_rr_scan else ORBConfig.max_risk_per_share),
        )

        provider = _ALPACA_PROVIDER

        # Determine symbols to scan.
        if symbols_list:
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

        # Per-chunk timeout so one bad/network-stuck chunk cannot hang the whole scan.
        # Tenant-compliant: on timeout we record the real failure and continue.
        chunk_timeout_s = _safe_float(params.get("chunk_timeout_s", os.getenv("ORB_CHUNK_TIMEOUT_S", "30"))) or 30.0
        if chunk_timeout_s <= 0:
            chunk_timeout_s = 30.0

        prefilter_sum: dict = {}
        reject_sum: dict = {}
        data_fail_sum: dict = {}
        all_candidates: list = []
        rejected_candidates_all: list = []
        candidates_total_est = 0
        seed_candidates_total_est = 0
        tradable_now_total_est = 0
        rejected_total_est = 0
        shortlisted_total = 0
        scanned = 0
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
            progress={"scanned": 0, "chunks_done": 0, "chunks_total": chunks_total, "offset": start, "end_offset": end},
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
                            exec_style=exec_style,
                            long_only=_is_truthy(params.get("long_only", "0")),
                            min_grade_enabled=_is_truthy(params.get("min_grade_enabled", "0")),
                            min_grade=str(params.get("min_grade", "B")).strip().upper() or "B",
                            min_combined_enabled=_is_truthy(params.get("min_combined_enabled", "0")),
                            min_combined_score=_param_float(params, "min_combined_score", 0.40),
                            no_chop_enabled=_is_truthy(params.get("no_chop_enabled", "0")),
                            min_vwap_enabled=_is_truthy(params.get("min_vwap_enabled", "0")),
                            min_pct_over_vwap=_param_float(params, "min_pct_over_vwap", 1.0),
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
            all_candidates.extend(out.get("candidates") or [])
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

        top = all_candidates[:limit]

        # Ensure candidates are JSON-serializable (Flask jsonify cannot handle dataclass objects).
        def _cand_to_dict(c):
            try:
                import dataclasses as _dc
                if _dc.is_dataclass(c):
                    return _dc.asdict(c)
            except Exception:
                pass
            if isinstance(c, dict):
                return c
            try:
                return dict(c)
            except Exception:
                pass
            try:
                return vars(c)
            except Exception:
                return {"value": str(c)}

        top = [_normalize_target_rr(_cand_to_dict(c)) for c in top]
        for _row in top:
            if isinstance(_row, dict):
                _row["strategy"] = _row.get("strategy") or params.get("strategy")
                _row["provider"] = _row.get("provider") or getattr(provider, "name", "alpaca")

        result = {
            "provider": getattr(provider, "name", "alpaca"),
            "strategy": params.get("strategy"),
            "count": len(top),
            "candidates_total": candidates_total_est,
            "seed_candidates_total": seed_candidates_total_est,
            "tradable_now_total": tradable_now_total_est,
            "rejected_total": rejected_total_est,
            "candidates": top,
            "rejected_candidates": rejected_candidates_all[:limit],
            "prefilter_counts": prefilter_sum,
            "prefilter_samples": prefilter_samples,
            "thresholds_used": thresholds_used,
            "reject_counts": reject_sum,
            "failure_samples_by_code": failure_samples_by_code,
            "data_failures": data_fail_sum,
            "shortlisted": shortlisted_total,
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
            },
            updated_at=time.time(),
        )
        if not _disable_auto_monitor(params):
            try:
                monitor_top_n = _param_int(params, "monitor_top_n", min(25, max(5, limit)))
                stream_cache = _ensure_stream(start=True, require=True)
                sess = _MONITOR.start_from_scan_candidates(
                    job_id=jid,
                    candidates=top,
                    top_n=monitor_top_n,
                    feed=(os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower() or "iex",
                    provider=provider,
                    stream_cache=stream_cache,
                    long_only=_is_truthy(params.get("long_only", "0")),
                    source="auto_scan_handoff",
                    promotion_candidates=slice_syms,
                )
                result["monitor_id"] = sess.monitor_id
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
    for k, v in request.args.items():
        data.setdefault(k, v)

    strategy = str(data.get("strategy", data.get("scan_strategy", "orb"))).strip().lower() or "orb"
    exec_style = str(data.get("exec_style", data.get("execution", "retest"))).strip().lower() or "retest"
    offset = _param_int(data, "offset", 0)
    max_symbols = _param_int(data, "max_symbols", 400)
    limit = _safe_int(data.get("top_n", data.get("limit", 25)), 25)

    data.setdefault("disable_auto_monitor", os.getenv("ORB_DISABLE_AUTO_MONITOR_DEFAULT", "1"))

    strategy_norm = str(strategy).strip().lower()
    max_price_for_profile = _param_float(data, "max_price", 30.0)
    orb_scalp_under10 = strategy_norm == "orb" and max_price_for_profile <= 10.0

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

    thresholds_used = {
        "strategy": strategy,
        "exec_style": exec_style,
        "offset": offset,
        "max_symbols": max_symbols,
        "limit": limit,
        "min_price": _param_float(data, "min_price", 1.0),
        "max_price": _param_float(data, "max_price", 30.0),
        "min_today_dollar_vol": _param_float(data, "min_today_dollar_vol", 2_000_000.0),
        "min_avg20_dollar_vol": _param_float(data, "min_avg20_dollar_vol", 1_000_000.0),
        "min_rvol": _param_float(data, "min_rvol", 1.5),
        "min_or_range_pct": _param_float(data, "min_or_range_pct", 0.6),
        "max_or_range_pct": _param_float(data, "max_or_range_pct", 6.0),
        "use_ml": _is_truthy(data.get("use_ml", "0")),
        "use_sentiment": _is_truthy(data.get("use_sentiment", "0")),
        "use_catalyst": _is_truthy(data.get("use_catalyst", "0")),
        "long_only": _is_truthy(data.get("long_only", "0")),
        "min_grade_enabled": _is_truthy(data.get("min_grade_enabled", "0")),
        "min_grade": str(data.get("min_grade", "B")).strip().upper() or "B",
        "min_combined_enabled": _is_truthy(data.get("min_combined_enabled", "0")),
        "min_combined_score": _param_float(data, "min_combined_score", 0.40),
        "no_chop_enabled": _is_truthy(data.get("no_chop_enabled", "0")),
        "min_vwap_enabled": _is_truthy(data.get("min_vwap_enabled", "0")),
        "min_pct_over_vwap": _param_float(data, "min_pct_over_vwap", 1.0),
        "orb_scalp_under10": bool(orb_scalp_under10),
        "orb_min_ml_score": _param_float(data, "orb_min_ml_score", 0.0 if orb_scalp_under10 else 0.35),
        "orb_max_chase_r": _param_float(data, "orb_max_chase_r", 0.35),
        "orb_retest_max_chase_r": _param_float(data, "orb_retest_max_chase_r", 0.20),
        "orb_breakout_now_max_chase_r": _param_float(data, "orb_breakout_now_max_chase_r", 0.10),
        "orb_breakout_now_min_ml_score": _param_float(data, "orb_breakout_now_min_ml_score", 0.45),
        "orb_retest_min_ml_score": _param_float(data, "orb_retest_min_ml_score", 0.40),
        "orb_min_minutes_after_open": _param_int(data, "orb_min_minutes_after_open", 10),
        "range_window_min": _param_int(data, "range_window_min", 60),
        "range_band_k": _param_float(data, "range_band_k", 2.0),
        "rr_touch_lookback_min": _param_int(data, "rr_touch_lookback_min", 15),
        "rr_stop_sigma_mult": _param_float(data, "rr_stop_sigma_mult", 0.75),
        "auto_monitor": not _disable_auto_monitor(data),
    }

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
    return jsonify(ok=True, job_id=jid, thresholds_used=thresholds_used)



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


@app.get("/api/broker_snapshot")
def api_broker_snapshot():
    provider = _BROKER_PROVIDER
    if provider is None:
        return jsonify(ok=False, error=(_BROKER_PROVIDER_ERROR or "provider_not_initialized"), broker=_BROKER_PROVIDER_NAME), 503
    if not hasattr(provider, "get_broker_snapshot"):
        return jsonify(ok=False, error="broker_snapshot_not_supported"), 501
    try:
        snap = provider.get_broker_snapshot()
        return jsonify(ok=True, snapshot=snap)
    except Exception as e:
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 500


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
    result = job.get("result") if status == "done" else None
    summary = None
    next_offset = None
    if result:
        try:
            end_offset = int(result.get("end_offset") or 0)
            universe_size = int(result.get("universe_size") or 0)
            next_offset = end_offset
            if universe_size > 0 and next_offset >= universe_size:
                next_offset = 0
        except Exception:
            next_offset = None

        # Return a rich result payload so the UI can display real scan counts even when zero candidates.
        summary = {
            "count": int(result.get("count") or 0),
            "scanned": int(result.get("scanned") or 0),
            "chunks": int(result.get("chunks") or 0),
            "end_offset": int(result.get("end_offset") or 0),
            "universe_size": int(result.get("universe_size") or 0),
            "shortlisted": int(result.get("shortlisted") or 0),
            "candidates_total": int(result.get("candidates_total") or 0),
            "seed_candidates_total": int(result.get("seed_candidates_total") or 0),
            "tradable_now_total": int(result.get("tradable_now_total") or 0),
            "rejected_total": int(result.get("rejected_total") or 0),
            "mode": result.get("mode"),
            "provider": result.get("provider"),
            "prefilter_counts": result.get("prefilter_counts") or {},
            "prefilter_samples": result.get("prefilter_samples") or [],
            "thresholds_used": result.get("thresholds_used") or {},
            "reject_counts": result.get("reject_counts") or {},
            "data_failures": result.get("data_failures") or {},
            "monitor_id": result.get("monitor_id"),
            "monitor_error": result.get("monitor_error"),
            "seed_symbols": result.get("seed_symbols") or [],
            # include candidates so the browser can render without a full page reload
            "candidates": _apply_live_state_to_candidates(result.get("candidates") or []),
            "rejected_candidates": result.get("rejected_candidates") or [],
        }
    if summary is not None and _is_truthy(request.args.get("debug", "")):
        summary["debug"] = result.get("debug") or {}
    return jsonify(ok=True, job_id=jid, status=status, progress=progress, error=err, result=summary, next_offset=next_offset)




@app.get("/api/debug_last_scan")
def api_debug_last_scan():
    req_jid = (request.args.get("job_id") or "").strip()

    def _job_payload(jid: str, job: dict[str, Any]) -> dict[str, Any]:
        result = dict((job or {}).get("result") or (job or {}).get("partial_result") or {})
        params = dict((job or {}).get("params") or {})
        candidates = result.get("candidates") or result.get("results") or []
        enriched_candidates = _apply_live_state_to_candidates(candidates)
        if "candidates" in result or candidates:
            result["candidates"] = enriched_candidates

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
    min_rvol = _param_float(request.args, "min_rvol", 1.5)
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

@app.get("/api/monitor_status")
def api_monitor_status():
    monitor_id = str(request.args.get("monitor_id") or "").strip()
    if not monitor_id:
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


if __name__ == "__main__":
    import os

    host = os.getenv("ORB_HOST", "127.0.0.1")
    port = int(os.getenv("ORB_PORT", "8050"))

    debug = str(os.getenv("FLASK_DEBUG", "0")).lower() in ("1","true","yes","on")

    app.run(host=host, port=port, debug=debug, threaded=True)
