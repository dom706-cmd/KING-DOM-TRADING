from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Set


@dataclass
class StreamState:
    streaming: bool
    connected: bool
    started_at: float | None
    last_event_at: float | None
    last_subscribe_at: float | None
    last_run_started_at: float | None
    last_run_exited_at: float | None
    run_attempt: int
    last_callback_kind: str | None
    last_callback_symbol: str | None
    symbols: list[str]
    error: str | None


class AlpacaStreamCache:
    """Background Alpaca market data stream (trades/quotes/minute bars) with a thread-safe cache.

    Real streaming data or real failure. No fake data.
    """

    _ALLOWED_FEEDS = {"iex", "sip", "otc"}

    def __init__(self, key: str, secret: str, feed: str = "iex"):
        self.key = str(key or "").strip()
        self.secret = str(secret or "").strip()
        self.feed = (feed or "iex").strip().lower() or "iex"
        if self.feed not in self._ALLOWED_FEEDS:
            raise RuntimeError(
                f"ALPACA_DATA_FEED must be one of {sorted(self._ALLOWED_FEEDS)}; got {self.feed!r}"
            )
        if not self.key or not self.secret:
            raise RuntimeError("alpaca stream requires non-empty api key and secret")

        self._lock = threading.RLock()
        self._started = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._fatal_stop = threading.Event()

        self._subscribed: Set[str] = set()
        self._subscribed_actual: Set[str] = set()

        self._last_trade: Dict[str, Dict[str, Any]] = {}
        self._last_quote: Dict[str, Dict[str, Any]] = {}
        self._bars_1m: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=600))

        self._connected = False
        self._started_at: float | None = None
        self._last_event_at: float | None = None
        self._last_subscribe_at: float | None = None
        self._last_run_started_at: float | None = None
        self._last_run_exited_at: float | None = None
        self._run_attempt: int = 0
        self._last_callback_kind: str | None = None
        self._last_callback_symbol: str | None = None
        self._error: str | None = None

        self._stream = None
        self._on_trade_cb = None
        self._on_quote_cb = None
        self._on_bar_cb = None

    # ---------------- public API ----------------

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stop.clear()
            self._fatal_stop.clear()
            self._started = True
            self._started_at = time.time()
            self._thread = threading.Thread(target=self._run, name="alpaca-stream", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._fatal_stop.set()
        try:
            st = getattr(self, "_stream", None)
            if st is not None:
                st.stop()
        except Exception:
            pass
        th = getattr(self, "_thread", None)
        if th is not None and th.is_alive():
            th.join(timeout=5.0)
        with self._lock:
            self._connected = False

    def ensure_symbols(self, symbols: Iterable[str]) -> None:
        syms = [str(s).strip().upper() for s in symbols if s and str(s).strip()]
        if not syms:
            return
        with self._lock:
            new = set(syms) - self._subscribed
            if not new:
                return
            self._subscribed |= new
        self._request_resubscribe()

    def state(self) -> StreamState:
        with self._lock:
            return StreamState(
                streaming=bool(self._thread and self._thread.is_alive()),
                connected=bool(self._connected),
                started_at=self._started_at,
                last_event_at=self._last_event_at,
                last_subscribe_at=self._last_subscribe_at,
                last_run_started_at=self._last_run_started_at,
                last_run_exited_at=self._last_run_exited_at,
                run_attempt=int(self._run_attempt),
                last_callback_kind=self._last_callback_kind,
                last_callback_symbol=self._last_callback_symbol,
                symbols=sorted(self._subscribed),
                error=self._error,
            )

    def latest_trade_price(self, symbol: str) -> float | None:
        with self._lock:
            payload = self._last_trade.get(str(symbol).upper()) or {}
            if isinstance(payload, dict):
                try:
                    return float(payload.get("price")) if payload.get("price") is not None else None
                except Exception:
                    return None
            return None

    def recent_1m_bars(self, symbol: str) -> list[Dict[str, Any]]:
        with self._lock:
            return list(self._bars_1m.get(str(symbol).upper(), []))

    def latest_trade(self, symbol: str):
        sym = str(symbol).strip().upper()
        with self._lock:
            v = self._last_trade.get(sym, {})
            return dict(v) if isinstance(v, dict) else {}

    def latest_quote(self, symbol: str):
        sym = str(symbol).strip().upper()
        with self._lock:
            v = self._last_quote.get(sym, {})
            return dict(v) if isinstance(v, dict) else {}

    # ---------------- internal ----------------

    def _set_error(self, msg: str) -> None:
        with self._lock:
            self._error = msg

    def _mark_event(self, kind: str, symbol: str | None = None) -> None:
        with self._lock:
            self._connected = True
            self._last_event_at = time.time()
            self._last_callback_kind = str(kind)
            self._last_callback_symbol = (str(symbol).upper() if symbol else None)
            if self._error and "connect" in self._error.lower():
                self._error = None

    def _request_resubscribe(self) -> None:
        if self._stop.is_set() or self._fatal_stop.is_set():
            return
        stream = getattr(self, "_stream", None)
        if stream is None:
            return
        try:
            with self._lock:
                pending = sorted(self._subscribed - self._subscribed_actual)
            if not pending:
                return
            if self._on_trade_cb is None or self._on_quote_cb is None or self._on_bar_cb is None:
                return
            stream.subscribe_trades(self._on_trade_cb, *pending)
            stream.subscribe_quotes(self._on_quote_cb, *pending)
            stream.subscribe_bars(self._on_bar_cb, *pending)
            with self._lock:
                self._subscribed_actual |= set(pending)
                self._last_subscribe_at = time.time()
                self._error = None
        except Exception as e:
            with self._lock:
                self._connected = False
            self._set_error(f"Stream resubscribe failed: {type(e).__name__}: {e}")

    def _resolve_feed_arg(self):
        feed_arg = self.feed
        try:
            from alpaca.data.enums import DataFeed  # type: ignore
            if isinstance(self.feed, str):
                f = self.feed.strip().lower() or "iex"
                mapping = {
                    "sip": getattr(DataFeed, "SIP", None),
                    "iex": getattr(DataFeed, "IEX", None),
                    "otc": getattr(DataFeed, "OTC", None),
                }
                feed_arg = mapping.get(f) or self.feed
        except Exception:
            pass
        return feed_arg

    @staticmethod
    def _is_fatal_stream_error(exc: Exception) -> bool:
        msg = f"{type(exc).__name__}: {exc}".lower()
        fatal_markers = (
            "auth failed",
            "forbidden",
            "unauthorized",
            "invalid credentials",
            "insufficient subscription",
            "subscription does not permit",
            "not entitled",
            "not authorized",
            "connection limit exceeded",
            "too many connections",
            "http 429",
        )
        return any(m in msg for m in fatal_markers)

    def _run(self) -> None:
        try:
            from alpaca.data.live import StockDataStream  # type: ignore
        except Exception as e:
            self._set_error(f"alpaca-py streaming import failed: {type(e).__name__}: {e}")
            return

        async def _on_trade(trade):
            sym = getattr(trade, "symbol", None)
            if not sym:
                return
            try:
                payload = {
                    "price": float(getattr(trade, "price")),
                    "size": int(getattr(trade, "size")) if getattr(trade, "size", None) is not None else None,
                    "exchange": str(getattr(trade, "exchange")) if getattr(trade, "exchange", None) is not None else None,
                    "conditions": list(getattr(trade, "conditions")) if getattr(trade, "conditions", None) is not None else None,
                    "timestamp": getattr(trade, "timestamp").isoformat() if getattr(trade, "timestamp", None) else None,
                }
            except Exception:
                return
            self._mark_event("trade", str(sym))
            with self._lock:
                self._last_trade[str(sym).upper()] = payload

        async def _on_quote(q):
            sym = getattr(q, "symbol", None)
            if not sym:
                return
            s0 = str(sym).upper()
            try:
                payload = {
                    "bid_price": float(getattr(q, "bid_price")),
                    "ask_price": float(getattr(q, "ask_price")),
                    "bid_size": int(getattr(q, "bid_size")) if getattr(q, "bid_size", None) is not None else None,
                    "ask_size": int(getattr(q, "ask_size")) if getattr(q, "ask_size", None) is not None else None,
                    "timestamp": getattr(q, "timestamp").isoformat() if getattr(q, "timestamp", None) else None,
                }
            except Exception:
                return
            self._mark_event("quote", s0)
            with self._lock:
                self._last_quote[s0] = payload

        async def _on_bar(bar):
            sym = getattr(bar, "symbol", None)
            if not sym:
                return
            s0 = str(sym).upper()
            try:
                row = {
                    "t": getattr(bar, "timestamp").isoformat() if getattr(bar, "timestamp", None) else None,
                    "o": float(getattr(bar, "open")),
                    "h": float(getattr(bar, "high")),
                    "l": float(getattr(bar, "low")),
                    "c": float(getattr(bar, "close")),
                    "v": int(getattr(bar, "volume")) if getattr(bar, "volume", None) is not None else None,
                    "vw": float(getattr(bar, "vwap")) if getattr(bar, "vwap", None) is not None else None,
                    "n": int(getattr(bar, "trade_count")) if getattr(bar, "trade_count", None) is not None else None,
                }
            except Exception:
                return
            self._mark_event("bar", s0)
            with self._lock:
                self._bars_1m[s0].append(row)

        self._on_trade_cb = _on_trade
        self._on_quote_cb = _on_quote
        self._on_bar_cb = _on_bar

        attempt = 0
        while not self._stop.is_set() and not self._fatal_stop.is_set():
            attempt += 1
            stream = None
            try:
                with self._lock:
                    self._run_attempt = attempt
                    self._last_run_started_at = time.time()
                    self._last_run_exited_at = None

                stream = StockDataStream(self.key, self.secret, feed=self._resolve_feed_arg())
                self._stream = stream

                with self._lock:
                    self._connected = False
                    self._subscribed_actual.clear()

                with self._lock:
                    syms = sorted(self._subscribed)
                if syms:
                    stream.subscribe_trades(_on_trade, *syms)
                    stream.subscribe_quotes(_on_quote, *syms)
                    stream.subscribe_bars(_on_bar, *syms)
                    with self._lock:
                        self._subscribed_actual |= set(syms)
                        self._last_subscribe_at = time.time()
                        self._error = None

                stream.run()

                if self._stop.is_set() or self._fatal_stop.is_set():
                    break

                with self._lock:
                    self._connected = False
                self._set_error("Stream run returned unexpectedly")
            except Exception as e:
                with self._lock:
                    self._connected = False
                    self._subscribed_actual.clear()

                if self._is_fatal_stream_error(e):
                    self._set_error(f"Fatal stream auth/subscription failure: {type(e).__name__}: {e}")
                    self._fatal_stop.set()
                    break

                self._set_error(f"Stream run failed: {type(e).__name__}: {e}")
            finally:
                with self._lock:
                    self._last_run_exited_at = time.time()
                self._stream = None

            if self._stop.is_set() or self._fatal_stop.is_set():
                break

            time.sleep(min(15.0, max(1.0, float(attempt) * 2.0)))

        with self._lock:
            self._connected = False
            self._started = False
