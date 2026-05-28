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

        # Halt / trading-status tracking
        self._last_halt_status: Dict[str, Dict[str, Any]] = {}        # symbol → latest status payload
        self._halt_events: Deque[Dict[str, Any]] = deque(maxlen=500)  # ring buffer of all status events

        # LULD price band tracking
        self._luld_bands: Dict[str, Dict[str, Any]] = {}
        self._luld_events: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._on_luld_cb = None

        # Order imbalance tracking (auction imbalances during halts)
        self._imbalances: Dict[str, Dict[str, Any]] = {}
        self._imbalance_events: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._on_imbalance_cb = None

        # Optional hook called on any halt/LULD event — set by app.py to wake SSE listeners
        self._market_event_hook = None

        # Spread history for pre-halt explosion detection
        # symbol → deque of (received_at_float, spread_pct, mid_price)
        self._quote_spreads: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=120))

        # Level 2 order book: symbol → {"bids": [(price, size), ...], "asks": [(price, size), ...], "ts": float}
        # Bids sorted descending (best bid first). Asks sorted ascending (best ask first).
        self._orderbook: Dict[str, Dict[str, Any]] = {}
        # alpaca-py 0.43.x does not implement subscribe_orderbooks — L2 unavailable via SDK
        self._orderbook_supported: bool = False

        # Time & Sales: symbol → deque of trade dicts (newest last, maxlen=200)
        self._trade_history: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=200))

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
        self._on_status_cb = None
        self._on_orderbook_cb = None

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

    def latest_halt_status(self, symbol: str) -> dict | None:
        """Return the most recent TradingStatus event for this symbol, or None."""
        sym = str(symbol).strip().upper()
        with self._lock:
            v = self._last_halt_status.get(sym)
            return dict(v) if isinstance(v, dict) else None

    def spread_explosion_check(self, symbol: str) -> dict:
        """Detect bid-ask spread blowing out — the pre-halt signature.

        Compares the baseline spread (quotes 30–120s ago) against the recent
        spread (last 10 quotes). A 3x expansion with spread > 1.5% of price
        is the pre-halt fingerprint seen 30–90s before most LULD halts.

        Returns dict with keys:
            is_exploding   – True if pre-halt pattern detected
            current_spread – current spread % of mid price
            baseline_spread– baseline spread % (recent calm)
            ratio          – current / baseline
            mid_price      – latest mid price
            data_points    – number of quotes in window
        """
        sym = str(symbol).upper()
        result: dict = {
            "is_exploding": False, "current_spread": None,
            "baseline_spread": None, "ratio": None,
            "mid_price": None, "data_points": 0,
        }
        try:
            with self._lock:
                history = list(self._quote_spreads.get(sym) or [])
            if len(history) < 15:
                result["data_points"] = len(history)
                return result

            result["data_points"] = len(history)
            now_t = time.time()

            # Baseline: quotes between 30s and 120s ago
            baseline = [s for (t, s, _m) in history if 30 <= (now_t - t) <= 120]
            # Recent: last 10 quotes (within ~10-15s at normal quote rate)
            recent_window = history[-10:]
            recent  = [s for (_t, s, _m) in recent_window]
            mid     = recent_window[-1][2] if recent_window else None

            if not baseline or not recent:
                return result

            # Use median to be robust against outliers
            def _median(lst):
                s = sorted(lst)
                n = len(s)
                return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

            baseline_spread = _median(baseline)
            current_spread  = _median(recent)
            result["baseline_spread"] = round(baseline_spread, 4)
            result["current_spread"]  = round(current_spread, 4)
            result["mid_price"]       = round(mid, 4) if mid else None

            if baseline_spread > 0:
                ratio = current_spread / baseline_spread
                result["ratio"] = round(ratio, 2)
                if ratio >= 3.0 and current_spread >= 1.5:
                    result["is_exploding"] = True
        except Exception:
            pass
        return result

    def latest_orderbook(self, symbol: str, *, levels: int = 10) -> dict:
        """Return current L2 book for symbol with wall detection.

        Returns dict with:
          bids / asks  — top N levels, each {"price": float, "size": int, "is_wall": bool}
          bid_wall     — nearest bid wall price or None
          ask_wall     — nearest ask wall price or None
          pressure     — float 0-1; 1.0 = all depth is on bid side (bullish)
          ts           — epoch float of last book update
          supported    — whether orderbook subscription succeeded
        """
        sym = str(symbol).strip().upper()
        with self._lock:
            book = self._orderbook.get(sym)
            supported = self._orderbook_supported
        if not book:
            return {"bids": [], "asks": [], "bid_wall": None, "ask_wall": None,
                    "pressure": None, "ts": None, "supported": supported}
        return self._annotate_book(book, levels=levels, supported=supported)

    @staticmethod
    def _annotate_book(book: dict, *, levels: int = 10, supported: bool = True) -> dict:
        bids_raw = book.get("bids") or []   # [(price, size), ...] descending
        asks_raw = book.get("asks") or []   # [(price, size), ...] ascending
        ts = book.get("ts")

        def _walls(levels_data: list) -> list:
            sizes = [s for (_, s) in levels_data if s > 0]
            if len(sizes) < 2:
                return [False] * len(levels_data)
            sizes_sorted = sorted(sizes)
            median = sizes_sorted[len(sizes_sorted) // 2]
            threshold = max(median * 3.0, 200)   # must be 3x median AND at least 200 shares
            return [s >= threshold for (_, s) in levels_data]

        top_bids = bids_raw[:levels]
        top_asks = asks_raw[:levels]
        bid_walls = _walls(top_bids)
        ask_walls = _walls(top_asks)

        bid_wall_price = next((p for (p, _), w in zip(top_bids, bid_walls) if w), None)
        ask_wall_price = next((p for (p, _), w in zip(top_asks, ask_walls) if w), None)

        total_bid = sum(s for (_, s) in top_bids)
        total_ask = sum(s for (_, s) in top_asks)
        total = total_bid + total_ask
        pressure = round(total_bid / total, 3) if total > 0 else None

        return {
            "bids": [{"price": round(p, 4), "size": s, "is_wall": w}
                     for (p, s), w in zip(top_bids, bid_walls)],
            "asks": [{"price": round(p, 4), "size": s, "is_wall": w}
                     for (p, s), w in zip(top_asks, ask_walls)],
            "bid_wall": round(bid_wall_price, 4) if bid_wall_price is not None else None,
            "ask_wall": round(ask_wall_price, 4) if ask_wall_price is not None else None,
            "pressure": pressure,
            "ts": ts,
            "supported": supported,
        }

    def recent_trades(self, symbol: str, *, limit: int = 50) -> list[dict]:
        """Return recent trade prints for Time & Sales, newest last."""
        sym = str(symbol).strip().upper()
        with self._lock:
            return list(self._trade_history.get(sym, []))[-limit:]

    def recent_halt_resume_events(self, *, max_age_sec: float = 1800.0) -> list[dict]:
        """Return halt/resume events from the last max_age_sec seconds (default 30 min)."""
        cutoff = time.time() - float(max_age_sec)
        with self._lock:
            return [dict(e) for e in self._halt_events if float(e.get("received_at", 0)) >= cutoff]

    def latest_luld(self, symbol: str) -> dict | None:
        """Return the most recent LULD band for a symbol, or None."""
        sym = str(symbol).strip().upper()
        with self._lock:
            return dict(self._luld_bands[sym]) if sym in self._luld_bands else None

    def recent_luld_events(self, *, max_age_sec: float = 3600.0) -> list[dict]:
        """Return all LULD events from the last max_age_sec seconds (default 1 hr)."""
        cutoff = time.time() - float(max_age_sec)
        with self._lock:
            return [dict(e) for e in self._luld_events if e.get('received_at', 0) >= cutoff]

    def recent_imbalances(self, *, max_age_sec: float = 1800.0) -> list[dict]:
        """Return all imbalance events from the last max_age_sec seconds (default 30 min)."""
        cutoff = time.time() - float(max_age_sec)
        with self._lock:
            return [dict(e) for e in self._imbalance_events if e.get('received_at', 0) >= cutoff]

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
            if self._on_status_cb is not None:
                try:
                    stream.subscribe_trading_statuses(self._on_status_cb, *pending)
                except Exception:
                    pass
            if self._on_luld_cb is not None:
                try:
                    stream._subscribe(self._on_luld_cb, pending, stream._handlers["lulds"])
                except Exception:
                    pass
            if self._on_imbalance_cb is not None:
                try:
                    stream.subscribe_imbalances(self._on_imbalance_cb, *pending)
                except Exception:
                    pass
            if self._on_orderbook_cb is not None and self._orderbook_supported:
                try:
                    stream.subscribe_orderbooks(self._on_orderbook_cb, *pending)
                except Exception as _ob_err:
                    _msg = f"{type(_ob_err).__name__}: {_ob_err}".lower()
                    if any(m in _msg for m in ("not entitled", "insufficient", "not authorized",
                                               "forbidden", "subscription", "attribute")):
                        self._orderbook_supported = False
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
                    "received_at": time.time(),
                }
            except Exception:
                return
            s0 = str(sym).upper()
            self._mark_event("trade", s0)
            with self._lock:
                self._last_trade[s0] = payload
                self._trade_history[s0].append(payload)

        async def _on_orderbook(ob):
            sym = getattr(ob, "symbol", None)
            if not sym:
                return
            s0 = str(sym).upper()
            try:
                raw_bids = getattr(ob, "bids", None) or []
                raw_asks = getattr(ob, "asks", None) or []
                bids = sorted(
                    [(float(getattr(lvl, "price", 0)), int(getattr(lvl, "size", 0))) for lvl in raw_bids],
                    key=lambda x: -x[0],  # best bid first (descending price)
                )
                asks = sorted(
                    [(float(getattr(lvl, "price", 0)), int(getattr(lvl, "size", 0))) for lvl in raw_asks],
                    key=lambda x: x[0],   # best ask first (ascending price)
                )
            except Exception:
                return
            self._mark_event("orderbook", s0)
            with self._lock:
                self._orderbook[s0] = {"bids": bids, "asks": asks, "ts": time.time()}

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
                try:
                    bid = payload["bid_price"]
                    ask = payload["ask_price"]
                    if bid > 0 and ask > bid:
                        mid = (bid + ask) / 2.0
                        spread_pct = (ask - bid) / mid * 100.0
                        self._quote_spreads[s0].append((time.time(), spread_pct, mid))
                except Exception:
                    pass

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

        async def _on_status(status):
            sym = getattr(status, "symbol", None)
            if not sym:
                return
            s0 = str(sym).upper()
            try:
                payload = {
                    "symbol":         s0,
                    "status_code":    str(getattr(status, "status_code",    "") or ""),
                    "status_message": str(getattr(status, "status_message", "") or ""),
                    "reason_code":    str(getattr(status, "reason_code",    "") or ""),
                    "reason_message": str(getattr(status, "reason_message", "") or ""),
                    "tape":           str(getattr(status, "tape",           "") or ""),
                    "timestamp":      getattr(status, "timestamp").isoformat() if getattr(status, "timestamp", None) else None,
                    "received_at":    time.time(),
                }
            except Exception:
                return
            self._mark_event("status", s0)
            with self._lock:
                self._last_halt_status[s0] = payload
                self._halt_events.append(payload)
            hook = self._market_event_hook
            if hook:
                try: hook()
                except Exception: pass

        async def _on_luld(luld) -> None:
            raw = luld if isinstance(luld, dict) else luld.__dict__
            sym = str(raw.get('symbol', '') or raw.get('S', '')).upper().strip()
            if not sym:
                return
            data = {
                'symbol':       sym,
                'limit_up':     float(raw.get('limit_up',   raw.get('u', 0)) or 0),
                'limit_down':   float(raw.get('limit_down', raw.get('d', 0)) or 0),
                'indicator':    str(raw.get('indicator',    raw.get('i', ''))),
                'ts':           str(raw.get('timestamp',    raw.get('t', ''))),
                'received_at':  time.time(),
            }
            with self._lock:
                self._luld_bands[sym] = data
                self._luld_events.append(data)
            self._mark_event('luld', sym)
            hook = self._market_event_hook
            if hook:
                try: hook()
                except Exception: pass

        async def _on_imbalance(imb) -> None:
            raw = imb if isinstance(imb, dict) else imb.__dict__
            sym = str(raw.get('symbol', '') or raw.get('S', '')).upper().strip()
            if not sym:
                return
            data = {
                'symbol':      sym,
                'price':       float(raw.get('price', raw.get('p', 0)) or 0),
                'tape':        str(raw.get('tape',  raw.get('z', ''))),
                'ts':          str(raw.get('timestamp', raw.get('t', ''))),
                'received_at': time.time(),
            }
            with self._lock:
                self._imbalances[sym] = data
                self._imbalance_events.append(data)
            self._mark_event('imbalance', sym)

        self._on_trade_cb = _on_trade
        self._on_quote_cb = _on_quote
        self._on_bar_cb = _on_bar
        self._on_status_cb = _on_status
        self._on_orderbook_cb = _on_orderbook
        self._on_luld_cb = _on_luld
        self._on_imbalance_cb = _on_imbalance

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
                    try:
                        stream.subscribe_trading_statuses(_on_status, *syms)
                    except Exception:
                        pass
                    try:
                        stream._subscribe(_on_luld, syms, stream._handlers["lulds"])
                    except Exception:
                        pass
                    try:
                        stream.subscribe_imbalances(_on_imbalance, *syms)
                    except Exception:
                        pass
                    try:
                        if self._orderbook_supported:
                            stream.subscribe_orderbooks(_on_orderbook, *syms)
                    except Exception as _ob_err:
                        _msg = f"{type(_ob_err).__name__}: {_ob_err}".lower()
                        if any(m in _msg for m in ("not entitled", "insufficient", "not authorized",
                                                    "forbidden", "subscription", "attribute")):
                            with self._lock:
                                self._orderbook_supported = False
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
