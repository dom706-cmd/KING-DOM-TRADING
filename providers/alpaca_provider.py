from __future__ import annotations

import os
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import pytz

from providers.base import BarsRequest, MarketDataProvider
from providers.symbols import to_provider_symbol

ET = pytz.timezone("America/New_York")


def _now_et() -> datetime:
    return datetime.now(tz=ET)


def _require_any_env(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    raise RuntimeError(f"Missing required env var: one of {', '.join(names)}")


def _period_to_start(period: str) -> datetime:
    """Convert '1d','5d','1mo','3mo','6mo','1y' to an ET datetime start."""
    p = (period or "5d").strip().lower()
    now = _now_et()
    try:
        if p.endswith("d"):
            return now - timedelta(days=int(p[:-1]))
        if p.endswith("mo"):
            return now - timedelta(days=30 * int(p[:-2]))
        if p.endswith("y"):
            return now - timedelta(days=365 * int(p[:-1]))
    except Exception:
        pass
    return now - timedelta(days=7)


def _tf(interval: str):
    """Map '1m','5m','1d' to alpaca-py TimeFrame."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    i = (interval or "1m").strip().lower()
    if i.endswith("m"):
        return TimeFrame(int(i[:-1]), TimeFrameUnit.Minute)
    if i.endswith("h"):
        return TimeFrame(int(i[:-1]), TimeFrameUnit.Hour)
    if i.endswith("d"):
        return TimeFrame(int(i[:-1]), TimeFrameUnit.Day)
    # default 1 minute
    return TimeFrame(1, TimeFrameUnit.Minute)


def _bars_df(barset_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize alpaca-py bars dataframe to expected schema.

    - Index: tz-aware ET DatetimeIndex
    - Columns: Open,High,Low,Close,Volume,Trades,VWAP (when available)
    """
    if barset_df is None or len(barset_df) == 0:
        return pd.DataFrame()

    df = barset_df.copy()

    # MultiIndex (symbol, timestamp) -> slice by symbol
    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.xs(symbol, level=0)
        except Exception:
            # fall back to filtering by column if present
            df = df.reset_index()
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol]
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")

    # Ensure timestamp index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True)

    # Normalize tz
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    df.index = df.index.tz_convert(ET)

    # Normalize columns
    colmap: dict[str, str] = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc == "open":
            colmap[c] = "Open"
        elif lc == "high":
            colmap[c] = "High"
        elif lc == "low":
            colmap[c] = "Low"
        elif lc == "close":
            colmap[c] = "Close"
        elif lc == "volume":
            colmap[c] = "Volume"
        elif lc in ("trade_count", "tradecount", "count"):
            colmap[c] = "Trades"
        elif lc == "vwap":
            colmap[c] = "VWAP"
    if colmap:
        df = df.rename(columns=colmap)

    return df


@dataclass
class AlpacaConfig:
    key: str
    secret: str
    data_feed: str = "iex"  # iex or sip
    paper: bool = False


class AlpacaProvider(MarketDataProvider):
    """Alpaca market data provider using alpaca-py.

    Tenants:
      - No fake data. Errors are real exceptions.
      - All returned market data is directly from Alpaca.
    """
    name = "alpaca"

    def __init__(self, cfg: Optional[AlpacaConfig] = None):
        if cfg is None:
            # Accept common Alpaca key env aliases across launch environments.
            key = _require_any_env("ALPACA_API_KEY", "APCA_API_KEY_ID", "ALPACA_KEY_ID")
            secret = _require_any_env("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY", "ALPACA_API_SECRET")
            feed = (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower()
            if feed not in {"iex", "sip", "otc"}:
                raise RuntimeError("ALPACA_DATA_FEED must be one of iex, sip, otc")
            paper = (os.getenv("ALPACA_PAPER") or "").strip().lower() in ("1", "true", "yes")
            cfg = AlpacaConfig(key=key, secret=secret, data_feed=feed, paper=paper)
        if (cfg.data_feed or "").strip().lower() not in {"iex", "sip", "otc"}:
            raise RuntimeError("ALPACA_DATA_FEED must be one of iex, sip, otc")
        self.cfg = cfg

        from alpaca.data.historical import StockHistoricalDataClient
        self._hist = StockHistoricalDataClient(cfg.key, cfg.secret)

    # ----------------------------
    # Symbol normalization helpers
    # ----------------------------
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Normalize symbols for Alpaca endpoints using the shared provider-symbol rules."""
        s = to_provider_symbol(symbol)
        return "" if s is None else s

    @staticmethod
    def _extract_invalid_symbol(err: Exception) -> str | None:
        msg = str(err)
        m = re.search(r"invalid symbol:\s*([A-Za-z0-9\.\$\-_]+)", msg)
        return m.group(1) if m else None

    @staticmethod
    def _is_transient_error(err: Exception) -> bool:
        msg = str(err).lower()
        # alpaca-py wraps HTTP errors; status code appears in message often
        transient_markers = [
            "429", "too many requests",
            "500", "502", "503", "504",
            "timeout", "timed out",
            "connection reset", "connection aborted", "connection error",
            "temporarily unavailable",
        ]
        return any(m in msg for m in transient_markers)

    def _call_with_retry(self, fn, *, retries: int = 3, base_sleep_s: float = 0.35):
        last = None
        for attempt in range(max(1, int(retries))):
            try:
                return fn()
            except Exception as e:
                last = e
                if attempt >= retries - 1 or not self._is_transient_error(e):
                    raise
                time.sleep(base_sleep_s * (2 ** attempt))
        raise last  # pragma: no cover

    # ----------------------------
    # Core market data
    # ----------------------------
    def get_bars(self, req: BarsRequest, timeout_s: int | None = None) -> pd.DataFrame:
        """Fetch bars for a single symbol over a period ending now (ET clock)."""
        from alpaca.data.requests import StockBarsRequest

        symbol_raw = (req.symbol or "").strip().upper()
        symbol = self.normalize_symbol(symbol_raw)
        if not symbol:
            raise ValueError("BarsRequest.symbol is required")

        interval = (req.interval or "1m").strip().lower()

        # Intraday bars should use the date-range path so session bounds are
        # aligned to ET and include_prepost is honored.
        if not interval.endswith("d"):
            start_et = _period_to_start(req.period)
            end_et = _now_et()
            return self.get_bars_range(
                symbol=symbol_raw,
                interval=interval,
                from_d=start_et.date(),
                to_d=end_et.date(),
                include_prepost=bool(req.include_prepost),
                timeout_s=timeout_s,
            )

        tf = _tf(interval)
        start_et = _period_to_start(req.period)
        end_et = _now_et()

        start_utc = start_et.astimezone(timezone.utc)
        end_utc = end_et.astimezone(timezone.utc)

        def _do():
            r = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start_utc,
                end=end_utc,
                feed=self.cfg.data_feed,
            )
            bars = self._hist.get_stock_bars(r)
            df = getattr(bars, "df", None)
            return _bars_df(df, symbol)

        try:
            return self._call_with_retry(_do, retries=3)
        except Exception as e:
            raise RuntimeError(f"Alpaca get_bars failed for {symbol} {req.interval} {req.period}: {repr(e)}") from e

    def get_bars_range(
        self,
        *,
        symbol: str,
        interval: str,
        from_d: date,
        to_d: date,
        include_prepost: bool = False,
        timeout_s: int | None = None,
    ) -> pd.DataFrame:
        """Fetch bars for a concrete date range (inclusive), aligned to US regular session by default."""
        from alpaca.data.requests import StockBarsRequest

        sym_raw = (symbol or "").strip().upper()
        sym = self.normalize_symbol(sym_raw)
        if not sym:
            raise ValueError("symbol is required")

        tf = _tf(interval)

        # Align to regular session window unless include_prepost requested.
        if include_prepost:
            start_et = ET.localize(datetime.combine(from_d, datetime.min.time()))
            end_et = ET.localize(datetime.combine(to_d, datetime.max.time()))
        else:
            start_et = ET.localize(datetime.combine(from_d, datetime.min.time().replace(hour=9, minute=30, second=0, microsecond=0)))
            end_et = ET.localize(datetime.combine(to_d, datetime.min.time().replace(hour=16, minute=0, second=0, microsecond=0)))

        # If querying "today", don't ask for future
        now_et = _now_et()
        if end_et > now_et:
            end_et = now_et

        if end_et <= start_et:
            return pd.DataFrame()

        start_utc = start_et.astimezone(timezone.utc)
        end_utc = end_et.astimezone(timezone.utc)

        def _do():
            r = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=tf,
                start=start_utc,
                end=end_utc,
                feed=self.cfg.data_feed,
            )
            bars = self._hist.get_stock_bars(r)
            df = getattr(bars, "df", None)
            return _bars_df(df, sym)

        try:
            return self._call_with_retry(_do, retries=3)
        except Exception as e:
            raise RuntimeError(f"Alpaca get_bars_range failed for {sym} {interval} {from_d}..{to_d}: {repr(e)}") from e

    def get_daily_history(self, symbol: str, period: str = "6mo", timeout_s: int | None = None) -> pd.DataFrame:
        return self.get_bars(BarsRequest(symbol=symbol, interval="1d", period=period, include_prepost=False), timeout_s=timeout_s)

    def download_daily_batch(self, symbols: list[str], period: str = "1mo", timeout_s: int | None = None) -> dict[str, pd.DataFrame]:
        """Download daily bars for many symbols.

        Important: Alpaca rejects the *whole request* if any one symbol is invalid (HTTP 400).
        We detect 'invalid symbol: XYZ', drop only that symbol, record a real per-symbol failure,
        and continue with the rest (tenant-compliant: real failure, no fake data).
        """
        from alpaca.data.requests import StockBarsRequest

        raw_syms = [s.strip().upper() for s in (symbols or []) if s and s.strip()]
        if not raw_syms:
            return {}

        # Map normalized -> originals (so callers can use their own symbol strings as keys)
        norm_to_raw: dict[str, list[str]] = {}
        out: dict[str, pd.DataFrame] = {rs: pd.DataFrame() for rs in raw_syms}

        for rs in raw_syms:
            ns = self.normalize_symbol(rs)
            if not ns:
                out[rs] = pd.DataFrame()
                continue
            norm_to_raw.setdefault(ns, []).append(rs)

        start_et = _period_to_start(period)
        end_et = _now_et()
        start_utc = start_et.astimezone(timezone.utc)
        end_utc = end_et.astimezone(timezone.utc)

        batch_size = 200
        norm_syms = list(norm_to_raw.keys())

        for i in range(0, len(norm_syms), batch_size):
            remaining = list(norm_syms[i:i + batch_size])
            removed_invalid: set[str] = set()

            while remaining:
                def _do():
                    r = StockBarsRequest(
                        symbol_or_symbols=remaining,
                        timeframe=_tf("1d"),
                        start=start_utc,
                        end=end_utc,
                        feed=self.cfg.data_feed,
                    )
                    return self._hist.get_stock_bars(r)

                try:
                    bars = self._call_with_retry(_do, retries=3)
                    df = getattr(bars, "df", None)
                    if df is None or len(df) == 0:
                        break

                    if isinstance(df.index, pd.MultiIndex):
                        for ns in remaining:
                            try:
                                sdf = df.xs(ns, level=0)
                            except Exception as _exc:
                                sdf = pd.DataFrame()
                            for rs in norm_to_raw.get(ns, []):
                                out[rs] = _bars_df(sdf, ns)
                    else:
                        # single-symbol fallthrough
                        if len(remaining) == 1:
                            ns = remaining[0]
                            for rs in norm_to_raw.get(ns, []):
                                out[rs] = _bars_df(df, ns)
                    break

                except Exception as e:
                    inv = self._extract_invalid_symbol(e)
                    if inv:
                        inv_norm = self.normalize_symbol(inv)
                        if inv_norm in remaining:
                            remaining.remove(inv_norm)
                            removed_invalid.add(inv_norm)
                            for rs in norm_to_raw.get(inv_norm, []):
                                out[rs] = pd.DataFrame()
                            continue
                    raise RuntimeError(
                        f"Alpaca download_daily_batch failed for {len(remaining)} symbols: {repr(e)}"
                    ) from e

        return out

    # ----------------------------
    # Latest data helpers (Analyze / Entry-Now)
    # ----------------------------
    def get_latest_trade(self, symbol: str) -> Dict[str, Any]:
        from alpaca.data.requests import StockLatestTradeRequest

        sym_raw = (symbol or "").strip().upper()
        sym = self.normalize_symbol(sym_raw)
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=sym, feed=self.cfg.data_feed)
            resp = self._hist.get_stock_latest_trade(req)
            t = resp[sym]
            return {
                "price": float(getattr(t, "price")),
                "size": int(getattr(t, "size")) if getattr(t, "size", None) is not None else None,
                "exchange": str(getattr(t, "exchange")) if getattr(t, "exchange", None) is not None else None,
                "conditions": list(getattr(t, "conditions")) if getattr(t, "conditions", None) is not None else None,
                "timestamp": getattr(t, "timestamp").isoformat() if getattr(t, "timestamp", None) else None,
            }
        except Exception as e:
            raise RuntimeError(f"Alpaca latest trade failed for {sym}: {e}") from e

    def get_latest_trade_price(self, symbol: str) -> float:
        from alpaca.data.requests import StockLatestTradeRequest

        sym_raw = (symbol or "").strip().upper()
        sym = self.normalize_symbol(sym_raw)
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=sym, feed=self.cfg.data_feed)
            resp = self._hist.get_stock_latest_trade(req)
            t = resp[sym]
            return float(getattr(t, "price"))
        except Exception as e:
            raise RuntimeError(f"Alpaca latest trade failed for {sym}: {e}") from e

    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        from alpaca.data.requests import StockLatestQuoteRequest

        sym_raw = (symbol or "").strip().upper()
        sym = self.normalize_symbol(sym_raw)
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=sym, feed=self.cfg.data_feed)
            resp = self._hist.get_stock_latest_quote(req)
            q = resp[sym]
            return {
                "bid_price": float(getattr(q, "bid_price")),
                "ask_price": float(getattr(q, "ask_price")),
                "bid_size": int(getattr(q, "bid_size")),
                "ask_size": int(getattr(q, "ask_size")),
                "timestamp": getattr(q, "timestamp").isoformat() if getattr(q, "timestamp", None) else None,
            }
        except Exception as e:
            raise RuntimeError(f"Alpaca latest quote failed for {sym}: {e}") from e

    def _alpaca_trading_base_url(self) -> str:
        return "https://paper-api.alpaca.markets" if bool(self.cfg.paper) else "https://api.alpaca.markets"

    def get_assets(
        self,
        *,
        status: str = "active",
        asset_class: str = "us_equity",
        exchange: str | None = None,
        timeout_s: float = 20.0,
    ) -> list[dict[str, Any]]:
        url = f"{self._alpaca_trading_base_url()}/v2/assets"
        payload = self._http_get_json(
            url,
            params={
                "status": str(status or "active"),
                "asset_class": str(asset_class or "us_equity"),
                "exchange": str(exchange).upper() if exchange else None,
            },
            timeout_s=timeout_s,
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected Alpaca assets response shape: {type(payload).__name__}")
        return [row for row in payload if isinstance(row, dict)]

    def get_scan_universe(
        self,
        *,
        include_etfs: bool = True,
        status: str = "active",
        asset_class: str = "us_equity",
        timeout_s: float = 20.0,
    ) -> list[str]:
        assets = self.get_assets(status=status, asset_class=asset_class, timeout_s=timeout_s)
        out: list[str] = []
        seen: set[str] = set()
        for asset in assets:
            if str(asset.get("status") or "").strip().lower() != "active":
                continue
            if not bool(asset.get("tradable")):
                continue
            if str(asset.get("exchange") or "").strip().upper() == "OTC":
                continue
            if not include_etfs:
                name = str(asset.get("name") or "").upper()
                if " ETF" in name or name.endswith("ETF"):
                    continue
            sym = self.normalize_symbol(str(asset.get("symbol") or ""))
            if not sym or sym in seen:
                continue
            seen.add(sym)
            out.append(sym)
        if not out:
            raise RuntimeError("Alpaca assets returned 0 active tradable symbols")
        return sorted(out)

    def get_snapshots(
        self,
        symbols: list[str],
        *,
        timeout_s: float = 20.0,
        feed: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        raw_syms = [str(s or "").strip().upper() for s in (symbols or []) if str(s or "").strip()]
        if not raw_syms:
            return {}

        norm_to_raw: dict[str, list[str]] = {}
        for rs in raw_syms:
            ns = self.normalize_symbol(rs)
            if not ns:
                continue
            norm_to_raw.setdefault(ns, []).append(rs)
        if not norm_to_raw:
            return {}

        url = "https://data.alpaca.markets/v2/stocks/snapshots"
        feed_value = str(feed or self.cfg.data_feed or "iex").strip().lower() or "iex"
        out: dict[str, dict[str, Any]] = {rs: {"symbol": rs, "error": "snapshot_unavailable"} for rs in raw_syms}

        def _parse_bar(bar: Any) -> dict[str, Any] | None:
            if not isinstance(bar, dict):
                return None
            return {
                "timestamp": bar.get("t"),
                "open": float(bar["o"]) if bar.get("o") is not None else None,
                "high": float(bar["h"]) if bar.get("h") is not None else None,
                "low": float(bar["l"]) if bar.get("l") is not None else None,
                "close": float(bar["c"]) if bar.get("c") is not None else None,
                "volume": int(bar["v"]) if bar.get("v") is not None else None,
                "trade_count": int(bar["n"]) if bar.get("n") is not None else None,
                "vwap": float(bar["vw"]) if bar.get("vw") is not None else None,
            }

        def _parse_trade(trade: Any) -> dict[str, Any] | None:
            if not isinstance(trade, dict):
                return None
            price = trade.get("p") if trade.get("p") is not None else trade.get("price")
            return {
                "timestamp": trade.get("t") or trade.get("timestamp"),
                "price": float(price) if price is not None else None,
                "size": int(trade["s"]) if trade.get("s") is not None else (int(trade["size"]) if trade.get("size") is not None else None),
                "exchange": trade.get("x") or trade.get("exchange"),
                "conditions": trade.get("c") or trade.get("conditions"),
            }

        def _parse_quote(quote: Any) -> dict[str, Any] | None:
            if not isinstance(quote, dict):
                return None
            bid = quote.get("bp") if quote.get("bp") is not None else quote.get("bid_price")
            ask = quote.get("ap") if quote.get("ap") is not None else quote.get("ask_price")
            return {
                "timestamp": quote.get("t") or quote.get("timestamp"),
                "bid_price": float(bid) if bid is not None else None,
                "ask_price": float(ask) if ask is not None else None,
                "bid_size": int(quote["bs"]) if quote.get("bs") is not None else (int(quote["bid_size"]) if quote.get("bid_size") is not None else None),
                "ask_size": int(quote["as"]) if quote.get("as") is not None else (int(quote["ask_size"]) if quote.get("ask_size") is not None else None),
            }

        def _convert_snapshot(sym: str, snap: dict[str, Any] | None) -> dict[str, Any]:
            if not isinstance(snap, dict):
                return {"symbol": sym, "error": "snapshot_missing"}
            latest_trade = _parse_trade(snap.get("latestTrade"))
            latest_quote = _parse_quote(snap.get("latestQuote"))
            minute_bar = _parse_bar(snap.get("minuteBar"))
            daily_bar = _parse_bar(snap.get("dailyBar"))
            prev_daily_bar = _parse_bar(snap.get("prevDailyBar"))
            reference_price = None
            reference_source = None
            if latest_trade and latest_trade.get("price") is not None:
                reference_price = float(latest_trade["price"])
                reference_source = "latest_trade"
            elif minute_bar and minute_bar.get("close") is not None:
                reference_price = float(minute_bar["close"])
                reference_source = "minute_bar_close"
            elif daily_bar and daily_bar.get("close") is not None:
                reference_price = float(daily_bar["close"])
                reference_source = "daily_bar_close"
            return {
                "symbol": sym,
                "latest_trade": latest_trade,
                "latest_quote": latest_quote,
                "minute_bar": minute_bar,
                "daily_bar": daily_bar,
                "prev_daily_bar": prev_daily_bar,
                "reference_price": reference_price,
                "reference_price_source": reference_source,
                "error": None if reference_price is not None or latest_quote is not None else "snapshot_missing_trade_quote_bar",
            }

        def _fetch_chunk(norm_chunk: list[str]) -> None:
            if not norm_chunk:
                return
            try:
                payload = self._http_get_json(
                    url,
                    params={"symbols": ",".join(norm_chunk), "feed": feed_value},
                    timeout_s=timeout_s,
                )
            except Exception as _exc:
                if len(norm_chunk) == 1:
                    err = f"snapshot_request_failed:{type(_exc).__name__}:{_exc}"
                    for rs in norm_to_raw.get(norm_chunk[0], []):
                        out[rs] = {"symbol": rs, "error": err}
                    return
                mid = max(1, len(norm_chunk) // 2)
                _fetch_chunk(norm_chunk[:mid])
                _fetch_chunk(norm_chunk[mid:])
                return

            snap_map: dict[str, Any] | None = None
            if isinstance(payload, dict):
                if isinstance(payload.get("snapshots"), dict):
                    snap_map = payload.get("snapshots")
                elif norm_chunk and all(isinstance(payload.get(sym), dict) for sym in norm_chunk if sym in payload):
                    # Alpaca multi-snapshot responses may be returned as a direct
                    # symbol->snapshot mapping instead of {"snapshots": {...}}.
                    snap_map = payload
                elif len(norm_chunk) == 1 and any(k in payload for k in ("latestTrade", "latestQuote", "minuteBar", "dailyBar", "prevDailyBar")):
                    # Defensive support for accidental single-snapshot response shape.
                    snap_map = {norm_chunk[0]: payload}
            if not isinstance(snap_map, dict):
                raise RuntimeError(
                    f"Unexpected Alpaca snapshots response shape: {type(payload).__name__} keys={sorted(payload.keys())[:12] if isinstance(payload, dict) else None}"
                )

            for ns in norm_chunk:
                converted = _convert_snapshot(ns, snap_map.get(ns))
                for rs in norm_to_raw.get(ns, []):
                    row = dict(converted)
                    row["symbol"] = rs
                    out[rs] = row

        norm_syms = list(norm_to_raw.keys())
        chunk_size = 200
        for i in range(0, len(norm_syms), chunk_size):
            _fetch_chunk(norm_syms[i:i + chunk_size])

        return out


    # ----------------------------
    # News helpers (Catalyst scoring)
    # ----------------------------
    def _alpaca_data_headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.cfg.key,
            "APCA-API-SECRET-KEY": self.cfg.secret,
            "accept": "application/json",
        }

    def _http_get_json(self, url: str, params: dict[str, Any] | None = None, timeout_s: float = 15.0) -> Any:
        full_url = url
        if params:
            q = {k: v for k, v in params.items() if v is not None and v != ''}
            if q:
                full_url = f"{url}?{urlencode(q, doseq=True)}"
        req = Request(full_url, headers=self._alpaca_data_headers(), method="GET")
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                data = resp.read()
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Alpaca HTTP GET failed for {url}: {e}") from e

    def get_account(self, timeout_s: float = 15.0) -> dict[str, Any]:
        url = f"{self._alpaca_trading_base_url()}/v2/account"
        payload = self._http_get_json(url, timeout_s=timeout_s)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Alpaca account response shape: {type(payload).__name__}")
        return payload

    def get_positions(self, timeout_s: float = 15.0) -> list[dict[str, Any]]:
        url = f"{self._alpaca_trading_base_url()}/v2/positions"
        payload = self._http_get_json(url, timeout_s=timeout_s)
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected Alpaca positions response shape: {type(payload).__name__}")
        out: list[dict[str, Any]] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            out.append({
                "symbol": str(row.get("symbol") or "").upper(),
                "side": str(row.get("side") or "").lower(),
                "qty": float(row.get("qty") or 0.0),
                "avg_entry_price": float(row.get("avg_entry_price") or 0.0),
                "market_value": float(row.get("market_value") or 0.0),
                "cost_basis": float(row.get("cost_basis") or 0.0),
                "unrealized_pl": float(row.get("unrealized_pl") or 0.0),
                "unrealized_plpc": float(row.get("unrealized_plpc") or 0.0),
                "current_price": float(row.get("current_price") or 0.0),
                "lastday_price": float(row.get("lastday_price") or 0.0),
                "change_today": float(row.get("change_today") or 0.0),
                "asset_class": row.get("asset_class"),
            })
        return out

    def get_broker_snapshot(self, timeout_s: float = 15.0) -> dict[str, Any]:
        acct = self.get_account(timeout_s=timeout_s)
        positions = self.get_positions(timeout_s=timeout_s)
        return {
            "broker": "alpaca",
            "account_id": acct.get("id"),
            "account_number": acct.get("account_number"),
            "status": acct.get("status"),
            "currency": acct.get("currency") or "USD",
            "buying_power": float(acct.get("buying_power") or 0.0),
            "cash": float(acct.get("cash") or 0.0),
            "equity": float(acct.get("equity") or 0.0),
            "last_equity": float(acct.get("last_equity") or 0.0),
            "portfolio_value": float(acct.get("portfolio_value") or acct.get("equity") or 0.0),
            "daytrade_count": int(acct.get("daytrade_count") or 0),
            "multiplier": acct.get("multiplier"),
            "pattern_day_trader": bool(acct.get("pattern_day_trader") or False),
            "positions": positions,
            "position_count": len(positions),
        }
    def get_news_batch(
        self,
        symbols: list[str],
        *,
        limit_per_symbol: int = 6,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch recent Alpaca news for multiple symbols and return per-symbol buckets.

        Important:
        - Try batch first for speed.
        - If a chunk 400s, fall back to per-symbol requests so one bad symbol
          does not poison the whole catalyst scan.
        """
        raw_syms = [str(s or "").strip().upper() for s in (symbols or []) if str(s or "").strip()]
        out: dict[str, list[dict[str, Any]]] = {s: [] for s in raw_syms}
        if not raw_syms:
            return out

        raw_to_norm: dict[str, str] = {}
        normalized: list[str] = []
        for s in raw_syms:
            y = to_provider_symbol(s)
            if y is None:
                continue
            raw_to_norm[s] = y
            normalized.append(y)

        if not normalized:
            return out

        url = "https://data.alpaca.markets/v1beta1/news"

        def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
            items = payload.get("news") or payload.get("articles") or []
            if not isinstance(items, list):
                raise RuntimeError("Unexpected Alpaca news response shape (missing news list)")
            return [row for row in items if isinstance(row, dict)]

        def _bucket_rows(rows: list[dict[str, Any]], target_syms: list[str]) -> dict[str, list[dict[str, Any]]]:
            bucketed: dict[str, list[dict[str, Any]]] = {s: [] for s in target_syms}
            target_set = set(target_syms)

            for row in rows:
                row_symbols = row.get("symbols") or []
                if isinstance(row_symbols, str):
                    row_symbols = [row_symbols]
                elif not isinstance(row_symbols, list):
                    row_symbols = []

                row_symbols_norm = {
                    str(x).strip().upper()
                    for x in row_symbols
                    if str(x).strip()
                }

                for sym in (row_symbols_norm & target_set):
                    if len(bucketed[sym]) < int(limit_per_symbol):
                        bucketed[sym].append(row)

            return bucketed

        def _fetch_chunk(chunk: list[str]) -> dict[str, list[dict[str, Any]]]:
            total_limit = max(len(chunk) * max(1, int(limit_per_symbol)) * 3, len(chunk) * 5)
            params = {
                "symbols": ",".join(chunk),
                "limit": min(1000, total_limit),
                "start": start,
                "end": end,
                "sort": "desc",
            }
            payload = self._http_get_json(url, params=params, timeout_s=20.0)
            return _bucket_rows(_extract_items(payload), chunk)

        def _fetch_one(sym: str) -> list[dict[str, Any]]:
            params = {
                "symbols": sym,
                "limit": max(1, int(limit_per_symbol)),
                "start": start,
                "end": end,
                "sort": "desc",
            }
            payload = self._http_get_json(url, params=params, timeout_s=20.0)
            rows = _extract_items(payload)

            matched: list[dict[str, Any]] = []
            for row in rows:
                row_symbols = row.get("symbols") or []
                if isinstance(row_symbols, str):
                    row_symbols = [row_symbols]
                elif not isinstance(row_symbols, list):
                    row_symbols = []

                row_symbols_norm = {
                    str(x).strip().upper()
                    for x in row_symbols
                    if str(x).strip()
                }

                if sym in row_symbols_norm and len(matched) < int(limit_per_symbol):
                    matched.append(row)

            return matched

        batch_size = 25
        for i in range(0, len(normalized), batch_size):
            chunk = normalized[i:i + batch_size]

            try:
                chunk_map = _fetch_chunk(chunk)
                for raw_sym, norm_sym in raw_to_norm.items():
                    if norm_sym in chunk:
                        out[raw_sym] = chunk_map.get(norm_sym, [])
            except Exception:
                # Fall back symbol-by-symbol so a single Alpaca 400 does not
                # wipe out the entire chunk.
                for raw_sym, norm_sym in raw_to_norm.items():
                    if norm_sym in chunk:
                        try:
                            out[raw_sym] = _fetch_one(norm_sym)
                        except Exception:
                            out[raw_sym] = []

        return out

    def get_news(
        self,
        symbol: str,
        *,
        limit: int = 10,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.get_news_batch(
            [symbol],
            limit_per_symbol=limit,
            start=start,
            end=end,
        ).get((symbol or "").strip().upper(), [])