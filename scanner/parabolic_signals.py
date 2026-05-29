"""
parabolic_signals.py — 6 predictive signals for parabolic moves.

All functions are defensively wrapped: they never raise, always return
a dict of scored outputs. Designed to be called from parabolic._one()
(signals 1–5) and scan_parabolic_symbols() post-scan (signal 6).
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


# ──────────────────────────────────────────────────────────────────────────────
# Signal 1 — Float Exhaustion Clock
# ──────────────────────────────────────────────────────────────────────────────

def float_exhaustion(pm_bars: Any, float_shares: float | None) -> dict[str, Any]:
    """
    Measure what fraction of the float has already traded in the PM session.
    A stock that has traded >80% of its float pre-open is approaching blow-off.

    Returns:
        exhaust_pct  – % of float traded (None if float unknown)
        eta_min      – estimated minutes until 150% float traded (blow-off ETA)
        factor       – score multiplier: 1.0 (normal) → 1.5 (extreme blow-off)
    """
    out: dict[str, Any] = {"exhaust_pct": None, "eta_min": None, "factor": 1.0}
    try:
        if pm_bars is None or pm_bars.empty:
            return out
        if not float_shares or float_shares <= 0:
            return out

        pm_volume = float(pm_bars["Volume"].sum())
        exhaust_pct = (pm_volume / float_shares) * 100.0
        out["exhaust_pct"] = round(exhaust_pct, 1)

        # ETA to 150% float traded (blow-off threshold)
        n_bars = len(pm_bars)
        if n_bars >= 3:
            vol_per_min = pm_volume / max(1, n_bars)
            remaining_vol = max(0.0, float_shares * 1.5 - pm_volume)
            eta_min = remaining_vol / max(1.0, vol_per_min)
            out["eta_min"] = round(eta_min, 1) if eta_min < 9999 else None

        # Smooth monotonic decay: high exhaustion = blow-off risk = penalty for longs
        # factor peaks at ~1.10 (50% floated, still building) and decays toward 0.50 at 200%+
        if exhaust_pct < 50:
            out["factor"] = 1.0    # low participation — neutral
        elif exhaust_pct < 80:
            out["factor"] = 1.10   # strong participation — still building
        else:
            # smooth exponential decay above 80%: 80%→0.85, 120%→0.70, 150%→0.58, 200%→0.50
            out["factor"] = round(max(0.50, 0.85 * math.exp(-0.007 * (exhaust_pct - 80))), 3)

    except Exception:
        pass
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Signal 2 — Halt Probability Score
# ──────────────────────────────────────────────────────────────────────────────

def halt_probability(
    symbol: str,
    pm_move_pct: float,
    float_shares: float | None,
    time_et: datetime | None,
    stream_cache: Any | None,
) -> dict[str, Any]:
    """
    Estimate the probability that this symbol will be halted (LULD or regulatory).
    Recently resumed halts are the most powerful signal — stock often re-runs.

    Returns:
        halt_prob        – 0.0–1.0 estimated halt probability
        is_halted        – currently halted per stream cache
        recently_resumed – halted & resumed within last 30 min
        resume_signal    – bool flag for UI badge
        factor           – score multiplier (recently resumed → 1.4x)
    """
    out: dict[str, Any] = {
        "halt_prob": 0.0,
        "is_halted": False,
        "recently_resumed": False,
        "resume_signal": False,
        "factor": 1.0,
    }
    try:
        sym = str(symbol).upper()
        prob = 0.0

        # PM move contribution: >30% meaningful halt risk; >50% high
        if pm_move_pct >= 50:
            prob += 0.35
        elif pm_move_pct >= 30:
            prob += 0.20
        elif pm_move_pct >= 20:
            prob += 0.10

        # Small float → more volatile, higher LULD halt risk
        if float_shares and float_shares < 5_000_000:
            prob += 0.20
        elif float_shares and float_shares < 15_000_000:
            prob += 0.10

        # Time of day: near open → LULD circuit breakers active
        if time_et:
            h = time_et.hour + time_et.minute / 60.0
            if 9.25 <= h <= 10.0:
                prob += 0.10

        # Live halt status from AlpacaStreamCache
        if stream_cache is not None:
            try:
                halt_rec = stream_cache.latest_halt_status(sym)
                if halt_rec:
                    sc = str(halt_rec.get("status_code") or "").upper()
                    if sc in ("T1", "T2", "T3", "H", "HALTED"):
                        out["is_halted"] = True
                        prob += 0.30

                # Recent halt + resume events (last 30 min)
                events = stream_cache.recent_halt_resume_events(max_age_sec=1800)
                sym_events = [e for e in events if str(e.get("symbol") or "").upper() == sym]
                if sym_events:
                    codes = [str(e.get("status_code") or "").upper() for e in sym_events]
                    had_halt = any(c in ("T1", "T2", "T3", "H", "HALTED") for c in codes)
                    had_resume = any(c in ("T", "TRADING") for c in codes)
                    if had_halt and had_resume:
                        out["recently_resumed"] = True
                        out["resume_signal"] = True
                        out["factor"] = 1.4
                        prob += 0.15
            except Exception:
                pass

        out["halt_prob"] = round(min(1.0, prob), 3)

    except Exception:
        pass
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Signal 3 — Options Gamma Exposure
# ──────────────────────────────────────────────────────────────────────────────

def gamma_exposure(
    symbol: str,
    current_price: float,
    provider: Any | None = None,
    store: Any | None = None,
) -> dict[str, Any]:
    """
    Find the nearest gamma wall (max open interest call strike within 10% above price).
    A gamma wall nearby forces market makers to buy as price rises → explosive accelerant.

    Primary: Alpaca OptionHistoricalDataClient (Pro tier, real-time IV/greeks).
    Fallback: yfinance option chain (has reliable OI).

    Returns:
        gamma_wall_strike – strike price of max OI call
        gamma_wall_oi     – open interest at that strike
        gamma_score       – 0.0–1.0 score
        near_gamma_wall   – True if within 2% of strike
        iv_rank           – normalized IV (0–100 scale, >50 = explosive)
        factor            – score multiplier (near wall → 1.2x)
    """
    out: dict[str, Any] = {
        "gamma_wall_strike": None,
        "gamma_wall_oi": None,
        "gamma_score": 0.0,
        "near_gamma_wall": False,
        "iv_rank": None,
        "factor": 1.0,
    }
    try:
        sym = str(symbol).upper()
        # Fix #3: Alpaca option chain does not return real OI — yfinance is authoritative
        # Try yfinance first (real OI), fall back to Alpaca only for IV data
        chain_data = _get_yfinance_option_chain(sym, current_price)
        if chain_data is None:
            chain_data = _get_alpaca_option_chain(sym, current_price)
        if not chain_data:
            return out

        calls = chain_data.get("calls") or []
        if not calls:
            return out

        # Filter to calls within 10% above current price
        nearby = [c for c in calls if current_price <= c["strike"] <= current_price * 1.10]
        if not nearby:
            nearby = [c for c in calls if current_price <= c["strike"] <= current_price * 1.15]
        if not nearby:
            return out

        wall = max(nearby, key=lambda c: c.get("oi") or 0)
        wall_strike = float(wall["strike"])
        wall_oi = int(wall.get("oi") or 0)
        if wall_oi < 10:
            return out

        out["gamma_wall_strike"] = round(wall_strike, 2)
        out["gamma_wall_oi"] = wall_oi

        dist_pct = abs(wall_strike - current_price) / max(0.01, current_price) * 100.0
        if dist_pct <= 2.0:
            out["near_gamma_wall"] = True
            out["factor"] = 1.2

        # log10 scale calibrated: 1k OI→0.60, 10k→0.80, 50k→0.94, 100k→1.0
        oi_score = min(1.0, math.log10(max(1, wall_oi)) / math.log10(100_000))
        # exponential proximity decay: 0%→1.0, 2%→0.67, 5%→0.22, 10%→0.05
        prox_score = math.exp(-0.35 * dist_pct)
        out["gamma_score"] = round(oi_score * 0.6 + prox_score * 0.4, 3)

        iv = chain_data.get("avg_iv")
        if iv is not None and float(iv) > 0:
            out["iv_rank"] = round(min(100.0, float(iv) * 50.0), 1)

    except Exception:
        pass
    return out


_OPTION_CHAIN_CACHE: dict[str, dict | None] = {}
_OPTION_CHAIN_TTL = 300  # seconds


def _get_alpaca_option_chain(symbol: str, current_price: float) -> dict | None:
    """Fetch option chain via Alpaca Pro OptionHistoricalDataClient (5s timeout)."""
    import concurrent.futures as _cf
    import time as _time

    cache_key = f"{symbol}:{int(_time.time() // _OPTION_CHAIN_TTL)}"
    if cache_key in _OPTION_CHAIN_CACHE:
        return _OPTION_CHAIN_CACHE[cache_key]

    def _fetch():
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            from alpaca.data.requests import OptionChainRequest

            api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
            api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
            if not api_key or not api_secret:
                return None

            client = OptionHistoricalDataClient(api_key=api_key, secret_key=api_secret)
            today = date.today()
            req = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date_gte=today,
                expiration_date_lte=today + timedelta(days=30),
                strike_price_gte=current_price * 0.90,
                strike_price_lte=current_price * 1.15,
                type="call",
            )
            chain = client.get_option_chain(req)
            if not chain:
                return None

            calls = []
            ivs = []
            for contract_sym, snap in chain.items():
                try:
                    m = re.search(r'[CP](\d{8})$', str(contract_sym))
                    if not m:
                        continue
                    strike = float(m.group(1)) / 1000.0
                    iv = getattr(snap, "implied_volatility", None)
                    oi = int(getattr(snap, "open_interest", None) or 0)
                    if oi <= 0:
                        continue  # skip — no real OI data available from Alpaca
                    calls.append({"strike": strike, "oi": oi, "iv": float(iv or 0)})
                    if iv and float(iv) > 0:
                        ivs.append(float(iv))
                except Exception:
                    continue

            if not calls:
                return None

            avg_iv = sum(ivs) / len(ivs) if ivs else None
            return {"calls": calls, "avg_iv": avg_iv, "source": "alpaca"}
        except ImportError:
            return None
        except Exception:
            return None

    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_fetch)
            result = fut.result(timeout=5)
    except Exception:
        result = None

    _OPTION_CHAIN_CACHE[cache_key] = result
    return result


def _get_yfinance_option_chain(symbol: str, current_price: float) -> dict | None:
    """Fetch option chain via yfinance (reliable OI, IV per strike), 5s timeout."""
    import concurrent.futures as _cf
    import time as _time

    cache_key = f"yf:{symbol}:{int(_time.time() // _OPTION_CHAIN_TTL)}"
    if cache_key in _OPTION_CHAIN_CACHE:
        return _OPTION_CHAIN_CACHE[cache_key]

    def _fetch():
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            exps = ticker.options
            if not exps:
                return None
            chain = ticker.option_chain(exps[0])
            calls_df = chain.calls
            calls = []
            ivs = []
            for _, row in calls_df.iterrows():
                try:
                    strike = float(row["strike"])
                    oi = int(row.get("openInterest") or 0)
                    iv = float(row.get("impliedVolatility") or 0)
                    calls.append({"strike": strike, "oi": oi, "iv": iv})
                    if iv > 0:
                        ivs.append(iv)
                except Exception:
                    continue
            if not calls:
                return None
            avg_iv = sum(ivs) / len(ivs) if ivs else None
            return {"calls": calls, "avg_iv": avg_iv, "source": "yfinance"}
        except Exception:
            return None

    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_fetch)
            result = fut.result(timeout=5)
    except Exception:
        result = None

    _OPTION_CHAIN_CACHE[cache_key] = result
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Signal 4 — Tape Velocity Fingerprinting
# ──────────────────────────────────────────────────────────────────────────────

def tape_velocity(
    pm_bars: Any,
    hist_avg_trade_size: float,
) -> dict[str, Any]:
    """
    Detect retail FOMO signature: trades/min spike + average trade size shrinking.
    Institutions stop buying; retail floods in with tiny market orders.

    Returns:
        trades_per_min  – recent trades/minute (last 10 bars)
        size_ratio      – recent avg trade size / historical avg (<1 = retail flood)
        velocity_score  – 0.0–1.0 composite score
        is_blow_off     – True if classic retail FOMO fingerprint
        factor          – score multiplier (blow-off → 1.3x)
    """
    out: dict[str, Any] = {
        "trades_per_min": None,
        "size_ratio": None,
        "velocity_score": 0.0,
        "is_blow_off": False,
        "factor": 1.0,
    }
    try:
        if pm_bars is None or pm_bars.empty:
            return out
        if "Trades" not in pm_bars.columns:
            return out

        trades_s = pm_bars["Trades"].fillna(0)
        volume_s = pm_bars["Volume"].fillna(0)
        total_trades = float(trades_s.sum())
        total_volume = float(volume_s.sum())
        if total_trades <= 0:
            return out

        # Recent window: last 10 bars (10 minutes)
        window = min(10, len(pm_bars))
        recent = pm_bars.iloc[-window:]
        recent_trades = float(recent["Trades"].fillna(0).sum())
        recent_volume = float(recent["Volume"].fillna(0).sum())

        trades_per_min = recent_trades / max(1, window)
        out["trades_per_min"] = round(trades_per_min, 1)

        recent_avg_sz = recent_volume / max(1, recent_trades)
        baseline_sz = hist_avg_trade_size if hist_avg_trade_size > 0 else (total_volume / max(1, total_trades))
        size_ratio = recent_avg_sz / max(1.0, baseline_sz)
        out["size_ratio"] = round(size_ratio, 3)

        # Score: high trades/min + small trade size = retail FOMO
        tpm_score = min(1.0, trades_per_min / 20.0)
        size_score = max(0.0, 1.0 - min(1.0, size_ratio))
        out["velocity_score"] = round(tpm_score * 0.6 + size_score * 0.4, 3)

        # Fix #2: blow-off = retail FOMO late-stage = PENALTY for long momentum plays
        if trades_per_min >= 5 and size_ratio < 0.7:
            out["is_blow_off"] = True
            out["factor"] = 0.65   # late-stage retail flood — move is ending
        elif trades_per_min >= 10:
            out["factor"] = 0.85   # very high trade rate alone = chop risk

    except Exception:
        pass
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Signal 5 — Social Velocity Delta
# ──────────────────────────────────────────────────────────────────────────────

def social_velocity(
    symbol: str,
    news_rows: list[dict] | None,
    finnhub_key: str | None = None,
) -> dict[str, Any]:
    """
    Measure social/news acceleration: recent 2h article rate vs prior 2-24h baseline.

    Tier 1: Finnhub social sentiment API (if FINNHUB_API_KEY set).
    Tier 2: Alpaca news timestamp analysis (always available via news_rows).

    Returns:
        mention_velocity – ratio of recent vs baseline (3x = 3× the normal buzz)
        buzz_score       – 0.0–1.0 normalized score
        social_signal    – 'neutral' | 'elevated' | 'buzzing' | 'viral'
        factor           – score multiplier (≥3× velocity → 1.25x)
    """
    out: dict[str, Any] = {
        "mention_velocity": None,
        "buzz_score": 0.0,
        "social_signal": "neutral",
        "factor": 1.0,
    }
    try:
        sym = str(symbol).upper()
        fh_key = finnhub_key or os.getenv("FINNHUB_API_KEY")

        if fh_key:
            fh_result = _finnhub_social_velocity(sym, fh_key)
            if fh_result is not None:
                out.update(fh_result)
                return out

        if not news_rows:
            return out

        now_utc = datetime.now(timezone.utc)
        cutoff_2h  = now_utc - timedelta(hours=2)
        cutoff_24h = now_utc - timedelta(hours=24)
        recent_count = 0
        baseline_count = 0

        for row in news_rows:
            ts_raw = row.get("created_at") or row.get("updated_at") or row.get("timestamp")
            if not ts_raw:
                continue
            try:
                if isinstance(ts_raw, str):
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                elif isinstance(ts_raw, datetime):
                    ts = ts_raw
                else:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ts = ts.astimezone(timezone.utc)
            except Exception:
                continue

            if ts >= cutoff_2h:
                recent_count += 1
            elif ts >= cutoff_24h:
                baseline_count += 1

        if recent_count == 0 and baseline_count == 0:
            return out

        # Hourly rate: recent 2h vs baseline 22h window
        recent_rate   = recent_count / 2.0
        baseline_rate = baseline_count / 22.0 if baseline_count > 0 else 0.5
        velocity = recent_rate / max(0.1, baseline_rate)
        out["mention_velocity"] = round(velocity, 2)
        out["buzz_score"] = round(min(1.0, velocity / 10.0), 3)

        if velocity >= 5:
            out["social_signal"] = "viral"
            out["factor"] = 1.25
        elif velocity >= 3:
            out["social_signal"] = "buzzing"
            out["factor"] = 1.15
        elif velocity >= 1.5:
            out["social_signal"] = "elevated"
            out["factor"] = 1.05

    except Exception:
        pass
    return out


def _finnhub_social_velocity(symbol: str, api_key: str) -> dict | None:
    """Fetch Finnhub social sentiment and compute 7-day vs 30-day velocity."""
    try:
        import urllib.request, json as _json
        url = f"https://finnhub.io/api/v1/stock/social-sentiment?symbol={symbol}&token={api_key}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = _json.loads(resp.read())

        reddit  = data.get("reddit")  or []
        twitter = data.get("twitter") or []

        def _mentions(items):
            return sum(
                int(r.get("positiveMention", 0)) + int(r.get("negativeMention", 0))
                for r in items
            )

        weekly  = _mentions(reddit[-7:])  + _mentions(twitter[-7:])
        monthly = _mentions(reddit)       + _mentions(twitter)
        if monthly == 0:
            return None

        velocity = (weekly / 7.0) / max(0.01, monthly / 30.0)
        buzz = round(min(1.0, velocity / 5.0), 3)
        if velocity >= 5:
            signal, factor = "viral",    1.25
        elif velocity >= 3:
            signal, factor = "buzzing",  1.15
        elif velocity >= 1.5:
            signal, factor = "elevated", 1.05
        else:
            signal, factor = "neutral",  1.0

        return {
            "mention_velocity": round(velocity, 2),
            "buzz_score": buzz,
            "social_signal": signal,
            "factor": factor,
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Signal 6 — Sector Cohort Amplification (post-scan, runs across all candidates)
# ──────────────────────────────────────────────────────────────────────────────

def sector_cohort(
    symbols: list[str],
    store: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Post-scan: if 3+ candidates share the same sector, they amplify each other
    (sector rotation wave). Must run after ThreadPoolExecutor completes.

    Returns a dict mapping each symbol to its cohort result:
        sector_name   – sector string (e.g. 'Technology')
        cohort_count  – how many candidates share this sector
        cohort_score  – 0.0–1.0 score
        factor        – score multiplier (≥3 cohort → 1.3x)
    """
    default: dict[str, Any] = {
        "sector_name": None,
        "cohort_count": 1,
        "cohort_score": 0.0,
        "factor": 1.0,
    }
    result: dict[str, dict[str, Any]] = {}

    if not symbols:
        return result

    sym_sector: dict[str, str | None] = {}
    for sym in symbols:
        sym_sector[sym] = _lookup_sector(sym, store)

    sector_counts = Counter(s for s in sym_sector.values() if s)

    for sym in symbols:
        sector = sym_sector.get(sym)
        if not sector:
            result[sym] = dict(default)
            continue

        count = sector_counts.get(sector, 1)
        if count >= 5:
            score, factor = 0.9, 1.3
        elif count >= 3:
            score, factor = 0.6, 1.3
        elif count >= 2:
            score, factor = 0.3, 1.15
        else:
            score, factor = 0.0, 1.0

        result[sym] = {
            "sector_name": sector,
            "cohort_count": count,
            "cohort_score": score,
            "factor": factor,
        }

    return result


# Module-level sector cache (survives for the process lifetime)
_sector_cache: dict[str, str | None] = {}


def _lookup_sector(symbol: str, store: Any | None) -> str | None:
    """Lookup sector via yfinance with in-process cache."""
    sym = str(symbol).upper()
    if sym in _sector_cache:
        return _sector_cache[sym]
    try:
        import yfinance as yf
        info = yf.Ticker(sym).fast_info
        sector: str | None = getattr(info, "sector", None) or None
        _sector_cache[sym] = sector
        return sector
    except Exception:
        _sector_cache[sym] = None
        return None
