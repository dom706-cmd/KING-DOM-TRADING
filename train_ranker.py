from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from providers.alpaca_provider import AlpacaProvider
from providers.base import BarsRequest



def _with_retries(fn, *, attempts: int, failures: List[dict], symbol: str, stage: str):
    """Run fn() up to attempts times. Record real failures; never fabricate data."""
    last = None
    for i in range(1, max(1, int(attempts)) + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            failures.append({"symbol": symbol, "stage": stage, "attempt": i, "error": f"{type(e).__name__}: {e}"})
    raise last  # type: ignore[misc]


def _is_common_equity(sym: str) -> bool:
    s = sym.strip().upper()
    if "$" in s:
        return False
    # Warrants / rights often show up as .W or .R
    if re.search(r"\.(W|R)$", s):
        return False
    return True


def _load_symbols(args) -> List[str]:
    if args.universe:
        uni = args.universe.strip().lower()
        if uni == "file":
            p = Path("symbols.txt")
            if not p.exists():
                raise SystemExit("symbols.txt not found for universe=file")
            syms = [
                ln.strip().upper()
                for ln in p.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
        else:
            from universe.nasdaq_symbols import get_nasdaq_symbols

            syms = get_nasdaq_symbols(
                include_non_common=bool(args.include_non_common),
                cache_dir=str(getattr(args, "cache_dir", "cache")),
                ttl_seconds=int(getattr(args, "ttl_seconds", 86400)),
            )

        if not args.include_non_common:
            syms = [s for s in syms if _is_common_equity(s)]

        start = int(args.offset)
        end = start + int(args.max_symbols)
        return syms[start:end]

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if not args.include_non_common:
            syms = [s for s in syms if _is_common_equity(s)]
        return syms

    if args.symbols_file:
        p = Path(args.symbols_file)
        df = pd.read_csv(p)
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        syms = [str(x).strip().upper() for x in df[col].dropna().tolist()]
        if not args.include_non_common:
            syms = [s for s in syms if _is_common_equity(s)]
        return syms

    raise SystemExit("Provide --symbols or --symbols_file or --universe")


_ET = ZoneInfo("America/New_York")

def _regular_session_day(df_5m: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Return 5m bars for the regular US equity session (09:30-16:00 ET) for a given day.

    We convert the provider's UTC index to America/New_York to slice the correct session,
    then convert back to UTC so downstream code remains consistent.
    """
    if df_5m is None or df_5m.empty:
        return pd.DataFrame()

    df = df_5m
    if getattr(df.index, "tz", None) is None:
        # Defensive: provider should return tz-aware UTC, but keep code robust.
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    df_et = df.tz_convert(_ET)

    # Interpret `day` as a calendar date in ET.
    d = pd.Timestamp(day)
    if d.tzinfo is None:
        d_et = d.tz_localize(_ET)
    else:
        d_et = d.tz_convert(_ET)

    start = d_et.normalize() + pd.Timedelta(hours=9, minutes=30)
    end = d_et.normalize() + pd.Timedelta(hours=16, minutes=0)
    out = df_et[(df_et.index >= start) & (df_et.index < end)]
    return out.tz_convert("UTC")


def _first_bar_or(df_5m: pd.DataFrame, day: pd.Timestamp) -> Tuple[float, float]:
    day_df = _regular_session_day(df_5m, day)
    if day_df is None or day_df.empty:
        raise RuntimeError("No regular-session bars for day")
    first = day_df.iloc[0]
    return float(first["High"]), float(first["Low"])



def _label_day(df_5m: pd.DataFrame, day: pd.Timestamp, *, label_minutes: int) -> int:
    """Label: 1 if +2R target reached before stop after breakout above OR high.

    The evaluation window is capped to `label_minutes` after the opening range bar.
    """
    day_df = _regular_session_day(df_5m, day)
    if day_df is None or day_df.empty or len(day_df) < 3:
        raise RuntimeError("Not enough intraday bars for day")

    or_high, or_low = _first_bar_or(df_5m, day)
    entry = or_high
    stop = or_low
    risk = entry - stop
    if risk <= 0:
        raise RuntimeError("Invalid OR risk")

    target = entry + 2.0 * risk

    in_pos = False
    max_bars = max(1, int(label_minutes) // 5)
    window = day_df.iloc[1 : 1 + max_bars]
    for _, row in window.iterrows():
        hi = float(row["High"])
        lo = float(row["Low"])

        if not in_pos:
            if hi >= entry:
                in_pos = True
            else:
                continue

        # Once in, see which hits first within the bar.
        if lo <= stop:
            return 0
        if hi >= target:
            return 1

    return 0


def _features_day(daily: pd.DataFrame, df_5m: pd.DataFrame, day: pd.Timestamp) -> Dict[str, float]:
    # Use daily features up to previous close (avoid lookahead)
    d = daily.copy().sort_index()
    d = d[d.index.normalize() < day.normalize()]
    if len(d) < 60:
        raise RuntimeError("Not enough daily history")

    close = d["Close"].astype(float)
    vol = d["Volume"].astype(float)

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    trend_20_50 = 1.0 if sma20 > sma50 else 0.0

    ret = close.pct_change()
    vol20 = float(ret.rolling(20).std().iloc[-1] or 0.0)

    avg20_vol = float(vol.rolling(20).mean().iloc[-1] or 0.0)
    last_close = float(close.iloc[-1])
    avg20_dollar_vol = float(avg20_vol * last_close)

    # Momentum
    if len(close) < 21:
        raise RuntimeError("Not enough daily history")
    mom5 = float(close.iloc[-1] / close.iloc[-6] - 1.0)
    mom20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)

    # Intraday-derived features (no lookahead beyond the OR window)
    day_df = _regular_session_day(df_5m, day)
    if day_df is None or day_df.empty or len(day_df) < 3:
        raise RuntimeError("Not enough intraday bars for day")

    first = day_df.iloc[0]
    or_high = float(first["High"])
    or_low = float(first["Low"])
    or_mid = (or_high + or_low) / 2.0
    or_range_pct = (or_high - or_low) / max(1e-9, or_mid) * 100.0

    # Early 15m dollar volume (first 3x 5m bars)
    early = day_df.iloc[:3]
    early_dv = float((early["Close"].astype(float) * early["Volume"].astype(float)).sum())

    # Gap %: today's first-bar open vs previous close
    prev_close = float(close.iloc[-1])
    today_open = float(first["Open"])
    gap_pct = (today_open / max(1e-9, prev_close) - 1.0) * 100.0

    # RVOL proxy: early 15m dollar vol vs expected (avg_daily / 26 trading periods)
    expected_15m_dv = float(avg20_dollar_vol) / 26.0
    rvol_proxy = float(early_dv / max(1.0, expected_15m_dv))

    return {
        "trend_20_50": float(trend_20_50),
        "vol20": float(vol20),
        "avg20_dollar_vol": float(avg20_dollar_vol),
        "mom5": float(mom5),
        "mom20": float(mom20),
        "gap_pct": float(gap_pct),
        "or_range_pct": float(or_range_pct),
        "early15_dollar_vol": float(early_dv),
        "rvol_proxy": float(rvol_proxy),
    }


def _fetch_intraday_5m_multi_year(
    provider: AlpacaProvider,
    sym: str,
    *,
    years: float,
    chunk_days: int,
    timeout_s: int,
    retries: int,
    failures: List[dict],
) -> pd.DataFrame:
    """Fetch multi-year 5m bars in small calendar chunks.

    Requesting huge multi-year intraday ranges in one call is fragile.
    Chunking keeps each HTTP request bounded while still using real provider data.

    This function NEVER fabricates data: provider failures are recorded via failures
    and we continue to the next chunk/symbol as appropriate.
    """
    end = date.today()
    start = end - timedelta(days=int(float(years) * 366) + 10)

    dfs: list[pd.DataFrame] = []
    cur = start
    cd = max(3, int(chunk_days))

    # Bound retries to something sane (still real failures).
    attempts = max(1, int(retries))

    while cur < end:
        nxt = min(end, cur + timedelta(days=cd))

        def _do_5m():
            return provider.get_bars_range(
                symbol=sym,
                interval="5m",
                from_d=cur,
                to_d=nxt,
                include_prepost=False,
                timeout_s=int(timeout_s),
            )

        try:
            df = _with_retries(_do_5m, attempts=attempts, failures=failures, symbol=sym, stage="fetch_5m_chunk")
        except Exception:
            # _with_retries already recorded the last real exception; skip this chunk.
            df = pd.DataFrame()

        if df is not None and not df.empty:
            dfs.append(df)

        cur = nxt

    if not dfs:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = pd.concat(dfs).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _select_top_liquid_symbols(
    provider: AlpacaProvider,
    symbols: list[str],
    *,
    top_n: int,
    daily_period: str,
    timeout_s: int,
    retries: int,
    failures: List[dict],
) -> list[str]:
    """Select top-N by avg20 dollar volume from daily data."""
    metrics: list[tuple[str, float]] = []
    for sym in symbols:
        try:
            def _do_daily():
                return provider.get_bars(
                    BarsRequest(symbol=sym, interval="1d", period=daily_period, include_prepost=False),
                    timeout_s=int(timeout_s),
                ).sort_index()

            d = _with_retries(_do_daily, attempts=retries, failures=failures, symbol=sym, stage="liquidity_daily_fetch")
            if d.empty or len(d) < 60:
                continue
            close = d["Close"].astype(float)
            vol = d["Volume"].astype(float)
            avg20_vol = float(vol.rolling(20).mean().iloc[-1] or 0.0)
            last_close = float(close.iloc[-1])
            adv = float(avg20_vol * last_close)
            if adv > 0:
                metrics.append((sym, adv))
        except Exception:
            continue
    metrics.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in metrics[: int(top_n)]]


def _time_holdout_split(
    X: pd.DataFrame,
    y_arr: np.ndarray,
    asof_days: np.ndarray,
    *,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Gold-standard time holdout split with guardrails.

    Sort by asof_days ascending and hold out the most recent `test_size` fraction.
    If the holdout has a single class, expand the window backward until both classes exist.
    """
    if len(X) != len(y_arr) or len(X) != len(asof_days):
        raise RuntimeError("Internal error: X/y/asof_days length mismatch")

    order = np.argsort(asof_days)
    Xo = X.iloc[order].reset_index(drop=True)
    yo = y_arr[order]
    tso = asof_days[order]

    n = len(yo)
    base_cut = int(n * (1.0 - float(test_size)))
    base_cut = max(1, min(n - 1, base_cut))

    # Move cut earlier if test set is single-class.
    cut = base_cut
    if len(set(yo[cut:])) < 2:
        step = max(1, int(n * 0.02))  # 2% steps
        found = False
        for c in range(base_cut - step, int(n * 0.50), -step):
            if 0 < c < n and len(set(yo[c:])) >= 2 and len(set(yo[:c])) >= 2:
                cut = c
                found = True
                break
        if not found:
            raise RuntimeError(
                "Time holdout test set collapsed to a single class. "
                "This is a real failure: broaden data horizon or adjust labeling."
            )

    meta = {
        "split_method": "time_holdout",
        "test_size": float(test_size),
        "train_start": str(pd.Timestamp(tso[0]).date()),
        "train_end": str(pd.Timestamp(tso[cut - 1]).date()),
        "test_start": str(pd.Timestamp(tso[cut]).date()),
        "test_end": str(pd.Timestamp(tso[-1]).date()),
        "train_pos_rate": float(np.mean(yo[:cut])),
        "test_pos_rate": float(np.mean(yo[cut:])),
    }
    return Xo.iloc[:cut], Xo.iloc[cut:], yo[:cut], yo[cut:], meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--universe", default="", help="nasdaq or file (symbols.txt)")
    ap.add_argument("--max_symbols", type=int, default=400)
    ap.add_argument(
        "--include_non_common",
        action="store_true",
        default=False,
        help="include symbols like preferreds/warrants/rights (contains $ or ends with .W/.R)",
    )
    ap.add_argument("--offset", type=int, default=0)

    ap.add_argument("--cache_dir", default="cache", help="cache dir for universe symbol list")
    ap.add_argument("--ttl_seconds", type=int, default=86400, help="TTL for cached universe symbols")

    ap.add_argument("--symbols_file", default="")

    # Gold-standard training controls
    ap.add_argument(
        "--mode",
        default="liquid",
        choices=["liquid", "outlier", "parabolic"],
        help="liquid=top liquid scalp ORB, outlier=<$20 outlier breaks, parabolic=extreme gap/RVOL setups",
    )
    ap.add_argument("--top_liquid", type=int, default=2000, help="top liquid symbols to keep in liquid mode")
    ap.add_argument("--intraday_years", type=float, default=2.0, help="intraday years to fetch (liquid: 2, outlier: 3)")
    ap.add_argument("--label_minutes", type=int, default=60, help="label window minutes after trigger (liquid: 60, outlier: 90)")
    ap.add_argument("--chunk_days", type=int, default=7, help="calendar days per intraday fetch chunk")
    ap.add_argument("--provider_timeout_s", type=int, default=25, help="hard timeout per provider request")
    ap.add_argument("--provider_retries", type=int, default=3, help="retries per provider request (real failures recorded)")

    # Evaluation split (gold standard defaults)
    ap.add_argument(
        "--split",
        default="time",
        choices=["time", "group", "random"],
        help="gold-standard evaluation split: time (default), group (by symbol), or random (rows; not recommended)",
    )
    ap.add_argument("--test_size", type=float, default=0.2, help="fraction of samples held out for evaluation")

    # Back-compat: if you pass --days, we map it to intraday_years ~= days/252
    ap.add_argument("--days", type=int, default=0, help="(compat) approximate trading days; overrides intraday_years")
    ap.add_argument("--out", default="models/orb_ranker.pkl")
    args = ap.parse_args()

    symbols = _load_symbols(args)
    provider = AlpacaProvider()

    X_rows: List[Dict[str, float]] = []
    y: List[int] = []
    row_days: List[pd.Timestamp] = []
    row_symbols: List[str] = []

    failures: List[dict] = []
    timeout_s = int(getattr(args, "provider_timeout_s", 25) or 25)
    retries = int(getattr(args, "provider_retries", 3) or 3)
    empty_5m: List[str] = []
    empty_daily: List[str] = []
    day_rows_attempted = 0
    day_rows_added = 0
    day_rows_skipped = 0
    symbols_with_rows = set()

    # Training horizons
    intraday_years = float(args.intraday_years)
    if int(args.days or 0) > 0:
        intraday_years = max(0.25, float(args.days) / 252.0)

    daily_period = "5y"
    mode = (args.mode or "liquid").strip().lower()
    if mode == "outlier" and int(args.label_minutes) <= 0:
        args.label_minutes = 90
    if mode == "liquid" and int(args.label_minutes) <= 0:
        args.label_minutes = 60

    # Liquid mode: reduce the universe up-front.
    if mode == "liquid":
        symbols = _select_top_liquid_symbols(
            provider,
            symbols,
            top_n=int(args.top_liquid),
            daily_period="1y",
            timeout_s=timeout_s,
            retries=retries,
            failures=failures,
        )

    print(
        f"Training universe symbols: {len(symbols)} (mode={mode}, intraday_years={intraday_years:.2f}, label_minutes={int(args.label_minutes)}, split={args.split})",
        flush=True,
    )

    for i, sym in enumerate(symbols, start=1):
        if i == 1 or (i % 50) == 0:
            print(f"[{i}/{len(symbols)}] fetch+featurize {sym}  (samples_so_far={len(X_rows)})", flush=True)

        # ---- 5m bars (required for labels) -----------------------------------
        try:
            df5 = _fetch_intraday_5m_multi_year(
                provider,
                sym,
                years=intraday_years,
                chunk_days=int(args.chunk_days),
                timeout_s=int(timeout_s),
                retries=int(retries),
                failures=failures,
            )
        except Exception as e:
            failures.append({"symbol": sym, "stage": "fetch_5m", "error": f"{type(e).__name__}: {e}"})
            continue

        df5 = df5.sort_index()
        if df5.empty:
            empty_5m.append(sym)
            failures.append({"symbol": sym, "stage": "empty_5m", "error": "empty dataframe"})
            continue

        # ---- daily history (required for features) ----------------------------
        try:
            daily = provider.get_daily_history(sym, period=daily_period)
        except Exception as e:
            failures.append({"symbol": sym, "stage": "fetch_daily", "error": f"{type(e).__name__}: {e}"})
            continue

        daily = daily.sort_index()
        if daily.empty:
            empty_daily.append(sym)
            failures.append({"symbol": sym, "stage": "empty_daily", "error": "empty dataframe"})
            continue

        # ---- build day-level samples ------------------------------------------
        # Use trading days present in intraday data.
        days = sorted({ts.normalize() for ts in df5.index})
        if not days:
            empty_5m.append(sym)
            failures.append({"symbol": sym, "stage": "no_days_in_5m", "error": "no normalized dates"})
            continue

        # Guard memory for multi-year intraday; keep most recent ~2000 distinct days.
        if len(days) > 2000:
            days = days[-2000:]

        symbol_added_any = False
        added_for_sym = 0
        for day in days:
            day_rows_attempted += 1
            try:
                # Outlier mode: only keep under-$20 days that look like outlier candidates.
                if mode == "outlier":
                    day_df = df5.loc[day.strftime("%Y-%m-%d")]
                    if day_df is None or day_df.empty:
                        raise RuntimeError("No bars for day")
                    first = day_df.iloc[0]
                    px = float(first["Open"])
                    if px >= 20.0:
                        raise ValueError("filtered_not_under_20")
                    feats_tmp = _features_day(daily, df5, day)
                    if not (
                        abs(float(feats_tmp.get("gap_pct", 0.0))) >= 4.0
                        or float(feats_tmp.get("early15_dollar_vol", 0.0)) >= 300_000.0
                        or float(feats_tmp.get("or_range_pct", 0.0)) >= 2.0
                    ):
                        raise ValueError("filtered_not_outlier")

                # Parabolic mode: extreme gap (≥20%) + early volume explosion + price <$50
                if mode == "parabolic":
                    day_df = df5.loc[day.strftime("%Y-%m-%d")]
                    if day_df is None or day_df.empty:
                        raise RuntimeError("No bars for day")
                    first = day_df.iloc[0]
                    px = float(first["Open"])
                    if px >= 50.0:
                        raise ValueError("filtered_price_above_50")
                    feats_tmp = _features_day(daily, df5, day)
                    gap = abs(float(feats_tmp.get("gap_pct", 0.0)))
                    rvol = float(feats_tmp.get("rvol_proxy", 0.0))
                    # Require extreme gap OR extreme early RVOL
                    if not (gap >= 20.0 or rvol >= 8.0):
                        raise ValueError("filtered_not_parabolic")

                feats = _features_day(daily, df5, day)
                label = _label_day(df5, day, label_minutes=int(args.label_minutes))
            except Exception as e:
                day_rows_skipped += 1
                failures.append({"symbol": sym, "stage": "feature_or_label", "error": f"{type(e).__name__}: {e}"})
                continue

            X_rows.append(feats)
            y.append(int(label))
            row_days.append(pd.Timestamp(day).normalize())
            row_symbols.append(sym)

            day_rows_added += 1
            symbol_added_any = True
            added_for_sym += 1

        if symbol_added_any:
            symbols_with_rows.add(sym)
            if added_for_sym > 0:
                print(f"    -> {sym}: added {added_for_sym} samples (total={len(X_rows)})", flush=True)

    # ---- summary -------------------------------------------------------------
    print("\n=== Training data summary ===")
    print(f"symbols attempted: {len(symbols)}")
    print(f"symbols with rows: {len(symbols_with_rows)}")
    print(f"day rows attempted: {day_rows_attempted}")
    print(f"day rows added: {day_rows_added}")
    print(f"day rows skipped: {day_rows_skipped}")
    print(f"empty 5m symbols: {len(empty_5m)}")
    print(f"empty daily symbols: {len(empty_daily)}")
    print(f"failures recorded: {len(failures)}")
    if empty_5m:
        print("empty_5m (first 25): " + ", ".join(empty_5m[:25]))
    if failures:
        print("failures (first 10):")
        for f in failures[:10]:
            print("  ", f)

    # ---- guardrails: real failure if insufficient data ----------------------
    MIN_ROWS = 500
    if len(X_rows) < MIN_ROWS:
        raise RuntimeError(
            f"Too few training samples: {len(X_rows)} rows collected (<{MIN_ROWS}). "
            "This is a real failure: provider data coverage was insufficient for the chosen universe/limits."
        )

    if len(set(y)) < 2:
        raise RuntimeError(
            f"Training labels collapsed to a single class (unique={set(y)}). "
            "This is a real failure: adjust labeling thresholds or expand data."
        )

    # ---- train ---------------------------------------------------------------
    X = pd.DataFrame(X_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_names = list(X.columns)
    y_arr = np.asarray(y, dtype=int)

    split_meta: dict = {"split_method": str(args.split), "test_size": float(args.test_size)}
    if args.split == "time":
        asof_days = np.asarray([pd.Timestamp(d).value for d in row_days], dtype=np.int64)
        X_train, X_test, y_train, y_test, meta = _time_holdout_split(X, y_arr, asof_days, test_size=float(args.test_size))
        split_meta.update(meta)
        print(
            f"\nSplit (time): train={len(y_train)} test={len(y_test)} | "
            f"train_pos={split_meta['train_pos_rate']:.4f} test_pos={split_meta['test_pos_rate']:.4f} | "
            f"train[{split_meta['train_start']}->{split_meta['train_end']}] "
            f"test[{split_meta['test_start']}->{split_meta['test_end']}]",
            flush=True,
        )
    elif args.split == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=42)
        groups = np.asarray(row_symbols)
        tr_idx, te_idx = next(gss.split(X, y_arr, groups=groups))
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y_arr[tr_idx], y_arr[te_idx]
        split_meta["split_method"] = "group_shuffle_by_symbol"
        split_meta["train_pos_rate"] = float(np.mean(y_train))
        split_meta["test_pos_rate"] = float(np.mean(y_test))
        print(
            f"\nSplit (group): train={len(y_train)} test={len(y_test)} | "
            f"train_pos={split_meta['train_pos_rate']:.4f} test_pos={split_meta['test_pos_rate']:.4f}",
            flush=True,
        )
        if len(set(y_test)) < 2:
            raise RuntimeError(
                "Group split test set collapsed to a single class. "
                "This is a real failure: increase data or adjust labeling."
            )
    else:
        # random rows (not recommended for time series)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_arr, test_size=float(args.test_size), random_state=42, stratify=y_arr
        )
        split_meta["split_method"] = "random_rows_stratified"
        split_meta["train_pos_rate"] = float(np.mean(y_train))
        split_meta["test_pos_rate"] = float(np.mean(y_test))
        print(
            f"\nSplit (random rows): train={len(y_train)} test={len(y_test)} | "
            f"train_pos={split_meta['train_pos_rate']:.4f} test_pos={split_meta['test_pos_rate']:.4f}",
            flush=True,
        )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else float("nan")
    print(f"\nTrained rows: {len(X)}  |  AUC: {auc}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "mode": mode,
        # audit metadata (gold standard)
        "label_minutes": int(args.label_minutes),
        "intraday_years": float(intraday_years),
        "chunk_days": int(args.chunk_days),
        "provider_timeout_s": int(timeout_s),
        "provider_retries": int(retries),
        "top_liquid": int(args.top_liquid) if mode == "liquid" else None,
        "universe": args.universe,
        "max_symbols": int(args.max_symbols),
        "include_non_common": bool(args.include_non_common),
        "auc": float(auc) if not (np.isnan(auc)) else None,
        "split": split_meta,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(X)),
        "symbols_attempted": int(len(symbols)),
        "symbols_with_rows": int(len(symbols_with_rows)),
    }

    joblib.dump(artifact, out)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
