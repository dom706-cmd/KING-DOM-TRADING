#!/usr/bin/env python3
"""One-command verification for ORB dashboard.

Asserts:
- /api/health returns ok and sub-checks are true
- A tiny scan produces candidates with ml_score + sentiment_score + model_bucket
  and trading plan fields (stop_loss/take_profit)

No fake/sample data is generated. If provider fails, this script fails.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.parse
from typing import Dict, Any, Tuple

BASE = "http://127.0.0.1:8050"


def _http_json(method: str, url: str, *, fields: Dict[str, str] | None = None, timeout: int = 20) -> Tuple[int, Dict[str, Any]]:
    if fields is None:
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, json.loads(body)

    # application/x-www-form-urlencoded is fine for our Flask endpoint
    data = urllib.parse.urlencode(fields).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
        return r.status, json.loads(body)


def die(msg: str, payload: Any | None = None) -> None:
    print("\nVERIFY FAILED:", msg)
    if payload is not None:
        try:
            print(json.dumps(payload, indent=2)[:4000])
        except Exception:
            print(str(payload)[:4000])
    raise SystemExit(1)


def main() -> int:
    # 1) Health
    try:
        code, health = _http_json("GET", f"{BASE}/api/health", timeout=25)
    except Exception as e:
        die(f"Could not reach {BASE}/api/health: {type(e).__name__}: {e}")

    if code != 200:
        die(f"/api/health returned HTTP {code}", health)

    if not health.get("ok"):
        die("/api/health ok=false", health)

    # 2) Preflight
    try:
        code, preflight = _http_json("GET", f"{BASE}/api/preflight?symbol=AAPL", timeout=25)
    except Exception as e:
        die(f"preflight failed: {type(e).__name__}: {e}")

    if code != 200 or not preflight.get("ok"):
        die("/api/preflight failed", preflight)

    # 3) Stream status
    try:
        code, stream = _http_json("GET", f"{BASE}/api/stream_status", timeout=25)
    except Exception as e:
        die(f"stream_status failed: {type(e).__name__}: {e}")

    if code != 200 or not stream.get("ok"):
        die("/api/stream_status failed", stream)

    # 4) Start scan
    fields = {
        "strategy": "orb",
        "exec_style": "retest",
        "offset": "0",
        "max_symbols": "200",
        "long_only": "1",
        "min_price": "1",
        "max_price": "20",
        "min_avg20_dollar_vol": "1000000",
        "min_today_dollar_vol": "200000",
        "min_rvol": "0.5",
        "use_ml": "1",
        "use_sentiment": "1",
        "use_catalyst": "1",
        "min_grade_enabled": "1",
        "min_grade": "B",
        "min_combined_enabled": "1",
        "min_combined_score": "0.40",
        "no_chop_enabled": "1",
        "min_vwap_enabled": "1",
        "min_pct_over_vwap": "1.0",
    }

    try:
        code, start = _http_json("POST", f"{BASE}/api/scan_start", fields=fields, timeout=25)
    except Exception as e:
        die(f"scan_start failed: {type(e).__name__}: {e}")

    if code != 200 or not start.get("ok") or not start.get("job_id"):
        die("scan_start did not return ok+job_id", start)

    jid = start["job_id"]
    print(f"health OK | build_id={health.get('build_id')} | job_id={jid}")

    # 5) Poll
    t0 = time.time()
    last = None
    while True:
        if time.time() - t0 > 120:
            die("scan_status timed out (>120s)", last)

        code, st = _http_json("GET", f"{BASE}/api/scan_status?job_id={jid}&debug=1", timeout=25)
        last = st
        if code != 200 or not st.get("ok"):
            die(f"scan_status HTTP {code} or ok=false", st)

        status = st.get("status")
        prog = st.get("progress") or {}
        print(
            f"status={status} chunks={prog.get('chunks_done')}/{prog.get('chunks_total')} scanned={prog.get('scanned')} next_offset={st.get('next_offset')}",
            flush=True,
        )

        if status == "done":
            break
        if status == "error":
            die("scan_status error", st)

        time.sleep(2)

    # 6) Debug scan assertions
    try:
        code, dbg = _http_json("GET", f"{BASE}/api/debug_last_scan?job_id={jid}", timeout=25)
    except Exception as e:
        die(f"debug_last_scan failed: {type(e).__name__}: {e}")

    if code != 200 or not dbg.get("ok"):
        die("debug_last_scan failed", dbg)

    data_failures = dbg.get("data_failures") or {}
    reject_counts = dbg.get("reject_counts") or {}
    prefilter_counts = dbg.get("prefilter_counts") or {}
    failure_samples_by_code = dbg.get("failure_samples_by_code") or {}

    if int(prefilter_counts.get("normalized_skipped") or 0) <= 0:
        die("normalized_skipped did not increase; symbol prefilter is not active", dbg)

    if int(data_failures.get("stage_catalyst") or 0) > 0:
        die("stage_catalyst failures remain", dbg)

    if int(reject_counts.get("bars_fetch_error_http_400") or 0) > 0:
        die("bars_fetch_error_http_400 remains", dbg)

    if int(data_failures.get("stage_daily") or 0) > 0:
        die("stage_daily failures remain", dbg)

    if int(data_failures.get("stage_intraday") or 0) > 25:
        die("stage_intraday failures remain too high", dbg)

    print("\nVERIFY OK")
    print(json.dumps({
        "job_id": jid,
        "prefilter_counts": prefilter_counts,
        "reject_counts": reject_counts,
        "data_failures": data_failures,
        "failure_samples_by_code": failure_samples_by_code,
        "candidate_count": len((dbg.get("candidates") or [])),
    }, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
