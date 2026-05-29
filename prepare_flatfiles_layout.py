#!/usr/bin/env python3
# prepare_flatfiles_layout.py
#
# Convert downloaded Massive flatfiles into the local day-wise layout expected by the
# AB dataset builder.
#
# Input:  data/flatfiles/raw/<FULL_S3_KEY_PATH>
#   e.g.  data/flatfiles/raw/us_stocks_sip/minute_aggs_v1/2025/03/2025-03-20.csv.gz
#         data/flatfiles/raw/us_stocks_sip/day_aggs_v1/2026/02/2026-02-17.csv.gz
#
# Output:
#   Minute: data/minute_1m_flat/YYYY/MM/YYYY-MM-DD.csv.gz
#   Daily : data/daily_1d_flat/YYYY/MM/YYYY-MM-DD.csv.gz
#
# Tenants:
# - No fake/sample data. This script only copies/merges real downloaded files.
# - Real failures only: if a file is unreadable or headers mismatch, we raise or count failure.
# - Complete file, runnable.

from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Prefer YYYY-MM-DD in filename; also accept .../YYYY/MM/DD/... paths.
_RE_YMD_DASH = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
_RE_YMD_SLASH = re.compile(r"(20\d{2})/(\d{2})/(\d{2})")


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _classify(p: Path) -> Optional[str]:
    """Return 'minute', 'daily', or None."""
    s = str(p).lower()
    name = p.name.lower()

    # Minute (1m)
    if (
        "minute_aggs_v1" in s
        or "minute_aggs" in s
        or "minute" in s
        or "minute" in name
        or "_1m" in name
        or "1m" in name
    ):
        return "minute"

    # Daily (1d) — Massive uses day_aggs_v1
    if (
        "day_aggs_v1" in s
        or "day_aggs" in s
        or "daily_aggs_v1" in s
        or "daily_aggs" in s
        or "daily" in s
        or "daily" in name
        or "_1d" in name
        or "1d" in name
    ):
        return "daily"

    return None


def _valid_ymd(y: int, m: int, d: int) -> bool:
    return 2000 <= y <= 2099 and 1 <= m <= 12 and 1 <= d <= 31


def _extract_day(p: Path) -> Optional[Tuple[int, int, int]]:
    """Extract (Y,M,D) from filename or path."""
    # 1) filename first (most reliable for Massive: YYYY-MM-DD.csv.gz)
    m = _RE_YMD_DASH.search(p.name)
    if m:
        y, mo, d = map(int, m.groups())
        if _valid_ymd(y, mo, d):
            return y, mo, d

    # 2) anywhere in path with slashes
    m = _RE_YMD_SLASH.search(str(p))
    if m:
        y, mo, d = map(int, m.groups())
        if _valid_ymd(y, mo, d):
            return y, mo, d

    # 3) dash date anywhere in full path
    m = _RE_YMD_DASH.search(str(p))
    if m:
        y, mo, d = map(int, m.groups())
        if _valid_ymd(y, mo, d):
            return y, mo, d

    return None


def _open_maybe_gzip(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")


def _append_csv_gz(dst_gz: Path, src: Path, *, expect_header: Optional[str]) -> str:
    """Append src rows to dst_gz.

    - If dst_gz does not exist, create it and write full content (including header).
    - If dst_gz exists, skip the header line of src and append remaining lines.
    - Returns the header string used/validated.

    This uses gzip 'ab' to append as a new gzip member (valid gzip stream).
    """

    dst_gz.parent.mkdir(parents=True, exist_ok=True)

    with _open_maybe_gzip(src) as f_in:
        header = f_in.readline()
        if not header:
            raise RuntimeError(f"Empty CSV file: {src}")

        header = header.rstrip("\n")
        if expect_header is not None and header != expect_header:
            raise RuntimeError(
                f"CSV header mismatch for {src}.\nExpected: {expect_header[:200]}\nGot: {header[:200]}"
            )

        if not dst_gz.exists():
            # create new gz
            with gzip.open(dst_gz, "wt", encoding="utf-8", newline="") as f_out:
                f_out.write(header + "\n")
                shutil.copyfileobj(f_in, f_out)
        else:
            # append rows only
            with gzip.open(dst_gz, "at", encoding="utf-8", newline="") as f_out:
                shutil.copyfileobj(f_in, f_out)

    return header


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_root",
        required=True,
        help="Where ingest_flatfiles.py downloaded files (e.g. data/flatfiles/raw)",
    )
    ap.add_argument(
        "--minute_out",
        default="data/minute_1m_flat",
        help="Output root for minute day files",
    )
    ap.add_argument(
        "--daily_out",
        default="data/daily_1d_flat",
        help="Output root for daily day files",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan and report without writing outputs",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Treat any merge/header error as fatal (default: keep going and report).",
    )
    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser()
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    minute_out = Path(args.minute_out).expanduser()
    daily_out = Path(args.daily_out).expanduser()

    # Collect files per (kind, y, m, d)
    grouped: Dict[Tuple[str, int, int, int], List[Path]] = defaultdict(list)

    counts = {"minute": 0, "daily": 0, "unknown": 0}
    unknown_examples: List[str] = []

    for f in _iter_files(raw_root):
        kind = _classify(f)
        day = _extract_day(f)
        if not kind or not day:
            counts["unknown"] += 1
            if len(unknown_examples) < 15:
                unknown_examples.append(str(f))
            continue
        y, mo, d = day
        grouped[(kind, y, mo, d)].append(f)
        counts[kind] += 1

    print("Summary:", counts)

    if counts["minute"] == 0 or counts["daily"] == 0:
        print("\nERROR: Could not locate BOTH minute and daily flatfiles in raw_root.")
        print(
            "Needed: files whose path indicates minute_aggs_v1 (minute) AND day_aggs_v1 (daily), "
            "and includes a day token like YYYY/MM/DD or YYYY-MM-DD."
        )
        if unknown_examples:
            print("\nExamples of unrecognized files:")
            for x in unknown_examples:
                print("  ", x)
        return 2

    if args.dry_run:
        print(f"Minute layout -> {minute_out}")
        print(f"Daily  layout -> {daily_out}")
        print(f"Groups: {len(grouped)}")
        return 0

    # Write outputs: for each day, merge all parts into one day file
    failures = 0
    merged_days = 0
    multi_part_days = 0

    # Ensure output dirs exist
    minute_out.mkdir(parents=True, exist_ok=True)
    daily_out.mkdir(parents=True, exist_ok=True)

    # Deterministic order
    for (kind, y, mo, d), files in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        dst_root = minute_out if kind == "minute" else daily_out
        day_str = f"{y:04d}-{mo:02d}-{d:02d}.csv.gz"
        dst = dst_root / f"{y:04d}" / f"{mo:02d}" / day_str

        # Sort files for stable merges
        files_sorted = sorted(files)
        if len(files_sorted) > 1:
            multi_part_days += 1

        try:
            header: Optional[str] = None
            # If dst already exists from a previous run, we don't want to silently keep stale output.
            # Remove and rebuild from the current raw inputs.
            if dst.exists():
                dst.unlink()
            for src in files_sorted:
                header = _append_csv_gz(dst, src, expect_header=header)
            merged_days += 1
        except Exception as e:
            failures += 1
            if args.strict:
                raise
            else:
                print(f"WARNING: failed to merge {kind} {y:04d}-{mo:02d}-{d:02d}: {e}")

    print(f"Minute layout -> {minute_out}")
    print(f"Daily  layout -> {daily_out}")
    print(f"Days merged: {merged_days}")
    print(f"Days with multiple parts: {multi_part_days}")
    if failures:
        print(f"WARNING: {failures} day merges failed. See warnings above.")
        return 2

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
