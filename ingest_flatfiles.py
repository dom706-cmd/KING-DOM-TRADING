#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


# ---------------------------- Config / Env ----------------------------

@dataclass
class S3Cfg:
    endpoint: str
    bucket: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"
    max_workers: int = 6
    verify_ssl: bool = True
    addressing_style: str = "path"  # important for many S3-compatible providers


def env_s3_config() -> S3Cfg:
    endpoint = os.getenv("MASSIVE_S3_ENDPOINT", "").strip()
    bucket = os.getenv("MASSIVE_S3_BUCKET", "").strip()
    access_key = os.getenv("MASSIVE_S3_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("MASSIVE_S3_SECRET_ACCESS_KEY", "").strip()

    if not endpoint or not bucket or not access_key or not secret_key:
        missing = [
            k for k, v in [
                ("MASSIVE_S3_ENDPOINT", endpoint),
                ("MASSIVE_S3_BUCKET", bucket),
                ("MASSIVE_S3_ACCESS_KEY_ID", access_key),
                ("MASSIVE_S3_SECRET_ACCESS_KEY", secret_key),
            ]
            if not v
        ]
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    region = os.getenv("MASSIVE_S3_REGION", "us-east-1").strip() or "us-east-1"
    verify_ssl_env = os.getenv("MASSIVE_S3_VERIFY_SSL", "").strip().lower()
    verify_ssl = True if verify_ssl_env in ("", "1", "true", "yes") else False

    return S3Cfg(
        endpoint=endpoint,
        bucket=bucket,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        verify_ssl=verify_ssl,
    )


def s3_client(cfg: S3Cfg):
    # S3-compatible endpoints often require path-style addressing.
    bcfg = Config(
        region_name=cfg.region,
        retries={"max_attempts": 10, "mode": "standard"},
        s3={"addressing_style": cfg.addressing_style},
        signature_version="s3v4",
    )
    sess = boto3.session.Session()
    return sess.client(
        "s3",
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        verify=cfg.verify_ssl,
        config=bcfg,
    )


# ---------------------------- Listing ----------------------------

def _ensure_prefix(prefix: str) -> str:
    prefix = prefix.lstrip("/")
    # we accept either "path" or "path/" – normalize for common-prefix listing
    return prefix


def list_common_prefixes(prefix: str, *, limit: int, cfg: S3Cfg) -> List[str]:
    """
    List "directories" immediately under prefix using Delimiter='/' (S3 CommonPrefixes).
    This is what you want for:
      python ingest_flatfiles.py --list --prefix us_stocks_sip/
    """
    prefix = _ensure_prefix(prefix)
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    s3 = s3_client(cfg)
    out: List[str] = []
    token: Optional[str] = None

    while True:
        kwargs: Dict[str, Any] = {
            "Bucket": cfg.bucket,
            "Prefix": prefix,
            "Delimiter": "/",
            "MaxKeys": 1000,
        }
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        for cp in resp.get("CommonPrefixes", []) or []:
            p = cp.get("Prefix")
            if p:
                out.append(p)
                if limit and len(out) >= limit:
                    return out[:limit]

        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break

    return out[:limit] if limit else out


def iter_keys(prefix: str, *, cfg: S3Cfg) -> Iterable[str]:
    """
    Iterate ALL object keys under prefix (no delimiter), with full pagination.
    """
    prefix = _ensure_prefix(prefix)
    s3 = s3_client(cfg)
    token: Optional[str] = None

    while True:
        kwargs: Dict[str, Any] = {
            "Bucket": cfg.bucket,
            "Prefix": prefix,
            "MaxKeys": 1000,
        }
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        contents = resp.get("Contents") or []
        for obj in contents:
            k = obj.get("Key")
            if k:
                yield k

        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break


# ---------------------------- Downloading ----------------------------

def key_to_local_path(out_dir: Path, key: str) -> Path:
    """
    CRITICAL: preserve the full key path under out_dir.
    Example:
      key = us_stocks_sip/minute_aggs_v1/2025/03/2025-03-20.csv.gz
      -> out_dir/us_stocks_sip/minute_aggs_v1/2025/03/2025-03-20.csv.gz
    """
    key = key.lstrip("/")
    return out_dir / key


def _atomic_write_stream(dst: Path, body_stream, chunk_size: int = 8 * 1024 * 1024) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with tmp.open("wb") as f:
            while True:
                chunk = body_stream.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dst)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def download_one(
    key: str,
    *,
    out_dir: Path,
    cfg: S3Cfg,
    overwrite: bool,
    no_head: bool,
) -> Tuple[str, str, Optional[str]]:
    """
    Returns: (key, status, error)
      status in {"downloaded","skipped","failed"}
    """
    s3 = s3_client(cfg)
    dst = key_to_local_path(out_dir, key)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        # optional: verify size via HeadObject (if allowed)
        if not no_head:
            try:
                h = s3.head_object(Bucket=cfg.bucket, Key=key)
                remote_size = int(h.get("ContentLength") or 0)
                local_size = dst.stat().st_size
                if remote_size > 0 and local_size == remote_size:
                    return (key, "skipped", None)
            except ClientError as e:
                # If HeadObject is forbidden, fall back to skip-by-existence.
                code = (e.response.get("Error") or {}).get("Code") if hasattr(e, "response") else None
                if code in ("403", "AccessDenied", "Forbidden"):
                    return (key, "skipped", None)
                # otherwise: real error
                return (key, "failed", f"HeadObject error: {repr(e)}")
            except BotoCoreError as e:
                return (key, "failed", f"HeadObject error: {repr(e)}")

        # no_head or couldn't verify size: conservatively skip if exists
        return (key, "skipped", None)

    try:
        resp = s3.get_object(Bucket=cfg.bucket, Key=key)
        body = resp["Body"]
        _atomic_write_stream(dst, body)
        return (key, "downloaded", None)
    except (ClientError, BotoCoreError) as e:
        return (key, "failed", f"GetObject error: {repr(e)}")
    except Exception as e:
        return (key, "failed", f"Unexpected error: {repr(e)}")


def download_prefix(
    prefix: str,
    *,
    out_dir: Path,
    cfg: S3Cfg,
    max_files: Optional[int],
    overwrite: bool,
    no_head: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Download all keys under prefix to out_dir, preserving key paths.
    Returns (downloaded_keys, manifest_dict).
    """
    t0 = time.time()
    prefix = _ensure_prefix(prefix)

    all_keys: List[str] = []
    for k in iter_keys(prefix, cfg=cfg):
        # ignore "folder marker" keys if any
        if k.endswith("/") and len(k) > 1:
            continue
        all_keys.append(k)
        if max_files and len(all_keys) >= max_files:
            break

    downloaded: List[str] = []
    skipped: List[str] = []
    failures: List[Dict[str, str]] = []

    # If nothing to do, return a truthful manifest.
    if not all_keys:
        manifest = {
            "endpoint": cfg.endpoint,
            "bucket": cfg.bucket,
            "prefix": prefix,
            "out_dir": str(out_dir),
            "max_files": max_files,
            "overwrite": overwrite,
            "no_head": no_head,
            "max_workers": cfg.max_workers,
            "counts": {"listed": 0, "downloaded": 0, "skipped": 0, "failed": 0},
            "elapsed_s": round(time.time() - t0, 3),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return [], manifest

    with cf.ThreadPoolExecutor(max_workers=max(1, int(cfg.max_workers))) as ex:
        futs = [
            ex.submit(
                download_one,
                k,
                out_dir=out_dir,
                cfg=cfg,
                overwrite=overwrite,
                no_head=no_head,
            )
            for k in all_keys
        ]
        for fut in cf.as_completed(futs):
            k, status, err = fut.result()
            if status == "downloaded":
                downloaded.append(k)
            elif status == "skipped":
                skipped.append(k)
            else:
                failures.append({"key": k, "error": err or "unknown"})

    manifest = {
        "endpoint": cfg.endpoint,
        "bucket": cfg.bucket,
        "prefix": prefix,
        "out_dir": str(out_dir),
        "max_files": max_files,
        "overwrite": overwrite,
        "no_head": no_head,
        "max_workers": cfg.max_workers,
        "counts": {
            "listed": len(all_keys),
            "downloaded": len(downloaded),
            "skipped": len(skipped),
            "failed": len(failures),
        },
        "downloaded": downloaded[:2000],  # cap to keep manifest sane
        "skipped": skipped[:2000],
        "failures": failures[:2000],
        "elapsed_s": round(time.time() - t0, 3),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return downloaded, manifest


# ---------------------------- CLI ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Download Massive Flat Files (S3-compatible).")
    p.add_argument("--prefix", required=True, help="S3 key prefix (e.g. us_stocks_sip/minute_aggs_v1/2025/).")
    p.add_argument("--out_dir", default="data/flatfiles/raw", help="Output directory for downloaded files.")
    p.add_argument("--max_files", type=int, default=0, help="Max files to download (0 = no limit).")
    p.add_argument("--max_workers", type=int, default=6, help="Concurrent download workers.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    p.add_argument("--manifest", default="data/flatfiles/manifest.json", help="Path to write manifest JSON.")
    p.add_argument("--list", action="store_true", help="List prefixes under --prefix and exit.")
    p.add_argument("--list_limit", type=int, default=200, help="Max prefixes to return in list mode.")
    p.add_argument(
        "--no_head",
        action="store_true",
        help="Avoid HeadObject calls (some providers forbid HeadObject). Uses GetObject streaming instead.",
    )

    args = p.parse_args(argv)

    cfg = env_s3_config()
    if args.max_workers:
        cfg.max_workers = int(args.max_workers)

    if args.list:
        items = list_common_prefixes(prefix=args.prefix, limit=int(args.list_limit), cfg=cfg)
        for x in items:
            print(x)
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded, manifest = download_prefix(
        prefix=args.prefix,
        out_dir=out_dir,
        cfg=cfg,
        max_files=int(args.max_files) if args.max_files and int(args.max_files) > 0 else None,
        overwrite=bool(args.overwrite),
        no_head=bool(args.no_head),
    )

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Downloaded {len(downloaded)} files -> {out_dir}")
    print(f"Manifest -> {manifest_path}")
    # surface failures loudly (real failures only)
    if manifest.get("counts", {}).get("failed", 0):
        print(f"WARNING: {manifest['counts']['failed']} downloads failed. See manifest failures[]", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
