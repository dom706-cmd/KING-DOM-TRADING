#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from urllib.parse import parse_qs, urlparse

from requests_oauthlib import OAuth1Session


def env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise SystemExit(f"Missing required env var: {name}")
    return v


def main() -> int:
    consumer_key = env("ETRADE_CONSUMER_KEY")
    consumer_secret = env("ETRADE_CONSUMER_SECRET")
    sandbox = (os.getenv("ETRADE_SANDBOX") or "1").strip().lower() in {"1", "true", "yes"}
    callback = os.getenv("ETRADE_CALLBACK_URL", "oob").strip() or "oob"

    base = "https://apisb.etrade.com" if sandbox else "https://api.etrade.com"
    request_token_url = f"{base}/oauth/request_token"
    authorize_url = f"{base}/e/t/etws/authorize"
    access_token_url = f"{base}/oauth/access_token"

    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        callback_uri=callback,
        signature_type="AUTH_HEADER",
    )

    try:
        fetch = oauth.fetch_request_token(request_token_url)
    except Exception as e:
        raise SystemExit(f"Failed to fetch request token: {e}")

    resource_owner_key = fetch.get("oauth_token")
    resource_owner_secret = fetch.get("oauth_token_secret")
    if not resource_owner_key or not resource_owner_secret:
        raise SystemExit("E*TRADE did not return request token/secret")

    auth_url = oauth.authorization_url(authorize_url)
    print("\n=== STEP 1: AUTHORIZE THIS APP ===")
    print(auth_url)
    print("\nAfter approving, E*TRADE will either:")
    print("- show a verifier code, or")
    print("- redirect to a callback URL containing oauth_verifier")
    print("\nPaste the FULL callback URL or just the verifier code below.\n")

    raw = input("Verifier or callback URL: ").strip()
    if not raw:
        raise SystemExit("No verifier entered")

    verifier = raw
    if "oauth_verifier=" in raw:
        parsed = urlparse(raw)
        qs = parse_qs(parsed.query)
        verifier = (qs.get("oauth_verifier") or [""])[0].strip()
    if not verifier:
        raise SystemExit("Could not parse oauth_verifier")

    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=resource_owner_key,
        resource_owner_secret=resource_owner_secret,
        verifier=verifier,
        signature_type="AUTH_HEADER",
    )

    try:
        final = oauth.fetch_access_token(access_token_url)
    except Exception as e:
        raise SystemExit(f"Failed to fetch access token: {e}")

    access_token = final.get("oauth_token")
    access_secret = final.get("oauth_token_secret")
    if not access_token or not access_secret:
        raise SystemExit("E*TRADE did not return final access token/secret")

    print("\n=== STEP 2: PUT THESE INTO ~/KingDom/.orb_env ===")
    print(f"ETRADE_OAUTH_TOKEN={access_token}")
    print(f"ETRADE_OAUTH_TOKEN_SECRET={access_secret}")
    print("\nThen restart the app and test /api/broker_snapshot.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
