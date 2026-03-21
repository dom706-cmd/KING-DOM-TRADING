from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests
from requests_oauthlib import OAuth1Session


@dataclass
class ETradeConfig:
    consumer_key: str
    consumer_secret: str
    oauth_token: str
    oauth_token_secret: str
    account_id_key: str | None = None
    sandbox: bool = True


class ETradeProvider:
    name = "etrade"

    def __init__(self, cfg: ETradeConfig | None = None):
        if cfg is None:
            cfg = ETradeConfig(
                consumer_key=os.getenv("ETRADE_CONSUMER_KEY", "").strip(),
                consumer_secret=os.getenv("ETRADE_CONSUMER_SECRET", "").strip(),
                oauth_token=os.getenv("ETRADE_OAUTH_TOKEN", "").strip(),
                oauth_token_secret=os.getenv("ETRADE_OAUTH_TOKEN_SECRET", "").strip(),
                account_id_key=(os.getenv("ETRADE_ACCOUNT_ID_KEY") or "").strip() or None,
                sandbox=(os.getenv("ETRADE_SANDBOX") or "1").strip().lower() in {"1", "true", "yes"},
            )
        missing = [
            name
            for name, value in {
                "ETRADE_CONSUMER_KEY": cfg.consumer_key,
                "ETRADE_CONSUMER_SECRET": cfg.consumer_secret,
                "ETRADE_OAUTH_TOKEN": cfg.oauth_token,
                "ETRADE_OAUTH_TOKEN_SECRET": cfg.oauth_token_secret,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing required E*TRADE env vars: {', '.join(missing)}")
        self.cfg = cfg

    def _base_url(self) -> str:
        return "https://apisb.etrade.com" if self.cfg.sandbox else "https://api.etrade.com"

    def _session(self) -> OAuth1Session:
        return OAuth1Session(
            self.cfg.consumer_key,
            client_secret=self.cfg.consumer_secret,
            resource_owner_key=self.cfg.oauth_token,
            resource_owner_secret=self.cfg.oauth_token_secret,
            signature_type="AUTH_HEADER",
        )

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None, json_body: dict[str, Any] | None = None, timeout_s: float = 20.0) -> Any:
        url = f"{self._base_url()}{path}"
        sess = self._session()
        headers = {"Accept": "application/json"}
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        resp = sess.request(method.upper(), url, params=params, json=json_body, headers=headers, timeout=timeout_s)
        if resp.status_code >= 400:
            raise RuntimeError(f"E*TRADE {method.upper()} {path} failed: HTTP {resp.status_code} {resp.text[:400]}")
        if not resp.text:
            return {}
        ctype = resp.headers.get("content-type", "")
        if "json" in ctype.lower() or resp.text.lstrip().startswith(("{", "[")):
            return resp.json()
        raise RuntimeError(f"E*TRADE {method.upper()} {path} returned non-JSON response")

    def list_accounts(self, timeout_s: float = 20.0) -> list[dict[str, Any]]:
        payload = self._request("GET", "/v1/accounts/list.json", timeout_s=timeout_s)
        accounts = (((payload or {}).get("AccountListResponse") or {}).get("Accounts") or {}).get("Account") or []
        return accounts if isinstance(accounts, list) else []

    def _resolve_account_id_key(self, timeout_s: float = 20.0) -> str:
        if self.cfg.account_id_key:
            return self.cfg.account_id_key
        accounts = self.list_accounts(timeout_s=timeout_s)
        if not accounts:
            raise RuntimeError("No E*TRADE accounts available")
        acct = accounts[0]
        key = str(acct.get("accountIdKey") or "").strip()
        if not key:
            raise RuntimeError("Could not resolve E*TRADE accountIdKey")
        self.cfg.account_id_key = key
        return key

    def get_account_balance(self, timeout_s: float = 20.0) -> dict[str, Any]:
        account_id_key = self._resolve_account_id_key(timeout_s=timeout_s)
        payload = self._request("GET", f"/v1/accounts/{account_id_key}/balance.json", params={"instType": "BROKERAGE"}, timeout_s=timeout_s)
        return (payload or {}).get("BalanceResponse") or {}

    def get_positions(self, timeout_s: float = 20.0) -> list[dict[str, Any]]:
        account_id_key = self._resolve_account_id_key(timeout_s=timeout_s)
        payload = self._request("GET", f"/v1/accounts/{account_id_key}/portfolio.json", timeout_s=timeout_s)
        portfolio = (payload or {}).get("PortfolioResponse") or {}
        acct_portfolios = portfolio.get("AccountPortfolio") or []
        if isinstance(acct_portfolios, dict):
            acct_portfolios = [acct_portfolios]
        out: list[dict[str, Any]] = []
        for acct in acct_portfolios:
            positions = acct.get("Position") or []
            if isinstance(positions, dict):
                positions = [positions]
            for row in positions:
                prod = row.get("Product") or {}
                symbol = str(prod.get("symbol") or "").upper()
                qty = float(row.get("quantity") or 0.0)
                price_paid = float(row.get("pricePaid") or 0.0)
                current_price = float(row.get("marketPrice") or 0.0)
                market_value = float(row.get("marketValue") or 0.0)
                pnl = float(row.get("totalGain") or 0.0)
                side = "long" if qty >= 0 else "short"
                qty_abs = abs(qty)
                cost_basis = qty_abs * price_paid
                unreal_pct = (pnl / cost_basis) if cost_basis else 0.0
                out.append({
                    "symbol": symbol,
                    "side": side,
                    "qty": qty_abs,
                    "avg_entry_price": price_paid,
                    "market_value": market_value,
                    "cost_basis": cost_basis,
                    "unrealized_pl": pnl,
                    "unrealized_plpc": unreal_pct,
                    "current_price": current_price,
                    "lastday_price": None,
                    "change_today": None,
                    "asset_class": prod.get("securityType"),
                })
        return out

    def get_broker_snapshot(self, timeout_s: float = 20.0) -> dict[str, Any]:
        account_id_key = self._resolve_account_id_key(timeout_s=timeout_s)
        balance = self.get_account_balance(timeout_s=timeout_s)
        computed = balance.get("Computed") or {}
        positions = self.get_positions(timeout_s=timeout_s)
        return {
            "broker": "etrade",
            "account_id": balance.get("accountId"),
            "account_number": balance.get("accountDescription") or account_id_key,
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": float(computed.get("cashBuyingPower") or computed.get("marginBuyingPower") or 0.0),
            "cash": float(computed.get("cashAvailableForInvestment") or computed.get("settledCashForInvestment") or 0.0),
            "equity": float(computed.get("netCash") or computed.get("totalAccountValue") or 0.0),
            "last_equity": float(computed.get("totalAccountValue") or 0.0),
            "portfolio_value": float(computed.get("totalAccountValue") or 0.0),
            "daytrade_count": 0,
            "multiplier": None,
            "pattern_day_trader": False,
            "positions": positions,
            "position_count": len(positions),
        }

    def submit_exit_order(self, *, symbol: str, qty: float | None = None, notional_pct: float | None = None, limit_price: float | None = None, timeout_s: float = 20.0) -> dict[str, Any]:
        account_id_key = self._resolve_account_id_key(timeout_s=timeout_s)
        positions = {str(p.get('symbol') or '').upper(): p for p in self.get_positions(timeout_s=timeout_s)}
        pos = positions.get(str(symbol or '').strip().upper())
        if not pos:
            raise RuntimeError(f"no_open_position:{symbol}")
        qty_open = float(pos.get("qty") or 0.0)
        if qty_open <= 0:
            raise RuntimeError(f"no_open_qty:{symbol}")
        final_qty = qty_open
        if qty is not None:
            final_qty = min(qty_open, max(0.0, float(qty)))
        elif notional_pct is not None:
            final_qty = min(qty_open, max(0.0, qty_open * float(notional_pct)))
        if final_qty <= 0:
            raise RuntimeError(f"invalid_exit_qty:{symbol}")
        order_action = "SELL" if str(pos.get("side") or "long") == "long" else "BUY_TO_COVER"
        price_type = "LIMIT" if limit_price is not None else "MARKET"
        body = {
            "PlaceOrderRequest": {
                "orderType": "EQ",
                "clientOrderId": f"kingdom-{symbol.lower()}-{int(os.times().elapsed*1000)}",
                "Order": {
                    "allOrNone": False,
                    "priceType": price_type,
                    "orderTerm": "GOOD_FOR_DAY",
                    "marketSession": "REGULAR",
                    "Instrument": {
                        "Product": {
                            "securityType": "EQ",
                            "symbol": str(symbol or '').strip().upper(),
                        },
                        "orderAction": order_action,
                        "quantityType": "QUANTITY",
                        "quantity": int(round(final_qty)),
                    },
                },
            }
        }
        if limit_price is not None:
            body["PlaceOrderRequest"]["Order"]["limitPrice"] = float(limit_price)
        return self._request("POST", f"/v1/accounts/{account_id_key}/orders/place.json", json_body=body, timeout_s=timeout_s)
