from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TypedFailure(Exception):
    code: str
    message: str
    stage: str | None = None
    symbol: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    cause_type: str | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.stage:
            out["stage"] = self.stage
        if self.symbol:
            out["symbol"] = self.symbol
        if self.cause_type:
            out["cause_type"] = self.cause_type
        if self.context:
            out["context"] = dict(self.context)
        return out


class SessionResolutionFailure(TypedFailure):
    pass


class PlanBuildFailure(TypedFailure):
    pass


class IntradayDataFailure(TypedFailure):
    pass


class TrendContextFailure(TypedFailure):
    pass


class EntryNowMLFailure(TypedFailure):
    pass


class MonitorRefreshFailure(TypedFailure):
    pass


class MonitorTradeRefreshFailure(MonitorRefreshFailure):
    pass


class MonitorQuoteRefreshFailure(MonitorRefreshFailure):
    pass


def failure_dict(
    exc: Any,
    *,
    code: str | None = None,
    message: str | None = None,
    stage: str | None = None,
    symbol: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(exc, TypedFailure):
        out = exc.to_dict()
        if code is not None:
            out["code"] = code
        if message is not None:
            out["message"] = message
        if stage is not None:
            out["stage"] = stage
        if symbol is not None:
            out["symbol"] = symbol
        if context:
            merged = dict(out.get("context") or {})
            merged.update(context)
            out["context"] = merged
        return out

    out: dict[str, Any] = {
        "code": code or "exception",
        "message": message or (str(exc) or type(exc).__name__),
        "cause_type": type(exc).__name__,
    }
    if stage is not None:
        out["stage"] = stage
    if symbol is not None:
        out["symbol"] = symbol
    if context:
        out["context"] = dict(context)
    return out


def failure_string(exc: Any) -> str:
    payload = exc if isinstance(exc, dict) else failure_dict(exc)
    code = str(payload.get("code") or "").strip()
    cause_type = str(payload.get("cause_type") or "").strip()
    message = str(payload.get("message") or payload.get("error") or "").strip()

    parts: list[str] = []
    if code:
        parts.append(code)
    if cause_type and cause_type != code:
        parts.append(cause_type)
    if message:
        parts.append(message)
    return ":".join(parts) if parts else "unknown_failure"
