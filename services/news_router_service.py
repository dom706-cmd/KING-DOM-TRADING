
from __future__ import annotations

import time
from app import _MONITOR, _ALPACA_PROVIDER, _STREAM, _CONTEXT_ENGINE, _RUNTIME_STORE, _PUBSUB

if __name__ == "__main__":
    _MONITOR.configure_runtime(provider=_ALPACA_PROVIDER, stream_cache=_STREAM, store=_RUNTIME_STORE, context_engine=_CONTEXT_ENGINE, pubsub=_PUBSUB)
    print("news_router_service running")
    while True:
        time.sleep(60)
