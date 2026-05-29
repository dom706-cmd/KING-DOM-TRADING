
from __future__ import annotations

import json
import time
from app import _RUNTIME_STORE, _PUBSUB

if __name__ == "__main__":
    seen = set()
    print("alert_service running")
    while True:
        alerts = _RUNTIME_STORE.recent_alerts(limit=100)
        for row in reversed(alerts):
            event_id = str(row.get("event_id") or "")
            if not event_id or event_id in seen:
                continue
            seen.add(event_id)
            print(json.dumps(row, default=str))
            if _PUBSUB is not None:
                try:
                    _PUBSUB.publish("alerts", row)
                except Exception as e:
                    print(f"pubsub_error:{e}")
        time.sleep(2.0)
