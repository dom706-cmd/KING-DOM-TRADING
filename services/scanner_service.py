
from __future__ import annotations

from app import app

if __name__ == "__main__":
    import os
    host = os.getenv("ORB_HOST", "127.0.0.1")
    port = int(os.getenv("ORB_PORT", "8050"))
    debug = str(os.getenv("FLASK_DEBUG", "0")).lower() in ("1","true","yes","on")
    app.run(host=host, port=port, debug=debug, threaded=True)
