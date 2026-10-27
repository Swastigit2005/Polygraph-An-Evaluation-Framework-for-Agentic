"""Run the web demo locally (static page + API) without the Vercel CLI.

    python scripts/dev_server.py        # http://localhost:8000
"""

from __future__ import annotations

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from ask import handler as ApiHandler  # noqa: E402


class DevHandler(SimpleHTTPRequestHandler):
    """Serve public/ and send /api/ask to the serverless handler."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/ask"):
            ApiHandler.do_GET(self)  # type: ignore[arg-type]
        else:
            super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/api/ask"):
            ApiHandler.do_POST(self)  # type: ignore[arg-type]
        else:
            self.send_error(404)

    _send = ApiHandler._send


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(DevHandler, directory=str(ROOT / "public")))
    print(f"Polygraph demo on http://localhost:{port}")
    server.serve_forever()
