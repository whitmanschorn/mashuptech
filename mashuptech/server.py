"""Local dashboard: static page + JSON state + Server-Sent Events + a couple of actions."""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import Config
from .pipeline import Pipeline
from .state import JobStore

WEB_DIR = Path(__file__).parent / "web"


def _reveal(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)] if path.is_file() else ["open", str(path)])
    elif sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", str(path.parent if path.is_file() else path)])
    else:  # pragma: no cover
        subprocess.Popen(["explorer", str(path.parent if path.is_file() else path)])


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, config: Config, store: JobStore, pipeline: Pipeline):
        super().__init__(address, Handler)
        self.config = config
        self.store = store
        self.pipeline = pipeline


class Handler(BaseHTTPRequestHandler):
    server: DashboardServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # silence per-request logging
        pass

    # ---------- helpers ----------
    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    # ---------- GET ----------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(self.server.store.snapshot())
        elif path == "/api/config":
            self._json(self.server.config.as_dict())
        elif path == "/api/events":
            self._sse()
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _sse(self) -> None:
        store = self.server.store
        q = store.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self._write_event(store.snapshot())
            while True:
                try:
                    snap = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                # Coalesce bursts: only send the newest snapshot waiting in the queue.
                while True:
                    try:
                        snap = q.get_nowait()
                    except queue.Empty:
                        break
                self._write_event(snap)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            store.unsubscribe(q)

    def _write_event(self, snap: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(snap).encode() + b"\n\n")
        self.wfile.flush()

    # ---------- POST ----------
    def do_POST(self) -> None:
        parts = [p for p in self.path.split("?", 1)[0].split("/") if p]
        store, pipeline, cfg = self.server.store, self.server.pipeline, self.server.config
        if parts[:2] == ["api", "retry"] and len(parts) == 3:
            ok = pipeline.retry(parts[2])
            self._json({"ok": ok}, 200 if ok else HTTPStatus.BAD_REQUEST)
        elif parts[:2] == ["api", "reveal"] and len(parts) == 3:
            job = store.get(parts[2])
            target = Path(job.output_dir) if job and job.output_dir else (Path(job.source) if job else None)
            if target and target.exists():
                _reveal(target)
                self._json({"ok": True})
            else:
                self._json({"ok": False, "error": "path missing"}, HTTPStatus.NOT_FOUND)
        elif parts == ["api", "open", "inbox"]:
            _reveal(cfg.inbox)
            self._json({"ok": True})
        elif parts == ["api", "open", "outbox"]:
            _reveal(cfg.outbox)
            self._json({"ok": True})
        elif parts == ["api", "dismiss"] and len(parts) == 2:
            # Remove finished/failed records from the dashboard (files are untouched).
            for job in store.jobs():
                if job.status in ("done", "failed"):
                    store.remove(job.id)
            self._json({"ok": True})
        elif parts[:2] == ["api", "dismiss"] and len(parts) == 3:
            job = store.get(parts[2])
            if job and job.status in ("done", "failed"):
                store.remove(job.id)
                self._json({"ok": True})
            else:
                self._json({"ok": False}, HTTPStatus.BAD_REQUEST)
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def start_server(config: Config, store: JobStore, pipeline: Pipeline) -> DashboardServer:
    server = DashboardServer(("127.0.0.1", config.port), config, store, pipeline)
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    return server
