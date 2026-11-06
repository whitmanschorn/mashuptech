"""Polls the inbox and queues files once they have finished copying."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from .config import AUDIO_EXTENSIONS, Config
from .state import JobStore


class InboxWatcher(threading.Thread):
    def __init__(self, config: Config, store: JobStore, stop: threading.Event):
        super().__init__(name="inbox-watcher", daemon=True)
        self.config = config
        self.store = store
        self.stop_event = stop
        self._seen_sizes: dict[Path, int] = {}

    def candidates(self) -> list[Path]:
        files = []
        for p in sorted(self.config.inbox.iterdir()):
            if p.is_dir() or p.name.startswith(".") or p.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            files.append(p)
        return files

    def scan_once(self) -> None:
        current: dict[Path, int] = {}
        for p in self.candidates():
            try:
                size = p.stat().st_size
            except FileNotFoundError:
                continue
            current[p] = size
            # Stable across two polls and non-empty -> ready.
            if size > 0 and self._seen_sizes.get(p) == size:
                job = self.store.add(p)
                if job:
                    self.store.log(f"Queued {p.name}", job_id=job.id)
        self._seen_sizes = current

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.scan_once()
            except Exception as exc:  # keep the watcher alive no matter what
                self.store.log(f"Watcher error: {exc}", level="error")
            self.stop_event.wait(self.config.poll_interval)
