"""Thread-safe job store, persisted to JSON, with change notifications for the SSE feed."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

QUEUED, PROCESSING, DONE, FAILED = "queued", "processing", "done", "failed"
STAGES = ("decode", "separate", "analyze", "encode", "finalize")

_pylog = logging.getLogger("mashuptech")
_LEVELS = {"info": logging.INFO, "warn": logging.WARNING, "error": logging.ERROR}


@dataclass
class Job:
    id: str
    source: str  # current path of the input file
    filename: str
    status: str = QUEUED
    stage: str | None = None
    progress: float = 0.0  # 0..1 within the current stage (separation mostly)
    artist: str | None = None
    title: str | None = None
    bpm: float | None = None
    key: str | None = None
    duration: float | None = None  # seconds of audio
    output_dir: str | None = None
    error: str | None = None
    added_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    device: str | None = None

    @property
    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.finished_at or time.time()) - self.started_at


class JobStore:
    def __init__(self, path: Path, log_lines: int = 200):
        self.path = path
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._log: deque[dict] = deque(maxlen=log_lines)
        self._subscribers: list[queue.Queue] = []
        self.info: dict = {"status": "starting", "device": None, "model": None}
        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except Exception:
            return
        for j in raw.get("jobs", []):
            job = Job(**{k: v for k, v in j.items() if k in Job.__dataclass_fields__})
            # Anything mid-flight when we died goes back to the queue.
            if job.status == PROCESSING:
                job.status, job.stage, job.progress, job.started_at = QUEUED, None, 0.0, None
            self._jobs[job.id] = job
            self._order.append(job.id)
        for line in raw.get("log", []):
            self._log.append(line)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"jobs": [asdict(self._jobs[i]) for i in self._order], "log": list(self._log)}))
        tmp.replace(self.path)

    def _changed(self) -> None:
        self._save()
        snapshot = self.snapshot()
        for q in list(self._subscribers):
            try:
                q.put_nowait(snapshot)
            except queue.Full:
                pass

    # ---------- subscriptions ----------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=50)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    # ---------- mutations ----------
    def set_info(self, **kwargs) -> None:
        with self._lock:
            self.info.update(kwargs)
            self._changed()

    def log(self, message: str, level: str = "info", job_id: str | None = None) -> None:
        _pylog.log(_LEVELS.get(level, logging.INFO), message)
        with self._lock:
            self._log.append({"t": time.time(), "level": level, "msg": message, "job": job_id})
            self._changed()

    def add(self, source: Path) -> Job | None:
        """Queue a file unless it is already known under that path and not finished."""
        with self._lock:
            for job in self._jobs.values():
                if job.source == str(source) and job.status in (QUEUED, PROCESSING):
                    return None
            job = Job(id=uuid.uuid4().hex[:10], source=str(source), filename=source.name)
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._changed()
            return job

    def next_queued(self) -> Job | None:
        with self._lock:
            for jid in self._order:
                job = self._jobs[jid]
                if job.status == QUEUED:
                    return job
            return None

    def update(self, job_id: str, **fields) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            for k, v in fields.items():
                setattr(job, k, v)
            self._changed()
            return job

    def start(self, job_id: str, device: str) -> Job:
        return self.update(job_id, status=PROCESSING, started_at=time.time(), finished_at=None, stage=STAGES[0],
                           progress=0.0, error=None, device=device)

    def stage(self, job_id: str, stage: str, progress: float = 0.0) -> None:
        self.update(job_id, stage=stage, progress=progress)

    def progress(self, job_id: str, progress: float) -> None:
        # Throttle: only persist/broadcast on visible change.
        with self._lock:
            job = self._jobs[job_id]
            if abs(progress - job.progress) >= 0.005 or progress >= 1.0:
                job.progress = progress
                self._changed()

    def finish(self, job_id: str, **fields) -> Job:
        return self.update(job_id, status=DONE, stage=None, progress=1.0, finished_at=time.time(), **fields)

    def fail(self, job_id: str, error: str, **fields) -> Job:
        return self.update(job_id, status=FAILED, finished_at=time.time(), error=error, **fields)

    def requeue(self, job_id: str, source: Path) -> Job:
        return self.update(job_id, status=QUEUED, stage=None, progress=0.0, error=None, started_at=None,
                           finished_at=None, source=str(source), output_dir=None)

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
            self._order = [i for i in self._order if i != job_id]
            self._changed()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def jobs(self) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in self._order]

    # ---------- derived ----------
    def average_seconds(self) -> float | None:
        """Mean wall-clock seconds of completed jobs (used for ETA)."""
        with self._lock:
            durations = [j.elapsed for j in self._jobs.values() if j.status == DONE and j.elapsed]
        return sum(durations) / len(durations) if durations else None

    def snapshot(self) -> dict:
        with self._lock:
            jobs = [asdict(j) | {"elapsed": j.elapsed} for j in (self._jobs[i] for i in self._order)]
            avg = self.average_seconds()
            queued = sum(1 for j in self._jobs.values() if j.status == QUEUED)
            current = next((j for j in self._jobs.values() if j.status == PROCESSING), None)
            eta = None
            if avg:
                remaining = queued * avg
                if current and current.elapsed is not None:
                    remaining += max(avg - current.elapsed, 0)
                eta = remaining
            return {
                "now": time.time(),
                "info": dict(self.info),
                "stats": {
                    "queued": queued,
                    "done": sum(1 for j in self._jobs.values() if j.status == DONE),
                    "failed": sum(1 for j in self._jobs.values() if j.status == FAILED),
                    "avg_seconds": avg,
                    "eta_seconds": eta,
                },
                "jobs": jobs,
                "log": list(self._log),
            }
