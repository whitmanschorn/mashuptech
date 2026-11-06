"""Per-job processing: decode -> separate -> analyze -> encode -> finalize."""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
from datetime import date, datetime
from pathlib import Path

from . import __version__, audio, naming
from .config import Config
from .separate import StemSeparator
from .state import JobStore, Job


class Pipeline:
    def __init__(self, config: Config, store: JobStore, separator: StemSeparator):
        self.config = config
        self.store = store
        self.separator = separator

    def process(self, job: Job) -> Path:
        cfg, store = self.config, self.store
        src = Path(job.source)
        if not src.exists():
            raise FileNotFoundError(f"{src.name} disappeared from the inbox")
        timings: dict[str, float] = {}
        t0 = time.time()

        # 1. decode + metadata
        store.stage(job.id, "decode")
        meta = naming.extract_meta(src)
        store.update(job.id, artist=meta.artist, title=meta.title)
        mix = audio.decode(src, sr=audio.SAMPLE_RATE)
        duration = audio.duration_seconds(mix)
        store.update(job.id, duration=duration)
        timings["decode"] = time.time() - t0
        store.log(f"{meta.artist} - {meta.title}: {duration/60:.1f} min, decoded in {timings['decode']:.1f}s", job_id=job.id)

        # 2. separate
        t = time.time()
        store.stage(job.id, "separate")
        stems = self.separator.separate(mix, audio.SAMPLE_RATE, progress=lambda p: store.progress(job.id, p))
        timings["separate"] = time.time() - t
        store.log(f"Separated in {timings['separate']/60:.1f} min on {self.separator.device}", job_id=job.id)

        # 3. analyze
        t = time.time()
        store.stage(job.id, "analyze")
        bpm = key = camelot = None
        key_conf = None
        try:
            from . import analyze

            bpm = analyze.estimate_bpm(stems["drums"], audio.SAMPLE_RATE, fallback=mix)
            key_result = analyze.estimate_key(mix, audio.SAMPLE_RATE)
            if key_result:
                key, camelot, key_conf = key_result
        except Exception as exc:  # analysis is best-effort; never fail the job over it
            store.log(f"Analysis failed, continuing without bpm/key: {exc}", level="warn", job_id=job.id)
        store.update(job.id, bpm=bpm, key=key)
        timings["analyze"] = time.time() - t
        store.log(f"Analysis: {bpm or '?'} bpm, key {key or '?'} ({camelot or '-'})", job_id=job.id)

        # 4. encode
        t = time.time()
        store.stage(job.id, "encode")
        today = date.today()
        basename = naming.build_basename(meta, bpm, key, today)
        out_dir = naming.unique_dir(cfg.outbox, basename)
        out_dir.mkdir(parents=True)
        tags_base = {"artist": meta.artist, "album": meta.album or f"{meta.artist} - {meta.title} (stems)",
                     "date": meta.year or ""}
        if bpm:
            tags_base["TBPM"] = str(int(round(bpm)))
        if key:
            tags_base["TKEY"] = key
        total = len(stems)
        for i, (stem, data) in enumerate(stems.items()):
            store.progress(job.id, i / total)
            tags = tags_base | {"title": f"{meta.title} ({stem})", "comment": f"mashuptech {__version__} {self.separator.model_name}"}
            audio.encode_mp3(data, out_dir / naming.stem_filename(basename, stem), sr=audio.SAMPLE_RATE,
                             bitrate=cfg.mp3_bitrate, tags=tags)
        timings["encode"] = time.time() - t

        # 5. finalize: move the original in and write a manifest
        store.stage(job.id, "finalize", 1.0)
        original_dest = out_dir / f"{basename} - original{src.suffix.lower()}"
        shutil.move(str(src), original_dest)
        info = {
            "artist": meta.artist, "title": meta.title, "album": meta.album, "year": meta.year,
            "metadata_source": meta.source, "source_filename": src.name,
            "bpm": bpm, "key": key, "camelot": camelot, "key_confidence": key_conf,
            "duration_seconds": round(duration, 2),
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            "model": self.separator.model_name, "device": self.separator.device,
            "shifts": self.separator.shifts, "overlap": self.separator.overlap,
            "bitrate": cfg.mp3_bitrate, "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
            "files": {stem: naming.stem_filename(basename, stem) for stem in stems} | {"original": original_dest.name},
            "mashuptech_version": __version__,
        }
        (out_dir / "info.json").write_text(json.dumps(info, indent=2))
        return out_dir

    def handle_failure(self, job: Job, exc: BaseException) -> None:
        """Move the input to inbox/failed with an error note."""
        src = Path(job.source)
        cfg = self.config
        cfg.failed_dir.mkdir(parents=True, exist_ok=True)
        dest = src
        if src.exists() and src.parent != cfg.failed_dir:
            dest = cfg.failed_dir / src.name
            if dest.exists():
                dest = cfg.failed_dir / f"{src.stem} ({int(time.time())}){src.suffix}"
            shutil.move(str(src), dest)
        (dest.with_name(dest.name + ".error.txt")).write_text(
            f"{datetime.now().isoformat(timespec='seconds')}\n{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        )
        self.store.fail(job.id, f"{type(exc).__name__}: {str(exc)[:300]}", source=str(dest))

    def retry(self, job_id: str) -> bool:
        """Move a failed file back into the inbox root and requeue it."""
        job = self.store.get(job_id)
        if not job or job.status != "failed":
            return False
        src = Path(job.source)
        if not src.exists():
            return False
        dest = self.config.inbox / src.name
        if src != dest:
            shutil.move(str(src), dest)
        err = src.with_name(src.name + ".error.txt")
        if err.exists():
            err.unlink()
        self.store.requeue(job_id, dest)
        self.store.log(f"Retrying {dest.name}", job_id=job_id)
        return True


class Worker(threading.Thread):
    """Drains the queue one job at a time."""

    def __init__(self, pipeline: Pipeline, stop: threading.Event):
        super().__init__(name="worker", daemon=True)
        self.pipeline = pipeline
        self.stop_event = stop

    def run(self) -> None:
        store = self.pipeline.store
        sep = self.pipeline.separator
        try:
            store.set_info(status="loading model", model=sep.model_name, device=sep.device)
            sep.load()
            store.set_info(status="idle", device=sep.device)
            store.log(f"Model {sep.model_name} ready on {sep.device}")
        except Exception as exc:
            store.set_info(status="error")
            store.log(f"Could not load model: {exc}", level="error")
            return
        while not self.stop_event.is_set():
            job = store.next_queued()
            if job is None:
                store.set_info(status="idle")
                self.stop_event.wait(1.0)
                continue
            store.set_info(status="working")
            store.start(job.id, sep.device)
            store.log(f"Processing {job.filename}", job_id=job.id)
            try:
                out_dir = self.pipeline.process(job)
                store.finish(job.id, output_dir=str(out_dir), device=sep.device)
                store.set_info(device=sep.device)
                store.log(f"Done: {out_dir.name}", job_id=job.id)
            except Exception as exc:
                store.log(f"Failed {job.filename}: {exc}", level="error", job_id=job.id)
                try:
                    self.pipeline.handle_failure(job, exc)
                except Exception as exc2:
                    store.fail(job.id, f"{exc} (and could not move to failed/: {exc2})")
