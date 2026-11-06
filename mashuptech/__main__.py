"""CLI: `mashuptech serve` (default), `process FILE...`, `warmup`, `doctor`."""
from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import platform
import shutil
import signal
import sys
import threading
import time
import webbrowser
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from . import __version__  # noqa: E402
from .config import Config  # noqa: E402

log = logging.getLogger("mashuptech")


def _setup_logging(cfg: Config, verbose: bool) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S")
    root = logging.getLogger("mashuptech")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(cfg.log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s"))
    root.addHandler(sh)
    root.addHandler(fh)


def _build(cfg: Config):
    from .pipeline import Pipeline
    from .separate import StemSeparator
    from .state import JobStore

    cfg.ensure_dirs()
    store = JobStore(cfg.state_file)
    separator = StemSeparator(model=cfg.model, device=cfg.device, shifts=cfg.shifts, overlap=cfg.overlap,
                              segment=cfg.segment)
    def _device_changed(device: str, reason: str | None) -> None:
        store.set_info(device=device)
        store.log(f"Metal (MPS) unavailable for this model ({reason or 'unknown error'}); using CPU for this run",
                  level="warn")

    separator.on_device_change = _device_changed
    store.set_info(status="starting", model=cfg.model, device=separator.device, config=cfg.as_dict(),
                   version=__version__, pid=os.getpid())
    return store, separator, Pipeline(cfg, store, separator)


def cmd_serve(cfg: Config, args) -> int:
    from .pipeline import Worker
    from .server import start_server
    from .watcher import InboxWatcher

    store, separator, pipeline = _build(cfg)
    stop = threading.Event()
    # Re-queue anything sitting in the inbox from a previous run before the watcher sees it.
    worker = Worker(pipeline, stop)
    watcher = InboxWatcher(cfg, store, stop)
    server = start_server(cfg, store, pipeline)
    url = f"http://localhost:{cfg.port}"
    store.log(f"mashuptech {__version__} watching {cfg.inbox} -> {cfg.outbox}; dashboard at {url}")
    worker.start()
    watcher.start()
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    def _stop(*_):
        store.log("Shutting down (jobs in progress will restart from scratch next launch)")
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    while not stop.is_set():
        time.sleep(0.5)
    server.shutdown()
    return 0


def cmd_process(cfg: Config, args) -> int:
    """Copy the given files into the inbox and run the worker until the queue is empty (no dashboard)."""
    from .pipeline import Worker

    store, separator, pipeline = _build(cfg)
    for f in args.files:
        src = Path(f).expanduser().resolve()
        if not src.exists():
            log.error("No such file: %s", src)
            return 2
        dest = cfg.inbox / src.name
        if src != dest:
            shutil.copy2(src, dest)
        job = store.add(dest)
        if job:
            store.log(f"Queued {dest.name}", job_id=job.id)
    stop = threading.Event()
    worker = Worker(pipeline, stop)
    worker.start()
    try:
        while worker.is_alive():
            if store.next_queued() is None and store.info.get("status") == "idle":
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    stop.set()
    worker.join(timeout=5)
    failed = [j for j in store.jobs() if j.status == "failed"]
    for j in store.jobs():
        if j.status == "done":
            print(f"OK    {j.output_dir}")
    for j in failed:
        print(f"FAIL  {j.filename}: {j.error}")
    return 1 if failed else 0


def cmd_warmup(cfg: Config, args) -> int:
    from .separate import StemSeparator

    sep = StemSeparator(model=cfg.model, device=cfg.device)
    print(f"Loading {cfg.model} on {sep.device} (downloads on first run)...")
    sep.load()
    print("Model ready.")
    return 0


def cmd_doctor(cfg: Config, args) -> int:
    ok = True

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  [{'ok' if good else 'XX'}] {label}" + (f": {detail}" if detail else ""))

    print(f"mashuptech {__version__}")
    machine = platform.machine()
    check("Python", True, f"{platform.python_version()} ({machine})")
    if sys.platform == "darwin":
        check("Native arm64 Python (not Rosetta)", machine == "arm64",
              "" if machine == "arm64" else "running under Rosetta; GPU unavailable. Re-run scripts/setup.sh")
    try:
        import torch

        mps = torch.backends.mps.is_available()
        check("torch", True, torch.__version__)
        check("Metal (MPS) available", mps or sys.platform != "darwin", "yes" if mps else "no, will use CPU")
    except Exception as exc:
        check("torch importable", False, str(exc))
    try:
        import demucs  # noqa: F401

        check("demucs", True)
    except Exception as exc:
        check("demucs importable", False, str(exc))
    try:
        from .audio import ffmpeg_path

        check("ffmpeg", True, ffmpeg_path())
    except Exception as exc:
        check("ffmpeg", False, str(exc))
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    torch_cache = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    wanted = cfg.model.replace("_", "-").lower()
    hits = [p.name for p in hf_cache.glob("models--*") if wanted in p.name.lower()] if hf_cache.exists() else []
    hits += [p.name for p in torch_cache.glob("*.th")] if torch_cache.exists() else []
    check(f"model {cfg.model} cached", bool(hits), ", ".join(hits) if hits else "not yet (run: .venv/bin/python -m mashuptech warmup)")
    for name, p in (("inbox", cfg.inbox), ("outbox", cfg.outbox)):
        check(f"{name} writable", os.access(p, os.W_OK) if p.exists() else os.access(p.parent, os.W_OK), str(p))
    usage = shutil.disk_usage(cfg.outbox if cfg.outbox.exists() else Path.home())
    free_gb = usage.free / 1e9
    check("Free disk for outbox", free_gb > 5, f"{free_gb:.1f} GB free (~30 MB per song)")
    print("All good." if ok else "Some checks failed.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mashuptech", description="Split songs into acapella / drums / instrumental.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd")
    p_serve = sub.add_parser("serve", help="Watch the inbox, process files, serve the dashboard (default)")
    p_serve.add_argument("--no-browser", action="store_true", help="Don't open the dashboard in a browser")
    p_proc = sub.add_parser("process", help="Process one or more files now, without the dashboard")
    p_proc.add_argument("files", nargs="+")
    sub.add_parser("warmup", help="Download/load the model and exit")
    sub.add_parser("doctor", help="Check the environment")
    args = parser.parse_args(argv)
    if args.cmd is None:
        args.cmd, args.no_browser = "serve", False

    cfg = Config()
    _setup_logging(cfg, args.verbose)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)  # kill -USR1 <pid> dumps every thread's stack
    return {"serve": cmd_serve, "process": cmd_process, "warmup": cmd_warmup, "doctor": cmd_doctor}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
