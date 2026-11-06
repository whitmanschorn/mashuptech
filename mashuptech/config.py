"""Runtime configuration. Every knob can be overridden with a MASHUP_* environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".aac", ".wav", ".aif", ".aiff", ".flac", ".ogg", ".opus", ".wma"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass
class Config:
    inbox: Path = field(default_factory=lambda: _env_path("MASHUP_INBOX", ROOT / "inbox"))
    outbox: Path = field(default_factory=lambda: _env_path("MASHUP_OUTBOX", ROOT / "outbox"))
    data_dir: Path = field(default_factory=lambda: _env_path("MASHUP_DATA", ROOT / "data"))
    # Demucs model. htdemucs_ft = best quality (4 fine-tuned models bagged, ~4x slower).
    # htdemucs = fast drafts.
    model: str = field(default_factory=lambda: os.environ.get("MASHUP_MODEL", "htdemucs_ft"))
    # "auto" picks mps on Apple Silicon, cuda if present, else cpu.
    device: str = field(default_factory=lambda: os.environ.get("MASHUP_DEVICE", "auto"))
    mp3_bitrate: str = field(default_factory=lambda: os.environ.get("MASHUP_BITRATE", "320k"))
    port: int = field(default_factory=lambda: int(os.environ.get("MASHUP_PORT", "8765")))
    poll_interval: float = field(default_factory=lambda: float(os.environ.get("MASHUP_POLL", "2")))
    # Demucs quality/speed knobs. shifts>0 averages several randomly shifted passes (slower, cleaner).
    shifts: int = field(default_factory=lambda: int(os.environ.get("MASHUP_SHIFTS", "1")))
    overlap: float = field(default_factory=lambda: float(os.environ.get("MASHUP_OVERLAP", "0.25")))
    # Segment length in seconds for chunked inference; lower = less memory.
    segment: float | None = field(
        default_factory=lambda: float(os.environ["MASHUP_SEGMENT"]) if os.environ.get("MASHUP_SEGMENT") else None
    )

    @property
    def failed_dir(self) -> Path:
        return self.inbox / "failed"

    @property
    def state_file(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def log_file(self) -> Path:
        return self.data_dir / "mashuptech.log"

    def ensure_dirs(self) -> None:
        for p in (self.inbox, self.outbox, self.data_dir, self.failed_dir):
            p.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict:
        return {
            "inbox": str(self.inbox),
            "outbox": str(self.outbox),
            "model": self.model,
            "device": self.device,
            "mp3_bitrate": self.mp3_bitrate,
            "port": self.port,
            "shifts": self.shifts,
            "overlap": self.overlap,
        }
