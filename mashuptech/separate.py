"""Demucs wrapper: loads the model once, reports progress, falls back from MPS to CPU."""
from __future__ import annotations

import os
import platform
import threading
from typing import Callable

# Must be set before torch is imported: lets unsupported Metal ops run on CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402

ProgressFn = Callable[[float], None]

# Demucs 4-stem output -> the three mashup samples.
STEM_MAP = {"acapella": ("vocals",), "drums": ("drums",), "instrumental": ("bass", "other")}


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available() and platform.machine() == "arm64":
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class StemSeparator:
    def __init__(self, model: str = "htdemucs_ft", device: str = "auto", shifts: int = 1,
                 overlap: float = 0.25, segment: float | None = None,
                 on_device_change: Callable[[str, str | None], None] | None = None):
        self.model_name = model
        self.device = pick_device(device)
        self.shifts = shifts
        self.overlap = overlap
        self.segment = segment
        self.on_device_change = on_device_change
        self._separator = None
        self._lock = threading.Lock()
        self._progress_fn: ProgressFn | None = None
        self._total_chunks = 1
        self._done_chunks = 0
        self.last_fallback_reason: str | None = None

    # ---------- model lifecycle ----------
    def load(self) -> None:
        """Download (first time) and load the model. Safe to call repeatedly."""
        with self._lock:
            if self._separator is not None:
                return
            from demucs.api import Separator

            self._separator = Separator(
                model=self.model_name,
                device=self.device,
                shifts=self.shifts,
                overlap=self.overlap,
                split=True,
                segment=self.segment,
                progress=False,
                callback=self._callback,
            )
            if self.device == "mps":
                self._probe_mps()

    def _probe_mps(self) -> None:
        """Run a 1 s silent clip through the model. PyTorch's Metal backend still rejects some Demucs ops
        ("Output channels > 65536 not supported"); if it fails, we settle on CPU before the first real job."""
        import torch

        try:
            with torch.no_grad():
                self._separator.separate_tensor(torch.zeros(2, self._separator.samplerate), self._separator.samplerate)
        except Exception as exc:  # NotImplementedError, RuntimeError (OOM) ...
            self.last_fallback_reason = str(exc).splitlines()[0][:200]
            self._switch_to_cpu()

    @property
    def samplerate(self) -> int:
        self.load()
        return self._separator.samplerate

    def _switch_to_cpu(self) -> None:
        self.device = "cpu"
        if self._separator is not None:
            self._separator.update_parameter(device="cpu")
            self._separator.model.to("cpu")
        if self.on_device_change:
            self.on_device_change("cpu", self.last_fallback_reason)

    # ---------- progress ----------
    def _expected_chunks(self, n_samples: int) -> int:
        """How many model forward passes separate_tensor will make (bag size x shifts x chunks)."""
        sep = self._separator
        model = sep.model
        segment = sep._segment if getattr(sep, "_segment", None) else getattr(model, "segment", 7.8)
        segment_length = int(sep.samplerate * segment)
        stride = max(int((1 - self.overlap) * segment_length), 1)
        shifts = max(self.shifts, 1)
        # Shifted passes see up to 0.5 s of extra padding, so allow one extra chunk.
        chunks = -(-(n_samples + (int(0.5 * sep.samplerate) if self.shifts else 0)) // stride)
        models = len(getattr(model, "models", [None]))
        return max(models * shifts * chunks, 1)

    def _callback(self, info: dict) -> None:
        if self._progress_fn is None or info.get("state") != "end":
            return
        self._done_chunks += 1
        self._progress_fn(min(self._done_chunks / self._total_chunks, 1.0))

    # ---------- separation ----------
    def separate(self, audio: np.ndarray, sr: int, progress: ProgressFn | None = None) -> dict[str, np.ndarray]:
        """audio: float32 (channels, samples). Returns {'acapella','drums','instrumental'} at self.samplerate."""
        import torch

        self.load()
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        n_samples = int(wav.shape[-1] * self.samplerate / sr)
        self._total_chunks = self._expected_chunks(n_samples)
        self._done_chunks = 0
        self._progress_fn = progress
        try:
            try:
                _, stems = self._separator.separate_tensor(wav, sr)
            except RuntimeError as exc:
                if self.device == "cpu":
                    raise
                # Typical MPS failures: unsupported op, out of memory. Retry on CPU for this and future jobs.
                self.last_fallback_reason = str(exc).splitlines()[0][:200]
                self._switch_to_cpu()
                self._done_chunks = 0
                if progress:
                    progress(0.0)
                _, stems = self._separator.separate_tensor(wav, sr)
        finally:
            self._progress_fn = None

        out: dict[str, np.ndarray] = {}
        for name, sources in STEM_MAP.items():
            mixed = None
            for src in sources:
                t = stems[src].detach().cpu().numpy().astype(np.float32)
                mixed = t if mixed is None else mixed + t
            out[name] = mixed
        if self.device == "mps":
            torch.mps.empty_cache()
        return out
