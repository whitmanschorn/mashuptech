"""Audio decode/encode via ffmpeg pipes. Arrays are float32, shape (channels, samples)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100
CHANNELS = 2


class FfmpegError(RuntimeError):
    pass


_FFMPEG: str | None = None


def ffmpeg_path() -> str:
    """Prefer the bundled arm64 build from imageio-ffmpeg; fall back to whatever is on PATH."""
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    try:
        import imageio_ffmpeg

        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        _FFMPEG = shutil.which("ffmpeg")
    if not _FFMPEG:
        raise FfmpegError("ffmpeg not found. Run scripts/setup.sh (installs the imageio-ffmpeg package).")
    return _FFMPEG


def decode(path: Path, sr: int = SAMPLE_RATE, channels: int = CHANNELS) -> np.ndarray:
    """Decode any audio file to float32 PCM at the given rate. Returns (channels, samples)."""
    cmd = [
        ffmpeg_path(), "-v", "error", "-nostdin",
        "-i", str(path),
        "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(channels), "-ar", str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise FfmpegError(f"ffmpeg failed to decode {path.name}: {proc.stderr.decode(errors='replace').strip()[:500]}")
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if data.size == 0:
        raise FfmpegError(f"{path.name} decoded to zero samples (not an audio file?)")
    frames = data.size // channels
    return np.ascontiguousarray(data[: frames * channels].reshape(frames, channels).T)


def limit_peak(audio: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    """Scale down (never up) so the peak sits under the ceiling. Summed stems can exceed 1.0."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return audio.astype(np.float32, copy=False)


def encode_mp3(audio: np.ndarray, out_path: Path, sr: int = SAMPLE_RATE, bitrate: str = "320k",
               tags: dict[str, str] | None = None) -> None:
    """Encode (channels, samples) float32 to MP3 with libmp3lame and optional ID3 tags."""
    audio = limit_peak(np.asarray(audio, dtype=np.float32))
    channels = audio.shape[0]
    cmd = [
        ffmpeg_path(), "-v", "error", "-nostdin", "-y",
        "-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-i", "-",
        "-codec:a", "libmp3lame", "-b:a", bitrate,
        "-id3v2_version", "3", "-write_id3v1", "1",
    ]
    for k, v in (tags or {}).items():
        if v:
            cmd += ["-metadata", f"{k}={v}"]
    cmd.append(str(out_path))
    proc = subprocess.run(cmd, input=np.ascontiguousarray(audio.T).tobytes(), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise FfmpegError(f"ffmpeg failed to encode {out_path.name}: {proc.stderr.decode(errors='replace').strip()[:500]}")


def duration_seconds(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    return audio.shape[-1] / sr
