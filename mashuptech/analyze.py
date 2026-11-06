"""Tempo and key estimation.

Tempo runs on the separated drum stem (clean onsets). Key uses a Krumhansl-Schmuckler
profile match against the mean chroma of the full mix.
"""
from __future__ import annotations

import numpy as np

ANALYSIS_SR = 22050

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl & Kessler (1982) key profiles.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Camelot wheel: handy for harmonic mixing. Index = pitch class.
CAMELOT_MAJOR = {"B": "1B", "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B", "A#": "6B",
                 "F": "7B", "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B"}
CAMELOT_MINOR = {"G#": "1A", "D#": "2A", "A#": "3A", "F": "4A", "C": "5A", "G": "6A",
                 "D": "7A", "A": "8A", "E": "9A", "B": "10A", "F#": "11A", "C#": "12A"}


def _mono(audio: np.ndarray, sr: int, target_sr: int = ANALYSIS_SR) -> np.ndarray:
    import librosa

    y = audio.mean(axis=0) if audio.ndim == 2 else audio
    y = np.asarray(y, dtype=np.float32)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y


def fold_tempo(bpm: float, low: float = 70.0, high: float = 175.0) -> float:
    """Resolve the half/double-time ambiguity into a DJ-friendly range."""
    if bpm <= 0:
        return bpm
    while bpm < low:
        bpm *= 2
    while bpm >= high:
        bpm /= 2
    return bpm


def estimate_bpm(audio: np.ndarray, sr: int, fallback: np.ndarray | None = None) -> float | None:
    """Tempo in BPM from the drum stem, or from `fallback` (the full mix) when the drums are near-silent.

    Uses librosa's global tempo estimate, which weighs the whole onset envelope; it is steadier than
    the median of local estimates on tracks with sparse drums.
    """
    import librosa

    y = _mono(audio, sr)
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    if rms < 0.005 and fallback is not None:
        y = _mono(fallback, sr)
        rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    if y.size == 0 or rms < 1e-4:
        return None
    onset_env = librosa.onset.onset_strength(y=y, sr=ANALYSIS_SR)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=ANALYSIS_SR, start_bpm=120)
    bpm = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(bpm) or bpm <= 0:
        return None
    return round(fold_tempo(bpm), 1)


def key_from_chroma(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    """Return (key name like 'F#m', camelot code, confidence 0..1) from a 12-bin chroma vector."""
    scores = []
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for pc in range(12):
            rotated = np.roll(profile, pc)
            r = np.corrcoef(chroma_mean, rotated)[0, 1]
            scores.append((r, pc, mode))
    scores.sort(reverse=True)
    best_r, pc, mode = scores[0]
    second_r = scores[1][0]
    name = NOTE_NAMES[pc]
    key = name + ("m" if mode == "minor" else "")
    camelot = (CAMELOT_MINOR if mode == "minor" else CAMELOT_MAJOR)[name]
    confidence = float(np.clip((best_r - second_r) * 5, 0, 1)) if np.isfinite(best_r) else 0.0
    return key, camelot, confidence


def estimate_key(audio: np.ndarray, sr: int) -> tuple[str, str, float] | None:
    import librosa

    y = _mono(audio, sr)
    if y.size == 0 or float(np.max(np.abs(y))) < 1e-4:
        return None
    y = librosa.effects.harmonic(y) if y.size < ANALYSIS_SR * 60 * 8 else y
    chroma = librosa.feature.chroma_cqt(y=y, sr=ANALYSIS_SR)
    chroma_mean = chroma.mean(axis=1)
    if not np.any(chroma_mean):
        return None
    return key_from_chroma(chroma_mean)
