import numpy as np
import pytest

from mashuptech import analyze


def test_fold_tempo():
    assert analyze.fold_tempo(60) == 120
    assert analyze.fold_tempo(200) == 100
    assert analyze.fold_tempo(128) == 128


def test_key_from_chroma_major_and_minor():
    c_major = np.zeros(12)
    for pc in (0, 4, 7, 2, 9, 5, 11):  # C major scale, triad weighted
        c_major[pc] = 1.0
    for pc in (0, 4, 7):
        c_major[pc] += 1.0
    key, camelot, conf = analyze.key_from_chroma(c_major)
    assert key == "C" and camelot == "8B"

    a_minor = np.zeros(12)
    for pc in (9, 11, 0, 2, 4, 5, 7):
        a_minor[pc] = 1.0
    for pc in (9, 0, 4):
        a_minor[pc] += 1.0
    key, camelot, conf = analyze.key_from_chroma(a_minor)
    assert key == "Am" and camelot == "8A"


def _click_track(bpm: float, seconds: float, sr: int) -> np.ndarray:
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float32)
    period = int(sr * 60 / bpm)
    click = (np.random.default_rng(0).standard_normal(int(0.01 * sr)) * np.linspace(1, 0, int(0.01 * sr))).astype(np.float32)
    for start in range(0, n - click.size, period):
        y[start:start + click.size] += click
    return np.stack([y, y])


def test_estimate_bpm_on_click_track():
    pytest.importorskip("librosa")
    sr = 44100
    bpm = analyze.estimate_bpm(_click_track(128, 20, sr), sr)
    assert bpm is not None
    assert abs(bpm - 128) < 2 or abs(bpm - 64) < 2 or abs(bpm - 256) < 4  # fold handles the octave


def test_estimate_bpm_silence_returns_none():
    pytest.importorskip("librosa")
    assert analyze.estimate_bpm(np.zeros((2, 44100)), 44100) is None


def test_estimate_key_on_synthetic_chord():
    pytest.importorskip("librosa")
    sr = 22050
    t = np.arange(sr * 6) / sr
    # A minor: A3, C4, E4, A4 plus a bit of the scale.
    freqs = [220.0, 261.63, 329.63, 440.0, 293.66, 349.23]
    y = sum(np.sin(2 * np.pi * f * t) * (1.0 if i < 4 else 0.3) for i, f in enumerate(freqs)).astype(np.float32)
    y /= np.max(np.abs(y))
    key, camelot, conf = analyze.estimate_key(np.stack([y, y]), sr)
    assert key in ("Am", "C")  # relative keys share pitch content; A minor expected
