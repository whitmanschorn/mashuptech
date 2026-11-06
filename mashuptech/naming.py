"""Metadata extraction and output naming.

Output layout (one folder per song):
    outbox/Artist - Title [123bpm F#m] [2026-11-05]/
        Artist - Title [123bpm F#m] [2026-11-05] - acapella.mp3
        Artist - Title [123bpm F#m] [2026-11-05] - drums.mp3
        Artist - Title [123bpm F#m] [2026-11-05] - instrumental.mp3
        Artist - Title [123bpm F#m] [2026-11-05] - original.mp3
        info.json
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

UNKNOWN_ARTIST = "Unknown Artist"
MAX_BASENAME = 150  # keep well under filesystem limits once suffixes are added

STEM_NAMES = ("acapella", "drums", "instrumental")


@dataclass
class TrackMeta:
    artist: str
    title: str
    album: str | None = None
    year: str | None = None
    source: str = "filename"  # "tags" or "filename"


def read_tags(path: Path) -> TrackMeta | None:
    """Read artist/title from embedded tags. Returns None if either is missing."""
    try:
        import mutagen
    except ImportError:  # pragma: no cover
        return None
    try:
        f = mutagen.File(path, easy=True)
    except Exception:
        return None
    if not f or not getattr(f, "tags", None):
        return None

    def first(key: str) -> str | None:
        vals = f.tags.get(key)
        if not vals:
            return None
        v = str(vals[0]).strip()
        return v or None

    artist = first("artist") or first("albumartist")
    title = first("title")
    if not artist or not title:
        return None
    year = first("date") or first("originaldate")
    if year:
        year = year[:4]
    return TrackMeta(artist=artist, title=title, album=first("album"), year=year, source="tags")


_FILENAME_SPLIT = re.compile(r"\s+[-–—]\s+")
_TRACK_PREFIX = re.compile(r"^\s*\d{1,3}\s*[-._)]\s*")


def parse_filename(path: Path) -> TrackMeta:
    """Parse 'Artist - Title.mp3' (optionally with a leading track number)."""
    stem = _TRACK_PREFIX.sub("", path.stem).strip()
    parts = _FILENAME_SPLIT.split(stem, maxsplit=1)
    if len(parts) == 2 and all(p.strip() for p in parts):
        return TrackMeta(artist=parts[0].strip(), title=parts[1].strip(), source="filename")
    return TrackMeta(artist=UNKNOWN_ARTIST, title=stem or path.stem, source="filename")


def extract_meta(path: Path) -> TrackMeta:
    return read_tags(path) or parse_filename(path)


_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def sanitize(text: str) -> str:
    """Make a string safe as a single path component on macOS/Windows/Linux."""
    text = _BAD_CHARS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text or "untitled"


def format_key(key: str | None) -> str | None:
    return key or None


def build_basename(meta: TrackMeta, bpm: float | None, key: str | None, when: date | None = None) -> str:
    """'Artist - Title [123bpm F#m] [2026-11-05]'"""
    when = when or date.today()
    artist = sanitize(meta.artist)
    title = sanitize(meta.title)
    base = f"{artist} - {title}"
    # Leave room for the analysis/date tags.
    if len(base) > MAX_BASENAME:
        base = base[:MAX_BASENAME].rstrip(" -")
    tags = []
    if bpm:
        tags.append(f"{int(round(bpm))}bpm")
    if key:
        tags.append(key)
    if tags:
        base += " [" + " ".join(tags) + "]"
    base += f" [{when.isoformat()}]"
    return base


def unique_dir(parent: Path, name: str) -> Path:
    """Return parent/name, or parent/'name (2)' etc. if it already exists."""
    candidate = parent / name
    n = 2
    while candidate.exists():
        candidate = parent / f"{name} ({n})"
        n += 1
    return candidate


def stem_filename(basename: str, stem: str, ext: str = "mp3") -> str:
    return f"{basename} - {stem}.{ext}"
