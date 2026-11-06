from datetime import date
from pathlib import Path

from mashuptech import naming


def test_parse_filename_artist_title():
    m = naming.parse_filename(Path("Daft Punk - One More Time.mp3"))
    assert (m.artist, m.title, m.source) == ("Daft Punk", "One More Time", "filename")


def test_parse_filename_strips_track_number_and_handles_en_dash():
    m = naming.parse_filename(Path("03 - Some Artist – Some Title.flac"))
    assert (m.artist, m.title) == ("Some Artist", "Some Title")


def test_parse_filename_without_separator_falls_back():
    m = naming.parse_filename(Path("mystery_track.mp3"))
    assert m.artist == naming.UNKNOWN_ARTIST
    assert m.title == "mystery_track"


def test_sanitize_removes_path_chars():
    assert naming.sanitize('AC/DC: "Back" <in> Black?') == "ACDC Back in Black"
    assert naming.sanitize("   ") == "untitled"


def test_build_basename_full():
    meta = naming.TrackMeta(artist="Daft Punk", title="One More Time")
    assert naming.build_basename(meta, 122.6, "F#m", date(2026, 11, 5)) == "Daft Punk - One More Time [123bpm F#m] [2026-11-05]"


def test_build_basename_without_analysis():
    meta = naming.TrackMeta(artist="A", title="B")
    assert naming.build_basename(meta, None, None, date(2026, 1, 2)) == "A - B [2026-01-02]"


def test_build_basename_caps_length():
    meta = naming.TrackMeta(artist="x" * 200, title="y" * 200)
    name = naming.build_basename(meta, 120, "Am", date(2026, 1, 2))
    assert len(name) < 200


def test_unique_dir(tmp_path):
    (tmp_path / "Song").mkdir()
    (tmp_path / "Song (2)").mkdir()
    assert naming.unique_dir(tmp_path, "Song") == tmp_path / "Song (3)"
    assert naming.unique_dir(tmp_path, "Other") == tmp_path / "Other"


def test_stem_filename():
    assert naming.stem_filename("A - B [2026-01-02]", "drums") == "A - B [2026-01-02] - drums.mp3"
