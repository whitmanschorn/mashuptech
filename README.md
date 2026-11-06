# mashuptech

A drop-folder pipeline that splits songs into the three samples a mashup is built from:

| stem | what's in it |
| --- | --- |
| **acapella** | vocals only |
| **drums** | drums only |
| **instrumental** | everything else: bass, keys, guitars, synths (no drums, no vocals) |

Drop MP3s into `inbox/`, leave it running overnight, and each song shows up in `outbox/`
as a folder of 320 kbps MP3 stems named with artist, title, BPM, key and the date it was
processed. A local dashboard shows what's cooking.

```
outbox/
  Daft Punk - One More Time [123bpm F#m] [2026-11-05]/
    Daft Punk - One More Time [123bpm F#m] [2026-11-05] - acapella.mp3
    Daft Punk - One More Time [123bpm F#m] [2026-11-05] - drums.mp3
    Daft Punk - One More Time [123bpm F#m] [2026-11-05] - instrumental.mp3
    Daft Punk - One More Time [123bpm F#m] [2026-11-05] - original.mp3
    info.json
```

Separation is done by [Demucs](https://github.com/adefossez/demucs) (`htdemucs_ft`), running
locally. Nothing leaves your machine.

## Setup (macOS, Apple Silicon)

```bash
git clone https://github.com/whitmanschorn/mashuptech.git
cd mashuptech
scripts/setup.sh
```

The setup script installs a **native arm64** Python toolchain via [uv](https://docs.astral.sh/uv/)
(even if your terminal runs under Rosetta), pulls the dependencies, bundles an arm64 `ffmpeg`,
and downloads the model (~330 MB). No Homebrew needed.

Check everything with:

```bash
.venv/bin/python -m mashuptech doctor
```

## Running

```bash
scripts/run.sh
```

This starts the inbox watcher, the worker and the dashboard at <http://localhost:8765>,
and wraps the process in `caffeinate` so the Mac won't sleep mid-batch. Leave it running;
drop files into `inbox/` whenever you like. Stop it with `Ctrl-C`. Anything still queued is
picked up again on the next start.

Accepted inputs: `.mp3 .m4a .aac .wav .aif .aiff .flac .ogg .opus .wma`.

Artist and title come from the file's tags; if those are missing, the filename is parsed as
`Artist - Title.mp3`. Files that fail end up in `inbox/failed/` with a `.error.txt` next to
them, and can be retried from the dashboard.

### One-off use, no dashboard

```bash
.venv/bin/python -m mashuptech process ~/Music/some-song.mp3 another.mp3
```

## The dashboard

- the song being worked on, with its three stem lanes filling as separation progresses
- the queue, with an estimate of time remaining based on songs finished so far
- finished songs with BPM and key, and a "Show in Finder" button
- failed songs with the error and a "Retry" button
- a live log

It updates over Server-Sent Events and falls back to polling.

## Configuration

Everything is an environment variable; put them in front of `scripts/run.sh`.

| variable | default | notes |
| --- | --- | --- |
| `MASHUP_INBOX` | `./inbox` | |
| `MASHUP_OUTBOX` | `./outbox` | point this at an external drive if space is tight |
| `MASHUP_MODEL` | `htdemucs_ft` | `htdemucs` is ~4x faster with slightly rougher stems |
| `MASHUP_DEVICE` | `auto` | `cpu`, `mps`, `cuda` |
| `MASHUP_BITRATE` | `320k` | MP3 bitrate for the stems |
| `MASHUP_PORT` | `8765` | dashboard port |
| `MASHUP_SHIFTS` | `1` | Demucs shift-averaging passes; 0 is faster, 2+ is cleaner and slower |
| `MASHUP_OVERLAP` | `0.25` | chunk overlap |
| `MASHUP_SEGMENT` | model default | chunk length in seconds; lower it if memory is tight |
| `MASHUP_POLL` | `2` | inbox poll interval in seconds |

Example: fast drafts into an external drive.

```bash
MASHUP_MODEL=htdemucs MASHUP_OUTBOX=/Volumes/Samples/stems scripts/run.sh
```

## How it works

```
inbox/ ──watch──▶ queue ──▶ decode ──▶ separate ──▶ analyze ──▶ encode ──▶ outbox/
                            ffmpeg      demucs      librosa      ffmpeg
```

1. **Watch.** The inbox is polled every two seconds. A file is queued once its size has
   stopped changing, so half-copied files are never picked up.
2. **Decode.** `ffmpeg` turns the input into 44.1 kHz stereo float PCM.
3. **Separate.** Demucs produces drums / bass / other / vocals. Vocals become the acapella,
   drums stay drums, and bass + other are summed into the instrumental.
4. **Analyze.** Tempo is estimated from the drum stem (clean onsets) and folded into the
   70–175 BPM range. Key is estimated from the full mix with a Krumhansl–Schmuckler profile
   match; `info.json` also records the Camelot code and a confidence score.
5. **Encode.** Each stem is written as a tagged 320 kbps MP3. The original is moved into the
   output folder, so the inbox drains as work completes.

The model is loaded once per run and reused for every song. Jobs run one at a time; the
model saturates the CPU on its own.

### Speed

On an M2 with `htdemucs_ft`, expect roughly twice a song's length in processing time
(a 4-minute song takes 8–9 minutes). `htdemucs` is about four times faster. An 8-hour
night is 50–60 songs at full quality or 200+ at draft quality.

### Why CPU and not the GPU?

PyTorch's Metal backend still rejects one of the convolutions in Demucs's hybrid
transformer (`Output channels > 65536 not supported at the MPS device`). The pipeline
probes Metal once at startup and falls back to CPU automatically; the dashboard and
`info.json` show which device was used. When a future PyTorch release fixes this, it will
start using the GPU without any changes here.

### Disk

Three stems plus the original is roughly 30 MB per song at 320 kbps. The toolchain and
model need about 2 GB.

## Development

```bash
.venv/bin/python -m pytest
```

Layout:

```
mashuptech/
  __main__.py   CLI: serve | process | warmup | doctor
  config.py     paths and knobs
  watcher.py    inbox polling
  pipeline.py   per-job stages + the worker thread
  separate.py   Demucs wrapper, progress, device fallback
  analyze.py    BPM and key
  audio.py      ffmpeg decode/encode
  naming.py     tags, filename parsing, output names
  state.py      job store, persistence, change feed
  server.py     HTTP + SSE API
  web/          the dashboard (one HTML file, no build step)
```

## Fair use

This tool is for making mashups from music you have the right to use. It performs no
downloading, no sharing and no uploading; it only processes files you put in the inbox.
