#!/usr/bin/env python3
"""Transcribe audio or video into TSV files for speaker labeling.

Design:
- Whisper does not distinguish speakers, so this tool emits TSV rows with an
    empty speaker column for a human to label. Diarization is deliberately
    kept out of this script; identify.py fills the column as an optional
    second pass so this pass stays cheap for users who label by hand.
- TSV supports bulk labeling in editors and spreadsheets. Each row represents
    one utterance and retains timestamps for source review.
- Output defaults to the repository's ignored output/ directory so complete
    transcripts cannot be committed accidentally.
- Inputs are pre-decoded through ffmpeg into a normalized mono 16kHz WAV temp
    file before reaching Whisper. This accepts any container/codec ffmpeg
    understands (audio or video) instead of relying on faster-whisper's own
    demuxer, which is pickier about formats.

Dependencies: pip install faster-whisper, and ffmpeg on PATH.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Keep generated transcripts in the ignored output directory by default.
DEFAULT_OUT = Path(__file__).resolve().parent / "output"

# Whisper's own encoder expects 16kHz mono audio, so normalize here rather
# than depend on faster-whisper's internal resampling for every format.
FFMPEG_SAMPLE_RATE = "16000"


class FFmpegError(RuntimeError):
    """Raised when ffmpeg fails to decode one input file.

    Kept distinct from other RuntimeErrors so the per-file skip-and-continue
    logic in main() cannot accidentally swallow unrelated failures (e.g. a
    broken CUDA setup, which affects every file and should abort loudly
    instead of being reported once per input as if it were an ffmpeg issue).
    """


def ensure_cuda_library_path() -> None:
    """Re-exec this process with pip-installed cuBLAS/cuDNN on LD_LIBRARY_PATH.

    glibc's dynamic linker only reads LD_LIBRARY_PATH once at process start,
    so setting os.environ after the interpreter is already running has no
    effect on ctranslate2's later dlopen() calls. Re-exec is therefore the
    only way to make a pip-only (no system CUDA) install work without asking
    users to export the variable by hand every time. No-op if the nvidia
    pip packages are not installed (e.g. CPU-only use) or already applied.
    """
    if os.environ.get("_TRANSCRIBE_CUDA_LIBRARY_PATH_SET"):
        return
    try:
        import nvidia.cublas
        import nvidia.cudnn
    except ImportError:
        return
    lib_dirs = [
        str(Path(nvidia.cublas.__path__[0]) / "lib"),
        str(Path(nvidia.cudnn.__path__[0]) / "lib"),
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    os.environ["_TRANSCRIBE_CUDA_LIBRARY_PATH_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def extract_audio(path: Path, tmp_dir: Path) -> Path:
    """Decode any ffmpeg-readable audio/video input into a mono 16kHz WAV temp file."""
    wav_path = tmp_dir / (path.stem + ".wav")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(path),
            "-vn",
            "-ac", "1",
            "-ar", FFMPEG_SAMPLE_RATE,
            "-f", "wav",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffmpeg failed for {path}:\n{result.stderr}")
    return wav_path


def transcribe(path: Path, model, out_dir: Path, tmp_dir: Path) -> Path:
    """Transcribe one input file and return the written TSV path."""
    wav_path = extract_audio(path, tmp_dir)
    segments, info = model.transcribe(
        str(wav_path),
        language="ja",
        # Remove silent and music-only segments to reduce false lyric matches.
        vad_filter=True,
    )
    out_path = out_dir / (path.stem + ".tsv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("start\tend\tspeaker\ttext\n")
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            f.write(f"{seg.start:.2f}\t{seg.end:.2f}\t\t{text}\n")
    wav_path.unlink()
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video and output a TSV for speaker labeling"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="audio/video files")
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name (large-v3 in int8 fits 6GB VRAM; use medium if it fails)",
    )
    parser.add_argument(
        "--compute",
        default="int8",
        help="compute type (int8 / int8_float16 / float16); int8 is the default for a GTX 1660 SUPER",
    )
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args()

    ensure_cuda_library_path()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper is not installed. Run:", file=sys.stderr)
        print("  pip3 install faster-whisper", file=sys.stderr)
        return 1

    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not installed or not on PATH.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute)

    with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for path in args.inputs:
            if not path.exists():
                print(f"Skipping missing input: {path}", file=sys.stderr)
                continue
            print(f"Transcribing: {path.name} ...")
            try:
                out_path = transcribe(path, model, args.out, tmp_dir)
            except FFmpegError as e:
                print(f"  Skipping (ffmpeg error): {e}", file=sys.stderr)
                continue
            print(f"  -> {out_path}")

    print()
    print("Next: label the TSV speaker column in an editor. See README.md for details.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
