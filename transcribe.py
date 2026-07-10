#!/usr/bin/env python3
"""Transcribe audio or video into TSV files for speaker labeling.

Design:
- Whisper does not distinguish speakers, so this tool emits TSV rows with an
    empty speaker column for a human to label. Speaker-diarization models are
    intentionally excluded because their setup cost does not justify the gain.
- TSV supports bulk labeling in editors and spreadsheets. Each row represents
    one utterance and retains timestamps for source review.
- Output defaults to the repository's ignored output/ directory so complete
    transcripts cannot be committed accidentally.

Dependency: pip install faster-whisper
"""

import argparse
import sys
from pathlib import Path

# Keep generated transcripts in the ignored output directory by default.
DEFAULT_OUT = Path(__file__).resolve().parent / "output"


def transcribe(path: Path, model, out_dir: Path) -> Path:
    """Transcribe one input file and return the written TSV path."""
    segments, info = model.transcribe(
        str(path),
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
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="音声・動画を文字起こしして話者ラベリング用 TSV を出力する"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="音声・動画ファイル")
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper モデル名 (VRAM 6GB なら large-v3 の int8 で動く。落ちる場合は medium)",
    )
    parser.add_argument(
        "--compute",
        default="int8",
        help="計算精度 (int8 / int8_float16 / float16)。GTX 1660 SUPER では int8 を既定とする",
    )
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="出力ディレクトリ")
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper is not installed. Run:", file=sys.stderr)
        print("  pip3 install faster-whisper", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute)

    for path in args.inputs:
        if not path.exists():
            print(f"Skipping missing input: {path}", file=sys.stderr)
            continue
        print(f"Transcribing: {path.name} ...")
        out_path = transcribe(path, model, args.out)
        print(f"  -> {out_path}")

    print()
    print("Next: label the TSV speaker column in an editor. See README.md for details.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
