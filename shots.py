#!/usr/bin/env python3
"""Grab one video frame per transcript row, for checking speaker labels by eye.

Design:
- Voice clustering cannot see who is on screen; a person reviewing the labels
    can. Each row gets the frame at the midpoint of its own time span, which
    is the moment most likely to show the speaker still talking rather than
    the cut before or after.
- Frames are named after the row's exact `start` string, so a tool holding a
    TSV line can find its image without an index file or any parsing beyond
    reading one column.
- Seeking happens before -i, so ffmpeg jumps to the nearest keyframe instead
    of decoding from the top of the file. That is the whole reason this is
    practical: a full-decode pass per row would take hours for a series.
- Existing frames are skipped, so an interrupted run resumes and a re-run
    after editing timings only fills the gaps.

Output: <out>/<tsv stem>/<start>.jpg
"""

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIDEO_SUFFIXES = (".mkv", ".mp4", ".avi", ".m4v", ".webm", ".ts")


def find_media(stem: str, media_dir: Path):
    for suffix in VIDEO_SUFFIXES:
        candidate = media_dir / (stem + suffix)
        if candidate.exists():
            return candidate
    matches = [p for p in media_dir.iterdir()
               if p.suffix.lower() in VIDEO_SUFFIXES and p.stem == stem]
    return matches[0] if matches else None


def grab(media: Path, when: float, dst: Path, width: int, quality: int) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-ss", f"{when:.2f}", "-i", str(media),
         "-frames:v", "1", "-vf", f"scale={width}:-2",
         "-q:v", str(quality), str(dst)],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and dst.exists()


def process(tsv: Path, media: Path, out_dir: Path, width: int, quality: int,
            workers: int, force: bool):
    with open(tsv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    shot_dir = out_dir / tsv.stem
    shot_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for row in rows:
        start = row.get("start", "").strip()
        end = row.get("end", "").strip()
        if not start:
            continue
        dst = shot_dir / f"{start}.jpg"
        if dst.exists() and not force:
            continue
        try:
            mid = (float(start) + float(end)) / 2 if end else float(start)
        except ValueError:
            continue
        jobs.append((mid, dst))

    if not jobs:
        return 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda j: grab(media, j[0], j[1], width, quality), jobs))
    return sum(results), len(results) - sum(results)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract one low-resolution frame per transcript row"
    )
    parser.add_argument("tsv", nargs="+", type=Path, help="labeled TSV files")
    parser.add_argument("--media-dir", required=True, type=Path,
                        help="directory holding the videos, matched by filename stem")
    parser.add_argument("--out", type=Path, default=Path("shots"),
                        help="where the frames go (default: shots/)")
    parser.add_argument("--width", type=int, default=320,
                        help="frame width in pixels; height follows the aspect ratio")
    parser.add_argument("--quality", type=int, default=6,
                        help="JPEG quality for ffmpeg -q:v, 2 best to 31 worst")
    parser.add_argument("--workers", type=int, default=6,
                        help="parallel ffmpeg processes")
    parser.add_argument("--force", action="store_true", help="re-extract existing frames")
    args = parser.parse_args()

    total_ok = total_bad = 0
    for tsv in args.tsv:
        if not tsv.exists():
            print(f"Skipping missing TSV: {tsv}", file=sys.stderr)
            continue
        media = find_media(tsv.stem, args.media_dir)
        if media is None:
            # A drama CD has no video; that is expected and not an error worth
            # aborting a batch over.
            print(f"{tsv.stem}: no video found, skipping")
            continue
        ok, bad = process(tsv, media, args.out, args.width, args.quality,
                          args.workers, args.force)
        total_ok += ok
        total_bad += bad
        print(f"{tsv.stem}: {ok} frame(s)" + (f", {bad} failed" if bad else ""))

    print(f"\n{total_ok} frame(s) under {args.out}" +
          (f"; {total_bad} failed" if total_bad else ""))
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
