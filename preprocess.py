#!/usr/bin/env python3
"""Turn video into speech-only mono WAV ready for transcription.

Design:
- Two stages, in this order: ffmpeg decodes the audio track to the mono
    16kHz WAV Whisper's encoder expects, then a source-separation model
    keeps the vocal stem and discards everything else.
- Music is the reason to do this at all. Over background music Whisper
    stops emitting utterances and instead returns one enormous segment
    holding a fragment of what was said, so whole minutes of dialogue
    disappear rather than come back wrong. Removing the music restores
    the utterance boundaries the rest of the pipeline depends on.
- Separation runs on the full-band 44.1kHz stereo audio, not on the 16kHz
    mono file, because the model was trained on music and throws away
    accuracy if fed something already downmixed and band-limited. The
    downmix therefore happens after separation, not before.
- Output is written next to the input under a directory of its own so a
    long batch can be resumed: an input whose WAV already exists is
    skipped unless --force is given.

Dependencies: ffmpeg on PATH, and `pip install demucs` for --separate.
"""

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path

FFMPEG_SAMPLE_RATE = "16000"
# htdemucs is the current default and the best of the bundled models at
# pulling a voice out of a dense mix; two-stems skips computing the drum
# and bass stems that would only be thrown away.
DEMUCS_MODEL = "htdemucs"


class PreprocessError(RuntimeError):
    """One input failed. Kept distinct so a batch can skip and continue."""


def run(cmd, what):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PreprocessError(f"{what} failed:\n{result.stderr[-2000:]}")
    return result


def duration_of(wav_path: Path) -> float:
    with wave.open(str(wav_path)) as w:
        return w.getnframes() / w.getframerate()


def decode_full_band(src: Path, dst: Path) -> Path:
    """Audio track only, untouched rate and channels, for the separator."""
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-c:a", "pcm_s16le", str(dst)],
        f"ffmpeg decode of {src.name}")
    return dst


def downmix(src: Path, dst: Path) -> Path:
    """Mono 16kHz, which is what Whisper's encoder consumes."""
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1",
         "-ar", FFMPEG_SAMPLE_RATE, "-c:a", "pcm_s16le", str(dst)],
        f"ffmpeg downmix of {src.name}")
    return dst


def separate_vocals(src: Path, work: Path, device: str, jobs: int) -> Path:
    """Return the vocal stem of src, written somewhere under work."""
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
           "-n", DEMUCS_MODEL, "-o", str(work), "-d", device]
    if jobs > 1 and device == "cpu":
        cmd += ["-j", str(jobs)]
    cmd.append(str(src))
    run(cmd, f"demucs on {src.name}")
    produced = list((work / DEMUCS_MODEL).glob(f"{src.stem}/vocals.*"))
    if not produced:
        raise PreprocessError(f"demucs produced no vocal stem for {src.name}")
    return produced[0]


def discard_intermediates(src: Path, tmp_dir: Path) -> None:
    """Drop this input's scratch files.

    The full-band decode and the separated stems run close to a gigabyte per
    episode, so keeping them for a whole series would cost more disk than the
    source video. They are reproducible from the input, so nothing is lost.
    """
    full = tmp_dir / (src.stem + ".full.wav")
    full.unlink(missing_ok=True)
    stem_dir = tmp_dir / "demucs" / DEMUCS_MODEL / (src.stem + ".full")
    if stem_dir.is_dir():
        shutil.rmtree(stem_dir, ignore_errors=True)


def preprocess(src: Path, out_dir: Path, tmp_dir: Path, separate: bool,
               device: str, jobs: int, keep: bool = False) -> Path:
    out_path = out_dir / (src.stem + ".wav")
    try:
        full = decode_full_band(src, tmp_dir / (src.stem + ".full.wav"))
        if separate:
            stem = separate_vocals(full, tmp_dir / "demucs", device, jobs)
            downmix(stem, out_path)
        else:
            downmix(full, out_path)
    finally:
        if not keep:
            discard_intermediates(src, tmp_dir)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract the audio track, drop the music, and write mono 16kHz WAV"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="audio/video files")
    parser.add_argument("--out", type=Path, default=Path("audio"),
                        help="directory for the finished WAVs (default: audio/)")
    parser.add_argument("--tmp", type=Path, default=None,
                        help="scratch directory (default: <out>/.work)")
    parser.add_argument("--no-separate", action="store_true",
                        help="skip source separation; only decode and downmix")
    parser.add_argument("--device", default="cuda",
                        help="demucs device: cuda / cpu (default: cuda)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="demucs parallel jobs, CPU only")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="keep the full-band decode and separated stems (roughly 1GB per input)")
    parser.add_argument("--force", action="store_true",
                        help="redo inputs whose output already exists")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.tmp or (args.out / ".work")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for src in args.inputs:
        if not src.exists():
            print(f"Skipping missing input: {src}", file=sys.stderr)
            failed += 1
            continue
        out_path = args.out / (src.stem + ".wav")
        if out_path.exists() and not args.force:
            print(f"{src.name}: already done, skipping")
            continue
        print(f"{src.name}: decoding{'' if args.no_separate else ' and separating'} ...", flush=True)
        try:
            written = preprocess(src, args.out, tmp_dir, not args.no_separate,
                                 args.device, args.jobs, args.keep_intermediates)
        except PreprocessError as e:
            # One unreadable input must not cost the rest of a long batch.
            print(f"  {e}", file=sys.stderr)
            failed += 1
            continue
        print(f"  -> {written} ({duration_of(written):.0f}s)")

    print()
    print(f"Wrote to {args.out}. Pass these WAVs to transcribe.py in place of the originals.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
