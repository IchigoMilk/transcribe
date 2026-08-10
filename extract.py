#!/usr/bin/env python3
"""Pull one speaker's lines out of labeled TSVs.

Design:
- Reading how a character actually talks means seeing their lines together
    and in order, not scrolling past everyone else's. That is the whole job:
    filter by the speaker column, keep the source position, print.
- The default is every transcript in the output directory, because the
    interesting questions ("does she always end sentences this way?") are
    about a whole series rather than one episode. --episodes narrows it.
- Episode selection matches loosely on purpose: `--episodes 1 5` finds
    sp_01.tsv and sp_05.tsv without the caller having to know the zero
    padding or the filename prefix used by whoever named the files.
- Speaker matching is exact by default. A near-miss silently returning
    nothing is worse than an error, so an unmatched name reports what is
    actually in the files instead.

Output is TSV so it can be piped onward; --plain drops everything but the
text for feeding into a text tool.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent / "output"
SPEAKER_COLUMN = "speaker"
TEXT_COLUMN = "text"


def read_rows(tsv_path: Path):
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader), reader.fieldnames


def episode_key(stem: str) -> str:
    """The trailing number of a filename, unpadded, or the stem itself."""
    m = re.search(r"(\d+)\s*$", stem)
    return str(int(m.group(1))) if m else stem


def select_files(paths, out_dir: Path, episodes):
    if paths:
        files = [Path(p) for p in paths]
    else:
        files = sorted(out_dir.glob("*.tsv"))
        # raw/ holds pre-correction copies and *.mapping.tsv are label tables;
        # neither is a transcript to search.
        files = [f for f in files if not f.name.endswith(".mapping.tsv")]
    if not episodes:
        return files
    wanted = {episode_key(str(e)) for e in episodes} | {str(e) for e in episodes}
    return [f for f in files if episode_key(f.stem) in wanted or f.stem in wanted]


def collect(files, speakers):
    """Return (matched rows, speaker -> line count over all files)."""
    hits = []
    seen = {}
    for path in files:
        if not path.exists():
            print(f"Skipping missing TSV: {path}", file=sys.stderr)
            continue
        rows, fieldnames = read_rows(path)
        if SPEAKER_COLUMN not in (fieldnames or []):
            print(f"Skipping {path.name}: no {SPEAKER_COLUMN!r} column", file=sys.stderr)
            continue
        for row in rows:
            name = (row.get(SPEAKER_COLUMN) or "").strip()
            if not name:
                continue
            seen[name] = seen.get(name, 0) + 1
            if not speakers or name in speakers:
                hits.append((path.stem, row))
    return hits, seen


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract one speaker's lines from labeled TSVs"
    )
    parser.add_argument("tsv", nargs="*", help="TSV files (default: every transcript in --out)")
    parser.add_argument("--speaker", "-s", action="append", default=[],
                        help="speaker to keep; repeat for several")
    parser.add_argument("--episodes", "-e", nargs="+", default=None,
                        help="only these episodes, by trailing number or file stem")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="directory to search when no files are given")
    parser.add_argument("--plain", action="store_true", help="print only the text")
    parser.add_argument("--list", action="store_true", help="list speakers and line counts, then exit")
    args = parser.parse_args()

    files = select_files(args.tsv, args.out, args.episodes)
    if not files:
        print("No transcripts matched.", file=sys.stderr)
        return 1

    hits, seen = collect(files, set(args.speaker))

    if args.list:
        for name, count in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"{count:6d}\t{name}")
        return 0

    if args.speaker:
        missing = [s for s in args.speaker if s not in seen]
        if missing:
            # An exact-match filter that quietly returns nothing looks like
            # "this character never speaks" rather than "you typed it wrong".
            print(f"Speaker not found in {len(files)} file(s): {', '.join(missing)}", file=sys.stderr)
            print("Known speakers: " + ", ".join(sorted(seen)), file=sys.stderr)
            return 1

    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    if not args.plain:
        writer.writerow(["file", "start", "end", "speaker", "text"])
    for stem, row in hits:
        if args.plain:
            print(row.get(TEXT_COLUMN, ""))
        else:
            writer.writerow([stem, row.get("start", ""), row.get("end", ""),
                             row.get(SPEAKER_COLUMN, ""), row.get(TEXT_COLUMN, "")])
    print(f"{len(hits)} line(s) from {len(files)} file(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
