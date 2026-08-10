#!/usr/bin/env python3
"""Apply hand-reviewed per-line overrides to corrected, speaker-labeled TSVs.

Design:
- correct.py fixes what a rule table can fix: spellings that recur across the
    whole series. What it cannot fix is a line that is wrong only once, or a
    speaker label that voice clustering got wrong. Those need a human reading
    the scene, and the result is data, not a rule.
- Overrides are keyed by (file, start) rather than row number, because start
    times survive re-running the correction pass while row numbers do not.
- Every override is optional per field: a blank speaker leaves the speaker
    alone, a blank text leaves the text alone. This keeps the file readable
    as "what a reviewer changed" instead of a full restatement of each row.
- Applying is idempotent and order-independent, so the file can be extended
    episode by episode and re-applied from scratch at any time.

Override file: TSV with columns file, start, speaker, text, note.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def read_overrides(path: Path):
    by_file = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"file", "start", "speaker", "text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s): {', '.join(sorted(missing))}")
        for row in reader:
            name = (row.get("file") or "").strip()
            start = (row.get("start") or "").strip()
            if not name or name.startswith("#") or not start:
                continue
            by_file[name][start] = (
                (row.get("speaker") or "").strip(),
                (row.get("text") or "").strip(),
            )
    return by_file


def apply_file(tsv_path: Path, overrides: dict):
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows, fieldnames = list(reader), reader.fieldnames

    changed_speaker = changed_text = 0
    seen = set()
    for row in rows:
        key = row["start"]
        if key not in overrides:
            continue
        seen.add(key)
        speaker, text = overrides[key]
        if speaker and row["speaker"] != speaker:
            row["speaker"] = speaker
            changed_speaker += 1
        if text and row["text"] != text:
            row["text"] = text
            changed_text += 1

    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    unmatched = sorted(set(overrides) - seen)
    return changed_speaker, changed_text, unmatched


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply per-line speaker/text overrides to labeled TSVs"
    )
    parser.add_argument("tsv", nargs="+", type=Path, help="TSV files to fix")
    parser.add_argument("--overrides", required=True, type=Path, help="override TSV")
    args = parser.parse_args()

    if not args.overrides.exists():
        print(f"Override file not found: {args.overrides}", file=sys.stderr)
        return 1
    by_file = read_overrides(args.overrides)

    total_s = total_t = 0
    problems = 0
    for tsv_path in args.tsv:
        if not tsv_path.exists():
            print(f"Skipping missing TSV: {tsv_path}", file=sys.stderr)
            continue
        overrides = by_file.get(tsv_path.name, {})
        if not overrides:
            continue
        s, t, unmatched = apply_file(tsv_path, overrides)
        total_s += s
        total_t += t
        print(f"  {tsv_path.name}: {s} speaker, {t} text")
        if unmatched:
            problems += len(unmatched)
            # A start time that matches no row means the override was written
            # against a stale transcript; silently ignoring it would hide a
            # correction the reviewer believed had been applied.
            print(f"    WARNING: {len(unmatched)} override(s) matched no row: "
                  f"{', '.join(unmatched[:8])}", file=sys.stderr)

    print(f"\nApplied {total_s} speaker and {total_t} text override(s).")
    if problems:
        print(f"{problems} override(s) matched nothing - check the start times.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
