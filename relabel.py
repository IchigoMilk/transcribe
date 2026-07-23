#!/usr/bin/env python3
"""Replace tentative speaker labels in a TSV with real names from a mapping file.

Design:
- identify.py assigns tentative labels (A, B, C...) without knowing real
    names. This script is the final step: the user prepares a small TSV
    mapping each tentative label to a real name, and this applies it in bulk
    instead of a manual find-and-replace in an editor.
- The mapping is a plain label -> name lookup, not limited to single-letter
    labels, so it also works for fixing typos in already-resolved names.
- Rows whose speaker value has no entry in the mapping are left untouched,
    so a partially-filled mapping file can be applied incrementally.
"""

import argparse
import csv
import sys
from pathlib import Path


def read_mapping(path: Path) -> dict:
    """Map label -> name, skipping rows with a blank name.

    Blank names mark labels the user has not decided on yet (as produced by
    make_mapping.py's template), so they must not overwrite the speaker
    column with an empty string.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {
            row["label"]: row["name"]
            for row in reader
            if row.get("name", "").strip()
        }


def read_rows(tsv_path: Path):
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader), reader.fieldnames


def write_rows(tsv_path: Path, rows, fieldnames) -> None:
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bulk-replace a TSV's speaker column with real names, per a mapping file"
    )
    parser.add_argument("tsv", type=Path, help="TSV produced by identify.py or similar")
    parser.add_argument("mapping", type=Path, help="mapping TSV with label and name columns")
    parser.add_argument(
        "--out", type=Path, default=None, help="output TSV (default: overwrite the input TSV)"
    )
    args = parser.parse_args()

    if not args.tsv.exists():
        print(f"TSV not found: {args.tsv}", file=sys.stderr)
        return 1
    if not args.mapping.exists():
        print(f"Mapping file not found: {args.mapping}", file=sys.stderr)
        return 1

    mapping = read_mapping(args.mapping)
    rows, fieldnames = read_rows(args.tsv)

    replaced_counts = {label: 0 for label in mapping}
    unmapped_counts = {}
    for row in rows:
        label = row.get("speaker", "")
        if label in mapping:
            row["speaker"] = mapping[label]
            replaced_counts[label] += 1
        elif label:
            unmapped_counts[label] = unmapped_counts.get(label, 0) + 1

    out_path = args.out or args.tsv
    write_rows(out_path, rows, fieldnames)

    print(f"-> {out_path}")
    for label, count in replaced_counts.items():
        print(f"  {label} -> {mapping[label]}: {count} row(s)")
    if unmapped_counts:
        print("Labels with no mapping entry (left unchanged):", file=sys.stderr)
        for label, count in unmapped_counts.items():
            print(f"  {label}: {count} row(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
