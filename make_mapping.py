#!/usr/bin/env python3
"""Generate a mapping.tsv template for relabel.py from an identify.py output TSV.

Design:
- Lists every tentative label found in the TSV, most-frequent first, with its
    row count and one example line, so the user can recognize who a cluster
    is without re-reading the whole transcript.
- The name column is intentionally left blank; filling it in by hand is the
    point of this template, not something to guess automatically.
- Candidate names come from profiles/<work-name>/*.yaml, printed for
    reference rather than written into the TSV, to keep the file's columns
    limited to what relabel.py reads.
"""

import argparse
import csv
import sys
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILES = TOOL_ROOT / "profiles"


def list_candidate_names(profiles_dir: Path) -> list[str]:
    """Read the `name` field from each character profile YAML."""
    import yaml

    names = []
    for path in sorted(profiles_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        name = str(data.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def collect_labels(tsv_path: Path) -> dict:
    """Return {label: {count, example}}, example being the first line seen."""
    labels = {}
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get("speaker", "").strip()
            if not label:
                continue
            if label not in labels:
                labels[label] = {"count": 0, "example": row.get("text", "").strip()}
            labels[label]["count"] += 1
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draft a relabel.py mapping template from an identify.py tentatively-labeled TSV"
    )
    parser.add_argument("tsv", type=Path, help="tentatively-labeled TSV")
    parser.add_argument(
        "--profiles", type=Path, default=DEFAULT_PROFILES, help="profiles directory to source candidate names from"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output path (default: <tsv>.mapping.tsv)"
    )
    args = parser.parse_args()

    if not args.tsv.exists():
        print(f"TSV not found: {args.tsv}", file=sys.stderr)
        return 1

    labels = collect_labels(args.tsv)
    if not labels:
        print("No labeled rows found.", file=sys.stderr)
        return 1

    out_path = args.out or args.tsv.with_name(args.tsv.stem + ".mapping.tsv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["label", "name", "count", "example"])
        for label, info in sorted(labels.items(), key=lambda kv: -kv[1]["count"]):
            writer.writerow([label, "", info["count"], info["example"]])

    print(f"-> {out_path}")
    print("Fill in the name column, then run: python relabel.py <tsv> <mapping>")

    if args.profiles.exists():
        names = list_candidate_names(args.profiles)
        if names:
            print()
            print("Candidate names from profiles (also valid: モブ for unlabeled background voices):")
            for name in sorted(names):
                print(f"  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
