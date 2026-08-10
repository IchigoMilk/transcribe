#!/usr/bin/env python3
"""Apply a work-specific correction dictionary to a transcribed TSV (2nd pass).

Design:
- Whisper has no knowledge of a work's proper nouns, so it renders them
    phonetically: a name whose kanji it cannot know comes back as the sounds
    it heard, and a prefix like 聖 becomes セイ. Those errors are systematic
    rather than random: the same wrong spelling recurs across every episode,
    which makes a plain rule table the right tool instead of an LLM pass. The
    table is deterministic, reviewable, and reproducible.
- Corrections run before identify.py. Speaker clustering only reads the
    timestamps, but a human reviewing the tentative labels reads the text, so
    the text should already be correct by the time labels are assigned.
- The untouched Whisper output is preserved under output/raw/ and every run
    re-applies the rules to that pristine copy. Editing the rule table and
    re-running therefore converges on the same result no matter how many times
    it happens, and no correction is ever applied twice on top of itself.
- Every substitution is logged with its rule, so the diff between raw and
    corrected text is always auditable.
- A separate dedupe pass drops Whisper's stuck-decoder repeats (the same
    sentence emitted two or three times back to back). Short interjections
    ("はい", "ええ") are genuinely repeated by different speakers, so only
    lines long enough to be unambiguous artifacts are collapsed.

Rule file: TSV with columns type, pattern, replacement, note.
  type = "literal" for a plain string swap, "regex" for a Python regular
  expression. Rules apply top to bottom, so list longer and more specific
  patterns before the shorter ones they contain.
"""

import argparse
import csv
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

RAW_DIR_NAME = "raw"
TEXT_COLUMN = "text"


class Rule:
    """One substitution, compiled once and reused across every file."""

    def __init__(self, kind: str, pattern: str, replacement: str, note: str, lineno: int):
        self.kind = kind
        self.pattern = pattern
        self.replacement = replacement
        self.note = note
        self.lineno = lineno
        if kind == "literal":
            self.regex = re.compile(re.escape(pattern))
        elif kind == "regex":
            self.regex = re.compile(pattern)
        else:
            raise ValueError(f"line {lineno}: unknown rule type {kind!r} (expected literal or regex)")

    def apply(self, text: str):
        new_text, count = self.regex.subn(self.replacement, text)
        return new_text, count

    def __str__(self) -> str:
        return f"{self.pattern} -> {self.replacement}"


def load_rules(path: Path):
    rules = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"type", "pattern", "replacement"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s): {', '.join(sorted(missing))}")
        for lineno, row in enumerate(reader, start=2):
            pattern = (row.get("pattern") or "").strip()
            # Blank lines and #-prefixed rows keep the table readable as a
            # grouped, commented document rather than a flat list.
            if not pattern or pattern.startswith("#"):
                continue
            rules.append(
                Rule(
                    kind=(row.get("type") or "literal").strip(),
                    pattern=pattern,
                    replacement=(row.get("replacement") or "").strip(),
                    note=(row.get("note") or "").strip(),
                    lineno=lineno,
                )
            )
    return rules


def read_rows(tsv_path: Path):
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader), reader.fieldnames


def write_rows(tsv_path: Path, rows, fieldnames) -> None:
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stash_raw(tsv_path: Path, raw_dir: Path) -> Path:
    """Return the pristine copy of this TSV, creating it on the first run."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / tsv_path.name
    if not raw_path.exists():
        shutil.copy2(tsv_path, raw_path)
    return raw_path


def dedupe_rows(rows, min_len: int, name: str, log_lines: list):
    """Collapse runs of consecutive rows carrying identical text.

    The surviving row keeps the start of the first and the end of the last,
    so the timestamps still cover the same stretch of audio for identify.py.
    """
    if min_len <= 0:
        return rows, 0
    kept = []
    dropped = 0
    for row in rows:
        text = row[TEXT_COLUMN].strip()
        if kept and len(text) >= min_len and kept[-1][TEXT_COLUMN].strip() == text:
            kept[-1]["end"] = row["end"]
            log_lines.append(f"{name}\t{row.get('start', '')}\tDEDUPE\t{text}")
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def existing_speakers(tsv_path: Path) -> dict:
    """Map start time -> speaker from an already-labeled TSV.

    Corrections regenerate from the pristine copy, whose speaker column is
    empty. Speaker labeling is a separate pass and often represents real
    review effort, so refining the rule table and re-running must not throw
    it away. Start times are stable across correction runs, which makes them
    a safe key to carry labels back over.
    """
    if not tsv_path.exists():
        return {}
    rows, fieldnames = read_rows(tsv_path)
    if "speaker" not in (fieldnames or []):
        return {}
    return {r["start"]: r["speaker"] for r in rows if r.get("speaker", "").strip()}


def correct_file(tsv_path: Path, rules, raw_dir: Path, dedupe_min_len: int, log_lines: list):
    carried = existing_speakers(tsv_path)
    raw_path = stash_raw(tsv_path, raw_dir)
    rows, fieldnames = read_rows(raw_path)
    if TEXT_COLUMN not in (fieldnames or []):
        raise ValueError(f"{raw_path}: no {TEXT_COLUMN!r} column")
    for row in rows:
        if not row.get("speaker", "").strip() and row["start"] in carried:
            row["speaker"] = carried[row["start"]]

    counts = Counter()
    for row in rows:
        original = row[TEXT_COLUMN]
        text = original
        for rule in rules:
            text, n = rule.apply(text)
            if n:
                counts[str(rule)] += n
        if text != original:
            log_lines.append(f"{tsv_path.name}\t{row.get('start', '')}\t{original}\t{text}")
        row[TEXT_COLUMN] = text

    rows, dropped = dedupe_rows(rows, dedupe_min_len, tsv_path.name, log_lines)
    write_rows(tsv_path, rows, fieldnames)
    return counts, dropped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply a work-specific correction dictionary to transcribed TSV files"
    )
    parser.add_argument("tsv", nargs="+", type=Path, help="TSV files produced by transcribe.py")
    parser.add_argument("--rules", required=True, type=Path, help="correction rule TSV")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="where pristine copies are kept (default: <tsv dir>/raw)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="write every changed line to this file (default: <tsv dir>/corrections.log)",
    )
    parser.add_argument(
        "--dedupe-min-len",
        type=int,
        default=8,
        help="collapse consecutive identical lines at least this long (0 disables)",
    )
    args = parser.parse_args()

    if not args.rules.exists():
        print(f"Rule file not found: {args.rules}", file=sys.stderr)
        return 1
    try:
        rules = load_rules(args.rules)
    except ValueError as e:
        print(f"Invalid rule file: {e}", file=sys.stderr)
        return 1
    print(f"Loaded {len(rules)} rule(s) from {args.rules}")

    base_dir = args.tsv[0].resolve().parent
    raw_dir = args.raw_dir or base_dir / RAW_DIR_NAME
    log_path = args.log or base_dir / "corrections.log"

    total = Counter()
    total_dropped = 0
    log_lines = ["file\tstart\tbefore\tafter"]
    for tsv_path in args.tsv:
        if not tsv_path.exists():
            print(f"Skipping missing TSV: {tsv_path}", file=sys.stderr)
            continue
        try:
            counts, dropped = correct_file(tsv_path, rules, raw_dir, args.dedupe_min_len, log_lines)
        except ValueError as e:
            # A shell glob easily catches sibling TSVs that are not transcripts
            # (mapping files, for one). Skipping them beats aborting a batch
            # that has already rewritten some of its files.
            print(f"  Skipping {tsv_path.name}: {e}", file=sys.stderr)
            continue
        total.update(counts)
        total_dropped += dropped
        print(f"  {tsv_path.name}: {sum(counts.values())} substitution(s), {dropped} duplicate line(s) dropped")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print()
    print(f"Total duplicate lines dropped: {total_dropped}")
    print(f"Total substitutions: {sum(total.values())}")
    for rule, n in total.most_common():
        print(f"  {n:5d}  {rule}")
    unused = [str(r) for r in rules if str(r) not in total]
    if unused:
        print(f"\nUnused rules ({len(unused)}):")
        for rule in unused:
            print(f"         {rule}")
    print(f"\nRaw copies: {raw_dir}")
    print(f"Change log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
