#!/usr/bin/env python3
"""Extract per-character speech-pattern material from labeled TSV files.

Design:
- Reports provide frequency tables that can be transferred into a profile's
    speech fields, replacing assumptions with transcript-based evidence.
- Ending detection uses a finite pattern table because fixed endings are more
    useful for character voice than a morphological analyzer. Unknown endings
    fall back to the final two characters for review.
- The target project's profiles are the source of truth for the cast. Passing
    --profiles avoids maintaining a second cast list in this repository.
- Reports contain transcript excerpts, so they are written under ignored
    output/ by default.

Dependency: PyYAML
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILES = TOOL_ROOT / "profiles"
DEFAULT_OUT = TOOL_ROOT / "output" / "reports"

# Count these first-person pronouns to infer each speaker's preferred form.
FIRST_PERSONS = ["わたくし", "わたし", "私", "あたし", "うち", "僕", "ぼく", "俺"]

# Check longer endings first to avoid matching a shorter suffix prematurely.
# Fixed endings distinguish character voices; unknown endings use a two-character fallback.
SENTENCE_ENDINGS = [
    "でございますわ", "でございます", "ではありませんか",
    "ですわよね", "ですわよ", "ですわね", "ですわ", "ですのよ", "ですのね", "ですの",
    "でしょうか", "でしょうね", "でしょう", "でしょ",
    "ですよね", "ですよ", "ですね", "ですか", "です",
    "ますでしょうか", "ましょうか", "ましょう", "ませんか", "ませんわ", "ません",
    "ますか", "ますね", "ますよ", "ます", "ませ",
    "かしらね", "かしら", "こと",
    "わよね", "わよ", "わね", "だわ",
    "のよね", "のよ", "のね", "なのよ", "なのね", "なのだ", "なんだ", "なの",
    "だもんね", "だもん", "もんね", "もん",
    "じゃないか", "じゃない", "じゃんか", "じゃん",
    "だよね", "だよな", "だよ", "だね", "だろう", "だろ",
    "だってば", "だって", "ってば", "って",
    "かなあ", "かな", "かい", "かよ", "のか", "っけ",
    "なさい", "ください", "ちょうだい", "頂戴",
    "のだ", "んだ", "のさ", "さ", "ぞ", "ぜ", "わ", "な", "ね", "よ", "の", "か",
]

# Detect forms of address by matching a name followed by one of these suffixes.
HONORIFICS = (
    "お姉さま|お姉様|おねえさま|お姉ちゃん|姉さま|さま|様|さん|ちゃん|くん|君|"
    "先輩|せんぱい|会長|部長|先生|殿"
)

# Do not split on commas because ending classification operates on sentences.
SENTENCE_SPLIT = re.compile(r"[。！？!?…]+")


def load_cast(profiles_dir: Path) -> dict[str, list[str]]:
    """Build {canonical name: variants} for forms-of-address detection.

    Include profile names and aliases. Also derive given names from full names,
    because dialogue commonly addresses a character by given name plus suffix.
    """
    import yaml

    cast = {}
    for path in sorted(profiles_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        name = str(data.get("name", "")).strip()
        if not name:
            continue
        variants = {name}
        for alias in data.get("aliases", []) or []:
            alias = str(alias).strip()
            # Split space-separated reading aliases into separate variants.
            variants.update(a for a in re.split(r"[\s　]+", alias) if len(a) >= 2)
        # Derive given names only from full names to avoid single-character false positives.
        for full in list(variants):
            m = re.fullmatch(r"([一-龥]{1,2})([一-龥々]{1,3})", full)
            if m and len(full) >= 3:
                variants.add(m.group(2))
        cast[name] = sorted(variants, key=len, reverse=True)
    return cast


def load_tsv(path: Path) -> list[dict]:
    """Read a labeled TSV while retaining empty-speaker rows as unlabeled."""
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header.startswith("start"):
            print(f"Warning: {path} does not start with the expected header.", file=sys.stderr)
        for lineno, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                print(f"Warning: skipping {path}:{lineno}; expected at least four columns.", file=sys.stderr)
                continue
            rows.append(
                {
                    "source": f"{path.stem}:{lineno}",
                    "speaker": cols[2].strip(),
                    "text": cols[3].strip(),
                }
            )
    return rows


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def classify_ending(sentence: str) -> str:
    """Classify an ending or return a marked two-character fallback."""
    for pattern in SENTENCE_ENDINGS:
        if sentence.endswith(pattern):
            return pattern
    return sentence[-2:] + " ?"


def analyze_speaker(rows: list[dict], cast: dict[str, list[str]]):
    """Aggregate ending, first-person, and form-of-address frequencies."""
    endings = Counter()
    first_persons = Counter()
    mentions = Counter()  # (canonical target, observed form of address) -> count
    mention_re = {
        canon: re.compile(
            "(" + "|".join(map(re.escape, variants)) + f")({HONORIFICS})?"
        )
        for canon, variants in cast.items()
    }

    for row in rows:
        text = row["text"]
        # Prevent watakushi from also being counted as watashi.
        without_watakushi = text.replace("わたくし", "")
        for fp in FIRST_PERSONS:
            target = without_watakushi if fp == "わたし" else text
            first_persons[fp] += target.count(fp)
        for canon, pat in mention_re.items():
            for m in pat.finditer(text):
                # One-character names collide with ordinary words, so require a suffix.
                if len(m.group(1)) == 1 and not m.group(2):
                    continue
                called = m.group(1) + (m.group(2) or "")
                mentions[(canon, called)] += 1
        for sentence in split_sentences(text):
            endings[classify_ending(sentence)] += 1

    return endings, first_persons, mentions


def sample_lines(rows: list[dict], endings: Counter, n: int = 8) -> list[dict]:
    """Select representative lines containing the most frequent endings."""
    picked = []
    seen = set()
    top_endings = [e for e, _ in endings.most_common(n) if not e.endswith("?")]
    for ending in top_endings:
        for row in rows:
            if row["source"] in seen:
                continue
            if any(s.endswith(ending) for s in split_sentences(row["text"])):
                picked.append(row)
                seen.add(row["source"])
                break
        if len(picked) >= n:
            break
    return picked


def write_report(out_dir: Path, speaker: str, rows, endings, first_persons, mentions):
    out_path = out_dir / f"{speaker}.md"
    lines = [f"# {speaker} の口調テンプレート素材", ""]
    lines += [f"- 台詞数: {len(rows)}", ""]

    lines += ["## 一人称 (出現回数)", ""]
    fp_items = [(fp, c) for fp, c in first_persons.most_common() if c > 0]
    if fp_items:
        lines += [f"- {fp}: {c}" for fp, c in fp_items]
    else:
        lines += ["- (検出なし)"]
    lines += [""]

    lines += ["## 文末表現 (頻度順、? はパターン表外)", ""]
    lines += [f"- {e}: {c}" for e, c in endings.most_common(25)]
    lines += [""]

    lines += ["## 呼称 (誰を何と呼ぶか)", ""]
    if mentions:
        by_target = defaultdict(list)
        for (canon, called), c in mentions.items():
            by_target[canon].append((called, c))
        for canon in sorted(by_target):
            calls = sorted(by_target[canon], key=lambda x: -x[1])
            rendered = ", ".join(f"{called} ({c})" for called, c in calls)
            lines += [f"- {canon}: {rendered}"]
    else:
        lines += ["- (検出なし)"]
    lines += [""]

    lines += ["## 実例 (頻出語尾を含む台詞)", ""]
    lines += [f"- {r['text']}  `{r['source']}`" for r in sample_lines(rows, endings)]
    lines += [""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ラベル済み TSV からキャラ別の口調テンプレート素材を抽出する"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="ラベル済み TSV (複数可)")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--min-lines", type=int, default=5,
        help="台詞数がこれ未満の話者はレポートを作らない (モブ除外の閾値)",
    )
    args = parser.parse_args()

    cast = load_cast(args.profiles)
    all_rows = []
    for path in args.inputs:
        all_rows += load_tsv(path)

    unlabeled = [r for r in all_rows if not r["speaker"]]
    labeled = [r for r in all_rows if r["speaker"]]
    print(f"Loaded: {len(all_rows)} rows (labeled {len(labeled)} / unlabeled {len(unlabeled)})")

    # Warn about labels outside the profile cast because they may be typos.
    # The reserved mob label marks rows intentionally excluded from analysis.
    known = {"モブ"} | set(cast)
    for canon, variants in cast.items():
        known.update(variants)
    unknown = sorted({r["speaker"] for r in labeled} - known)
    if unknown:
        print(f"Unknown speaker labels (check for typos): {', '.join(unknown)}", file=sys.stderr)

    by_speaker = defaultdict(list)
    for row in labeled:
        by_speaker[row["speaker"]].append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    for speaker, rows in sorted(by_speaker.items(), key=lambda kv: -len(kv[1])):
        if len(rows) < args.min_lines:
            continue
        # Exclude self-references from address detection; pronoun counts cover them.
        cast_others = {c: v for c, v in cast.items() if speaker not in ([c] + v)}
        endings, first_persons, mentions = analyze_speaker(rows, cast_others)
        out_path = write_report(args.out, speaker, rows, endings, first_persons, mentions)
        print(f"  {speaker}: {len(rows)} rows -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
