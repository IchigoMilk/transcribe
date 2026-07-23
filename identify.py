#!/usr/bin/env python3
"""Identify recurring speakers in an already-transcribed TSV (2nd pass).

Design:
- Whisper's segments (produced by transcribe.py) already carry reliable
    timestamps, so this pass reuses them instead of re-running voice-activity
    detection or segmentation. Only speaker-embedding extraction and
    clustering are added on top, which keeps this pass cheap.
- Rows are grouped into anonymous clusters labeled A, B, C, ... ordered by
    first appearance. This tool never guesses real names; the user bulk
    find-and-replaces each letter with a real name afterward, which is far
    less error-prone than asking a model to name speakers directly.
- Rows that already carry a speaker label are left untouched, so this can be
    re-run after partial manual correction without discarding it.

Dependencies: pip install pyannote.audio scikit-learn
Uses pyannote/wespeaker-voxceleb-resnet34-LM, which is openly licensed, so no
HuggingFace token is required. --hf-token / HF_TOKEN are still accepted for
users who swap in a gated embedding model.
"""

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

from transcribe import extract_audio

SPEAKER_LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def read_rows(tsv_path: Path):
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader), reader.fieldnames


def write_rows(tsv_path: Path, rows, fieldnames) -> None:
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


def embed_segments(wav_path: Path, rows):
    from pyannote.audio import Inference, Model
    from pyannote.core import Segment

    model = Model.from_pretrained(EMBEDDING_MODEL)
    inference = Inference(model, window="whole")
    embeddings = []
    for row in rows:
        segment = Segment(float(row["start"]), float(row["end"]))
        embeddings.append(inference.crop(str(wav_path), segment))
    return embeddings


def cluster_embeddings(embeddings, num_speakers):
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    X = np.vstack(embeddings)
    n = len(X)
    if n == 1:
        return np.zeros(1, dtype=int)

    if num_speakers:
        k_candidates = [min(num_speakers, n)]
    else:
        # Sweep candidate speaker counts and keep the one with the best
        # cluster separation, since the true speaker count is unknown.
        k_candidates = list(range(2, min(10, n - 1) + 1)) or [2]

    best_k, best_score, best_labels = None, -1.0, None
    for k in k_candidates:
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, metric="cosine") if 1 < k < n else 0.0
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if num_speakers:
        print(f"  Speaker count fixed at: {best_k}")
    else:
        print(f"  Auto-selected speaker count: {best_k} (silhouette={best_score:.3f})")
    return best_labels


def assign_labels(cluster_ids):
    order = {}
    labels = []
    for cid in cluster_ids:
        if cid not in order:
            if len(order) >= len(SPEAKER_LABELS):
                raise RuntimeError("Too many speakers (max 26 supported)")
            order[cid] = SPEAKER_LABELS[len(order)]
        labels.append(order[cid])
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill empty speaker rows in a TSV with tentative labels (A, B, C...) by clustering voice similarity"
    )
    parser.add_argument("tsv", type=Path, help="TSV produced by transcribe.py")
    parser.add_argument("media", type=Path, help="original audio/video file")
    parser.add_argument(
        "--num-speakers", type=int, default=None, help="fix the speaker count (default: auto-detect)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output TSV (default: overwrite the input TSV)"
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (default: HF_TOKEN env var; usually unneeded since the default embedding model is public)",
    )
    args = parser.parse_args()

    if not args.tsv.exists():
        print(f"TSV not found: {args.tsv}", file=sys.stderr)
        return 1
    if not args.media.exists():
        print(f"Media not found: {args.media}", file=sys.stderr)
        return 1
    if args.hf_token:
        from huggingface_hub import login

        login(token=args.hf_token)

    rows, fieldnames = read_rows(args.tsv)
    target_rows = [r for r in rows if not r.get("speaker", "").strip()]
    if not target_rows:
        print("No unlabeled rows to identify.")
        return 0

    with tempfile.TemporaryDirectory(prefix="identify-") as tmp_dir_name:
        wav_path = extract_audio(args.media, Path(tmp_dir_name))
        print(f"Embedding {len(target_rows)} segment(s) ...")
        embeddings = embed_segments(wav_path, target_rows)

    print("Clustering ...")
    cluster_ids = cluster_embeddings(embeddings, args.num_speakers)
    labels = assign_labels(cluster_ids)
    for row, label in zip(target_rows, labels):
        row["speaker"] = label

    out_path = args.out or args.tsv
    write_rows(out_path, rows, fieldnames)
    print(f"-> {out_path}")
    print()
    print("Next: check the tentative labels (A, B, C...) and fix mistakes directly in the TSV.")
    print("Once confirmed, bulk find-and-replace each letter with a real name in the speaker column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
