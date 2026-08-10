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


def label_for(index: int) -> str:
    """A, B, ... Z, AA, AB, ... so an over-segmented file still gets labels."""
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, len(SPEAKER_LABELS))
        name = SPEAKER_LABELS[rem] + name
    return name


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

# Whisper rounds segment boundaries outward and can emit an end time slightly
# past the real end of the audio, which pyannote rejects outright. It also
# emits very short segments ("はい") that carry too few samples for the
# embedding model. Both are fixed by fitting each segment into the file
# instead of failing, since a whole episode should not be lost to one bad row.
MIN_SEGMENT_DURATION = 0.5


def fit_segment(start: float, end: float, duration: float):
    """Clamp a Whisper segment into [0, duration] and widen it to the minimum length.

    Returns None only when the file itself is shorter than the minimum, i.e.
    when no usable window exists at all.
    """
    from pyannote.core import Segment

    if duration < MIN_SEGMENT_DURATION:
        return None
    start = max(0.0, min(start, duration))
    end = max(start, min(end, duration))
    if end - start < MIN_SEGMENT_DURATION:
        # Grow around the midpoint, then slide back inside the file if the
        # segment sits against either edge.
        mid = (start + end) / 2
        start = mid - MIN_SEGMENT_DURATION / 2
        end = mid + MIN_SEGMENT_DURATION / 2
        if start < 0.0:
            start, end = 0.0, MIN_SEGMENT_DURATION
        elif end > duration:
            start, end = duration - MIN_SEGMENT_DURATION, duration
    return Segment(start, end)


def resolve_device(name: str):
    """Pick the torch device to embed on. "auto" prefers CUDA when present."""
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


# A broken CUDA stack (a cuDNN built against a different torch, say) fails on
# every single segment rather than on a few odd ones. Retrying the whole file
# on CPU is far better than emitting an unlabeled transcript, and the check is
# cheap: if none of the first few segments embed, the device is the problem,
# not the audio.
GPU_PROBE_SEGMENTS = 3


def _embed_on(wav_path: Path, rows, torch_device):
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained(EMBEDDING_MODEL)
    inference = Inference(model, window="whole", device=torch_device)
    duration = model.audio.get_duration(str(wav_path))
    embeddings = []
    failures = 0
    for row in rows:
        segment = fit_segment(float(row["start"]), float(row["end"]), duration)
        if segment is None:
            embeddings.append(None)
            failures += 1
            continue
        try:
            embeddings.append(inference.crop(str(wav_path), segment))
        except Exception as e:  # noqa: BLE001 - one bad row must not lose the file
            if failures == 0:
                print(f"  Segment [{segment.start:.2f}, {segment.end:.2f}] failed: {e}", file=sys.stderr)
            embeddings.append(None)
            failures += 1
        if len(embeddings) == GPU_PROBE_SEGMENTS and failures == GPU_PROBE_SEGMENTS:
            return None, failures
    return embeddings, failures


def embed_segments(wav_path: Path, rows, device: str = "auto"):
    """Embed one row per entry, or None where the segment could not be embedded."""
    torch_device = resolve_device(device)
    print(f"  Embedding device: {torch_device}")
    embeddings, failures = _embed_on(wav_path, rows, torch_device)

    if embeddings is None:
        if torch_device.type == "cpu":
            raise RuntimeError("Embedding failed on CPU; see the error above.")
        print("  Every probe segment failed on this device; falling back to CPU.", file=sys.stderr)
        import torch

        embeddings, failures = _embed_on(wav_path, rows, torch.device("cpu"))
        if embeddings is None:
            raise RuntimeError("Embedding failed on both CUDA and CPU; see the errors above.")

    if failures:
        print(f"  {failures} segment(s) could not be embedded and stay unlabeled.", file=sys.stderr)
    return embeddings


# Silhouette scores on speaker embeddings are nearly flat across candidate
# speaker counts, so the peak is not a trustworthy answer on its own. Among
# the counts that score close to the best, prefer the largest: merging two
# labels that turned out to be one character is a one-line edit in the
# mapping file, while splitting one label that covered two characters means
# re-labeling by hand. Over-segmenting is the cheaper mistake.
SILHOUETTE_TOLERANCE = 0.9
MAX_AUTO_SPEAKERS = 12


def normalize_embeddings(embeddings):
    """Length-normalize, then subtract the mean of the file.

    Every segment comes from the same recording, so all embeddings share a
    large channel component: the encode, the background music, the room. That
    shared direction dominates cosine distance and collapses clustering into
    one giant cluster plus a few outliers. Removing the per-file mean cancels
    it and leaves the between-speaker variation the clustering actually needs.
    """
    import numpy as np

    X = np.vstack(embeddings).astype(float)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X -= X.mean(axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # A segment sitting exactly on the mean would divide by zero.
    norms[norms == 0] = 1.0
    return X / norms


def cluster_embeddings(embeddings, num_speakers):
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    X = normalize_embeddings(embeddings)
    n = len(X)
    if n == 1:
        return np.zeros(1, dtype=int)

    if num_speakers:
        k = max(1, min(num_speakers, n))
        labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
        print(f"  Speaker count fixed at: {k}")
        return labels

    scored = []
    for k in range(2, min(MAX_AUTO_SPEAKERS, n - 1) + 1) or [2]:
        labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
        score = silhouette_score(X, labels, metric="cosine") if 1 < k < n else 0.0
        scored.append((k, score, labels))
    if not scored:
        return np.zeros(n, dtype=int)

    best_score = max(score for _, score, _ in scored)
    cutoff = best_score * SILHOUETTE_TOLERANCE if best_score > 0 else best_score
    k, score, labels = max((c for c in scored if c[1] >= cutoff), key=lambda c: c[0])
    print(f"  Auto-selected speaker count: {k} (silhouette={score:.3f}, best={best_score:.3f})")
    return labels


def assign_labels(cluster_ids):
    order = {}
    labels = []
    for cid in cluster_ids:
        if cid not in order:
            order[cid] = label_for(len(order))
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
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto for the embedding model")
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
        # The default embedding model is public, so a token is a convenience
        # for users who swapped in a gated one. A stale or unusable token must
        # not take down a run that never needed authentication in the first
        # place; if the model really is gated, from_pretrained fails later with
        # a message that actually names the problem.
        from huggingface_hub import login

        try:
            login(token=args.hf_token)
        except Exception as e:  # noqa: BLE001
            print(f"HuggingFace login failed, continuing unauthenticated: {e}", file=sys.stderr)

    rows, fieldnames = read_rows(args.tsv)
    target_rows = [r for r in rows if not r.get("speaker", "").strip()]
    if not target_rows:
        print("No unlabeled rows to identify.")
        return 0

    with tempfile.TemporaryDirectory(prefix="identify-") as tmp_dir_name:
        wav_path = extract_audio(args.media, Path(tmp_dir_name))
        print(f"Embedding {len(target_rows)} segment(s) ...")
        embeddings = embed_segments(wav_path, target_rows, args.device)

    embedded = [(row, emb) for row, emb in zip(target_rows, embeddings) if emb is not None]
    if not embedded:
        print("No segment could be embedded; nothing to cluster.", file=sys.stderr)
        return 1

    print("Clustering ...")
    cluster_ids = cluster_embeddings([emb for _, emb in embedded], args.num_speakers)
    labels = assign_labels(cluster_ids)
    for (row, _), label in zip(embedded, labels):
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
