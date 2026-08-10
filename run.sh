#!/usr/bin/env bash
set -euo pipefail

# Design: chains transcribe.py -> correct.py -> identify.py, the order the
# three passes are always run in. Correction sits in the middle because a
# human reviewing the tentative speaker labels reads the text, so the proper
# nouns should already be right by the time labels exist.
#
# Multiple inputs are accepted and transcribed in one process, since loading
# the Whisper model costs far more than transcribing a single file. Passing a
# whole series at once is therefore much faster than looping over this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") <input-file>... [options]

Options:
  --model NAME       Whisper model, forwarded to transcribe.py
  --compute TYPE     Compute type, forwarded to transcribe.py
  --device DEV       cuda / cpu / auto, forwarded to transcribe.py
  --out DIR          Output directory for the TSVs (default: ${SCRIPT_DIR}/output)
  --rules FILE       Correction rule TSV, e.g. profiles/<work>/corrections.tsv
                     (default: no correction pass)
  --num-speakers N   Fix the speaker count, forwarded to identify.py (default: auto)
  --hf-token TOKEN   HuggingFace token, forwarded to identify.py (default: \$HF_TOKEN)
  --skip-identify    Stop after correction, without speaker clustering
EOF
    exit 1
}

OUT_DIR="${SCRIPT_DIR}/output"
RULES=""
SKIP_IDENTIFY=0
INPUTS=()
TRANSCRIBE_ARGS=()
IDENTIFY_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --model|--compute)
            TRANSCRIBE_ARGS+=("$1" "$2")
            shift 2
            ;;
        --device)
            # Both passes accept a device, and they should agree.
            TRANSCRIBE_ARGS+=("$1" "$2")
            IDENTIFY_ARGS+=("$1" "$2")
            shift 2
            ;;
        --out)
            OUT_DIR="$2"
            TRANSCRIBE_ARGS+=("--out" "$2")
            shift 2
            ;;
        --rules)
            RULES="$2"
            shift 2
            ;;
        --num-speakers|--hf-token)
            IDENTIFY_ARGS+=("$1" "$2")
            shift 2
            ;;
        --skip-identify)
            SKIP_IDENTIFY=1
            shift
            ;;
        -*)
            usage
            ;;
        *)
            INPUTS+=("$1")
            shift
            ;;
    esac
done

[ ${#INPUTS[@]} -ge 1 ] || usage

for input in "${INPUTS[@]}"; do
    [ -f "$input" ] || { echo "Input file not found: $input" >&2; exit 1; }
done
[ -z "$RULES" ] || [ -f "$RULES" ] || { echo "Rule file not found: $RULES" >&2; exit 1; }

if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

python3 "${SCRIPT_DIR}/transcribe.py" "${INPUTS[@]}" "${TRANSCRIBE_ARGS[@]}"

TSVS=()
for input in "${INPUTS[@]}"; do
    stem="$(basename "$input")"
    TSVS+=("${OUT_DIR}/${stem%.*}.tsv")
done

if [ -n "$RULES" ]; then
    echo
    python3 "${SCRIPT_DIR}/correct.py" "${TSVS[@]}" --rules "$RULES"
fi

if [ "$SKIP_IDENTIFY" -eq 1 ]; then
    echo
    echo "Done (identification skipped)."
    exit 0
fi

# identify.py takes one media file at a time, so this loop is unavoidable;
# the embedding model is small enough that reloading it per file is cheap.
for i in "${!INPUTS[@]}"; do
    echo
    echo "Identifying speakers: $(basename "${INPUTS[$i]}")"
    python3 "${SCRIPT_DIR}/identify.py" "${TSVS[$i]}" "${INPUTS[$i]}" "${IDENTIFY_ARGS[@]}"
done

echo
echo "Done: ${OUT_DIR}"
echo "Next: draft a mapping with make_mapping.py, fill in real names, then apply it with relabel.py."
