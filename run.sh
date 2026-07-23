#!/usr/bin/env bash
set -euo pipefail

# Design: chains transcribe.py and identify.py into a single command, since
# that pair (transcribe -> tentative speaker labels) is always run together.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") <input-file> [options]

Options:
  --model NAME       Whisper model, forwarded to transcribe.py
  --compute TYPE     Compute type, forwarded to transcribe.py
  --device DEV       cuda / cpu / auto, forwarded to transcribe.py
  --out DIR          Output directory for the TSV (default: ${SCRIPT_DIR}/output)
  --num-speakers N   Fix the speaker count, forwarded to identify.py (default: auto)
  --hf-token TOKEN   HuggingFace token, forwarded to identify.py (default: \$HF_TOKEN)
EOF
    exit 1
}

[ $# -ge 1 ] || usage
INPUT="$1"
shift

OUT_DIR="${SCRIPT_DIR}/output"
TRANSCRIBE_ARGS=()
IDENTIFY_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --model|--compute|--device)
            TRANSCRIBE_ARGS+=("$1" "$2")
            shift 2
            ;;
        --out)
            OUT_DIR="$2"
            TRANSCRIBE_ARGS+=("--out" "$2")
            shift 2
            ;;
        --num-speakers|--hf-token)
            IDENTIFY_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

[ -f "$INPUT" ] || { echo "Input file not found: $INPUT" >&2; exit 1; }

if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

python3 "${SCRIPT_DIR}/transcribe.py" "$INPUT" "${TRANSCRIBE_ARGS[@]}"

STEM="$(basename "$INPUT")"
STEM="${STEM%.*}"
TSV="${OUT_DIR}/${STEM}.tsv"

python3 "${SCRIPT_DIR}/identify.py" "$TSV" "$INPUT" "${IDENTIFY_ARGS[@]}"

echo
echo "Done: ${TSV}"
echo "Next: draft a mapping with make_mapping.py, fill in real names, then apply it with relabel.py."
