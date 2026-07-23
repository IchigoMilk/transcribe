# Transcribe

This tool transcribes audio and video into a speaker-labeled TSV: Whisper does the transcription, then an optional voice-clustering pass groups lines by speaker so a human only has to assign real names instead of labeling every line by hand.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install faster-whisper pyyaml
```

`ffmpeg` is also required on the system (e.g. `apt install ffmpeg`). `transcribe.py` automatically converts input files to a mono 16kHz WAV temp file via ffmpeg, so any audio/video format ffmpeg supports can be passed in.

Instead of the system `pip3`, activate the project-local `.venv` and use `python -m pip`. This avoids package-management errors from the Linux system Python. Run the remaining commands in the same shell while the virtual environment is active.

The model is downloaded automatically on its first use (about 3 GB for `large-v3`). The default `int8` setting is intended for hardware around a GTX 1660 SUPER with 6 GB of VRAM. If the process runs out of VRAM, use `--model medium`.

GPU use also needs cuBLAS/cuDNN, which a plain pip install of faster-whisper does not pull in on a machine without a system-wide CUDA toolkit:

```sh
python -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

`transcribe.py` detects these and re-execs itself with `LD_LIBRARY_PATH` set, since ctranslate2 cannot find them otherwise.

### Character profiles (optional)

`profiles/<work-name>/*.yaml` holds one file per character, with a `name` field. `make_mapping.py` reads these to print a candidate-name reference list when drafting a mapping file, so you don't have to remember or retype exact names by hand. Create a new subdirectory per work, e.g. `profiles/serial-experiments-lain/`.

```yaml
name: 岩倉玲音
aliases: [玲音, いわくら れいん, レイン]
```

`aliases` is optional and not read by any script; keep it only as your own reference if useful.

`profiles/` is gitignored: character data for a specific work is not meant to be pushed to this repo, so it only ever lives on disk locally.

### Additional setup for speaker identification (optional)

`identify.py` clusters speakers by voice similarity using pyannote.audio.

```sh
python -m pip install pyannote.audio scikit-learn omegaconf
```

The default embedding model (`pyannote/wespeaker-voxceleb-resnet34-LM`) is openly licensed, so no HuggingFace account or token is required. pyannote.audio depends on torch, so the first install downloads several GB, and the model itself is downloaded automatically on first use.

If you swap in a gated embedding model, pass a HuggingFace token via `identify.py --hf-token`, or set the `HF_TOKEN` environment variable.

## Usage

### 1. Transcribe

```sh
python transcribe.py input.mkv
# Creates output/input.tsv
```

Enter speaker names directly in the generated TSV file's `speaker` column, or use the steps below to speed that up.

### 2. Identify speakers (optional)

Instead of labeling every line by hand, you can group lines by voice similarity to get tentative labels.

```sh
python identify.py output/input.tsv input.mkv
# Fills empty rows in the speaker column with A, B, C... (overwrites the TSV)
```

Lines in the same cluster just get the same tentative label; real names are never resolved. Review the TSV and fix any mistakes directly. Pass `--num-speakers 3` to fix the speaker count if it's known. Rows that are already labeled by hand are left untouched.

`run.sh` runs steps 1 and 2 together.

```sh
./run.sh input.mkv
# Runs transcribe.py then identify.py, producing output/input.tsv with tentative labels
./run.sh input.mkv --model medium --num-speakers 3
```

### 3. Apply real names

Instead of a manual find-and-replace in an editor, prepare a small mapping TSV with `label` and `name` columns and apply it in bulk. `make_mapping.py` drafts this file for you: it lists every tentative label with its row count and an example line, and prints candidate names from the profiles directory for reference.

```sh
python make_mapping.py output/input.tsv --profiles profiles/serial-experiments-lain
# Creates output/input.mapping.tsv with label, name (blank), count, example columns
```

Fill in the `name` column by hand, then apply it:

```sh
python relabel.py output/input.tsv output/input.mapping.tsv
# Replaces speaker values matching a label in mapping.tsv (overwrites the TSV)
```

Labels present in the TSV but missing from the mapping are left unchanged and reported, so the mapping file can be filled in incrementally.
