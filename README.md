# Speech Pattern Template Extractor

This tool transcribes audio and video, then aggregates character speech patterns from speaker-labeled TSV files. It helps verify a profile's first-person pronouns, sentence endings, and forms of address using dialogue examples.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install faster-whisper pyyaml
```

Instead of the system `pip3`, activate the project-local `.venv` and use `python -m pip`. This avoids package-management errors from the Linux system Python. Run the remaining commands in the same shell while the virtual environment is active.

The model is downloaded automatically on its first use (about 3 GB for `large-v3`). The default `int8` setting is intended for hardware around a GTX 1660 SUPER with 6 GB of VRAM. If the process runs out of VRAM, use `--model medium`.

## Usage

### 1. Transcribe

```sh
python transcribe.py input.mkv
# Creates output/input.tsv
```

Enter speaker names in the generated TSV file's `speaker` column. Names must exactly match the `name` field in the `profiles/*.yaml` files being analyzed.

### 2. Aggregate speech patterns

```sh
python analyze.py output/input.tsv --profiles /path/to/profiles
# Creates output/reports/<speaker-name>.md
```

The report contains first-person pronouns, sentence endings, forms of address, and representative dialogue.

## Example

`example.tsv` and `profiles/` contain fictional example lines and minimal profiles for testing.

```sh
python analyze.py example.tsv --profiles profiles --min-lines 2 --out /tmp/reports
```
