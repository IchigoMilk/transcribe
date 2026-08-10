# Transcribe

This tool transcribes audio and video into a speaker-labeled TSV: Whisper does the transcription, an optional rule table fixes the proper nouns Whisper cannot know, and an optional voice-clustering pass groups lines by speaker so a human only has to assign real names instead of labeling every line by hand.

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

### 2. Correct proper nouns (optional)

Whisper knows nothing about a specific work, so it spells its proper nouns phonetically: a school name written with 聖 comes back as カタカナ, a character's surname is transcribed by sound rather than by its kanji. These errors are systematic — the same wrong spelling recurs in every episode — so a rule table fixes them deterministically.

```sh
python correct.py output/*.tsv --rules profiles/serial-experiments-lain/corrections.tsv
```

The rule file is a TSV with `type`, `pattern`, `replacement`, `note` columns. `type` is `literal` for a plain string swap or `regex` for a Python regular expression. Rules apply top to bottom, so list longer and more specific patterns before the shorter ones they contain, and blank or `#`-prefixed patterns act as section headings.

```tsv
type	pattern	replacement	note
literal	レイン	玲音	the given name is written 玲音
regex	岩倉(?:れいん|レイン)	岩倉玲音	full name, however the reading came out
```

The untouched Whisper output is kept in `output/raw/`, and every run re-applies the rules to that pristine copy. Editing the table and re-running therefore always converges on the same result, and no correction is ever stacked on top of itself. Every changed line is written to `output/corrections.log` (`file`, `start`, `before`, `after`), and the run prints per-rule hit counts plus the rules that never fired, which is how the table gets refined.

The same pass also collapses runs of consecutive identical lines, a Whisper decoder artifact where one sentence is emitted two or three times in a row. Short interjections are genuinely repeated by different speakers, so only lines of at least `--dedupe-min-len` characters (default 8) are collapsed; the surviving row keeps the full time span.

Corrections run before speaker identification. Clustering itself only reads the timestamps, but a human reviewing the tentative labels reads the text, so the text should already be right by then.

### 3. Fix what a rule cannot (optional)

A rule table only handles what recurs. What is left over is a line that is wrong exactly once, or a speaker label that voice clustering got wrong — both need a human reading the scene, and the result is data rather than a rule. `fix.py` applies those per-line overrides.

```sh
python fix.py output/*.tsv --overrides profiles/<work-name>/overrides.tsv
```

```tsv
file	start	speaker	text	note
input.tsv	353.60	岩倉玲音		clustered with the wrong speaker
input.tsv	428.85		あまり寝顔が可愛いから	clipped あまり
```

Overrides are keyed by `(file, start)` rather than by row number, because start times survive re-running the correction pass while row numbers do not. Each field is optional: a blank speaker leaves the speaker alone, a blank text leaves the text alone, so the file reads as "what a reviewer changed" instead of a restatement of every row. Applying is idempotent and order-independent, so it can be extended file by file and re-applied from scratch at any time. An override whose start time matches no row is reported rather than ignored, since that means it was written against a stale transcript.

Because corrections regenerate from `output/raw/`, whose speaker column is empty, `correct.py` carries the existing speaker column back over by start time. Refining the rule table and re-running therefore does not throw away labeling work.

### 4. Identify speakers (optional)

Instead of labeling every line by hand, you can group lines by voice similarity to get tentative labels.

```sh
python identify.py output/input.tsv input.mkv
# Fills empty rows in the speaker column with A, B, C... (overwrites the TSV)
```

Lines in the same cluster just get the same tentative label; real names are never resolved. Review the TSV and fix any mistakes directly. Pass `--num-speakers 3` to fix the speaker count if it's known. Rows that are already labeled by hand are left untouched.

Every segment of one file shares a recording channel — the same encode, room, and background music — and that shared component dominates cosine distance strongly enough to collapse a whole episode into one cluster plus a handful of outliers. The embeddings are therefore length-normalized and the per-file mean is subtracted before clustering, which cancels the channel and leaves the between-speaker variation.

When the speaker count is not given, it is chosen by a silhouette sweep. Those scores are nearly flat across candidate counts, so the peak is not meaningful on its own; among the counts scoring within 10% of the best, the largest is used. Over-segmenting is the cheaper mistake, because merging two labels that turned out to be one character is a one-line edit in the mapping file, while splitting one label that covered two characters means re-labeling by hand.

Whisper rounds segment boundaries outward and sometimes reports an end time a fraction of a second past the end of the file, which the embedding model rejects. Segments are clamped into the file and widened to a minimum length, so no episode is lost to one bad row. Segments that still fail to embed are reported and left unlabeled rather than aborting the run.

The embedding model runs on CUDA when it is available; `--device cpu` forces it off. On a whole series this is the difference between minutes and most of an hour.

### 5. Run the whole chain

`run.sh` runs the transcribe, correct and identify steps in order and accepts several inputs at once. Passing a whole series in one command is much faster than looping over the script, because the Whisper model is loaded once instead of per file.

```sh
./run.sh input.mkv
./run.sh ~/Videos/lain/*.mkv --rules profiles/serial-experiments-lain/corrections.tsv
./run.sh input.mkv --model medium --num-speakers 3
```

Without `--rules` the correction pass is skipped. `--skip-identify` stops after correction.

### 6. Apply real names

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
