# Transcribe Frame Preview

Shows the video frame belonging to the transcript row under the cursor, so
speaker labels can be checked against who is actually on screen.

## Use

1. Generate the frames:

   ```sh
   python shots.py output/<work>/*.tsv --media-dir ~/Videos/<work> --out shots/<work>
   ```

2. Point the extension at that directory. In workspace settings:

   ```json
   { "transcribeFrames.shotsRoot": "shots/strawberry-panic" }
   ```

3. Open a labeled TSV. The frame follows the cursor in a panel beside the
   editor, and hovering a row shows it inline.

Commands: *Transcribe: Show frame for current row*, *Transcribe: Toggle
following the cursor*. Settings: `shotsRoot`, `follow`, `hover`.

## How the lookup works

`shots.py` names each frame after the row's exact `start` value, so the
extension reads the first column of the current line and opens
`<shotsRoot>/<tsv stem>/<start>.jpg`. There is no index to keep in sync. If
that exact file is missing it falls back to the nearest frame within one
second, which covers rounding differences after an edit.
