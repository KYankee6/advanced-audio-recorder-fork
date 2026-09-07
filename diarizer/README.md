# Local speaker diarization (offline, gated-model-free)

`diarize.py` attaches speaker labels (`발화자1`, `발화자2`, ...) to whisper
segments, 100% offline. The plugin calls it automatically after a local
whisper.cpp transcription when the diarizer paths are configured (Settings →
Transcription → Local diarizer).

- Architecture: [pyannote](https://github.com/pyannote/pyannote-audio)
  segmentation-3.0 loaded from the **ungated mirror**
  `ubitec/pyannote-segmentation-3.0` (no Hugging Face token needed). 10-second
  windows with 5-second hops are embedded and clustered; each whisper segment
  gets the overlap-weighted majority speaker of its covering windows.
- Far-field friendly: windowed segmentation embeddings survive meeting-room
  distance far better than ECAPA voice similarity, which collapses to a single
  mode (~0.4 similarity) on far-mic audio.
- If you know the headcount, set `Max speakers` in the settings — fixed-k is
  measurably better than auto-estimation. `0` = auto.

## Setup

```bash
python3.12 -m venv ~/diarize-env
~/diarize-env/bin/pip install -r requirements.txt
```

Then in the plugin settings:

| Setting | Value |
| --- | --- |
| Python path | `~/diarize-env/bin/python` (absolute path) |
| Script path | absolute path to this repo's `diarizer/diarize.py` |
| Max speakers | meeting headcount, or `0` for auto |

MPS is not used (PyanNet instance-norm lacks an MPS kernel); inference is
pinned to CPU. A 45-minute meeting takes a few minutes.

## Standalone use

```bash
python diarize.py --wav rec.wav --segments segs.json --max-speakers 3
# stdout: {"speakers": N, "assignments": [{"index": i, "speaker": "발화자1"}]}
```

`segs.json` is a list of `{"start": seconds, "end": seconds, "text": "..."}`
objects — the same shape whisper.cpp segments produce.
