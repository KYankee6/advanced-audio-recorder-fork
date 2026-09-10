# Local speaker diarization (offline, pyannote community-1)

`diarize.py` attaches speaker labels (`발화자1`, `발화자2`, ...) to whisper
segments, 100% offline at inference time. The plugin calls it automatically
after a local whisper.cpp transcription when the diarizer paths are
configured (Settings -> Transcription -> Local diarizer).

- Architecture: [pyannote](https://github.com/pyannote/pyannote-audio)
  **speaker-diarization-community-1** = segmentation + WeSpeaker embedding +
  VBx (Bayesian HMM) clustering. On Apple Silicon the segmentation model runs
  on MPS (about 2x end-to-end vs CPU) with automatic CPU fallback.
- First run downloads the pipeline from
  [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1):
  accept the conditions and pass a read token once, then it runs from the
  local HF cache offline. The pipeline repo bundles its segmentation,
  embedding and PLDA models, so no other gated repos are needed.
- The script self-heals two macOS traps: torchcodec cannot find Homebrew
  FFmpeg shared libraries (it re-executes itself with
  DYLD_FALLBACK_LIBRARY_PATH), and unassigned segments are labeled `-`.
- Speaker labels are ordered by total speaking time (the largest talker is
  발화자1) and are stable across reruns.

See the plugin settings guide for wiring it up. Standalone development
history: https://github.com/KYankee6/obsidian-diarizer
