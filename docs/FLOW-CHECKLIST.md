# Flow checklist (copy once per video)

Video: ______  Date: ______  Run folder: ______  Seed(s) tried: ______

Gates are in [FLOW.md](FLOW.md). Tick a box only after its gate passes.

- [ ] 1 Script: continuous speech, 2.8-3.1 words/s, last word by about 28.3 s, one gesture per beat, no silent gaps
- [ ] 2 Prompt: card form, time-coded arc, no camera moves, `olv lint` clean
- [ ] 3 Assets: FLUX look image (about 720p), empty set plate (no person), voice reference you own
- [ ] 4 Recipe: copied from `configs/flow.default.yaml`; guidance 1.0, Euler, 8 steps, identity 0.8, realism 0.3, VSA 10 %, seed 7329
- [ ] 5 Render: 1280x720, 719 frames, one shot; render seconds: ______
- [ ] 6 Best of seeds: chosen seed ______; every 5 s window at 0 % WER
- [ ] 7 QA pass 1: face, hands, lip sync, motion, words; vision model read only the flagged parts
- [ ] 8 Upscale: 1920x1080; upscale seconds: ______
- [ ] 9 Edit: repeated closing lines cut
- [ ] 10 Voice (skip if unused): exact length, word and lip-sync checks pass, human listening check approved
- [ ] 11 Finish: warm grade; encode CRF 18, AAC 192k
- [ ] 12 Final QA: 0 % WER in every window, voice match, lip sync, generation time reported
- [ ] `olv flow-check <run folder>` exits 0

## `flow_run.json` (in the run folder)

```json
{
  "render_s": 635, "upscale_s": 20, "grade": true,
  "wer_raw": "wer_raw.json", "wer_final": "wer_final.json",
  "raw":   {"width": 1280, "height": 720, "frames": 719},
  "final": {"width": 1920, "height": 1080, "vcodec": "h264", "crf": 18, "acodec": "aac", "audio_kbps": 192},
  "qa":    {"stage1_pass": true, "flagged_reviewed": true, "speaker_match": 0.7, "lip_sync_ok": true},
  "edit":  {"repeats_cut": true},
  "voice": {"used": true, "len_diff": 0, "wer_guard": true, "listening_approved": true}
}
```

`wer_raw.json` and `wer_final.json` are the output of `olv wer <wav> <lines.txt> --out <file>`. Set `"voice": {"used": false}` when you keep the model voice.
