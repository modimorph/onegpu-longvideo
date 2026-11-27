# Limitations

Open problems, stated plainly. Then model licences.

Our runs used one 24 GB consumer GPU (RTX 4090). Nothing here was tested on other hardware.

## Open problems

### Speech quality past about 15 s

- The model was trained on clips of 5, 10 and 15 s.
- 30 s one-shot speech works only with strict prompt discipline: continuous speech at about 2.8-3.1 words per second, no planned silence, one line per beat. Even then, one of our three full 30 s runs failed (WER 0.365), and another scored WER 0.133 with 12 inserted words near the end.
- Every 45 s and 60 s attempt failed. Examples: 66 invented words in a 45 s run; 126 invented words after about 17 s in a 60 s run; WER 0.495 with clean speech only to about 11 s in another.
- In every longer run, speech held for 15-20 s, then the audio lost its place in the script. It repeated lines and invented filler.
- We do not know why. Our guess is the audio timeline past 30 s. It is untested.
- Chaining keeps the words (WER 6 % on 60 s) but shows joins: colour steps 7-24x typical, face match drops (worst 0.59 to 0.50), digital silence after each join.

### Repeats

- A 60 s run replayed the same five-step gesture sequence at 8 s and 35 s.
- Two seeds of the same 30 s prompt differed: WER 0.024 with 0 gesture events per minute, and WER 0.012 with 10 gesture events per minute.
- Closing lines can be said twice. One run repeated them between 20.5 s and 27.5 s. We removed 3.4 s in the edit.

### Drift

- Face match fell in the second clip of a 2 x 15 s chain (0.33 to 0.29).
- Sharpness rises segment by segment in a chain.
- Background and lighting can step. Our detector flagged a luma step at 14.1 s in a 30 s clip.

### Quality loss from the speed tricks

- VSA made the face 19 % softer on our test clip (sharpness 969 vs 1200). We tested quality on one 10 s clip per arm. Behaviour over many seeds and over a full 30 s is not measured.
- ComfyUI's sparse node differs from the training mode (64-token cubes and an 8-bit kernel vs 128-token tiles). We do not know which difference causes the softness.
- SageAttention on every row raised one clip's WER by 0.263. We do not use it for speech.
- SPAN x2 is fast and adds little detail. SeedVR2 3B draws better skin and face detail but costs about 90 s per video second, so it is not a production route here.
- FlashVSR+ did not run on our Windows setup (no C compiler for Triton). Its quality on clean AI faces is untested.

### Hardware limits

- Native 1080p does not fit: a 5.2 s probe used 21.1 GB on the card plus 8.8 GB shared memory. Thirty seconds would need about 31 GB (est).
- Spill into shared GPU memory makes a run slow enough that we treat it as out of memory.
- torch.compile failed with weight streaming.
- One GPU job at a time. A second process on the same card slowed a 10 s render from about 265 s to 668 s.

### Prompt and camera control

- A silent beat in the prompt becomes invented speech.
- Two-hand gestures cross and fuse.
- An untimed camera instruction can be ignored (a "push-in" measured zoom 1.009x). Timed pans, arcs and push-ins were verified. Crane and orbit moves were not.
- A proper noun the model mispronounces stays mispronounced after voice conversion. Fix it in the prompt.

### QA

- Stage 2 (VLM on flagged windows) has not been scored. Its gain is unproven.
- The lip-sync offset estimator and the blink rule failed calibration and are report-only.
- Hand flags fire about twice per minute on real footage.
- Thresholds come from a small real-footage sample (4 segments of one speaker).
- The planted-artifact recall test is designed and not run.
- Some scorers crashed or scored partial clips in our runs. We picked one clip with its own check unfinished.

### Voice post step

- Seed-VC output is 22.05 kHz.
- Speaker match rose (0.717 to 0.760 on a joined cut) while DNSMOS fell slightly (3.50 to 3.43).
- Not tested by ear. No metric here measures "robotic".
- A cleanup model after conversion lost most of the gain, so we do not use it.

### Evidence quality

- Most timings and scores are single runs.
- Identity transfer from the training base to Omni is unmeasured (0.19 on Omni vs 0.22 on the base, three clips, different render settings).
- Several speed numbers are projections (est), not runs. The text marks each one.

## Models and licences

You bring your own weights. This repo does not ship or redistribute any model. Before you download or run a model, read its licence and accept it. Licences change. Check the current text on each model's page.

| Component | What we used | What our notes say about the licence |
|---|---|---|
| Joint audio-video DiT | FastH3 Omni, from the MiniMax-H3 family | **The MiniMax-H3 family licence has territory limits.** It excludes some regions. Read the licence text and confirm that you may use the model where you live and where you ship output. We do not summarise the list here, because it may change. |
| Single-identity LoRA | trained by us on consented data | Not distributed. Train your own only on data you have written consent to use. |
| Upscaler | SPAN x2 | not verified in our notes; check the weights' licence |
| Upscaler | SeedVR2 3B | Apache-2.0 per our notes |
| Upscaler | FlashVSR+ | Apache-2.0 per our notes (not run successfully) |
| Upscalers | Real-ESRGAN, RealPLKSR | not verified in our notes; check the weights' licence |
| Voice conversion | Seed-VC | not verified in our notes; check the code and weights |
| ASR | faster-whisper (Whisper weights) | not verified in our notes; check |
| Face and identity | ArcFace (insightface) | not verified in our notes. Some face-model weights restrict commercial use. Check. |
| Lip-sync | SyncNet v2 | MIT per our notes |
| VLM (stage 2) | Qwen3.8-27B | Apache-2.0 per its model card (our notes) |
| Hand and face detectors | VBench-2.0 anomaly weights | licence field was empty when we looked; code Apache-2.0 |
| Another human-artifact detector (not used) | HADM | no licence found |

Also:

- Some text-to-speech models we tried carry non-commercial output terms. We do not use them here. Check any TTS you add.
- A voice or face of a real person needs that person's written consent. Do not use this repo to imitate someone without it.
- You are responsible for complying with each licence, with local law, and with any platform rule on synthetic media. Label synthetic output where required.

## Speech past about 15 s (summary of EXPERIMENTS.md section 10)
- At 30 s in one pass, every speech error we measured started at 14.5 s or later.
- One seed can pass and the next seed can fail on the same recipe. Plan for 2 or more takes and pick by per-window WER.
- Our published tutor take still has one 5 s window with errors (WER 3.2 % overall).
