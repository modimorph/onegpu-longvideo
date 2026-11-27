# Experiments

What we tried, what we measured, what failed, and what we learned.

## Conventions

- Our runs used one 24 GB consumer GPU (RTX 4090). This is what we had. It is not a requirement.
- Video model: FastH3 Omni int8 (MiniMax-H3 family), a joint audio-video DiT, run through ComfyUI. Distilled 8-step checkpoint, Euler sampler, seed fixed per comparison.
- Most numbers are single runs (n = 1). Where a number comes from a bigger set, the text says so.
- "Est" means arithmetic or extrapolation. We did not run it. "Untested" means no run exists.
- "Look" is a face-match score (cosine similarity, 1 = identical) against held-out real footage of the speaker. "Voice" is a speaker-match score on the same scale. WER is word error rate of an ASR transcript against the script. WER above 1.0 is possible because extra words count as errors.
- Identity work used a single-identity LoRA trained on consented data. We do not publish it, its data, or any real person's likeness.

## 1. Route to a 30 s 1080p file

Goal: 30 s of talking-head video at 1920x1080 in under 20 min from prompt to file, on one 24 GB GPU.

| Route | Generate | Upscale | Total | Status |
|---|---|---|---|---|
| 432p dense attention, then SeedVR2 3B | 8.6 min (measured) | about 44 min (est) | about 53 min (est) | not run end to end |
| Native 1080p, VSA sparse attention | 55-60 min (est) | none | 55-60 min (est) | out: about 31 GB activations (est), does not fit |
| **720p, VSA keep 10 %, then streaming SPAN x2** | 10.6 min | about 20 s | **12.7 min (762 s)** | measured |

Three full 30 s runs of the final route: 762 s, 758 s and 744 s. Parts of the 762 s run: 637 s generate and decode, about 100 s load and transfer, about 20 s upscale. One of the three runs failed quality (see section 4).

## 2. Attention and speed experiments

Fixture: 10 s clip, 1280x720, 8 steps, same seed, same graph. Dense attention is the reference (463 s).

| Change | Time | vs dense | Quality (n = 1) | Verdict |
|---|---|---|---|---|
| VSA sparse attention, keep 10 %, sparse from step 0 | 136 s | 3.4x (2.7x if a possible cached text encode, up to 34 s, is charged to it) | look +0.011 and +0.020 per 5 s window; WER 0.045 vs 0.136; AV offset 42 ms vs 292-458 ms; face sharpness 969 vs 1200 (-19 %) | keep |
| SageAttention on video rows only | 322 s | 1.44x | not scored | drop (slower than VSA) |
| torch.compile (inductor) | failed at step 1 | - | - | drop |
| No `--reserve-vram` flag (3 steps) | 207 s vs 216 s with it | about 1.04x (noise) | - | drop |
| Native 1920x1088 + VSA, 5.2 s probe | 203 s, 21.1 GB on card + 8.8 GB shared | - | - | native 1080p out |

Details are in PERFORMANCE.md.

Older speed tests (10 s clips, 8 steps unless noted, Omni, no VSA):

| Test | Size | Sampling time | Note |
|---|---|---|---|
| Base, no Sage | 768x768 | 338 s | baseline |
| Sage | 768x768 | 161 s | same output by eye |
| Face-crop prompt | 576x576 | 70 s | head fully framed |
| + EasyCache 0.2 | 576x576 | 52 s | SSIM 0.967 vs previous; kept |
| 6 steps | 576x576 | 43 s | SSIM 0.84; borderline |
| 448 px | 448x448 | 18 s | soft face; rejected |

Fixture, 15 s, 1024x576 (362 frames): 8 steps 397 s; 12 steps 539 s (+36 %, no gain we could measure); realism LoRA 0.3 404 s (look +0.035, kept). The same fixture at 768x432: 162-171 s with equal look.

### Sage on every row: faster, but it changed words

Five paired 10 s clips (same checkpoint, prompt, seed). Mean render time 367 s dense vs 159 s Sage (2.31x). One dense clip shared the GPU with another model (668 s); without that pair the gain is 1.82x (arithmetic).

The quality gate failed on two checks. Mean reward change was -0.033 (limit -0.02). One clip's WER rose by 0.263 (limit +0.10). In this model the same attention layers write the speech. A backend that changes the words fails a talking-head gate. We turned Sage off for the quality rounds.

### EasyCache

On a 10 s test it cut sampling from 70 s to 52 s. We kept it off for long speech. We did not test it at 30 s with a gate.

### Render graph from the model card

Guidance 1.0, Euler, LoRA only, cache on, against our older graph (guidance 0.9, two LoRAs): look +0.027, voice +0.024, 141-167 s per 10 s 1024x576 clip against about 285 s (about -45 %). The gains belong to the bundle. Part of the speed-up has a simple cause: guidance above 1.0 runs a second model pass at each step. The model was distilled to run without it.

## 3. Resolution and length

Dense attention, 30 s (719 frames), one pass, unless noted.

| Size | Time | Note |
|---|---|---|
| 768x432 | 504-515 s | sampling about 89 %; 93 % of the compute-only bound (see PERFORMANCE.md) |
| 1024x576 | 1313 s | look about +0.05 vs 768x432; 13 invented words |
| 1280x720 | about 54 min (projected from 6 of 8 steps at 383 s/step) | stopped at step 6 |
| 1024x576 x 736 frames (earlier try) | stopped at about 1750 s | spilled 14 GB into shared memory |

## 4. Length of one-shot speech

The model was trained on clips of 5, 10 and 15 s. We wanted 30 s and more with no cuts.

| Clip | Size | Time | Result |
|---|---|---|---|
| 30 s, first try, old graph | 768x432 | 763 s | WER 0.12; voice falls after 5 s |
| 30 s, card graph, voice reference, 6 lines at 3.07 words/s | 768x432 | 504 s | WER 0.087; one garbled second at 5-10 s |
| 30 s, continuous speech 2.84 words/s, realism 0.3 | 768x432 | 524 s | **WER 0.024**, look 0.64-0.66 facing camera |
| same, other seed | 768x432 | 489 s | WER 0.012; 10 gesture events/min (first seed: 0) |
| 30 s, six-section prompt with silent action beats | 768x432 | 515 s | speech passed; look failed |
| 30 s, 1024x576 | 1024x576 | 1313 s | 13 invented words; over our 20 min cap |
| 45 s | 640x352 | 533 s | 66 invented words |
| 60 s | 640x352 | 667.6 s | WER 0.495; clean speech only to about 11 s; gesture replays |
| 60 s | 640x352 | 870 s | WER 1.62; 126 invented words after about 17 s |

What made 30 s work:

- Continuous speech with no planned silence. A silent beat in the prompt is a gap the model fills with invented speech.
- About 2.8-3.1 words per second, one line per beat.
- One new one-hand gesture per spoken line. Two-hand gestures cross and fuse. No props that change state.
- A short voice reference cut at a natural pause (4.1 s worked better than 5.2 s).
- A time-coded camera plan. An untimed "push-in" came out locked off (zoom 1.009x). A timed push-in measured 1.105x over its first 22.5 s. A timed arc panned 8.3 % of frame width.

Prompt discipline bought about twice the training length. It did not buy three or four times.

## 5. Chaining

Five segments of about 13 s (5 x 311 frames, 640x352), each started from the last frames and audio of the previous one: 447 s for 60 s, WER 6 %, no invented words. It looked like five joined clips. Frame-by-frame:

- Colour: each join is a colour step 7-24 times the typical frame-to-frame change.
- Identity: face match drops at joins, worst from 0.59 to 0.50.
- Sound: room tone falls to digital silence for 0.3-2 s after each join.
- Motion: segments were prompted "silent and still", so each join reads as stop, freeze, restart.
- Drift: sharpness rises segment by segment.

A 2 x 15 s chain at 768x432 read cleaner in speech, but look fell from 0.33 to 0.29 in the second clip. We dropped chaining. Our untested guess: a chain needs a shared audio timeline and colour anchoring.

## 6. Upscalers

4 s clip, 768x432 to 1920x1080, ComfyUI path. See PERFORMANCE.md for the table. Short version: SeedVR2 3B gives the best face detail (face Laplacian 278 vs 15 for plain resizing and 54 for Real-ESRGAN x2plus) but costs about 90 s per video second. SPAN x2 in a streaming pipeline costs about 20 s per 30 s clip.

## 7. LoRA rounds (single-identity, consented data)

Each round was tested on 10 s clips at about 576p with no image or audio references (three clips in rounds 0-1, five later). Real footage scores 0.81 look and 0.72 voice on this scale.

| Round | Change | Look | Voice | Outcome |
|---|---|---|---|---|
| 0 | no LoRA, spoken line | -0.05 | 0.16 | baseline |
| 1 | LoRA, epoch 1 | 0.15 | - | gibberish was a prompt problem: no spoken words. With the spoken line in the model's prompt grammar, WER was 0 |
| 2 | audio encoder on, 348 voice clips | 0.23 | 0.53 | keep. Round 1 had cached clips without audio, so no voice was learned |
| 3 | reference distillation, epoch 2 | 0.34 | 0.59 | keep |
| 4 | 2.4x data (231 to 560 clips, 232 to 554 stills, 348 to 1064 voice clips), LoRA+ ratio 4 | 0.48 | 0.70 | keep. New-scene probe 0.52 vs 0.30 for round 3, so not memorisation |
| 5 | tight face crops, distillation off | 0.50 | - | sync guard failed; not final |
| 6 | voice re-captioned | 0.53 | 0.65 | sync guard failed |
| 6B | model-card render graph | 0.56 | 0.675 | pass; standard from here |
| 7 | more weight on low-noise steps (60 % to 75 %) | -0.011 vs 6B | - | failed its +0.03 test; training stopped |

A three-clip transfer probe scored look 0.19 on Omni and 0.22 on the training base, under different render settings. The size of any transfer penalty is open.

Training speed: with 32 of 50 blocks streamed from system RAM, the step took 2.29 s and the card drew 122 W at "97 %" utilisation. Streaming 16 blocks cut it to 1.64 s (-28 %). Card memory rose from 13.8 GB to 19.1 GB. Loss stayed in the same band (0.343-0.386 against an epoch mean of 0.382). A round took about 72 min: 55 % training, 34 % test renders.

Review findings that changed numbers:

- Self-training that picks the best samples ranked them with hold-out references. That leaks the test set. We switched to train-only references.
- A face-weight flag did nothing, because distillation already covered every still step.
- Baselines must use the same attention backend as the arm they are compared with.
- A teacher probe scored 0.48 against about 0.52 for the round-4 student. Distillation could only pull identity down, so we turned it off.
- The "sync" metric measured delay, not whether the mouth followed the audio. We retired it as a gate.

## 8. Voice post step

Seed-VC v1 converts the generated speech toward the real voice. It uses the existing speech as the timing guide, so it does not shift timing.

Settings: 30 diffusion steps, length-adjust 1.0, target = a 20 s reference cut from training clips only. Run on CPU. After conversion we trim or pad to the exact input sample count.

| 60 s test audio | Samples | Speaker match | WER vs script | Time |
|---|---|---|---|---|
| original | 2,877,440 | 0.7474 | 1.1 % | - |
| UniPASE restoration only | 2,877,440 | 0.7384 | 1.1 % | 110 s |
| Seed-VC v1 | 2,877,440 | 0.7801 | 2.8 % | 420 s |
| Seed-VC v1 + UniPASE | 2,877,440 | 0.7604 | 1.7 % | 420 s + about 90 s |

- Sample count identical to the input in every output.
- Envelope lag between -8 and +6 ms, median -2 ms, no drift.
- SyncNet offset unchanged (-2 frames at 25 fps) in three windows, before and after.
- On a joined 56 s cut: speaker match 0.717 to 0.760, WER unchanged at 0.56 %, DNSMOS 3.50 to 3.43 (a small drop), SyncNet offset the same 40 ms in four windows.
- UniPASE after conversion lost most of the gain. We dropped it.
- A mispronounced proper noun stayed mispronounced. Fix names in the prompt.
- Output is 22.05 kHz, so nothing above 11 kHz is added.
- Not tested by ear. No metric here measures "robotic" directly. A blind A/B listen is still needed.

## 9. Idle GPU lessons

Most lost GPU time had nothing to do with kernels.

| Cause | Cost | Fix |
|---|---|---|
| Job script without the executable bit | 55 min idle | check mode before queueing |
| Loop whose SSH calls read the job list from stdin | 2 min idle | `ssh -n` |
| Blocking file upload inside the GPU loop | idle GPU | upload in background |
| Foreign vision model on the GPU | 10 s render took 668 s instead of about 265 s | GPU guard and reservation file |
| Pause flag fired one epoch early | 19 min idle | step-based pause |
| Test ComfyUI holding about 16 GB with nothing queued | blocked next job | the stage that starts a server stops it |
| Subprocess timeout of 900 s aborted a render set | 15 min | render the fixed set first |

## 10. Learnings

- Compute the compute-only bound of a step before tuning. Ours ran at 93 % of it. A faster kernel could save little. Doing less work could save a lot.
- Cut attention work first. It gave the largest speed-up of every option we tried.
- A faster backend is not a win if the gate fails. Gate on WER, not only on look.
- Check how a prompt is shaped before blaming a model. The "gibberish LoRA" was a prompt with no words.
- Run every detector on real footage before you trust it. Two of ours failed that check (see QA.md).
- Score both arms of a comparison with the same attention backend.

## 10. Clean native speech over 30 s (the published samples)

Goal: the model itself speaks every line cleanly for 30 s, with no TTS and no lip-sync model. Pass rule: WER at most 3 % in EVERY 5 s window (about zero errors per window at ~14 words per window).
All arms: FastH3 Omni int8, card graph (guidance 1, Euler, 8 steps), realism LoRA 0.3, no identity LoRA, a stock voice reference, 719 frames, chunked FFN 16384, `--reserve-vram 4`. Render time = ComfyUI execution time on our one 24 GB GPU. n = 1 per row.

| Arm | Change | Seed | Render | WER | Per-window WER (0-5 ... 25-30 s) | Pass |
|---|---|---|---|---|---|---|
| baseline podcast | 1280x720, VSA 10 % | 7329 | 544 s | 19.8 % | 0, 0, 0, .50, .85, 0 | no |
| podcast | 768x432, dense attention | 7329 | 506 s | 7.0 % | 0, 0, .23, .25, 0, 0 | no |
| **podcast** | **768x432, VSA 10 %** | 7329 | **163 s** | **0.0 %** | 0, 0, 0, 0, 0, 0 | **yes (published)** |
| podcast | 768x432, VSA 10 % | 7330 | 163 s | 4.7 % | 0, 0, 0, 0, .57, 0 | no |
| tutor | 768x432, VSA 10 % | 7329 | 165 s | 6.5 % | 0, 0, 0, .55, 0, 0 | no |
| **tutor** | 768x432, VSA 10 % | 7330 | 167 s | 3.2 % | 0, 0, 0, .27, 0, 0 | no (published as the best take) |
| podcast | 1024x576, VSA 10 % | 7329 | 288 s | 0.0 % | all 0 | yes, but 14 GB spilled into shared memory |
| podcast | 768x432, VSA + temporal RoPE "yarn" change | 7329 | 164 s | 27.9 % | errors from 0 s | no |
| tutor | 768x432, VSA + temporal RoPE "yarn" change | 7329 | 165 s | 16.1 % | 0, .35, .50, 0, 0, 0 | no |

Readings:
- VSA does not hurt speech here. Code read: under VSA the audio query rows stay dense and read all keys; only video query rows are sparse. Same seed: dense 7.0 % vs VSA 0.0 %. This checkpoint was distilled with VSA.
- Fewer tokens (768x432) helped but did not fix it: 3 of 4 takes at 768x432 still failed one window.
- Every failing window in every take started at 14.5 s or later. 15 s is the trained length. The temporal RoPE component k = 8 has a period of about 15.7 s (`onegpu_longvideo.frames.rope_time_periods`), so gaps past ~15 s are out of the trained range. This fits the onset. It is a hypothesis, not a proof.
- Stretching the temporal RoPE ("yarn" on components k >= 8) made speech much worse. Do not use it as a fix.
- Voice post (Seed-VC) on both published takes: the converted voice is kept on both (27 Nov 2026). Podcast: speaker match 0.616 to 0.789, WER 0 in every 5 s window, SyncNet offsets -1/-1/-1 unchanged. Tutor: first pass (30 steps, cfg 0.7, two reference clips) failed the sync guard (offsets 0/-1/0 to -1/-1/0). One change at a time, CPU, n = 1 each: same settings again, 50 steps, cfg 0.5 all failed the same way; a neutral-only reference clip passed (offsets 0/-1/0 unchanged, WER 0 in all windows, speaker match 0.613 to 0.732 against that clip, DNSMOS OVR 3.04 to 3.23, 269 s on CPU). The reference differs between rows, so scores are not comparable across them. The guard compares whole-frame SyncNet offsets at 25 fps; a one-frame flip may be measurement noise (untested).
- Published potter documentary (no speech): 1280x720, VSA 10 %, render 541.5 s, SPAN x2 upscale 20.0 s.
