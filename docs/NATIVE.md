# Native code: what we added, what it gained, what it did not

Measured on one RTX 4090 (24 GB), 28 Nov 2026. The recipe is identical to the pure-Python flow in every row: 1280x720, 243 frames, 8 steps, guidance 1.0, realism LoRA 0.3, VSA keep 10 %, seed 7329. Single runs per lane. The quality gates are the same as for any change: WER 0 per 5 s window (or equal to main), voice match (ECAPA) and face match (ArcFace) within 0.02 of main, 243 frames, audio equal to video length.

## a. Why

The goal was a 2x faster video than the all-Python flow, from non-Python code (Rust, C++/CUDA, Triton, TensorRT, int8 and INT4 kernels), at equal quality. Speed matters because one take costs about 3 minutes of one GPU. Faster takes mean more takes per hour and a cheaper repo to run. Result: **1.17-1.19x, not 2x**.

## b. Result vs the pure-Python flow

| Lane | Python flow (s) | This branch (s) | Speed-up | Gates |
|---|---|---|---|---|
| Cold 10 s clip | 168.18 | 143.20 | 1.174x | WER 0/0, ECAPA 0.779 vs 0.770 |
| Cold 10 s clip (earlier, 2 runs each) | 177.04 | 158.74 | 1.115x | all pass |
| Warm, same prompt re-render | 169.23 | 126.93 | 1.333x | all pass |
| 1-min film (6 x 10 s) | 1148.09 | 968.76 | 1.185x | WER equal window by window; ECAPA and ArcFace within 0.02 |
| 4-bit DiT (not shipped) | 177.04 | 140.95 | 1.256x | face match 0.683 vs 0.859: fail |

Why not 2x. Half of 177 s is 88.5 s. The branch spends 54.7 s outside the 8 DiT forwards, which leaves 4.2-5.0 s per step. Measured per step: int8 GEMMs 5.08 s, sparse attention 4.1 s, other ops about 2.4 s. The int8 GEMMs alone exceed the budget. INT4 is the only 2x tensor path on this GPU, and every 4-bit variant tested changed the face (full W4A4 -0.176 ArcFace, FFN-only -0.061).

## d. What helped (kept) and what did not (dropped)

Kept:
- int8 video VAE decoder: decode 30.4 s to 13.2 s.
- `--fast fp16_accumulation` server flag: render 160.4 s to 135.7 s (with the decoder).
- Readiness wait instead of two fixed 3 s sleeps: 177.0 s to 169.2 s (harness).
- Upscale x2 and encode inside the ComfyUI graph: 158.7 s to 147.1 s.
- Upscale overlap (pinned memory, copy stream), bit-exact: pipeline -12 %.
- Rust WER alignment: 23-53x on Windows, under 2 ms end to end.
- History poll every 0.5 s: about 2 s per render.

Dropped:
- VRAM reserve 1.5 GB: no gain, near spill.
- Full and FFN-only W4A4: fail the face gate. W4 weights with int8 activations: slower.
- VAE tile sizes: at most 0.65 s.
- Pre-merged LoRA: the patch costs about 0 s.
- Triton frame post-process: 5.7x faster kernel, net loss once JIT time is counted.
- NVDEC decode (not bit-exact), TorchScript cache of the upscaler (not bit-exact), cudnn/channels_last/half interpolation (no gain), Rust envelope (numpy is faster).

## e. Untested

- A same-selection sparse-attention kernel for Ada (estimate 0.8-1.5 s per forward).
- Scoped `torch.compile` or Triton fusion of norm and residual ops, and CUDA graphs.
- A resident or prefetched text encoder: measured 14.2 s cold conditioning vs 3.8 s with the encoder loaded, so about 10 s is loading. Not built.
- Faster reference-encode and conditioning caches.
- The upscaler as a TensorRT engine.
- Calibrated INT4 with outlier handling.
- A W4A8 checkpoint for faster loading (no compute gain).
- A warm-server film for both arms, and a matched pair on the very last commit.

## Defaults and fallbacks

The defaults need the int8 VAE file and the in-graph upscale node. To fall back, set `upscale_in_graph: false` and use the fp16 VAE. See the README install section. Build scripts for the Rust core are under `tools/pc/`.
