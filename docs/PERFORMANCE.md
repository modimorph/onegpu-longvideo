# Performance

Profile, attention options, memory notes, upscaler comparison and end-to-end timings.

Our runs used one 24 GB consumer GPU (RTX 4090) with 128 GB of system RAM over PCIe 4.0. Numbers are single runs unless stated. "Est" means we did the arithmetic and did not run it.

## 1. Where the time goes

The cost of a pixel: attention is quadratic in token count. Double the tokens and attention work goes up 4x.

| 30 s shape | Tokens (S) | Dense time | Note |
|---|---|---|---|
| 768x432, 719 frames | about 74k | 504-515 s (measured) | sampling about 89 % of wall time |
| 1024x576 | about 125k | 1313 s (measured) | 2.55x the time for 1.78x the pixels |
| 1280x720 | about 198k | about 54 min (projected from 6 of 8 steps at 383 s/step) | stopped at step 6 |
| 1920x1088 | about 436k | about 4 h (est) | about 31 GB activations (est) |

### Compute-only bound

For the 768x432, 30 s run, one forward pass covers about 74,000 tokens. That is about 7.9e15 attention operations and 2.9e15 int8 operations in the other layers per step. At the card's peak rates, run one after the other, that is 52.3 s. The measured step took 56.3 s: 93 % of the bound. A perfect kernel would save about 4 s per step (7 %). The kernel was not the problem. The amount of work was.

### Profile

PyTorch profiler, 10 s clip at 720p, three sampling steps, 155 s window.

| Item | Share |
|---|---|
| Attention (one fused kernel) | 87.2 % of summed kernel time |
| int8 matrix multiplies | 8.9 % |
| Unpacking int8 weights to 16-bit | 2.6 % |
| Weight copies from system RAM | 1.9 % of the window |
| GPU idle | 2.9 % of the window |

If every non-attention kernel took zero time, summed kernel time would drop about 13 % (a 1.15x ceiling). Attention is the only target worth the effort.

### Time model (fit to one run)

Per forward pass: t = 9.17e-9 x S^2 x f + 7.7e-5 x S seconds, fitted to the 768x432 30 s dense run. f is the attention cost relative to dense. At 720p, 10 s with VSA we measured f of about 0.15 (arithmetic from 136.3 s minus about 45 s of non-sampling time).

Projection for 30 s with VSA at f = 0.15-0.30 (est, not run):

| Size | Total | Fits 24 GB |
|---|---|---|
| 768x432 | 2.7-3.7 min | yes |
| 1024x576 | 5.1-8.3 min | yes |
| 1280x720 | 10.2-18 min | yes, with weight streaming |
| 1536x864 | 17-30 min | no (sparse producer needs about 20 GB) |
| 1920x1088 | 39-75 min | no |

Our measured 30 s run at 720p with VSA was 637 s generate and decode, inside the 10.2-18 min band.

## 2. VSA vs SageAttention vs torch.compile

Fixture: 10 s, 1280x720, 8 steps, same seed. Dense reference: 462.8 s.

| Option | Time | vs dense | Verdict |
|---|---|---|---|
| Dense (PyTorch SDPA) | 463 s | 1x | reference |
| **VSA, keep 10 %** | 136.3 s | 3.4x (2.7x if a cached text encode of up to 34 s is charged) | keep |
| SageAttention on video rows | 322 s | 1.44x | slower than VSA; both replace the same attention call, so pick one |
| SageAttention on every row (5 paired clips, 10 s) | 159 s vs 367 s mean | 2.31x (1.82x without a contaminated pair) | gate failed: one clip WER +0.263 |
| torch.compile | failed at step 1 | - | weight streaming asks for a CUDA stream property that does not exist |

VSA at 30 s: 637 s against a projected 3265 s dense run, about 5x. The gain grows with length because attention is a larger share of each step.

### How VSA works here

- Video tokens are grouped into cubes (4 x 4 x 4 tokens in the model's training mode; ComfyUI's sparse node uses 64-token cubes and an 8-bit kernel, where training used 128-token tiles).
- A cheap first pass compares each cube with a one-vector summary of every other cube and keeps the top 10 %.
- Text, reference images and audio rows keep full attention, so the prompt and the voice are never skipped.
- The distilled checkpoint was trained at 90 % video sparsity and carries the small extra layers for the first pass. ComfyUI's default graph ignored them and ran dense attention.
- We start sparse at step 0 (every forward is sparse) and require the sparse-producer path in the log.

### Cost of VSA

- Face sharpness fell 19 % on the test clip (969 vs 1200). One clip cannot say which difference from training caused it.
- Quality was compared on one 10 s clip per arm (look, WER, AV offset). We did not run a gate across many seeds.
- Why 3.4x and not 10x (likely, untested): the scoring pass costs time, cubes are padded, text/image/audio rows run in full, and the other 13 % of the work does not shrink.

### Caches and other options

| Option | Result |
|---|---|
| EasyCache 0.2 (10 s, 576x576) | sampling 70 s to 52 s, SSIM 0.967. Off for long speech. |
| 6 steps (10 s, 576x576) | 43 s, SSIM 0.84. Borderline. |
| 12 steps instead of 8 (15 s, 1024x576) | 539 s vs 397 s, no measurable gain |
| Guidance 1.0 (model-card graph) vs 0.9 | about -45 % time per 10 s clip (141-167 s vs about 285 s) |
| Compile, TensorRT, fp8 GEMM | touch only the non-attention 10-15 %. Untested beyond the failed compile. |

## 3. Reserve-VRAM, offload and memory

- `--reserve-vram` (keeps part of card memory free): a three-step job took 207.5 s without it and 216.1 s with it (about 4 %, one pair). The run with the flag was also the profiled run, so the pair is weak evidence. We kept the original setting.
- Weight streaming (block offload to system RAM) costs under 2 % of time, because the copies overlap with compute. A 10 s VSA run at 720p streamed about 17 GB.
- Activations are the real limit. A 30 s render at about 576p (736 frames) reported 14 GB of shared GPU memory and was stopped at about 1750 s. We treat that spill like an out-of-memory failure.
- Native 1080p: a 5.2 s probe with VSA took 203 s and needed 21.1 GB on the card plus 8.8 GB shared. A memory model put 30 s at about 31 GB (est).
- A foreign process on the same GPU can make a 10 s render take 668 s instead of about 265 s. Run one GPU job at a time and guard with a reservation file.
- A test ComfyUI instance can hold about 16 GB of weights with nothing queued. Stop servers when the stage ends.
- Training (LoRA, same card): 32 of 50 blocks streamed gave 2.29 s/step at 13.8 GB; 16 streamed gave 1.64 s/step at 19.1 GB (-28 %), no OOM in 260 steps.

## 4. Upscaler comparison

4 s clip, 768x432 to 1920x1080, ComfyUI path.

| Upscaler | Time (4 s clip) | Per 30 s (extrapolated) | Quality note |
|---|---|---|---|
| Lanczos resize (CPU) | - | - | face Laplacian 15 |
| SPAN x2 | 18 s | 136 s in ComfyUI (est); about 20 s in the streaming pipeline (measured) | fast; less detail |
| Real-ESRGAN x2plus | 27 s | about 3.5 min (est) | waxy; face sharpness 54 |
| Real-ESRGAN general x4v3 | 31 s | - | - |
| RealPLKSR x4 | 49 s | - | - |
| SeedVR2 3B (int8) | 361 s (includes load) | about 44 min (est) | best detail; face sharpness 278 |
| FlashVSR+ Turbo int8 | failed | - | Triton found no C compiler on Windows; the sdpa fallback stopped after 145 s (one attention mask alone was estimated at 33 GiB) |

SeedVR2 on a 60 s cut: 5324 s, about 90 s per video second.

Sharpness is a Laplacian-variance score on the face region. It is a proxy, not a quality verdict. We judged final output by eye as well.

### Streaming SPAN x2 pipeline

Three overlapping stages: one thread decodes frames with ffmpeg, the GPU runs SPAN x2 in fp16 on batches, a third thread feeds the hardware video encoder (NVENC). 720p input becomes 1440p, then a bicubic resize to 1920x1080. Source audio is copied. Result: a 30 s clip in about 20 s (36 fps). The ComfyUI and pipeline timings use different inputs and boundaries, so the ratio does not split the gain by stage.

```python
# reader thread: ffmpeg -f rawvideo -pix_fmt rgb24 -> qin (Queue(4))
# writer thread: qout (Queue(4)) -> ffmpeg -c:v h264_nvenc, source audio copied
model = ModelLoader().load_from_file(span_x2).model.eval().cuda().half()
with torch.inference_mode():
    while (x := qin.get()) is not None:             # uint8 batch, N x H x W x 3
        t = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).half().div_(255)
        y = model(t)                                 # 1280x720 -> 2560x1440
        y = F.interpolate(y.float(), size=(1080, 1920),
                          mode="bicubic", antialias=True).clamp_(0, 1)
        qout.put(y.mul_(255).round_().byte().permute(0, 2, 3, 1).contiguous().cpu().numpy())
```

### Not tested

FlashVSR+ at 1080p on our card (paper-based estimates only: 2.2-3.2 min per 720 frames, est), NVIDIA RTX VSR, RelayVSR, SeedVR2 on a face crop only with a feathered paste, and a two-pass render (early steps at low resolution, a latent upscale, last steps at 720p).

## 5. End-to-end timings

30 s, 720p, VSA keep 10 %, 8 steps, then streaming SPAN x2 to 1080p. The clock runs from prompt to the 1080p file on the render machine.

| Run | Generate + decode | Load + transfer | Upscale | Total | Quality |
|---|---|---|---|---|---|
| 1 | 637 s | about 100 s | about 20 s | 762 s | WER 0.133; 7 flags |
| 2 | 635 s | about 100 s | about 20 s | 758 s | scorer crashed; picked by eye |
| 3 | 604 s | 110 s | 20 s | 744 s | WER 0.365; 9 flags; failed quality |

Checks, voice conversion and review run after the clock stops.

Other end-to-end reference points (generation only):

| Setup | Time |
|---|---|
| 768x432, 30 s, dense | 504-524 s |
| 1024x576, 30 s, dense | 1313 s |
| 60 s one-shot, 640x352 | 667.6-870 s (speech fails; see EXPERIMENTS.md) |
| 5 x 311-frame chain, 640x352 | 447 s |
| Stage 1 QA, CPU only | about 3.5 min per 30 s clip in our notes; one logged run read 588.5 s total |
| Seed-VC voice step, CPU, 60 s audio | 420 s |

Single runs. Parts in the table do not sum exactly to the totals (run 3 leaves about 10 s unaccounted). Run totals were 762, 758 and 744 s. A 10 s render took 668 s instead of about 265 s when another process shared the GPU.
