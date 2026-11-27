# Rules for coding agents in this repo

- Run `pytest -q` and `ruff check src tests` before and after every change. Both run on CPU.
- Pure logic lives in `src/onegpu_longvideo/` and must stay importable without torch, CUDA or ComfyUI. Heavy imports go inside functions.
- Render settings belong in `configs/*.yaml`. Do not hard-code machine paths, hostnames or model folders.
- A new speed or quality claim needs a measured number with its context (resolution, frames, steps, GPU) in `docs/EXPERIMENTS.md` or `docs/PERFORMANCE.md`. Write "untested" when there is none.
- One GPU job at a time. Free ComfyUI VRAM (`ComfyClient.free()`) before an upscale or QA job on the same GPU.
- Never commit model weights, datasets, credentials, `.env` files, or any real person's likeness or voice.
- Every video follows the 12-step flow in `docs/FLOW.md`. Finish with `olv flow-check <run folder>`. Do not skip a step or report a result from a run that skipped one.
