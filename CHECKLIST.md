# Checklist (v0.1.0)

## Done
- [x] Package `src/onegpu_longvideo/`: graph builder, ComfyUI client, render CLI (`olv`), streaming SPAN upscaler, QA stage 1 (per-window WER, identity, gesture repeat, seams, AV offset) and stage 2 (VLM on flagged windows), voice post (sample-exact, lag, WER guard), camera check, timing and profiler helpers.
- [x] Custom ComfyUI nodes `comfyui_nodes/onegpu_nodes.py` (chunked FFN, Sage on video rows, temporal RoPE patch).
- [x] Recipes and exact prompts for the three samples plus one cinematic example.
- [x] Docs: EXPERIMENTS, PERFORMANCE, QA, LIMITATIONS.
- [x] 49 CPU unit tests, ruff, CI on Python 3.10 and 3.12.
- [x] Three sample videos in `media/` and a GitHub Pages page.
- [x] Secret, path and forbidden-term scans before the first push.

## TODO
- [ ] End-to-end GPU test of `olv render` from this package (the graph matches our production graph node for node, but this exact code path has not rendered yet).
- [ ] Stage-1 feature extraction (face embeddings, pose, mouth opening) as a packaged command; the detectors take per-frame series today.
- [ ] Score QA stage 2 against a labelled set.
- [ ] Chain mode CLI (`workflow.add_video_guide` exists; no driver).
- [ ] Verify the licences of SPAN weights, Seed-VC and QA models and list them in LIMITATIONS.md.
- [ ] Tests on other GPUs and on Linux.
