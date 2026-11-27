"""Two-stage QA. Stage 1: cheap detectors on every frame and word (CPU). Stage 2: a vision-language model looks
only at the windows stage 1 flagged, plus a few random unflagged windows to measure what stage 1 misses."""
