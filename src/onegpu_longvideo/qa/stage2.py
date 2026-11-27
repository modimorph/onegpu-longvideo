"""Stage 2: ask a vision-language model about the flagged windows only.

Works with any OpenAI-compatible chat endpoint that accepts images (for example a local llama.cpp or vLLM server).
Set --vlm-url or VLM_URL. Speech-only flags (wrong words, invented words) are NOT sent: the VLM cannot hear.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

AUDIO_ONLY = {"wrong_words", "invented_words", "av_offset"}

PROMPT = ("You check an AI-generated video for visual defects. The frames are consecutive, with time stamps. "
          "Detector evidence: {evidence}. Answer JSON only: {{\"confirm\": true|false, \"what\": \"...\", \"t_start\": s, \"t_end\": s}}.")


def extract_frames(video: str, t0: float, t1: float, fps: float = 8.0, width: int = 512) -> list[tuple[float, bytes]]:
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", video, "-vf", f"fps={fps},scale={width}:-2",
                        "-q:v", "4", str(Path(d) / "f_%04d.jpg")], check=True)
        files = sorted(Path(d).glob("f_*.jpg"))
        return [(t0 + i / fps, f.read_bytes()) for i, f in enumerate(files)]


def build_message(frames: list[tuple[float, bytes]], evidence: dict) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": PROMPT.format(evidence=json.dumps(evidence))}]
    for t, jpg in frames:
        content.append({"type": "text", "text": f"t={t:.3f} s"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}})
    return [{"role": "user", "content": content}]


def ask(messages: list[dict], url: str | None = None, model: str = "local", max_tokens: int = 400, timeout: int = 600) -> dict:
    url = (url or os.environ.get("VLM_URL", "http://127.0.0.1:8090")).rstrip("/")
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = json.loads(r.read())["choices"][0]["message"]["content"]
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except ValueError:
        return {"confirm": None, "raw": text[:500]}


def visual_flags(flags: list[dict]) -> list[dict]:
    return [f for f in flags if f.get("type") not in AUDIO_ONLY]
