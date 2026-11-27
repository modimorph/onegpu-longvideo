"""Minimal ComfyUI HTTP client: queue a graph, wait, download the output, read ComfyUI's own execution time.

Set the server with --server or the COMFY_URL environment variable (default http://127.0.0.1:8188).
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def default_url() -> str:
    return os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")


class ComfyClient:
    def __init__(self, url: str | None = None, timeout: float = 30.0):
        self.url = (url or default_url()).rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    def _req(self, path: str, data: dict | None = None):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(self.url + path, data=body, headers={"Content-Type": "application/json"} if body else {})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            raw = r.read()
        return json.loads(raw) if raw else {}

    def queue(self, graph: dict) -> str:
        r = self._req("/prompt", {"prompt": graph, "client_id": self.client_id})
        if r.get("node_errors"):
            raise RuntimeError(f"ComfyUI rejected the graph: {json.dumps(r['node_errors'])[:800]}")
        return r["prompt_id"]

    def history(self, prompt_id: str) -> dict:
        return self._req(f"/history/{prompt_id}").get(prompt_id, {})

    def wait(self, prompt_id: str, timeout_s: float = 3600, poll_s: float = 5.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            h = self.history(prompt_id)
            st = h.get("status", {})
            if st.get("completed"):
                return h
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                raise RuntimeError(f"ComfyUI execution error: {json.dumps(msgs)[:800]}")
            time.sleep(poll_s)
        raise TimeoutError(f"prompt {prompt_id} not done after {timeout_s} s")

    def free(self) -> None:
        """Unload models and free VRAM (before an upscale or QA job on the same GPU)."""
        self._req("/free", {"unload_models": True, "free_memory": True})

    def download(self, item: dict, dest: str | Path) -> Path:
        q = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")})
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{self.url}/view?{q}", dest)
        return dest

    def run(self, graph: dict, dest: str | Path, timeout_s: float = 3600) -> dict:
        pid = self.queue(graph)
        h = self.wait(pid, timeout_s)
        item = first_output(h)
        path = self.download(item, dest)
        return {"prompt_id": pid, "file": str(path), "exec_s": exec_seconds(h), "remote_file": item["filename"]}


def first_output(history_entry: dict) -> dict:
    for node_out in history_entry.get("outputs", {}).values():
        for key in ("videos", "gifs", "images", "audio"):
            if node_out.get(key):
                return node_out[key][0]
    raise RuntimeError("no output file in the ComfyUI history entry")


def exec_seconds(history_entry: dict) -> float | None:
    """ComfyUI's own execution_start -> execution_success time (server side, excludes queue and transfer)."""
    ts = {}
    for m in history_entry.get("status", {}).get("messages", []):
        if isinstance(m, list) and len(m) > 1 and isinstance(m[1], dict):
            ts[m[0]] = m[1].get("timestamp")
    if ts.get("execution_start") and ts.get("execution_success"):
        return round((ts["execution_success"] - ts["execution_start"]) / 1000.0, 1)
    return None
