"""Required-flow gate check: validate one run folder against the numeric gates of docs/FLOW.md.

The folder holds `flow_run.json` (a manifest you fill; see docs/FLOW-CHECKLIST.md) and the `olv wer --out` JSON files it names.
Human gates (listening check, vision review) are read from the manifest; this code cannot judge them, it only refuses a missing "yes".
"""
from __future__ import annotations

import json
from pathlib import Path

GATES = {"frames": 719, "raw_size": (1280, 720), "final_size": (1920, 1080), "last_word_s": 28.3, "crf": 18, "audio_kbps": 192}


def _wer_ok(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing {path.name}"
    w = json.loads(path.read_text()).get("win_wer")
    return (bool(w) and all(x == 0 for x in w)), f"win_wer={w}"


def check(run_dir: str | Path) -> list[dict]:
    d = Path(run_dir)
    m = json.loads((d / "flow_run.json").read_text())
    raw, fin, voice, qa = m.get("raw", {}), m.get("final", {}), m.get("voice") or {}, m.get("qa", {})
    rows: list[tuple[int, str, bool, str]] = []

    def add(step, name, ok, note=""):
        rows.append((step, name, bool(ok), note))

    wp = d / m.get("wer_raw", "wer_raw.json")
    lw = json.loads(wp.read_text()).get("last_word_s") if wp.exists() else None
    add(1, "last word by 28.3 s", lw is not None and lw <= GATES["last_word_s"], f"last_word_s={lw}")
    add(5, "719 frames", raw.get("frames") == GATES["frames"], f"frames={raw.get('frames')}")
    add(5, "raw 1280x720", (raw.get("width"), raw.get("height")) == GATES["raw_size"], f"{raw.get('width')}x{raw.get('height')}")
    add(5, "render time logged", isinstance(m.get("render_s"), (int, float)), f"render_s={m.get('render_s')}")
    ok, note = _wer_ok(wp)
    add(6, "0 % WER in every 5 s window (raw take)", ok, note)
    add(7, "QA stage 1 passed; flagged parts reviewed", qa.get("stage1_pass") is True and qa.get("flagged_reviewed") is True)
    add(8, "final 1920x1080", (fin.get("width"), fin.get("height")) == GATES["final_size"], f"{fin.get('width')}x{fin.get('height')}")
    add(9, "repeated closing lines cut", m.get("edit", {}).get("repeats_cut") is True)
    if voice.get("used"):
        add(10, "voice length sample-exact", voice.get("len_diff") == 0, f"len_diff={voice.get('len_diff')}")
        add(10, "WER guard passed", voice.get("wer_guard") is True)
        add(10, "human listening check approved", voice.get("listening_approved") is True)
    add(11, "colour grade applied", m.get("grade") is True)
    enc_ok = fin.get("vcodec") == "h264" and fin.get("crf") == GATES["crf"] and fin.get("acodec") == "aac" and fin.get("audio_kbps") == GATES["audio_kbps"]
    add(11, "encode h264 CRF 18, AAC 192k", enc_ok, f"{fin.get('vcodec')} crf={fin.get('crf')} {fin.get('acodec')} {fin.get('audio_kbps')}k")
    ok, note = _wer_ok(d / m.get("wer_final", "wer_final.json"))
    add(12, "final: 0 % WER in every window", ok, note)
    add(12, "final: voice match checked", isinstance(qa.get("speaker_match"), (int, float)), f"speaker_match={qa.get('speaker_match')}")
    add(12, "final: lip sync ok", qa.get("lip_sync_ok") is True)
    add(12, "generation time reported", isinstance(m.get("render_s"), (int, float)) and isinstance(m.get("upscale_s"), (int, float)))
    return [{"step": s, "gate": n, "pass": ok, "note": t} for s, n, ok, t in rows]


def passed(rows: list[dict]) -> bool:
    return all(r["pass"] for r in rows)
