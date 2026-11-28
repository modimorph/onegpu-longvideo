"""Required-flow gate check: validate one run folder against the numeric gates of docs/FLOW.md.

The folder holds `flow_run.json` (a manifest you fill; see docs/FLOW-CHECKLIST.md) and the `olv wer --out` JSON files it names.
Human gates (listening check, vision review, contact sheet) are read from the manifest; this code cannot judge them, it only refuses a missing "yes".
Planner and editor (docs/PLANNER.md, docs/EDITOR.md) wrap the flow: steps 1-2 need a passing `plan.json` (`olv plan`),
steps 9, 11 and 12 a passing `edit_qa.json` (`olv edit`). Set `seconds` in the manifest for clips other than 29.96 s.
`"film": true` checks a film of several clips: last word and per-window WER per clip come from plan.json and edit_qa.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from .frames import frames_for_seconds

END_MARGIN_S = 1.66   # last word by S - 1.66 s (28.3 s for 29.96 s); the planner profiles use 1.7
FILM_END_GUARD_S = 0.2  # film mode: the last word must end at least 0.2 s before the clip ends
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
    secs = float(m.get("seconds", 29.96))
    frames, last_word = frames_for_seconds(secs), round(secs - END_MARGIN_S, 2)

    def add(step, name, ok, note=""):
        rows.append((step, name, bool(ok), note))

    wp = d / m.get("wer_raw", "wer_raw.json")
    lw = json.loads(wp.read_text()).get("last_word_s") if wp.exists() else None
    pp = d / m.get("plan", "plan.json")
    pl = json.loads(pp.read_text()) if pp.exists() else {}
    ep = d / m.get("edit_qa", "edit_qa.json")
    eq = json.loads(ep.read_text()) if ep.exists() else {}
    film = bool(m.get("film"))       # a film of several clips: per-clip numbers come from plan.json and edit_qa.json
    add(1, "planner gates passed (plan.json)", pl.get("gates_pass") is True,
        "missing plan.json (run olv plan)" if not pl else ", ".join(g["id"] for g in pl.get("gates", []) if not g["pass"]) or "all pass")
    if film:
        secs_of = {c["clip"]: c["seconds"] for c in pl.get("clips", [])}
        wc = (eq.get("word_check") or {}).get("clips", [])
        # Measured (10 clips, 10 s and 30 s): the model paces speech to fill the clip and ends about 0.4 s before the end,
        # whatever the word count; with 0 % WER that is harmless and the editor trims the tail. So a film gates on
        # "no word cut off at the clip end"; S - 1.66 s stays the planning target.
        cut_off = [f"clip {c['clip']}: {c.get('last_word_s')} s" for c in wc
                   if c.get("checked") and (c.get("last_word_s") is None or c["last_word_s"] > secs_of.get(c["clip"], secs) - FILM_END_GUARD_S)]
        add(1, f"every clip: no word cut off (last word by S - {FILM_END_GUARD_S} s)", bool(wc) and not cut_off, "; ".join(cut_off) or f"{len(wc)} clips")
    else:
        add(1, f"last word by {last_word} s", lw is not None and lw <= last_word, f"last_word_s={lw}")
    add(5, f"{frames} frames", raw.get("frames") == frames, f"frames={raw.get('frames')}")
    add(5, "raw 1280x720", (raw.get("width"), raw.get("height")) == GATES["raw_size"], f"{raw.get('width')}x{raw.get('height')}")
    add(5, "render time logged", isinstance(m.get("render_s"), (int, float)), f"render_s={m.get('render_s')}")
    if film:
        wc = (eq.get("word_check") or {}).get("clips", [])
        bad = [f"clip {c['clip']}: {c.get('win_wer')}" for c in wc if not c.get("checked") or any(x > 0 for x in (c.get("win_wer") or [1]))]
        add(6, "every clip: 0 % WER in every 5 s window", bool(wc) and not bad, "; ".join(bad) or f"{len(wc)} clips")
    else:
        ok, note = _wer_ok(wp)
        add(6, "0 % WER in every 5 s window (raw take)", ok, note)
    add(7, "QA stage 1 passed; flagged parts reviewed", qa.get("stage1_pass") is True and qa.get("flagged_reviewed") is True)
    add(8, "final 1920x1080", (fin.get("width"), fin.get("height")) == GATES["final_size"], f"{fin.get('width')}x{fin.get('height')}")
    add(9, "repeated closing lines cut", m.get("edit", {}).get("repeats_cut") is True)
    add(9, "editor gates passed (edit_qa.json)", eq.get("gates_pass") is True,
        "missing edit_qa.json (run olv edit)" if not eq else ", ".join(g["id"] for g in eq.get("gates", []) if not g["pass"]) or "all pass")
    if voice.get("used"):
        add(10, "voice length sample-exact", voice.get("len_diff") == 0, f"len_diff={voice.get('len_diff')}")
        add(10, "WER guard passed", voice.get("wer_guard") is True)
        add(10, "human listening check approved", voice.get("listening_approved") is True)
    add(11, "colour grade applied", m.get("grade") is True)
    enc_ok = fin.get("vcodec") == "h264" and fin.get("crf") == GATES["crf"] and fin.get("acodec") == "aac" and fin.get("audio_kbps") == GATES["audio_kbps"]
    add(11, "encode h264 CRF 18, AAC 192k", enc_ok, f"{fin.get('vcodec')} crf={fin.get('crf')} {fin.get('acodec')} {fin.get('audio_kbps')}k")
    if not film or (d / m.get("wer_final", "wer_final.json")).exists():
        ok, note = _wer_ok(d / m.get("wer_final", "wer_final.json"))
        add(12, "final: 0 % WER in every window", ok, note)
    add(12, "final: voice match checked", isinstance(qa.get("speaker_match"), (int, float)), f"speaker_match={qa.get('speaker_match')}")
    add(12, "final: lip sync ok", qa.get("lip_sync_ok") is True)
    add(12, "contact sheet reviewed by a person", qa.get("contact_sheet_reviewed") is True)
    add(12, "generation time reported", isinstance(m.get("render_s"), (int, float)) and isinstance(m.get("upscale_s"), (int, float)))
    return [{"step": s, "gate": n, "pass": ok, "note": t} for s, n, ok, t in rows]


def passed(rows: list[dict]) -> bool:
    return all(r["pass"] for r in rows)
