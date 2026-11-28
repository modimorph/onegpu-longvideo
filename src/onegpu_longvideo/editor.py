"""Studio editor (flow steps 9, 11, 12): from rendered clips and the film plan to a finished, measured film.

Stages (docs/EDITOR.md):
  E1 word check     per-clip WER per 5 s window from `olv wer --out` files; a window over the limit = gibberish = FAIL.
  E2 cut list       IN/OUT per clip around the speech, plus cuts of repeated runs, stutters and fillers the script does not have.
                    Cut points sit in the silence between words; every audio boundary gets a short fade.
  E3 colour match   every clip matched to the anchor grade (Reinhard transfer in Lab, one LUT per clip).
  E4 assemble       punch-in where the plan asks, optional lower third, frame-exact concat, x264 encode.
  E5 loudness       gain to the profile target (YouTube: -14 LUFS) plus a peak limiter, true peak measured on the AAC file.
  E6 review         7x6 contact sheet (42 frames evenly spaced) and QA numbers with pass/fail gates.
Needs ffmpeg and ffprobe on PATH. No GPU. Source clips are never modified.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from . import colour as C

FILLERS = {"um", "umm", "uh", "uhh", "erm", "er", "hmm", "mm"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9']", "", s.lower())


# ---------------------------------------------------------------- inputs
def find_clips(clips_dir: str | Path, wer_dir: str | Path | None = None) -> list[dict]:
    """clipN/video.mp4 (layout of the production folders) or clipN.mp4; WER file <wer_dir>/clipN.wer.json, clipN/wer.json or clipN.wer.json."""
    d = Path(clips_dir)
    wd = Path(wer_dir) if wer_dir else None
    found = {}
    for p in list(d.glob("clip*/video.mp4")) + list(d.glob("clip*.mp4")):
        m = re.match(r"clip(\d+)", p.parent.name if p.name == "video.mp4" else p.stem)
        if m:
            found.setdefault(int(m.group(1)), p)
    out = []
    for n in sorted(found):
        p = found[n]
        cands = ([wd / f"clip{n}.wer.json"] if wd else []) + [p.parent / "wer.json", d / f"clip{n}.wer.json", d / f"clip{n}" / "wer_raw.json"]
        wer = next((w for w in cands if w.exists()), None)
        out.append({"clip": n, "video": str(p), "wer": str(wer) if wer else None})
    return out


def heard_words(wer_json: dict) -> list[dict]:
    """Word list with times from an `olv wer --out` file ('words': [{w, t0, t1}])."""
    return [{"w": w["w"], "t0": float(w["t0"]), "t1": float(w["t1"])} for w in wer_json.get("words", []) if _norm(w["w"])]


# ---------------------------------------------------------------- E1
def word_check(clips: list[dict], prof: dict, require: bool = True) -> dict:
    wc = prof["editor"]["word_check"]
    rows, errors = [], []
    for c in clips:
        if not c["wer"]:
            rows.append({"clip": c["clip"], "checked": False})
            if require:
                errors.append({"where": f"clip {c['clip']}", "type": "no_wer_file", "severity": "fail"})
            continue
        r = json.loads(Path(c["wer"]).read_text())
        wins = r.get("windows", [])
        bad = [w for w in wins if w["wer"] > wc["max_window_wer"]]
        gib = [w for w in wins if w.get("invented", 0) >= wc["gibberish_invented"]]
        rows.append({"clip": c["clip"], "checked": True, "wer": r.get("wer"), "win_wer": r.get("win_wer"),
                     "last_word_s": r.get("last_word_s"), "gibberish_windows": [w["window"] for w in gib]})
        for w in bad:
            errors.append({"where": f"clip {c['clip']} {w['window'][0]:g}-{w['window'][1]:g} s",
                           "type": "gibberish" if w in gib else "word_errors", "severity": "fail", "wer": w["wer"],
                           "action": "needs regeneration of this clip (never an automatic re-render)"})
    return {"clips": rows, "errors": errors}


# ---------------------------------------------------------------- E2
def find_flaws(words: list[dict], reference: str, earlier: list[str] | None = None, ngram: int = 4) -> list[dict]:
    """Repeated runs, stutters and fillers in the heard words that the reference text does not contain.

    earlier = heard tokens of the previous clip (a chained clip can replay the end of the last one)."""
    toks = [_norm(w["w"]) for w in words]
    ref = [_norm(x) for x in reference.split() if _norm(x)]
    ref_grams = Counter(tuple(ref[i:i + ngram]) for i in range(len(ref) - ngram + 1))
    ref_bigrams = Counter(zip(ref, ref[1:]))
    flaws, i = [], 0
    # stutters: the same word twice in a row with a short gap -> drop the first copy
    while i < len(words) - 1:
        if toks[i] == toks[i + 1] and words[i + 1]["t0"] - words[i]["t1"] < 0.8 and ref_bigrams[(toks[i], toks[i])] == 0:
            flaws.append({"kind": "stutter", "i0": i, "i1": i + 1})
            i += 2
        else:
            i += 1
    for i, t in enumerate(toks):
        if t in FILLERS and t not in ref:
            flaws.append({"kind": "filler", "i0": i, "i1": i + 1})
    # repeated runs: a run of >= ngram words heard more often than the reference has it -> cut the later copy
    ctx = list(earlier or [])
    seen = Counter(tuple(ctx[j:j + ngram]) for j in range(len(ctx) - ngram + 1))
    i = 0
    while i <= len(toks) - ngram:
        g = tuple(toks[i:i + ngram])
        if seen[g] >= max(1, ref_grams[g]) and g in seen:
            L = ngram
            flaws.append({"kind": "repeat", "i0": i, "i1": min(len(toks), i + L)})
            for k in range(i, min(len(toks), i + L) - ngram + 1):
                seen[tuple(toks[k:k + ngram])] += 1
            i += L
            continue
        seen[g] += 1
        i += 1
    return flaws


def cut_list(words: list[dict], duration: float, flaws: list[dict], prof: dict) -> dict:
    """Segments to keep [(a, b)] in clip time. Cut points sit in the gaps between words."""
    cu = prof["editor"]["cuts"]
    if not words:
        return {"segments": [[0.0, round(duration, 3)]], "cuts": [], "in": 0.0, "out": round(duration, 3)}
    t_in = max(0.0, words[0]["t0"] - cu["lead_pad_s"]) if cu["trim_in"] else 0.0
    t_out = min(duration, words[-1]["t1"] + cu["tail_pad_s"]) if cu["trim_out"] else duration
    cuts = []
    if cu["internal"]:
        for f in flaws:
            if f["kind"] not in cu["remove"]:
                continue
            i0, i1 = f["i0"], f["i1"]
            a = (words[i0 - 1]["t1"] + words[i0]["t0"]) / 2 if i0 > 0 else words[i0]["t0"]
            b = (words[i1 - 1]["t1"] + words[i1]["t0"]) / 2 if i1 < len(words) else words[-1]["t1"]
            cuts.append([round(a, 3), round(b, 3), f["kind"]])
    cuts.sort()
    merged = []
    for a, b, k in cuts:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][2] += "+" + k
        else:
            merged.append([a, b, k])
    segs, t = [], t_in
    for a, b, _ in merged:
        if a > t:
            segs.append([round(t, 3), round(min(a, t_out), 3)])
        t = max(t, b)
    if t < t_out:
        segs.append([round(t, 3), round(t_out, 3)])
    segs = [s for s in segs if s[1] - s[0] >= cu["min_segment_s"]]
    return {"segments": segs, "cuts": merged, "in": round(t_in, 3), "out": round(t_out, 3)}


def cuts_in_silence(words: list[dict], segments: list[list[float]], tol: float = 0.02) -> list[float]:
    """Boundaries that fall inside a word (empty list = guard passes)."""
    bad = []
    for a, b in segments:
        for t in (a, b):
            if any(w["t0"] + tol < t < w["t1"] - tol for w in words):
                bad.append(t)
    return bad


# ---------------------------------------------------------------- helpers
def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({r.returncode}): {' '.join(cmd[:6])} ... {r.stderr[-600:]}")
    return r


def measure_loudness(path: str) -> dict:
    """Integrated loudness, LRA and true peak of the file as it is (ffmpeg ebur128 with true-peak)."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    s = r.stderr[r.stderr.rfind("Summary:"):]
    get = lambda k: float(re.search(k + r":\s+(-?[\d.]+|-inf)", s).group(1).replace("-inf", "-99"))  # noqa: E731
    return {"I": get("I"), "LRA": get("LRA"), "TP": get("Peak")}


def _pcm(path: str, start: float, dur: float, sr: int = 8000):
    import numpy as np

    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.4f}", "-t", f"{dur:.4f}", "-i", path, "-map", "0:a:0",
                        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.float32)


def av_offset_ms(src: str, src_t: float, fin: str, fin_t: float, dur: float = 4.0, search_ms: float = 400, sr: int = 8000) -> float | None:
    """Lag (ms) of the final's audio against the source audio at the same picture position (positive = final audio late)."""
    import numpy as np

    m = search_ms / 1000
    b0 = max(0.0, fin_t - m)                 # the search window cannot start before the file
    a = _pcm(src, src_t, dur, sr)
    b = _pcm(fin, b0, dur + (fin_t - b0) + m, sr)
    if len(a) < sr // 4 or len(b) < len(a) or float(np.std(a)) < 1e-4:
        return None
    a = (a - a.mean()) / (a.std() + 1e-9)
    n = len(b) - len(a) + 1
    c = np.array([float(np.dot(a, b[k:k + len(a)])) for k in range(0, n)])
    return round((int(np.argmax(c)) / sr - (fin_t - b0)) * 1000, 1)


def av_check(qa: dict, final: str, fps: float) -> dict:
    """Per segment: cross-correlate the source audio with the final audio at the frame-exact picture position."""
    tol = qa.get("_sync_tol_ms", 10.0)
    rows, t = [], 0.0
    for c in qa["clips"]:
        for f0, f1 in c["segments_frames"]:
            d = (f1 - f0) / fps
            w = min(4.0, d - 0.1)                      # stay inside the segment (and past its 30 ms fades)
            off = max(0.05, (d - w) / 2)
            lag = av_offset_ms(c["video"], f0 / fps + off, final, t + off, dur=w) if w >= 0.4 else None
            rows.append({"clip": c["clip"], "t_final": round(t, 3), "offset_ms": lag})
            t += d
    p = C.probe(final)
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", final],
                             capture_output=True, text=True).stdout
    durs = {x["codec_type"]: float(x.get("duration", 0) or 0) for x in json.loads(streams).get("streams", [])}
    gap = abs(durs.get("video", 0) - durs.get("audio", 0))
    ok = all(r["offset_ms"] is None or abs(r["offset_ms"]) <= tol for r in rows) and gap <= 1.0 / fps + 0.03
    return {"segments": rows, "stream_gap_s": round(gap, 3), "duration": p["duration"], "pass": ok,
            "note": "lip sync inside a clip (model output) still needs SyncNet or a look; this checks the joins only"}


# ---------------------------------------------------------------- the edit
def edit(clips_dir: str | Path, plan: dict | None, prof: dict, out_dir: str | Path, dry_run: bool = False, require_wer: bool = True,
         title: str = "", subtitle: str = "", wer_dir: str | Path | None = None) -> dict:
    ed = prof["editor"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clips = find_clips(clips_dir, wer_dir)
    if not clips:
        raise FileNotFoundError(f"no clipN/video.mp4 or clipN.mp4 in {clips_dir}")
    plan = plan or {}
    ref_lines = {c["clip"]: " ".join(c["lines"]) for c in plan.get("clips", [])}
    punch = {a["clip"]: a.get("punch_in", False) for a in plan.get("angles", {}).get("clips", [])}
    qa: dict = {"profile": prof["name"], "clips": [], "_sync_tol_ms": ed["sync"]["max_offset_ms"]}

    # E1
    e1 = word_check(clips, prof, require=require_wer)
    qa["word_check"] = e1

    # E2
    prev_tokens: list[str] = []
    errors = list(e1["errors"])
    for c in clips:
        info = C.probe(c["video"]) if not dry_run or Path(c["video"]).exists() else {"duration": 0, "width": 0, "height": 0, "fps": 24, "has_audio": True}
        words = heard_words(json.loads(Path(c["wer"]).read_text())) if c["wer"] else []
        flaws = find_flaws(words, ref_lines.get(c["clip"], ""), prev_tokens, ngram=ed["cuts"]["repeat_ngram"]) if words else []
        cl = cut_list(words, info["duration"], flaws, prof)
        bad = cuts_in_silence(words, cl["segments"])
        if bad:
            errors.append({"where": f"clip {c['clip']}", "type": "cut_inside_speech", "severity": "fail", "t": bad})
        prev_tokens = [_norm(w["w"]) for w in words][-40:]
        qa["clips"].append({**c, **info, "flaws": [{"kind": f["kind"], "t0": words[f["i0"]]["t0"], "t1": words[f["i1"] - 1]["t1"]} for f in flaws],
                            **cl, "punch_in": punch.get(c["clip"], False)})

    W, H = qa["clips"][0]["width"] or 1280, qa["clips"][0]["height"] or 720
    fps = qa["clips"][0]["fps"] or 24
    W, H = (ed["encode"].get("width") or W), (ed["encode"].get("height") or H)

    # E3 colour stats (on kept frames)
    tmp = Path(tempfile.mkdtemp(prefix="olv_edit_"))
    col = ed["color"]
    target = (plan.get("color") or {}).get("target")
    stats = []
    if col["method"] != "none" and not dry_run:
        for c in qa["clips"]:
            a, b = c["segments"][0][0], c["segments"][-1][1]
            stats.append(C.lab_stats(list(C.frames_rgb(c["video"], step=col["sample_step"], start=a, end=b))))
        anchor = target or stats[col["anchor_clip"] - 1]
        for c, s in zip(qa["clips"], stats):
            c["colour_before"] = {"stats": s, "dE76_vs_anchor": C.de76(s, anchor)}
            if c["clip"] == col["anchor_clip"] and not target or C.de76(s, anchor) < col["skip_below_dE"]:
                c["cube"] = None
            else:
                c["cube"] = str(tmp / f"clip{c['clip']}.cube")
                C.write_cube(s, anchor, c["cube"], strength=col["strength"])
        qa["anchor"] = {"source": "plan target" if target else f"clip {col['anchor_clip']}", "stats": anchor}

    # E4 filter graph
    fade = ed["cuts"]["audio_fade_ms"] / 1000.0
    z = ed["pacing"]["punch_in_scale"]
    inputs, fc, labels, k = [], [], [], 0
    sr = 48000
    for ci, c in enumerate(qa["clips"]):
        inputs += ["-i", c["video"]]
        n_frames = int((c.get("duration") or 0) * (c.get("fps") or fps) + 1e-6) if c.get("duration") else None
        c["segments_frames"] = []
        for a, b in c["segments"]:
            # Sample-exact join: snap both ends to the frame grid, trim video by frame and audio by sample, so every segment's
            # audio lasts exactly frames/fps. Unequal lengths add an audio lead at every join (seen: +48 ms per join).
            f0, f1 = round(a * fps), round(b * fps)
            if n_frames:
                f1 = min(f1, n_frames)
            if f1 <= f0:
                continue
            c["segments_frames"].append([f0, f1])
            s0, s1 = round(f0 * sr / fps), round(f1 * sr / fps)
            vf = f"[{ci}:v]trim=start_frame={f0}:end_frame={f1},setpts=PTS-STARTPTS"
            if c.get("punch_in"):
                vf += f",crop=iw/{z}:ih/{z}"
            # Untagged sources are read as BT.709 TV range (same as the colour stats). Without explicit matrices, concat lets
            # swscale guess per segment, and an untagged clip next to a tagged one shifted by dE76 1.7 (measured).
            vf += f",scale={W}:{H}:flags=lanczos:in_color_matrix=bt709:out_color_matrix=bt709:in_range=tv:out_range=tv,setsar=1,fps={fps}"
            if c.get("cube"):
                vf += "," + C.LUT_CHAIN.format(cube=c["cube"])
            vf += f",format=yuv420p,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv[v{k}]"
            d = (f1 - f0) / fps
            af = (f"[{ci}:a]aresample={sr},aformat=sample_rates={sr}:channel_layouts=stereo,apad,"
                  f"atrim=start_sample={s0}:end_sample={s1},asetpts=PTS-STARTPTS")
            if fade > 0 and d > 4 * fade:
                af += f",afade=t=in:st=0:d={fade},afade=t=out:st={d - fade:.4f}:d={fade}"
            af += f"[a{k}]"
            fc += [vf, af]
            labels.append(f"[v{k}][a{k}]")
            k += 1
    fc.append("".join(labels) + f"concat=n={k}:v=1:a=1[vc][ac]")
    vlast = "[vc]"
    lt = ed["titles"]
    if title and lt.get("lower_third"):
        (tmp / "title.txt").write_text(title)
        (tmp / "sub.txt").write_text(subtitle or " ")
        en = f"enable='between(t,{lt['start_s']},{lt['start_s'] + lt['hold_s']})'"
        y = int(H * 0.78)
        fc.append(f"[vc]drawtext=textfile='{tmp / 'title.txt'}':fontcolor=white:fontsize={int(H * 0.045)}:x={int(W * 0.06)}:y={y}:"
                  f"box=1:boxcolor=black@0.55:boxborderw={int(H * 0.012)}:{en},"
                  f"drawtext=textfile='{tmp / 'sub.txt'}':fontcolor=white@0.85:fontsize={int(H * 0.03)}:x={int(W * 0.06)}:y={y + int(H * 0.07)}:"
                  f"box=1:boxcolor=black@0.55:boxborderw={int(H * 0.01)}:{en}[vt]")
        vlast = "[vt]"
    enc = ed["encode"]
    assembled = out / "assembled.mkv"
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fc), "-map", vlast, "-map", "[ac]",
           "-c:v", "libx264", "-preset", enc["preset"], "-crf", str(enc["crf"]), "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", "-c:a", "pcm_s16le", str(assembled)]
    qa["ffmpeg_assemble"] = cmd
    edl = {"segments": [{"clip": c["clip"], "segments": c["segments"], "cuts": c["cuts"], "punch_in": c.get("punch_in"), "lut": bool(c.get("cube"))}
                        for c in qa["clips"]]}  # frame-exact ranges are added to edit_qa.json (segments_frames)
    (out / "edl.json").write_text(json.dumps(edl, indent=1))
    if dry_run:
        qa["dry_run"] = True
        (out / "edit_qa.json").write_text(json.dumps(qa, indent=1, default=str))
        return qa
    try:
        _run(cmd)
    except RuntimeError as e:
        if vlast == "[vt]":   # drawtext missing or no font: finish without the lower third, report it
            qa.setdefault("warnings", []).append(f"lower third skipped: {str(e)[-200:]}")
            cmd[cmd.index("-filter_complex") + 1] = ";".join(fc[:-1])
            cmd[cmd.index("[vt]")] = "[vc]"
            _run(cmd)
        else:
            raise

    # E5 loudness: measure, apply gain to the target plus a peak limiter, measure the AAC file, correct gain and limit, retry.
    # loudnorm in linear mode cannot raise peaky dialogue to -14 LUFS under a -2 dBTP ceiling (it fell back and gave -16.9 LUFS),
    # so the limiter takes the peaks and the loop checks the true peak after the AAC encode.
    target_i, target_tp, tol = ed["audio"]["lufs"], ed["audio"]["true_peak"], ed["audio"]["lufs_tol"]
    final = out / "final.mp4"
    m0 = measure_loudness(str(assembled))
    gain, limit_db = target_i - m0["I"], target_tp - 1.0
    tries = []
    au = ed["audio"]
    for _ in range(6):
        # rumble cut, gain, delay-compensated peak limiter (alimiter without latency=1 shifted the audio 5 ms; caught by G13).
        # A fixed-threshold compressor before the limiter ate the gain (measured: -15.9 LUFS after 6 tries); not used.
        af = (f"highpass=f={au['highpass_hz']},volume={gain:.2f}dB,"
              f"alimiter=limit={10 ** (limit_db / 20):.4f}:attack=5:release=50:level=false:latency=1,aresample=48000")
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(assembled), "-map", "0:v", "-map", "0:a", "-c:v", "copy", "-af", af,
              "-c:a", "aac", "-b:a", f"{enc['audio_kbps']}k", "-movflags", "+faststart", str(final)])
        lm = measure_loudness(str(final))
        tries.append({"gain_db": round(gain, 2), "limit_db": round(limit_db, 2), **lm})
        ok_i, ok_tp = abs(lm["I"] - target_i) <= min(tol, 0.5), lm["TP"] <= target_tp + 0.05
        if ok_i and ok_tp:
            break
        if not ok_tp:
            limit_db -= max(0.3, lm["TP"] - target_tp + 0.1)
        # secant step: under limiting, 1 dB of gain buys less than 1 LU, so use the slope of the last two tries
        slope = 1.0
        if len(tries) >= 2 and tries[-1]["gain_db"] != tries[-2]["gain_db"]:
            slope = min(1.0, max(0.2, (tries[-1]["I"] - tries[-2]["I"]) / (tries[-1]["gain_db"] - tries[-2]["gain_db"])))
        gain += (target_i - lm["I"]) / slope
    reduction = round(tries[-1]["gain_db"] - (tries[-1]["I"] - m0["I"]), 1)   # loudness lost to compression and limiting
    if reduction > au["max_reduction_db"]:
        qa.setdefault("warnings", []).append(f"dynamics took {reduction} dB of loudness (> {au['max_reduction_db']}): listen for pumping")
    qa["loudness"] = {"target": ed["audio"], "input": m0, "tries": tries, "final": tries[-1], "dynamics_reduction_db": reduction}

    # E3 measured after: per clip in the final timeline + joins
    fin = C.probe(str(final))
    qa["final"] = {"path": str(final), **fin}
    t, pos = 0.0, []
    for c in qa["clips"]:
        d = sum(f1 - f0 for f0, f1 in c["segments_frames"]) / fps
        pos.append((t, t + d))
        t += d
    qa["av"] = av_check(qa, str(final), fps)
    if stats:
        after = [C.lab_stats(list(C.frames_rgb(str(final), step=col["sample_step"], start=a, end=b))) for a, b in pos]
        anchor = qa["anchor"]["stats"]
        for c, s in zip(qa["clips"], after):
            c["colour_after"] = {"stats": s, "dE76_vs_anchor": C.de76(s, anchor)}
        joins = []
        for (a0, b0), (a1, b1), c0, c1 in zip(pos, pos[1:], qa["clips"], qa["clips"][1:]):
            x = C.lab_stats(list(C.frames_rgb(str(final), step=1, start=max(a0, b0 - 0.5), end=b0)))
            y = C.lab_stats(list(C.frames_rgb(str(final), step=1, start=a1, end=min(b1, a1 + 0.5))))
            joins.append({"join": f"{c0['clip']}->{c1['clip']}", "t": round(a1, 2), "dE76": C.de76(x, y)})
        qa["colour_joins"] = joins

    # E6 contact sheet: cols x rows frames evenly spaced over the whole film
    cols, rows = ed["review"]["contact_sheet"]
    n = cols * rows
    sheet = out / "contact_sheet.jpg"
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(final), "-vf",
          f"fps={n}/{fin['duration']:.4f},scale={ed['review']['thumb_width']}:-2,tile={cols}x{rows}:padding=4:margin=4:color=white",
          "-frames:v", "1", "-q:v", "3", str(sheet)])
    qa["contact_sheet"] = {"path": str(sheet), "grid": [cols, rows], "frames": n, "t": [round(i * fin["duration"] / n, 2) for i in range(n)]}

    # gates
    cg = (plan.get("color") or {}).get("gates") or prof["planner"]["color"]["gates"]
    col_ok = all(c.get("colour_after", {}).get("dE76_vs_anchor", 0) <= cg["clip_dE76_max"] for c in qa["clips"])
    join_ok = all(j["dE76"] <= cg["join_dE76_max"] for j in qa.get("colour_joins", []))
    L = qa["loudness"]["final"]
    gates = [
        {"id": "G8", "gate": "every 5 s window WER inside the limit (no gibberish)", "pass": not e1["errors"]},
        {"id": "G9", "gate": "no cut inside speech", "pass": not [e for e in errors if e["type"] == "cut_inside_speech"]},
        {"id": "G10", "gate": f"clip dE76 <= {cg['clip_dE76_max']} vs anchor, joins <= {cg['join_dE76_max']}",
         "pass": (col_ok and join_ok) or col["method"] == "none"},
        {"id": "G11", "gate": f"loudness {ed['audio']['lufs']} +/- {ed['audio']['lufs_tol']} LU, true peak <= {ed['audio']['true_peak']} dBTP (AAC file)",
         "pass": abs(L["I"] - ed["audio"]["lufs"]) <= ed["audio"]["lufs_tol"] and L["TP"] <= ed["audio"]["true_peak"] + 0.05},
        {"id": "G12", "gate": f"contact sheet {cols}x{rows} written", "pass": sheet.exists() and sheet.stat().st_size > 0},
        {"id": "G13", "gate": f"sample-exact joins: every segment's audio within {ed['sync']['max_offset_ms']} ms of its source, "
                              f"streams within one frame", "pass": qa["av"]["pass"]},
    ]
    qa["errors"] = errors
    qa["gates"] = gates
    qa["gates_pass"] = all(g["pass"] for g in gates)
    qa["human"] = "open contact_sheet.jpg and watch final.mp4; record qa.contact_sheet_reviewed = true in flow_run.json"
    if plan:
        qa["plan"] = {"angle_changes": plan.get("film", {}).get("angle_changes"), "gestures": plan.get("film", {}).get("gestures")}
    assembled.unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    (out / "edit_qa.json").write_text(json.dumps(qa, indent=1, default=str))
    return qa


def summary(qa: dict) -> dict:
    if qa.get("dry_run"):
        return {"dry_run": True, "segments": sum(len(c["segments"]) for c in qa["clips"]), "cuts": sum(len(c["cuts"]) for c in qa["clips"])}
    return {"edit_gates_pass": qa["gates_pass"], "failed": [g["id"] + " " + g["gate"] for g in qa["gates"] if not g["pass"]],
            "duration_s": round(qa["final"]["duration"], 2), "loudness": qa["loudness"]["final"],
            "clip_dE76_after": [c.get("colour_after", {}).get("dE76_vs_anchor") for c in qa["clips"]],
            "join_dE76": [j["dE76"] for j in qa.get("colour_joins", [])], "cuts": sum(len(c["cuts"]) for c in qa["clips"]),
            "av_offsets_ms": [r["offset_ms"] for r in qa["av"]["segments"]], "stream_gap_s": qa["av"]["stream_gap_s"],
            "contact_sheet": qa["contact_sheet"]["path"], "final": qa["final"]["path"], "warnings": qa.get("warnings", [])}
