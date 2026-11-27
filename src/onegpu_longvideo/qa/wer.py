"""Per-window word error rate against the intended lines.

Speech faults in long one-shot clips are local (invented words in one 5 s window), so a clip-level WER hides them.
We align the reference words to the heard words (Levenshtein), time each error by the heard word, and count per window.
ASR: any engine that returns word timestamps; `transcribe()` uses faster-whisper if installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


def normalize(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("-", " ")).split()


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein alignment. Ops: M match, S substitution, D deletion (missing), I insertion (invented)."""
    n, k = len(ref), len(hyp)
    D = [[0] * (k + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(k + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops = []
    i, j = n, k
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("M" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            ops.append(("D", i - 1, None))
            i -= 1
        else:
            ops.append(("I", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def score(reference_text: str, heard: list[Word], duration_s: float, window_s: float = 5.0) -> dict:
    ref = normalize(reference_text)
    hw = [Word(t, w.start, w.end, w.prob) for w in heard for t in normalize(w.text)]
    hyp = [w.text for w in hw]
    ops = align(ref, hyp)
    errs, ref_t = [], []
    last_t = 0.0
    names = {"S": "wrong", "I": "invented", "D": "missing"}
    for op, ri, hj in ops:
        t = hw[hj].start if hj is not None else last_t
        last_t = t
        if op != "I":
            ref_t.append(t)
        if op != "M":
            errs.append({"t": round(t, 2), "op": names[op], "ref": ref[ri] if ri is not None else "", "heard": hyp[hj] if hj is not None else ""})
    wins = []
    edges = [(w0, w0 + window_s) for w0 in [i * window_s for i in range(max(1, int(round(duration_s / window_s))))]]
    for idx, (a, b) in enumerate(edges):
        last = idx == len(edges) - 1

        def inw(t, a=a, b=b, last=last):
            return a <= t < b or (last and t >= b)

        sel = [e for e in errs if inw(e["t"])]
        nref = sum(1 for t in ref_t if inw(t))
        wins.append({"window": [a, b], "ref_words": nref, "wrong": sum(e["op"] == "wrong" for e in sel),
                     "invented": sum(e["op"] == "invented" for e in sel), "missing": sum(e["op"] == "missing" for e in sel),
                     "wer": round(len(sel) / max(1, nref), 3)})
    return {"wer": round(len(errs) / max(1, len(ref)), 3), "n_ref": len(ref), "n_hyp": len(hyp),
            "windows": wins, "win_wer": [w["wer"] for w in wins], "errors": errs,
            "first_word_s": round(hw[0].start, 2) if hw else None, "last_word_s": round(hw[-1].end, 2) if hw else None}


def passes(result: dict, max_window_wer: float = 0.03) -> bool:
    """Our pass rule: WER <= 3 % in EVERY window (about zero errors per window at ~14 words per 5 s)."""
    return all(w <= max_window_wer for w in result["win_wer"])


def transcribe(audio_path: str, model: str = "large-v3-turbo", device: str = "cpu", compute_type: str = "int8") -> list[Word]:
    """faster-whisper with word timestamps. condition_on_previous_text=False avoids repeat hallucinations at the end."""
    from faster_whisper import WhisperModel

    m = WhisperModel(model, device=device, compute_type=compute_type)
    segs, _ = m.transcribe(audio_path, language="en", word_timestamps=True, vad_filter=False, beam_size=5,
                           condition_on_previous_text=False)
    return [Word(w.word, w.start, w.end, w.probability) for s in segs for w in (s.words or [])]
