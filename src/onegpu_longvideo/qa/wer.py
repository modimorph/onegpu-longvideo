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


_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                                     "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


def _numbers(tokens: list[str]) -> list[str]:
    """Spoken numbers -> digits, so a script that spells numbers matches an ASR that writes digits.
    "fifteen" -> 15, "eighty seven" -> 87, "two hundred and five" -> 205, "ten eighty" -> 1080 (two 10-99 groups in a row)."""
    out, i = [], 0

    def group(j):            # one number below 1000 starting at j -> (value, next index) or (None, j)
        val, k, seen = 0, j, False
        if k < len(tokens) and tokens[k] in _UNITS and k + 1 < len(tokens) and tokens[k + 1] == "hundred":
            val, k, seen = _UNITS[tokens[k]] * 100, k + 2, True
            if k < len(tokens) and tokens[k] == "and" and k + 1 < len(tokens) and (tokens[k + 1] in _UNITS or tokens[k + 1] in _TENS):
                k += 1
        if k < len(tokens) and tokens[k] in _TENS:
            val, k, seen = val + _TENS[tokens[k]], k + 1, True
            if k < len(tokens) and tokens[k] in _UNITS and _UNITS[tokens[k]] < 10:
                val, k = val + _UNITS[tokens[k]], k + 1
        elif k < len(tokens) and tokens[k] in _UNITS:
            val, k, seen = val + _UNITS[tokens[k]], k + 1, True
        return (val, k) if seen else (None, j)

    while i < len(tokens):
        v, j = group(i)
        if v is None:
            out.append(tokens[i])
            i += 1
            continue
        v2, j2 = group(j)
        if v2 is not None and 10 <= v <= 99 and 10 <= v2 <= 99:      # "ten eighty", "nineteen ninety"
            out.append(f"{v}{v2:02d}")
            i = j2
        else:
            out.append(str(v))
            i = j
    return out


# British -> American spellings (Whisper writes American; a British script made "colour" vs "color" a word error).
_OUR = "colour favour behaviour honour labour neighbour flavour humour harbour rumour vapour savour armour endeavour".split()
_SPELL = {**{w: w.replace("our", "or") for w in _OUR}, "grey": "gray", "centre": "center", "metre": "meter", "litre": "liter",
          "theatre": "theater", "fibre": "fiber", "programme": "program", "catalogue": "catalog", "dialogue": "dialog",
          "analyse": "analyze", "organise": "organize", "recognise": "recognize", "realise": "realize", "optimise": "optimize",
          "apologise": "apologize", "customise": "customize", "prioritise": "prioritize", "summarise": "summarize",
          "travelled": "traveled", "travelling": "traveling", "modelling": "modeling", "cancelled": "canceled", "labelled": "labeled"}


def _spell(w: str) -> str:
    if w in _SPELL:
        return _SPELL[w]
    for base, us in _SPELL.items():          # colours, coloured, organised, organising, analysed ...
        if base.endswith("e") and w.startswith(base[:-1]) and w[len(base) - 1:] in ("es", "ed", "ing", "er", "ers"):
            return us[:-1] + w[len(base) - 1:]
        if not base.endswith("e") and w.startswith(base) and w[len(base):] in ("s", "ed", "ing", "ful", "ite", "ites", "less"):
            return us + w[len(base):]
    return w


def normalize(s: str) -> list[str]:
    """Lower-case words; Unicode letters stay inside one word (an ASCII-only class split "größe" into two tokens);
    numbers in one form (digits): "fifteen" and "15" match, "10%" and "ten percent" match, "1,080" = "1080"."""
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s.lower()).replace("%", " percent ")
    s = re.sub(r"\b(\d+)p\b", r"\1", s)                         # "1080p" -> "1080"
    return [_spell(w) for w in _numbers(re.sub(r"[^\w' ]+|_", " ", s.replace("-", " ")).split())]


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
