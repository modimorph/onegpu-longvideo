"""Whole-film planner (flow steps 1-2): read the whole script once, write one film plan, then build every clip prompt from it.

"Decode the whole, then encode the parts." Stages (docs/PLANNER.md):
  P1 text pass      fake actions auto-fixed (stutters, coughs, fillers, self-corrections, fake stops), repeated sentences
                    and repeated ideas flagged, weak joins between clips flagged; optional LLM review request.
  P2 speech density words per clip inside the profile band, last word by S - margin, no word after the training-window edge.
  P3 angle plan     camera set-ups changed only at clip joins, at the biggest topic shifts, inside the profile band, bookend.
  P4 gesture budget each gesture type at most its film budget, matched to the meaning of the line, spread out; rest by default.
  P5 colour anchor  the target grade (a still or clip 1) and the colour gates the editor enforces.
  P6 prompts        one prompt per clip built with `prompt.build` from the plan, then `prompt.lint`.
No model call and no GPU unless the optional LLM review is asked for.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import prompt as P

STOP = set("""a an the and or but so to of in on at for with from by is are was were be been it its it's this that these those we our us you your i i'm
my me he his she her they them their there here what which who how when then than just only also too very really about into over out up down not no
one two three four five six seven eight nine ten got get make makes made like kind thing first next now back way well per each every""".split())


@dataclass
class Line:
    text: str
    raw: str = ""
    cue: str = ""          # optional hint from the writer (old gesture column); never copied into the prompt


@dataclass
class Clip:
    idx: int
    title: str
    seconds: float
    lines: list[Line] = field(default_factory=list)
    side: str = ""         # optional: which side of frame the second camera sits for this clip


@dataclass
class Script:
    title: str
    clips: list[Clip]
    speaker: dict = field(default_factory=dict)   # tag, role, description, language, voice_ref
    scene: dict = field(default_factory=dict)     # refs, description, sound
    facts: list = field(default_factory=list)     # fact sheet: [{id, keywords, allowed, forbid, note}]


# ---------------------------------------------------------------- input
def _segments_md(text: str, clip_seconds: float | None) -> list[Clip]:
    """Markdown script: '## Segment N (t0-t1 s): title' + table rows '| n | a-b | line | gesture |'."""
    clips = []
    for m in re.finditer(r"^## Segment (\d+) \(([\d.]+)-([\d.]+) s\):\s*(.*?)\n(.*?)(?=^## Segment|\Z)", text, re.S | re.M):
        rows = []
        for row in m.group(5).splitlines():   # one table row per line; the gesture column is optional
            cells = [x.strip() for x in row.strip().strip("|").split("|")]
            if len(cells) >= 3 and re.fullmatch(r"\d+", cells[0]) and re.fullmatch(r"[\d.]+-[\d.]+", cells[1]) and cells[2]:
                rows.append((cells[2], cells[3] if len(cells) > 3 else ""))
        secs = clip_seconds or (float(m.group(3)) - float(m.group(2)))
        clips.append(Clip(int(m.group(1)), m.group(4).strip(), secs, [Line(r[0], r[0], r[1]) for r in rows]))
    return clips


def load_facts(path: str | Path | None) -> list:
    """Fact sheet YAML: a list of {id, keywords: [all must appear], allowed: [one must appear], forbid: [regex], note}."""
    import yaml

    if not path:
        return []
    d = yaml.safe_load(Path(path).read_text()) or []
    return d.get("facts", []) if isinstance(d, dict) else d


def fact_check(script: "Script") -> list[dict]:
    """Lines whose claim contradicts the fact sheet (production data, article, logs). Each hit is a fail: rewrite before rendering."""
    out = []
    for c in script.clips:
        for n, ln in enumerate(c.lines, 1):
            t = ln.text.lower()
            for f in script.facts:
                kws = [k.lower() for k in f.get("keywords", [])]
                if kws and not all(k in t for k in kws):
                    continue
                bad = [rx for rx in f.get("forbid", []) if re.search(rx, t, re.I)]
                allowed = [a.lower() for a in f.get("allowed", [])]
                miss = bool(allowed) and kws and not any(a in t for a in allowed)
                if bad or miss:
                    out.append({"where": f"line {n} of clip {c.idx}", "type": "claim_vs_facts", "severity": "fail", "fact": f.get("id", ""),
                                "note": f.get("note", "")})
    return out


def _line(x, clip_no: int) -> Line:
    """A script line from YAML. YAML turns "Step one: open it" into a mapping and "yes" into True; both become text again."""
    if isinstance(x, dict) and "text" in x:
        return Line(str(x["text"]), str(x["text"]), str(x.get("cue", "")))
    if isinstance(x, dict) and len(x) == 1:
        k, v = next(iter(x.items()))
        t = f"{k}: {v}"
        return Line(t, t)
    if isinstance(x, (str, int, float, bool)):
        t = {True: "yes", False: "no"}.get(x, str(x)) if isinstance(x, bool) else str(x)
        return Line(t, t)
    raise ValueError(f"clip {clip_no}: cannot read line {x!r}; quote it as a string")


def load_script(path: str | Path, clip_seconds: float | None = None, scene_path: str | Path | None = None) -> Script:
    """Load a YAML script (see configs/scripts/example_talking_head.yaml) or a Markdown segment-table script."""
    import yaml

    p = Path(path)
    txt = p.read_text()
    extra = yaml.safe_load(Path(scene_path).read_text()) if scene_path else {}
    if p.suffix in (".yaml", ".yml", ".json"):
        d = yaml.safe_load(txt)
        d = {**d, **{k: v for k, v in (extra or {}).items() if k in ("speaker", "scene")}}
        default_s = float(d.get("clip_seconds", 29.96))
        clips = []
        for i, c in enumerate(d["clips"], 1):
            lines = [_line(x, i) for x in c["lines"]]
            secs = float(clip_seconds) if clip_seconds else float(c.get("seconds", default_s))   # --clip-seconds overrides all
            clips.append(Clip(i, str(c.get("title", f"clip {i}")), secs, lines, c.get("side", "")))
        return Script(d.get("title", p.stem), clips, d.get("speaker", {}), d.get("scene", {}), d.get("facts", []))
    clips = _segments_md(txt, clip_seconds)
    if not clips:
        raise ValueError(f"no clips found in {p}: use the YAML format or '## Segment N (t0-t1 s): title' tables")
    title = (re.search(r"^# (.*)$", txt, re.M) or [None, p.stem])[1]
    return Script(title, clips, (extra or {}).get("speaker", {}), (extra or {}).get("scene", {}))


# ---------------------------------------------------------------- P1 text pass
# Words a speaker may legitimately say twice in a row ("I know that that is true", "had had", "very very").
LEGIT_DOUBLES = {"that", "had", "is", "do", "very", "bye", "no", "so", "really", "many", "far", "long", "now", "again", "more", "round", "over"}

FAKE_ACTIONS = [  # (kind, pattern, replacement); applied in this order. Each pattern is anchored so normal text does not match.
    # a bracket holding only a stage direction: [cough], (laughs), *sighs*; not "(breathing room matters)" or "(laughter track)"
    ("cough", r"\s*[\[(*]\s*(?:a |one )?(?:soft |small |little |short )?(?:cough|coughs|coughing|laugh|laughs|laughing|sigh|sighs|"
              r"clears throat|clears his throat|clears her throat|pause|pauses|breath|breathes|inhales|exhales)\s*[\])*]\s*", " "),
    # "an hour... well, nearly an hour": an ellipsis, a hedge word, a correction word, then the restated phrase
    ("self_correction", r"\s*\.\.\.\s*(?:well|no|sorry|i mean)\b,?\s*(?:nearly|actually|almost|rather|i mean)\b[^.?!]*", ""),
    # mid-sentence "..., no, actually, ...": needs a comma before it, so a sentence that starts "No, actually" is kept
    ("self_correction", r",\s*(?:no|sorry|i mean),\s*actually,?\s*", ", "),
    ("filler", r"(?<![\w'])(?:{fillers})(?![\w'])[,.]?\s*", ""),
    # "you'd, you'd" (comma) is always a stutter; "the the" is one unless the word is in LEGIT_DOUBLES (handled in code)
    ("stutter", r"\b(\w+(?:'\w+)?), \1\b", r"\1"),
    ("stutter", r"\b(\w+(?:'\w+)?) \1\b", "LEGIT"),
    ("fake_stop", r"\s*(?:\.\.\.|…)\s*", ". \x01"),        # \x01 marks a new sentence for capitalisation
]
# Fillers by language; "er" is English only (German "Er sagte" is a pronoun).
FILLERS = {"english": "um+|uh+|erm|er|hmm+", "german": "äh+m?|öhm|hmm+", "french": "euh|hmm+", "_other": "um+|uh+|hmm+"}



def _cap(s: str) -> str:
    """Tidy spaces and punctuation; capitalise the first letter and letters after a sentence break WE inserted (\x01).
    Text the writer wrote ("e.g. a still") keeps its case."""
    s = re.sub(r"\s*(?:—|\s--\s)\s*", ", ", s)          # a dash pause is read as a comma, not a fake stop
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.?!])", r"\1", s)
    s = re.sub(r"([.?!])\s*\.", r"\1", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\x01\s*(\w)", lambda m: m.group(1).upper(), s).replace("\x01", "")
    return s[:1].upper() + s[1:]


def _patterns(language: str = "English"):
    fill = FILLERS.get(language.lower(), FILLERS["_other"])
    return [(k, p.replace("{fillers}", fill), r) for k, p, r in FAKE_ACTIONS]


def fix_fake_actions(text: str, banned: list[str], language: str = "English") -> tuple[str, list[str]]:
    """Remove banned fake actions that have a mechanical fix. Returns (clean text, kinds fixed)."""
    out, kinds = text, []
    for kind, pat, rep in _patterns(language):
        if kind not in banned:
            continue
        if rep == "LEGIT":
            new = re.sub(pat, lambda m: m.group(0) if m.group(1).lower() in LEGIT_DOUBLES else m.group(1), out, flags=re.I)
        else:
            new = re.sub(pat, rep, out, flags=re.I)
        if new != out:
            kinds.append(kind)
            out = new
    return _cap(out), kinds


def norm_words(s: str) -> list[str]:
    """Lower-case words, Unicode letters kept ("größe" stays one word; an ASCII-only class split it in two)."""
    return re.sub(r"[^\w' ]+|_", " ", s.lower().replace("-", " ")).split()


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[^\W\d_][\w'-]+", s.lower()) if w not in STOP and len(w) > 3}


def repeated_runs(lines: list[tuple[str, str]], n: int) -> list[dict]:
    """Runs of n+ words said twice anywhere in the film. lines = [(where, text)]."""
    seen: dict[tuple, tuple[str, int]] = {}
    hits = []
    for where, text in lines:
        w = norm_words(text)
        for i in range(len(w) - n + 1):
            g = tuple(w[i:i + n])
            if all(x in STOP for x in g):
                continue
            # a second copy counts even inside the same line, but not an overlapping one ("very very very ...")
            if g in seen and (seen[g][0] != where or i - seen[g][1] >= n):
                hits.append({"where": where, "type": "repeated_sentence", "first": seen[g][0], "words": " ".join(g)})
            seen.setdefault(g, (where, i))
    # one hit per (where, first) pair is enough
    out, keys = [], set()
    for h in hits:
        k = (h["where"], h["first"])
        if k not in keys:
            keys.add(k)
            out.append(h)
    return out


def text_pass(script: Script, prof: dict) -> dict:
    tp = prof["planner"]["text"]
    banned = prof["planner"]["banned_actions"]
    fixes, flags = [], []
    lang = (script.speaker or {}).get("language", "English")
    for c in script.clips:
        for n, ln in enumerate(c.lines, 1):
            ln.text, kinds = fix_fake_actions(ln.raw or ln.text, banned, lang)
            if kinds:
                fixes.append({"where": f"line {n} of clip {c.idx}", "fixed": kinds})
            if not re.search(r"\w", ln.text):
                fixes.append({"where": f"line {n} of clip {c.idx}", "fixed": ["empty_line_dropped"]})
            if re.search(r"\d", ln.text):
                flags.append({"where": f"line {n} of clip {c.idx}", "type": "digits", "severity": "warn", "note": "write numbers as words"})
            if len(ln.text.split()) > tp["max_line_words"]:
                flags.append({"where": f"line {n} of clip {c.idx}", "type": "long_line", "severity": "warn", "note": f"> {tp['max_line_words']} words"})
            if ln.text and ln.text[-1] not in ".?!":
                ln.text += "."
    for c in script.clips:              # a line that was only a stage direction ("[laughs]") is gone, not an empty beat
        c.lines = [ln for ln in c.lines if re.search(r"\w", ln.text)]
    where_text = [(f"line {n} of clip {c.idx}", ln.text) for c in script.clips for n, ln in enumerate(c.lines, 1)]
    errors = []
    # banned actions that survive the mechanical fix (e.g. a stutter across punctuation) are errors
    for where, t in where_text:
        for kind, pat, rep in _patterns(lang):
            if kind not in banned:
                continue
            hits = [m for m in re.finditer(pat, t, re.I) if rep != "LEGIT" or m.group(1).lower() not in LEGIT_DOUBLES]
            if hits:
                errors.append({"where": where, "type": f"banned_{kind}", "severity": "fail"})
    if "repeated_sentence" in banned:
        errors += [{**h, "severity": "fail"} for h in repeated_runs(where_text, tp["repeat_ngram"])]
    errors += fact_check(script)
    # repeated ideas across clips (content-word overlap): a warning for the writer, not a fail
    L = [(c.idx, n, content_words(ln.text)) for c in script.clips for n, ln in enumerate(c.lines, 1)]
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            if L[i][0] == L[j][0] or not L[i][2] or not L[j][2]:
                continue
            a, b = L[i][2], L[j][2]
            jac = len(a & b) / len(a | b)
            if jac >= tp["idea_overlap_jaccard"] and len(a & b) >= 3:
                flags.append({"where": f"line {L[j][1]} of clip {L[j][0]}", "type": "repeated_idea", "severity": "warn",
                              "first": f"line {L[i][1]} of clip {L[i][0]}", "overlap": round(jac, 2)})
    joins = []
    hook = re.compile(tp["connector_regex"], re.I)
    for a, b in zip(script.clips, script.clips[1:]):
        if not a.lines or not b.lines:
            continue
        first = b.lines[0].text
        shared = content_words(a.lines[-1].text) & content_words(first)
        ok = bool(hook.match(first)) or bool(shared)
        joins.append({"join": f"{a.idx}->{b.idx}", "linked": ok})
        if not ok:
            flags.append({"where": f"join {a.idx}->{b.idx}", "type": "weak_join", "severity": "warn", "note": "add a bridge clause"})
    return {"fixes": fixes, "errors": errors, "warnings": flags, "joins": joins}


def llm_review_request(script: Script, tp_result: dict) -> str:
    """The whole-script review request for a writer model (logic, grammar, flow). Its answer is advice; it is never auto-applied."""
    body = "\n".join(f"clip {c.idx} ({c.title}), line {n}: {ln.text}" for c in script.clips for n, ln in enumerate(c.lines, 1))
    flags = json.dumps(tp_result["errors"] + tp_result["warnings"], indent=0)
    facts = "\n".join(f"- {f.get('id', '')}: {f.get('note', '')}" for f in script.facts) or "- none given"
    return ("You are a script editor for a spoken video. Read the WHOLE script below as one text.\n"
            "Check: logic from start to end, grammar, flow between clips, repeated ideas or sentences, claims that contradict each other, "
            "and any fake spoken action (stutter, cough, filler, self-correction, fake stop). Keep every line's word count within 2 words.\n"
            "Answer only as lines: 'clip M line N: <problem type> -> <rewrite>'. Write 'OK' if nothing needs a change.\n\n"
            "Every claim must agree with the fact sheet; a claim the facts contradict is an error, not style.\n\n"
            f"Fact sheet:\n{facts}\n\nAutomatic flags:\n{flags}\n\nScript:\n{body}\n")


def llm_review(request: str, url: str, model: str, key: str = "", timeout: float = 120) -> str:
    """POST the request to an OpenAI-compatible /chat/completions endpoint."""
    import urllib.request

    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", method="POST",
                                 data=json.dumps({"model": model, "messages": [{"role": "user", "content": request}], "temperature": 0}).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- P2 speech density
def speech_window(seconds: float, sp: dict) -> tuple[float, float]:
    """(first word s, last word deadline s) for a clip of this length."""
    end = seconds - sp["end_margin_s"]
    if seconds <= sp["train_window_s"] + 1e-6:
        end = min(end, sp["window_speech_end_s"])
    return sp["start_s"], end


def density(script: Script, prof: dict) -> dict:
    sp = prof["planner"]["speech"]
    lo, hi = sp["wps"]
    rows, errors = [], []
    for c in script.clips:
        t0, t1 = speech_window(c.seconds, sp)
        avail = max(0.1, t1 - t0)
        words = sum(len(ln.text.split()) for ln in c.lines)
        band = [math.ceil(lo * avail), math.floor(hi * avail)]
        # The clip is spoken continuously, so estimate at the clip's own rate clamped into the band (a fixed nominal rate made
        # in-band scripts fail the deadline: 36 words in a 33-37 band ran to 13.6 s against 13.3 s).
        rate = min(hi, max(lo, words / avail))
        est_last = t0 + words / rate
        t, times = t0, []
        for ln in c.lines:
            d = len(ln.text.split()) / rate
            times.append([round(t, 2), round(t + d, 2)])
            t += d
        row = {"clip": c.idx, "seconds": c.seconds, "words": words, "band": band, "wps": round(words / avail, 2), "rate_used": round(rate, 2),
               "first_word_s": t0, "deadline_s": round(t1, 2), "est_last_word_s": round(est_last, 2), "line_times": times}
        rows.append(row)
        if not band[0] <= words <= band[1]:
            kind = "too_few_words (leftover time invites invented words)" if words < band[0] else "too_many_words (rushed, clipped endings)"
            errors.append({"where": f"clip {c.idx}", "type": kind, "severity": "fail", "words": words, "band": band})
        if est_last > t1 + sp["deadline_tolerance_s"]:
            errors.append({"where": f"clip {c.idx}", "type": "last_word_late", "severity": "fail",
                           "est_last_word_s": round(est_last, 2), "deadline_s": round(t1, 2)})
    warnings = []
    lang = (script.speaker or {}).get("language", "English")
    if lang.lower() != "english":
        warnings.append({"where": "film", "type": "speech_band_untested_language", "severity": "warn",
                         "note": f"the words-per-second band was measured on English; check {lang} with per-window WER on a first clip"})
    hk = prof["planner"].get("hook") or {}
    if hk.get("first_line_by_s") and rows and rows[0]["line_times"] and rows[0]["line_times"][0][1] > hk["first_line_by_s"]:
        warnings.append({"where": "line 1 of clip 1", "type": "slow_hook", "severity": "warn",
                         "note": f"first line ends at about {rows[0]['line_times'][0][1]} s; the profile wants the promise by {hk['first_line_by_s']} s"})
    return {"clips": rows, "errors": errors, "warnings": warnings}


# ---------------------------------------------------------------- P3 angle plan
def _pick_joins(nj: int, want: int, open_k: int, gap: int, book: bool, shift) -> tuple:
    """Joins (1..nj) for `want` set-up changes with the largest summed topic shift. Dynamic programming, O(nj^2 * want).

    Rules: the first change not before join `open_k`; changes at least `gap` joins apart; with `book`, the last change is the
    final join (back to set-up A) and is exempt from the gap rule. Returns () when nothing fits."""
    m = want - 1 if book else want
    hi = nj - 1 if book else nj
    cand = list(range(open_k, hi + 1))
    if m == 0:
        return (nj,) if book and nj >= 1 else ()
    if len(cand) < m:
        return ()
    NEG = float("-inf")
    best = {(j, 1): (shift(j), (j,)) for j in cand}
    for c in range(2, m + 1):
        for j in cand:
            prev = [best[(i, c - 1)] for i in cand if i <= j - gap and (i, c - 1) in best]
            if prev:
                v, path = max(prev, key=lambda x: (x[0], [-q for q in x[1]]))
                best[(j, c)] = (v + shift(j), path + (j,))
    ends = [best[(j, m)] for j in cand if (j, m) in best]
    if not ends:
        return ()
    v, path = max(ends, key=lambda x: (x[0], [-q for q in x[1]]))
    if v == NEG:
        return ()
    return path + ((nj,) if book else ())


def _run_setups(seq: list[str], n_changes: int, book: bool) -> list[str]:
    """Set-up per run of clips: cycle through the sequence, never the same set-up twice in a row.
    With `book`, the last run returns to seq[0] when the set-ups allow it (with only two set-ups and an odd number of
    changes it cannot: then the cycle simply continues)."""
    runs, i = [seq[0]], 0

    def nxt(avoid: set[str]) -> str:
        nonlocal i
        for _ in range(2 * len(seq)):
            i += 1
            x = seq[i % len(seq)]
            if x not in avoid:
                return x
        for x in seq:                      # fall back: anything that is a change
            if x != runs[-1]:
                return x
        return runs[-1]

    for r in range(1, n_changes + 1):
        if book and r == n_changes and runs[-1] != seq[0]:
            runs.append(seq[0])
        elif book and r == n_changes - 1:
            runs.append(nxt({runs[-1], seq[0]}))   # keep seq[0] free for the bookend
        else:
            runs.append(nxt({runs[-1]}))
    return runs


def angle_plan(script: Script, prof: dict) -> dict:
    ap = prof["planner"]["angles"]
    clips = script.clips
    dur = sum(c.seconds for c in clips)
    lo_f, hi_f = (x * dur / 180.0 for x in ap["changes_per_3min"])
    nj = len(clips) - 1
    band = [math.floor(lo_f + 0.02), max(math.ceil(hi_f - 0.02), math.floor(lo_f + 0.02))]   # 0.02 absorbs 29.96 s clips
    band = [min(b, nj) for b in band]          # a film with n clips has only n - 1 joins to cut at
    want = min(nj, max(band[0], math.ceil((lo_f + hi_f) / 2 - 0.02)))
    terms = {c.idx: Counter(w for ln in c.lines for w in content_words(ln.text)) for c in clips}

    def shift(k):  # topic shift at join k (between clips[k-1] and clips[k]) = 1 - Jaccard of content words
        a, b = set(terms[clips[k - 1].idx]), set(terms[clips[k].idx])
        return 1 - len(a & b) / max(1, len(a | b))

    seq = ap["sequence"]
    choice: tuple = ()
    if want > 0 and ap["mode"] != "none":
        open_k = min(max(1, ap["min_open_clips"]), nj)
        # relax in this order: the gap between changes, then the opening hold, and only then the number of changes
        for w in range(want, 0, -1):
            for ok in range(open_k, 0, -1):
                for gap in range(ap["min_clips_between"], 0, -1):
                    choice = _pick_joins(nj, w, ok, gap, ap["bookend"] and w >= 2, shift)
                    if choice:
                        break
                if choice:
                    break
            if choice:
                break
    runs = _run_setups(seq, len(choice), ap["bookend"] and len(choice) >= 2)
    setup_of, k = {}, 0
    for i, c in enumerate(clips):
        if i in choice:
            k += 1
        setup_of[c.idx] = runs[k]
    rows, run_side, prev = [], {}, None
    for c in clips:
        s = setup_of[c.idx]
        changed = prev is not None and s != prev
        if changed or prev is None:
            run_side[s] = c.side or ap["default_side"]
        text = ap["setups"][s].format(side=run_side[s])
        if prev is None:
            pic = "first shot"
        elif ap["mode"] == "punch_in":
            pic = "same camera; the editor punches in" if s != seq[0] else "same camera"
        elif changed:
            pic = "new set-up: no last-frame chaining; the set-up reference still is the framing picture"
        else:
            pic = "same set-up: chain the colour-corrected last frame of the previous clip"
        rows.append({"clip": c.idx, "setup": s, "text": text, "change_from_prev": changed, "picture3": pic,
                     "punch_in": ap["mode"] == "punch_in" and s != seq[0]})
        prev = s
    n_changes = sum(r["change_from_prev"] for r in rows)
    refs = {}
    for r in rows:   # one reference-still prompt per set-up that appears after the first shot (production makes these stills)
        if r["setup"] != rows[0]["setup"] and r["setup"] not in refs and ap["mode"] == "generate":
            tpl = (ap.get("ref_prompts") or {}).get(r["setup"]) or f"same person and place, seen as: {ap['setups'][r['setup']]}"
            refs[r["setup"]] = tpl.format(side=run_side.get(r["setup"], ap["default_side"]))
    errors = [] if band[0] <= n_changes <= band[1] or ap["mode"] == "none" else [
        {"where": "film", "type": "angle_changes_out_of_band", "severity": "fail", "changes": n_changes, "band": band}]
    return {"band": band, "changes": n_changes, "joins": [clips[k].idx for k in choice], "clips": rows, "errors": errors, "setup_refs": refs,
            "mode": ap["mode"], "rule": "one locked camera per clip; a change is a hard cut at a clip join (sentence end)"}


# ---------------------------------------------------------------- P4 gesture budget
def gesture_plan(script: Script, prof: dict) -> dict:
    gp = dict(prof["planner"]["gestures"])
    if script.speaker.get("rest"):        # the scene knows its props: "keeps both hands on the steering wheel"
        gp["rest"] = script.speaker["rest"]
    types = gp["types"]
    n_lines = sum(len(c.lines) for c in script.clips)
    max_total = round(gp["density"] * n_lines)
    starts, t = {}, 0.0
    for c in script.clips:
        starts[c.idx] = t
        t += c.seconds
    dens = {r["clip"]: r["line_times"] for r in density(script, prof)["clips"]}
    cands = []
    for c in script.clips:
        for n, ln in enumerate(c.lines, 1):
            for g, spec in types.items():
                if re.search(spec["cue"], ln.text, re.I):
                    cands.append((c.idx, n, g, starts[c.idx] + dens[c.idx][n - 1][0]))
    order = list(types)
    used, last_t, per_clip, pick = Counter(), {}, Counter(), {}
    finger = set(gp["finger_types"])
    for cid, n, g, tf in sorted(cands, key=lambda x: (order.index(x[2]), x[3])):
        if (cid, n) in pick or used[g] >= types[g]["budget"] or sum(used.values()) >= max_total:
            continue
        if g in last_t and tf - last_t[g] < gp["min_gap_s"]:
            continue
        if per_clip[cid] >= gp["max_per_clip"]:
            continue
        if g in finger and sum(used[x] for x in finger) >= gp["finger_max"]:
            continue
        pick[(cid, n)] = g
        used[g] += 1
        last_t[g] = tf
        per_clip[cid] += 1
    rows = []
    variants = gp.get("rest_variants") or []
    if script.speaker.get("rest"):
        variants = []                     # the script's own rest wording wins
    k_rest = 0
    for c in script.clips:
        for n, ln in enumerate(c.lines, 1):
            g = pick.get((c.idx, n))
            m = re.search(r"\b(two|three|four)\b", ln.text.lower())
            if g:
                wording = types[g]["wording"].format(n=m.group(1) if m else "two")
            elif variants:                # cycle neutral poses so no clip holds one frozen pose (hypothesis, see docs)
                wording = variants[k_rest % len(variants)]
                k_rest += 1
            else:
                wording = gp["rest"]
            rows.append({"clip": c.idx, "line": n, "type": g or "rest", "wording": wording})
    errors = [{"where": "film", "type": f"gesture_over_budget:{g}", "severity": "fail"} for g in used if used[g] > types[g]["budget"]]
    if sum(used[x] for x in finger) > gp["finger_max"]:
        errors.append({"where": "film", "type": "too_many_finger_shapes", "severity": "fail"})
    return {"used": dict(used), "total": sum(used.values()), "lines": n_lines, "max_total": max_total, "lines_plan": rows, "errors": errors}


# ---------------------------------------------------------------- P5 colour anchor
def lab_stats_image(path: str | Path) -> dict:
    """Lab mean/std of one image (decoded by ffmpeg, 320x180)."""
    from .colour import frames_rgb, lab_stats

    return lab_stats(list(frames_rgb(str(path), step=1)))


def colour_plan(prof: dict, anchor_still: str | None = None) -> dict:
    cp = prof["planner"]["color"]
    out = {"anchor": "still" if anchor_still else cp["anchor"], "method": prof["editor"]["color"]["method"], "gates": cp["gates"],
           "pre_render": "match the chained seed frame to the anchor before it becomes the next clip's framing picture",
           "post": "per-clip LUT to the anchor in the editor (olv edit), before the join"}
    if anchor_still:
        out["anchor_path"] = Path(anchor_still).name
        try:
            out["target"] = lab_stats_image(anchor_still)
        except Exception as e:  # ffmpeg missing or unreadable image: the editor falls back to clip 1
            out["target"] = None
            out["note"] = f"anchor still not measured ({e}); the editor uses clip {prof['editor']['color']['anchor_clip']}"
    return out


# ---------------------------------------------------------------- P6 prompts
def build_prompts(script: Script, prof: dict, dens: dict, angles: dict, gestures: dict) -> list[dict]:
    sp, pp = script.speaker, prof["planner"]["prompt"]
    tag, lang = sp.get("tag", "S1"), sp.get("language", "English")
    role = sp.get("role", "the speaker")
    out = []
    for i, c in enumerate(script.clips):
        a = angles["clips"][i]
        parts = []
        if script.scene.get("refs"):
            parts.append(script.scene["refs"].strip())
        if i > 0:
            if a["change_from_prev"] and angles["mode"] == "generate":
                parts.append(pp["picture_setup"].format(setup=a["setup"]))
            elif not a["change_from_prev"] or angles["mode"] != "generate":
                parts.append(pp["picture_chain"])
        if sp.get("description"):
            parts.append(sp["description"].strip())
        if script.scene.get("description"):
            parts.append(script.scene["description"].strip())
        framing = a["text"] if angles["mode"] == "generate" else angles["clips"][0]["text"]
        parts.append(f"The camera frames {role} in a {framing}.")
        parts.append(pp["camera_tail"])
        times = dens["clips"][i]["line_times"]
        gl = [g for g in gestures["lines_plan"] if g["clip"] == c.idx]
        beats = [P.Beat(t=times[n][0], action=f"{role} {gl[n]['wording']}", line=ln.text, speaker=tag, language=lang) for n, ln in enumerate(c.lines)]
        scene = P.Scene(description=" ".join(parts), sound=script.scene.get("sound", ""), beats=beats,
                        end_silence_s=prof["planner"]["speech"]["end_margin_s"], voice_ref=bool(sp.get("voice_ref", True)))
        text = P.build(scene, c.seconds, trigger=pp.get("trigger") or None)
        problems = P.lint(text, c.seconds, max_wps=prof["planner"]["speech"]["lint_max_wps"])
        out.append({"clip": c.idx, "seconds": c.seconds, "prompt": text, "words": len(text.split()), "lint": problems})
    return out


# ---------------------------------------------------------------- the plan
def plan(script: Script, prof: dict, anchor_still: str | None = None) -> dict:
    tp = text_pass(script, prof)          # mutates line text (clean) before everything else reads it
    dens = density(script, prof)
    angles = angle_plan(script, prof)
    gestures = gesture_plan(script, prof)
    colour = colour_plan(prof, anchor_still)
    prompts = build_prompts(script, prof, dens, angles, gestures)
    gates = [
        {"id": "G1", "gate": "no banned fake action left", "pass": not [e for e in tp["errors"] if e["type"].startswith("banned_")]},
        {"id": "G2", "gate": "no repeated sentence", "pass": not [e for e in tp["errors"] if e["type"] == "repeated_sentence"]},
        {"id": "G2F", "gate": "no claim contradicts the fact sheet", "pass": not [e for e in tp["errors"] if e["type"] == "claim_vs_facts"]},
        {"id": "G3", "gate": "words per clip inside the band", "pass": not [e for e in dens["errors"] if "words" in e["type"]]},
        {"id": "G4", "gate": "estimated last word by the deadline", "pass": not [e for e in dens["errors"] if e["type"] == "last_word_late"]},
        {"id": "G5", "gate": "angle changes inside the band, only at clip joins", "pass": not angles["errors"]},
        {"id": "G6", "gate": "gesture budget and finger cap kept", "pass": not gestures["errors"]},
        {"id": "G7", "gate": "every prompt lint clean", "pass": all(not p["lint"] for p in prompts)},
    ]
    film = {"title": script.title, "profile": prof["name"], "clips": len(script.clips), "duration_s": round(sum(c.seconds for c in script.clips), 2),
            "angle_changes": angles["changes"], "gestures": gestures["total"], "lines": gestures["lines"]}
    return {"version": 1, "film": film, "gates": gates, "gates_pass": all(g["pass"] for g in gates),
            "text_pass": tp, "density": dens, "angles": angles, "gestures": gestures, "color": colour,
            "clips": [{"clip": c.idx, "title": c.title, "seconds": c.seconds, "lines": [ln.text for ln in c.lines]} for c in script.clips],
            "prompts": prompts}


def write(plan_d: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    for old in (out / "prompts").glob("clip*.txt"):   # a re-plan with fewer clips must not leave stale prompts behind
        if re.fullmatch(r"clip\d+(\.lines)?\.txt", old.name):
            old.unlink()
    for p in plan_d["prompts"]:
        (out / "prompts" / f"clip{p['clip']}.txt").write_text(p["prompt"] + "\n")
        (out / "prompts" / f"clip{p['clip']}.lines.txt").write_text(" ".join(next(c["lines"] for c in plan_d["clips"] if c["clip"] == p["clip"])) + "\n")
    (out / "plan.json").write_text(json.dumps(plan_d, indent=1))
    return out / "plan.json"


def summary(plan_d: dict) -> dict:
    return {"plan_gates_pass": plan_d["gates_pass"], "failed": [g["id"] + " " + g["gate"] for g in plan_d["gates"] if not g["pass"]],
            "errors": [e for part in ("text_pass", "density", "angles", "gestures") for e in plan_d[part]["errors"]],
            "angle_setups": [a["setup"] for a in plan_d["angles"]["clips"]], "gestures_used": plan_d["gestures"]["used"],
            "words_per_clip": [r["words"] for r in plan_d["density"]["clips"]],
            "band": plan_d["density"]["clips"][0]["band"] if plan_d["density"]["clips"] else None,
            "fixes": len(plan_d["text_pass"]["fixes"]), "warnings": len(plan_d["text_pass"]["warnings"]) + len(plan_d["density"].get("warnings", []))}


# ---------------------------------------------------------------- calibration from measured renders
def _load_words(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    w = d.get("words", []) if isinstance(d, dict) else d
    return [{"w": x.get("w", x.get("word", "")), "t0": float(x.get("t0", x.get("start"))), "t1": float(x.get("t1", x.get("end")))}
            for x in w if re.search(r"\w", str(x.get("w", x.get("word", ""))))]


def calibrate(plan_d: dict, words_dir: str | Path, prof: dict) -> dict:
    """Measured speaking rate per clip (word-times files clipN.json or clipN.wer.json) against what the plan assumed.

    The planner assumes the model speaks inside the profile band. If the renders speak slower or faster, the deadline
    (S - margin) and the prompt time codes are wrong. This returns the measured rates and a suggested speech band; it
    changes nothing by itself (use `--write-profile` and re-plan, then test on one clip)."""
    d = Path(words_dir)
    sp = prof["planner"]["speech"]
    rows = []
    for c in plan_d["clips"]:
        f = next((p for p in (d / f"clip{c['clip']}.json", d / f"clip{c['clip']}.wer.json", d / f"clip{c['clip']}" / "wer.json") if p.exists()), None)
        if not f:
            continue
        w = _load_words(f)
        if len(w) < 3:
            continue
        span = max(0.1, w[-1]["t1"] - w[0]["t0"])
        t0, t1 = speech_window(c["seconds"], sp)
        rows.append({"clip": c["clip"], "seconds": c["seconds"], "words": len(w), "first_word_s": round(w[0]["t0"], 2),
                     "last_word_s": round(w[-1]["t1"], 2), "deadline_s": round(t1, 2), "late_s": round(w[-1]["t1"] - t1, 2),
                     "rate_wps": round(len(w) / span, 2), "planned_band": [sp["wps"][0], sp["wps"][1]]})
    if not rows:
        return {"clips": [], "note": "no word files found"}
    by_len: dict[float, list] = {}
    for r in rows:
        by_len.setdefault(round(r["seconds"], 1), []).append(r["rate_wps"])
    from statistics import median    # true median (two clips: their mean, not the faster one)

    med = {k: round(median(v), 2) for k, v in by_len.items()}
    overall = round(median(r["rate_wps"] for r in rows), 2)
    late = [r for r in rows if r["late_s"] > sp["deadline_tolerance_s"]]
    sug = {"wps": [round(overall * 0.95, 2), round(overall * 1.05, 2)], "wps_nominal": round(overall, 2),
           "lint_max_wps": max(sp["lint_max_wps"], round(overall * 1.1, 2))}
    return {"clips": rows, "median_rate_by_clip_seconds": med, "median_rate_wps": overall, "clips_past_deadline": [r["clip"] for r in late],
            "planned_band": sp["wps"], "suggested_speech": sug,
            "hypothesis": "re-planning at the measured rate keeps the last word by the deadline; untested: the model may stretch fewer words "
                          "to fill the clip. Test on one clip before re-rendering a film."}


def calibration_profile(cal: dict, base_profile: str) -> str:
    import yaml

    return ("# Generated by `olv calibrate`: speech band from measured renders. Untested until one clip is rendered with it.\n"
            + yaml.safe_dump({"extends": base_profile, "description": f"{base_profile}, speech band calibrated from {len(cal['clips'])} rendered clips",
                              "planner": {"speech": cal["suggested_speech"]}}, sort_keys=False))
