"""Timeline prompt builder and linter for one-shot generation.

A prompt is a scene description plus time-coded beats ("At 00:06 ..."). Spoken lines use the model's
dialogue tag: <d>[English] words</d>. Our findings (docs/EXPERIMENTS.md): leftover speaking time gets
filled with invented words, so the lines should fill the planned speech time at a natural pace and the
speaker should close their lips and stay silent after the last line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

REALISM_TRIGGER = "r34l1sm"
_D_TAG = re.compile(r"<d>(.*?)</d>", re.S)


def fmt_time(t: float) -> str:
    """Seconds -> 'MM:SS' (rounded down to whole seconds)."""
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


@dataclass
class Beat:
    t: float
    action: str = ""
    line: str = ""
    speaker: str = "S1"
    language: str = "English"


@dataclass
class Scene:
    description: str
    camera: str = ""
    sound: str = ""
    beats: list[Beat] = field(default_factory=list)
    end_silence_s: float = 1.0
    voice_ref: bool = False


def build(scene: Scene, duration_s: float, trigger: str | None = None) -> str:
    """Render a Scene into one prompt string."""
    parts = []
    if scene.voice_ref:
        parts.append("<Audio 1> is the voice-timbre reference for the speaker (S1); its words are not copied.")
    parts.append(scene.description.strip())
    if scene.camera:
        parts.append(scene.camera.strip())
    if scene.sound:
        parts.append(scene.sound.strip())
    last_line_t = None
    for b in sorted(scene.beats, key=lambda x: x.t):
        s = f"At {fmt_time(b.t)} {b.action.strip()}".rstrip(" .") + "."
        if b.line:
            line = b.line.strip()
            if line[-1] not in ".?!":
                line += "."
            s += f" ({b.speaker}) says: <d>[{b.language}] {line}</d>"
            last_line_t = b.t
        parts.append(s)
    if last_line_t is not None:
        end_t = max(0.0, duration_s - scene.end_silence_s)
        parts.append(f"At {fmt_time(end_t)} the speaker finishes speaking, closes their lips and stays silent and still to the last frame.")
    p = " ".join(parts)
    if trigger and not p.startswith(trigger):
        p = f"{trigger} {p}"
    return p


def spoken_lines(prompt: str) -> list[str]:
    """Text inside every <d>...</d> tag, without the [Language] prefix."""
    return [re.sub(r"^\s*\[[^\]]+\]\s*", "", m).strip() for m in _D_TAG.findall(prompt)]


def lint(prompt: str, duration_s: float, max_wps: float = 3.2, max_words: int = 600) -> list[str]:
    """Return a list of problems (empty list = clean). Checks are cheap text rules from our runs."""
    problems = []
    words = len(prompt.split())
    if words > max_words:
        problems.append(f"prompt has {words} words (> {max_words})")
    for m in re.finditer(r"<d>(.*?)</d>", prompt, re.S):
        if m.group(1) != m.group(1).rstrip():
            problems.append(f"trailing space before </d> at char {m.end()} (acts as a wildcard)")
        if not m.group(1).strip():
            problems.append(f"empty <d></d> at char {m.start()}")
    if prompt.count("<d>") != prompt.count("</d>"):
        problems.append("unbalanced <d> tags")
    n_spoken = sum(len(x.split()) for x in spoken_lines(prompt))
    if n_spoken and duration_s > 0 and n_spoken / duration_s > max_wps:
        problems.append(f"speech rate {n_spoken / duration_s:.2f} words/s > {max_wps}")
    if prompt.count(REALISM_TRIGGER) > 1:
        problems.append("realism trigger appears more than once")
    return problems
