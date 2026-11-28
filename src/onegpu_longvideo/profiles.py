"""Scenario profiles for the planner and the editor (configs/profiles/*.yaml).

A profile holds only data: speech band, angle plan, gesture budget, colour gates, cut, loudness and review rules.
Every profile is deep-merged over `_base.yaml`, so a profile file lists only what differs from the base.
`extends: talking_head` merges a variant over another profile.
`_replace: [planner.gestures.types]` replaces that key instead of merging it (a cinematic short has no desk gestures).
Sources for the numbers: docs/PLANNER.md and docs/EDITOR.md (rule ids point to the research notes).
"""
from __future__ import annotations

import copy
from pathlib import Path


def _profile_dir() -> Path:
    """OLV_PROFILE_DIR, else the repo's configs/profiles (source checkout), else ./configs/profiles (installed package)."""
    import os

    for d in (os.environ.get("OLV_PROFILE_DIR"), Path(__file__).resolve().parents[2] / "configs" / "profiles", Path.cwd() / "configs" / "profiles"):
        if d and Path(d, "_base.yaml").exists():
            return Path(d)
    return Path(__file__).resolve().parents[2] / "configs" / "profiles"


PROFILE_DIR = _profile_dir()


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def names(profile_dir: str | Path | None = None) -> list[str]:
    d = Path(profile_dir) if profile_dir else PROFILE_DIR
    return sorted(p.stem for p in d.glob("*.yaml") if not p.stem.startswith("_"))


def load(name_or_path: str, profile_dir: str | Path | None = None) -> dict:
    """Load a profile by name (`talking_head`) or by path to a YAML file, merged over `_base.yaml`."""
    import yaml

    p = Path(name_or_path)
    if not p.suffix:
        p = (Path(profile_dir) if profile_dir else PROFILE_DIR) / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"profile not found: {p} (known: {', '.join(names(profile_dir))})")
    over = yaml.safe_load(p.read_text()) or {}
    if over.get("extends"):          # a variant profile: merge over its parent (which is merged over the base)
        sib = p.parent / f"{over['extends']}.yaml"       # parent next to this file, else in the profile folder
        base = load(str(sib) if sib.exists() else over["extends"], profile_dir)
        base.pop("name", None)
        base.pop("_replace", None)
    else:
        base_p = p.parent / "_base.yaml"
        base = yaml.safe_load(base_p.read_text()) if base_p.exists() else {}
    prof = _merge(base, over)
    for dotted in over.get("_replace", []):   # keys the profile replaces instead of merging (e.g. planner.gestures.types)
        src, dst = over, prof
        *head, last = dotted.split(".")
        for k in head:
            src, dst = src[k], dst[k]
        dst[last] = copy.deepcopy(src[last])
    prof.setdefault("name", p.stem)
    return prof
