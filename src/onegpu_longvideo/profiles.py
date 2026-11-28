"""Scenario profiles for the planner and the editor (configs/profiles/*.yaml).

A profile holds only data: speech band, angle plan, gesture budget, colour gates, cut, loudness and review rules.
Every profile is deep-merged over `_base.yaml`, so a profile file lists only what differs from the base.
Sources for the numbers: docs/PLANNER.md and docs/EDITOR.md (rule ids point to the research notes).
"""
from __future__ import annotations

import copy
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parents[2] / "configs" / "profiles"


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
    base_p = p.parent / "_base.yaml"
    base = yaml.safe_load(base_p.read_text()) if base_p.exists() else {}
    prof = _merge(base, yaml.safe_load(p.read_text()) or {})
    prof.setdefault("name", p.stem)
    return prof
