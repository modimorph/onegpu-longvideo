"""Command line: olv plan | calibrate | graph | render | upscale | lint | wer | camera | edit | flow-check | profiles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import prompt as P
from . import recipe as R
from . import workflow as W
from .frames import tokens


def _prompt_for(rc: R.Recipe, cfg_path: str, override: str | None) -> str:
    pf = override or rc.prompt_file
    if not pf:
        sys.exit("no prompt: set prompt_file in the recipe or pass --prompt")
    p = Path(pf)
    if not p.is_absolute():
        p = Path(cfg_path).parent / p
    return p.read_text().strip()


def cmd_graph(a):
    rc = R.load(a.recipe)
    g = W.build(_prompt_for(rc, a.recipe, a.prompt), rc, output_prefix=f"onegpu/{rc.name}")
    errs = W.validate(g)
    if errs:
        sys.exit("graph errors:\n" + "\n".join(errs))
    out = json.dumps(g, indent=1)
    if a.out:
        Path(a.out).write_text(out)
        print(json.dumps({"graph": a.out, "nodes": len(g), "tokens": tokens(rc.frames, rc.width, rc.height), "custom_nodes": W.uses_custom_nodes(g)}))
    else:
        print(out)


def cmd_render(a):
    from .comfy import ComfyClient
    from .timing import TimingLog

    rc = R.load(a.recipe)
    pr = _prompt_for(rc, a.recipe, a.prompt)
    problems = P.lint(pr, rc.frames / 24)
    if problems and not a.force:
        sys.exit("prompt lint failed (use --force to render anyway):\n" + "\n".join(problems))
    g = W.build(pr, rc, output_prefix=f"onegpu/{rc.name}")
    out_dir = Path(a.out_dir)
    log = TimingLog(out_dir / "timings.jsonl")
    cli = ComfyClient(a.server)
    with log.stage("render", frames=rc.frames, size=[rc.width, rc.height]) as row:
        res = cli.run(g, out_dir / (f"{rc.name}_{int(rc.upscale_to[1])}p.mp4" if rc.upscale_in_graph else f"{rc.name}_raw.mp4"), timeout_s=a.timeout,
                      raw_dest=out_dir / f"{rc.name}_raw.mp4" if rc.upscale_in_graph else None)
        res["upscaled_in_graph"] = bool(rc.upscale_in_graph)
        row.update(exec_s=res["exec_s"])
    (out_dir / f"{rc.name}_recipe.json").write_text(json.dumps(rc.to_dict(), indent=1))
    print(json.dumps({**res, "timings": log.rows}))


def cmd_upscale(a):
    from .upscale import upscale

    print(json.dumps(upscale(a.input, a.output, a.model, a.width, a.height, batch=a.batch, quality=a.quality, x264=a.x264)))


def cmd_lint(a):
    pr = Path(a.prompt).read_text()
    problems = P.lint(pr, a.seconds)
    print(json.dumps({"words": len(pr.split()), "spoken_words": sum(len(x.split()) for x in P.spoken_lines(pr)), "problems": problems}))
    sys.exit(1 if problems else 0)


def cmd_wer(a):
    from .qa import wer

    ref = Path(a.reference).read_text() if Path(a.reference).exists() else a.reference
    if a.words:   # word times made elsewhere (e.g. faster-whisper on another machine): [{"w"|"word", "t0"|"start", "t1"|"end"}]
        heard = [wer.Word(x.get("w", x.get("word", "")), float(x.get("t0", x.get("start"))), float(x.get("t1", x.get("end"))), float(x.get("p", 1.0)))
                 for x in json.loads(Path(a.words).read_text())]
    else:
        heard = wer.transcribe(a.audio, model=a.model)
    aliases = dict(x.split("=", 1) for x in (a.alias or []))
    res = wer.score(ref, heard, a.seconds, aliases=aliases)
    res["pass"] = wer.passes(res, a.max_window_wer)
    res["words"] = [{"w": w.text.strip(), "t0": round(w.start, 3), "t1": round(w.end, 3), "p": round(w.prob, 3)} for w in heard]  # olv edit reads these
    print(json.dumps({k: v for k, v in res.items() if k not in ("errors", "words")}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_plan(a):
    import os

    from . import planner as PL
    from . import profiles

    try:
        prof = profiles.load(a.profile)
        sc = PL.load_script(a.script, clip_seconds=a.clip_seconds, scene_path=a.scene)
        sc.facts += PL.load_facts(a.facts)
    except (FileNotFoundError, ValueError, KeyError) as e:
        sys.exit(f"olv plan: {e}")
    plan = PL.plan(sc, prof, anchor_still=a.anchor_still)
    path = PL.write(plan, a.out)
    if a.llm:
        url, model = os.environ.get("OLV_LLM_URL"), os.environ.get("OLV_LLM_MODEL", "")
        req = PL.llm_review_request(sc, plan["text_pass"])
        (Path(a.out) / "text_review_request.md").write_text(req)
        if url:
            (Path(a.out) / "text_review.md").write_text(PL.llm_review(req, url, model, os.environ.get("OLV_LLM_KEY", "")))
    print(json.dumps({"plan": str(path), **PL.summary(plan)}))
    sys.exit(0 if plan["gates_pass"] or a.no_gate else 1)


def cmd_calibrate(a):
    from . import planner as PL
    from . import profiles

    plan = json.loads(Path(a.plan).read_text())
    name = a.profile or plan.get("film", {}).get("profile") or "talking_head"
    cal = PL.calibrate(plan, a.words_dir, profiles.load(name))
    if a.write_profile and cal.get("clips"):
        Path(a.write_profile).write_text(PL.calibration_profile(cal, Path(name).stem))
        cal["profile_written"] = a.write_profile
    print(json.dumps(cal, indent=1))
    sys.exit(0 if cal.get("clips") else 1)


def cmd_edit(a):
    from . import editor as ED
    from . import profiles

    plan = json.loads(Path(a.plan).read_text()) if a.plan else None
    prof = profiles.load(a.profile or (plan or {}).get("film", {}).get("profile") or "talking_head")
    try:
        qa = ED.edit(a.clips_dir, plan, prof, a.out, dry_run=a.dry_run, require_wer=not a.no_wer, title=a.title or "", subtitle=a.subtitle or "",
                     wer_dir=a.wer_dir, captions=a.captions, normalize_framing=a.normalize_framing)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"olv edit: {e}")
    print(json.dumps(ED.summary(qa)))
    sys.exit(0 if a.dry_run or qa["gates_pass"] else 1)


def cmd_profiles(a):
    from . import profiles

    found = profiles.names()
    if not found:
        sys.exit(f"no profiles in {profiles.PROFILE_DIR}: run from the repo checkout (pip install -e .) or set OLV_PROFILE_DIR")
    for n in found:
        print(f"{n:20s} {profiles.load(n).get('description', '')}")


def cmd_camera(a):
    from .camera import analyze

    print(json.dumps(analyze(a.video, plan=a.plan)))


def cmd_flow_check(a):
    from . import flow

    rows = flow.check(a.run_dir)
    for r in rows:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  step {r['step']:>2}  {r['gate']}  {r['note']}")
    sys.exit(0 if flow.passed(rows) else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="olv", description="Long one-shot audio-video generation on one GPU")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("graph", help="build the ComfyUI API graph (no GPU)")
    s.add_argument("recipe"); s.add_argument("--prompt"); s.add_argument("--out"); s.set_defaults(fn=cmd_graph)
    s = sub.add_parser("render", help="render through a running ComfyUI")
    s.add_argument("recipe"); s.add_argument("--prompt"); s.add_argument("--out-dir", default="out"); s.add_argument("--server")
    s.add_argument("--timeout", type=float, default=3600); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_render)
    s = sub.add_parser("upscale", help="streaming x2 upscale (CUDA)")
    s.add_argument("input"); s.add_argument("output"); s.add_argument("model"); s.add_argument("--width", type=int, default=1920)
    s.add_argument("--height", type=int, default=1080); s.add_argument("--batch", type=int, default=8); s.add_argument("--quality", type=int, default=18)
    s.add_argument("--x264", action="store_true"); s.set_defaults(fn=cmd_upscale)
    s = sub.add_parser("lint", help="check a prompt")
    s.add_argument("prompt"); s.add_argument("--seconds", type=float, default=29.96); s.set_defaults(fn=cmd_lint)
    s = sub.add_parser("wer", help="per-window WER of an audio file against the intended lines")
    s.add_argument("audio"); s.add_argument("reference"); s.add_argument("--seconds", type=float, default=29.96)
    s.add_argument("--model", default="large-v3-turbo"); s.add_argument("--max-window-wer", type=float, default=0.03); s.add_argument("--out")
    s.add_argument("--words", help="JSON word times from another ASR run; skips transcription (audio may be '-')")
    s.add_argument("--alias", action="append", help="'heard form=script form' for a name the ASR cannot spell (logged in the output)")
    s.set_defaults(fn=cmd_wer)
    s = sub.add_parser("camera", help="estimate the camera move and compare with the plan")
    s.add_argument("video"); s.add_argument("--plan", choices=["push", "pull", "arc", "slide", "locked"]); s.set_defaults(fn=cmd_camera)
    s = sub.add_parser("plan", help="whole-film planner (flow steps 1-2): plan.json + one prompt per clip")
    s.add_argument("script", help="script YAML or Markdown segment tables"); s.add_argument("--profile", default="talking_head")
    s.add_argument("--clip-seconds", type=float, help="override every clip length"); s.add_argument("--scene", help="YAML with speaker and scene text")
    s.add_argument("--anchor-still", help="image whose grade is the colour anchor"); s.add_argument("--out", default="plan")
    s.add_argument("--facts", help="fact sheet YAML (production data, article, logs); a contradicted claim fails G2F")
    s.add_argument("--llm", action="store_true", help="write the whole-script review request; send it if OLV_LLM_URL is set")
    s.add_argument("--no-gate", action="store_true", help="exit 0 even when a plan gate fails"); s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("edit", help="studio editor (flow steps 9, 11, 12): cut, colour match, loudness, contact sheet, QA")
    s.add_argument("clips_dir"); s.add_argument("--plan"); s.add_argument("--profile"); s.add_argument("--out", default="edit")
    s.add_argument("--title"); s.add_argument("--subtitle"); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--wer-dir", help="folder with clipN.wer.json files (keeps the clip folders clean)")
    s.add_argument("--captions", dest="captions", action="store_true", default=None, help="burn captions in (default: the profile)")
    s.add_argument("--no-captions", dest="captions", action="store_false", help="do not burn captions in (captions.srt is still written)")
    s.add_argument("--normalize-framing", action="store_true", help="crop each clip to the tightest camera view of its set-up (needs opencv)")
    s.add_argument("--no-wer", action="store_true", help="allow clips without wer.json (G8 then only covers checked clips)"); s.set_defaults(fn=cmd_edit)
    s = sub.add_parser("profiles", help="list scenario profiles"); s.set_defaults(fn=cmd_profiles)
    s = sub.add_parser("bgcheck", help="background consistency of a clip vs a reference clip or image (needs opencv + mediapipe)")
    s.add_argument("video"); s.add_argument("reference"); s.add_argument("--fps", type=float, default=2.0); s.add_argument("--out")
    s.set_defaults(fn=lambda a: __import__("onegpu_longvideo.qa.background", fromlist=["main"]).main(
        [a.video, a.reference, "--fps", str(a.fps)] + (["--out", a.out] if a.out else [])))
    s = sub.add_parser("calibrate", help="measured speaking rate of rendered clips vs the plan; suggests a speech band")
    s.add_argument("plan"); s.add_argument("words_dir", help="folder with clipN.json word times or clipN.wer.json files")
    s.add_argument("--profile"); s.add_argument("--write-profile", help="write a profile YAML (extends the plan's profile) with the suggested band")
    s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser("flow-check", help="validate a run folder against the required-flow gates (docs/FLOW.md)")
    s.add_argument("run_dir"); s.set_defaults(fn=cmd_flow_check)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
