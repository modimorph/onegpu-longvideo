"""Command line: olv graph | render | upscale | lint | wer | camera."""
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
        res = cli.run(g, out_dir / f"{rc.name}_raw.mp4", timeout_s=a.timeout)
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
    heard = wer.transcribe(a.audio, model=a.model)
    res = wer.score(ref, heard, a.seconds)
    res["pass"] = wer.passes(res, a.max_window_wer)
    print(json.dumps({k: v for k, v in res.items() if k != "errors"}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_camera(a):
    from .camera import analyze

    print(json.dumps(analyze(a.video, plan=a.plan)))


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
    s.set_defaults(fn=cmd_wer)
    s = sub.add_parser("camera", help="estimate the camera move and compare with the plan")
    s.add_argument("video"); s.add_argument("--plan", choices=["push", "pull", "arc", "slide", "locked"]); s.set_defaults(fn=cmd_camera)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
