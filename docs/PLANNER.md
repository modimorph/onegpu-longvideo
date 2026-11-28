# Planner (`olv plan`, flow steps 1-2)

The planner reads the whole script once and writes one film plan before any clip prompt exists. Then it builds every clip prompt from that plan.
Rule: decode the whole, then encode the parts. A clip planned alone repeats the framing, the gestures and the grade drift of the clip before it.

```mermaid
flowchart LR
    S["script.yaml or Markdown segment tables"] --> P1["P1 text pass"] --> P2["P2 speech density"] --> P3["P3 angle plan"]
    P3 --> P4["P4 gesture budget"] --> P5["P5 colour anchor"] --> P6["P6 prompts + lint"]
    PR["configs/profiles/NAME.yaml"] --> P1
    P6 --> O["plan.json, prompts/clipN.txt, prompts/clipN.lines.txt"]
```

## Run it

```bash
olv profiles                                    # list scenario profiles
olv plan configs/scripts/example_talking_head.yaml --profile talking_head --out plan/
olv plan SCRIPT.md --scene scene.yaml --clip-seconds 10 --profile talking_head --out plan/   # Markdown script + speaker/scene text
olv plan script.yaml --anchor-still look.png --llm --out plan/                              # colour anchor from a still; write the review request
olv plan script.yaml --facts facts.yaml --out plan/                                         # fail lines whose claims the facts contradict
```

Exit code 1 when a gate fails (`--no-gate` to inspect anyway). `prompts/clipN.lines.txt` is the reference text for `olv wer`.
`--llm` writes `text_review_request.md` (the whole script plus the automatic flags). With `OLV_LLM_URL` (an OpenAI-compatible endpoint), `OLV_LLM_MODEL` and optional `OLV_LLM_KEY` set, it also sends it and saves `text_review.md`. The answer is advice for the writer. It is never applied automatically.

## Script format
YAML: `title`, `clip_seconds`, `speaker` (`tag`, `role`, `language`, `voice_ref`, `description`), `scene` (`refs`, `description`, `sound`), `clips` (each: `title`, optional `seconds`, optional `side`, `lines`). See `configs/scripts/example_talking_head.yaml`.
Markdown: `## Segment N (t0-t1 s): title` followed by a table `| # | a-b | spoken line | gesture |`. The gesture column is read as a hint only and never copied into a prompt. Speaker and scene text come from `--scene`.

## Stages and gates
| Stage | What it does | Gate (plan fails when) | Rule ids |
|---|---|---|---|
| P1 text pass | Auto-fixes fake actions: stutters ("you'd, you'd"), coughs and other bracketed actions, fillers, self-corrections ("... well, nearly"), fake stops (ellipsis, dash). Flags repeated sentences (a 5-word run said twice in the film), repeated ideas across clips, weak joins between clips, digits, long lines. | G1 a banned action survives; G2 a repeated sentence | R-SP3 |
| P1b fact check | `--facts facts.yaml` (or `facts:` in the script): each fact has `keywords` (all must appear in the line), `allowed` (one must appear) and `forbid` (regex). Use the production data, article and logs. The fact sheet also goes into the LLM review request. | G2F a claim contradicts the fact sheet | R-SP4 |
| P2 speech density | Words per clip must fit `wps` x (deadline - first word). Deadline = S - 1.7 s; for clips of 15 s or less also 14.5 s (this only binds if a profile sets `end_margin_s` below 0.5 s). Line times are estimated at the clip's own rate clamped into the band and become the prompt time codes. Scripts in another language get a warning: the band was measured on English. Hook check: the first line should end by `hook.first_line_by_s` (warning). | G3 words outside the band (too few invites invented words; too many rushes); G4 estimated last word late | R-SP1, R-SP2, R-HK1 |
| P3 angle plan | Target changes = profile band scaled to the film length (talking head: 2-3 per 3 min), capped by the number of joins. The joins are chosen by dynamic programming (largest summed topic shift); the set-up sequence cycles for long films. Changes only at clip joins (one locked camera per clip), at the joins with the largest topic shift, never before `min_open_clips`, at least `min_clips_between` apart, last join back to set-up A (bookend). New set-up: no last-frame chaining, use that set-up's reference still. `punch_in` mode leaves the camera and lets the editor crop. | G5 changes outside the band | R-AN1, R-AN2, R-AN3 |
| P4 gesture budget | Each gesture type has a film budget (1-2), a meaning cue (regex), and a minimum gap (40 s). At most `density` x lines in total, `max_per_clip` per clip, finger shapes capped at `finger_max`. Every other line rests. | G6 a budget or the finger cap broken (a consistency check of the written plan: the selection already keeps the budgets) | R-GE1, R-GE2 |
| P5 colour anchor | Anchor = clip 1 (default) or a still (`--anchor-still`, measured Lab mean/std). Records the gates the editor enforces. | (checked in the editor, G10) | R-CO1, R-CO2 |
| P6 prompts | One prompt per clip with `prompt.build`: references, framing picture sentence from the angle plan, speaker, scene, framing, locked camera, time-coded lines with the gesture or rest wording, end silence. Then `prompt.lint`. | G7 any lint problem | flow step 2 |

Talking head, 6 x 30 s (our 3-min demo script): set-ups A A B B C A (changes at 60, 120 and 150 s), 11 gestures in 36 lines (was 36 cues, 8 finger shapes), 6 fake actions auto-fixed, G3 failed on clip 2 (75 words after the fixes, band 79-86), and a fact sheet from the production data failed 6 lines (G2F: render time, gestures, camera plan, voice conversion, speech limit, takes). The writer fixed both before any render.

## Calibrate the speech band from renders (`olv calibrate`)
The band is a prediction. The model sets its own pace: in our first 10 s talking-head clip it spoke 24 words at 2.46 w/s and ended at 9.76 s, against a planned 3.0 w/s and a deadline of 8.3 s.
```bash
olv calibrate plan/plan.json words/ --write-profile calibrated.yaml   # words/: clipN.json word times or clipN.wer.json
olv plan script.yaml --profile calibrated.yaml --out plan_cal/          # the calibrated profile extends the original one
```
It reports the measured rate per clip, the clips past the deadline and a band of +/- 5 % around the median rate. Untested: whether fewer words keep the last word earlier, or whether the model stretches them to fill the clip. Test it on one clip before re-rendering a film.

## Profiles
`configs/profiles/_base.yaml` holds every key with the talking-head numbers. A profile lists only what differs.

| Profile | Speech w/s | Angles | Gestures | Editor differences |
|---|---|---|---|---|
| talking_head | 2.8-3.1 | generate, 2-3 per 3 min, bookend | 0.4 per line | jump cuts for flaws only |
| talking_head_varied | as talking_head | as talking_head | rest lines cycle 4 relaxed poses (A/B variant) | as talking_head |
| tutorial | 2.5-2.9 | punch_in, 3-6 per 3 min | 0.25, pointing x4 | punch-in 1.15, lower third, captions |
| podcast | 2.6-3.0 | generate, 1-2 per 3 min (wide, close) | 0.5 | fillers and stutters cut, lower third |
| cinematic_short | 1.2-2.8 | a new shot at every join (wide, medium, close) | 0.3 | no internal cuts |
| ad | 2.8-3.2 | a new shot at every join, hook by 3 s | 0.5 | short tail, lower third at 0.5 s, captions |
| documentary | 2.4-2.9 | 3-5 per 3 min, waist-up, close for emphasis | 0.35 | max shot 40 s, lower third |

## Limits (untested until the production test)
- Fewer gestures (0.4 per line) may change lip sync or WER. The old recipe tied one gesture to each line.
- A second camera set-up needs a reference still at that angle; identity hold at 30 degrees is untested.
- The text pass is rules plus an optional LLM review. It does not judge facts or logic by itself.
- Speech-rate bands for non-talking-head profiles come from sources and are not measured on this model yet.
