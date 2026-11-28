# Editor (`olv edit`, flow steps 9, 11, 12)

The editor turns rendered clips and the film plan into a finished, measured film: word check, cuts, colour match, assembly, loudness, contact sheet and QA numbers.
It needs `ffmpeg` and `ffprobe` on PATH. It runs on the CPU. Source clips are never changed.

```mermaid
flowchart LR
    C["clipN/video.mp4 + clipN/wer.json"] --> E1["E1 word check"] --> E2["E2 cut list"] --> E3["E3 colour match"]
    PJ["plan.json"] --> E2
    PJ --> E3
    E3 --> E4["E4 assemble<br/>punch-in, lower third, x264"] --> E5["E5 loudness<br/>2-pass, AAC measured"] --> E6["E6 review<br/>7x6 sheet + QA"]
    E6 --> O["final.mp4, contact_sheet.jpg, edit_qa.json, edl.json"]
```

## Run it

```bash
# per clip, after the render (word times are stored in the --out file)
olv wer clips/clip1/audio.wav plan/prompts/clip1.lines.txt --seconds 10 --out clips/clip1/wer.json
# whole film
olv edit clips/ --plan plan/plan.json --out edit/ [--title "Title" --subtitle "Line two"] [--dry-run]
```

Exit code 1 when an editor gate fails. `--dry-run` writes `edl.json` and the ffmpeg command without encoding. `--no-wer` allows clips without `wer.json`.

## Stages and gates
| Stage | What it does | Gate | Rule ids |
|---|---|---|---|
| E1 word check | Reads each clip's per-window WER. A window above 3 % fails; 2 or more invented words in a window is labelled gibberish. The fix is a new take of that clip, never an automatic re-render. | G8 | R-ED1 |
| E2 cut list | IN 0.15 s before the first word, OUT 0.25 s after the last. Cuts the later copy of a repeated 4-word run, the first copy of a stutter, and fillers, but only when the script does not have them. A chained clip that replays the end of the previous clip is cut too. Cut points sit in the middle of the gap between words; every audio boundary gets a 30 ms fade. | G9 no boundary inside a word | R-ED2, R-ED3, R-ED4, R-SP3 |
| E3 colour match | Lab mean/std per clip on the kept frames. Anchor = plan target (still) or clip 1. One 33^3 LUT per clip (Reinhard transfer), applied with `lut3d`. Measured again on the final file, per clip and across each join (last 0.5 s vs first 0.5 s). | G10 clip dE76 <= 2.0, join dE76 <= 1.0 | R-CO1, R-CO2 |
| E4 assemble | Frame-exact trims, punch-in crop where the plan says `punch_in`, scale to one size and frame rate, optional lower third (`drawtext`, skipped with a warning when the filter or font is missing), concat, x264 CRF 18. | | R-DO2 |
| E5 loudness | `loudnorm` pass 1 measures, pass 2 normalises in linear mode to -14 LUFS, AAC 192k. `ebur128` measures the AAC file; if the true peak is above -2 dBTP or the level is off by more than 1 LU, it tightens the peak target and runs again (at most 3 tries). | G11 | R-AU1, R-AU2 |
| E6 review | Contact sheet: 7 x 6 = 42 frames evenly spaced over the whole film (`fps=42/duration`, `tile=7x6`). QA numbers in `edit_qa.json`: WER windows, cuts, colour before and after, joins, loudness tries, angle changes and gestures from the plan. A person opens the sheet and watches the film; `flow-check` needs `qa.contact_sheet_reviewed = true`. | G12 sheet written | |

## Measured
- Synthetic 3-clip test (`tests/test_editor.py`, 320x180, 3 s clips with different colour casts): integrated loudness within 1 LU of -14, true peak at or below -2 dBTP, worst clip dE76 more than halved, 42-frame sheet.
- Real 3-min talking-head edit (6 x 30 s, 1920x1080): see `docs/EXPERIMENTS.md`, section "Studio editor on the 3-min demo".

## Not built yet (gaps)
- J and L cuts and B-roll: the generator makes no B-roll yet, so documentary and podcast profiles cannot split audio over a cutaway.
- Captions: the `captions` flag is recorded in the profile; burning them in is not built (word times exist in `wer.json`).
- Blink-aware cut points (cut 1-2 frames before a blink, R-ED5) need a face model; cuts are placed by word gaps only.
- MKL or histogram colour matching (color-matcher is GPL-3.0; run it as a separate program if a test shows it beats Reinhard on skin).
