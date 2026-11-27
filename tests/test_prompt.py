from pathlib import Path

from onegpu_longvideo import prompt as P

PROMPTS = Path(__file__).resolve().parents[1] / "configs" / "prompts"


def test_fmt_time():
    assert P.fmt_time(0) == "00:00" and P.fmt_time(65.9) == "01:05"


def test_build_and_lines():
    sc = P.Scene("A host in a studio.", camera="Locked-off.", beats=[P.Beat(0, "he smiles", "Hello there"), P.Beat(5, "he nods", "Welcome back!")])
    p = P.build(sc, 10.0, trigger=P.REALISM_TRIGGER)
    assert p.startswith(P.REALISM_TRIGGER)
    assert P.spoken_lines(p) == ["Hello there.", "Welcome back!"]
    assert "At 00:09" in p and P.lint(p, 10.0) == []


def test_lint_catches_trailing_space_and_rate():
    bad = "x <d>[English] one two three four five six seven eight nine ten </d>"
    probs = P.lint(bad, 2.0)
    assert any("trailing space" in s for s in probs)
    assert any("speech rate" in s for s in probs)


def test_lint_unbalanced_and_length():
    assert any("unbalanced" in s for s in P.lint("<d>[English] hi.", 5))
    assert any("words" in s for s in P.lint("w " * 700, 30))


def test_shipped_prompts_lint_clean():
    for f in PROMPTS.glob("*.txt"):
        if f.name.endswith(".lines.txt"):
            continue
        assert P.lint(f.read_text(), 29.96) == [], f.name
