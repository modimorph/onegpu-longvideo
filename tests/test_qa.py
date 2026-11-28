import numpy as np
import pytest

from onegpu_longvideo.qa import detectors as D
from onegpu_longvideo.qa import wer as Wr


def _words(text, t0=0.0, step=0.4):
    return [Wr.Word(w, t0 + i * step, t0 + i * step + 0.3) for i, w in enumerate(text.split())]


def test_wer_clean():
    ref = "one two three four five six"
    r = Wr.score(ref, _words(ref), 10.0)
    assert r["wer"] == 0 and Wr.passes(r) and len(r["windows"]) == 2


def test_wer_invented_words_located_in_window():
    ref = " ".join(f"w{i}" for i in range(20))
    heard = _words(ref, step=0.5)
    heard.insert(15, Wr.Word("blah", 7.6, 7.8))
    heard.insert(16, Wr.Word("blah", 7.8, 7.9))
    r = Wr.score(ref, heard, 10.0)
    assert r["windows"][0]["invented"] == 0
    assert r["windows"][1]["invented"] == 2
    assert not Wr.passes(r)


def test_wer_missing_and_wrong():
    r = Wr.score("a b c d", _words("a x d"), 5.0)
    ops = sorted(e["op"] for e in r["errors"])
    assert ops == ["missing", "wrong"]


def test_runs_and_stepdet():
    assert D.runs([0, 1, 1, 0, 1, 0, 0, 1], gap=1) == [(1, 4), (7, 7)]
    x = np.r_[np.zeros(50), np.ones(50)]
    s = D.stepdet(x, k=10)
    assert s[50] == 1.0 and s[20] == 0.0


def test_identity_flags():
    c = np.r_[np.full(48, 0.8), np.full(48, 0.3)]
    types = {f["type"] for f in D.identity_flags(c, fps=24)}
    assert types == {"identity_drift", "identity_jump"}


def test_seam_flags():
    d = np.abs(np.random.default_rng(0).normal(1, 0.1, 200))
    d[120] = 10
    f = D.seam_flags(d, fps=24)
    assert [x["t"] for x in f] == [5.0]


def test_gesture_repeat_detects_replay():
    rng = np.random.default_rng(1)
    fps = 24
    g = np.sin(np.linspace(0, 6, 48))[:, None] * np.array([1.0, -0.5, 0.8, 0.3])
    noise = rng.normal(0, 0.05, (400, 4))
    X = noise.copy()
    X[24:72] += g
    X[240:288] += g
    hits = D.gesture_repeats(X, fps)
    assert any(abs(h["t_first"] - 1.0) < 0.01 and abs(h["t_repeat"] - 10.0) < 0.01 for h in hits)


def test_av_offset():
    fps = 24
    rng = np.random.default_rng(2)
    m = np.convolve(rng.random(300), np.ones(3) / 3, "same")
    a = np.r_[np.zeros(2), m[:-2]]
    ms, c = D.av_offset_ms(m, a, fps)
    assert abs(ms - 2 * 1000 / 24) < 1e-6 and c > 0.9


def test_merge_windows():
    w = D.merge_windows([{"t_start": 1, "t_end": 2}, {"t_start": 2.2, "t_end": 3}, {"t": 10}], pad_s=0.5, max_len_s=3)
    assert w[0] == (0.5, 3.5) and w[-1] == (9.5, 10.5) and all(b - a <= 3 for a, b in w)


def test_normalize_keeps_unicode_words():
    from onegpu_longvideo.qa import wer
    assert wer.normalize("Größe, naïve café") == ["größe", "naïve", "café"]



@pytest.mark.parametrize("a,b", [("fifteen seconds", "15 seconds"), ("eighty-seven words", "87 words"), ("ten-eighty", "1080"),
                                 ("at full ten-eighty", "at full 1080p"), ("forty five", "45"), ("ten percent", "10%"),
                                 ("two hundred and five", "205"), ("one thousand", "one thousand"), ("twenty four gigabytes", "24 gigabytes")])
def test_numbers_match_in_either_form(a, b):
    from onegpu_longvideo.qa import wer
    assert wer.normalize(a) == wer.normalize(b)


def test_fifteen_vs_15_is_not_an_error():
    from onegpu_longvideo.qa import wer
    heard = [wer.Word(t, i * 0.4, i * 0.4 + 0.3) for i, t in enumerate("Push it past 15 and it slips.".split())]
    assert wer.score("Push it past fifteen and it slips.", heard, 5.0)["wer"] == 0



@pytest.mark.parametrize("uk,us", [("colour", "color"), ("the colours", "the colors"), ("colour graded", "color graded"),
                                   ("grey", "gray"), ("organised", "organized"), ("analysing", "analyzing"), ("favourite", "favorite")])
def test_british_and_american_spellings_match(uk, us):
    from onegpu_longvideo.qa import wer
    assert wer.normalize(uk) == wer.normalize(us)
    assert wer.normalize("promise") == ["promise"] and wer.normalize("our hour") == ["our", "hour"]



def test_room_review_message_puts_reference_first():
    from onegpu_longvideo.qa import stage2
    m = stage2.build_room_message(b"REF", [(1.5, b"F1"), (2.0, b"F2")], {"kinds": ["step"]})
    c = m[0]["content"]
    assert "same_room" in c[0]["text"] and "step" in c[0]["text"]
    assert c[1]["type"] == "image_url" and c[2]["text"] == "t=1.50 s" and len(c) == 6


def test_background_compare_flags_a_new_object():
    cv2 = pytest.importorskip("cv2")
    from onegpu_longvideo.qa import background as B
    rng = np.random.default_rng(0)
    ref = (rng.random((B.H, B.W, 3)) * 60 + 100).astype(np.uint8)
    ref = cv2.GaussianBlur(ref, (5, 5), 0)
    same = ref.copy()
    lamp = ref.copy()
    lamp[60:200, 10:110] = (230, 200, 150)          # a bright new object at the left
    bg = np.ones((B.H, B.W), bool)
    a, b = B.compare(same, ref, bg), B.compare(lamp, ref, bg)
    assert a["ssim"] > 0.99 and a["cell_ssim"] > 0.99
    assert b["cell_ssim"] < 0.3 and B.where(b["cell"]).endswith("left")
