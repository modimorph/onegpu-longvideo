import pytest

from onegpu_longvideo import frames as F


@pytest.mark.parametrize("s,n", [(29.96, 719), (15, 362), (10, 243), (5, 124)])
def test_frames_for_seconds(s, n):
    assert F.frames_for_seconds(s) == n


def test_align_is_17k_plus_5():
    for n in range(1, 2000, 37):
        a = F.align_frames(n)
        assert a >= max(5, n) and a % 17 == 5 and a - max(5, n) < 17


def test_latent_t():
    assert F.latent_t(5) == 2
    assert F.latent_t(719) == (714 // 17) * 5 + 2


def test_tokens_768x432():
    t = F.tokens(719, 768, 432)
    assert t["video_rows"] == F.latent_t(719) * 24 * 14
    assert t["audio_rows"] == 2 * round(719 / 24 * 40)


def test_rope_period_component_8_matches_trained_length():
    p = F.rope_time_periods()
    assert abs(p[0] - 0.157) < 0.001
    assert 15.5 < p[8] < 15.9


def test_windows():
    assert F.windows(29.96) == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 25.0), (25.0, 29.96)]
    assert F.windows(0) == []
