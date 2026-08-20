from __future__ import annotations

import pytest

from ai_girlfriend.avatar.squash import squash_stretch_scale


def test_neutral_at_the_midpoint_between_squash_and_stretch() -> None:
    # Not literally 1.0/1.0 (that's the amount=0 case) — this checks the
    # x/y scale move in opposite directions by the same margin.
    scale_x, scale_y = squash_stretch_scale(0.0, amount=0.08)
    assert scale_y == pytest.approx(1.08)
    assert scale_x == pytest.approx(0.92)


def test_stretches_taller_and_narrower_at_the_center() -> None:
    scale_x, scale_y = squash_stretch_scale(0.0, amount=0.08)
    assert scale_y > 1.0
    assert scale_x < 1.0


def test_squashes_shorter_and_wider_at_the_top_extreme() -> None:
    scale_x, scale_y = squash_stretch_scale(1.0, amount=0.08)
    assert scale_y < 1.0
    assert scale_x > 1.0


def test_squashes_shorter_and_wider_at_the_bottom_extreme() -> None:
    scale_x, scale_y = squash_stretch_scale(-1.0, amount=0.08)
    assert scale_y < 1.0
    assert scale_x > 1.0


def test_symmetric_between_the_two_extremes() -> None:
    assert squash_stretch_scale(1.0, amount=0.08) == squash_stretch_scale(-1.0, amount=0.08)
