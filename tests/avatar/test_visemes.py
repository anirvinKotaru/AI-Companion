from __future__ import annotations

from ai_girlfriend.avatar.visemes import IDLE_SHAPE, MOUTH_SHAPES, mouth_shape_for


def test_every_rhubarb_shape_is_mapped() -> None:
    # Rhubarb's full basic + default extended shape set (see its README /
    # --extendedShapes default "GHX").
    assert set(MOUTH_SHAPES) == set("ABCDEFGHX")


def test_idle_shape_is_fully_closed() -> None:
    idle = mouth_shape_for(IDLE_SHAPE)
    assert idle.open_amount == 0.0
    assert idle.width == 0.0


def test_unknown_viseme_falls_back_to_idle() -> None:
    assert mouth_shape_for("?") == mouth_shape_for(IDLE_SHAPE)


def test_wide_open_shape_opens_more_than_closed_shape() -> None:
    assert mouth_shape_for("D").open_amount > mouth_shape_for("A").open_amount
