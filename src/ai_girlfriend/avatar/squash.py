from __future__ import annotations

# Max scale change in either axis, e.g. 0.08 == up to 8% taller/narrower or
# shorter/wider.
SQUASH_STRETCH_AMOUNT = 0.08


def squash_stretch_scale(
    normalized_offset: float, amount: float = SQUASH_STRETCH_AMOUNT
) -> tuple[float, float]:
    """Return (scale_x, scale_y) for a squash-and-stretch effect driven by
    the idle bob's current position.

    `normalized_offset` is the bob's offset divided by its amplitude, so it
    ranges -1 (bottom) to 1 (top). Stretches (taller, narrower) through the
    middle of the bob, where it's moving fastest, and squashes (shorter,
    wider) at the top/bottom, where it momentarily reverses direction — the
    classic squash-and-stretch pairing with a bounce.
    """
    k = 1 - 2 * abs(normalized_offset)  # +1 at the middle, -1 at either extreme
    return (1 - k * amount, 1 + k * amount)
