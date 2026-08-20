from __future__ import annotations

import time
from collections.abc import Callable

_BOB_PERIOD_SECONDS = 3.5

# Public: player.py also uses this to normalize the current offset to -1..1
# for squash.py's squash-and-stretch scale.
BOB_AMPLITUDE_PX = 6


def _ease(u: float) -> float:
    """Smootherstep (Ken Perlin): an S-curve from 0 to 1 with zero first
    *and* second derivative at both ends — noticeably flatter near 0 and 1
    than the more common cubic smoothstep, so a value driven by it holds
    near its endpoints and snaps quickly through the middle.
    """
    return u * u * u * (u * (u * 6 - 15) + 10)


class IdleBob:
    """Continuous, subtle vertical bob for the whole avatar frame.

    Traces an eased triangle wave — one "V" per period, dipping down and
    rising back up — rather than a smoothly rounded sine, so the motion
    reads as a small deliberate nod rather than an organic breathing sway.
    The V's straight lines are themselves eased (see `_ease`) so she holds
    at the top/bottom of the bob and snaps quickly through the middle,
    instead of moving at one constant rate the whole time — more a held
    pose with a quick snap between poses than a smooth, even bob. Runs
    continuously through both idle and speech, so the avatar is never
    perfectly still.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        period_seconds: float = _BOB_PERIOD_SECONDS,
        amplitude_px: float = BOB_AMPLITUDE_PX,
    ) -> None:
        self._clock = clock
        self._period = period_seconds
        self._amplitude = amplitude_px
        self._start = clock()

    def offset_px(self) -> float:
        """Current vertical offset in pixels, ranging -amplitude..+amplitude."""
        elapsed = self._clock() - self._start
        t = (elapsed % self._period) / self._period  # 0..1
        # Progress through the current half-cycle (0..0.5 or 0.5..1), eased,
        # then remapped back into a full 0..1 phase before the triangle wave
        # — same sync points (0, 0.5, 1) as the unmodified wave, but eased
        # in between.
        half_progress = _ease((t % 0.5) / 0.5)
        eased_t = half_progress / 2 if t < 0.5 else 0.5 + half_progress / 2
        triangle = 4 * abs(eased_t - 0.5) - 1  # 1 -> -1 -> 1: one held V per period
        return self._amplitude * triangle
