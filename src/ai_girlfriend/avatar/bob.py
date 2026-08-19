from __future__ import annotations

import time
from collections.abc import Callable

_BOB_PERIOD_SECONDS = 3.5

# Public: player.py also needs this to size the replicated-edge padding it
# adds around each frame, so bobbing never exposes empty space at the top or
# bottom of the window.
BOB_AMPLITUDE_PX = 6


class IdleBob:
    """Continuous, subtle vertical bob for the whole avatar frame.

    Traces a triangle wave — a straight-line dip down and a straight-line
    rise back up each cycle, i.e. one "V" per period — rather than a
    smoothly rounded sine, so the motion reads as a small deliberate nod
    rather than an organic breathing sway. Runs continuously through both
    idle and speech, so the avatar is never perfectly still.
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
        triangle = 4 * abs(t - 0.5) - 1  # 1 -> -1 -> 1: one V per period
        return self._amplitude * triangle
