from __future__ import annotations

import random
import time
from collections.abc import Callable

_JITTER_INTERVAL_SECONDS = 0.15

# Public: player.py also needs this to size the replicated-edge padding it
# adds around each frame, so jitter never exposes empty space at an edge.
JITTER_AMPLITUDE_PX = 2


class Jitter:
    """Small, abrupt random (x, y) offset layered on top of the idle bob.

    Picks a new offset every `interval_seconds` and holds it steady in
    between rather than interpolating toward it — a sudden small jump reads
    as a nervous twitch, where smoothing it out would just look like a
    slightly shaky version of the bob instead of a distinct effect.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        interval_seconds: float = _JITTER_INTERVAL_SECONDS,
        amplitude_px: float = JITTER_AMPLITUDE_PX,
    ) -> None:
        self._clock = clock
        self._rng = rng or random.Random()
        self._interval = interval_seconds
        self._amplitude = amplitude_px
        self.dx = 0.0
        self.dy = 0.0
        self._deadline = clock() + interval_seconds

    def tick(self) -> bool:
        """Advance to a new offset if the interval has elapsed.

        Returns True if `dx`/`dy` changed as a result.
        """
        now = self._clock()
        if now < self._deadline:
            return False
        self.dx = self._rng.uniform(-self._amplitude, self._amplitude)
        self.dy = self._rng.uniform(-self._amplitude, self._amplitude)
        self._deadline = now + self._interval
        return True
