from __future__ import annotations

import random
import time
from collections.abc import Callable

# "Every once in a blue moon" — much rarer than the blink or bob/jitter.
_MIN_INTERVAL_SECONDS = 25.0
_MAX_INTERVAL_SECONDS = 75.0

# A quick snap one way, back the other way, and a smaller settle — a
# decaying oscillation rather than a single clean swing, so it reads as a
# nervous jerk rather than a deliberate head-shake. (angle_degrees, hold_seconds).
_STEPS: tuple[tuple[float, float], ...] = (
    (5.0, 0.06),
    (-4.0, 0.06),
    (2.0, 0.06),
    (-1.0, 0.06),
)

# Public: the largest |angle| any step reaches. player.py needs this to size
# how much extra replicated-edge padding a rotated frame requires so the
# twitch never exposes empty space at a window edge.
TWITCH_MAX_ANGLE_DEG = max(abs(angle) for angle, _ in _STEPS)


class HeadTwitch:
    """Drives a rare, brief head twitch: a quick rotation back and forth
    that settles to neutral, on a long random interval.

    Call `tick()` on a regular cadence and read `angle_deg` for the current
    rotation to apply (0 when idle, which is nearly all the time).
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        min_interval_seconds: float = _MIN_INTERVAL_SECONDS,
        max_interval_seconds: float = _MAX_INTERVAL_SECONDS,
        steps: tuple[tuple[float, float], ...] = _STEPS,
    ) -> None:
        self._clock = clock
        self._rng = rng or random.Random()
        self._min_interval = min_interval_seconds
        self._max_interval = max_interval_seconds
        self._steps = steps
        self.angle_deg = 0.0
        self._step_index = -1  # -1 == idle, waiting for the next twitch
        self._deadline = self._schedule_next_twitch()

    def _schedule_next_twitch(self) -> float:
        return self._clock() + self._rng.uniform(self._min_interval, self._max_interval)

    def tick(self) -> bool:
        """Advance the state machine if its deadline has passed.

        Returns True if `angle_deg` changed as a result.
        """
        now = self._clock()
        if now < self._deadline:
            return False

        self._step_index += 1
        if self._step_index >= len(self._steps):
            self._step_index = -1
            self.angle_deg = 0.0
            self._deadline = self._schedule_next_twitch()
            return True

        angle, hold = self._steps[self._step_index]
        self.angle_deg = angle
        self._deadline = now + hold
        return True
