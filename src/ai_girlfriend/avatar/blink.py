from __future__ import annotations

import random
import time
from collections.abc import Callable

# Wall-clock gap between the start of one blink and the next.
_MIN_BLINK_INTERVAL_SECONDS = 2.5
_MAX_BLINK_INTERVAL_SECONDS = 6.0

# Gap between one eyelid moving and the other following it, and how long both
# stay shut in between — together these give the blink its lizard-like,
# one-eye-then-the-other feel instead of both eyes snapping shut in sync.
_EYE_STAGGER_SECONDS = 0.1
_BOTH_CLOSED_HOLD_SECONDS = 0.1


class BlinkScheduler:
    """Drives an asymmetric two-eye blink: one eyelid closes, then the other
    follows a beat later; they reopen in the same order.

    Call `tick()` on a regular cadence (a few times a second is plenty) and
    redraw whenever it returns True. `closed` holds the current open/shut
    state per eye id, keyed the same way the caller's eye overlays are.
    """

    def __init__(
        self,
        eye_ids: tuple[str, str],
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self._eye_ids = eye_ids
        self._clock = clock
        self._rng = rng or random.Random()
        self.closed: dict[str, bool] = {eye_ids[0]: False, eye_ids[1]: False}
        self._phase = "idle"
        self._deadline = self._schedule_next_blink()

    def _schedule_next_blink(self) -> float:
        return self._clock() + self._rng.uniform(
            _MIN_BLINK_INTERVAL_SECONDS, _MAX_BLINK_INTERVAL_SECONDS
        )

    def tick(self) -> bool:
        """Advance the state machine if its deadline has passed.

        Returns True if `closed` changed as a result (caller should redraw).
        """
        now = self._clock()
        if now < self._deadline:
            return False

        first, second = self._eye_ids
        if self._phase == "idle":
            self.closed[first] = True
            self._phase = "closing_second"
            self._deadline = now + _EYE_STAGGER_SECONDS
        elif self._phase == "closing_second":
            self.closed[second] = True
            self._phase = "opening_first"
            self._deadline = now + _BOTH_CLOSED_HOLD_SECONDS
        elif self._phase == "opening_first":
            self.closed[first] = False
            self._phase = "opening_second"
            self._deadline = now + _EYE_STAGGER_SECONDS
        else:  # "opening_second"
            self.closed[second] = False
            self._phase = "idle"
            self._deadline = self._schedule_next_blink()
        return True
