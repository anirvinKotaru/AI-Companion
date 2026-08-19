from __future__ import annotations

import random

from ai_girlfriend.avatar.blink import BlinkScheduler


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _scheduler(clock: _FakeClock) -> BlinkScheduler:
    # Fixed seed so the randomized inter-blink interval is deterministic.
    return BlinkScheduler(("left", "right"), clock=clock, rng=random.Random(0))


def test_starts_with_both_eyes_open() -> None:
    scheduler = _scheduler(_FakeClock())
    assert scheduler.closed == {"left": False, "right": False}


def test_tick_before_deadline_does_nothing() -> None:
    scheduler = _scheduler(_FakeClock())
    assert scheduler.tick() is False
    assert scheduler.closed == {"left": False, "right": False}


def test_blink_closes_one_eye_before_the_other() -> None:
    clock = _FakeClock()
    scheduler = _scheduler(clock)
    clock.advance(10.0)  # past even the longest possible inter-blink interval

    assert scheduler.tick() is True
    closed_eyes = [eye for eye, shut in scheduler.closed.items() if shut]
    assert len(closed_eyes) == 1

    # Not enough time has passed yet for the second eye to follow.
    assert scheduler.tick() is False

    clock.advance(0.2)  # past the stagger gap
    assert scheduler.tick() is True
    assert all(scheduler.closed.values())


def test_blink_reopens_in_the_same_order_it_closed() -> None:
    clock = _FakeClock()
    scheduler = _scheduler(clock)
    clock.advance(10.0)
    scheduler.tick()  # first eye closes
    first_eye = next(eye for eye, shut in scheduler.closed.items() if shut)

    clock.advance(0.2)
    scheduler.tick()  # second eye closes
    clock.advance(0.2)
    scheduler.tick()  # first eye reopens

    assert scheduler.closed[first_eye] is False
    other_eye = "right" if first_eye == "left" else "left"
    assert scheduler.closed[other_eye] is True


def test_both_eyes_end_open_after_a_full_blink_cycle() -> None:
    clock = _FakeClock()
    scheduler = _scheduler(clock)
    clock.advance(10.0)
    for _ in range(4):
        scheduler.tick()
        clock.advance(0.2)
    assert scheduler.closed == {"left": False, "right": False}
