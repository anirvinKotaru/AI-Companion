from __future__ import annotations

import random

from ai_girlfriend.avatar.jitter import Jitter


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_starts_with_no_offset() -> None:
    jitter = Jitter(clock=_FakeClock(), rng=random.Random(0))
    assert jitter.dx == 0.0
    assert jitter.dy == 0.0


def test_tick_before_interval_does_nothing() -> None:
    jitter = Jitter(clock=_FakeClock(), rng=random.Random(0))
    assert jitter.tick() is False
    assert jitter.dx == 0.0
    assert jitter.dy == 0.0


def test_tick_after_interval_picks_a_new_offset_within_amplitude() -> None:
    clock = _FakeClock()
    jitter = Jitter(clock=clock, rng=random.Random(0), interval_seconds=0.15, amplitude_px=2.0)
    clock.advance(0.15)
    assert jitter.tick() is True
    assert -2.0 <= jitter.dx <= 2.0
    assert -2.0 <= jitter.dy <= 2.0


def test_offset_holds_steady_between_ticks() -> None:
    clock = _FakeClock()
    jitter = Jitter(clock=clock, rng=random.Random(0), interval_seconds=0.15, amplitude_px=2.0)
    clock.advance(0.15)
    jitter.tick()
    dx, dy = jitter.dx, jitter.dy
    clock.advance(0.05)  # still within the same interval
    assert jitter.tick() is False
    assert (jitter.dx, jitter.dy) == (dx, dy)
