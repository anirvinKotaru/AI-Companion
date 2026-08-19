from __future__ import annotations

from ai_girlfriend.avatar.bob import IdleBob


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_starts_at_the_top_of_its_range() -> None:
    bob = IdleBob(clock=_FakeClock(), period_seconds=4.0, amplitude_px=3.0)
    assert bob.offset_px() == 3.0


def test_reaches_the_bottom_at_the_midpoint() -> None:
    clock = _FakeClock()
    bob = IdleBob(clock=clock, period_seconds=4.0, amplitude_px=3.0)
    clock.advance(2.0)  # half the period
    assert bob.offset_px() == -3.0


def test_back_to_the_top_after_a_full_period() -> None:
    clock = _FakeClock()
    bob = IdleBob(clock=clock, period_seconds=4.0, amplitude_px=3.0)
    clock.advance(4.0)
    assert bob.offset_px() == 3.0


def test_stays_within_the_amplitude_bounds() -> None:
    clock = _FakeClock()
    bob = IdleBob(clock=clock, period_seconds=4.0, amplitude_px=3.0)
    for _ in range(41):
        clock.advance(0.1)
        assert -3.0 <= bob.offset_px() <= 3.0
