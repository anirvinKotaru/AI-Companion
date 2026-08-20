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


def test_moves_slowly_near_the_top_and_quickly_through_the_middle() -> None:
    # Just after the top of the range (t=0): the eased wave should barely
    # have moved from its extreme.
    near_top_clock = _FakeClock()
    near_top_bob = IdleBob(clock=near_top_clock, period_seconds=4.0, amplitude_px=3.0)
    start_near_top = near_top_bob.offset_px()
    near_top_clock.advance(0.1)
    near_top_delta = abs(near_top_bob.offset_px() - start_near_top)

    # A quarter-period in (t=0.25): the steepest, fastest-moving part of the
    # swing, between the top and bottom holds.
    mid_clock = _FakeClock()
    mid_bob = IdleBob(clock=mid_clock, period_seconds=4.0, amplitude_px=3.0)
    mid_clock.advance(1.0)
    start_mid = mid_bob.offset_px()
    mid_clock.advance(0.1)
    mid_delta = abs(mid_bob.offset_px() - start_mid)

    assert mid_delta > near_top_delta
