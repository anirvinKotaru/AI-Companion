from __future__ import annotations

import random

from ai_girlfriend.avatar.twitch import HeadTwitch


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _twitch(clock: _FakeClock) -> HeadTwitch:
    return HeadTwitch(clock=clock, rng=random.Random(0))


def test_starts_at_neutral() -> None:
    assert _twitch(_FakeClock()).angle_deg == 0.0


def test_tick_before_deadline_does_nothing() -> None:
    twitch = _twitch(_FakeClock())
    assert twitch.tick() is False
    assert twitch.angle_deg == 0.0


def test_twitch_rotates_away_from_neutral_then_returns() -> None:
    clock = _FakeClock()
    twitch = _twitch(clock)
    clock.advance(100.0)  # past even the longest possible interval

    seen_nonzero = False
    for _ in range(10):
        twitch.tick()
        if twitch.angle_deg != 0.0:
            seen_nonzero = True
        clock.advance(0.06)
    assert seen_nonzero
    assert twitch.angle_deg == 0.0


def test_twitch_alternates_direction() -> None:
    clock = _FakeClock()
    twitch = _twitch(clock)
    clock.advance(100.0)

    angles = []
    for _ in range(4):
        twitch.tick()
        angles.append(twitch.angle_deg)
        clock.advance(0.06)

    assert angles[0] > 0
    assert angles[1] < 0
    assert angles[2] > 0
    assert angles[3] < 0


def test_stays_rare_between_twitches() -> None:
    clock = _FakeClock()
    twitch = _twitch(clock)
    clock.advance(1.0)  # nowhere near even the shortest interval
    assert twitch.tick() is False
    assert twitch.angle_deg == 0.0
