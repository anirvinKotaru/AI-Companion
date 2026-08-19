from __future__ import annotations

from dataclasses import dataclass

# Rhubarb Lip Sync's mouth-shape set (see docs/design/004-talking-head.md).
# "X" is the idle/relaxed shape used between and after phrases.
IDLE_SHAPE = "X"


@dataclass(frozen=True)
class MouthShape:
    """How far to displace the mouth region for one Rhubarb viseme.

    `open_amount` is 0 (closed) to 1 (wide open), driving vertical
    displacement. `width` is -1 (puckered/narrow) to 1 (wide/stretched),
    driving horizontal displacement. Values are hand-picked approximations
    of Rhubarb's shape descriptions (see its README) — there's no ground
    truth to calibrate against for a warped-photo effect like this one.
    """

    open_amount: float
    width: float


# Every shape Rhubarb can emit (its 6 basic shapes A-F, plus the default
# extended set G, H, X — see --extendedShapes in its CLI reference).
MOUTH_SHAPES: dict[str, MouthShape] = {
    "A": MouthShape(open_amount=0.0, width=0.0),  # closed: P, B, M
    "B": MouthShape(open_amount=0.15, width=0.3),  # slightly open, clenched teeth
    "C": MouthShape(open_amount=0.5, width=0.1),  # open: EH, AE
    "D": MouthShape(open_amount=1.0, width=0.0),  # wide open: AA
    "E": MouthShape(open_amount=0.4, width=-0.3),  # rounded: AO, ER
    "F": MouthShape(open_amount=0.2, width=-0.7),  # puckered: UW, OW, W
    "G": MouthShape(open_amount=0.1, width=0.0),  # teeth on lip: F, V
    "H": MouthShape(open_amount=0.35, width=0.0),  # tongue raised: long L
    "X": MouthShape(open_amount=0.0, width=0.0),  # idle / relaxed
}


def mouth_shape_for(viseme: str) -> MouthShape:
    """Look up a viseme's mouth shape, falling back to idle for anything unknown."""
    return MOUTH_SHAPES.get(viseme, MOUTH_SHAPES[IDLE_SHAPE])
