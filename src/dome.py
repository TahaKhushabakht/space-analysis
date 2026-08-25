"""Phase 5a — the third dimension: vertical reach, and the dome it produces.

From docs/concept.md: vertical extent does NOT grow with the horizon the way horizontal
range does. A jump adds a fixed height and needs a plant-and-leap window, so the height
a player can reach at a point depends on the time left AFTER arriving there:

    z_reach(q) = standing_reach + jump(tau − t_arrive(q))

Near the player there is slack to set and leap — full reach. At the fringe of horizontal
range, arrival consumes the whole horizon and only standing reach remains. That is the
dome: a wide base tapering to a cap, with the cap set by stature and leap (physical
attributes) and the base radius by speed and acceleration (kinematic attributes).

WHY THIS NEEDED NO REFACTOR. reach.py was deliberately written as arrival-TIME fields
over arbitrary target points rather than as a 2D grid primitive, precisely so the z
dimension could be layered on as a function of arrival time. `arrival_times` is reused
unchanged here; nothing from Phases 1-4 had to move.

THE JUMP MODEL is ballistic, not a lookup. To rise h a player leaves the ground at
v = sqrt(2gh) and reaches the apex after t = sqrt(2h/g); inverted, flight time t buys
height g·t²/2. Subtracting a plant time for the leap itself:

    jump(s) = clip( g · max(s − plant_time, 0)² / 2 , 0 , max_jump )

With the defaults below a full leap needs ~0.60s of slack (0.25s plant + 0.35s rise to
0.6m), which is the right order for a standing header. The taper between 0 and 0.6s of
slack is what curves the dome.

UNIFORM PARAMETERS, and that is the honest limit of Phase 5a. Every player gets the same
standing reach and leap, because Metrica is anonymised — there are no heights to attach.
Per-player caps are Phase 5b and are blocked on named tracking (docs/data-scope.md).
Consequently, in 5a all vertical differences between players come from SLACK TIME alone:
whoever arrives earlier can jump higher. That is a real, testable claim, and W6a tests it
against aerial-duel outcomes — but it is not the attribute-driven dome the concept doc
describes, and results here must not be read as if it were.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81  # m/s²

# Population-average values, documented as such. Standing reach ~2.30m corresponds to a
# ~1.83m player with arms raised; 0.60m is a reasonable athletic standing vertical.
# Phase 5b replaces these per player; nothing here should be read as player-specific.
DEFAULT_STANDING_REACH_M = 2.30
DEFAULT_MAX_JUMP_M = 0.60
DEFAULT_PLANT_TIME_S = 0.25


@dataclass
class DomeParams:
    standing_reach_m: float | np.ndarray = DEFAULT_STANDING_REACH_M
    max_jump_m: float | np.ndarray = DEFAULT_MAX_JUMP_M
    plant_time_s: float | np.ndarray = DEFAULT_PLANT_TIME_S

    @property
    def full_jump_slack_s(self) -> float:
        """Slack time needed for a maximal leap: plant + rise to max_jump."""
        mj = float(np.max(self.max_jump_m))
        pt = float(np.max(self.plant_time_s))
        return pt + float(np.sqrt(2.0 * mj / G))


def jump_height(slack_s: np.ndarray, params: DomeParams | None = None) -> np.ndarray:
    """Height added by a leap given `slack_s` seconds spare after arriving.

    Zero while the slack is consumed by planting, then ballistic (g t²/2), capped at
    max_jump. Negative slack (the point is not reachable at all) yields 0.
    """
    params = params or DomeParams()
    t = np.maximum(np.asarray(slack_s, dtype=float) - params.plant_time_s, 0.0)
    return np.minimum(0.5 * G * t**2, params.max_jump_m)


def z_reach(arrivals: np.ndarray, tau: float, params: DomeParams | None = None) -> np.ndarray:
    """Vertical reach per (target, player). NaN where the target is out of horizontal range.

    arrivals: (n_targets, n_players) from reach.arrival_times. Returns the same shape.
    """
    params = params or DomeParams()
    slack = tau - np.asarray(arrivals, dtype=float)
    z = np.asarray(params.standing_reach_m, dtype=float) + jump_height(slack, params)
    return np.where(slack >= 0.0, z, np.nan)


def team_z_reach(arrivals: np.ndarray, tau: float, params: DomeParams | None = None) -> np.ndarray:
    """Highest point any player of the team can reach at each target. (n_targets,).

    The team's positive space in 3D is the union of individual domes, and a union of
    domes over the same ground is just the pointwise MAX of their heights — no
    independence assumption needed, unlike the 2D probabilistic coverage union.
    """
    z = z_reach(arrivals, tau, params)
    if z.shape[1] == 0:
        return np.zeros(z.shape[0])
    with np.errstate(invalid="ignore"):
        out = np.nanmax(z, axis=1)
    return np.nan_to_num(out, nan=0.0)


def reachable_volume(
    arrivals: np.ndarray, tau: float, cell_area_m2: float, params: DomeParams | None = None
) -> float:
    """Volume of the team's 3D positive space (m³): the height field integrated over ground.

    This is the literal `C_D` of the concept doc taken into three dimensions — the space
    the defense occupies in reach terms, not merely the ground it stands on.
    """
    return float(team_z_reach(arrivals, tau, params).sum()) * cell_area_m2


def can_contest(
    arrivals: np.ndarray, tau: float, ball_height_m: float, params: DomeParams | None = None
) -> np.ndarray:
    """Boolean (n_targets, n_players): can this player meet a ball at `ball_height_m` here?

    The aerial top slice. An aerial duel is exactly the contest between overlapping dome
    caps at the ball's arrival point, so this is the primitive W6a's duel test uses.
    """
    z = z_reach(arrivals, tau, params)
    return np.nan_to_num(z, nan=-np.inf) >= ball_height_m
