"""Reachable sets: arrival-time fields and coverage probability.

The core of the positive/negative-space framework (docs/concept.md). A player's
POSITIVE SPACE is the region they can reach within a horizon tau; a team's is the union
over its players. Everything downstream — negative space, creation, entry, destruction —
is derived from the arrival-time field this module produces.

RELATIONSHIP TO PITCH CONTROL (validation gate W1). The concept claims pitch control is
the 2D uniform-parameter special case of this model. That is a testable claim, not a
slogan: `arrival_times` here must reproduce `pitch_control_ref._time_to_intercept`
exactly, and `control_probability` must reproduce its team-race sigmoid. src/reach.py is
the generalization; pitch_control_ref.py is the frozen oracle it is measured against.

WHAT THE GENERALIZATION ADDS (beyond the frozen model):
1. **Per-player parameters.** max_speed / reaction_time accept a scalar (uniform, the
   pitch-control case) OR a per-player array. Public pitch-control models assume uniform
   physical parameters; the concept doc names attribute-parameterized domes as one of the
   genuinely underexplored pieces. Phase 1 only plumbs it through — nothing estimates
   per-player values yet.
2. **Coverage within a horizon**, not just a race between teams. Pitch control asks "who
   gets there first"; negative space asks "can ANYONE on the defense get there at all
   within tau". Different question, same arrival times.
3. **Room for the z dimension.** Deliberately written as arrival-time fields over target
   points rather than as a 2D grid primitive, because the 3D dome (Phase 5) is the same
   computation with a z-dependent reach test layered on arrival time. Nothing here is 2D
   in a way that has to be torn up later.

MOTION MODEL (inherited from the frozen oracle, unchanged so W1 can pass):
a player continues at their CURRENT velocity for `reaction_time`, then travels in a
straight line at `max_speed`. Note `max_speed` (5.0 m/s default) is the model's assumed
sustained/controlled movement speed, NOT an observed sprint top speed, and NOT the 12 m/s
noise clip applied to measured velocities in `compute_velocities` — three different
"speeds" that are easy to conflate (see NOTES.md Phase 0 findings).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Defaults match pitch_control_ref.PitchControlParams exactly — W1 depends on it.
DEFAULT_MAX_SPEED = 5.0  # m/s, sustained controlled speed (not sprint top speed)
DEFAULT_REACTION_TIME = 0.7  # s, spent continuing at current velocity
DEFAULT_SIGMA = 0.45  # s, arrival-time uncertainty


@dataclass
class ReachParams:
    """Motion parameters. Each may be a scalar (uniform) or a per-player array.

    Uniform values reproduce the frozen pitch-control model. Per-player values are the
    generalization the concept doc calls for; no estimator populates them yet.
    """

    max_speed: float | np.ndarray = DEFAULT_MAX_SPEED
    reaction_time: float | np.ndarray = DEFAULT_REACTION_TIME
    sigma: float | np.ndarray = DEFAULT_SIGMA


def arrival_times(
    positions: np.ndarray,
    velocities: np.ndarray | None,
    targets: np.ndarray,
    params: ReachParams | None = None,
) -> np.ndarray:
    """Time for each player to reach each target point.

    positions:  (n_players, 2) metric coords
    velocities: (n_players, 2) m/s, or None for stationary (e.g. static freeze frames)
    targets:    (n_targets, 2)
    Returns:    (n_targets, n_players) — same orientation as the frozen oracle's
                `_time_to_intercept`, so the two can be compared elementwise.

    Model: reaction_time at current velocity, then straight-line at max_speed. Because
    the reaction phase uses current velocity, the reachable set SKEWS in the direction of
    travel — a defender sprinting away from a point is correctly treated as further from
    it than their static position suggests. That skew is the main thing lost on
    velocity-free data.
    """
    params = params or ReachParams()
    positions = np.asarray(positions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    n_players = len(positions)

    velocities = np.zeros_like(positions) if velocities is None else np.asarray(velocities, dtype=float)

    # Broadcast scalar-or-array parameters to one value per player.
    reaction = np.broadcast_to(np.asarray(params.reaction_time, dtype=float), (n_players,))
    speed = np.broadcast_to(np.asarray(params.max_speed, dtype=float), (n_players,))

    reaction_pos = positions + velocities * reaction[:, None]  # (n_players, 2)
    diff = targets[:, None, :] - reaction_pos[None, :, :]  # (n_targets, n_players, 2)
    dist = np.linalg.norm(diff, axis=-1)  # (n_targets, n_players)
    return reaction[None, :] + dist / speed[None, :]


def coverage_probability(
    arrivals: np.ndarray, tau: float, params: ReachParams | None = None
) -> np.ndarray:
    """P(each player reaches each target within horizon tau), from arrival times.

    Logistic in the slack `tau - t_arrive`, with scale sigma — the same arrival-time
    uncertainty the frozen model uses for its team race. A player expected to arrive well
    inside tau covers the point with probability ~1; well outside, ~0; sigma sets how
    sharp that transition is.

    NOTE this is a DIFFERENT question from pitch control's, using the SAME arrival times.
    Pitch control asks who wins a race between teams; this asks whether a given player can
    get there at all in the time available. Negative space needs the latter, because space
    is uncovered when nobody can reach it — regardless of who would win a race for it.

    KNOWN BIAS — the reaction-time floor. The inherited motion model charges every player
    `reaction_time` before they move, so a player's arrival time at THEIR OWN LOCATION is
    not zero but

        t_own = reaction_time * (1 + speed / max_speed)

    (0.7s standing still; ~1.7s at a 7 m/s sprint, since the reaction phase carries them
    away and they must come back). Coverage of one's own position is therefore capped at
    sigmoid((tau - t_own)/sigma) — about 0.95 at tau=2s standing still, and lower when
    moving. In a RACE this is harmless because both teams pay it; for COVERAGE it is a
    real conservative bias that understates reach near defenders and so OVERSTATES
    negative space around them, most at high speed. Left in place because W1 requires the
    arrival-time function to stay identical to the frozen oracle; revisit if Phase 2 value
    weighting proves sensitive to it. Verified as exact algebra in validate_phase1 W2(d).

    arrivals: (n_targets, n_players). Returns the same shape.
    """
    params = params or ReachParams()
    sigma = np.asarray(params.sigma, dtype=float)
    return 1.0 / (1.0 + np.exp(-(tau - arrivals) / sigma))


def team_coverage(
    arrivals: np.ndarray, tau: float, params: ReachParams | None = None
) -> np.ndarray:
    """P(at least one player of the team reaches each target within tau).

    Union over players via the complement of independent failures:
        P(covered) = 1 - prod_p (1 - P(player p arrives in time))

    ASSUMPTION, stated because it is not innocuous: arrival-time uncertainties are treated
    as INDEPENDENT across players. They are not — teammates share tracking error, react to
    the same cue, and are physically obstructed by the same bodies. Independence therefore
    OVERSTATES union coverage where several defenders are similarly placed (a compact
    block), which biases measured negative space DOWNWARD exactly where defenders cluster.
    Recorded as a known bias rather than silently assumed away; the deterministic variant
    (`covered_deterministic`) is free of it and is the cross-check.

    arrivals: (n_targets, n_players). Returns (n_targets,).
    """
    if arrivals.shape[1] == 0:
        return np.zeros(arrivals.shape[0])
    p = coverage_probability(arrivals, tau, params)
    return 1.0 - np.prod(1.0 - p, axis=1)


def covered_deterministic(arrivals: np.ndarray, tau: float) -> np.ndarray:
    """Hard-edged coverage: is ANY player's expected arrival time within tau?

    The deterministic reading of the concept doc's `C_D(t, tau)`. No independence
    assumption and no sigma — useful as a cross-check on the probabilistic version, and
    as the definition when a crisp set (rather than a field) is wanted.

    arrivals: (n_targets, n_players). Returns (n_targets,) boolean.
    """
    if arrivals.shape[1] == 0:
        return np.zeros(arrivals.shape[0], dtype=bool)
    return arrivals.min(axis=1) <= tau


def control_probability(
    att_arrivals: np.ndarray, def_arrivals: np.ndarray, params: ReachParams | None = None
) -> np.ndarray:
    """Attacking-team control probability — the pitch-control race, from arrival times.

    Present ONLY so validation gate W1 can demonstrate that this module subsumes the
    frozen oracle. Negative space does not use it; `team_coverage` answers the question
    this project actually asks. Kept here rather than in the validation script so the
    claim "pitch control is a special case" is expressed in the model code itself.

    Returns (n_targets,).
    """
    params = params or ReachParams()
    sigma = np.asarray(params.sigma, dtype=float)
    tau_att = att_arrivals.min(axis=1)
    tau_def = def_arrivals.min(axis=1)
    return 1.0 / (1.0 + np.exp(-(tau_def - tau_att) / sigma))


def pitch_grid(
    cell_m: float = 1.0,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regular grid of target points over the pitch, centre-origin metric coords.

    Returns (grid_x, grid_y, targets) where grid_x/grid_y are (ny, nx) meshgrids for
    plotting and `targets` is (nx*ny, 2) for the arrival-time functions.

    Cell CENTRES are spaced `cell_m` apart and inset half a cell from each touchline, so
    the cells tile the pitch exactly and every cell has area cell_m^2. That matters
    because negative-space AREA is computed as (cell count x cell area) — an endpoint-
    inclusive grid would make edge cells half-width and quietly inflate the total.
    """
    nx = int(round(pitch_length / cell_m))
    ny = int(round(pitch_width / cell_m))
    xs = -pitch_length / 2 + (np.arange(nx) + 0.5) * cell_m
    ys = -pitch_width / 2 + (np.arange(ny) + 0.5) * cell_m
    grid_x, grid_y = np.meshgrid(xs, ys)
    targets = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return grid_x, grid_y, targets


def team_positions_velocities(
    row, team: str, ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Positions/velocities (n_on_pitch, 2) for players of `team` present at this frame.

    Mirrors pitch_control_ref._team_positions_velocities. Reimplemented (not imported)
    only because that one is private; behaviour is intentionally identical, and W1 checks
    that by feeding both paths the same frames.
    """
    import pandas as pd

    positions, velocities = [], []
    for pid in ids:
        x, y = row.get(f"{team}_{pid}_x"), row.get(f"{team}_{pid}_y")
        if pd.isna(x) or pd.isna(y):
            continue
        vx, vy = row.get(f"{team}_{pid}_vx", 0.0), row.get(f"{team}_{pid}_vy", 0.0)
        vx = 0.0 if pd.isna(vx) else vx
        vy = 0.0 if pd.isna(vy) else vy
        positions.append((x, y))
        velocities.append((vx, vy))
    return np.array(positions, dtype=float), np.array(velocities, dtype=float)
