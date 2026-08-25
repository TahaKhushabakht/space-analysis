"""Valuable negative space (VNS): value weighting + the offside/exploitability mask.

Phase 2 of NOTES.md. Phase 1 measured that raw |N| is 68–79% of the pitch — mostly
worthless acreage. This module produces the number that means something:

    VNS(t) = sum over cells in [ N(t, tau, theta) ∩ exploitable(t, tau) ] of
             v(cell) * cell_area

with v from the season-scale xT adapter (value_surface.py) and `exploitable` the hard
offside mask of validation.md Gap B (option B2). Units: xT·m² — "expected-threat-
weighted square meters of space the defense cannot reach and the attack may legally
use". Comparable across frames at fixed (tau, theta) only; both knobs stay explicit.

THE OFFSIDE MASK, and its direction of error. Space behind the second-last defender is
exactly where xT concentrates, so unmasked VNS overstates the headline number in the
most valuable region — the hole caught at planning review (Gap B). Rule implemented,
in the ATTACKING team's oriented frame (attack -> +x):

    line     = max( second-highest oriented defender x, 0 )        [halfway-line rule]
    a cell with oriented x <= line is exploitable as-is;
    a cell beyond the line is exploitable iff SOME attacker who is currently onside
    (oriented x <= line) can reach it within TAU_RUN.

TWO HORIZONS, NOT ONE — the correction that made this gate meaningful. `tau` is the
DEFENSIVE horizon: how long the defense has to cover a point. `tau_run` is the
ATTACKING horizon for a run into space behind the line, and it is strictly longer,
because exploiting that space means a pass FLIGHT plus a run, and the offside law only
requires the runner to be onside when the ball is played — not to already be there.
Using a single tau (the first implementation) confined exploitable behind-line space to
a ~6.5m band and stripped 92.6% of final-third VNS, inverting the thirds ordering.
Measured, diagnosed, fixed rather than tuned away; `tau_run` is swept in
validate_phase2 alongside tau and theta (Gap E).

Simplifications, both stated: (1) the real law also lets a player level with the ball
be onside — ignored; current attacker positions proxy for positions at the pass moment.
This makes the mask slightly STRICTER than the law, so masked VNS errs conservative
(understates). (2) "can reach within tau_run" uses the same motion model as coverage, so
the reaction-time floor bias (reach.py) applies to attackers here too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import reach
from reach import ReachParams
from value_surface import ValueSurface


@dataclass
class VNSResult:
    """Value-weighted negative space at one frame, one (tau, theta)."""

    tau: float
    theta: float
    attacking_team: str
    raw_area_m2: float  # |N| unweighted, unmasked — Phase 1's non-discriminating number
    vns_unmasked: float  # value-weighted, ignoring offside (Gap B option B1)
    vns: float  # value-weighted, offside-masked — THE Phase 2 number
    behind_line_share: float  # fraction of vns_unmasked removed by the mask


def offside_line(def_pos: np.ndarray, direction: int) -> float:
    """Oriented-x of the offside line: second-last defender, floored at halfway.

    def_pos is (n, 2) metric; direction orients the ATTACKING team toward +x, so the
    defending team's goal is at +x and the deepest defenders have the largest oriented x
    (normally the GK is last, a centre-back second-last).
    """
    if len(def_pos) < 2:
        return 0.0
    ox = np.sort(def_pos[:, 0] * direction)
    return max(float(ox[-2]), 0.0)


def compute_vns(
    row: pd.Series,
    attacking_team: str,
    player_ids: dict[str, list[str]],
    direction: int,
    surface: ValueSurface,
    tau: float,
    theta: float,
    targets: np.ndarray,
    cell_area_m2: float,
    params: ReachParams | None = None,
    return_cells: bool = False,
    tau_run: float = 4.0,
) -> VNSResult | tuple[VNSResult, dict] | None:
    """VNS at one frame. Returns None if either team has no tracked players.

    `targets` from reach.pitch_grid — any resolution; validation uses 2m cells for
    speed (16x fewer cells than 1m at ~0.3% relative area error), renders use 1m.

    return_cells=True additionally returns the per-cell arrays (n_mask, exploitable,
    value, oriented x) so validation can slice VNS by pitch region without duplicating
    this code path.
    """
    params = params or ReachParams()
    defending_team = "away" if attacking_team == "home" else "home"

    def_pos, def_vel = reach.team_positions_velocities(row, defending_team, player_ids[defending_team])
    att_pos, att_vel = reach.team_positions_velocities(row, attacking_team, player_ids[attacking_team])
    if len(def_pos) == 0 or len(att_pos) == 0:
        return None

    # Negative space: defense's coverage below theta.
    cov = reach.team_coverage(reach.arrival_times(def_pos, def_vel, targets, params), tau, params)
    n_mask = cov < theta

    # Value per cell, oriented for the attacking team.
    v = surface.value_at_targets(targets, direction)

    # Offside mask.
    line = offside_line(def_pos, direction)
    tx = targets[:, 0] * direction  # oriented x of each cell
    beyond = tx > line

    onside = att_pos[:, 0] * direction <= line
    if onside.any():
        att_arr = reach.arrival_times(att_pos[onside], att_vel[onside], targets, params)
        beyond_reachable = att_arr.min(axis=1) <= tau_run
    else:
        beyond_reachable = np.zeros(len(targets), dtype=bool)
    exploitable = ~beyond | beyond_reachable

    vns_unmasked = float(v[n_mask].sum()) * cell_area_m2
    vns_masked = float(v[n_mask & exploitable].sum()) * cell_area_m2
    result = VNSResult(
        tau=tau,
        theta=theta,
        attacking_team=attacking_team,
        raw_area_m2=float(n_mask.sum()) * cell_area_m2,
        vns_unmasked=vns_unmasked,
        vns=vns_masked,
        behind_line_share=0.0 if vns_unmasked == 0 else 1.0 - vns_masked / vns_unmasked,
    )
    if return_cells:
        return result, {"n_mask": n_mask, "exploitable": exploitable, "value": v, "oriented_x": tx}
    return result
