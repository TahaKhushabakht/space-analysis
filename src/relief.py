"""Relief field — the "space gravity" rendering of docs/idea-space-gravity.md.

    h(x, t) = v(x) · (1 − 2·C_D(x, t))

  h > 0  peak        valuable ground the defense cannot reach   (negative space that matters)
  h < 0  depression  valuable ground the defense owns
  h ≈ 0  flat        worthless ground, whoever holds it

The `(1 − 2C)` term maps coverage from [0,1] to [+1,−1], so the sign carries "who owns
it" and the magnitude carries "how much it matters". Multiplying by value is what
flattens the acreage Phase 1 showed dominates raw |N| (68–79% of the pitch open, nearly
all of it worthless).

STATUS: this is a RENDERING of quantities already validated in W1–W3, not a new claim.
It introduces no gate. The dynamical reading of gravity — ∇h as a force field that
predicts where players and the ball flow — is deliberately NOT implemented here; it is
a falsifiable claim fenced off in the idea doc until its own tests are run.

SIGN CONVENTION is the free choice recorded in the idea doc: open-and-valuable renders
as a peak. Flip both terms if the "valuable space is a well that draws the ball in"
metaphor ever communicates better; nothing downstream depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import reach
from reach import ReachParams
from value_surface import ValueSurface
from vns import offside_line


@dataclass
class ReliefFrame:
    """Per-frame fields, kept SEPARATE rather than pre-combined into h.

    The explorer needs them apart so the browser can toggle the offside mask and compute
    differences against a pinned reference frame client-side, without shipping a
    precomputed field for every (mask, t0, t) combination.
    """

    coverage: np.ndarray  # (ny, nx) P(some defender reaches within tau)
    exploitable: np.ndarray  # (ny, nx) bool — passes the offside mask
    att_xy: np.ndarray  # (n, 2) oriented attacking player positions
    def_xy: np.ndarray  # (n, 2) oriented defending player positions
    ball_xy: tuple[float, float] | None
    offside_x: float  # oriented x of the offside line


def relief(coverage: np.ndarray, value: np.ndarray) -> np.ndarray:
    """h = v · (1 − 2C). Elementwise; shapes must match."""
    return value * (1.0 - 2.0 * coverage)


def compute_frame(
    row: pd.Series,
    attacking_team: str,
    player_ids: dict[str, list[str]],
    direction: int,
    tau: float,
    targets: np.ndarray,
    grid_shape: tuple[int, int],
    params: ReachParams | None = None,
    tau_run: float = 4.0,
) -> ReliefFrame | None:
    """Coverage + offside mask + oriented positions for one frame.

    Returns fields in GRID orientation (row 0 = lowest y, matching reach.pitch_grid);
    the explorer builder handles attack-direction flipping and canvas row order.
    """
    params = params or ReachParams()
    defending_team = "away" if attacking_team == "home" else "home"

    def_pos, def_vel = reach.team_positions_velocities(row, defending_team, player_ids[defending_team])
    att_pos, att_vel = reach.team_positions_velocities(row, attacking_team, player_ids[attacking_team])
    if len(def_pos) == 0 or len(att_pos) == 0:
        return None

    cov = reach.team_coverage(reach.arrival_times(def_pos, def_vel, targets, params), tau, params)

    line = offside_line(def_pos, direction)
    tx = targets[:, 0] * direction
    beyond = tx > line
    onside = att_pos[:, 0] * direction <= line
    if onside.any():
        arr = reach.arrival_times(att_pos[onside], att_vel[onside], targets, params)
        reachable = arr.min(axis=1) <= tau_run
    else:
        reachable = np.zeros(len(targets), dtype=bool)
    exploitable = ~beyond | reachable

    bx, by = row.get("ball_x"), row.get("ball_y")
    ball = None if (pd.isna(bx) or pd.isna(by)) else (float(bx) * direction, float(by) * direction)

    return ReliefFrame(
        coverage=cov.reshape(grid_shape),
        exploitable=exploitable.reshape(grid_shape),
        att_xy=att_pos * direction,
        def_xy=def_pos * direction,
        ball_xy=ball,
        offside_x=float(line),
    )


def value_grid(surface: ValueSurface, targets: np.ndarray, direction: int, grid_shape) -> np.ndarray:
    """xT per grid cell, in grid orientation, for a team attacking `direction`."""
    return surface.value_at_targets(targets, direction).reshape(grid_shape)
