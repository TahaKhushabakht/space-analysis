"""Negative space: the region the defense cannot reliably reach.

From docs/concept.md, defined from the DEFENSE's point of view:

    C_D(t, tau) = union over defenders p of R_p(t, tau)     (covered volume)
    N(t, tau)   = F \\ C_D(t, tau)                           (negative space)

with the probabilistic reading preferred: each point carries P(some defender reaches it
within tau), and negative space is where that probability falls below a threshold theta.

TWO KNOBS, STATED NOT HIDDEN. `tau` (how much time the defense gets) and `theta` (how
confident "reliably reach" means) are modelling choices, and every number this module
produces is conditional on them: "space no defender reaches in 2s with 80% confidence".
Gate W5/Gap E in docs/validation.md requires headline metrics to be rank-stable across a
grid of both before any claim is made. Callers must pass them explicitly — there are
deliberately NO default values for tau and theta in the public functions.

WHOSE negative space: N is always computed against a specific DEFENDING team, so
possession determines roles. `possession_by_frame` resolves that from events. The
goalkeeper is included in the defensive union — they are a defender for reachability
purposes regardless of how they are treated elsewhere in the workspace (the surplus
project excludes GKs from its ROLE space, a different question entirely).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import reach
from reach import ReachParams

# Events that establish which team is in possession, forward-filled between them.
# CHALLENGE is excluded deliberately: it is a contested moment where possession is
# precisely what is undecided, so letting it set possession would flip roles mid-duel.
POSSESSION_TYPES = ("PASS", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST")


@dataclass
class NegativeSpace:
    """Negative space at one frame, for one defending team, at one (tau, theta)."""

    tau: float
    theta: float
    defending_team: str
    coverage: np.ndarray  # (ny, nx) P(some defender reaches within tau)
    mask: np.ndarray  # (ny, nx) bool, True where coverage < theta
    cell_area_m2: float

    @property
    def area_m2(self) -> float:
        """Total uncovered area. Raw area only — value weighting arrives in Phase 2."""
        return float(self.mask.sum()) * self.cell_area_m2

    @property
    def fraction(self) -> float:
        """Uncovered share of the pitch, in [0, 1]."""
        return float(self.mask.mean())


def defensive_coverage(
    row: pd.Series,
    defending_team: str,
    player_ids: dict[str, list[str]],
    tau: float,
    targets: np.ndarray,
    params: ReachParams | None = None,
    deterministic: bool = False,
) -> np.ndarray:
    """P(some defender reaches each target within tau). Returns (n_targets,).

    `deterministic=True` returns the hard-edged variant (any defender's expected arrival
    within tau) as a 0/1 float — the cross-check that is free of the independence
    assumption documented in reach.team_coverage.
    """
    pos, vel = reach.team_positions_velocities(row, defending_team, player_ids[defending_team])
    if len(pos) == 0:
        return np.zeros(len(targets))
    arrivals = reach.arrival_times(pos, vel, targets, params)
    if deterministic:
        return reach.covered_deterministic(arrivals, tau).astype(float)
    return reach.team_coverage(arrivals, tau, params)


def negative_space(
    row: pd.Series,
    defending_team: str,
    player_ids: dict[str, list[str]],
    tau: float,
    theta: float,
    grid_shape: tuple[int, int],
    targets: np.ndarray,
    cell_area_m2: float,
    params: ReachParams | None = None,
    deterministic: bool = False,
) -> NegativeSpace:
    """Negative space at one frame. tau and theta are required — see module docstring.

    grid_shape is (ny, nx) matching `targets` as produced by reach.pitch_grid.
    """
    cov = defensive_coverage(
        row, defending_team, player_ids, tau, targets, params, deterministic
    ).reshape(grid_shape)
    return NegativeSpace(
        tau=tau,
        theta=theta,
        defending_team=defending_team,
        coverage=cov,
        mask=cov < theta,
        cell_area_m2=cell_area_m2,
    )


def possession_by_frame(events: pd.DataFrame, tracking: pd.DataFrame) -> np.ndarray:
    """Team in possession at each tracking frame, as lowercase "home"/"away" (or None).

    Forward-fills the team of the most recent possession-establishing event. Frames before
    the first such event get None.
    """
    ev = events.dropna(subset=["Start Frame"])
    ev = ev[ev["Type"].isin(POSSESSION_TYPES)].sort_values("Start Frame")
    if ev.empty:
        return np.full(len(tracking), None, dtype=object)

    ev_frames = ev["Start Frame"].to_numpy(dtype=float)
    ev_teams = ev["Team"].astype("string").str.lower().to_numpy()

    idx = np.searchsorted(ev_frames, tracking["Frame"].to_numpy(dtype=float), side="right") - 1
    out = np.full(len(tracking), None, dtype=object)
    valid = idx >= 0
    out[valid] = ev_teams[idx[valid]]
    return out


def _dead_intervals(events: pd.DataFrame) -> list[tuple[float, float]]:
    """Frame intervals [stop, restart) during which the ball was out of play.

    DERIVED BACKWARDS FROM RESTARTS, deliberately. The obvious approach — start dead time
    at a BALL OUT and end it at the next SET PIECE — silently misses every stoppage with
    another cause. Measured on the sample games: 77 SET PIECEs but only 51 BALL OUTs in
    game 1 (80 vs 49 in game 2), with the remaining restarts preceded by FAULT RECEIVED,
    CARD, CHALLENGE, BALL LOST or SHOT. That reconstruction put the ball in play 74-80% of
    the time, well above football's ~55-60%.

    A SET PIECE is an unambiguous marker that play HAD stopped, whatever stopped it, so
    each one is walked back to the end of the preceding event. Robust to the cause.

    End Frame is unreliable on some events (set pieces carry End Frame 0), so the stoppage
    moment falls back to the preceding event's Start Frame when its End Frame is not later.
    The half-time gap is handled for free: period 2's kick-off is a SET PIECE whose
    preceding event is the last action of period 1.
    """
    ev = events.dropna(subset=["Start Frame"]).sort_values("Start Frame").reset_index(drop=True)
    if ev.empty:
        return []

    intervals: list[tuple[float, float]] = []
    for i in ev.index[ev["Type"] == "SET PIECE"]:
        restart = float(ev.loc[i, "Start Frame"])
        if i == 0:
            intervals.append((0.0, restart))  # pre-kick-off
            continue
        prev = ev.loc[i - 1]
        prev_start = float(prev["Start Frame"])
        prev_end = float(prev["End Frame"]) if pd.notna(prev["End Frame"]) else prev_start
        stop = prev_end if prev_end > prev_start else prev_start
        if stop < restart:
            intervals.append((stop, restart))
    return intervals


def alive_by_frame(events: pd.DataFrame, tracking: pd.DataFrame) -> np.ndarray:
    """Boolean per tracking frame: was the ball in play?

    Dead time comes from `_dead_intervals` (see its docstring for why restarts, not
    stoppages, drive the reconstruction), plus the tail after the final event — tracking
    usually runs past the closing whistle.

    Ball-in-play typically lands near 55-60% of elapsed match time; a figure far from that
    means this reconstruction is wrong, and validate_phase1 gates on it.
    """
    frames = tracking["Frame"].to_numpy(dtype=float)
    alive = np.ones(len(tracking), dtype=bool)

    for lo, hi in _dead_intervals(events):
        alive &= ~((frames >= lo) & (frames < hi))

    ev = events.dropna(subset=["Start Frame"])
    if not ev.empty:
        last = max(float(ev["Start Frame"].max()), float(ev["End Frame"].max(skipna=True) or 0.0))
        alive &= frames <= last
    return alive


def analysable_frames(events: pd.DataFrame, tracking: pd.DataFrame) -> np.ndarray:
    """Boolean mask: ball in play AND possession resolved. The frames Phase 1+ works on."""
    poss = possession_by_frame(events, tracking)
    return alive_by_frame(events, tracking) & np.array([p is not None for p in poss])
