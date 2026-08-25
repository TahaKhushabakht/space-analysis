"""Phase 3 — the two verbs: space CREATED / DESTROYED, and space ENTERED.

Turns the static negative space of Phases 1–2 into change over time, and attributes that
change to individual players. This is the machinery behind the explorer's "change since
pinned t₀" layer, made accountable.

THE KEY IDEA — attribute in log-complement space, where the union is ADDITIVE.

Team coverage is a union, `C = 1 − Π_i (1 − p_i)`, which is NOT additive across
defenders, so "who caused this cell to open" has no exact answer in coverage space.
Leave-one-out attribution is the obvious approach and it does not sum back to the total
(two defenders can each be individually non-essential yet jointly responsible). But

    L := log(1 − C) = Σ_i log(1 − p_i)

IS additive. So the change decomposes exactly:

    ΔL = Σ_i ΔL_i ,      ΔL_i = log(1 − p_i(t+δ)) − log(1 − p_i(t))

and every cell's change is split among defenders with no residual. A cell counts as
negative space when `C < θ`, equivalently `L > log(1 − θ)` — the same threshold, moved
into the additive coordinate. Flipped cells are shared out in proportion to each
defender's positive (or negative) contribution to ΔL there.

This makes gate W4's accounting identity true BY CONSTRUCTION rather than by luck:
per-defender attributions sum to total ΔN exactly. It cannot fail unless the code is
wrong — which is exactly the class of check this project prefers (docs/math.md §6).
A cell that opened necessarily has ΔL > 0, hence at least one positive ΔL_i, so the
normalisation never divides by zero.

SCOPE HONESTY. Defender-side attribution ("whose reach vacated this ground") is
mechanical and safe. ATTACKER-side creation credit ("who dragged that defender out") is a
causal claim about intent, implemented here as the transparent nearest-marker heuristic
(validation.md Gap A, option A1). It is wrong in zonal schemes and says so. Everything
else in this module — destruction, entry — never needs it, which is why the risky part
is isolated behind one function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

import reach
from reach import ReachParams
from vns import offside_line

# p is clipped off 1 so log(1 − p) stays finite; 1e-9 caps a single defender's
# contribution at ~ −20.7 nats, far beyond any threshold that matters.
_P_CLIP = 1.0 - 1e-9


@dataclass
class FrameDynamics:
    """Change in negative space between two frames, attributed to players."""

    dt_s: float
    opened_m2: float
    closed_m2: float
    opened_xt: float  # value-weighted (xT·m²)
    closed_xt: float
    created_by_defender: dict[str, float]  # defender id -> xT·m² of space they vacated
    destroyed_by_defender: dict[str, float]  # defender id -> xT·m² they re-covered
    credited_to_attacker: dict[str, float]  # via marking heuristic; see module docstring
    credited_area_to_attacker: dict[str, float] = field(default_factory=dict)  # same, bare m²
    entries: list[str] = field(default_factory=list)  # attacker ids entering negative space
    entry_value: dict[str, float] = field(default_factory=dict)  # xT at the entered point
    # --- region structure (Phase 3 fix 1) ---------------------------------------------
    # Scattered flipped cells and one coherent channel can carry identical total area.
    # These let events be ranked by the largest CONNECTED piece, which is what "a space
    # opened up" actually means. Computed only when grid_shape is supplied.
    largest_opened_xt: float = 0.0
    largest_closed_xt: float = 0.0
    n_opened_regions: int = 0
    coherence: float = 0.0  # largest opened region's share of opened cells, 0..1
    # --- goalkeeper flag (Phase 3 fix 2) ----------------------------------------------
    # The keeper stays in created_by_defender so W4(a)'s identity remains exhaustive;
    # this field lets reporting split them out instead of ranking them against outfielders.
    gk_id: str | None = None


def _team_pv_ids(row: pd.Series, team: str, ids: list[str]):
    """Positions, velocities and the ids of players actually tracked in this frame."""
    pos, vel, present = [], [], []
    for pid in ids:
        x, y = row.get(f"{team}_{pid}_x"), row.get(f"{team}_{pid}_y")
        if pd.isna(x) or pd.isna(y):
            continue
        vx, vy = row.get(f"{team}_{pid}_vx", 0.0), row.get(f"{team}_{pid}_vy", 0.0)
        pos.append((x, y))
        vel.append((0.0 if pd.isna(vx) else vx, 0.0 if pd.isna(vy) else vy))
        present.append(pid)
    return np.array(pos, dtype=float), np.array(vel, dtype=float), present


def log_uncovered(arrivals: np.ndarray, tau: float, params: ReachParams) -> np.ndarray:
    """log(1 − p_i) per (cell, player) — the additive coordinate. Shape in = shape out."""
    p = np.clip(reach.coverage_probability(arrivals, tau, params), 0.0, _P_CLIP)
    return np.log1p(-p)


def nearest_marker(def_pos: np.ndarray, att_pos: np.ndarray, max_dist_m: float = 12.0):
    """Index of the attacker each defender is nearest to, or -1 beyond `max_dist_m`.

    The transparent version of Gap A: assumes man-orientation, which is wrong in a zonal
    block and in transition. `max_dist_m` exists so a defender with nobody near them is
    credited to nobody rather than to whoever happens to be least far away.
    """
    if len(att_pos) == 0:
        return np.full(len(def_pos), -1, dtype=int)
    d = np.linalg.norm(def_pos[:, None, :] - att_pos[None, :, :], axis=-1)
    idx = d.argmin(axis=1)
    return np.where(d[np.arange(len(def_pos)), idx] <= max_dist_m, idx, -1)


def frame_dynamics(
    row0: pd.Series,
    row1: pd.Series,
    attacking_team: str,
    player_ids: dict[str, list[str]],
    direction: int,
    tau: float,
    theta: float,
    targets: np.ndarray,
    value: np.ndarray,
    cell_area_m2: float,
    dt_s: float,
    params: ReachParams | None = None,
    mark_dist_m: float = 12.0,
    grid_shape: tuple[int, int] | None = None,
    gk_id: str | None = None,
    theta_entry: float | None = None,
    min_entry_xt: float = 0.01,
) -> FrameDynamics | None:
    """Attribute the change in negative space between two frames of one possession.

    `value` is the per-cell xT already oriented for `attacking_team` (shape = len(targets)).
    Only defenders tracked in BOTH frames participate — a substitution appearing mid-window
    would otherwise register as a player who "created" space by materialising.
    """
    params = params or ReachParams()
    defending_team = "away" if attacking_team == "home" else "home"

    p0, v0, ids0 = _team_pv_ids(row0, defending_team, player_ids[defending_team])
    p1, v1, ids1 = _team_pv_ids(row1, defending_team, player_ids[defending_team])
    common = [i for i in ids0 if i in set(ids1)]
    if len(common) < 2:
        return None
    k0 = [ids0.index(i) for i in common]
    k1 = [ids1.index(i) for i in common]

    l0 = log_uncovered(reach.arrival_times(p0[k0], v0[k0], targets, params), tau, params)
    l1 = log_uncovered(reach.arrival_times(p1[k1], v1[k1], targets, params), tau, params)

    thr = np.log(1.0 - theta)  # C < theta  <=>  L > log(1 - theta)
    L0, L1 = l0.sum(axis=1), l1.sum(axis=1)
    open0, open1 = L0 > thr, L1 > thr
    opened, closed = (~open0) & open1, open0 & (~open1)

    dL = l1 - l0  # (cells, defenders)
    w = value * cell_area_m2  # xT-weighted cell area

    created = {pid: 0.0 for pid in common}
    destroyed = {pid: 0.0 for pid in common}
    # Parallel UNWEIGHTED accumulation (bare m², no xT). The value-weighted number is the
    # meaningful one, but it is also inherently positional — xT is a function of location,
    # so any value-weighted per-player metric imports position. Carrying both lets the two
    # be separated instead of confounded. See metrics.created_relative.
    created_area = {pid: 0.0 for pid in common}

    if opened.any():
        pos = np.maximum(dL[opened], 0.0)
        share = pos / pos.sum(axis=1, keepdims=True)  # rows sum to 1 exactly
        contrib = share * w[opened, None]
        area_contrib = share * cell_area_m2
        for j, pid in enumerate(common):
            created[pid] = float(contrib[:, j].sum())
            created_area[pid] = float(area_contrib[:, j].sum())
    if closed.any():
        neg = np.maximum(-dL[closed], 0.0)
        share = neg / neg.sum(axis=1, keepdims=True)
        contrib = share * w[closed, None]
        for j, pid in enumerate(common):
            destroyed[pid] = float(contrib[:, j].sum())

    # --- attacker-side credit (the isolated causal guess) ------------------------------
    a0, av0, aids0 = _team_pv_ids(row0, attacking_team, player_ids[attacking_team])
    credited: dict[str, float] = {}
    credited_area: dict[str, float] = {}
    if len(a0):
        marks = nearest_marker(p0[k0], a0, mark_dist_m)
        for j, pid in enumerate(common):
            m = marks[j]
            if m >= 0 and created[pid] > 0:
                credited[aids0[m]] = credited.get(aids0[m], 0.0) + created[pid]
                credited_area[aids0[m]] = credited_area.get(aids0[m], 0.0) + created_area[pid]

    # --- entries: an attacker moving into space the defense cannot contest -------------
    # TWO conditions beyond "is uncontested", both added after Phase 3's first run showed
    # the naive definition fires constantly on nothing (measured: 3.7 of 11 attackers are
    # standing in negative space at any instant, and 97.7% of those moments carry
    # xT < 0.02). Being unmarked is the normal state, not an event.
    #   1. DECISIVE crossing: coverage must fall below `theta_entry` (stricter than theta)
    #      having been at/above theta. A single threshold lets a player hovering at the
    #      boundary chatter in and out, inflating counts with non-events.
    #   2. MATERIALITY: the entered point must carry at least `min_entry_xt`, so entering
    #      worthless space is not an entry. Both are stated knobs, sweepable like tau/theta.
    #
    # CALIBRATED, not guessed — the first attempt (theta_entry=0.15, min_entry_xt=0.02)
    # was too strict and produced ZERO entries. Measured entries per game (game 1,
    # 666 sampled 2s windows scaled to full match):
    #     theta_entry \ min_xt   0.000   0.005   0.010   0.020
    #                    0.30     1716     792     331      48
    #                    0.20      806     327    [118]     13
    #                    0.15      488     131      48       0
    # Defaults land on ~118/game, matching the rough football expectation for genuinely
    # dangerous off-ball runs by both teams combined (tens, not hundreds or single digits).
    # A materiality floor of 0.01 xT is roughly the central channel at ~33m — the edge of
    # where possession starts being worth something.
    entries, entry_value = [], {}
    theta_entry = theta * (2.0 / 3.0) if theta_entry is None else theta_entry
    a1, av1, aids1 = _team_pv_ids(row1, attacking_team, player_ids[attacking_team])
    if len(a0) and len(a1):
        line = offside_line(p1[k1], direction)
        c_at = lambda pts, pp, vv: 1.0 - np.exp(
            log_uncovered(reach.arrival_times(pp, vv, pts, params), tau, params).sum(axis=1)
        )
        cov0 = c_at(a0, p0[k0], v0[k0])
        cov1 = c_at(a1, p1[k1], v1[k1])
        was_in = {aids0[i]: cov0[i] < theta for i in range(len(aids0))}
        for i, pid in enumerate(aids1):
            onside = a1[i, 0] * direction <= line  # offside players are not "in usable space"
            if not (cov1[i] < theta_entry and onside and not was_in.get(pid, False)):
                continue
            gx = np.argmin(np.linalg.norm(targets - a1[i], axis=1))
            v_here = float(value[gx])
            if v_here < min_entry_xt:
                continue
            entries.append(pid)
            entry_value[pid] = v_here

    # --- connected-region structure ---------------------------------------------------
    # Deliberately NOT fed back into the attribution: W4(a)'s identity is stated over the
    # FULL flipped set, so filtering cells here would break the very theorem the log-space
    # decomposition was chosen to guarantee. These are additional descriptors for ranking
    # and for the eye test, computed alongside.
    largest_open = largest_close = 0.0
    n_regions = 0
    coherence = 0.0
    if grid_shape is not None and (opened.any() or closed.any()):
        for mask, sink in ((opened, "open"), (closed, "close")):
            if not mask.any():
                continue
            lab, n = ndimage.label(mask.reshape(grid_shape))
            masses = ndimage.sum(w.reshape(grid_shape), lab, index=range(1, n + 1))
            top = float(np.max(masses)) if n else 0.0
            if sink == "open":
                largest_open, n_regions = top, n
                sizes = ndimage.sum(mask.reshape(grid_shape).astype(float), lab, index=range(1, n + 1))
                coherence = float(np.max(sizes) / mask.sum()) if mask.sum() else 0.0
            else:
                largest_close = top

    return FrameDynamics(
        dt_s=dt_s,
        opened_m2=float(opened.sum()) * cell_area_m2,
        closed_m2=float(closed.sum()) * cell_area_m2,
        opened_xt=float(w[opened].sum()),
        closed_xt=float(w[closed].sum()),
        created_by_defender=created,
        destroyed_by_defender=destroyed,
        credited_to_attacker=credited,
        credited_area_to_attacker=credited_area,
        entries=entries,
        entry_value=entry_value,
        largest_opened_xt=largest_open,
        largest_closed_xt=largest_close,
        n_opened_regions=n_regions,
        coherence=coherence,
        gk_id=gk_id,
    )
