"""Phase 4 — per-player and per-team space metrics, aggregated over a full match.

Turns Phase 3's per-window attributions into rates. Five per-player quantities:

  DEFENSIVE (accrue while the player's team is defending)
    space_conceded   xT·m² of valuable space the player's own movement vacated
    space_recovered  xT·m² they re-covered
  ATTACKING (accrue while their team is attacking)
    space_created    xT·m² credited via the marking heuristic (they pulled a defender)
    entries          count of decisive, material moves into uncontested space
    entry_value      xT at the points entered

EXPOSURE, and why it is not just "minutes played". A defensive rate must be normalised
by time the player's team spent DEFENDING, not by total time on the pitch. Otherwise a
team that dominates possession looks defensively excellent purely for having defended
less — the denominator would encode possession share rather than defending quality. So
each player carries two clocks, `def_minutes` and `att_minutes`, and each metric is
divided by the one that actually exposed them to the opportunity.

WINDOWS TILE, they do not overlap. Time is partitioned into consecutive δ-length windows
so each moment is counted exactly once; overlapping windows would multiply-count the same
space change and inflate every rate by the overlap factor.

GOALKEEPERS are flagged, not dropped (`is_gk`). Phase 3 measured them taking 38–41% of
creation credit because they guard the highest-value ground — real, but not comparable to
outfield players, so they must be reported separately rather than ranked alongside.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

import dynamics as dyn
import negative_space as ns
import reach
import vns as vns_mod
from orientation import goalkeeper_ids
from reach import ReachParams
from value_surface import ValueSurface


# Oriented zone grid for exposure-adjusting creation credit (see `created_relative`).
# Deliberately coarse: with 2 games a finer grid leaves too few player-minutes per zone to
# estimate a league rate, the same sparsity trap the sibling project hit with a 12x8 xT
# grid on 2 Metrica games (~20 actions/zone) before moving to 16x12 at season scale.
ZONE_NX, ZONE_NY = 6, 4
PITCH_L, PITCH_W = 105.0, 68.0


def zone_index(x_oriented, y_oriented) -> np.ndarray:
    """Zone id in [0, ZONE_NX*ZONE_NY) from ORIENTED metric coords (attack toward +x)."""
    xf = np.clip((np.asarray(x_oriented) / PITCH_L + 0.5), 0, 1 - 1e-9)
    yf = np.clip((np.asarray(y_oriented) / PITCH_W + 0.5), 0, 1 - 1e-9)
    return (yf * ZONE_NY).astype(int) * ZONE_NX + (xf * ZONE_NX).astype(int)


@dataclass
class PlayerMetrics:
    team: str
    pid: str
    is_gk: bool = False
    def_minutes: float = 0.0
    att_minutes: float = 0.0
    space_conceded: float = 0.0
    space_recovered: float = 0.0
    space_created: float = 0.0
    space_created_area: float = 0.0  # bare m², no xT weighting — position-neutral
    entries: int = 0
    entry_value: float = 0.0
    # Per-zone attacking exposure and creation credit, for the position-confound fix.
    zone_att_minutes: np.ndarray = field(default_factory=lambda: np.zeros(ZONE_NX * ZONE_NY))
    zone_created: np.ndarray = field(default_factory=lambda: np.zeros(ZONE_NX * ZONE_NY))

    def _per90(self, total: float, minutes: float) -> float:
        return float("nan") if minutes <= 0 else total / minutes * 90.0

    @property
    def conceded_p90(self) -> float:
        return self._per90(self.space_conceded, self.def_minutes)

    @property
    def recovered_p90(self) -> float:
        return self._per90(self.space_recovered, self.def_minutes)

    @property
    def net_defensive_p90(self) -> float:
        """Recovered minus conceded. Positive = re-covers more valuable space than it gives up."""
        return self.recovered_p90 - self.conceded_p90

    @property
    def created_p90(self) -> float:
        return self._per90(self.space_created, self.att_minutes)

    @property
    def created_area_p90(self) -> float:
        """Space created per 90 in bare m², with NO value weighting.

        The position-neutral half of creation. `created_p90` multiplies by xT, and xT is a
        function of location spanning three orders of magnitude, so the value-weighted
        version necessarily imports position. Reporting both separates "how much space do
        you open" from "how valuable is the ground you open it on".
        """
        return self._per90(self.space_created_area, self.att_minutes)

    @property
    def entries_p90(self) -> float:
        return self._per90(self.entries, self.att_minutes)

    @property
    def entry_value_p90(self) -> float:
        return self._per90(self.entry_value, self.att_minutes)


@dataclass
class MatchMetrics:
    game_id: int
    players: dict[tuple[str, str], PlayerMetrics] = field(default_factory=dict)
    team_vns_conceded: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    n_windows: int = 0

    def team_vns_conceded_mean(self) -> dict[str, float]:
        """Mean valuable negative space a team allowed while defending (xT·m²)."""
        return {t: float(np.mean(v)) for t, v in self.team_vns_conceded.items() if v}


def match_metrics(
    game,
    surface: ValueSurface,
    params: ReachParams | None = None,
    tau: float = 2.0,
    theta: float = 0.3,
    delta_s: float = 2.0,
    cell_m: float = 2.0,
    period: int | None = None,
    max_windows: int | None = None,
    include_team_vns: bool = True,
) -> MatchMetrics:
    """Aggregate space metrics over one match (or one period of it).

    `period` restricts to a single half — used by W5's split-half stability check.
    `max_windows` subsamples for the knob sweep, where absolute rates do not matter and
    only the RANKING is compared; leave None for real rates.
    """
    params = params or ReachParams()
    gx, gy, targets = reach.pitch_grid(cell_m=cell_m)
    grid_shape, cell_area = gx.shape, cell_m**2

    poss = ns.possession_by_frame(game.events, game.tracking)
    alive = ns.analysable_frames(game.events, game.tracking)
    gks = goalkeeper_ids(game.tracking, game.player_ids)
    periods = game.tracking["Period"].to_numpy()
    step = int(round(delta_s / game.dt))

    out = MatchMetrics(game_id=game.game_id)

    def slot(team: str, pid: str) -> PlayerMetrics:
        key = (team, pid)
        if key not in out.players:
            out.players[key] = PlayerMetrics(team=team, pid=pid, is_gk=(gks.get(team) == pid))
        return out.players[key]

    # --- exposure clocks --------------------------------------------------------------
    # Counted per FRAME (not per window) so a player subbed mid-window is charged only for
    # the time they were actually on the pitch. Vectorised over frames — the scalar loop
    # was 3.2M Python iterations per match and dominated runtime.
    minutes_per_frame = game.dt / 60.0
    frame_ok = alive.copy()
    if period is not None:
        frame_ok &= periods == period
    poss_arr = np.array([p if p is not None else "" for p in poss], dtype=object)
    frame_ok &= poss_arr != ""

    for team in ("home", "away"):
        # A team's attacking direction depends only on the period, not on who has the ball.
        team_dir = np.array(
            [game.direction(team, int(p)) for p in periods], dtype=float
        )
        is_attacking = frame_ok & (poss_arr == team)
        is_defending = frame_ok & (poss_arr != team) & (poss_arr != "")
        for pid in game.player_ids[team]:
            x = game.tracking[f"{team}_{pid}_x"].to_numpy(float)
            y = game.tracking[f"{team}_{pid}_y"].to_numpy(float)
            on = ~np.isnan(x)
            att_mask = is_attacking & on
            def_mask = is_defending & on
            if not (att_mask.any() or def_mask.any()):
                continue
            p = slot(team, pid)
            p.att_minutes += float(att_mask.sum()) * minutes_per_frame
            p.def_minutes += float(def_mask.sum()) * minutes_per_frame
            if att_mask.any():
                z = zone_index(x[att_mask] * team_dir[att_mask], y[att_mask] * team_dir[att_mask])
                p.zone_att_minutes += np.bincount(z, minlength=ZONE_NX * ZONE_NY) * minutes_per_frame

    # --- non-overlapping windows ------------------------------------------------------
    starts = np.arange(0, len(game.tracking) - step, step)
    if period is not None:
        starts = starts[periods[starts] == period]
    if max_windows is not None and len(starts) > max_windows:
        starts = starts[np.linspace(0, len(starts) - 1, max_windows).astype(int)]

    vcache: dict[int, np.ndarray] = {}
    for i in starts:
        j = i + step
        if not (alive[i] and alive[j]):
            continue
        if poss[i] is None or poss[i] != poss[j] or periods[i] != periods[j]:
            continue
        row0, row1 = game.tracking.iloc[i], game.tracking.iloc[j]
        att = poss[i]
        dfn = game.defending_team(att)
        d = game.direction(att, int(periods[i]))
        if d not in vcache:
            vcache[d] = surface.value_at_targets(targets, d)

        r = dyn.frame_dynamics(
            row0, row1, att, game.player_ids, d, tau, theta, targets, vcache[d],
            cell_area, delta_s, params, grid_shape=grid_shape, gk_id=gks.get(dfn),
        )
        if r is None:
            continue
        out.n_windows += 1

        for pid, v in r.created_by_defender.items():
            slot(dfn, pid).space_conceded += v
        for pid, v in r.destroyed_by_defender.items():
            slot(dfn, pid).space_recovered += v
        for pid, v in r.credited_area_to_attacker.items():
            slot(att, pid).space_created_area += v
        for pid, v in r.credited_to_attacker.items():
            p = slot(att, pid)
            p.space_created += v
            # Record WHERE the crediting happened, so creation can later be compared
            # against what a typical player produces from the same ground.
            px, py = row0.get(f"{att}_{pid}_x"), row0.get(f"{att}_{pid}_y")
            if not (px is None or np.isnan(px) or np.isnan(py)):
                p.zone_created[int(zone_index(px * d, py * d))] += v
        for pid in r.entries:
            p = slot(att, pid)
            p.entries += 1
            p.entry_value += r.entry_value.get(pid, 0.0)

        if include_team_vns:
            res = vns_mod.compute_vns(
                row0, att, game.player_ids, d, surface, tau, theta, targets, cell_area, params
            )
            if res is not None:
                out.team_vns_conceded[dfn].append(res.vns)

    return out


def created_relative(
    matches: list["MatchMetrics"], min_minutes: float = 20.0, outfield_only: bool = True
) -> dict[tuple[str, str], float]:
    """Position-adjusted creation: observed credit ÷ credit expected from where they played.

    THE PROBLEM THIS FIXES. Raw `created_p90` correlates **+0.956** with a player's mean
    oriented pitch position — so nearly collinear that it is closer to a relabel of "how
    far forward you play" than a measure of creation skill. The mechanism is not subtle:
    an advanced player is marked by defenders standing on high-xT ground, so when those
    defenders move, high-value cells flip and the credit is large. A deep player's marker
    guards worthless ground and can never generate much credit however well they are
    dragged around. Same failure the sibling project hit with `cpv_avg` (zone-confounding
    0.387), fixed there by two-stage normalisation into `cpv_zz` (→ 0.000).

    THE FIX — exposure-adjusted, not residualised. For each zone z, the league rate is

        rate(z) = total creation credit in z / total attacking player-minutes in z

    and a player's expectation is `Σ_z minutes_in_z(player) · rate(z)`. The metric is
    `observed / expected`: 1.0 means exactly what a typical player produces from that
    player's own distribution of positions, >1 means more.

    Chosen over regressing the metric on mean_x and taking residuals, which would assume
    the position→value relationship is LINEAR. It is not: xT spans three orders of
    magnitude across the pitch (0.0016 → 0.397 by column), so a linear control would
    under-correct the final third and over-correct midfield. Exposure adjustment makes no
    functional-form assumption at all — it compares like ground with like ground.

    CAVEATS, both real. (1) The league baseline here is two matches and four teams, so
    `rate(z)` is thinly estimated; the coarse 6x4 grid is a deliberate concession to that.
    (2) The adjustment removes positional ADVANTAGE, which is the point, but a player
    whose value genuinely IS getting into good positions will now score lower. That is a
    definitional choice: this metric answers "how well do they create relative to peers on
    the same ground", not "how much do they create". Report alongside raw, never instead.
    """
    agg: dict[tuple[str, str], PlayerMetrics] = {}
    for m in matches:
        for key, p in m.players.items():
            if key not in agg:
                agg[key] = PlayerMetrics(team=p.team, pid=p.pid, is_gk=p.is_gk)
            a = agg[key]
            a.att_minutes += p.att_minutes
            a.space_created += p.space_created
            a.zone_att_minutes = a.zone_att_minutes + p.zone_att_minutes
            a.zone_created = a.zone_created + p.zone_created

    pool = [p for p in agg.values() if not (outfield_only and p.is_gk)]
    tot_credit = np.sum([p.zone_created for p in pool], axis=0)
    tot_minutes = np.sum([p.zone_att_minutes for p in pool], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(tot_minutes > 0, tot_credit / tot_minutes, 0.0)

    out: dict[tuple[str, str], float] = {}
    for p in pool:
        if p.att_minutes < min_minutes:
            continue
        expected = float(p.zone_att_minutes @ rate)
        if expected <= 0:
            continue
        out[(p.team, p.pid)] = float(p.space_created) / expected
    return out


def leaderboard(
    matches: list[MatchMetrics], metric: str, min_minutes: float = 20.0, outfield_only: bool = True
) -> list[tuple[str, str, float, float]]:
    """Pool matches and rank players by a per-90 metric.

    Returns (team, pid, value, minutes). Players below `min_minutes` of the RELEVANT
    exposure are dropped — a per-90 rate off a few minutes is noise, and this project has
    already been bitten by small-sample instability once (the sibling project's S2).
    """
    agg: dict[tuple[str, str], PlayerMetrics] = {}
    for m in matches:
        for key, p in m.players.items():
            if key not in agg:
                agg[key] = PlayerMetrics(team=p.team, pid=p.pid, is_gk=p.is_gk)
            a = agg[key]
            a.def_minutes += p.def_minutes
            a.att_minutes += p.att_minutes
            a.space_conceded += p.space_conceded
            a.space_recovered += p.space_recovered
            a.space_created += p.space_created
            a.space_created_area += p.space_created_area
            a.entries += p.entries
            a.entry_value += p.entry_value

    defensive = metric in ("conceded_p90", "recovered_p90", "net_defensive_p90")
    rows = []
    for p in agg.values():
        if outfield_only and p.is_gk:
            continue
        minutes = p.def_minutes if defensive else p.att_minutes
        if minutes < min_minutes:
            continue
        val = getattr(p, metric)
        if not np.isnan(val):
            rows.append((p.team, p.pid, float(val), float(minutes)))
    return sorted(rows, key=lambda r: -r[2])
