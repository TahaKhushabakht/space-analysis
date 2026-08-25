"""Attack-direction resolution for Metrica games.

ADAPTED (not copied) from surplus_value_model/src/expected_threat.py
`_attack_directions` / `orient_events`. Three deliberate differences:

1. **Keys are lowercase** ("home"/"away"), matching TRACKING column prefixes. Metrica
   events spell the team "Home"/"Away" while tracking columns are `home_11_x` — an
   adapter detail that would otherwise silently produce empty direction lookups.
2. **Provenance is tracked** (`from_shots`). Upstream fills a shot-less (team, period)
   by negating the team's other period. That fallback makes the "teams switch ends"
   invariant true BY CONSTRUCTION, so checking it there proves nothing. We record which
   keys came from real shot evidence so validation can report how many of the four
   (team, period) pairs were independently derived. A check satisfied by construction
   is not evidence.
3. **Metric-coordinate helpers**. Upstream orients normalized [0,1] event coords
   (x -> 1-x). Tracking here is metric and centre-origin, where the same 180-degree
   rotation is just (x, y) -> (-x, -y). Simpler, and it applies to velocities too.

WHY THIS MATTERS HERE: Metrica teams switch ends at half-time. Every value-weighted
quantity in this project (Phase 2 onward) asks "how much threat does this location carry
for the team attacking it", so the direction must be resolved BEFORE any xT lookup.
Getting it wrong flips the value surface end-for-end and quietly inverts the headline
metric rather than crashing.

CONVENTION: direction = +1 means the team attacks toward +x (normalized x=1); -1 means
they attack toward -x. `orient_xy` maps raw coords into that team's attacking frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class GKEvidence:
    """Independent direction evidence from goalkeeper position (see `gk_attack_directions`)."""

    direction: int
    gk_player: str
    gk_mean_x: float
    margin_m: float  # how much deeper the GK sits than the next-deepest teammate


@dataclass
class AttackDirections:
    """Attack direction per (team, period), plus which keys had real shot evidence."""

    directions: dict[tuple[str, int], int]
    from_shots: set[tuple[str, int]] = field(default_factory=set)

    def __call__(self, team: str, period: int) -> int:
        """Direction for a (team, period); defaults to +1 if unknown (upstream behaviour)."""
        return self.directions.get((str(team).lower(), int(period)), 1)

    @property
    def n_independent(self) -> int:
        """How many keys were derived from shots rather than filled by the switch-ends fallback."""
        return len(self.from_shots)


def resolve_attack_directions(events: pd.DataFrame) -> AttackDirections:
    """Infer, per (team, period), which goal that team attacks, from their shot locations.

    A team shooting on average from x > 0.5 (normalized event coords) is attacking the
    x=1 goal. A (team, period) with no shots falls back to the negation of that team's
    other period, then to +1 — same ladder as upstream, but the fallback keys are left
    out of `from_shots`.
    """
    ev_team = events["Team"].astype("string").str.lower()
    periods = sorted({int(p) for p in events["Period"].dropna().unique()})

    directions: dict[tuple[str, int], int] = {}
    from_shots: set[tuple[str, int]] = set()

    shots = events[events["Type"] == "SHOT"]
    if len(shots):
        shot_team = shots["Team"].astype("string").str.lower()
        for (team, period), grp in shots.groupby([shot_team, shots["Period"]]):
            key = (str(team), int(period))
            directions[key] = 1 if grp["Start X"].mean() > 0.5 else -1
            from_shots.add(key)

    for team in sorted(ev_team.dropna().unique()):
        for period in periods:
            key = (str(team), int(period))
            if key in directions:
                continue
            # Teams switch ends between the two halves.
            other = 2 if period == 1 else 1
            directions[key] = -directions.get((str(team), other), -1)

    return AttackDirections(directions=directions, from_shots=from_shots)


def gk_attack_directions(
    tracking: pd.DataFrame, player_ids: dict[str, list[str]]
) -> dict[tuple[str, int], GKEvidence]:
    """Derive attack direction per (team, period) from goalkeeper position instead of shots.

    INDEPENDENT of `resolve_attack_directions`: it reads tracking positions, not events,
    so agreement between the two is real corroboration rather than a restatement. A team
    defends the goal its keeper stands in front of, so it ATTACKS the opposite way:
    direction = -sign(gk_mean_x) in centre-origin metric coords.

    The keeper is identified as the team's most extreme mean-x player within the period —
    no position labels needed (Metrica is anonymized). `margin_m` reports how far clear of
    the next-deepest teammate they were; a small margin means the identification is shaky
    and any agreement it produces is correspondingly weak evidence. Upstream hit exactly
    this problem from the other side (surplus_value_model HISTORY §27/§29: a high defensive
    line pushes outfield players into keeper-like positions), so the margin is reported
    rather than assumed comfortable.
    """
    evidence: dict[tuple[str, int], GKEvidence] = {}
    for period, period_df in tracking.groupby("Period"):
        for team, ids in player_ids.items():
            means = {pid: period_df[f"{team}_{pid}_x"].mean() for pid in ids}
            means = {pid: mx for pid, mx in means.items() if not np.isnan(mx)}
            if not means:
                continue
            ranked = sorted(means.items(), key=lambda kv: abs(kv[1]), reverse=True)
            gk_pid, gk_mean_x = ranked[0]
            margin = abs(gk_mean_x) - abs(ranked[1][1]) if len(ranked) > 1 else float("nan")
            evidence[(team, int(period))] = GKEvidence(
                direction=int(-np.sign(gk_mean_x)),
                gk_player=gk_pid,
                gk_mean_x=float(gk_mean_x),
                margin_m=float(margin),
            )
    return evidence


def goalkeeper_ids(tracking: pd.DataFrame, player_ids: dict[str, list[str]]) -> dict[str, str | None]:
    """The goalkeeper's jersey id per team — the modal deepest player across periods.

    Reuses `gk_attack_directions`' identification (most extreme mean-x player), which
    Phase 0 measured as unambiguous on both sample games: keepers sat 22-26m clear of the
    next-deepest teammate, and all 8 (team, period) picks agreed with shot-derived
    attack directions.

    Needed because the keeper distorts CREDIT attribution in Phase 3 — they guard the
    highest-value ground, so any movement they make flips high-xT cells and they collect
    ~37% of all creation credit. The fix is to report them separately, NOT to drop them
    from the coverage union: the keeper genuinely covers the deep zone in front of goal
    (Phase 1's eye test confirmed exactly that), so excluding them would manufacture
    phantom negative space in the most valuable region on the pitch.
    """
    evidence = gk_attack_directions(tracking, player_ids)
    out: dict[str, str | None] = {}
    for team in player_ids:
        picks = [e.gk_player for (t, _), e in evidence.items() if t == team]
        out[team] = max(set(picks), key=picks.count) if picks else None
    return out


def orient_xy(x, y, direction: int):
    """Rotate metric centre-origin coords into the attacking frame (attack -> +x).

    In centre-origin coordinates a 180-degree rotation about the pitch centre is simply
    a sign flip on both axes, so this is `(x, y) * direction`. Works elementwise on
    scalars or arrays, and applies unchanged to VELOCITY components (a rotated frame
    rotates velocities the same way).
    """
    return x * direction, y * direction
