"""Synthetic tracking with SCRIPTED causality — ground truth for attribution.

WHY THIS EXISTS. Every metric in this project is validated against invariants, face
validity, or outcomes. None of that can test whether CREATION CREDIT points at the right
player, because real tracking never says who caused what. The `created_p90` confound was
only caught because position happened to be an obvious proxy (ρ = +0.956); a subtler
mis-attribution would pass every gate silently.

Here the causality is authored. A scenario states "attacker 10 runs, defender 4 follows
them, therefore attacker 10 opened the channel defender 4 vacated", the frames are
generated to match, and the real `dynamics.frame_dynamics` is then asked who did it. The
answer is checkable.

DESIGN CHOICE — feed the REAL pipeline, do not reimplement it. `build_scenario` emits a
tracking DataFrame with exactly the column layout `metrica_io` produces, and velocities
come from the same frozen `compute_velocities`. So the code under test is the production
path, including its velocity smoothing and its reaction-time motion model. A simulator
that reimplemented any of that would validate the reimplementation.

NO SYNTHETIC EVENTS. Possession and ball-in-play are supplied directly by the caller
rather than reconstructed from a fake event stream — the attribution code takes the
attacking team as an argument, so inventing events would add a second thing that could be
wrong without testing anything extra.

HONEST LIMIT. Scripted players move on smooth interpolated paths and mark perfectly or
not at all. Real defenders hedge, pass players on, and react late. So a heuristic that
scores well here is not thereby validated on real football — but one that scores BADLY
here is definitively broken, since this is the easy case. Read the results as an upper
bound on attribution quality, never as an estimate of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitch_control_ref import compute_velocities

DT = 0.04  # 25 Hz, matching Metrica so the pipeline sees a familiar frame rate


@dataclass
class Run:
    """One scripted movement: `player` travels to `to_xy` between t_start and t_end.

    `team` is REQUIRED in spirit even though it defaults: both teams use jersey ids
    "1".."11", so a run keyed on the number alone matches a player on each side. An early
    version did exactly that and silently moved the mirror-numbered defender too, which
    showed up as defender-side attribution scoring 62% where it should have been 100%.
    """

    player: str
    to_xy: tuple[float, float]
    t_start: float
    t_end: float
    team: str = "home"


@dataclass
class Scenario:
    """A scripted passage with a stated ground truth.

    `follows` encodes the marking relationship the SIMULATION obeys: {defender: attacker}.
    A defender in this map shadows their attacker's movement; one absent from it holds
    position (or moves only via its own scripted Run, e.g. a zonal slide).

    `true_creator` is the attacker whose movement is responsible for the space that opens
    — None when the correct answer is "nobody", which is the case the false-positive
    checks depend on.
    """

    name: str
    description: str
    home: dict[str, tuple[float, float]]  # attacking team, jersey -> start xy
    away: dict[str, tuple[float, float]]  # defending team
    runs: list[Run] = field(default_factory=list)
    follows: dict[str, str] = field(default_factory=dict)
    ball_xy: tuple[float, float] = (0.0, 0.0)
    duration_s: float = 3.0
    true_creator: str | None = None
    true_vacater: str | None = None  # defender expected to be charged with the space
    true_enterer: str | None = None


def _smoothstep(f: np.ndarray) -> np.ndarray:
    """Ease-in-out on [0,1]. Gives continuous velocity at both ends of a run.

    A linear ramp would start and stop instantaneously, producing velocity step changes
    that the 7-frame smoothing window then smears — an artefact of the script rather than
    of football.
    """
    f = np.clip(f, 0.0, 1.0)
    return f * f * (3.0 - 2.0 * f)


def _path(start: tuple[float, float], end: tuple[float, float], t: np.ndarray,
          t0: float, t1: float) -> np.ndarray:
    """(n_frames, 2) positions moving start->end over [t0, t1], holding either side."""
    f = _smoothstep((t - t0) / max(t1 - t0, 1e-9))
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    return s[None, :] + (e - s)[None, :] * f[:, None]


def build_scenario(
    sc: Scenario, jitter_m: float = 0.0, rng: np.random.Generator | None = None
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Render a scenario to a tracking DataFrame + player_ids, ready for the real pipeline.

    `jitter_m` perturbs every starting position, so a scenario can be run many times and
    the attribution scored over configurations rather than over one lucky geometry.
    """
    rng = rng or np.random.default_rng(0)
    n = int(round(sc.duration_s / DT))
    t = np.arange(n) * DT

    def jit(xy):
        if jitter_m <= 0:
            return tuple(xy)
        return (xy[0] + rng.normal(0, jitter_m), xy[1] + rng.normal(0, jitter_m))

    home = {k: jit(v) for k, v in sc.home.items()}
    away = {k: jit(v) for k, v in sc.away.items()}
    # Keyed by TEAM as well as jersey — see the note in Run about the id collision.
    home_runs = {r.player: r for r in sc.runs if r.team == "home"}
    away_runs = {r.player: r for r in sc.runs if r.team == "away"}

    pos: dict[str, np.ndarray] = {}
    for pid, start in home.items():
        if pid in home_runs:
            r = home_runs[pid]
            pos[f"home_{pid}"] = _path(start, r.to_xy, t, r.t_start, r.t_end)
        else:
            pos[f"home_{pid}"] = np.repeat(np.asarray(start, float)[None, :], n, axis=0)

    for pid, start in away.items():
        if pid in away_runs:  # a defender with its own script (e.g. a zonal slide)
            r = away_runs[pid]
            pos[f"away_{pid}"] = _path(start, r.to_xy, t, r.t_start, r.t_end)
        elif pid in sc.follows:
            # Shadow the marked attacker: hold the initial offset for the whole run.
            marked = sc.follows[pid]
            att = pos[f"home_{marked}"]
            offset = np.asarray(away[pid], float) - att[0]
            pos[f"away_{pid}"] = att + offset[None, :]
        else:
            pos[f"away_{pid}"] = np.repeat(np.asarray(away[pid], float)[None, :], n, axis=0)

    data: dict[str, np.ndarray] = {
        "Period": np.ones(n, dtype=int),
        "Frame": np.arange(1, n + 1),
        "Time [s]": t,
    }
    for key, arr in pos.items():
        data[f"{key}_x"] = arr[:, 0]
        data[f"{key}_y"] = arr[:, 1]
    data["ball_x"] = np.full(n, sc.ball_xy[0])
    data["ball_y"] = np.full(n, sc.ball_xy[1])

    tracking = pd.DataFrame(data)
    player_ids = {"home": list(home), "away": list(away)}
    return compute_velocities(tracking, player_ids, dt=DT), player_ids


# ---------------------------------------------------------------------------------------
# Scenario library. Attack is toward +x throughout; "home" attacks, "away" defends.
# Geometry: away back line ~x=+28, away keeper x=+50, home front line ~x=+18.
# ---------------------------------------------------------------------------------------

def _base_shape() -> tuple[dict, dict]:
    """A plausible 11v11 shape: away defending a mid-block, home attacking into it."""
    away = {
        "1": (50.0, 0.0),  # keeper
        "2": (28.0, -20.0), "3": (28.0, -7.0), "4": (28.0, 7.0), "5": (28.0, 20.0),
        "6": (16.0, -12.0), "7": (16.0, 0.0), "8": (16.0, 12.0),
        "9": (4.0, -8.0), "10": (4.0, 8.0), "11": (0.0, 0.0),
    }
    home = {
        "1": (-45.0, 0.0),  # keeper
        "2": (-15.0, -22.0), "3": (-18.0, -8.0), "4": (-18.0, 8.0), "5": (-15.0, 22.0),
        "6": (0.0, -14.0), "7": (2.0, 0.0), "8": (0.0, 14.0),
        "9": (20.0, -6.0), "10": (18.0, 6.0), "11": (22.0, 18.0),
    }
    return home, away


def scenario_clean_drag() -> Scenario:
    """The textbook case: one runner, one marker, one vacated channel."""
    home, away = _base_shape()
    return Scenario(
        name="clean_drag",
        description="Attacker 10 drags marker 4 wide; the central channel 4 vacated opens.",
        home=home, away=away,
        runs=[Run("10", (20.0, 26.0), 0.5, 2.5)],
        follows={"4": "10"},
        ball_xy=(0.0, 0.0),
        true_creator="10", true_vacater="4",
    )


def scenario_zonal_slide() -> Scenario:
    """The known failure mode: the block slides for the BALL, not for the runner."""
    home, away = _base_shape()
    runs = [Run("10", (18.0, -6.0), 0.5, 2.5, team="home")]  # unrelated run, marked by nobody
    for d, y in (("2", -8.0), ("3", 5.0), ("4", 19.0), ("5", 30.0)):
        runs.append(Run(d, (28.0, y), 0.5, 2.5, team="away"))  # whole line shifts zonally
    return Scenario(
        name="zonal_slide",
        description="Back line slides zonally with the ball; attacker 10's run causes none of it.",
        home=home, away=away,
        runs=runs, follows={},
        ball_xy=(10.0, 20.0),
        true_creator=None, true_vacater=None,
    )


def scenario_decoy_and_beneficiary() -> Scenario:
    """Attacker 9 pulls the marker; attacker 11 runs into what opened. Credit belongs to 9."""
    home, away = _base_shape()
    return Scenario(
        name="decoy_and_beneficiary",
        description="9 drags marker 3 wide; 11 exploits the vacated space. Creator is 9, enterer is 11.",
        home=home, away=away,
        runs=[Run("9", (20.0, -26.0), 0.5, 2.5), Run("11", (26.0, -6.0), 1.0, 3.0)],
        follows={"3": "9"},
        ball_xy=(0.0, 0.0),
        true_creator="9", true_vacater="3", true_enterer="11",
    )


def scenario_static_control() -> Scenario:
    """Nobody moves. Any creation credit produced here is spurious by construction."""
    home, away = _base_shape()
    return Scenario(
        name="static_control",
        description="No movement at all. Correct answer is zero creation and zero credit.",
        home=home, away=away, runs=[], follows={},
        ball_xy=(0.0, 0.0),
        true_creator=None, true_vacater=None,
    )


def scenario_crossed_marking() -> Scenario:
    """Defender 4 tracks attacker 11, but starts NEARER attacker 10.

    Directly probes the A1 marking heuristic's core assumption (nearest attacker == marked
    attacker). Truth is 11; proximity says 10.
    """
    home, away = _base_shape()
    home = dict(home)
    home["10"] = (22.0, 9.0)  # parked right next to defender 4, but not the one being tracked
    home["11"] = (20.0, 14.0)
    return Scenario(
        name="crossed_marking",
        description="Defender 4 tracks attacker 11 while standing nearer attacker 10.",
        home=home, away=away,
        runs=[Run("11", (20.0, 30.0), 0.5, 2.5)],
        follows={"4": "11"},
        ball_xy=(0.0, 0.0),
        true_creator="11", true_vacater="4",
    )


ALL_SCENARIOS = [
    scenario_clean_drag,
    scenario_decoy_and_beneficiary,
    scenario_crossed_marking,
    scenario_zonal_slide,
    scenario_static_control,
]
