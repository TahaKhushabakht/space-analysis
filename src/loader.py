"""One-call loading of a Metrica game, ready for reachable-set analysis.

Wraps the frozen parser (`metrica_io`) and velocity computation (`pitch_control_ref`)
and adds the orientation resolution every later phase needs, so no phase has to
re-derive the setup or accidentally derive it differently.

DATA LOCATION: the Metrica sample data lives in the sibling project
(`../surplus_value_model/data/metrica/data`) and is read READ-ONLY from there rather
than duplicated — consistent with the one-way dependency in NOTES.md. Override with
`data_dir` if it moves.

VELOCITIES: computed by the frozen `compute_velocities` (smoothed central difference,
per period, clipped at 12 m/s). Imported rather than re-implemented — importing the
oracle's utilities is fine; only its MODEL math is off-limits to duplicate (see the
provenance header in pitch_control_ref.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import metrica_io
from orientation import AttackDirections, resolve_attack_directions
from pitch_control_ref import compute_velocities, player_ids_for_team

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "surplus_value_model" / "data" / "metrica" / "data"
)

# Metrica sample games are 25 Hz. Derived from the data rather than assumed (see `_frame_dt`),
# but a game far from this is a red flag worth failing on.
EXPECTED_DT_S = 0.04


@dataclass
class Game:
    """A loaded Metrica game with everything the later phases need."""

    game_id: int
    tracking: pd.DataFrame  # metric centre-origin coords, with _vx/_vy columns added
    events: pd.DataFrame  # RAW / unoriented — frames still align with tracking
    player_ids: dict[str, list[str]]  # {"home": [...], "away": [...]}
    directions: AttackDirections
    dt: float  # seconds per frame

    def defending_team(self, attacking_team: str) -> str:
        return "away" if attacking_team == "home" else "home"

    def direction(self, team: str, period: int) -> int:
        return self.directions(team, period)


def _frame_dt(tracking: pd.DataFrame) -> float:
    """Median seconds per frame, computed within periods (never across the half-time gap)."""
    deltas = tracking.groupby("Period")["Time [s]"].diff().dropna()
    return float(deltas.median())


def load_game(game_id: int, data_dir: Path | str | None = None, smoothing_window: int = 7) -> Game:
    """Load Metrica Sample_Game_{game_id} (1 or 2) with velocities and attack directions.

    Sample_Game_3 ships in a different format and is not supported (see metrica_io).
    """
    if game_id not in (1, 2):
        raise ValueError(f"Only Metrica Sample_Game_1 and _2 are parseable; got {game_id}")

    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Metrica data directory not found: {data_dir}\n"
            "Expected it in the sibling surplus_value_model project; pass data_dir= to override."
        )

    tracking, events = metrica_io.load_game(data_dir, game_id)

    player_ids = {
        "home": player_ids_for_team(tracking.columns, "home"),
        "away": player_ids_for_team(tracking.columns, "away"),
    }

    dt = _frame_dt(tracking)
    if not np.isclose(dt, EXPECTED_DT_S, atol=0.005):
        raise ValueError(f"Unexpected frame interval {dt:.4f}s (expected ~{EXPECTED_DT_S}s @ 25Hz)")

    tracking = compute_velocities(tracking, player_ids, dt=dt, smoothing_window=smoothing_window)
    directions = resolve_attack_directions(events)

    return Game(
        game_id=game_id,
        tracking=tracking,
        events=events,
        player_ids=player_ids,
        directions=directions,
        dt=dt,
    )


def speeds(tracking: pd.DataFrame, player_ids: dict[str, list[str]]) -> np.ndarray:
    """Flat array of every player-frame speed (m/s), NaNs (off-pitch players) dropped.

    Diagnostic helper for velocity sanity checks.
    """
    out = []
    for team, ids in player_ids.items():
        for pid in ids:
            vx = tracking[f"{team}_{pid}_vx"].to_numpy(dtype=float)
            vy = tracking[f"{team}_{pid}_vy"].to_numpy(dtype=float)
            out.append(np.hypot(vx, vy))
    allspeeds = np.concatenate(out)
    return allspeeds[~np.isnan(allspeeds)]
