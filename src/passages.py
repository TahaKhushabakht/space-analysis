"""Passage extraction — windows of play worth looking at.

Data layer, not analysis: it decides WHICH stretches of a match get precomputed for the
explorer (and later, any dashboard view). Kept separate from rendering so the same
passage definitions can feed a different visual, or a per-player aggregation, without
duplicating the selection logic.

DEFAULT SELECTION — windows ending in a shot. Chosen over random sampling because this
is an inspection tool, not a validation gate: the point is to watch space in the moments
it mattered. That IS a biased sample (success-conditioned on reaching a shot), which is
fine here and would NOT be fine for any measurement — the W-gates deliberately sample
randomly instead. Do not compute statistics over these passages and report them as
match-level facts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import negative_space as ns


@dataclass
class Passage:
    """A contiguous stretch of frames ending at a shot."""

    game_id: int
    period: int
    frame_indices: np.ndarray  # positional indices into tracking (already subsampled)
    attacking_team: str
    direction: int  # +1 / -1, attack direction for attacking_team in this period
    label: str
    shot_subtype: str


def _fmt_clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def extract_shot_passages(
    game,
    window_s: float = 8.0,
    sample_hz: float = 5.0,
    max_passages: int | None = 10,
    min_frames: int = 12,
) -> list[Passage]:
    """Windows of `window_s` seconds ending at each shot, subsampled to `sample_hz`.

    Only frames with the ball in play AND possession resolved are kept, and only frames
    where the shooting team is the team in possession — a window that starts before a
    turnover would otherwise render the wrong team as "attacking" for its first half.
    Passages shorter than `min_frames` after filtering are dropped.

    Shots are taken in chronological order; when `max_passages` is set, the passages with
    the most usable frames are kept (longest uninterrupted build-ups), then re-sorted
    chronologically for a sensible dropdown order.
    """
    tracking, events = game.tracking, game.events
    poss = ns.possession_by_frame(events, tracking)
    analysable = ns.analysable_frames(events, tracking)
    frame_numbers = tracking["Frame"].to_numpy(dtype=float)

    step = max(1, int(round((1.0 / sample_hz) / game.dt)))
    window_frames = window_s / game.dt

    shots = events[events["Type"] == "SHOT"].dropna(subset=["Start Frame"])
    passages: list[Passage] = []

    for _, ev in shots.iterrows():
        shot_frame = float(ev["Start Frame"])
        team = str(ev["Team"]).lower()
        lo = shot_frame - window_frames

        # Positional indices whose Frame lies in the window.
        idx = np.flatnonzero((frame_numbers >= lo) & (frame_numbers <= shot_frame))
        if len(idx) == 0:
            continue
        idx = idx[::step]
        keep = np.array([analysable[i] and poss[i] == team for i in idx], dtype=bool)
        idx = idx[keep]
        if len(idx) < min_frames:
            continue

        period = int(tracking.iloc[idx[0]]["Period"])
        t0 = float(tracking.iloc[idx[0]]["Time [s]"])
        subtype = "" if pd.isna(ev.get("Subtype")) else str(ev.get("Subtype"))
        passages.append(
            Passage(
                game_id=game.game_id,
                period=period,
                frame_indices=idx,
                attacking_team=team,
                direction=game.direction(team, period),
                label=f"P{period} {_fmt_clock(t0)} — {team} shot" + (f" ({subtype})" if subtype else ""),
                shot_subtype=subtype,
            )
        )

    if max_passages is not None and len(passages) > max_passages:
        passages = sorted(passages, key=lambda p: -len(p.frame_indices))[:max_passages]
        passages.sort(key=lambda p: p.frame_indices[0])
    return passages
