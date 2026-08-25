"""Phase 0 smoke test: does the scaffold load real data correctly?

Not a W-gate (those start at W1 in docs/validation.md) — this is the "the plumbing is
sound before anything is modelled" check. It fails loudly rather than printing warnings,
because every later phase silently inherits whatever this gets wrong.

Checks:
  P0.1  Both parseable games load; frame rate, player counts, on-pitch counts sane.
  P0.2  Velocities sane, INCLUDING at the half-time boundary (where players swap ends
        and a naive global diff would manufacture a ~1000 m/s teleport).
  P0.3  Orientation invariants: teams switch ends between halves; the two teams attack
        opposite ways within a half. Reported alongside how many of the four keys had
        real shot evidence vs. came from the switch-ends fallback (a fallback-filled key
        satisfies the switch-ends invariant by construction and proves nothing).
  P0.4  Orientation corroborated by an INDEPENDENT signal: goalkeeper position from
        tracking vs. shot locations from events. Two different data sources; agreement
        is real evidence.

Run:  python src/validate_phase0.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loader import load_game, speeds  # noqa: E402
from orientation import gk_attack_directions  # noqa: E402

GAMES = (1, 2)

# Plausibility bands for elite football, deliberately loose — they exist to catch a
# BROKEN pipeline (all-zero velocities from over-smoothing, or clip-saturated noise),
# not to assert anything about these particular players.
#
# Deliberately NOT gating on max speed. Measured on both games, the max is 16.9706 m/s =
# exactly 12*sqrt(2): the frozen compute_velocities clips vx and vy PER COMPONENT at
# 12 m/s, so the resultant magnitude can reach 12*sqrt(2) even though its docstring reads
# like a speed cap. (Upstream inconsistency, left alone — pitch_control_ref is a frozen
# oracle.) The max is therefore a property of the clip and of tracking-noise spikes, not
# of football: only ~0.01-0.03% of 3.1M player-frames exceed 10 m/s. The meaningful gates
# are a high percentile (real sprinting) plus a ceiling on HOW MUCH of the data is
# implausible.
P99_9_SPEED_BAND = (6.0, 10.0)  # p99.9 player-frame speed: genuine elite sprinting, m/s
NOISE_CEILING_MS = 12.0  # above this is tracking noise, not a player
MAX_NOISE_FRACTION = 0.001  # at most 0.1% of player-frames may exceed the ceiling
MAX_SUBS_OVERLAP_FRACTION = 0.001  # frames with >11 tracked: substitution overlap, must be rare
MIN_FULL_STRENGTH_FRACTION = 0.95  # frames with <11: red cards / tracking gaps


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _ok(msg: str) -> None:
    print(f"  ok    {msg}")


def check_game(game_id: int) -> bool:
    print(f"\n{'=' * 70}\nGame {game_id}\n{'=' * 70}")
    passed = True

    game = load_game(game_id)
    tr, ev = game.tracking, game.events

    # ---- P0.1 load sanity -------------------------------------------------------
    print("\nP0.1 load sanity")
    n_home, n_away = len(game.player_ids["home"]), len(game.player_ids["away"])
    print(f"  frames={len(tr)}  dt={game.dt:.3f}s ({1 / game.dt:.0f} Hz)  events={len(ev)}")
    print(f"  squad columns: home={n_home} away={n_away}  periods={sorted(tr['Period'].unique())}")

    # A team is 11 players. Two documented deviations, both checked as FRACTIONS rather
    # than via the mean (which hides both): brief >11 spikes are substitution overlap,
    # where the outgoing and incoming player are tracked together for a few frames; <11
    # would be a red card or a tracking gap.
    for team, ids in game.player_ids.items():
        on_pitch = np.asarray(sum(tr[f"{team}_{pid}_x"].notna() for pid in ids))
        n = len(on_pitch)
        over, under = int((on_pitch > 11).sum()), int((on_pitch < 11).sum())
        full = 1 - (over + under) / n
        print(
            f"  {team}: exactly 11 in {full:.4%} of frames "
            f"(>11: {over} frames = subs overlap, <11: {under} frames)"
        )
        if over / n > MAX_SUBS_OVERLAP_FRACTION:
            _fail(f"{team}: {over / n:.3%} of frames track >11 players — more than subs overlap")
            passed = False
        elif full >= MIN_FULL_STRENGTH_FRACTION:
            _ok(f"{team}: squad-size sane ({full:.2%} at exactly 11)")
        else:
            _fail(f"{team}: only {full:.2%} of frames at 11 players")
            passed = False

    # ---- P0.2 velocity sanity ---------------------------------------------------
    print("\nP0.2 velocity sanity")
    sp = speeds(tr, game.player_ids)
    p99, p99_9, smean = (float(np.percentile(sp, 99)), float(np.percentile(sp, 99.9)), float(sp.mean()))
    noise_frac = float((sp > NOISE_CEILING_MS).mean())
    print(
        f"  speed: mean={smean:.2f}  p99={p99:.2f}  p99.9={p99_9:.2f}  max={sp.max():.2f} m/s "
        f"(n={len(sp):,} player-frames)"
    )
    if P99_9_SPEED_BAND[0] <= p99_9 <= P99_9_SPEED_BAND[1]:
        _ok(f"p99.9 speed {p99_9:.2f} m/s within {P99_9_SPEED_BAND} — real sprinting present, not over-smoothed")
    else:
        _fail(f"p99.9 speed {p99_9:.2f} m/s outside {P99_9_SPEED_BAND}")
        passed = False
    if noise_frac <= MAX_NOISE_FRACTION:
        _ok(f"only {noise_frac:.4%} of player-frames exceed {NOISE_CEILING_MS} m/s (noise spikes, as expected)")
    else:
        _fail(f"{noise_frac:.4%} of player-frames exceed {NOISE_CEILING_MS} m/s — too noisy to trust")
        passed = False

    # Half-time boundary: players swap ends, so a velocity computed ACROSS the period
    # break would read as a full-pitch teleport in one frame (~50m / 0.04s). The frozen
    # compute_velocities groups by Period to prevent this; verify it actually happened.
    #
    # NOTE the clip would MASK a raw teleport (1250 m/s clips down to 12*sqrt(2)), so the
    # discriminating signal is not "below some huge number" — it is that boundary speeds
    # sit in NORMAL PLAY range. Compared against the game's own p99.9 for exactly that
    # reason: leakage would peg these frames at the clip ceiling, ~2x p99.9.
    p2_start = tr.index[tr["Period"] == 2][0]
    boundary = tr.loc[p2_start:p2_start + 2]
    bmax = 0.0
    for team, ids in game.player_ids.items():
        for pid in ids:
            v = np.hypot(boundary[f"{team}_{pid}_vx"], boundary[f"{team}_{pid}_vy"])
            v = v[~np.isnan(v)]
            if len(v):
                bmax = max(bmax, float(v.max()))
    if bmax <= p99_9:
        _ok(f"half-time boundary max speed {bmax:.2f} m/s <= p99.9 ({p99_9:.2f}) — no cross-period smoothing")
    else:
        _fail(f"half-time boundary speed {bmax:.2f} m/s exceeds p99.9 — velocities leak across the break")
        passed = False

    # ---- P0.3 orientation invariants --------------------------------------------
    print("\nP0.3 orientation invariants")
    d = game.directions
    for (team, period), val in sorted(d.directions.items()):
        src = "shots" if (team, period) in d.from_shots else "FALLBACK"
        print(f"  {team:5s} period {period}: attacks {'+x' if val == 1 else '-x'}   [{src}]")
    print(f"  independently derived from shots: {d.n_independent}/4 keys")

    for team in ("home", "away"):
        if d(team, 1) == -d(team, 2):
            _ok(f"{team} switches ends between halves")
        else:
            _fail(f"{team} attacks the same way in both halves")
            passed = False
    for period in (1, 2):
        if d("home", period) == -d("away", period):
            _ok(f"period {period}: teams attack opposite goals")
        else:
            _fail(f"period {period}: both teams attack the same goal")
            passed = False

    # ---- P0.4 independent corroboration -----------------------------------------
    print("\nP0.4 goalkeeper cross-check (tracking) vs shot inference (events)")
    gk = gk_attack_directions(tr, game.player_ids)
    for key in sorted(gk):
        team, period = key
        e = gk[key]
        agree = e.direction == d(team, period)
        src = "shots" if key in d.from_shots else "fallback"
        line = (
            f"{team:5s} period {period}: GK={e.gk_player:>3s} at x={e.gk_mean_x:+6.1f}m "
            f"(margin {e.margin_m:+5.1f}m) -> {'+x' if e.direction == 1 else '-x'}  "
            f"vs {src}-derived {'+x' if d(team, period) == 1 else '-x'}"
        )
        if agree:
            _ok(line)
        else:
            _fail(line + "  MISMATCH")
            passed = False
        if e.margin_m < 5.0:
            print(f"        note: GK identified by only {e.margin_m:.1f}m — weak corroboration here")

    return passed


def main() -> int:
    print("Phase 0 smoke test — scaffold plumbing (space_analysis)")
    results = {g: check_game(g) for g in GAMES}

    print(f"\n{'=' * 70}")
    for g, ok in results.items():
        print(f"Game {g}: {'PASS' if ok else 'FAIL'}")
    allpass = all(results.values())
    print(f"Phase 0: {'PASS — scaffold sound, ready for Phase 1' if allpass else 'FAIL'}")
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
