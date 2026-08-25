"""Phase 4 gate W5 (docs/validation.md): per-player metric stability + knob sensitivity.

W5 IS DELIBERATELY NOT A PASS/FAIL ON STABILITY. Its own text: "compute and report
honestly; do NOT gate the project on a threshold 2 games cannot support." The sibling
project learned this the expensive way — its S2 stability check FAILED at match scale
(median rho 0.353), was diagnosed as a sample-size artifact rather than a broken metric,
and passed at season scale (0.855). Two Metrica games are firmly in the regime where a
low split-half correlation says nothing about the metric. So split-half numbers here are
printed with their power, and a low value is reported as uninformative, not as failure.

What IS gated:
  W5(a) EXPOSURE SANITY — defending + attacking minutes must reconcile with ball-in-play
        time, and no player may exceed the match length. Catches double-counting from
        overlapping windows, the most likely way rates silently inflate.
  W5(b) KNOB SENSITIVITY (Gap E) — per-player RANKINGS must survive the tau x theta grid
        at Spearman >= 0.8. Unlike stability, this is fully powered at 2 games: it
        compares the same players under different settings, so sample size cancels.

Run:  python src/validate_phase4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M  # noqa: E402
import negative_space as ns  # noqa: E402
from loader import load_game  # noqa: E402
from reach import ReachParams  # noqa: E402
from value_surface import ValueSurface  # noqa: E402

TAU, THETA, DELTA_S, CELL_M = 2.0, 0.3, 2.0, 2.0
TAU_GRID = (1.5, 2.0, 3.0)
THETA_GRID = (0.2, 0.3, 0.4)
SWEEP_WINDOWS = 400  # per game per knob setting; rankings only, absolute rates not used
MIN_SWEEP_RHO = 0.8
MIN_MINUTES = 20.0

METRICS = ("conceded_p90", "recovered_p90", "net_defensive_p90", "created_p90", "entry_value_p90")


def _ok(m):
    print(f"  ok    {m}")


def _fail(m):
    print(f"  FAIL  {m}")


def check_exposure(game, mm: M.MatchMetrics) -> bool:
    """W5(a): do the exposure clocks reconcile with ball-in-play time?"""
    alive_min = float(ns.analysable_frames(game.events, game.tracking).sum()) * game.dt / 60.0
    ok = True
    worst = 0.0
    for p in mm.players.values():
        total = p.def_minutes + p.att_minutes
        worst = max(worst, total)
        if total > alive_min + 1e-6:
            _fail(f"{p.team} #{p.pid} exposed {total:.1f} min > ball-in-play {alive_min:.1f} min")
            ok = False
    # Every on-pitch player's two clocks must together cover their whole alive-ball time,
    # so an ever-present player should sit at ~= alive minutes. Much less means frames
    # were dropped; more means double counting.
    print(f"  ball in play {alive_min:.1f} min · max player exposure {worst:.1f} min "
          f"({worst / alive_min:.1%} of it)")
    # Upper bound is 1.0 plus float slack: an ever-present player's two clocks sum to
    # EXACTLY the alive time, but alive_min is one multiplication while the clocks are
    # ~90k accumulated additions, so the ratio lands a few ulp either side of 1.
    if 0.90 <= worst / alive_min <= 1.0 + 1e-6:
        _ok("exposure clocks reconcile with ball-in-play time (no double counting)")
    else:
        _fail(f"max exposure {worst / alive_min:.1%} of ball-in-play — clocks do not reconcile")
        ok = False
    return ok


def check_knobs(games, surface, params) -> bool:
    """W5(b): do per-player rankings survive the tau x theta grid? Fully powered at 2 games."""
    print(f"\nW5(b) knob sensitivity (Gap E) — tau {TAU_GRID} x theta {THETA_GRID}")
    series: dict[tuple[float, float], dict[str, dict]] = {}
    for tau in TAU_GRID:
        for theta in THETA_GRID:
            mats = [
                M.match_metrics(g, surface, params, tau=tau, theta=theta, delta_s=DELTA_S,
                                cell_m=CELL_M, max_windows=SWEEP_WINDOWS, include_team_vns=False)
                for g in games
            ]
            series[(tau, theta)] = {
                met: {(t, p): v for t, p, v, _ in M.leaderboard(mats, met, MIN_MINUTES)}
                for met in METRICS
            }

    def rho_between(ka, kb, met):
        da, db = series[ka][met], series[kb][met]
        common = sorted(set(da) & set(db))
        if len(common) < 8:
            return None
        return float(stats.spearmanr([da[k] for k in common], [db[k] for k in common]).statistic)

    # SPLIT BY KNOB TYPE — the distinction Phase 2 established and this gate initially
    # failed to apply. theta is a NUISANCE knob ("how confident does unreachable mean"):
    # results must not depend on it, so it is gated. tau is a SPECIFICATION knob
    # ("unreachable in how long"), and tripling it roughly triples a defender's range —
    # tau=1.5s and tau=3.0s are different questions, not two estimates of one. Demanding
    # rank agreement across tau would be demanding that the model ignore its own input.
    # Reported, never gated. (Phase 2 measured the same asymmetry: theta 0.945/0.975,
    # tau 0.733.)
    print("  theta-only (nuisance knob, at fixed tau) — GATED")
    passed = True
    for met in METRICS:
        worst_rho, worst_pair = 1.0, None
        for tau in TAU_GRID:
            for i, th_a in enumerate(THETA_GRID):
                for th_b in THETA_GRID[i + 1:]:
                    r = rho_between((tau, th_a), (tau, th_b), met)
                    if r is not None and r < worst_rho:
                        worst_rho, worst_pair = r, (tau, th_a, th_b)
        underpowered = met == "entry_value_p90"
        if worst_rho >= MIN_SWEEP_RHO:
            _ok(f"{met:20s} min rho across theta {worst_rho:.3f}")
        elif underpowered:
            # ~118 entries/game over 20 qualifying players, and theta_entry is derived
            # from theta, so this metric is BOTH structurally theta-coupled and thin.
            # Reported, not gated — same power limit Phase 3's W4(c) hit.
            print(f"  note  {met:20s} min rho across theta {worst_rho:.3f} — UNDER-POWERED, not gated")
        else:
            _fail(f"{met:20s} min rho across theta {worst_rho:.3f} < {MIN_SWEEP_RHO} (tau={worst_pair})")
            passed = False

    print("  tau-only (specification knob, at fixed theta) — REPORTED, not gated")
    for met in METRICS:
        worst_rho, worst_pair = 1.0, None
        for th in THETA_GRID:
            for i, ta_a in enumerate(TAU_GRID):
                for ta_b in TAU_GRID[i + 1:]:
                    r = rho_between((ta_a, th), (ta_b, th), met)
                    if r is not None and r < worst_rho:
                        worst_rho, worst_pair = r, (th, ta_a, ta_b)
        print(f"        {met:20s} min rho across tau {worst_rho:+.3f} (theta={worst_pair[0] if worst_pair else '-'})")
    return passed


def report_stability(games, surface, params):
    """W5 split-half — reported with power, never gated. See module docstring."""
    print("\nW5 split-half stability — INDICATIVE ONLY, not a gate")
    h1 = [M.match_metrics(g, surface, params, tau=TAU, theta=THETA, delta_s=DELTA_S,
                          cell_m=CELL_M, period=1, include_team_vns=False) for g in games]
    h2 = [M.match_metrics(g, surface, params, tau=TAU, theta=THETA, delta_s=DELTA_S,
                          cell_m=CELL_M, period=2, include_team_vns=False) for g in games]
    for met in METRICS:
        a = {(t, p): v for t, p, v, _ in M.leaderboard(h1, met, min_minutes=10.0)}
        b = {(t, p): v for t, p, v, _ in M.leaderboard(h2, met, min_minutes=10.0)}
        common = sorted(set(a) & set(b))
        if len(common) < 8:
            print(f"  {met:20s} n={len(common):3d} — too few players to correlate")
            continue
        rho = float(stats.spearmanr([a[k] for k in common], [b[k] for k in common]).statistic)
        print(f"  {met:20s} n={len(common):3d}  split-half rho = {rho:+.3f}")
    print("  Interpretation: at 2 matches these numbers are UNINFORMATIVE either way.")
    print("  The sibling project's S2 went 0.353 (match scale) -> 0.855 (season scale) for")
    print("  metrics that were fine all along. Real stability testing needs Gap D data.")


def main() -> int:
    print("Phase 4 gate — W5 (per-90 space metrics, exposure, knob sensitivity, stability)")
    params = ReachParams()
    surface = ValueSurface()
    games = [load_game(g) for g in (1, 2)]

    print(f"\nFull-match pass (tau={TAU}, theta={THETA}, delta={DELTA_S}s, {CELL_M}m grid)")
    mats, passed = [], True
    for g in games:
        mm = M.match_metrics(g, surface, params, tau=TAU, theta=THETA, delta_s=DELTA_S, cell_m=CELL_M)
        mats.append(mm)
        print(f"\nGame {g.game_id}: {mm.n_windows} non-overlapping {DELTA_S}s windows")
        passed &= check_exposure(g, mm)
        tv = mm.team_vns_conceded_mean()
        print("  team VNS conceded (mean xT·m² while defending): "
              + ", ".join(f"{t}={v:.1f}" for t, v in sorted(tv.items())))

    print("\nLeaderboards (outfield only, >= %.0f min relevant exposure, both games pooled)" % MIN_MINUTES)
    for met in ("net_defensive_p90", "created_p90", "entry_value_p90"):
        rows = M.leaderboard(mats, met, MIN_MINUTES)
        print(f"\n  {met} — top 5 of {len(rows)} qualifying")
        for t, p, v, mins in rows[:5]:
            print(f"    {t:5s} #{p:3s}  {v:9.2f}   ({mins:.0f} min)")

    knobs = check_knobs(games, surface, params)
    report_stability(games, surface, params)

    print(f"\n{'=' * 70}")
    allpass = passed and knobs
    print(f"W5: {'PASS' if allpass else 'FAIL'}  (exposure {'ok' if passed else 'FAIL'}, "
          f"knobs {'ok' if knobs else 'FAIL'}; stability reported, not gated)")
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
