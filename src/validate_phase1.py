"""Phase 1 gates W1 and W2 (docs/validation.md).

W1 — REDUCTION TO PITCH CONTROL. The concept claims pitch control is the 2D
uniform-parameter special case of the reachable-volume model. This makes that testable
rather than rhetorical: with uniform parameters, reach.py must reproduce the frozen
oracle's arrival times (tol 1e-9) and its team-race control surface (tol 1e-6) on
randomly sampled frames. Any failure is a bug, not a modelling disagreement.

W2 — NEGATIVE-SPACE INVARIANTS. Four properties that are THEOREMS given the model, so a
violation is likewise a bug rather than a finding:
  (a) coverage is non-decreasing in tau     -> |N| non-increasing in tau
  (b) |N| non-decreasing in theta           -> stricter confidence, more space uncovered
  (c) removing a defender never shrinks N   -> union monotonicity
  (d) coverage ~= 1 at each defender's own location
Plus ball-in-play reconstruction sanity, which everything downstream is filtered on.

The eye test (10 randomly sampled frames, no cherry-picking) is a separate visual step —
see viz_phase1.py. This script covers everything mechanically checkable.

Run:  python src/validate_phase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import negative_space as ns  # noqa: E402
import pitch_control_ref as ref  # noqa: E402
import reach  # noqa: E402
from loader import load_game  # noqa: E402
from reach import DEFAULT_MAX_SPEED as DEFAULT_MS  # noqa: E402
from reach import DEFAULT_REACTION_TIME as DEFAULT_RT  # noqa: E402
from reach import ReachParams  # noqa: E402

GAMES = (1, 2)
N_FRAMES = 20  # randomly sampled analysable frames per game
SEED = 20260730

ARRIVAL_TOL = 1e-9
CONTROL_TOL = 1e-6

# Ball-in-play is typically ~55-60% of elapsed time in professional football. Loose band —
# it is here to catch a BROKEN reconstruction (e.g. 5% or 100%), not to assert a rate.
ALIVE_FRACTION_BAND = (0.45, 0.75)

TAU_GRID = (1.5, 2.0, 3.0)
THETA_GRID = (0.2, 0.3, 0.4)


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _ok(msg: str) -> None:
    print(f"  ok    {msg}")


def _sample_frames(game, rng, n=N_FRAMES):
    """Random ANALYSABLE frames (ball in play, possession resolved). No cherry-picking."""
    mask = ns.analysable_frames(game.events, game.tracking)
    idx = np.flatnonzero(mask)
    take = rng.choice(idx, size=min(n, len(idx)), replace=False)
    return np.sort(take), mask


def check_w1(game, frames) -> bool:
    """reach.py must reproduce the frozen oracle exactly, with uniform parameters."""
    print("\nW1 reduction to pitch control (frozen oracle = pitch_control_ref)")
    passed = True
    params = ReachParams()  # uniform defaults, matching PitchControlParams
    ref_params = ref.PitchControlParams()

    worst_arrival, worst_control = 0.0, 0.0
    for fi in frames:
        row = game.tracking.loc[fi]
        for attacking in ("home", "away"):
            defending = game.defending_team(attacking)

            # Same grid the oracle builds internally (its own default 50x34 layout).
            gx, gy, control_ref = ref.pitch_control_grid(
                row, attacking, game.player_ids, ref_params, nx=50, ny=34
            )
            targets = np.stack([gx.ravel(), gy.ravel()], axis=1)

            att_pos, att_vel = reach.team_positions_velocities(row, attacking, game.player_ids[attacking])
            def_pos, def_vel = reach.team_positions_velocities(row, defending, game.player_ids[defending])
            if len(att_pos) == 0 or len(def_pos) == 0:
                continue

            # (a) arrival times vs the oracle's private _time_to_intercept
            for pos, vel in ((att_pos, att_vel), (def_pos, def_vel)):
                mine = reach.arrival_times(pos, vel, targets, params)
                theirs = ref._time_to_intercept(pos, vel, targets, ref_params)
                worst_arrival = max(worst_arrival, float(np.abs(mine - theirs).max()))

            # (b) derived control surface vs the oracle's grid
            mine_ctrl = reach.control_probability(
                reach.arrival_times(att_pos, att_vel, targets, params),
                reach.arrival_times(def_pos, def_vel, targets, params),
                params,
            ).reshape(control_ref.shape)
            worst_control = max(worst_control, float(np.abs(mine_ctrl - control_ref).max()))

    print(f"  frames checked: {len(frames)} x 2 attacking sides, 1700 grid points each")
    for label, val, tol in (
        ("arrival time", worst_arrival, ARRIVAL_TOL),
        ("control probability", worst_control, CONTROL_TOL),
    ):
        if val <= tol:
            _ok(f"max abs {label} difference {val:.3e} <= {tol:.0e}")
        else:
            _fail(f"max abs {label} difference {val:.3e} exceeds {tol:.0e}")
            passed = False
    return passed


def check_w2(game, frames, alive_mask) -> bool:
    """Structural invariants of negative space. Violations are bugs, not findings."""
    print("\nW2 negative-space invariants")
    passed = True
    params = ReachParams()
    cell_m = 1.0
    gx, gy, targets = reach.pitch_grid(cell_m=cell_m)
    grid_shape = gx.shape
    cell_area = cell_m**2

    poss = ns.possession_by_frame(game.events, game.tracking)

    # --- ball-in-play reconstruction ------------------------------------------------
    alive_frac = float(alive_mask.mean())
    if ALIVE_FRACTION_BAND[0] <= alive_frac <= ALIVE_FRACTION_BAND[1]:
        _ok(f"ball in play {alive_frac:.1%} of frames — plausible reconstruction")
    else:
        _fail(f"ball in play {alive_frac:.1%} of frames, outside {ALIVE_FRACTION_BAND}")
        passed = False

    # --- possession is not inverted -------------------------------------------------
    # Negative space is defined against the DEFENDING team, so swapping the roles would
    # invert every downstream number while leaving all the monotonicity invariants intact
    # (they hold for any team). Nothing else here would catch it. Model-free test: the
    # ball should sit at the POSSESSING team's feet, not the defenders'. Run over a large
    # random sample rather than the 20 rendered frames, since it is cheap.
    rng_p = np.random.default_rng(SEED)
    all_idx = np.flatnonzero(alive_mask & np.array([p is not None for p in poss]))
    sample = rng_p.choice(all_idx, size=min(3000, len(all_idx)), replace=False)
    d_att, d_def = [], []
    for fi in sample:
        row = game.tracking.loc[fi]
        bx, by = row.get("ball_x"), row.get("ball_y")
        if np.isnan(bx) or np.isnan(by):
            continue
        att = poss[fi]
        pa, _ = reach.team_positions_velocities(row, att, game.player_ids[att])
        pdf, _ = reach.team_positions_velocities(row, game.defending_team(att), game.player_ids[game.defending_team(att)])
        if len(pa) == 0 or len(pdf) == 0:
            continue
        d_att.append(np.hypot(pa[:, 0] - bx, pa[:, 1] - by).min())
        d_def.append(np.hypot(pdf[:, 0] - bx, pdf[:, 1] - by).min())
    d_att, d_def = np.array(d_att), np.array(d_def)
    closer = float((d_att < d_def).mean())
    print(
        f"  ball to nearest possessor: median {np.median(d_att):.2f}m  "
        f"vs nearest defender: median {np.median(d_def):.2f}m  (n={len(d_att)})"
    )
    # Not ~100%: the ball is legitimately nearer a defender during passes in flight and
    # contested duels. Well above half is what distinguishes correct from inverted.
    if closer >= 0.70:
        _ok(f"possession not inverted — possessor nearer the ball in {closer:.1%} of frames")
    else:
        _fail(f"possession likely inverted — possessor nearer the ball in only {closer:.1%} of frames")
        passed = False

    # --- (a) monotone in tau, (b) monotone in theta ---------------------------------
    tau_viol = theta_viol = 0
    for fi in frames:
        row = game.tracking.loc[fi]
        attacking = poss[fi]
        defending = game.defending_team(attacking)

        areas_by_tau = []
        for tau in TAU_GRID:
            n = ns.negative_space(
                row, defending, game.player_ids, tau, 0.3, grid_shape, targets, cell_area, params
            )
            areas_by_tau.append(n.area_m2)
        if any(b > a + 1e-9 for a, b in zip(areas_by_tau, areas_by_tau[1:])):
            tau_viol += 1

        areas_by_theta = []
        cov = ns.defensive_coverage(row, defending, game.player_ids, 2.0, targets, params)
        for theta in THETA_GRID:
            areas_by_theta.append(float((cov < theta).sum()) * cell_area)
        if any(b < a - 1e-9 for a, b in zip(areas_by_theta, areas_by_theta[1:])):
            theta_viol += 1

    for label, viol in (("|N| non-increasing in tau", tau_viol), ("|N| non-decreasing in theta", theta_viol)):
        if viol == 0:
            _ok(f"{label}: holds on all {len(frames)} frames")
        else:
            _fail(f"{label}: violated on {viol}/{len(frames)} frames")
            passed = False

    # --- (c) union monotonicity: removing a defender never shrinks N ----------------
    union_viol = 0
    for fi in frames:
        row = game.tracking.loc[fi]
        defending = game.defending_team(poss[fi])
        ids = game.player_ids[defending]
        pos, vel = reach.team_positions_velocities(row, defending, ids)
        if len(pos) < 2:
            continue
        full = reach.team_coverage(reach.arrival_times(pos, vel, targets, params), 2.0, params)
        base_area = float((full < 0.3).sum())
        for drop in range(len(pos)):
            keep = [i for i in range(len(pos)) if i != drop]
            reduced = reach.team_coverage(
                reach.arrival_times(pos[keep], vel[keep], targets, params), 2.0, params
            )
            if float((reduced < 0.3).sum()) < base_area - 1e-9:
                union_viol += 1
    if union_viol == 0:
        _ok(f"union monotonicity: removing any one defender never shrank N ({len(frames)} frames x ~11 drops)")
    else:
        _fail(f"union monotonicity violated {union_viol} times")
        passed = False

    # --- (d) own-location reach -----------------------------------------------------
    # NOT "coverage ~= 1 at a defender's own location", which is what docs/validation.md
    # originally specified by analogy with the main project's V1 check. That analogy does
    # not transfer: V1 tested a RACE (both teams pay the reaction cost, so the nearer team
    # wins and control ~ 1), whereas coverage is absolute and the reaction-time floor caps
    # own-location coverage at sigmoid((tau - t_own)/sigma) — ~0.95 standing still, less
    # when moving. Asserting ~1 would demand behaviour the model cannot and should not
    # produce. Gate spec corrected (see docs/validation.md W2); the invariants below are
    # what is actually provable.
    own_t, own_cov, speeds_at = [], [], []
    for fi in frames:
        row = game.tracking.loc[fi]
        defending = game.defending_team(poss[fi])
        pos, vel = reach.team_positions_velocities(row, defending, game.player_ids[defending])
        if len(pos) == 0:
            continue
        arrivals = reach.arrival_times(pos, vel, pos, params)  # players as their own targets
        own_t.append(np.diag(arrivals))
        own_cov.append(np.diag(reach.coverage_probability(arrivals, 2.0, params)))
        speeds_at.append(np.linalg.norm(vel, axis=1))
    own_t, own_cov, speeds_at = np.concatenate(own_t), np.concatenate(own_cov), np.concatenate(speeds_at)

    # (d1) closed-form identity: t_own == reaction_time * (1 + speed/max_speed), exactly.
    # A real theorem given the motion model — catches indexing/transpose bugs that a loose
    # probability threshold would sail past.
    expected = DEFAULT_RT * (1.0 + speeds_at / DEFAULT_MS)
    ident_err = float(np.abs(own_t - expected).max())
    if ident_err <= 1e-9:
        _ok(f"own-location arrival time matches closed form (max err {ident_err:.2e})")
    else:
        _fail(f"own-location arrival time deviates from closed form by {ident_err:.3e}")
        passed = False

    # (d2) stationary defenders should sit at the model's structural ceiling.
    ceiling = 1.0 / (1.0 + np.exp(-(2.0 - DEFAULT_RT) / 0.45))
    still = speeds_at < 0.5
    if still.any():
        gap = float(np.abs(own_cov[still] - ceiling).max())
        if gap <= 0.02:
            _ok(f"near-stationary defenders at ceiling {ceiling:.4f} (max gap {gap:.4f}, n={still.sum()})")
        else:
            _fail(f"near-stationary own-location coverage off ceiling by {gap:.4f}")
            passed = False

    # (d3) ordering: a defender covers their own location better than the far side of the
    # pitch. Weak but direction-critical — a sign error would flip it.
    far = np.array([[-52.0, -33.0], [52.0, 33.0]])
    worst_order = 1.0
    for fi in frames:
        row = game.tracking.loc[fi]
        defending = game.defending_team(poss[fi])
        pos, vel = reach.team_positions_velocities(row, defending, game.player_ids[defending])
        if len(pos) == 0:
            continue
        own = np.diag(reach.coverage_probability(reach.arrival_times(pos, vel, pos, params), 2.0, params))
        far_cov = reach.coverage_probability(reach.arrival_times(pos, vel, far, params), 2.0, params)
        worst_order = min(worst_order, float((own[None, :] > far_cov).all(axis=0).mean()))
    if worst_order == 1.0:
        _ok("every defender covers their own location more than either far corner")
    else:
        _fail(f"own-location coverage not dominant for some defenders (worst frame {worst_order:.2%})")
        passed = False

    print(
        f"  own-location coverage: mean={own_cov.mean():.4f} min={own_cov.min():.4f} "
        f"(ceiling {ceiling:.4f} at rest; lower when moving — documented reaction-time floor)"
    )

    # --- deterministic vs probabilistic cross-check ---------------------------------
    # The independence assumption in team_coverage (documented in reach.py) biases
    # probabilistic coverage upward where defenders cluster. Report the gap rather than
    # assert a bound on it — it is a known bias to size, not a pass/fail condition.
    gaps = []
    for fi in frames:
        row = game.tracking.loc[fi]
        defending = game.defending_team(poss[fi])
        prob = ns.defensive_coverage(row, defending, game.player_ids, 2.0, targets, params)
        det = ns.defensive_coverage(row, defending, game.player_ids, 2.0, targets, params, deterministic=True)
        gaps.append(float((prob < 0.3).sum() - (det < 0.5).sum()) * cell_area)
    print(
        f"  probabilistic vs deterministic |N| gap: mean {np.mean(gaps):+.0f} m^2 "
        f"(tau=2.0, theta=0.3) — sizing the independence bias, not a gate"
    )
    return passed


def check_game(game_id: int) -> bool:
    print(f"\n{'=' * 70}\nGame {game_id}\n{'=' * 70}")
    rng = np.random.default_rng(SEED + game_id)
    game = load_game(game_id)
    frames, alive_mask = _sample_frames(game, rng)
    w1 = check_w1(game, frames)
    w2 = check_w2(game, frames, alive_mask)
    return w1 and w2


def main() -> int:
    print("Phase 1 gates — W1 (reduction to pitch control) + W2 (negative-space invariants)")
    results = {g: check_game(g) for g in GAMES}
    print(f"\n{'=' * 70}")
    for g, ok in results.items():
        print(f"Game {g}: {'PASS' if ok else 'FAIL'}")
    allpass = all(results.values())
    print(f"Phase 1 mechanical gates: {'PASS' if allpass else 'FAIL'}")
    print("(eye test is separate — run viz_phase1.py)")
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
