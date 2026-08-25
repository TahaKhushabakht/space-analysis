"""Attribution validated against SCRIPTED ground truth (see simulate.py).

The question no real-data gate can answer: when the model says a player created space,
is it pointing at the player who actually created it?

Two attributions are tested separately, because they have very different standing:

  DEFENDER-SIDE ("which defender's reach vacated this ground") is mechanical — it falls
  out of the additive log-complement decomposition and W4 already proved it is
  exhaustive. Here it is checked for CORRECTNESS rather than just completeness: the
  defender the script actually moved should receive the credit. Expect near-perfect; a
  failure would mean the decomposition is exact but pointed at the wrong player.

  ATTACKER-SIDE ("who pulled that defender") is the A1 marking heuristic — a causal guess
  that real data measured as badly position-confounded (ρ = +0.956 with mean pitch
  position, and stripping the value weighting only moved it to +0.806, which located the
  problem in this heuristic rather than in the xT surface). These scenarios say what its
  accuracy actually is, and on which failure modes.

SCORING. Each scenario runs `n_trials` times with jittered starting positions, so results
describe a family of configurations rather than one geometry. A scenario with a
`true_creator` scores a HIT when that player receives the largest share of creation
credit. A scenario with `true_creator=None` is a false-positive probe: any confident
credit there is wrong by construction.

Run:  python src/validate_attribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dynamics as dyn  # noqa: E402
import reach  # noqa: E402
import simulate as sim  # noqa: E402
from reach import ReachParams  # noqa: E402
from value_surface import ValueSurface  # noqa: E402

TAU, THETA = 2.0, 0.3
CELL_M = 2.0
DELTA_S = 2.0
DIRECTION = 1  # home attacks toward +x in every scenario
N_TRIALS = 40
JITTER_M = 1.5

# A scenario whose truth is "nobody created anything" should not produce a confident
# winner. Credit is judged spurious when one attacker takes more than this share.
SPURIOUS_SHARE = 0.50


def _ok(m):
    print(f"  ok    {m}")


def _fail(m):
    print(f"  FAIL  {m}")


def run_trial(sc: sim.Scenario, rng, surface, params, targets, grid_shape, value):
    """One jittered rendering of a scenario -> the model's attribution for it."""
    tracking, player_ids = sim.build_scenario(sc, jitter_m=JITTER_M, rng=rng)
    step = int(round(DELTA_S / sim.DT))
    # Compare the frame before the run starts against one a full delta later, so the
    # window spans the scripted movement rather than clipping it.
    i0 = max(int(round(0.25 / sim.DT)), 1)
    i1 = min(i0 + step, len(tracking) - 1)
    return dyn.frame_dynamics(
        tracking.iloc[i0], tracking.iloc[i1], "home", player_ids, DIRECTION,
        TAU, THETA, targets, value, CELL_M**2, DELTA_S, params, grid_shape=grid_shape,
    )


def score(sc: sim.Scenario, results) -> dict:
    """Aggregate hits/misses across trials for one scenario."""
    def_hits = att_hits = spurious = 0
    n_scored = n_att_scored = 0
    opened = []
    att_winners: dict[str, int] = {}
    for r in results:
        opened.append(r.opened_xt)
        # Only trials where space actually opened are scoreable. Otherwise `max()` over an
        # all-zero dict returns whichever key happens to be first, which is not an
        # attribution at all — scoring it would measure dict ordering.
        if sum(r.created_by_defender.values()) > 0:
            n_scored += 1
            top_def = max(r.created_by_defender.items(), key=lambda kv: kv[1])[0]
            if sc.true_vacater is not None and top_def == sc.true_vacater:
                def_hits += 1
        total = sum(r.credited_to_attacker.values())
        if total > 0:
            n_att_scored += 1
            top_att, top_val = max(r.credited_to_attacker.items(), key=lambda kv: kv[1])
            att_winners[top_att] = att_winners.get(top_att, 0) + 1
            if sc.true_creator is not None and top_att == sc.true_creator:
                att_hits += 1
            if sc.true_creator is None and top_val / total > SPURIOUS_SHARE:
                spurious += 1
    return dict(
        n=len(results), n_scored=n_scored, n_att_scored=n_att_scored,
        def_hits=def_hits, att_hits=att_hits, spurious=spurious,
        mean_opened=float(np.mean(opened)) if opened else 0.0,
        att_winners=att_winners,
    )


def main() -> int:
    print("Attribution vs scripted ground truth (synthetic tracking)")
    print(f"tau={TAU} theta={THETA} delta={DELTA_S}s cell={CELL_M}m  "
          f"{N_TRIALS} jittered trials/scenario (sigma={JITTER_M}m)\n")

    params = ReachParams()
    surface = ValueSurface()
    gx, gy, targets = reach.pitch_grid(cell_m=CELL_M)
    value = surface.value_at_targets(targets, DIRECTION)

    passed = True
    summary = []
    for factory in sim.ALL_SCENARIOS:
        sc = factory()
        rng = np.random.default_rng(4242)
        results = []
        for _ in range(N_TRIALS):
            r = run_trial(sc, rng, surface, params, targets, gx.shape, value)
            if r is not None:
                results.append(r)
        s = score(sc, results)
        summary.append((sc, s))

        print(f"{sc.name}")
        print(f"  {sc.description}")
        print(f"  mean space opened: {s['mean_opened']:.2f} xT·m² "
              f"({s['n_scored']}/{s['n']} trials opened any space)")

        if sc.true_vacater is not None and s["n_scored"]:
            acc = s["def_hits"] / s["n_scored"]
            line = f"defender-side: #{sc.true_vacater} charged in {acc:.0%} of scoreable trials"
            _ok(line) if acc >= 0.9 else _fail(line + "  (mechanical step should be near-perfect)")
            passed &= acc >= 0.9

        top = sorted(s["att_winners"].items(), key=lambda kv: -kv[1])[:3]
        got = ", ".join(f"#{k}:{v}" for k, v in top) or "none"
        if sc.true_creator is not None:
            acc = s["att_hits"] / max(s["n_att_scored"], 1)
            print(f"  attacker-side: true creator #{sc.true_creator} credited in {acc:.0%} "
                  f"of {s['n_att_scored']} scoreable trials   (most-credited: {got})")
        else:
            rate = s["spurious"] / max(s["n_att_scored"], 1)
            print(f"  attacker-side: truth is NOBODY. Confident credit given in {rate:.0%} "
                  f"of {s['n_att_scored']} trials that produced any   (most-credited: {got})")
        print()

    print("=" * 70)
    print("Attacker-side accuracy summary (the A1 marking heuristic under test):")
    for sc, s in summary:
        if sc.true_creator is not None:
            print(f"  {sc.name:24s} {s['att_hits'] / max(s['n_att_scored'], 1):5.0%} correct")
        else:
            print(f"  {sc.name:24s} {s['spurious'] / max(s['n_att_scored'], 1):5.0%} spurious (lower is better)")
    print("\nDefender-side attribution is the mechanical half and is gated; attacker-side is")
    print("reported, not gated — it is a causal guess and these numbers say how good a guess.")
    print("Scripted marking is cleaner than real defending, so treat attacker-side accuracy")
    print("as an UPPER BOUND on real-data performance, never as an estimate of it.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
