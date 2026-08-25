"""Phase 3 gate W4 (docs/validation.md): the dynamics decomposition.

W4(a) ACCOUNTING IDENTITY — theorem class. Per-defender attributions of created and
destroyed space must sum back to the frame's total exactly (exhaustive and exclusive).
True by construction given the additive log-complement decomposition in dynamics.py, so
any failure is a bug, never a finding. Checked to 1e-9 relative.

W4(b) EYE TEST — the 10 largest creation and 10 largest destruction events rendered for
inspection (viz_phase3.py), 8/10 defensible required. Failures are the most informative
output; log them.

W4(c) KNOWN-PATTERN CHECK, direction only. Advanced players should dominate final-third
creation and entry; deep players should dominate destruction. Metrica is anonymised, so
role comes from mean oriented x — a crude proxy the main project's HISTORY §29 showed is
unreliable for outliers (a high defensive line pushes centre-backs into midfield
positions). Treated as INDICATIVE, printed and sanity-checked, never a hard gate.

NOT covered here: attacker-side creation credit is reported but not gated. It is a causal
guess (marking heuristic, Gap A) and W4 was specified to be passable without it.

Run:  python src/validate_phase3.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dynamics as dyn  # noqa: E402
import negative_space as ns  # noqa: E402
import reach  # noqa: E402
from loader import load_game  # noqa: E402
from orientation import goalkeeper_ids  # noqa: E402
from reach import ReachParams  # noqa: E402
from value_surface import ValueSurface  # noqa: E402

SEED = 20260731
TAU, THETA = 2.0, 0.3
CELL_M = 2.0
# delta RAISED from 0.6s after the first Phase 3 run failed its eye test. Measured
# coherence (largest connected opened region as a share of opened cells):
#   0.6s -> 29.2% over 11.5 fragments | 1.2s -> 39.0% | 2.0s -> 48.8% | 3.0s -> 57.1%
# At 0.6s the decomposition measures the coverage frontier JITTERING, not space being
# created. 2.0s is chosen over the more coherent 3.0s because attribution blurs as the
# window grows: over 3s several defenders move for unrelated reasons and "who opened
# this" becomes progressively less meaningful. 2.0s is the coherence/causality trade.
DELTA_S = 2.0
N_WINDOWS = 400  # random (t, t+delta) pairs per game
IDENT_TOL = 1e-9
MIN_COHERENCE = 0.40  # largest opened region must hold >=40% of opened cells on average
MIN_ENTRY_XT = 0.01  # materiality floor for an entry — calibrated, see dynamics.py table


def _fail(m):
    print(f"  FAIL  {m}")


def _ok(m):
    print(f"  ok    {m}")


def run_game(game_id: int):
    game = load_game(game_id)
    surface = ValueSurface()
    params = ReachParams()
    gx, gy, targets = reach.pitch_grid(cell_m=CELL_M)
    cell_area = CELL_M**2

    poss = ns.possession_by_frame(game.events, game.tracking)
    analysable = ns.analysable_frames(game.events, game.tracking)
    gks = goalkeeper_ids(game.tracking, game.player_ids)
    step = int(round(DELTA_S / game.dt))

    rng = np.random.default_rng(SEED + game_id)
    cand = np.flatnonzero(analysable)
    cand = cand[cand + step < len(game.tracking)]
    sel = rng.choice(cand, size=min(N_WINDOWS, len(cand)), replace=False)

    value_cache: dict[int, np.ndarray] = {}
    results = []
    worst_open = worst_close = 0.0
    created_tot = defaultdict(float)
    destroyed_tot = defaultdict(float)
    entered_tot = defaultdict(float)
    mean_x = defaultdict(list)

    for i in sel:
        j = i + step
        # Same possession across the window, or "who is defending" changes mid-difference.
        if not analysable[j] or poss[i] != poss[j] or poss[i] is None:
            continue
        row0, row1 = game.tracking.iloc[i], game.tracking.iloc[j]
        if int(row0["Period"]) != int(row1["Period"]):
            continue
        att = poss[i]
        d = game.direction(att, int(row0["Period"]))
        if d not in value_cache:
            value_cache[d] = surface.value_at_targets(targets, d)

        r = dyn.frame_dynamics(
            row0, row1, att, game.player_ids, d, TAU, THETA, targets, value_cache[d],
            cell_area, DELTA_S, params, grid_shape=gx.shape,
            gk_id=gks.get(game.defending_team(att)), min_entry_xt=MIN_ENTRY_XT,
        )
        if r is None:
            continue
        results.append(r)

        # --- W4(a) identity, per window ---
        s_open = sum(r.created_by_defender.values())
        s_close = sum(r.destroyed_by_defender.values())
        for total, parts, store in ((r.opened_xt, s_open, "open"), (r.closed_xt, s_close, "close")):
            denom = max(abs(total), 1e-12)
            err = abs(total - parts) / denom
            if store == "open":
                worst_open = max(worst_open, err)
            else:
                worst_close = max(worst_close, err)

        for pid, v in r.created_by_defender.items():
            created_tot[(game.defending_team(att), pid)] += v
        for pid, v in r.destroyed_by_defender.items():
            destroyed_tot[(game.defending_team(att), pid)] += v
        for pid in r.entries:
            entered_tot[(att, pid)] += r.entry_value.get(pid, 0.0)

        # role proxy: mean oriented x of each player while their team attacks
        for team in ("home", "away"):
            sign = d if team == att else -d
            for pid in game.player_ids[team]:
                x = row0.get(f"{team}_{pid}_x")
                if not np.isnan(x):
                    mean_x[(team, pid)].append(x * sign)

    return dict(
        results=results, worst_open=worst_open, worst_close=worst_close,
        created=created_tot, destroyed=destroyed_tot, entered=entered_tot,
        mean_x={k: float(np.mean(v)) for k, v in mean_x.items() if len(v) > 20},
        gks=gks,
    )


def check(game_id: int) -> bool:
    print(f"\n{'=' * 70}\nGame {game_id}\n{'=' * 70}")
    R = run_game(game_id)
    n = len(R["results"])
    passed = True
    print(f"  {n} windows of {DELTA_S}s (tau={TAU}, theta={THETA}, {CELL_M}m grid)")

    tot_open = sum(r.opened_xt for r in R["results"])
    tot_close = sum(r.closed_xt for r in R["results"])
    print(f"  total opened {tot_open:.1f} xT·m²   destroyed {tot_close:.1f} xT·m² "
          f"(net {tot_open - tot_close:+.1f})")

    print("\nW4(a) accounting identity — per-defender attributions vs frame total")
    for label, err in (("created", R["worst_open"]), ("destroyed", R["worst_close"])):
        if err <= IDENT_TOL:
            _ok(f"{label}: max relative residual {err:.2e} <= {IDENT_TOL:.0e} (exhaustive & exclusive)")
        else:
            _fail(f"{label}: max relative residual {err:.2e} exceeds {IDENT_TOL:.0e}")
            passed = False

    # --- W4(b-quant): are flipped cells coherent regions or scattered jitter? ----------
    # The qualitative eye test lives in viz_phase3.py; this is its measurable half, added
    # after the delta=0.6s run produced thin fragmented shells rather than channels.
    print("\nW4(b) region coherence — is the opened space a channel or frontier jitter?")
    coh = np.array([r.coherence for r in R["results"] if r.opened_m2 > 0])
    nreg = np.array([r.n_opened_regions for r in R["results"] if r.opened_m2 > 0])
    if len(coh):
        print(f"  largest opened region holds {coh.mean():.1%} of opened cells "
              f"(median {np.median(coh):.1%}), across {nreg.mean():.1f} regions on average")
        if coh.mean() >= MIN_COHERENCE:
            _ok(f"coherence {coh.mean():.1%} >= {MIN_COHERENCE:.0%} at delta={DELTA_S}s")
        else:
            _fail(f"coherence {coh.mean():.1%} below {MIN_COHERENCE:.0%} — still measuring jitter")
            passed = False

    # --- goalkeeper split (fix 2) -----------------------------------------------------
    # The keeper stays in the coverage union and in the identity; only the REPORTING is
    # split, because guarding the highest-value ground is not the same as creating space.
    print("\nGoalkeeper vs outfield creation credit (keeper kept in coverage, split in reporting)")
    gk_c = sum(v for (t, pid), v in R["created"].items() if R["gks"].get(t) == pid)
    all_c = sum(R["created"].values()) or 1.0
    gk_d = sum(v for (t, pid), v in R["destroyed"].items() if R["gks"].get(t) == pid)
    all_d = sum(R["destroyed"].values()) or 1.0
    print(f"  keepers {R['gks']}: {gk_c / all_c:.1%} of creation credit, {gk_d / all_d:.1%} of destruction")
    print(f"  outfield-only totals: created {all_c - gk_c:.1f}, destroyed {all_d - gk_d:.1f} xT·m²")

    print("\nW4(c) known-pattern check — INDICATIVE ONLY (anonymised role proxy, outfield only)")
    mx = {k: v for k, v in R["mean_x"].items() if R["gks"].get(k[0]) != k[1]}
    if not mx:
        print("  (insufficient data for role proxy)")
        return passed

    # Split each squad into thirds by mean oriented x: deep / middle / advanced.
    for team in ("home", "away"):
        players = [(pid, x) for (t, pid), x in mx.items() if t == team]
        if len(players) < 9:
            continue
        players.sort(key=lambda kv: kv[1])
        k = len(players) // 3
        groups = {"deep": players[:k], "middle": players[k:-k], "advanced": players[-k:]}
        line = []
        for gname, grp in groups.items():
            c = sum(R["created"].get((team, pid), 0.0) for pid, _ in grp)
            d = sum(R["destroyed"].get((team, pid), 0.0) for pid, _ in grp)
            e = sum(R["entered"].get((team, pid), 0.0) for pid, _ in grp)
            line.append((gname, c, d, e))
        tot_c = sum(v[1] for v in line) or 1.0
        tot_d = sum(v[2] for v in line) or 1.0
        tot_e = sum(v[3] for v in line) or 1.0
        print(f"  {team}:")
        for gname, c, d, e in line:
            print(f"    {gname:9s} vacated {c / tot_c:5.1%}   re-covered {d / tot_d:5.1%}   entered {e / tot_e:5.1%}")

    n_entries = sum(len(r.entries) for r in R["results"])
    adv_entry = sum(
        R["entered"].get((t, pid), 0.0)
        for (t, pid), x in mx.items()
        if x > np.median([v for v in mx.values()])
    )
    all_entry = sum(R["entered"].values()) or 1.0
    print(f"  {n_entries} entries in {len(R['results'])} windows "
          f"(theta_entry={THETA * 2 / 3:.2f}, min xT={MIN_ENTRY_XT})")
    print(f"  advanced-half players account for {adv_entry / all_entry:.1%} of entered value")
    # Power check before reading anything into the split. The sample covers ~12% of the
    # match (400 windows x 2s of ~5800s), so entries land in the tens — far too few to
    # split across 3 role groups x 2 teams and interpret. Saying "entries do not skew
    # advanced" off n=14 would be reporting noise as a finding.
    if n_entries < 50:
        print(f"  UNDER-POWERED: n={n_entries} entries cannot support a role split — not interpreted.")
        print("  (entry RATE is calibrated and plausible; the direction check needs more data,")
        print("   the same 2-game power limit W5 already documents)")
    elif adv_entry / all_entry > 0.5:
        _ok("entries skew to advanced players, as expected (direction only, not gated)")
    else:
        print("  note: entries do NOT skew advanced — worth a look, but the role proxy is crude")

    print("\nAttacker-side creation credit (marking heuristic — reported, NOT gated)")
    cred = defaultdict(float)
    for r in R["results"]:
        for pid, v in r.credited_to_attacker.items():
            cred[pid] += v
    top = sorted(cred.items(), key=lambda kv: -kv[1])[:5]
    print("  top credited: " + ", ".join(f"{pid}={v:.1f}" for pid, v in top))
    return passed


def main() -> int:
    print("Phase 3 gate — W4 (creation / destruction / entry decomposition)")
    res = {g: check(g) for g in (1, 2)}
    print(f"\n{'=' * 70}")
    for g, ok in res.items():
        print(f"Game {g}: {'PASS' if ok else 'FAIL'}")
    allpass = all(res.values())
    print(f"W4 mechanical: {'PASS' if allpass else 'FAIL'}   (eye test separate — viz_phase3.py)")
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
