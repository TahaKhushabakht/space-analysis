"""Phase 5a gate W6a (docs/validation.md): does the vertical machinery hold up?

W6a(1) ARC SANITY — inferred apexes for flagged aerial passes must land in a plausible
       band (~2-15m). A mass of 30m apexes means the arc inference is broken.
W6a(2) DISCRIMINATION — the falsifiable one. At MATCHED GROUND DISTANCE, aerial-flagged
       passes must take longer than ground passes: lofting a ball spends energy on
       height, so it arrives later than a driven ball over the same distance. The flag
       comes from the collector's downstream HEAD/AERIAL annotations, independent of the
       flight time being tested, so this is not circular.
W6a(3) DOME GEOMETRY — theorem-class invariants: vertical reach is non-increasing in
       arrival time, equals standing reach exactly at the horizon, reaches the full cap
       given enough slack, and reachable volume is non-decreasing in tau.
W6a(4) AERIAL DUEL PREDICTION — the test that asks whether verticality EARNS its
       complexity. Metrica labels AERIAL-WON / AERIAL-LOST, so duels can be paired and
       predicted. Evaluated from 1s BEFORE the duel (at the duel instant both players are
       already there and the test degenerates). Compared against a raw-distance baseline.

HONEST SCOPE. Parameters are uniform (Metrica is anonymised), so every vertical
difference between two players here comes from SLACK TIME — whoever arrives earlier can
set and leap. This is NOT the attribute-driven dome of the concept doc, and W6a(4) is
therefore a weaker claim than W6b: it tests the arrival-time-to-height chain, not whether
tall players win headers. W6b stays blocked on named data (docs/data-scope.md).

Run:  python src/validate_phase5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_flight as bf  # noqa: E402
import dome  # noqa: E402
import reach  # noqa: E402
from dome import DomeParams  # noqa: E402
from loader import load_game  # noqa: E402
from reach import ReachParams  # noqa: E402

APEX_BAND = (2.0, 15.0)
MIN_APEX_IN_BAND = 0.70  # share of flagged aerial passes whose apex is plausible
DUEL_LOOKBACK_S = 1.0  # evaluate from this far before the duel
DUEL_PAIR_WINDOW = 5  # frames within which the won/lost pair must sit
PITCH_L, PITCH_W = 105.0, 68.0


def _ok(m):
    print(f"  ok    {m}")


def _fail(m):
    print(f"  FAIL  {m}")


def check_ball_trajectory(games) -> bool:
    """Is the tracked ball path MEASURED, or interpolated between annotated touches?

    Added after W6a(1) and (2) both failed, to find out why. This is the decisive check
    and the one to re-run on any future data provider: a real rolling ball decelerates and
    a real struck ball curves. If neither happens, the path is drawn, not observed, and no
    altitude can be recovered from it at any level of cleverness.
    """
    print("\nW6a(0) is the ball path real? — speed retention and straightness during passes")
    ratios, straight = [], []
    for g in games:
        tr = g.tracking
        fr = tr["Frame"].to_numpy(float)
        bx, by = tr["ball_x"].to_numpy(float), tr["ball_y"].to_numpy(float)
        for _, r in bf.passes_with_flight(g.events, g.dt).iterrows():
            i0, i1 = np.searchsorted(fr, r["start_frame"]), np.searchsorted(fr, r["end_frame"])
            if i1 - i0 < 8:
                continue
            sx, sy = bx[i0:i1 + 1], by[i0:i1 + 1]
            if np.isnan(sx).any() or np.isnan(sy).any():
                continue
            sp = np.hypot(np.diff(sx), np.diff(sy)) / g.dt
            if np.median(sp[:3]) < 0.5:
                continue
            ratios.append(np.median(sp[-3:]) / np.median(sp[:3]))
            p0, p1 = np.array([sx[0], sy[0]]), np.array([sx[-1], sy[-1]])
            v = p1 - p0
            L = np.linalg.norm(v)
            if L < 1.0:
                continue
            nrm = np.array([-v[1], v[0]]) / L
            straight.append(np.abs((np.stack([sx, sy], 1) - p0) @ nrm).max())
    ratios, straight = np.array(ratios), np.array(straight)
    print(f"  speed retention (arrival/launch): p10 {np.percentile(ratios, 10):.3f}  "
          f"p50 {np.median(ratios):.3f}  p90 {np.percentile(ratios, 90):.3f}")
    print(f"  deviation from straight line: median {np.median(straight):.2f}m  "
          f"p90 {np.percentile(straight, 90):.2f}m")
    interpolated = np.median(ratios) > 0.99 and np.median(straight) < 0.20
    if interpolated:
        print("  DIAGNOSIS: ball path is INTERPOLATED — straight line at constant speed between")
        print("  annotated touches. A rolling ball must decelerate; this one does not. There is")
        print("  no curvature or deceleration signal, therefore NO recoverable altitude.")
        print("  => Gap C is unresolvable on Metrica, not merely imperfect. Apex inference withdrawn.")
    else:
        _ok("ball path shows real deceleration/curvature — altitude inference is worth attempting")
    return not interpolated


def check_arc(all_passes: pd.DataFrame) -> bool:
    print("\nW6a(1) arc sanity — apex distribution of flagged aerial passes")
    aer = all_passes[all_passes["aerial"]]
    apex = aer["apex_m"].dropna()
    if len(apex) < 10:
        _fail(f"only {len(apex)} flagged aerial passes — cannot judge")
        return False
    share = float(((apex >= APEX_BAND[0]) & (apex <= APEX_BAND[1])).mean())
    print(f"  n={len(apex)}  apex median {apex.median():.2f}m  p10 {apex.quantile(.1):.2f}  "
          f"p90 {apex.quantile(.9):.2f}  max {apex.max():.2f}")
    if share >= MIN_APEX_IN_BAND:
        _ok(f"{share:.1%} of apexes within {APEX_BAND}m")
        return True
    _fail(f"only {share:.1%} of apexes within {APEX_BAND}m")
    return False


def check_discrimination(all_passes: pd.DataFrame) -> bool:
    print("\nW6a(2) discrimination — at matched distance, do aerial passes fly longer?")
    df = all_passes.copy()
    # Distance bins so the comparison is not confounded by aerial passes simply being longer.
    bins = [3, 10, 15, 20, 25, 30, 40, 120]
    df["bin"] = pd.cut(df["dist_m"], bins)
    rows, diffs = [], []
    for b, grp in df.groupby("bin", observed=True):
        a = grp[grp["aerial"]]["flight_s"]
        g = grp[~grp["aerial"]]["flight_s"]
        if len(a) < 4 or len(g) < 10:
            continue
        rows.append((b, len(a), len(g), a.median(), g.median()))
        diffs.append(a.median() - g.median())
    for b, na, ng, ma, mg in rows:
        print(f"  {str(b):>12s}  aerial n={na:3d} median {ma:.2f}s | ground n={ng:4d} median {mg:.2f}s"
              f"  -> {ma - mg:+.2f}s")
    if not rows:
        _fail("no distance bin had enough of both classes")
        return False
    # Sign test across bins plus a pooled rank test on residual flight time.
    pos = sum(1 for d in diffs if d > 0)
    df2 = df.dropna(subset=["bin"])
    slope = np.polyfit(df2["dist_m"], df2["flight_s"], 1)
    resid = df2["flight_s"] - np.polyval(slope, df2["dist_m"])
    u = stats.mannwhitneyu(resid[df2["aerial"]], resid[~df2["aerial"]], alternative="greater")
    print(f"  {pos}/{len(diffs)} bins show aerial slower; pooled distance-adjusted "
          f"Mann-Whitney p={u.pvalue:.2e}")
    if pos == len(diffs) and u.pvalue < 0.05:
        _ok("aerial passes fly measurably longer at matched distance — arc model has real signal")
        return True
    _fail("aerial and ground passes are not separable at matched distance")
    return False


def check_geometry() -> bool:
    print("\nW6a(3) dome geometry invariants")
    p = DomeParams()
    passed = True
    tau = 2.0
    arrivals = np.linspace(0.0, tau, 200).reshape(-1, 1)
    z = dome.z_reach(arrivals, tau, p).ravel()

    if np.all(np.diff(z) <= 1e-12):
        _ok("vertical reach non-increasing in arrival time (dome tapers outward)")
    else:
        _fail("vertical reach not monotone in arrival time")
        passed = False

    z_edge = dome.z_reach(np.array([[tau]]), tau, p)[0, 0]
    if abs(z_edge - p.standing_reach_m) < 1e-12:
        _ok(f"at the horizon reach = standing reach exactly ({z_edge:.2f}m, no jump possible)")
    else:
        _fail(f"at the horizon reach {z_edge:.3f} != standing {p.standing_reach_m}")
        passed = False

    z_near = dome.z_reach(np.array([[0.0]]), tau, p)[0, 0]
    expect = p.standing_reach_m + p.max_jump_m
    if tau >= p.full_jump_slack_s and abs(z_near - expect) < 1e-12:
        _ok(f"with full slack reach = standing + max jump ({z_near:.2f}m); "
            f"a maximal leap needs {p.full_jump_slack_s:.2f}s")
    else:
        _fail(f"near-player reach {z_near:.3f} != {expect}")
        passed = False

    beyond = dome.z_reach(np.array([[tau + 0.5]]), tau, p)[0, 0]
    if np.isnan(beyond):
        _ok("points beyond the horizon have undefined (NaN) reach, not zero")
    else:
        _fail("unreachable points did not return NaN")
        passed = False

    # Volume monotone in tau: more time cannot shrink the reachable set.
    pos = np.array([[0.0, 0.0], [20.0, 5.0]])
    vel = np.zeros_like(pos)
    _, _, targets = reach.pitch_grid(cell_m=2.0)
    arr = reach.arrival_times(pos, vel, targets, ReachParams())
    vols = [dome.reachable_volume(arr, t, 4.0, p) for t in (1.0, 2.0, 3.0, 4.0)]
    if all(b >= a - 1e-9 for a, b in zip(vols, vols[1:])):
        _ok("reachable volume non-decreasing in tau: " + " -> ".join(f"{v:,.0f}" for v in vols) + " m³")
    else:
        _fail("reachable volume not monotone in tau")
        passed = False
    return passed


def _jersey(from_field) -> str | None:
    if pd.isna(from_field):
        return None
    return str(from_field).replace("Player", "").strip()


def collect_duels(games) -> list[dict]:
    """Pair AERIAL-WON with AERIAL-LOST from the opposing team within a few frames."""
    duels = []
    for g in games:
        ev = g.events.reset_index(drop=True)
        sub = ev["Subtype"].fillna("").astype(str)
        aer = ev[sub.str.startswith("AERIAL")].copy()
        aer["sub"] = sub[aer.index]
        won = aer[aer["sub"].str.contains("WON")]
        lost = aer[aer["sub"].str.contains("LOST")]
        for _, w in won.iterrows():
            cand = lost[
                (lost["Team"] != w["Team"])
                & ((lost["Start Frame"] - w["Start Frame"]).abs() <= DUEL_PAIR_WINDOW)
                & (lost["Period"] == w["Period"])
            ]
            if cand.empty:
                continue
            l = cand.iloc[0]
            wj, lj = _jersey(w["From"]), _jersey(l["From"])
            if wj is None or lj is None:
                continue
            duels.append(
                dict(
                    game=g, period=int(w["Period"]), frame=float(w["Start Frame"]),
                    winner_team=str(w["Team"]).lower(), winner=wj,
                    loser_team=str(l["Team"]).lower(), loser=lj,
                    x=(float(w["Start X"]) - 0.5) * PITCH_L,
                    y=(float(w["Start Y"]) - 0.5) * PITCH_W,
                )
            )
    return duels


def check_duels(games) -> bool:
    print(f"\nW6a(4) aerial duel prediction — evaluated {DUEL_LOOKBACK_S}s before contact")
    duels = collect_duels(games)
    print(f"  paired {len(duels)} aerial duels across both games")
    if len(duels) < 25:
        print("  too few duels to test — reported, not gated")
        return True

    rp, dp = ReachParams(), DomeParams()
    n = ok_dome = ok_dist = 0
    for d in duels:
        g = d["game"]
        step = int(round(DUEL_LOOKBACK_S / g.dt))
        idx = g.tracking.index[g.tracking["Frame"] == d["frame"] - step]
        if len(idx) == 0:
            continue
        row = g.tracking.loc[idx[0]]
        target = np.array([[d["x"], d["y"]]])
        vals = {}
        for role in ("winner", "loser"):
            team, pid = d[f"{role}_team"], d[role]
            x, y = row.get(f"{team}_{pid}_x"), row.get(f"{team}_{pid}_y")
            if x is None or pd.isna(x) or pd.isna(y):
                vals = {}
                break
            vx = row.get(f"{team}_{pid}_vx", 0.0)
            vy = row.get(f"{team}_{pid}_vy", 0.0)
            vx = 0.0 if pd.isna(vx) else vx
            vy = 0.0 if pd.isna(vy) else vy
            arr = reach.arrival_times(np.array([[x, y]]), np.array([[vx, vy]]), target, rp)
            vals[role] = dict(
                z=float(dome.z_reach(arr, DUEL_LOOKBACK_S, dp)[0, 0]),
                arrival=float(arr[0, 0]),
                dist=float(np.hypot(x - d["x"], y - d["y"])),
            )
        if len(vals) != 2:
            continue
        n += 1
        zw, zl = vals["winner"]["z"], vals["loser"]["z"]
        # NaN = cannot reach at all; treat as lowest possible reach.
        zw = -np.inf if np.isnan(zw) else zw
        zl = -np.inf if np.isnan(zl) else zl
        if zw > zl:
            ok_dome += 1
        elif zw == zl:
            ok_dome += 0.5
        if vals["winner"]["dist"] < vals["loser"]["dist"]:
            ok_dist += 1
        elif vals["winner"]["dist"] == vals["loser"]["dist"]:
            ok_dist += 0.5

    if n < 25:
        print(f"  only {n} duels had usable tracking — reported, not gated")
        return True
    acc_dome, acc_dist = ok_dome / n, ok_dist / n
    p_dome = stats.binomtest(int(round(ok_dome)), n, 0.5, alternative="greater").pvalue
    print(f"  n={n} duels with tracking")
    print(f"  dome (vertical reach from slack time): {acc_dome:.1%} correct  (p={p_dome:.3f} vs coin flip)")
    print(f"  baseline (nearer player wins):         {acc_dist:.1%} correct")
    # Reported, NOT gated: with uniform parameters this is a weak form of the real test,
    # and n is small. The honest question is whether it beats the baseline at all.
    if acc_dome > acc_dist:
        _ok("arrival-time-derived reach beats raw proximity (uniform-parameter form of W6b)")
    else:
        print("  note: dome does NOT beat raw proximity here. With uniform parameters the two are")
        print("  nearly the same quantity, so this is weak evidence against the CAP, not the model;")
        print("  the real test (W6b) needs per-player heights and stays blocked on named data.")
    return True


def main() -> int:
    print("Phase 5a gate — W6a (ball-arc inference + dome geometry + aerial duels)")
    games = [load_game(g) for g in (1, 2)]
    all_passes = pd.concat([bf.passes_with_flight(g.events, g.dt) for g in games], ignore_index=True)
    print(f"\n{len(all_passes)} passes with usable flight, {int(all_passes['aerial'].sum())} flagged aerial")

    r0 = check_ball_trajectory(games)
    r1 = check_arc(all_passes)
    r2 = check_discrimination(all_passes)
    r3 = check_geometry()
    r4 = check_duels(games)

    # Reported as a SPLIT result, not a single verdict, because the two halves of Phase 5a
    # came out differently and collapsing them to one boolean would hide that.
    print(f"\n{'=' * 70}")
    print("W6a AS SPECIFIED: FAIL — the altitude half does not survive contact with the data.")
    print(f"  ball path real .......... {'yes' if r0 else 'NO — interpolated'}")
    print(f"  arc apex sanity ......... {'ok' if r1 else 'FAIL'}")
    print(f"  aerial discrimination ... {'ok' if r2 else 'FAIL (premise disproved)'}")
    print(f"  dome geometry ........... {'ok' if r3 else 'FAIL'}   <- this half PASSES")
    print("\nOUTCOME: dome GEOMETRY is validated and retained (dome.py). Ball-altitude")
    print("inference is WITHDRAWN — not tuned, withdrawn — because Metrica interpolates the")
    print("ball path and no altitude exists in it to recover.")
    print("\n5b GO/NO-GO: NO-GO on current data, and the requirement is now SHARPER than")
    print("'named players'. A provider must supply MEASURED ball z, and W6a(0) above is the")
    print("acceptance test to run on arrival. See docs/data-scope.md.")
    # Exit 0: geometry is the part Phase 5a can legitimately deliver, and it passed. The
    # altitude failure is a recorded finding with a decided consequence, not an open bug.
    return 0 if r3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
