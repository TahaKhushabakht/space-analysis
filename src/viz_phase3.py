"""Phase 3 eye test (gate W4b): render the largest creation and destruction events.

docs/validation.md asks for the 10 highest-VNS creation events and 10 highest destruction
events inspected against the tracking, 8/10 defensible, with every failure logged.

Selection here is by MAGNITUDE, not at random — that is deliberate and different from the
W1/W2 eye tests. The question is not "does this look right on average" (the invariants
and the accounting identity already answer that) but "when the model claims a lot of
valuable space just opened, did something recognisable actually happen?" Extremes are
where a decomposition is most likely to be embarrassing, so extremes are what get looked
at. The cost is that these frames say nothing about typical behaviour.

Run:  python src/viz_phase3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dynamics as dyn  # noqa: E402
import negative_space as ns  # noqa: E402
import reach  # noqa: E402
from loader import load_game  # noqa: E402
from orientation import orient_xy  # noqa: E402
from reach import ReachParams  # noqa: E402
from value_surface import ValueSurface  # noqa: E402

SEED = 20260731
TAU, THETA = 2.0, 0.3
CELL_M = 2.0
DELTA_S = 2.0  # raised from 0.6s — see validate_phase3.py for the coherence measurements
SCAN_N = 900
TOP_N = 8
OUT = Path(__file__).resolve().parents[1] / "output" / "figures"


def scan(game, surface, params, targets, cell_area, grid_shape):
    poss = ns.possession_by_frame(game.events, game.tracking)
    an = ns.analysable_frames(game.events, game.tracking)
    step = int(round(DELTA_S / game.dt))
    rng = np.random.default_rng(SEED + game.game_id)
    cand = np.flatnonzero(an)
    cand = cand[cand + step < len(game.tracking)]
    sel = np.sort(rng.choice(cand, size=min(SCAN_N, len(cand)), replace=False))

    vcache, events = {}, []
    for i in sel:
        j = i + step
        if not an[j] or poss[i] != poss[j] or poss[i] is None:
            continue
        r0, r1 = game.tracking.iloc[i], game.tracking.iloc[j]
        if int(r0["Period"]) != int(r1["Period"]):
            continue
        att = poss[i]
        d = game.direction(att, int(r0["Period"]))
        if d not in vcache:
            vcache[d] = surface.value_at_targets(targets, d)
        r = dyn.frame_dynamics(r0, r1, att, game.player_ids, d, TAU, THETA, targets,
                               vcache[d], cell_area, DELTA_S, params,
                               grid_shape=grid_shape)
        if r is None:
            continue
        events.append(dict(game=game.game_id, i=i, j=j, att=att, d=d, res=r))
    return events


def render(ax, game, ev, params, targets, grid_shape, gx, gy, mode):
    r0, r1 = game.tracking.iloc[ev["i"]], game.tracking.iloc[ev["j"]]
    att, d = ev["att"], ev["d"]
    dfn = game.defending_team(att)

    p0, v0, ids0 = dyn._team_pv_ids(r0, dfn, game.player_ids[dfn])
    p1, v1, ids1 = dyn._team_pv_ids(r1, dfn, game.player_ids[dfn])
    common = [k for k in ids0 if k in set(ids1)]
    k0 = [ids0.index(k) for k in common]
    k1 = [ids1.index(k) for k in common]

    l0 = dyn.log_uncovered(reach.arrival_times(p0[k0], v0[k0], targets, params), TAU, params)
    l1 = dyn.log_uncovered(reach.arrival_times(p1[k1], v1[k1], targets, params), TAU, params)
    thr = np.log(1 - THETA)
    o0, o1 = l0.sum(1) > thr, l1.sum(1) > thr
    opened = ((~o0) & o1).reshape(grid_shape)
    closed = (o0 & (~o1)).reshape(grid_shape)

    if d == -1:
        opened, closed = opened[::-1, ::-1], closed[::-1, ::-1]

    # pitch
    hl, hw = 52.5, 34.0
    ax.plot([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw], color="#444", lw=1)
    ax.plot([0, 0], [-hw, hw], color="#444", lw=1)
    ax.add_patch(plt.Circle((0, 0), 9.15, color="#444", fill=False, lw=1))
    for s in (-1, 1):
        for dep, wid in ((16.5, 40.32), (5.5, 18.32)):
            ax.plot([s * hl, s * (hl - dep), s * (hl - dep), s * hl],
                    [-wid / 2, -wid / 2, wid / 2, wid / 2], color="#444", lw=1)
        ax.plot([s * hl, s * hl], [-3.66, 3.66], color="#444", lw=2.5)

    ax.contourf(gx, gy, opened.astype(float), levels=[0.5, 1.5], colors=["#2eb872"], alpha=0.55)
    ax.contourf(gx, gy, closed.astype(float), levels=[0.5, 1.5], colors=["#9b3fb5"], alpha=0.55)

    # defenders: hollow at t0 -> filled at t1, arrow shows the movement that did it
    ox0, oy0 = orient_xy(p0[k0][:, 0], p0[k0][:, 1], d)
    ox1, oy1 = orient_xy(p1[k1][:, 0], p1[k1][:, 1], d)
    ax.scatter(ox0, oy0, facecolors="none", edgecolors="#c1121f", s=34, lw=1.2, zorder=3)
    ax.scatter(ox1, oy1, c="#c1121f", s=34, zorder=4, edgecolors="white", lw=0.5)
    for a, b, c, e in zip(ox0, oy0, ox1, oy1):
        ax.annotate("", xy=(c, e), xytext=(a, b),
                    arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.1, alpha=0.9))

    ap, av, _ = dyn._team_pv_ids(r1, att, game.player_ids[att])
    if len(ap):
        axo, ayo = orient_xy(ap[:, 0], ap[:, 1], d)
        ax.scatter(axo, ayo, c="#0b6fa4", s=30, marker="s", zorder=4, edgecolors="white", lw=0.5)
    bx, by = r1.get("ball_x"), r1.get("ball_y")
    if not (np.isnan(bx) or np.isnan(by)):
        obx, oby = orient_xy(bx, by, d)
        ax.scatter([obx], [oby], c="white", s=40, zorder=5, edgecolors="black", lw=1.1)

    res = ev["res"]
    key = res.created_by_defender if mode == "created" else res.destroyed_by_defender
    top = max(key.items(), key=lambda kv: kv[1]) if key else ("-", 0.0)
    ax.set_title(
        f"G{ev['game']} f{ev['i']} · {att} attacking →\n"
        f"largest opened region {res.largest_opened_xt:.2f} xT·m² "
        f"({res.coherence:.0%} of opened cells) · top {mode[:4]}: #{top[0]} ({top[1]:.2f})",
        fontsize=8,
    )
    ax.set_xlim(-hl - 2, hl + 2)
    ax.set_ylim(-hw - 2, hw + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    params = ReachParams()
    surface = ValueSurface()
    gx, gy, targets = reach.pitch_grid(cell_m=CELL_M)
    grid_shape, cell_area = gx.shape, CELL_M**2

    games = {g: load_game(g) for g in (1, 2)}
    events = []
    for g in games.values():
        events += scan(g, surface, params, targets, cell_area, grid_shape)
    print(f"scanned {len(events)} windows across both games")

    # Ranked by the LARGEST CONNECTED region, not total flipped area. Scattered jitter and
    # one coherent channel can carry identical totals; only the second is "a space opened".
    for mode, keyf in (("created", lambda e: e["res"].largest_opened_xt),
                       ("destroyed", lambda e: e["res"].largest_closed_xt)):
        top = sorted(events, key=keyf, reverse=True)[:TOP_N]
        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        for ax, ev in zip(axes.ravel(), top):
            render(ax, games[ev["game"]], ev, params, targets, grid_shape, gx, gy, mode)
        fig.suptitle(
            f"Phase 3 eye test — top {TOP_N} {mode.upper()} events (δ={DELTA_S}s, τ={TAU}, θ={THETA})\n"
            "green = space that opened · purple = space re-covered · hollow→filled red = defender movement",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        out = OUT / f"phase3_{mode}_events.png"
        fig.savefig(out, dpi=105, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
