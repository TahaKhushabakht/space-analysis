"""Phase 1 eye test: render negative space on randomly sampled frames (gate W2).

The mechanical half of W2 lives in validate_phase1.py. This is the other half — does the
negative space LOOK like football? docs/validation.md asks for 10 randomly sampled frames
(explicitly no cherry-picking) and 8/10 defensible on inspection:
  - channels behind a stretched back line show up as negative space
  - a compact block kills central negative space
  - the GK's dome accounts for the deep zone in front of goal

Frames are drawn with the SAME seed and the same analysable-frame filter as
validate_phase1, so the pictures show the frames the invariants were checked on.

Every panel is oriented so the team IN POSSESSION attacks toward +x (right). That makes
panels comparable at a glance and doubles as a visual check on the Phase 0 orientation
code: if it were wrong, teams would be attacking the wrong way in half the panels.

Run:  python src/viz_phase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import negative_space as ns  # noqa: E402
import reach  # noqa: E402
from loader import load_game  # noqa: E402
from orientation import orient_xy  # noqa: E402
from reach import ReachParams  # noqa: E402

SEED = 20260730  # same as validate_phase1
N_PANELS = 10
TAU = 2.0
THETA = 0.3
CELL_M = 1.0

PITCH_L, PITCH_W = 105.0, 68.0
OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "figures"


def draw_pitch(ax, lw=1.0, color="#3a3a3a"):
    """Standard pitch markings in centre-origin metric coords."""
    hl, hw = PITCH_L / 2, PITCH_W / 2
    ax.plot([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw], color=color, lw=lw)
    ax.plot([0, 0], [-hw, hw], color=color, lw=lw)
    circle = plt.Circle((0, 0), 9.15, color=color, fill=False, lw=lw)
    ax.add_patch(circle)
    for sign in (-1, 1):
        # penalty area (16.5m deep, 40.32m wide) and six-yard box (5.5m, 18.32m)
        for depth, width in ((16.5, 40.32), (5.5, 18.32)):
            x0 = sign * hl
            x1 = sign * (hl - depth)
            ax.plot([x0, x1, x1, x0], [-width / 2, -width / 2, width / 2, width / 2], color=color, lw=lw)
        ax.plot([sign * (hl - 11)], [0], marker=".", color=color, ms=2)
        # goal
        ax.plot([sign * hl, sign * hl], [-3.66, 3.66], color=color, lw=lw * 2.5)
    ax.set_xlim(-hl - 2, hl + 2)
    ax.set_ylim(-hw - 2, hw + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def render_frame(ax, game, fi, attacking, params, targets, grid_shape, gx, gy):
    """One panel: negative space against the defending team, oriented attack -> +x."""
    row = game.tracking.loc[fi]
    defending = game.defending_team(attacking)
    period = int(row["Period"])
    d = game.direction(attacking, period)

    n = ns.negative_space(
        row, defending, game.player_ids, TAU, THETA, grid_shape, targets, CELL_M**2, params
    )

    draw_pitch(ax)

    # Orienting the FIELD means flipping both axes, which for a regular centred grid is
    # equivalent to reversing both array axes — cheaper and exact.
    cov = n.coverage if d == 1 else n.coverage[::-1, ::-1]
    mask = n.mask if d == 1 else n.mask[::-1, ::-1]

    ax.contourf(gx, gy, cov, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r", alpha=0.55, zorder=0)
    # Negative space itself, outlined so the theta boundary is unambiguous.
    ax.contour(gx, gy, mask.astype(float), levels=[0.5], colors="#0b2a4a", linewidths=1.4, zorder=1)

    for team, color, marker in ((defending, "#c1121f", "o"), (attacking, "#0b6fa4", "s")):
        pos, vel = reach.team_positions_velocities(row, team, game.player_ids[team])
        if len(pos) == 0:
            continue
        ox, oy = orient_xy(pos[:, 0], pos[:, 1], d)
        ovx, ovy = orient_xy(vel[:, 0], vel[:, 1], d)
        ax.scatter(ox, oy, c=color, s=26, marker=marker, zorder=3, edgecolors="white", linewidths=0.5)
        ax.quiver(ox, oy, ovx, ovy, color=color, angles="xy", scale_units="xy", scale=3.0,
                  width=0.004, zorder=2, alpha=0.8)

    bx, by = row.get("ball_x"), row.get("ball_y")
    if not (np.isnan(bx) or np.isnan(by)):
        obx, oby = orient_xy(bx, by, d)
        ax.scatter([obx], [oby], c="white", s=42, marker="o", zorder=4, edgecolors="black", linewidths=1.1)

    ax.set_title(
        f"frame {fi}  P{period}  {attacking} attacking →\n"
        f"|N| = {n.area_m2:,.0f} m² ({n.fraction:.0%} of pitch)",
        fontsize=8,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = ReachParams()
    gx, gy, targets = reach.pitch_grid(cell_m=CELL_M)
    grid_shape = gx.shape

    for game_id in (1, 2):
        game = load_game(game_id)
        rng = np.random.default_rng(SEED + game_id)
        mask = ns.analysable_frames(game.events, game.tracking)
        poss = ns.possession_by_frame(game.events, game.tracking)
        idx = np.flatnonzero(mask)
        frames = np.sort(rng.choice(idx, size=min(N_PANELS, len(idx)), replace=False))

        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        for ax, fi in zip(axes.ravel(), frames):
            render_frame(ax, game, fi, poss[fi], params, targets, grid_shape, gx, gy)
        fig.suptitle(
            f"Negative space — Metrica Sample Game {game_id}  "
            f"(tau={TAU}s, theta={THETA}, {CELL_M}m grid)\n"
            "green = uncovered, red = covered by defense; navy outline = negative space boundary; "
            "red circles = defenders, blue squares = attackers",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = OUT_DIR / f"phase1_negative_space_game{game_id}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")
        print(f"  frames: {list(frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
