"""Build the self-contained space explorer (one HTML file per game).

Precomputes per-frame fields for a handful of shot-ending passages, quantises them, and
embeds everything in `explorer_template.html`. No server, no dependencies at view time —
open the file and scrub.

WHAT SHIPS TO THE BROWSER, and why it is split that way:
  value    uint8 per cell, ONE grid per passage (xT depends only on attack direction)
  coverage uint8 per cell per frame (P * 255)
  exploit. ONE BIT per cell per frame (numpy packbits), the offside mask
The browser computes `h = v·(1 − 2C)` itself, and differences against a pinned frame by
subtracting two h fields. That is the reason coverage is shipped rather than a
precomputed relief: otherwise every (offside on/off) x (pinned t0) x (t) combination
would need its own precomputed field. Bit-packing the mask keeps the payload ~8x smaller
than a byte-per-cell would.

QUANTISATION ERROR is bounded and stated: coverage to 1/255 (~0.4% of its range) and
value to vmax/255. Both are far below the modelling uncertainty they represent — this is
a viewer, not a measurement path, and no number here feeds a gate.

Run:  python src/build_explorer.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reach  # noqa: E402
import relief as relief_mod  # noqa: E402
from loader import load_game  # noqa: E402
from passages import extract_shot_passages  # noqa: E402
from reach import ReachParams  # noqa: E402
from value_surface import ValueSurface  # noqa: E402

TAU = 2.0
TAU_RUN = 4.0
CELL_M = 2.0
WINDOW_S = 8.0
SAMPLE_HZ = 5.0
MAX_PASSAGES = 10

OUT_DIR = Path(__file__).resolve().parents[1] / "output"
TEMPLATE = Path(__file__).resolve().parent / "explorer_template.html"


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _to_canvas(field: np.ndarray, direction: int) -> np.ndarray:
    """Grid orientation -> canvas orientation (attack right, row 0 = top of pitch).

    reach.pitch_grid rows run from lowest y upward. Orienting the attack toward +x for
    direction -1 is a 180-degree rotation = reversing both axes. Then row order is
    flipped so row 0 is the TOP of the pitch, which is what a canvas expects.
    """
    if direction == -1:
        field = field[::-1, ::-1]
    return field[::-1, :]


def build_game(game_id: int) -> Path:
    game = load_game(game_id)
    surface = ValueSurface()
    params = ReachParams()
    gx, gy, targets = reach.pitch_grid(cell_m=CELL_M)
    grid_shape = gx.shape
    ny, nx = grid_shape

    passages = extract_shot_passages(
        game, window_s=WINDOW_S, sample_hz=SAMPLE_HZ, max_passages=MAX_PASSAGES
    )
    print(f"game {game_id}: {len(passages)} passages")

    value_scale = float(surface.xt.max()) / 255.0
    out_passages, h_all, d_all = [], [], []

    for p in passages:
        vgrid = relief_mod.value_grid(surface, targets, p.direction, grid_shape)
        vq = np.clip(np.round(_to_canvas(vgrid, p.direction) / value_scale), 0, 255).astype(np.uint8)

        cov_chunks, expl_chunks, frames, hs = [], [], [], []
        for fi in p.frame_indices:
            fr = relief_mod.compute_frame(
                game.tracking.iloc[fi], p.attacking_team, game.player_ids, p.direction,
                TAU, targets, grid_shape, params, tau_run=TAU_RUN,
            )
            if fr is None:
                continue
            cov = _to_canvas(fr.coverage, p.direction)
            exp = _to_canvas(fr.exploitable, p.direction)
            cov_chunks.append(np.clip(np.round(cov * 255), 0, 255).astype(np.uint8).ravel())
            expl_chunks.append(np.packbits(exp.ravel()))

            h = relief_mod.relief(cov, vq.astype(float) * value_scale)
            hs.append(h)

            row = game.tracking.iloc[fi]
            frames.append(
                dict(
                    t=round(float(row["Time [s]"]), 2),
                    ox=round(fr.offside_x, 2),
                    a=[[round(float(x), 2), round(float(y), 2)] for x, y in fr.att_xy],
                    d=[[round(float(x), 2), round(float(y), 2)] for x, y in fr.def_xy],
                    b=None if fr.ball_xy is None else [round(fr.ball_xy[0], 2), round(fr.ball_xy[1], 2)],
                )
            )
        if len(frames) < 5:
            continue

        hs = np.array(hs)
        h_all.append(np.abs(hs))
        d_all.append(np.abs(hs - hs[0]))

        out_passages.append(
            dict(
                label=p.label, period=p.period, attacking=p.attacking_team,
                direction=p.direction, frames=frames,
                value_b64=_b64(vq.ravel()),
                cov_b64=_b64(np.concatenate(cov_chunks)),
                expl_b64=_b64(np.concatenate(expl_chunks)),
            )
        )

    # Colour scales from high percentiles, so a single extreme cell doesn't wash out the
    # ramp for every other frame.
    h_scale = float(np.percentile(np.concatenate([a.ravel() for a in h_all]), 99.5))
    d_scale = float(np.percentile(np.concatenate([a.ravel() for a in d_all]), 99.5))

    payload = dict(
        game=game_id,
        grid=dict(nx=nx, ny=ny, cell_m=CELL_M),
        value_scale=value_scale,
        h_scale=max(h_scale, 1e-6),
        d_scale=max(d_scale, 1e-6),
        passages=out_passages,
    )

    html = TEMPLATE.read_text(encoding="utf-8")
    html = (
        html.replace("__TITLE__", f"Space Explorer — Metrica Sample Game {game_id}")
        .replace(
            "__SUBTITLE__",
            f"Relief of valuable space over time · {len(out_passages)} passages ending in a shot · "
            f"{CELL_M:g}m grid · inspection tool, not a validation gate",
        )
        .replace("__TAU__", f"{TAU:g}")
        .replace("__TAURUN__", f"{TAU_RUN:g}")
        .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"space_explorer_game{game_id}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  wrote {out}  ({out.stat().st_size / 1e6:.2f} MB, h_scale={h_scale:.4f}, d_scale={d_scale:.4f})")
    return out


def main() -> int:
    for gid in (1, 2):
        build_game(gid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
