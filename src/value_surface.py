"""Value surface adapter: season-scale xT, looked up from metric tracking coords.

Loads `../surplus_value_model/output/sb_xt.npy` READ-ONLY (reuse policy, NOTES.md):
a (12, 16) = (ny, nx) Expected Threat grid built from 29,675 StatsBomb passes at season
scale, normalized coords, EVERY attack oriented toward x=1. Median 212 actions/zone —
far better estimated than anything rebuildable from 2 Metrica games (~20/zone). Value
is a property of location, not provider, so carrying it across is legitimate; the
caveat carried with it is that the surface is ~63% Leverkusen passes (league average is
Leverkusen-tinted).

COORDINATE CHAIN (the part that must not be wrong):
  metric centre-origin (x_m, y_m)
    --orient_xy(d)-->  attacking frame, attack toward +x   (180° rotation = sign flip)
    --/L, /W + 0.5-->  normalized [0,1]^2, attack toward x=1
    --zone index-->    (row, col) into the grid
Getting direction d wrong flips the surface end-for-end and silently inverts every
value-weighted number — the failure Phase 0's dual-source direction check guards.

Y-SYMMETRIZATION (default ON, measured before deciding): the surface's y-asymmetry is
negligible everywhere except the last column (goal mouth), where the two central cells
disagree 0.259 vs 0.397 — exactly the small-sample box noise the main project's V3
validation flagged ("small-sample noise in box goal_prob"). Max asymmetry outside the
final two columns is < 0.008. Averaging the surface with its y-flip (a) removes noise
football gives no reason to believe (first-order, threat is y-symmetric), and (b) makes
the lookup INVARIANT to the Metrica-vs-StatsBomb y-axis convention, which would
otherwise need to be established across two providers' documentation to be trusted.
Two birds, documented, reversible via symmetrize=False.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PITCH_L, PITCH_W = 105.0, 68.0

DEFAULT_XT_PATH = (
    Path(__file__).resolve().parents[2] / "surplus_value_model" / "output" / "sb_xt.npy"
)


class ValueSurface:
    def __init__(self, path: Path | str | None = None, symmetrize: bool = True):
        path = Path(path) if path is not None else DEFAULT_XT_PATH
        xt = np.load(path)
        if xt.ndim != 2:
            raise ValueError(f"expected 2D xT grid, got shape {xt.shape}")
        if symmetrize:
            xt = 0.5 * (xt + xt[::-1, :])
        self.xt = xt
        self.ny, self.nx = xt.shape

        # Monotonicity guard: mean value must rise toward the attack end (the main
        # project's V3 property). If this fails the file isn't what we think it is.
        col_means = xt.mean(axis=0)
        if not (np.diff(col_means) > 0).all():
            raise ValueError("xT column means not monotone toward attack end — wrong artifact?")

    def value(self, x_m, y_m, direction: int) -> np.ndarray:
        """xT at metric centre-origin coords, for a team attacking `direction`.

        Vectorized over arrays. Points off the pitch clamp to the edge zone.
        """
        x_m, y_m = np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float)
        ox, oy = x_m * direction, y_m * direction  # orient_xy inlined for array speed
        xn = np.clip(ox / PITCH_L + 0.5, 0.0, 1.0 - 1e-9)
        yn = np.clip(oy / PITCH_W + 0.5, 0.0, 1.0 - 1e-9)
        col = (xn * self.nx).astype(int)
        row = (yn * self.ny).astype(int)
        return self.xt[row, col]

    def value_at_targets(self, targets: np.ndarray, direction: int) -> np.ndarray:
        """xT for an (n, 2) array of metric target points. Returns (n,)."""
        return self.value(targets[:, 0], targets[:, 1], direction)
