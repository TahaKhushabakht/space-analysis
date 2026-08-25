"""Ball altitude inference for Metrica — **WITHDRAWN. The apex model does not work here.**

READ THIS BEFORE USING `apex_m` OR `ball_height_at_frame`. Phase 5a tested the arc model
and it failed, then the root cause was found and it is fatal on this data:

**Metrica's ball trajectory between events is INTERPOLATED, not measured.** Measured over
game 1's passes: the ball's speed-retention ratio is 1.000 at the 10th, 50th and 90th
percentiles (47% of passes exactly 1.000), and the path deviates from the straight line
joining its endpoints by a median of 3cm (62% under 5cm). A real rolling ball decelerates
and a real struck ball curves; neither happens here. The path is a straight line at
constant speed drawn between two annotated endpoints.

Consequences, all of them structural rather than fixable:
  * There is NO curvature or deceleration signal, so nothing in the data distinguishes an
    airborne ball from a ground ball by physics.
  * `apex = g·T²/8` applied to this data describes the interpolation, not the ball.
  * The one falsifiable prediction — that aerial passes fly LONGER at matched distance —
    failed decisively (2/7 distance bins, pooled p = 0.999 in the wrong direction). The
    premise was wrong anyway: lofted balls are struck hard while ground passes roll
    gently, so aerial balls are FASTER over the same distance, not slower.

WHAT SURVIVES. Flight TIME is real (both endpoints are collector-annotated), and the
AERIAL FLAG is real — it comes from the collector's downstream HEAD/AERIAL annotations,
not from physics, so it still identifies which passes arrived in the air. Use
`passes_with_flight` for flagging. Do not use the apex numbers for anything.

WHAT THIS CHANGES DOWNSTREAM. Gap C is not "imperfect on Metrica", it is UNRESOLVABLE on
Metrica. Any future provider must supply MEASURED ball z, and the check above is the one
to run on arrival: if the ball path is straight at constant speed between touches, the
provider is interpolating and the z channel (if any) may be synthetic too.

---
Original intent, retained for context:

Metrica's CSVs give the ball's ground position and nothing else, so height must be
inferred. For a ball leaving and returning to ground level over flight time T, the
drag-free symmetric arc gives height as a function of the fraction f of the flight
elapsed, and an apex at the midpoint:

Metrica's CSVs give the ball's ground position and nothing else, so height must be
inferred. For a ball leaving and returning to ground level over flight time T, the
drag-free symmetric arc gives height as a function of the fraction f of the flight
elapsed, and an apex at the midpoint:

    z(f) = (g·T²/2)·f·(1 − f)          apex = g·T²/8

FIRST ORDER ONLY, and the violations matter. Drag is ignored; driven balls, bounces,
deflections and mid-flight contests all break the ground-to-ground assumption — and a
contested aerial, which is exactly what this is used for, is the case most likely to end
mid-flight. So the inference is applied ONLY to passes independently flagged as airborne
(below), never to every pass, and W6a checks the resulting apex distribution rather than
trusting it. That check is what caught the problem described at the top of this file.

FLAGGING AIRBORNE PASSES without a height channel. Metrica has no "aerial" flag on
passes, but the event stream betrays it two ways:
  1. The pass's own subtype is CROSS or GOAL KICK — airborne by nature.
  2. The NEXT event involves HEAD or AERIAL. You cannot head a ball that is on the
     ground, so a pass followed by a headed action or an aerial duel arrived in the air.
Rule 2 is the useful one: it uses the collector's own downstream annotation as an
INDEPENDENT signal, rather than inferring aerial-ness from the same flight time the arc
model then consumes. That independence is what makes W6a's discrimination test
meaningful instead of circular.

Measured separation on game 1: flagged passes run 1.92s median flight over 26.6m, ground
passes 1.20s over 14.0m — but they also differ in DISTANCE, so any honest comparison has
to be made at matched distance (W6a does this).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.81

PITCH_L, PITCH_W = 105.0, 68.0

# Flight times outside this band are almost certainly mis-parsed rather than real passes
# (instant tap-ons at one end, tracking gaps at the other).
MIN_FLIGHT_S, MAX_FLIGHT_S = 0.2, 6.0
MIN_DIST_M = 3.0


def apex_m(flight_time_s):
    """Apex of a ground-to-ground drag-free arc: g·T²/8."""
    T = np.asarray(flight_time_s, dtype=float)
    return G * T**2 / 8.0


def height_at(fraction: float | np.ndarray, flight_time_s: float | np.ndarray):
    """Ball height at `fraction` of the way through the flight (0 = struck, 1 = arrival)."""
    f = np.asarray(fraction, dtype=float)
    T = np.asarray(flight_time_s, dtype=float)
    return (G * T**2 / 2.0) * f * (1.0 - f)


def flight_time_for_apex(apex_height_m: float) -> float:
    """Inverse of `apex_m` — the flight time an arc of this apex implies."""
    return float(np.sqrt(8.0 * apex_height_m / G))


def passes_with_flight(events: pd.DataFrame, dt: float) -> pd.DataFrame:
    """All passes with flight time, ground distance, an airborne flag and inferred apex.

    Returns a frame with columns: start_frame, end_frame, team, flight_s, dist_m,
    speed_ms, aerial (bool), apex_m (NaN for non-flagged passes — deliberately, since the
    arc model is not valid for a rolling ball).
    """
    ev = events.reset_index(drop=True)
    sub = ev["Subtype"].fillna("").astype(str)
    next_sub = sub.shift(-1).fillna("")

    is_pass = ev["Type"] == "PASS"
    own_aerial = sub.str.contains("CROSS|GOAL KICK", regex=True)
    next_aerial = next_sub.str.contains("HEAD|AERIAL", regex=True)
    aerial = is_pass & (own_aerial | next_aerial)

    d = ev[is_pass].copy()
    d = d.dropna(subset=["Start Frame", "End Frame", "Start X", "Start Y", "End X", "End Y"])

    flight = (d["End Frame"] - d["Start Frame"]) * dt
    dist = np.hypot((d["End X"] - d["Start X"]) * PITCH_L, (d["End Y"] - d["Start Y"]) * PITCH_W)

    out = pd.DataFrame(
        {
            "start_frame": d["Start Frame"].to_numpy(float),
            "end_frame": d["End Frame"].to_numpy(float),
            "team": d["Team"].astype("string").str.lower().to_numpy(),
            "flight_s": flight.to_numpy(float),
            "dist_m": dist.to_numpy(float),
            "aerial": aerial.loc[d.index].to_numpy(bool),
        }
    )
    out["speed_ms"] = out["dist_m"] / out["flight_s"].replace(0, np.nan)
    ok = (
        out["flight_s"].between(MIN_FLIGHT_S, MAX_FLIGHT_S)
        & (out["dist_m"] >= MIN_DIST_M)
    )
    out = out[ok].reset_index(drop=True)
    out["apex_m"] = np.where(out["aerial"], apex_m(out["flight_s"]), np.nan)
    return out


def ball_height_at_frame(pass_row, frame: float, dt: float) -> float:
    """Inferred ball height at a given frame during a flagged aerial pass. 0 if not aerial."""
    if not bool(pass_row["aerial"]):
        return 0.0
    T = float(pass_row["flight_s"])
    if T <= 0:
        return 0.0
    f = (frame - float(pass_row["start_frame"])) * dt / T
    if f < 0.0 or f > 1.0:
        return 0.0
    return float(height_at(f, T))
