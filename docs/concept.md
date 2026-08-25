# Concept: Positive/Negative Space & 3D Reachable Volumes

STATUS: in development — this folder (`space_analysis/`) is the live home. Originally
parked as `surplus_value_model/docs/idea-positive-negative-space.md`; that copy is now
historical. Still separate from the surplus-value pipeline: crossover points are
flagged at the bottom, dependency is one-way (this project may READ surplus_value_model
artifacts; nothing in the CPV build depends on this).

## The framing

Borrowed from visual art, applied to the pitch, defined from the DEFENSE's point of view:

- **Whitespace** — the empty pitch: the full field volume before anyone occupies it.
  The canvas. Formally `F ⊂ R³` (105m x 68m x reachable height).
- **Positive space** — the region a player physically occupies *in reach terms*: every
  point they can reach within a time horizon τ, given reaction time, acceleration,
  top speed, and vertical reach. A team's positive space is the union over its players.
- **Negative space** — space no defender can reliably reach within τ. This unifies two
  intuitions: (a) statically, it is simply the complement of the defense's positive
  space; (b) dynamically, it can be *created* — a striker dragging a center-back wide
  carves new negative space in the channel they vacated. Same object, two verbs:
  attackers **enter** negative space (off-ball runs into it) or **create** it
  (movement that reshapes the defensive union).

The offensive question becomes: how much *valuable* negative space exists, who
creates it, and who exploits it. The defensive question is the mirror: who destroys
(re-covers) negative space fastest.

## Formal sketch

### Player reachable volume (the "dome")

For player `p` at time `t` with horizon `τ`, the reachable set is

```
R_p(t, τ) = { (x, y, z) : p can occupy (x, y, z) within τ }
```

Horizontal extent: from a motion model — reaction delay `t_r`, then acceleration-
capped movement (current velocity matters: the set skews in the direction of travel).
Roughly a disc of radius `r(τ)` centered ahead of the player, as in pitch control.

Vertical extent: does NOT grow with τ the way horizontal range does. A jump adds a
fixed `Δz` and needs a plant-and-leap window (~0.3–0.5s). So the reachable height at
a horizontal point depends on *time remaining after arriving there*:

```
z_reach(d) = standing_reach + jump(τ − t_arrive(d))
```

Near the player's current position there is slack time to set and jump — full vertical
reach. At the fringe of horizontal range, arrival consumes the whole horizon — ground-
level reach only. This yields the dome naturally: a wide base tapering to a cap, with
the cap height set by stature + leap (physical attributes) and the base radius by
speed/acceleration (kinematic attributes). Verticality is therefore not a separate
mechanic — aerial dominance is just the *top slice* of positive space, and an aerial
duel is a contest between overlapping dome caps at the ball's arrival point.

**BUILT AND VALIDATED (Phase 5a, `src/dome.py`).** The geometry above is implemented and
its invariants pass: reach is monotone in arrival time, equals standing reach exactly at
the horizon (no slack, no jump), reaches the full cap given enough slack, and reachable
VOLUME is monotone in τ. The jump model is ballistic rather than a lookup — flight time
`t` buys height `g·t²/2`, so a maximal leap needs ~0.60s of slack (0.25s plant + 0.35s
rise). One pleasing consequence of the union being a MAX over heights rather than a
probabilistic union: the 3D positive space needs no independence assumption, unlike the
2D coverage field.

**What Phase 5a could NOT establish**, and it matters: with uniform parameters every
vertical difference between two players reduces to slack time, so the dome adds nothing
over plain proximity. Measured on 46 paired aerial duels, vertical reach picked the
winner 66.3% of the time versus 68.5% for "whoever is nearer" — i.e. no benefit. That is
not evidence against the dome; it is what a uniform cap MUST produce, since identical
caps cannot differentiate players. **The value of verticality is entirely contingent on
per-player attributes**, which is exactly what W6b was written to test and what remains
blocked on named data. Until then the dome is validated machinery with no demonstrated
payoff, and should be described that way.

### Negative space, precisely

Defense `D`'s covered volume is `C_D(t, τ) = ∪_{p ∈ D} R_p(t, τ)`. Negative space is

```
N(t, τ) = F \ C_D(t, τ)      (deterministic version)
```

Better, probabilistic: with arrival-time uncertainty (as in pitch control), each point
gets `P(some defender reaches it within τ)`, and negative space is the region where
that probability is below a threshold θ. The choice of τ and θ is a modeling knob —
"space no defender reaches in 2s with 80% confidence" — and should be stated, not hidden.

### Creation vs. entry (the two verbs)

- **Entry**: attacker `a` moves so that `R_a` intersects `N` — occupying space the
  defense cannot contest. Off-ball runs into the channel are entries.
- **Creation**: an attacker's movement changes the *defense's* union — dragging a
  defender shifts their dome, opening a hole. Measured as the change in `N` between
  consecutive times, attributed to the movements that caused the defensive displacement:
  `ΔN = N(t+δ) − N(t)`, decomposed by which defender moved and who pulled them.
- Symmetrically, defensive **destruction**: recovering runs that shrink `N`.

### Value weighting (the crucial refinement)

Raw volume is misleading — acres of negative space in your own half are worthless.
Weight each point of negative space by a value surface (xT/EPV): *valuable negative
space* is the integral of value over `N`. A striker who creates 2m² behind the back
line creates more than a fullback who creates 40m² near halfway.

## What could be measured (if ever built)

- Valuable negative space created per 90 (off-ball contribution, invisible to on-ball stats)
- Negative space entered per 90 (run timing/intelligence)
- Defensive destruction rate (space re-covered after disruption)
- Aerial margin: dome-cap overlap advantage at cross/long-ball arrival points
- Team-level: average valuable negative space conceded (defensive shape quality)

## Data requirements (honest accounting)

- **Kinematic attributes** (dome base): derivable per player from tracking history —
  observed top speed, acceleration percentiles, reaction proxies. Self-contained.
- **Vertical attributes** (dome cap): NOT in tracking data — needs external sources
  (height/reach from bios; jump/aerial data from e.g. FIFA/FM attribute databases).
  Requires named players → StatsBomb-scale data, not anonymized Metrica.
- **Ball altitude**: 3D pass evaluation needs ball z. Metrica CSV games have x,y only.
  Some providers (FIFA EPTS, SkillCorner) carry ball z. Without it, verticality can
  only activate on inferred aerial events (long balls, crosses) rather than continuously.
  **UPDATE (Phase 5a, 2026-07-31) — this paragraph was too optimistic and is now
  corrected by measurement.** The plan to infer altitude from flight time via a
  ballistic arc (`apex = g·T²/8`) was built and TESTED, and it fails on Metrica for a
  reason that no modelling can fix: **the tracked ball path is interpolated, not
  measured** — a straight line at constant speed drawn between annotated touches
  (speed-retention ratio 1.000 at p10/p50/p90; 3cm median deviation from the straight
  line; 62% of passes under 5cm). A real rolling ball decelerates and a real struck ball
  curves. Neither happens. So there is no curvature or deceleration signal, and no
  altitude to recover at any level of cleverness. The falsifiable prediction that aerial
  passes fly LONGER at matched distance failed decisively (pooled p=0.999 in the wrong
  direction) — and the premise was wrong regardless, since lofted balls are struck hard
  while ground passes roll gently, making aerial balls FASTER over equal distance.
  Ball-altitude inference is **withdrawn, not tuned**. Verticality therefore cannot
  activate at all on Metrica, continuously or on flagged events. Requirement for any
  future provider is now sharper than "carries ball z": the z must be MEASURED, and the
  acceptance test is `validate_phase5.check_ball_trajectory` — if the ball path is
  straight at constant speed between touches, the provider is interpolating and any z
  channel may be synthetic too.

## Relation to prior work

Pitch control (Spearman) covers the 2D probabilistic-reachability core. Fernández &
Bornn's "Wide Open Spaces" formalizes space creation/occupation value in 2D. Off-ball
scoring opportunity (OBSO) values space by scoring probability. The genuinely
underexplored pieces here: (1) full-3D reachable volumes with the dome geometry and
attribute-driven caps, (2) the positive/negative decomposition as a unified language
for creation, entry, and destruction, (3) per-player attribute-parameterized domes
(public models mostly assume uniform physical parameters).

## Crossover points with the surplus-value project

- Our pitch control model is the 2D uniform-parameter special case of the dome.
- Value-weighted negative space uses the same xT surface already built.
- The parked "off-ball run counterfactual" (NOTES) is exactly "which negative space
  *should* have been entered" — this framework is its natural language.
- Attribute-parameterized domes would sharpen P(complete) in CPV (a slow defender's
  nominal position overstates their real coverage).
