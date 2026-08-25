# Math & Algorithms in space_analysis — Context and Usage

LIVING DOC: one entry per mathematical idea or algorithm, saying what it is, where it
is used (file/function + validation gate), why it was chosen, and its caveats. Update
when a phase adds or retires one. Mirrors surplus_value_model/docs/math-concepts.md.
Entries are grouped: IN USE (built and gated, Phases 0–2), SCHEDULED (Phases 3–5),
IDEAS (parked docs). Move an entry between groups when its phase lands; renumber
sequentially.

---

## In use

### 1. Motion model: reaction delay + constant-speed travel
**What.** A player continues at current velocity `v` for reaction time `t_r`, then moves
straight-line at `max_speed` s. Arrival time at target `x`:
`t(x) = t_r + ‖x − (p + v·t_r)‖ / max_speed`.
**Where.** `reach.arrival_times` (heart of everything); inherited unchanged from the
frozen oracle `pitch_control_ref._time_to_intercept` so gate **W1** can demand exact
reproduction (achieved: max abs diff 0.0).
**Why.** Simplest model that captures the two first-order truths: players can't react
instantly, and current momentum skews reach in the direction of travel.
**Caveats.** No acceleration phase, no turning cost. Three distinct "speeds" in play —
model `max_speed` 5 m/s (assumed sustained speed), observed p99.9 ≈ 8 m/s, and the
12 m/s per-component noise clip in `compute_velocities` — do not conflate (NOTES,
Phase 0 findings).

### 2. Closed-form own-location arrival time
**What.** Setting target = player's own position gives `t_own = t_r·(1 + ‖v‖/max_speed)`
exactly (the reaction phase carries them away; they return at max_speed).
**Where.** `validate_phase1` W2(d1) — an equality test to 1e-9, replacing the original
(wrong) "coverage ≈ 1 at own location" criterion; full correction reasoning in
docs/validation.md W2.
**Why.** Exact algebra catches transpose/indexing bugs that loose probability
thresholds sail past. Also documents the REACTION-TIME FLOOR bias: own-location
coverage is capped at `σ((τ − t_own)/sigma)` ≈ 0.947 at rest — coverage near defenders
is understated, negative space around them overstated, worst at speed.

### 3. Logistic (sigmoid) soft threshold
**What.** `σ(z) = 1/(1+e^(−z))` converts a time margin into a probability with scale
`sigma` (0.45s): coverage `P = σ((τ − t_arrive)/sigma)`; pitch-control race
`P = σ((t_def − t_att)/sigma)`.
**Where.** `reach.coverage_probability`, `reach.control_probability`; the latter exists
solely so W1 can show the race is a special case of the same arrival-time field.
**Why.** Arrival times are uncertain (tracking noise, model error); a hard cutoff would
make N's boundary jump discontinuously frame to frame. `sigma` is a stated knob.
**Caveat.** Coverage asks "can this player get there in τ"; control asks "who wins the
race". Same arrival times, different questions — the project needs the former.

### 4. Union of independent events (complement product)
**What.** `P(some defender covers x) = 1 − ∏_p (1 − P_p(x))`.
**Where.** `reach.team_coverage`, feeding `negative_space.defensive_coverage`.
**Why.** The team's positive space is a union over players; this is the standard
probabilistic union under independence.
**Caveats.** Independence is FALSE across teammates (shared tracking error, shared
cues, mutual obstruction) and overstates union coverage where defenders cluster —
biasing |N| DOWN in compact blocks. Sized empirically at ~670–744 m² (~10% of pitch)
against the deterministic variant at τ=2, θ=0.3; reported every run, not gated
(`validate_phase1` W2). `reach.covered_deterministic` (`min_p t_p ≤ τ`) is the
assumption-free cross-check.

### 5. Negative space as a thresholded sub-level set
**What.** `N(t, τ, θ) = { x : C_D(x, t, τ) < θ }` — the region where defensive coverage
falls below confidence θ. Deterministic reading: complement of the union of reach sets.
**Where.** `negative_space.negative_space` / `NegativeSpace`.
**Why.** Direct transcription of concept.md's probabilistic definition. τ and θ are
REQUIRED arguments everywhere — no defaults — so no number can hide its knobs.
**Caveat.** Raw |N| ≈ 68–79% of the pitch at τ=2s: it does not discriminate and must
never be a headline number. Value weighting (Phase 2) is what gives it meaning.

### 6. Monotonicity invariants as theorems
**What.** Given the model: coverage is non-decreasing in τ ⟹ |N| non-increasing in τ;
|N| non-decreasing in θ; removing a defender never shrinks N (unions only grow).
**Where.** `validate_phase1` W2(a–c), checked on random frames.
**Why.** They are provable from the definitions, so ANY violation is a bug, never a
finding — the sharpest possible class of test. (Same philosophy as the ΔN accounting
identity scheduled for Phase 3.)

### 7. Cell-centred grid integration (Riemann sum)
**What.** Area = (count of cells in mask) × cell area, on a grid whose cell CENTRES are
inset half a cell from the touchlines so cells tile the pitch exactly.
**Where.** `reach.pitch_grid`, `NegativeSpace.area_m2`.
**Why.** An endpoint-inclusive linspace grid makes edge cells half-width and silently
inflates areas; centring makes every cell worth exactly cell_m².

### 8. Event-stream interval reconstruction (searchsorted forward-fill)
**What.** Point events → piecewise-constant state per tracking frame via
`np.searchsorted(event_frames, frame, side="right") − 1`: possession (last
possession-establishing event's team) and ball-in-play (dead intervals).
**Where.** `negative_space.possession_by_frame`, `alive_by_frame`, `_dead_intervals`.
**Why + lesson.** Dead time is derived BACKWARDS from restarts: every SET PIECE proves
play had stopped, whatever stopped it. The forward rule (dead starts at BALL OUT)
missed 26 of 77 stoppages in game 1 (fouls/cards/challenges) and gave 74–80% ball-in-play
vs football's ~55–60%; backwards gives 65.6/69.6%. Anchor reconstructions on the
unambiguous marker.
**Caveat.** CHALLENGE deliberately does not set possession (contested moment).
Still at the high end of the plausible band — likely residual over-count; revisit if
Phase 4 per-90 rates depend on it.

### 9. Model-free inversion test (order statistic)
**What.** Median distance from ball to nearest player of the possessing team (0.26m)
vs nearest defender (5.03m); possessor nearer in ~80% of frames.
**Where.** `validate_phase1` (possession gate).
**Why.** Swapping attacker/defender would invert every downstream number while leaving
ALL monotonicity invariants intact (they hold for any team) — nothing model-based
catches it. A model-free physical fact does. Not ~100% by design: passes in flight and
duels legitimately put the ball nearer a defender.

### 10. Smoothed central-difference velocity (inherited, frozen)
**What.** Position diff / dt, centred rolling mean (window 7), computed per period,
clipped at 12 m/s PER COMPONENT.
**Where.** `pitch_control_ref.compute_velocities`, called by `loader.load_game`.
**Caveats.** Per-component clip ⟹ speed can reach 12√2 = 16.97 m/s (observed exact max
in both games — the clip's signature, not football). ~0.01–0.03% of player-frames
exceed 12 m/s; noise spikes seed mislocated reach sets. Phase 1 accepted this;
revisit a physical speed clamp if per-frame maps look wrong.

### 11. Value-surface lookup with orientation transform
**What.** Zone lookup into the season-scale xT grid (`sb_xt.npy`, (12,16) = (ny,nx),
normalized coords, attack→+x) from metric centre-origin tracking coords. The 180°
rotation for direction −1 is a sign flip on both axes: `(x,y) → (−x,−y)`
(`orientation.orient_xy`), applying identically to velocities.
**Where.** `value_surface.ValueSurface.value` / `.value_at_targets`.
**Guard.** Constructor asserts column means are monotone toward the attack end (the main
project's V3 property) — wrong artifact fails loudly. Direction errors would flip the
surface end-for-end and silently invert every value-weighted number: the failure Phase
0's dual-source direction check exists to prevent.

### 12. Y-symmetrization of the value surface
**What.** `xt ← ½(xt + xt[::-1, :])`, default ON.
**Why (measured before deciding).** Asymmetry is < 0.008 everywhere except the goal-mouth
column, where the two central cells disagree 0.259 vs 0.397 — the small-sample box noise
the main project's V3 flagged. Averaging removes noise football gives no reason to
believe (threat is y-symmetric to first order) AND makes the lookup invariant to the
Metrica-vs-StatsBomb y-axis convention, which would otherwise require cross-provider
documentation to trust. Reversible via `symmetrize=False`.

### 13. Valuable negative space (VNS)
**What.** `VNS(t) = Σ_{cells ∈ N ∩ exploitable} v(cell)·cell_area`, units xT·m².
**Where.** `vns.compute_vns`; gate W3.
**Why it earns its place.** Tested, not assumed: added to a ball-x baseline, VNS improves
held-out log-likelihood for "shot within 10s" in BOTH cross-game directions while raw |N|
does not. Comparable across frames at fixed (τ, θ, τ_run) only.

### 14. Offside mask with two time horizons (Gap B + Gap F)
**What.** `line = max(second-largest oriented defender x, 0)`; a cell beyond the line is
exploitable iff some currently-onside attacker reaches it within **τ_run**, a horizon
distinct from and longer than the defensive τ.
**Where.** `vns.offside_line`, `vns.compute_vns`.
**Why two horizons.** Exploiting behind-line space is pass FLIGHT plus run, and the law
only requires the runner onside when the ball is played. Sharing one τ confined
exploitable space to a ~6.5m band, removed 92.6% of final-third VNS, and inverted the
thirds ordering. Removes ~69% of aggregate unmasked VNS at τ_run=4s — Gap B was a real
hole, not a theoretical one.
**Caveats.** Ignores the ball-level clause (mask slightly stricter than the law ⟹
conservative); current attacker positions proxy for positions at the pass moment.

### 15. Knob taxonomy: nuisance vs specification (Gap E, resolved)
**What.** Rank-stability sweeps distinguish knobs the metric must be ROBUST to from knobs
that DEFINE the question. Measured per-knob min pairwise Spearman on per-frame VNS:
θ 0.945/0.975 and τ_run 0.857/0.892 (nuisance, stable) but τ 0.733/0.872.
**Why.** τ ∈ {1.5, 3.0}s changes the coverage radius ~3x — "space unreachable in 1.5s"
and "in 3s" are different objects with no reason to rank frames alike. So τ is a
SPECIFICATION: state it with every number; cross-τ comparisons are invalid.
**Where.** `validate_phase2.check_knob_sweep`, which gates on nuisance knobs and prints
the full decomposition every run.

---

## Scheduled (Phases 3–5)

### 16b. Additive log-complement decomposition — **IMPLEMENTED** (`dynamics.py`)
**What.** Team coverage `C = 1 − Π_i(1 − p_i)` is not additive, so "which defender caused
this cell to open" has no exact answer in coverage space. But
`L := log(1 − C) = Σ_i log(1 − p_i)` IS additive, so
`ΔL = Σ_i ΔL_i` with `ΔL_i = log(1−p_i(t+δ)) − log(1−p_i(t))` splits every cell's change
across defenders with **zero residual**. The negative-space test moves along unchanged:
`C < θ ⟺ L > log(1 − θ)`. Flipped cells are shared in proportion to each defender's
positive (resp. negative) contribution.
**Where.** `dynamics.frame_dynamics`, gated by W4(a) — passed at 5.8e-16.
**Why.** Leave-one-out attribution, the obvious alternative, does not sum back to the
total: two defenders can each be individually non-essential yet jointly responsible.
Log-space makes the accounting identity true BY CONSTRUCTION, so W4(a) can only fail if
the code is wrong — the theorem-class check this project prefers (§6).
**Guarantee.** A cell that opened necessarily has ΔL > 0, hence at least one positive
ΔL_i, so the share normalisation never divides by zero.
**Caveats.** (1) `p` is clipped at 1−1e-9 to keep the log finite. (2) The identity says
the split is EXHAUSTIVE, not that it is CAUSALLY right — it apportions by who changed
their reach, which rewards guarding valuable ground (the GK takes ~37% of creation
credit at every δ tested). (3) δ matters: at 0.6s the flipped set is boundary jitter
(largest connected blob 29% of cells); at 3.0s it is coherent (57%).

### 15b. Ballistic jump model + dome z_reach — **IMPLEMENTED** (`dome.py`, Phase 5a)
**What.** `z_reach(q) = standing_reach + jump(τ − t_arrive(q))`, where the jump is
ballistic rather than tabulated: rising to height h needs takeoff `v = √(2gh)` reaching
apex after `t = √(2h/g)`, so inverted, flight time `t` buys `g·t²/2`. With a plant time
subtracted: `jump(s) = clip(g·max(s − t_plant, 0)²/2, 0, max_jump)`. Defaults give a
maximal leap at ~0.60s of slack (0.25 plant + 0.35 rise to 0.6m).
**Where.** `dome.z_reach` / `team_z_reach` / `reachable_volume` / `can_contest`; gated by
W6a(3), all invariants pass.
**Why it needed no refactor.** reach.py was written in Phase 1 as arrival-TIME fields
over arbitrary targets, not as a 2D grid primitive, precisely so z could be layered on as
a function of arrival time.
**Nice property.** The 3D team union is a pointwise MAX over heights — unlike the 2D
probabilistic coverage union, it carries no independence assumption (§4's known bias
does not apply here).
**Limit.** Uniform parameters mean every vertical difference reduces to slack time, so
the dome cannot differentiate players: measured 66.3% vs 68.5% for plain proximity on 46
aerial duels. Verticality's value is contingent on per-player caps (Phase 5b, blocked).

### 15c. Ballistic ball arc — **WITHDRAWN, does not work on Metrica** (`ball_flight.py`)
**What was intended.** Ground-to-ground drag-free arc: `z(f) = (g·T²/2)·f(1−f)`,
`apex = g·T²/8`, applied only to independently flagged aerial passes.
**Why it is void here.** Metrica's ball path is INTERPOLATED — straight line, constant
speed between annotated touches (speed-retention 1.000 at p10/p50/p90; 3cm median
deviation from straight; 62% under 5cm). No deceleration, no curvature, no altitude
signal. The formula describes the interpolation, not the ball.
**The reasoning error worth remembering.** The falsifiable test assumed aerial passes fly
LONGER at matched distance because energy goes vertical. It failed at pooled p=0.999 in
the wrong direction — the premise ignored ground friction. Lofted balls are struck hard;
ground passes roll gently and decelerate. Aerial is FASTER over equal distance.
**What survives.** Flight time (collector-annotated endpoints) and the aerial FLAG (from
downstream HEAD/AERIAL annotations, independent of physics). Only apex numbers are void.
**Reusable test.** `validate_phase5.check_ball_trajectory` is now the acceptance check for
any future provider — cheap, and it is what silently failed here.

### 16. ΔN attribution accounting identity (Phase 3)
Pointwise set difference between frames δ apart, each flipped cell attributed to the
defender(s) whose individual reachability over it changed; attributions must sum back
to ΔN exactly (exhaustive + exclusive). Theorem-class check, like #6.

### 17. Ballistic arc inference (Phase 5a, Gap C)
Drag-free projectile between endpoints with flight time T from frame counts: for
ground-to-ground, apex = gT²/8; z(t) recoverable along the path. First-order only —
driven balls, bounces, and mid-flight contests violate it; W6a gates on the apex
distribution being plausible (~2–15m).

### 18. Dome z-reach (Phase 5)
`z_reach(d) = standing_reach + jump(τ − t_arrive(d))` — vertical reach as a function of
slack time after arrival. The reason reach.py is written as arrival-time fields: the
z-dimension is a function OF arrival time, so Phase 5 layers on top without rework.

---

### 21. Opportunity-normalised rates (Phase 4, `metrics.py`) — **IMPLEMENTED**
**What.** Each per-90 rate divides by the exposure that actually created the
opportunity: defensive metrics by the player's team's DEFENDING minutes, attacking
metrics by its ATTACKING minutes — not by total minutes played.
**Why.** Total-minutes normalisation puts possession share in the denominator: a team
that dominates the ball defends less, so its defenders would post better "space
conceded per 90" for reasons that have nothing to do with defending. Two clocks per
player, each metric divided by the right one.
**Where.** `PlayerMetrics.def_minutes` / `att_minutes`; gated by W5(a), which checks the
clocks reconcile with ball-in-play time (catching the double-counting that overlapping
windows would cause — windows TILE, they do not overlap).

### 22. Nuisance vs specification knobs (project-wide discipline)
**What.** A sweep knob is a NUISANCE knob if results must not depend on it, and a
SPECIFICATION knob if changing it changes the question asked. They demand different
tests: nuisance knobs are gated on rank stability, specification knobs are reported.
**Where.** θ (nuisance) vs τ (specification), established in Phase 2 and applied in
W5(b). Measured: θ gives 0.858–0.979 across the four usable Phase 4 metrics, τ gives
0.557–0.901. Tripling τ roughly triples a defender's reachable radius, so demanding
rank agreement across τ would demand the model ignore its own input.
**Lesson.** W5 initially lumped both and "failed" 3 of 5 metrics. Splitting them was a
correction to the TEST, not a loosening of it — the third time in this project the test,
not the code, turned out to be wrong (cf. Phase 0 max-speed, Phase 1 W2(d)).

## Ideas (parked docs, not scheduled)

### 19. Relief field — **IMPLEMENTED** (`relief.py`, shipped in the space explorer)
`h(x,t) = v(x)·(1 − 2·C_D(x,t))`: valuable-open renders as peaks, valuable-covered as
depressions, worthless ground flat. `(1−2C)` maps coverage to [+1,−1] so sign carries
"who owns it" and magnitude carries "how much it matters". Sign convention is a
documented free choice. A RENDERING of already-gated quantities — introduces no gate and
makes no new claim; the dynamical reading (∇h as a force field) stays fenced off.
**Rendering caveat learned the hard way:** colour must stay LINEAR in h. A signed-√
scale was tried to make midfield legible, and it amplified near-worthless open space
into the first "valuable" colour band — manufacturing exactly the illusion the value
weighting exists to destroy. Reverted; an unweighted coverage layer serves that need
honestly instead.

### 20. Connected-component labelling + region lineage (idea-space-gravity.md)
`scipy.ndimage.label` on the N mask; per-region mass `M(R) = ∫_R v`; frame-to-frame
overlap gives birth/growth/merge/split/death — the structural summary of Phase 3's
creation/destruction verbs.

### 21. Zero-dimensional persistence over θ (idea-space-gravity.md)
Sweep θ as a sub-level-set filtration of the coverage field; union-find over
sorted cell values yields each negative region's birth/merge span (its persistence)
with no topology library needed. Regions persistent across wide θ windows are
structurally real; narrow-window regions are knob artifacts. A principled attack on
Gap E's knob-dependence problem.

### 22. Gradient-flow dynamics test (idea-space-gravity.md, explicitly later)
`F = ∇h`; falsifiable only: run directions vs ∇h against a ∇v baseline; pass flow vs
region mass against distance baselines. The metaphor earns physics only if these pass.
