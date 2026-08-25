# Space Analysis — Validation Criteria & Gap Options

Same discipline as surplus_value_model/docs/prototype-validation.md: (1) the finish
line per phase, defined BEFORE building; (2) options for each known gap, with a
recommended pick. Decisions here are provisional until building starts. If a gate
proves impossible at prototype scale, document why and relax explicitly or defer —
don't silently tune forever.

Naming: W-gates (whitespace), to keep them distinct from the main project's V/S gates.

---

## Part 1: Validation gates

### W1. Reduction to pitch control (regression gate)
The reachable-set model, run with uniform parameters in its 2D ground slice, must
reproduce the existing `pitch_control.py` outputs: per-defender arrival times equal to
`_time_to_intercept` within numerical tolerance (1e-9), and the derived attack/defense
control probability equal to the existing sigmoid on ~20 randomly sampled frames
(max abs diff < 1e-6). The dome model claims to GENERALIZE pitch control — this gate
proves the special case before trusting the generalization. Purely mechanical; any
failure is a bug, not a modeling disagreement.

### W2. Negative-space invariants + face validity
Structural invariants, checked over ~100 random alive-ball frames:
- Coverage probability at every point is non-decreasing in τ (more time, more reach)
  ⟹ deterministic N(τ₂) ⊆ N(τ₁) for τ₂ > τ₁; probabilistic |N| non-increasing in τ.
- |N| non-decreasing in θ (stricter confidence ⟹ more space counts as uncovered).
- Union monotonicity: removing a defender never shrinks N.
- ~~Coverage ≈ 1 at each defender's own location (mirror of the main project's V1
  own-location check, 92.3% there).~~ **CORRECTED during Phase 1 — the analogy does not
  transfer.** V1 tested a RACE: both teams pay the reaction-time cost, so the nearer team
  wins and control → 1. Coverage is ABSOLUTE, and the inherited motion model charges every
  player `reaction_time` before moving, so arrival at one's own location is
  `t_own = reaction_time * (1 + speed/max_speed)` — 0.7s at rest, ~1.7s at a 7 m/s sprint.
  Own-location coverage is therefore capped at `sigmoid((tau - t_own)/sigma)` ≈ 0.95 at
  rest and lower when moving. The original criterion demanded behaviour the model cannot
  and should not produce. Replaced by three provable checks:
  (d1) `t_own` matches the closed form exactly (catches transpose/indexing bugs a loose
  probability threshold would sail past); (d2) near-stationary defenders sit at the
  structural ceiling; (d3) every defender covers their own location more than either far
  corner (weak, but a sign error would flip it).
  The underlying reaction-time floor is a real conservative bias for coverage — it
  understates reach near defenders and so OVERSTATES negative space around them, most at
  speed. Documented in reach.py rather than fixed, because W1 requires the arrival-time
  function to remain identical to the frozen oracle. Revisit if Phase 2 value weighting
  proves sensitive to it.
Face validity, eye test on 10 randomly sampled frames (no cherry-picking): channels
behind a stretched line show up as N, a compact block kills central N, the GK's dome
accounts for the deep zone in front of goal.
*Pass/fail: all invariants hold exactly (they're theorems given the model — violations
are bugs); eye test 8/10 defensible.*

### W3. Value weighting sanity + outcome linkage
(a) Face validity: over both Metrica games, mean valuable-negative-space (VNS) should
concentrate where football says it should — final third and half-spaces dominate;
raw-area N (mostly own-half acreage) and VNS must visibly diverge (that divergence is
the whole point of the weighting; quantify as rank correlation between raw-|N| and VNS
per frame — expect it LOW).
(b) Outcome linkage (the falsifiable part): within possessions, does VNS entered (or
created then used) in the preceding k seconds predict possession outcome (shot or box
entry) better than a ball-position-only baseline? At 2-game scale demand only
direction + significance vs. the baseline, not effect-size claims — same modesty as V2's
±10pp tolerance rationale.

### W4. Dynamics decomposition (creation / entry / destruction)
- Accounting identity: pointwise ΔN attribution is exhaustive and exclusive — the
  per-defender attributions of newly-uncovered (and re-covered) area sum back to total
  ΔN exactly. Mechanical; violations are bugs.
- Eye test (mirrors V4): the 10 highest-VNS creation events and 10 highest destruction
  events across both games, inspected against tracking animation. ≥8/10 defensible as
  a real space-opening run / a real recovering action. Log every failure — failures
  are the most informative output.
- Known-pattern check: forwards/wide players should dominate final-third creation and
  entry; centre-backs and holding mids should dominate destruction. Direction only —
  no magnitude claims. (Anonymized Metrica: infer role from mean position, accepting
  the crudeness — the main project's HISTORY §29 lesson says spatial role inference is
  unreliable for outliers, so treat this check as indicative, not a hard gate.)
NOTE: this gate covers DEFENDER-side attribution only (who vacated / who re-covered).
Attacker-side credit ("who pulled the defender") is Gap A and gated separately — W4
must be passable without it.

### W5. Player-metric stability (indicative only at this scale)
Per-player VNS-created, VNS-entered, destruction per 90. Split-half (per-half of each
match) rank correlation as an INDICATIVE number, reported with the explicit caveat
that S2 taught us match-scale stability tests fail even for metrics that are fine at
season scale (median rho 0.353 → 0.855). Gate: compute and report honestly; do NOT
gate the project on a threshold 2 games cannot support. Real stability testing waits
for a larger tracking corpus (Gap D).

### W6. Verticality (3D dome)
- (a) Uniform-parameter phase, Metrica: inferred ball-arc sanity — apex distribution
  of inferred airborne passes falls in a plausible range (roughly 2–15m; a mass of
  30m apexes means the arc inference is broken); flagged-aerial events (long balls)
  coincide with frames where inferred ball z exceeds uniform ground-reach.
- (b) Named-data phase (only if Gap D resolves): at cross/long-ball arrival points,
  dome-cap overlap margin predicts the aerial-duel winner better than a
  height-difference-only baseline (log-loss or AUC, held-out). This is the one gate
  that tests whether the 3D machinery earns its complexity — if height difference
  alone does as well, the dome cap is decoration and we say so.

**Explicitly NOT prototype goals:** stable per-player rankings (2 games can't give
that), per-player attribute-parameterized domes on Metrica (anonymized — impossible),
price/scarcity linkage, claims about specific players. The prototype validates
*machinery*, exactly as the CPV prototype did.

---

## Part 2: Options for each known gap

### Gap A: Attacker-side creation credit ("who pulled the defender")
Defender-side attribution (who vacated) is mechanical and safe. Crediting the ATTACKER
whose movement caused the vacating is a causal claim about defender intent.

| Option | Idea | Cost | Verdict |
|---|---|---|---|
| A1. Marking heuristic | Assign each defender a marked attacker (nearest / trailing-window movement correlation); credit that attacker when the defender vacates | Low | **Recommended first** — transparent, wrong in zonal schemes and admits it |
| A2. Counterfactual freeze | Recompute coverage with attacker a frozen at t−w; credit the difference | Medium-high | Deceptively attractive but requires a defender-RESPONSE model (the observed defense already reacted) — that's ghosting, a research project in itself. Park. |
| A3. Learned marking assignment | Fit assignment from tracking corpora | High | Needs far more data than 2 games. Park. |

Ship defender-side attribution + entry metrics regardless — destruction and entry
never need attacker credit. Creation-credit is the research risk, isolated by design.

**MEASURED CONSEQUENCE (2026-07-31) — A1 is more compromised than expected, and the
problem is NOT where it was assumed to be.** `created_p90` correlates **+0.956** with a
player's mean oriented pitch position. Two fixes were built and both largely failed:

| variant | overall ρ vs position | within advanced band |
|---|---|---|
| `created_p90` (xT-weighted, raw) | +0.956 | +0.929 |
| `created_relative` (zone-exposure adjusted) | +0.868 | +0.929 |
| `created_area_p90` (**no value weighting at all**) | +0.806 | +0.714 |

The decisive line is the third. Stripping xT *entirely* — removing the only quantity that
is a function of location — still leaves ρ = +0.806. **So the confound is not in the value
weighting; it is in the A1 marking heuristic itself.** The mechanism credits an attacker
only when a defender is within `mark_dist_m` of them, so attackers who are marked (i.e.
advanced players in the defensive block's zone) accrue credit while deep players in
possession, who have no marker nearby, accrue none. The metric substantially measures
*"are you being marked by someone who moves"*.

Consequences, decided:
- **`created_p90` is not a cross-position player-evaluation metric and normalisation
  cannot make it one.** Do not rank players across roles with it. Within-band comparison
  is defensible but still carries ρ ≈ +0.71 to +0.93 inside the band.
- Zone-exposure adjustment (`created_relative`) is retained as a diagnostic — it does
  shrink the between-band gap from 4.3x to 1.8x — but is NOT a fix.
- The root cause sits in A2/A3 territory: a defender-RESPONSE model, which is ghosting and
  a research project. A1 was always flagged as "wrong in zonal schemes and admits it";
  this quantifies how wrong.
- Unaffected: destruction, entry, and all defender-side attribution never use the marking
  heuristic. Only attacker-side creation credit is compromised.

**CONFIRMED AGAINST SCRIPTED GROUND TRUTH (2026-07-31, `validate_attribution.py`).** The
real-data evidence above is circumstantial — position is a proxy, not proof. Synthetic
tracking with authored causality settles it, 40 jittered trials per scenario:

| scenario | defender-side | A1 attacker-side |
|---|---|---|
| clean drag | 100% | 74% correct |
| decoy + beneficiary | 100% | 100% correct |
| crossed marking | 100% | **5% correct** |
| zonal slide (truth: nobody) | — | **92% spurious** |
| static control (truth: nobody) | — | 0% spurious |

Defender-side attribution is now validated as CORRECT, not just exhaustive — W4 proved it
sums, this proves it points at the right player. A1 is reliably WRONG (not noisy) under
crossed marking, and fabricates credit in 92% of zonal-slide trials. Since scripted
marking is far cleaner than real defending, 74% on the easy case is a ceiling.
**A1 is not salvageable by normalisation; only A2 (defender-response modelling) reaches
the failure modes above.**

### Gap B: Offside / exploitability mask (hole in the original concept doc)
Raw N over-counts the most valuable region: space behind the back line is only
exploitable if a runner can be onside at the pass moment. Not mentioned in concept.md
— caught at planning review.

| Option | Idea | Cost | Verdict |
|---|---|---|---|
| B1. Ignore | Report raw VNS | Free | Overstates behind-line space — the single most valuable kind. Not acceptable past Phase 2. |
| B2. Hard offside mask | Point behind the second-last defender counts only if some attacker is currently onside and can reach it within τ | Low | **Recommended** — coarse but honest |
| B3. Pass-feasibility weighting | Weight each point by P(a feasible pass arrives there) — reuses CPV completion machinery | High | Right long-term answer; crossover with CPV; defer |

### Gap C: No ball z in Metrica — **CLOSED AS UNRESOLVABLE (Phase 5a, 2026-07-31)**
The mitigation below was built (`ball_flight.py`) and tested, and it fails for a reason
that is structural rather than fixable: **Metrica's ball path is interpolated, not
measured** — straight line, constant speed, drawn between annotated touches
(speed-retention 1.000 at p10/p50/p90; median 3cm deviation from straight). No
deceleration, no curvature, therefore no altitude signal of any kind.
The gate's own falsifiable test — aerial passes should fly LONGER at matched distance —
failed at pooled p=0.999 in the wrong direction, and the premise was wrong anyway (lofted
balls are struck hard, ground passes roll gently, so aerial is FASTER over equal
distance). Apex inference is **withdrawn, not tuned**.
Consequence: Gap C cannot be mitigated on Metrica. It converts into a DATA REQUIREMENT —
a provider must supply MEASURED ball z, verified with `validate_phase5.check_ball_trajectory`
on arrival. Original text retained below for context.


For airborne passes, endpoints + flight time (frame count between pass and reception)
+ a drag-free symmetric arc determine altitude along the path (apex ≈ gT²/8 for
ground-to-ground). First-order only: driven balls, bounces, and mid-flight contests
violate it — and contested aerials are precisely the frames we care about. Mitigation:
apply only to event-flagged long balls/crosses, exclude by speed/curvature heuristics,
W6a sanity-checks the apex distribution. Continuous ball z needs a different provider
(Gap D).

### Gap D: Vertical attributes + named players
Per-player dome caps need identity (heights, jump proxies) AND velocities. StatsBomb
360 is disqualified here — freeze frames have no velocities and partial visibility,
and reachable volumes are fundamentally kinematic. Candidate: SkillCorner Open Data
(broadcast tracking, ~9 matches, named players; VERIFY current availability, frame
rate, and whether ball z is included before committing — same verify-against-primary-
source discipline as the StatsBomb recipient-attribute gate). Attribute databases
(FM/FIFA) carry licensing questions — resolve before use, or fall back to aerial-duel
win rates from StatsBomb events as an empirical jump proxy. Decision deferred to the
Phase 5 go/no-go.

### Gap E: Knob dependence (τ, θ, and τ_run)
Every headline number depends on knobs. Discipline (V5b style): headline metrics
must be rank-stable (Spearman ≥ 0.8) across a stated grid of reasonable settings
(τ ∈ {1.5, 2, 3}s × θ ∈ {0.2, 0.3, 0.4} — final grid fixed before running). If
rankings flip on arbitrary knobs, the metric isn't measuring something real yet.
Report the sweep in every phase that produces a per-player or per-team number.

**RESOLVED IN PHASE 2, with a distinction the original framing missed.** Measured on the
full 27-cell grid (a third knob, τ_run, was added in Phase 2 — see below), per-knob
decomposition of min pairwise Spearman on per-frame VNS:

| knob | game 1 | game 2 | verdict |
|---|---|---|---|
| θ (confidence) | 0.945 | 0.975 | nuisance knob, very stable |
| τ_run (run horizon) | 0.857 | 0.892 | nuisance knob, stable |
| **τ (defensive horizon)** | **0.733** | 0.872 | **NOT a nuisance knob** |

τ spans 1.5→3.0s, roughly a 3x change in coverage radius. That is not a perturbation of
one question, it is a DIFFERENT question — "space no defender reaches in 1.5s" versus
"in 3s" are distinct objects, and there is no reason they should rank frames alike.
**τ is therefore a SPECIFICATION, not a robustness range**: every VNS number must state
its τ, and cross-τ comparisons are invalid. The gate accordingly requires stability
across the nuisance knobs (θ × τ_run at fixed τ; 0.884+ observed) and reports the
wide-τ figure as a documented limitation rather than a failure. This is a narrowing of
the criterion with a stated principle, not a threshold retreat — the raw numbers are
printed on every run.

### Gap G: Shooting is invisible to the model (raised by the user, 2026-07-31)
Three distinct problems, ranked by tractability. Prompted by the observation that long
shots carry value the relief does not show. Measured context: central-channel xT runs
0.328 (~3m) / 0.052 (~16m) / 0.014 (~30m), so long-shot range sits ~11x below the
six-yard box.

**G1 — Per-cell rendering hides an integral (rendering only, cheapest fix).** NOT a
modelling error, but it actively misled a reader, which is worse than a caveat nobody
reads. Measured over 3,000 random frames: **76.9% of exploitable VNS lies IN FRONT of the
offside line** (3,595 m²/frame at 0.0042 xT/m²), only 23.1% behind (225 m²/frame at
0.0201 xT/m² — 4.8x denser). The relief colours each cell by its own value, so the
front-of-line majority renders flat while the dense minority glows, and the picture says
"valuable space only exists behind the line" while the accounting says the opposite.
A per-REGION view (connected components sized by mass `M(R) = ∫_R v`, already specified
in idea-space-gravity.md) shows it honestly. Note this is the mirror of the reverted √
contrast scale: that made worthless space look valuable, this makes modest-but-plentiful
space look worthless. Any fix must not reintroduce the other.

**G2 — No shooting lanes (real modelling gap, structural).** Coverage asks whether a
defender can REACH a point; nothing asks whether a defender stands BETWEEN that point and
the goal. Wide-open space at 25m with six bodies in the lane and the same space with a
clear sight of goal are currently identical. Fix would be a line-of-sight term from each
cell to the goal mouth against defender positions — a real addition that changes numbers,
not pixels. No parameter tweak reaches it.

**G3 — xT prices "ball here", the relief uses it for "free space here" (deepest).**
The two diverge most exactly where being unpressured unlocks a specific action, and a
long shot is the clearest case: the zone average blends every contested, back-to-goal
possession with the rare unpressured one. So open space in long-shot range is underpriced
BY CONSTRUCTION, not by a bad estimate. Proper fix needs a value surface conditioned on
being unpressured, which two Metrica games cannot estimate. Connects to the main
project's documented scope limit ("CPV covers passing decisions ONLY — no shooting").

None of these block Phase 3.

### Gap F: Two time horizons in the offside mask (discovered in Phase 2)
The first B2 implementation used a single τ for both "the defense can cover this point"
and "an onside attacker can arrive at this point". Those are different horizons:
exploiting space behind the line means a pass FLIGHT plus a run, and the law only
requires the runner to be onside *when the ball is played*. Using τ=2s for both confined
exploitable behind-line space to a ~6.5m band, stripped **92.6% of final-third VNS**,
and inverted the thirds ordering (final third fell to 26% of total VNS when unmasked it
is 79.8%). Fixed by a separate `tau_run` (default 4.0s), swept as a nuisance knob above.
Lesson for later phases: any rule that mixes defensive reachability with attacking
reachability needs two horizons, not one.
