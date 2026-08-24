# Corpus audit (clean at strict level) + closed-loop entropy experiment + a proposal: treat environment memory as a learning space

*Everything below is reproducible: `python test_concept_novelty.py` from repo root (CPU-only, numpy + scikit-learn). Reports attached.*

## I tested your corpus before saying anything

Your paper closes on the idea that the next scaling law may be the rate
at which intelligence creates valid, diverse, executable experience at
its own frontier. This is a genuinely interesting area of thought, and
I started from that sentence and audited the released
pool against it: all 7,872 environments, compared by what each game
*does* — the mechanics; comparisons, accumulations, container
operations, reward dependencies, names and literals discarded — rather
than what it is called.

Result: **0/7,872 strict concept-duplicates. 7,872 distinct mechanics
signatures. 0 parse failures. 0 duplicate groups crossing declared
skills.** At looser thresholds, 0.3% of environments have a near-twin
at cosine ≥ 0.99 over op-distributions — and when I opened the
top-scoring pair by hand (game_203412 / game_204723, causal_inference),
I found two genuinely different games: different hidden mechanisms,
different endgames, a shared engine style. That is a huge credit to your
grounding-and-validation pipeline and a caught limitation of my own
instrument.

## Is there a gap in the mechanism itself that can be refined?

Your paper flags repetitive generation as a known risk. The selection
path contains no explicit mechanism for it:
`grep -rE "divers|novel|simil|dedup" spade/core/` returns nothing —
`EnvironmentMemory.add()` accepts unconditionally and
`high_regret_seeds()` ranks by regret alone. To be fair about what DOES
guard diversity implicitly: corpus grounding injects outside variety at
generation, and hint-regret is partly self-correcting — a flooded family
that the agent masters stops being high-regret. Diversity in stock SPADE
is an emergent byproduct of those two processes, not an enforced
property; the question is what happens where they thin out — if they
do — at frontier levels.

So I closed the loop and measured it, with the control I would demand
of anyone else: a designer whose proposal taste follows regret feedback,
proposing into the real `EnvironmentMemory` class, twelve cycles, three
seeds per condition — and a **dose-response sweep** over how much regret
advantage one skill family starts with. Under a perfectly symmetric
regret landscape the stock loop is metastable (entropy ~0.95). Give any
family a realistic edge and collapse is graded in the dose: +10%
advantage → 0.90 mean (worst seed 0.80); +20% → 0.86 (worst 0.69);
+40% → 0.75; +80% → 0.68. Skills are never equally hard, so the
asymmetric case is the real one. The same loop with a concept-gated
filter (a cap, not a ban: at most k near-identical mechanics per
neighborhood, threshold calibrated from measured similarity separation)
**holds at 1.00 across every dose and every seed.**

This looked unusually good — which is usually a warning. So I attacked
my own result the way I expect you would. Two rebuttals
follow from your design: grounding should dilute seed-driven drift, and
regret should self-correct (mastering the flooded family collapses its
regret). Both are right, and both are measurable: injecting grounding
at 30% proposal coverage lifts stock's worst seed from 0.76 to 0.93;
self-correcting regret alone lifts it to 0.96; with both healthy the
stock loop holds ~0.99 and there is no vulnerability. So the honest,
narrower claim: **entropy decay is a conditional failure mode — it
opens where grounding coverage is low and where high-regret regions
resist mastery** (too-hard families, or reward-hacked regret — exactly
the pockets your limitations section worries about). The explicit gate
holds 1.00 under every condition tested, including the adversarial
ones. As self-play scales toward greater designer autonomy, it may simply
need more guard rails. The gate's value is not that stock SPADE is
currently collapsing — your
corpus shows it is not — but defense-in-depth: it converts diversity
from an emergent byproduct of two contingent processes into an enforced
invariant, which matters most exactly where scaled self-play is headed —
greater designer autonomy (less grounding) and harder frontiers (slower
mastery). And the difference
teaches: a proxy world-model trained on the diversity-held pool
predicts unseen skill *combinations* 2.7× better than one trained on
the drifted pool (held-out masked-prediction error 0.0095 vs 0.0260;
the ratio reproduces at ~3× across feature configurations).

## Should we build more than a dedup patch?

Treat what the memory knows as a **learning space**: a diffuse cloud of
mechanics with no discrete addresses (in the learned version, this
cloud literally lives in the encoder's weights). New knowledge is distance from
the cloud — an environment is redundant iff a world-model trained on
memory already predicts its mechanics from a masked view. Blind
clustering over each candidate's profile (distance to nearest
pair-midpoint, distance to nearest known concept, pull-balance between
parents) recovers the curriculum's natural families with no labels
anywhere — mirroring, at least in miniature, how humans seem to learn:
in the vicinity of, and by combining, what they already know.

The dense middle of that cloud is where combinations live, and it has
computable addresses: the **mass-weighted midpoint of two mastered
concepts** (each parent pulling in proportion to how much mechanism it
contributes) marks where their fusion should condense — an address at
which to *commission* the environment that does not exist yet. The
interesting measurement is the deviation from it: a genuinely
integrated fusion (one mechanic's outcome gating another's rules) sits
measurably off its midpoint — 0.075 in my toy runs — and that residual
quantifies interaction rather than coexistence. (Calibration, stated
plainly: concatenations land on the weighted midpoint at distance ~0 by
linear identity; the naive unweighted midpoint fails even that.)

Commissioned worlds are accepted by a **four-gate test** (whether these
gates need to be hand-written at all is its own question — in the
learned-encoder version they become learned decision boundaries): anchored
between the right parents; **a null gate — the result must be neither
parent**, its distance to each above the measured reskin/garnish band;
displaced from the midpoint (emergence); and not a copy of anything
else known. In my battery the null gate alone caught three cheats the
emergence test passed (parent-plus-decoration in both directions, and a
reskin); concatenation died on emergence exactly as the linear identity
demands; one lopsided-but-genuine integration was falsely rejected —
the conservative direction, which is the correct asymmetry for a
curriculum, since a false accept pollutes the pool and compounds
through seeding while a false reject costs one re-commission.

My experiments also found the limit of doing any of this with flat
features: the wall is always **interaction structure** — same-template
vs same-game, decoration vs lopsided integration, compositional novelty
itself. A mixture of known distributions is maximally predictable to a
linear model; my teaching-value experiment confirmed the corresponding
null (commissioned *concatenations* add nothing a linear proxy can
see). So the real version lives in a **learned structural encoder
co-evolving with the designer** — embed environments with the policy
model plus an online masked-prediction head, so the representation can
encode which mechanic feeds which — anchored by the frozen signature
layer as a tamper alarm, with divergence between the two layers logged
per cycle: the quantity that distinguishes an evolving curriculum from
a drifting one, and from a designer gaming its own metric.

## The ablation I cannot run and you can

Against stock SPADE on the 30B-A3B games recipe (`cmd/ablations/`),
GPT-5.5 static corpus as control: (1) concept-gated `add()`;
(2) novelty-penalized seed slates (regret − λ·max concept-similarity to
the slate so far); (3) the learned encoder as live scorer with frozen
signatures as anchor; (4) one designer-prompt slot per cycle reserved
for an unpopulated midpoint commission, graded by its integration
residual through the four gates. Track pool skill-entropy and pairwise
concept-similarity per cycle, fraction of accepted twists and true
fusions, and your held-out suites at matched rollout budget.

**Falsifiable predictions.** Stock curriculum entropy shows decay
episodes concentrated where the two implicit defenses thin — cycles
with low grounding coverage, and high-regret regions that resist
mastery — and the size of those episodes grows with model scale, since
a stronger designer exploits regret faster than grounding can dilute
it. (This is checkable in your existing training logs: designer
proposal distributions per cycle, conditioned on grounding-doc coverage
and per-family solve-rate trajectories.) The gated, commissioned version holds
entropy roughly flat while staying on the regret frontier, and the
held-out gap widens where stock saturates: the fixed-env baseline
saturates when the learner masters the pool; stock self-play saturates
when the designer quietly fixes its own pool. If commissioning works,
gains concentrate on composition-heavy benchmarks (multi-turn tool
use), because integrated fusions are the training signal for exactly
that — and they are the category every flat selection mechanism
silently discards. My intuition, stated as intuition: this creates
diverse experience slowly at first — it will need heavy RL early —
then compounds as mastered concepts give the commissioning engine more
addresses to build from, including across domains. I tested that
compounding mechanism in miniature: a loop that commissions one empty
midpoint per cycle, gates the result, and promotes accepted fusions to
full concepts grew from 5 concepts and 10 open addresses to 20 concepts
and **155 open addresses** in fifteen cycles, reaching third-order
compositions (fusions of fusions of fusions), while an external-drip
baseline receiving new base concepts grew linearly to 36 addresses and
composition depth zero. The integration operator there is synthetic, so
this demonstrates the growth *dynamics*, not designer competence — but
the dynamics are the point: every mastered composition multiplies the
addresses available next, which is what makes creation rate at the
frontier a candidate scaling law at all.

I need to test whether this is true, and the 30B loop is where it can
be tested. Instruments, closed-loop evidence, honest nulls, and reports
attached; MIT like the repo. Happy to extend or run the small-scale
pieces.

---

## Repository contents & how to reproduce

| file | what it does | runtime |
|---|---|---|
| `test_concept_novelty.py` | Sections A–E: drift reproduction on `EnvironmentMemory`, mechanics-signature vs text identity, distance-from-cloud novelty, blind self-clustering, mass-weighted midpoints + integration residual | ~30 s CPU |
| `experiments_loop.py` | Dose-response entropy sweep (label-free gate), author-rebuttal conditions (grounding / self-correcting regret), teaching-value comparison, compounding-addresses growth | ~2 min CPU |
| `corpus_audit_v3.py` | Strict concept-duplicate audit of the released 7,872-environment corpus (reads the HF snapshot) | ~2 min after download |
| `corpus_neardup.py` | Near-twin rates at cosine 0.90/0.95/0.98/0.99 + top named pairs for manual inspection | ~5 min CPU |
| `audit_report.json`, `neardup_report.json` | Outputs of the two corpus audits as cited above | — |

All CPU-only; `pip install numpy scikit-learn huggingface_hub`. Corpus scripts
run against `spade-rl/SPADE-Environment-Pool-GPT5.5-Games`; loop scripts run
inside the SPADE repo root (they import `spade.core.env_memory`).
MIT, same as the repo.
