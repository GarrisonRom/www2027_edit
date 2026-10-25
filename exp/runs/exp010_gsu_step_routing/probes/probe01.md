# Strengthen probe01: risk location to actual update direction

exp_id: exp010_gsu_step_routing; date: 2026-10-22; cycle: one proposed bounded round.
Failed/partial references: exp006 A2/A3 and exp007 B2/B3, exact run IDs in parent design.

## Observations and broken link

B2 covered recall44/75 vs B0 52/75, but F0 13/33 vs8/33 and overall hop2
49.44% vs44.94%. A3 overall32.21% is better than B2. B2 answer LP on covered
queries is higher, not lower. A positive component signal is not an effective
within-component forgetting direction or guaranteed overall transfer.

## Experimental reverse derivation

Audit g_guided versus full AdamW displacement, per-component geometry and each
probability flip. C1 tests routing after Adam, C2/C3 add true spans, Cg isolates
F0 progress guard; C3D controls generic rank-limited damping; C4 tests whether
components add anything. Raw/final FP/LP behavior and fallback rates distinguish
effective routing from norm damping and feedback undoing.

## Logical derivation

Stationary positive scalar gradient rescaling approximately cancels in Adam
when moments scale proportionally; epsilon, nonstationarity, initialization,
weight decay and cross-step normalization break exact invariance. Thus do not
declare this the measured cause before diagnostics. Apply geometry to d0 instead.
F0 progress is a necessary local condition, not a guarantee of global forgetting.
Projection of F0 proposal into F1 gradient spans is label-informed filtering,
not equivalent to adding a signed F1 gradient; test sign h^T*d and actual LP.

## Analogy transfer

Trust-region/line-search logic bounds actual parameter changes; low-rank projection
separates location from direction. These are design analogies, not proof of
unlearning benefit. Gradient spans can be stale or wrong; refresh three times
and compare matched random spans rather than assuming shared causal circuitry.

## Candidates and discriminating test

C1 post-Adam routing; C2 A-score+span; C3 B-score+span. Each has the same progress
guard, trust radius and feedback. If Cg explains all gains, routing hypothesis is
unsupported. If C3D matches C3, generic damping is sufficient. If C2>=C3, there is
no added evidence for historicalD; if C4>=C2/C3, components are unnecessary.
Main C3 is prespecified, no outcome-based rename. Method: GSU-v8 draft.

## Budget and kill rule

Eight single-seed groups,912 steps, proposed16 GPU-hour ceiling; stage0 tests
first. No auto sweeps. Candidate kills include direct/overall regression, domination
by guard-only, ineffective geometry and >50% full fallbacks. Details in card.md.
Remaining global strengthen budget was never frozen; do not imply project-wide
budget approval. This document proposes only this round and does not launch it.

## Outcome / gate

run_ids: none; verdict: untested; claim headroom: unknown.
Not permission to pivot or to claim optimality. New confirmation data and budget
required if the mechanism screen succeeds.
