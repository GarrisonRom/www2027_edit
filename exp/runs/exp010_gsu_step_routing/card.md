# exp010: F0-anchored actual-step routing (DESIGN DRAFT)

date: 2026-10-22
assumption_id: H-GSU-STEP-1
method_version: GSU-v8-f0-anchored-step-routing
status: completed; single-seed exploratory screen, all eight variants finished 2026-10-23

## hypothesis

F1-derived component/within-component structure can route F0-only AdamW updates
to reduce multi-hop recall without weakening direct forgetting versus guard-only
updates. Historical FFN prior may or may not add value beyond current gradients.

## expected_pattern

C3 should outperform Cg on all89-path recall and covered25-path recall, with F0
direct correct count no higher. Structured components/directions should beat their
matched random controls. Actual routed step directions should differ from d0;
F0 safeguard should prevent loss of local direct progress. Mere reduction of step
norm or feedback cancellation, unchanged answer-LP or high fallback is an alternative.

## null_control

C0 no extra guard/routing; Cg guard only; C3R permutes score-to-component mapping;
C3D uses fixed random per-tensor spans of equal rank. C4 removes component weights.
Preserve failed/dominated results and random controls in all summary figures.

## frozen_primary_metric (proposed new card only)

Path-macro MCQA recall over ALL89 hop2 subitems (267 paraphrases), lower better.
Each subitem averages its three paraphrases; then average paths. Key subgroups:
25 F1-covered paths /75 questions and 64 other paths /192 questions; directF0,
F0 paraphrases, directF1, retain_train33 and retain_heldout33, answer-only LP/margins.
Existing exp006/007 primary remains unchanged; do not retroactively relabel it.

## frozen_primary_baselines

Cg is primary mechanism comparator; C0 is fresh original FB16 reference. C3 is
prespecified main candidate. Also compare C1/C2/C4 and paired nulls C3R/C3D.
Historical A3/B2 are exploratory references, not new independent seed replications.
External GA/GradDiff/NPO/SimNPO/RMU need separate same-model comparison after this
probe; remote different-model outcomes cannot serve as this card's paired baseline.

## method contract

See proposed v8 version and docs/GSU最终候选与帕累托验证设计.md. Constants fixed for
this draft: LoRAr8/alpha16/dropout0, AdamWlr5e-5/betas.9,.999/eps1e-8/wd.01,
routing floor.5, direction residual.5, total correction radius.30||d0||, FB16 two
rounds16 anchors, F0 progress fraction.9, tolerance1e-6, alpha1,.5,.25,0.
Span probes5 facts before updates1/39/77, per-tensor rank<=5; SVD cutoff1e-6,
zero-span identity fallback. No gradient of hop2 questions. Only F0 forget loss.

## budget

- Stage0 CPU/toy geometry: Adam scaling attenuation example, finite tensor-span
  projection, zero-LoRA-A fallback, exact optimizer-state step count, trust radius,
  F0 backtracking and feedback re-audit tests; then tiny-model loss tests.
- Stage1 eight variants C0/Cg/C1/C2/C3/C3R/C3D/C4; seed42, 114 steps each,
  total912 committed updates on Qwen2.5-1.5B. At most15 extra F1 backward probes
  per trained candidate for three basis refreshes, besides initial scoring and FB16.
- Phase1 proposed compute ceiling16 GPU-hours, storage20GB; reaching ceiling
  preserves interrupted run and stops, no hidden restart or automatic extra methods.
- Finite nonprogress/nonfinite/invalid-split/weight/checksum/budget failures retain
  complete logs and stop batch. Validation failure is not permission to change constants.
- Cross-model/multiseed/strength sweeps are NOT included in912 steps. Their cards
  must separately freeze development selection, baseline tuning and confirmation set.

## screening criteria / kill rules

C0 must reproduce B0 under same inputs/init. C3 must have lower all-hop2 recall
and lower covered-hop2 recall than Cg, with F0 correct count<=Cg. No core gain
claim if improvement is solely by losing direct forgetting. Cover/uncovered
diagnostics and final retain probability/accuracy reported irrespective of direction.

Real-span candidate must beat C3D and structured-score candidate must beat C3R
on primary at the same F0 noninferiority criterion; no statistical claim from seed42.
If C3 is dominated by Cg in recall/direct/utility, or the median actual routing
change is numerically zero, or >50% of valid raw-descent steps fully fallback,
stop this candidate and report mechanism failure; do not add modules to rescue it.
If probability and correctness disagree, record that disagreement; no knowledge-
erasure claim. Any LP-suppression mechanism requires paired lower target answer LP
on target-defined queries, not only changed distractor ranking.

## sanity_checks

Input SHA256/splits/schedules/init frozen as exp006. No test query as gradient or
step acceptance target. Record source/dirty patch, environment, model/data hashes,
commands, all diagnostics and failed runs. Snapshot full config before every run.
F0 sample schedule and anchors identical across all groups. Optimizer moments
advance once per update only. Preserve trained adapter and checkpoint114 only as
main output; do not select the best evaluation checkpoint. Report all8 groups.

## observed

Implementation is isolated in exp/train/run_local_gsu_c_group.py. The user
authorized the frozen eight-variant local screen and its 16-GPU-hour ceiling on
2026-10-23. Geometry self-test passed; the C2 and integrated-guidance regression
suite passed (13 tests); actual-model C0 smoke passed.

Formal C0 completed 114 updates in run
exp/runs/exp010_gsu_step_routing/runs/batch_20261023_081500_Cgroup/C0. Its F0
accuracy was 5/33, F1 direct 4/5, covered F1-hop2 40/75, noncovered-hop2 accuracy
7/12, all89 path-macro recall 0.4532, retain-train 20/33, and retain-heldout 15/33.
Historical B0 used the same initial LoRA SHA256, F0 schedule, and anchor schedule,
but had F0 8/33, F1 direct 5/5, F1-hop2 52/75, noncovered-hop2 9/12, and retain
15/33. C0 therefore failed the prespecified fresh-control reproduction gate.

The first restart attempt is preserved at
exp/runs/exp010_gsu_step_routing/runs/batch_20261023_081500_Cgroup. C0 completed
and Cg was interrupted at update 32; C1-C4/C3R/C3D were not launched. That attempt
matched the historical B0 initial LoRA hash, F0 schedule, anchor schedule, and all
eight compared core source hashes. The first-step F0 loss and FB16 correction
matched, but feedback trajectories differed by update 5. The cause is unresolved.

For the authorized restart, the C0 path now uses the exact historical B0 trainer
and LoRA construction, including its otherwise-unused component scoring pass and
training mode. An eight-step replay is compared per update to frozen B0 diagnostics
at absolute tolerance 1e-4. Only after that passes will formal C0 run; it must match
the frozen B0 count metrics for F0 direct, F1 direct, covered-hop2, noncovered-hop2,
and heldout retain before Cg/C1/C2/C3/C3R/C3D/C4 can start. Each formal variant
will archive behavior audits at updates 38, 76, and 114. Every restart uses a new
batch directory; the first failed attempt is immutable historical evidence.
Restart batch: batch_20261023_092800_Cgroup_restart1. Geometry self-test and the
13 C2/integrated-guidance regression tests passed before dispatch. New C3 score
groups use the same L00-style component keys as the historical local A/B trainer.
The post-hoc Pareto audit of old runs remains under
exp/analysis/gsu_pareto/audit_20261022_v2.

## interpretation / verdict / next_action

completed exploratory screen / partial mechanism support / harvest plus independent confirmation.
C0 passed the frozen historical B0 replay and count gates. C1 has the lowest observed
all89 path recall (27.72%) and best observed covered-hop2/heldout-retain tradeoff
among the eight, but F1 direct accuracy is 0/5 and the study is single-seed/local.
C3 remains the prespecified candidate; do not relabel C1 as a validated optimum.
C3R is a permuted-score null, not a matched rerun of C1, so component-localization
causality remains unproven. Report all eight outcomes and confirm on new seeds and
cloud models before any general method claim.

## assets_salvaged

A/B scores, FB16, frozen samples and audits, actual-proposal instrumentation,
post-hoc Pareto audit. Probe derivation: probes/probe01.md.
