# Local F0-only baseline comparison

Hypothesis: C1 association-guided routing reduces all89 multihop recall relative to
F0-only baseline losses; this is untested, not a claimed win.
Expected pattern: C1 lower all89 recall, with F0 and independent retain reported
alongside it; no assumption of a Pareto improvement.
Null control: plain GA without association guidance; existing C3R is not a matched C1 control.
Frozen primary metric: all89 path-macro recall (lower better), covered75 reported
in raw behavior audit; F0/F1 MCQA lower better, independent retain MCQA higher better.
Frozen baselines: GA, GradDiff, NPO, SimNPO, RMU, DPOmis, DPOrej, CEU, PDU,
WGA, SatImp, UNDIAL (matched LoRA); KLUE (separate native track).
Budget: Qwen2.5-1.5B-Instruct local model, seed42, F0=33; matched LoRA r8 alpha16,
lr5e-5, batch1, 114 updates, training retain first33. Native KLUE uses all66 retain,
batch4, lr1e-5, max150epochs/1350updates with source training-UA early stop.
Each method requires two-update smoke before formal run; per job6h, queue24h.
Evaluation: frozen local behavior, full89 and covered75; MMLU/PPL explicitly pending,
not replaced by retain. Native KLUE last33 retain are NOT heldout.
Sanity checks: finite loss/gradients, frozen source/input/model hashes, separate
F0 supervision versus C1 F1 guidance access, no test-based tuning. KLUE is an explicit
paper reimplementation, not certified author-code parity. PDU five-epoch warmup may
disable dual updates within114steps. CEU skipped single-token answers must be disclosed.
Observed: pending; raw metrics and failures live under runs, never overwrite old runs.
Interpretation: pending, no robustness or global-optimum claim from seed42.
Verdict: untested.
Cause guess: pending.
Next action: smoke then formal local jobs, report failures and missing capability metrics.
Assets salvaged: source-aligned loss bridge, native neuron optimizer, frozen behavior evaluator.
