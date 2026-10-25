# GSU Pareto evidence audit

2026-10-22. Analysis only; no training or formal figures rendered.

- `audit_20261022/pareto_audit.json`: initial descriptive covered-slice audit.
- `audit_20261022_v2/pareto_audit.json`: source SHA256 and covered/all-hop2,
  two/three-objective and F0-only analyses. Keeps initial artifact unchanged.
- `remote_audit_20261022/remote_pareto_audit.json`: read-only SSH collection of
  exp008/009 completed outcomes, original source paths/hashes, per-model frontiers.
- Scripts: `../audit_gsu_pareto.py`, `../audit_remote_baseline_pareto.py`.
- Plot style proposed in STYLE_BRIEF.md; awaiting approval before formal rendering.
- Full scientific design: `docs/GSU最终候选与帕累托验证设计.md` at workspace root.

Local comparison includes all seven A and six B outcomes. Identical A0/B0 and
A1/B1 outcomes are not independent replications. No pooled cross-model frontier.
Covered-slice/heldout-retain choices are post-hoc descriptive audits, not changed
exp006/007 success criteria. All-hop2 is the proposed new experiment endpoint.
Current heldout retain is neither wholly independent nor proven unrelated.

Remote full89-path values are missing, not imputed. Add separate evaluations
of saved adapters before a full-hop2 formal figure. Old local literature baselines
exist in exp/results/eval_results_gsu.jsonl and gsu_sixset.jsonl; training supervision,
update counts and absence of heldout33 audits prevent unconditional pooling.

Verified: dominance unit tests2 passed; local13 outcome records complete;
B2 is covered-slice F0-only non-dominated but all-hop2 dominated by A3.
This audit does not establish global or statistically confirmed Pareto optimality.
