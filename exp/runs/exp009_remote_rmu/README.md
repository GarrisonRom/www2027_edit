# exp009 dispatch and localization review

Date: 2026-10-22. RMU queue dispatched, not yet trained at verification.

- Remote snapshot: `/root/autodl-tmp/www2027_rmu_20261022`.
- Queue PID at dispatch: 7813; state verified `waiting_for_dependency`.
- Status: snapshot + `/exp/runs/exp009_remote_rmu/runs/batch_20261022_01/status.json`.
- Dependency: exp008 `batch_20261022_01/status.json` must reach completed.
- On dependency failure stop; otherwise smoke both models, then RMU F0-only
  for Qwen3.5-4B and Llama-3.2-3B-Instruct, sequentially, 114 steps each.
- Tests: three masked-loss/module-rejection tests passed locally and remotely.
- No active exp008 source files were edited. Parent snapshot code is copied.

## Localization candidates

KLUE and CLUE are distinct names, not assumed spelling variants.

KLUE: the EMNLP2025 FaithUn paper introduces knowledge-localization unlearning;
the public yny0506/faithun repository currently documents the benchmark. Its
landing page alone is not sufficient to establish an available executable KLUE
implementation. Paper algorithm details/official code provenance and an audited
localization adapter are prerequisites; do not label an invented mask as KLUE.
This is especially relevant because our data is FaithUn.

CLUE: official repository cloned read-only for review at
`/root/autodl-tmp/CLUE_review_20261022`, commit
`96f78b9788801f643ce40c41e40cb33b7c856b89`.
README specifies noising/denoising edge-pruning circuits, refined logical edges,
conflict solver, masks, then separate conflict-aware unlearning. Inspected
conflict.py, mask_generation.py, unlearn_model_conlict.py and model mask handling.
Main example uses Mistral cyber pruning; Qwen3.5 hybrid architecture requires
additional instrumented-model support, not just swapping a model path.
Neither CLUE nor KLUE training has been launched or claimed reproduced.

Next: audit KLUE paper localization objective and code availability, then assess
a Llama-first adapter with frozen F0-only supervision, equal training budget,
matched random localization and separately reported localization cost. For CLUE
include both circuit construction and safe/conflict training policies; component
ranking alone is an ablation, not the full framework. Preregister before runs.

Primary sources inspected: ACL Anthology 2025.emnlp-main.657;
GitHub yny0506/faithun and Zodiark-ch/CLUE; local open-unlearning RMU class/config.
