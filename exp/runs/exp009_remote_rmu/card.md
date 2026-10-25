# exp009 RMU-LoRA representation matching

Preregistered exploratory follow-up, 2026-10-22. Not official full-weight RMU.

- hypothesis: representation intervention may reduce associated recall beyond
  output-loss methods; fixed layer intervention may also fail to generalize.
- expected_pattern: F0 recall below Original, and a potentially different
  F1_hop2/retain tradeoff from GA and GradDiff; no predicted guaranteed winner.
- null_control: Original model from exp008, same weights/environment/questions.
- frozen_primary_metric: accuracy of fixed F1_hop2 75 dependent paraphrases, lower better.
- frozen_primary_baselines: exp008 Original, GA, GradDiff, NPO, SimNPO F0-only.
- budget: two models, seed42, 114 steps, lr5e-5, AdamW defaults as exp008;
  LoRA r8 alpha16 dropout0 only MLP down_proj at zero-based layers5/6/7.
- mechanism: layer7 output matches a fixed positive uniform random unit vector
  times20 for forget answers; retained answer representations match disabled-adapter
  frozen base. Both masked MSE terms have coefficient1. Raw QA prompt.
- sources: local open-unlearning RMU class supports EMBED_DIFF; config defaults
  steer2 and all parameters, while class defaults steer20 and selected matrices.
  This experiment explicitly selects steer20, retain weight1 and restricted LoRA;
  it is not the default YAML and does not claim a FaithUn-tuned reproduction.
- data/evaluation: identical exp008 fixed-input FaithUn and historical five F1 facts;
  training F0-only33, retain pool33, heldout33. Same answer LP/MCQA audits and paraphrases.
- sanity_checks: both actual model loss/backward/audit/save smoke tests must finish
  before either formal job; finite losses/grads, unique layer hook, three targets.
  Queue starts only after exp008 completed; dependency failure aborts the follow-up.
- provenance: isolated remote source snapshot, source/data/model hashes inherited
  from exp008 batch manifest plus exp009 source hashes, actual pip freeze, dirty
  patch, exact commands, modules and parameter counts saved before training.
- observed: pending; verdict: pending; cause_guess: pending.
- next_action: run after exp008; any tuning requires another preregistered experiment.
- assets_salvaged: reusable hooks/masked MSE and representation matching diagnostic.
