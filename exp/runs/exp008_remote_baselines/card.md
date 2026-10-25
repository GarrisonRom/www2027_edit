# exp008: remote cross-architecture baseline screen

Status: preregistered exploratory single-seed screen, 2026-10-22.

- hypothesis: F0-only direct unlearning can leave associated multi-hop recall;
  expanded supervision may reduce it, with method- and model-dependent effects.
- expected_pattern: direct F0 recall decreases versus Original; F1_hop2 can remain
  higher than GA with F0+F1. Failure to observe this pattern is retained evidence.
- null_control: unchanged Original model for each model family. No claim about
  structure specificity is tested here; random structural controls belong to RQ3.
- frozen_primary_metric: MCQA accuracy on the frozen F1_hop2 75 paraphrases,
  lower means less observed recall. These are dependent paraphrases, not 75 paths.
- frozen_primary_baselines: Original, GA, GradDiff, NPO, SimNPO with F0-only;
  GA F0+F1 is an explicitly separate expanded-supervision reference.
- budget: Qwen3.5-4B and Llama-3.2-3B-Instruct; seed 42; 114 AdamW steps per
  trained method, batch size 1, lr 5e-5, weight decay .01, LoRA r8 alpha16 dropout0.
  Ten training runs and two Original evaluations, sequential on remote RTX4090.
- data: frozen FaithUn SHA256 d570cc3e30c083357ca45ab4dd9df807abbd800f51a657d89629e4fe7b3adbb0;
  forget33, retain66, test464. Training retain pool first33, heldout last33.
  Existing five F1 facts/active paths were selected using the historical Qwen2.5
  pipeline: this is fixed-input transfer, NOT model-specific path discovery.
- secondary: F0 direct/paraphrase, F1 direct, answer-only logprob, legacy
  full-sequence MCQA margin, related historical slices and separate retain pools.
- architecture: linear modules inside language decoder attention/MLP receive LoRA;
  visual/MTP modules excluded. Qwen3.5 hybrid and Llama have different modules and
  trainable parameter counts, recorded per run. This is not matched-parameter
  evidence for model-size superiority. No GSU component prior is transferred.
- loss: reuse train_baselines.compute_loss without changing formulas; gamma1,
  alpha1, beta1, delta0. NPO uses summed answer NLL versus frozen disabled adapter;
  SimNPO uses mean answer NLL. These are local LoRA implementations, not claimed
  to reproduce every upstream full-finetuning recipe or tuned hyperparameter.
- prompt: legacy raw question-space-answer, no chat template or sampled reasoning.
  Model differences are not evidence about chain-of-thought reasoning abilities.
- sanity_checks: per-model loss/backward smoke test for all four methods before
  training; fresh base per job; frozen source/data/model hashes; finite loss/grads;
  independent outputs, no overwrite, all methods use same installed environment.
- observed: pending; raw metrics and per-item records in each run directory.
- verdict: pending, not support/partial/reject before observation.
- cause_guess: pending.
- next_action: run compatibility smoke tests, then baseline screen; do not select
  a new method or tune using the same evaluation and call it confirmation.
- assets_salvaged: isolated architecture-aware runner, audited datasets and losses.

Run artifacts: config, schedule/retain indices, modules/parameter counts, adapter,
diagnostics, metrics, status, stdout/stderr; batch records environment, source,
data and model manifests. Snapshot includes current dirty patch and Git provenance.
