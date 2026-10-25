# Method Version: GSU-v7-integrated-damage-gradient-path

date: 2026-10-22
parent: GSU-v6-memory-guided
status: implementation frozen; untested
experiment: exp007_gsu_integrated_guidance

## motivation

exp006 separated two signals that should be connected but were not: the
historical post-hoc damage attribution and the training-time F0/F1 gradient
sharing score. A2/A3 did not replace F1 supervision, while A3 versus A3R
left a weak structured signal. This version tests whether a frozen historical
damage prior can route the current gradient/path signal more usefully.

## what_changed

The new entry point is `exp/train/diagnose_gsu_integrated_guidance.py`.
It reuses the exp006 trainer, schedule, FB16 actual-logprob feedback, audit
fields, and comparison checks. It changes only the frozen component score:

```text
gradient_path_S(j) = A_j * max(P_j, 0)
damage_prior(l) = sqrt(norm_D_G1(l) * norm_D_G2(l))
integrated_S(j) = gradient_path_S(j) * damage_prior(layer(j))  # MLP
integrated_S(j) = gradient_path_S(j)                            # attention
```

The two normalized FFN profiles are the frozen `per_layer.D` values from
the existing G1/G2 attribution artifacts. They are not recomputed from the
new run and are not training targets.

## evidence boundary

The historical evidence is strongest at MLP layer and selected neuron
granularity: G2 FFN damage peaks at layer 27, the G1/G2 shared neuron list is
concentrated in layers 22--27, and layer/module rollback results exist.
There is no defensible historical q/k/v/o matrix-level damage score in the
current artifacts. Therefore attention components use a neutral prior, and
the run records this limitation explicitly. The method does not claim a
complete component-causal map.

## variants

- B0/B1 are fresh no-guidance baselines.
- B2/B3 use integrated soft/hard guidance.
- B2R/B3R are score-permutation controls.

The existing exp006 A0--A3 results are references, not overwritten baselines.

## reproducibility

- Model/data/split/F1/active-path inputs inherit the frozen FaithUn protocol.
- Historical source files, hashes, layer priors, raw gradient/path scores and
  integrated scores are copied to each run snapshot.
- The run launcher must use `D:\Anaconda\envs\cuda126_env\python.exe`.
- No full training is started by the implementation patch; run directories
  are new and immutable.
