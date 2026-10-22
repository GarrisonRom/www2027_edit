# Method Version: GSU-v3

date: 2026-10-19
parent: GSU-v2
status: validated as a candidate; failed the frozen single-seed gate

## what_changed
- Proposed M3A update rule: replace M3U's single aggregate retain-gradient half-space with separate first-order constraints for each of the four sampled retain anchors.
- Keep the same AdamW proposal, LoRA targets, seed, anchor count, data, and training budget; correct the proposal to the nearest update satisfying every sampled-anchor constraint.
- Log each anchor's pre/post correction directional dot product and the projected-to-raw update norm ratio.

## why
- Trigger: H-GSU-UPD-1 was not supported by the frozen single-seed screen. M3U improved the primary by only 1/12 and retain fell by 3/33 despite correction on 112/114 steps.
- Proposed mechanism: a nonnegative dot product with the sum of anchor gradients does not guarantee a nonnegative dot product for each anchor. See [`probe01`](../../exp/runs/exp002_gsu_update_projection/probes/probe01.md) and [`Decision Log`](../06_decisions.md).
- This is a candidate method with partial mechanism evidence; it is not a validated final method or novelty claim.

## reproducibility
- config snapshot: `exp/runs/exp003_gsu_m3a/card.md`
- code path: `exp/train/diagnose_gsu_m3a.py`
- code commit / patch: working tree dirty; run snapshot records the exact script hash and patch
- first verified run: `run_20261020_1332_m3a_train_diagnostic`

## validation result

- Primary `D_hop2_no_F1_overlap`: `3/12` versus M3 `0/12`.
- Retain: `11/33` versus M3 `16/33` and M3U `13/33`.
- F1_hop2: `9/75`; direct sampled-anchor log-probability improved over raw AdamW on `439/456` paired observations.
- The local constraint mechanism is supported, but the frozen retain gate is not. M3A does not enter multi-seed confirmation.

## notes
- Must preserve the exp002 primary metric and pass gate. No multi-seed run unless the single-seed candidate first meets the original thresholds.
- The candidate is being tested under one fixed-seed, 114-update validation budget. No multi-seed run is authorized by this card.
