# Method Version: GSU-v5-actual-feedback

date: 2026-10-20
parent: GSU-v4-retain-protection
status: completed single-seed screen; rejected for advancement

## motivation

GSU-v4 COV16 proved that wider retain-anchor coverage improves the final retain point estimate (`20/33` versus M3A `11/33`), but its local first-order constraint did not guarantee the actual nonlinear answer logprob after writeback: `866/1824` corrected observations still decreased and `9/1824` actual residuals crossed the `1e-5` audit threshold. CUM16 further showed that summing linearized floors does not protect the real multi-step trajectory; its worst fixed-anchor actual gap was `-25.5761`.

The new mechanism therefore feeds the actual post-proposal answer logprob back into the correction. It keeps the coverage intervention fixed and changes only the missing feedback link.

## algorithm

For each forget example:

1. Sample 16 fresh retain anchors from the fixed 33-anchor pool.
2. Compute the original forget-gradient proposal and execute one AdamW step.
3. Measure each sampled anchor's actual answer logprob change relative to the value immediately before the proposal.
4. For anchors with an actual drop below `-1e-4`, compute gradients at the current post-proposal parameters and solve the minimum-norm compensation whose first-order predicted improvement covers the observed drop.
5. Apply the compensation only up to a trust budget of `0.30 * ||delta_raw||`; remeasure all 16 anchors.
6. Repeat steps 4–5 at most once more if any anchor remains below the actual-drop tolerance. The optimizer moment state is unchanged by the post-hoc compensation.

The correction is a feedback controller, not a cumulative linear floor: the reference is the immediately preceding state and the error signal is the measured nonlinear logprob change.

## frozen constants

```text
anchors_per_update = 16
actual_drop_tolerance = 1e-4
max_feedback_rounds = 2
max_correction_to_raw_norm_ratio = 0.30
seed = 42
epochs = 3
updates = 114
```

The `0.30` budget is frozen from the COV16 correction-ratio 90th percentile (`0.285745`, rounded upward). The `1e-4` trigger is ten times the exp004 actual-residual audit tolerance and is intended to suppress numerical chatter. These constants are not selected from the FB16 outcome.

## scope and limits

This version is a one-seed mechanism screen on Qwen2.5-1.5B. It does not claim cumulative retain guarantees, because the controller only observes the current 16-anchor batch. The final retain and the frozen `D_hop2_no_F1_overlap` metric remain necessary checks alongside the direct logprob diagnostics.

## source records

- exp004 COV16: `exp/runs/exp004_gsu_retain_protection/runs/run_20261020_cov16_rerun`
- exp004 CUM16: `exp/runs/exp004_gsu_retain_protection/runs/run_20261020_cum16`
- strengthening probe: `exp/runs/exp005_gsu_actual_feedback/probes/probe01.md`

## validation result

Run `run_20261020_1714_fb16` completed all 114 updates and covered all `33/33` retain identities. The actual feedback reduced the candidate's raw AdamW anchor decreases from `1780/1824` to `838/1824`, and `1750/1824` paired observations improved over raw AdamW. However, relative to the COV16 reference it reduced final decreases by only `1.54` percentage points (`838/1824` versus `866/1824`), with a worse final mean delta (`-0.053329` versus `+0.026811`) and lower retain (`19/33` versus `20/33`). The primary was `3/12` and F1_hop2 was `22/75`.

The bounded controller and solver passed runtime sanity, but the mechanism and final retain gates were not met. The candidate is retained as diagnostic evidence and does not enter multi-seed confirmation.
