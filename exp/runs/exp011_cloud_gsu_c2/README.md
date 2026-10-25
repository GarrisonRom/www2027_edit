# Selected cloud GSU comparison

GSU-C2 is the single selected candidate, public current-model component/span
routing without historical layer prior. See card.md for frozen contract.
Actual cloud smoke and dispatch status will be recorded separately.

## Required localization baseline

FaithUn is already the benchmark. KLUE, the method associated with the FaithUn
paper (EMNLP2025, ACL2025.emnlp-main.657), is NOT in completed exp008/009.
KLUE is now a required pending localization comparison, not optional omission.
Do not conflate it with CLUE (ICLR2026 conflict-guided circuits).
Before running: verify original localization objective, trainable scope, official
implementation provenance, data permissions, model adapters and compute budget.
Only audited/validated code may be called KLUE; no invented mask-based proxy.
If official executable code is unavailable, retain this missing comparison as
pending and document the exact implementation derivation before a separate run.

Completed baseline metrics: GA/GradDiff/NPO/SimNPO/RMU-LoRA and Original, plus
expandedGA, two models. New C2 actual results will be compared to these; KLUE
remains unranked until executed, no predicted score is inserted.
