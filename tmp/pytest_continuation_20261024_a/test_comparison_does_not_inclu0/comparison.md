# Local C-group and baseline comparison

Qwen2.5-1.5B, seed42; provisional descriptive ranking, no significance claim.
KLUE is a separate native paper reimplementation track; its retain33 were trained.
C variants use F1 guidance; C0 omits the F0 guard. Data access and optimization differ.
MMLU/PPL pending. Missing/failed methods have no performance row.

| Method | All89 recall ↓ | Covered /75 ↓ | F0 /33 ↓ | F1 /5 ↓ | Retain /33 ↑ | Heldout |
|---|---:|---:|---:|---:|---:|---|
| KLUE | 50.00% | 3 | 1 | 2 | 28 | False |

## Raw sources

- KLUE: raw/metrics.json
