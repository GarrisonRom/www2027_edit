# exp016: Qwen3-8B and Llama3.1-8B, complete baselines plus C1/C2

Date:2026-10-24 Asia/Shanghai. Status: source/tests/download/dispatch receipts distinguish pending and completed.

Hypothesis: C1/C2 effects and baseline tradeoffs may or may not transfer to8B and repeat over seeds.
Expected pattern: no assumed winner; report degradation, nulls, failed gates and all requested methods.
Null: one unchanged Original per model, not three copied scores represented as independent runs.
Primary: all89 path macro recall (lower), conditional on separately reported direct F0 effectiveness.
Secondary: F0/paraphrase/covered75/retain, MMLU57x10 five-shot fixed sample plan, full WikiText2
NLL/PPL, test464 AA/MA under disclosed raw QA score, costs and diagnostics.

## Models and downloads

ModelScope Qwen/Qwen3-8B -> /root/models8b/Qwen3-8B (system disk).
ModelScope LLM-Research/Meta-Llama-3.1-8B-Instruct -> /root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct.
Download API resolves master revision metadata and freezes full remote file inventory plus per-file SHA256.
Do not substitute Qwen3.5-9B or quantized weights and call them8B.
Only root-level safetensors/tokenizer/config/license/docs; no duplicate original/PyTorch shard formats.
Both models must complete integrity verification before GPU queue starts. No deletion of old models/runs.

## Exact matrix

Seeds42,43,44 independently initialize adapters, forgetting schedule, retain choices and random controls.
MMLU evaluation sample seed42 remains fixed; train seeds do not change data split/test questions.
All13 baselines: GA, GradDiff, NPO, SimNPO, RMU, DPOmis, DPOrej, CEU, PDU, WGA, SatImp, UNDIAL, KLUE.
Per model+seed:12 matched-LoRA baselines +13 native baselines (including KLUE) +C1+C2 =27.
2models x3seeds x27 =162 trained endpoints, plus2 Original evaluations.
Each model/method/track has a separate seed42 smoke before its three formal seeds.
No KLUE-LoRA proxy. No combined native/LoRA rank or unmeasured score.

## Alignment

C1/C2 execute the exact shared local C-group worker with architecture enumeration.
LoRA r8 alpha16 dropout0, bf16 backbone, train mode, CAUSAL_LM wrapper, lr5e-5,
114updates, seed-specific F0 and16-anchor schedules, feedback and progress guard.
C1 has no span; C2 refreshes true F1 basis at1/39/77. Nonreentrant activation checkpointing
reduces memory, must pass numerical/real-model smoke, does not authorize dropping modules.
Matched baselines use the same seed-specific F0 ordering as C1/C2; preference alternate sampling
has its own frozen data plan, not a change to the forgetting permutation.

Native update domains/losses/profiles follow exp015, with disclosed source fixes.
8B native execution streams CPU-resident linear weights and saves backward tensors on CPU.
Native parameter storage BF16, Adam moments FP32 on CPU. Full-native optimizer omits duplicate
FP32 masters to fit120GiB RAM; this execution/numeric difference is disclosed, not bitwise original-paper fidelity.
KLUE continues its explicit native selected-neuron Adam convention. Checkpoints keep exact BF16
weights in sparse or lossless XOR formats; no silent replacement by LoRA/4bit.
Reference logits frozen before first update, actual8B model manifest attached per run.
Native uses all66 retain; retain_last33 is NOT independent for this track, flag saved.

## Scheduling and limits

Downloads CPU/network-only now. GPU work waits for exp015 terminal state to avoid contention.
A previous completed_with_failures is not inherited as successful evidence; its failures remain visible.
Queue ceiling21days, per training worker72hours, command watchdog74hours.
50GiB data disk/30GiB system disk still constrain saved checkpoints: explicit storage failure,
no automatic deletion. Disk gate failure means requested run incomplete, not a reproduction success.
Multi-seed results retain each seed and show mean/sample standard deviation with n; incomplete
seed groups cannot be described as three-seed results. Primary bootstrap should cluster by source fact/path,
not count dependent paraphrases as independent. Multi-seed alone does not establish significance.

Sanity: model checksum, immutable source, disjoint split, nonfinite values, fixed evaluator, native
streamed forward/gradient parity, native optimizer toy parity, checkpoint roundtrip, seed matrix completeness.
Observed: no8B formal result at preregistration; download and queue receipts supply later state.
Verdict: untested. Next: download verification, true8B smoke, run complete matrix, merge measured rows into original master table.

