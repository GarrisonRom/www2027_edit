# exp012: cloud MMLU and general-corpus PPL audit

Date: 2026-10-22. Status: running in a separate snapshot after exp011 completed.
No training; one capability audit per existing baseline and final C2 adapter.

## Hypothesis

Some methods that lower target recall may do so through broad capability degradation.
General MMLU and WikiText-2 shifts provide complementary, imperfect indicators.

## Expected pattern and measures

No claim that C2 must have zero capability loss. Compare per-model change from that
model's Original and relate it to target recall. MMLU accuracy higher is better;
token PPL lower is better within the same tokenizer/model. Neither alone proves
general reasoning, factuality, generation quality or absence of collateral harm.

MMLU pilot: OpenUnlearning-pinned lm-evaluation-harness0.4.11; MMLU group;
5-shot from official dev; no chat template; seed42 samples10 test indices without
replacement per each of57 subjects (570 items); batch1; preserve indices, per-subject
and per-sample output. Name it MMLU-10/subject, not full MMLU.

PPL: Salesforce/wikitext, wikitext-2-raw-v1 test, pin dataset commit and corpus SHA256.
Concatenate official text fields with two newlines; add one BOS token when available;
context<=2048, stride512, each next-token label scored exactly once. Report total NLL,
valid token count, token PPL, delta NLL and ratio to same-model Original. Do not rank
absolute PPL across tokenizers.

## Frozen endpoints and controls

16 outcomes: Original, GA-F0, GradDiff-F0, NPO-F0, SimNPO-F0, RMU-LoRA-F0,
GA-F0F1 for each of two models, plus Llama3.2 C2 and Qwen3.5 C2. Identical data,
task samples, harness, prompt and tokenizer for each model family. All GPU evaluation
starts after C2 training batch completion; pip install occurs only in a separate
venv with system packages shared, after the C2 dependency gate.

## Sanity, provenance and budget

Run a smoke on Original Llama first: per-subject MMLU limit1, short PPL token prefix,
then require at least50 MMLU subjects and exact PPL token accounting. Only then run
all16 endpoints sequentially. No automatic retries. Each endpoint gets config,
source/adapter/model/data hashes, environment, raw subject/sample output, PPL NLL,
status, logs, peak memory and wall time. Evaluation does not change adapters.

Budget ceiling: 24 GPU-hours for 16 sequential endpoints plus one smoke; timeout
per endpoint 90 minutes. On ceiling/failure keep partial data and stop; no silent
retraining, package changes in the base training environment or altered sampling.

## Interpretation boundary

Compare within model against Original; MMLU subset is a broad knowledge/academic QA
audit, PPL is general text prediction under a fixed corpus. Report all outcomes,
including regressions, no cross-model absolute PPL rank, no MMLU-subset-as-full score.
Results supplement rather than replace all-hop2, direct forgetting, locality, KLUE,
MQuAKE/RWKU protocols, matched random controls or multi-seed confirmation.

## Status

Latest repair (2026-10-23): harness raw metadata is now converted to JSON-safe
values before serialization, preserving numeric metrics; the queue anchors its
working directory to WS. Four capability tests passed, including torch.dtype
regression. A new snapshot www2027_sync_20261023_capability_eval_v3 retains the
failed v2 batches. batch_20261023_06 passed the 57-subject Original Llama smoke and
entered the first formal endpoint qwen35_4b_Original_F0. Proxy acceleration and
HF_HUB_DISABLE_XET=1 are enabled. Previous _05 failure below is historical.
Existing report_v2 and report_v3_klue_pending tables now include both exp011 C2
endpoints, with all measured behavior/LP/margin/cost ranks rebuilt from raw evidence.

Observed as of 2026-10-23: no formal endpoint completed. The earlier 20261023_01
controller exited while waiting for already-completed exp011. A retry in _02 failed
to reach Hugging Face; _03 found the dataset but failed the source hash check because
it was launched outside the frozen snapshot root; _04 passed that gate but MMLU data
download failed through Xet with HTTP 401. These failed directories are retained.
After enabling AutoDL academic network acceleration and HF_HUB_DISABLE_XET=1, the
_05 batch completed WikiText smoke token accounting at 5,276 tokens and full89-path
behavior audit, and downloaded/generated MMLU subject data. It then failed while
serializing mmlu.json because raw_result contains a torch.dtype object. Thus MMLU
smoke did not pass, and no formal endpoint started. Repair serialization, then use a
new frozen batch; retain _05 as failed evidence.

Current batch:
`/root/autodl-tmp/www2027_sync_20261023_capability_eval_v2/exp/runs/exp012_cloud_capability_audit/runs/batch_20261023_05`.
All 16 formal endpoints include existing baseline adapters and C2; smoke is excluded
from ranking. Baseline full89 metrics use the C2 per-item evaluator. No training or
adapter modification is performed. Reader-facing explanation of every metric and
method distinction is in docs/云端GSU-C2与指标说明.md.
