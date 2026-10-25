# exp015: complete baseline reimplementation and alignment audit

Status: implementation validated locally; cloud dispatch recorded separately.
Methods: GA, GradDiff, NPO, SimNPO, RMU, DPOmis, DPOrej, CEU, PDU, WGA, SatImp, UNDIAL, KLUE.
No result from exp008/009/014 is overwritten or renamed an official reproduction.

## Frozen question and outcomes

Hypothesis: source-aligned losses/native update domains yield different behavior from historical
constructor-default LoRA ports; any apparent GSU gain must survive valid strong baselines.
Expected pattern: no assumed GSU or baseline winner. Regressions and no-effect runs remain evidence.
Null control: each unchanged base model; reference logits cached before the first update and
native KLUE unselected neuron entries verified unchanged, including optimizer momentum effects.
Primary metric: all89 hop2 path macro recall (lower), with direct F0 as a feasibility measure.
Unchanged Original scores are inherited from frozen exp012 on the same local model files,
tokenizer and capability protocol; source/model manifests must match before reuse.
Secondary: covered75, retain33, MMLU57x10 five-shot, full WikiText2 PPL, costs, and raw-QA
AA/MA across forget33 and test464, preserving sample counts and the formula distinction.

## Two explicit tracks

Matched LoRA: rank8 alpha16 dropout0, all text decoder attention/MLP linears, lr5e-5,
batch1, 114 committed updates, seed42, retain first33, no test-based tuning.
The backbone runs BF16, returned logits/loss are FP32 as in Accelerate output conversion.
Native: actual text weights with CPU FP32-master AdamW; no PEFT conversion. Actual batch4,
fixed right-padding to the training pool maximum, answer-only labels, linear LR decay and
max gradient norm1 as in the HF Trainer default profile (not constructor-only defaults). Each method
uses frozen OpenUnlearning YAML loss coefficients/epochs. This is an upstream-native
reimplementation on our data/models, NOT a claim to reproduce every author's published number.
Native KLUE: gamma.7 alpha1, batch4, max150epochs, stop training-UA<=1/3, lr1e-5
as an explicitly chosen first source-range trial (not the paper's completed LR search),
attribution alpha10, five mismatches, global top5% FFN intermediate neurons.
Gated-neuron mapping: pre-down activation, gate/up weight rows and down weight columns.
Probability attribution is the full-answer sequence probability, never silently replaced by logprob.
Native optimizer updates only selected entries; visited FP32 masters/moments are CPU-resident.
KLUE uses constant LR and norm1 on selected gradients, explicitly declared optimizer choices.
Native uses all66 training retain examples: its last33 are NOT heldout. Metrics expose
retain_is_heldout=false; test464 and capability metrics provide independent audits.

## Differences that must remain visible

- Model families differ from the author's full study; Qwen3.5 and Llama3.2 tested separately.
- Prompt/scoring remain frozen raw-QA/full-sequence mean LP for compatibility. Exact author
  prompt alignment is not certified. Paper AA/MA formulas alone do not certify identical evaluation.
- No author KLUE source is public in the checked faithun repository. Gated FFN convention,
  probability/token pooling, mismatch exclusions and optimizer are declared implementation choices.
- DPOmis uses same question with a different answer drawn from training retain; DPOrej uses
  the explicit fixed response "I don't know." Neither substitutes alternate paraphrase questions.
- PDU executes primal-dual source code. Source YAML retain_loss_eps is unresolved ???;
  initial training-retain mean NLL is our explicit constraint. Five-epoch warmup is preserved;
  fixed114 matched runs may never enable dual updates and must be labeled accordingly.
- CEU executes no added retain term. Cases whose answers are entirely masked by the upstream
  first-token exclusion are skipped and counted, not converted into another CEU formula.
- UNDIAL repairs the frozen source's negative ignored-label indexing and prompt/padding CE:
  teacher targets and cross-entropy are restricted to valid shifted answer labels. Upstream-as-written
  remains in the audit namespace; corrected results are explicitly not bitwise upstream parity.
- RMU repairs the source BxB normalization broadcast for batch>1: masked hidden MSE is normalized
  within each sequence then averaged. Batch1 is unchanged; variable-length counterexample tested.
- RMU uses source YAML steering2 and EMBED_DIFF; matched LoRA updates all selected text modules,
  native updates text weights, not historical down_proj-only rank8 steering20.

## Budget and fail-closed behavior

Two models; seed42 first validation, no multi-seed robustness claim. Exactly25 endpoint
specifications per model: KLUE native +12 matched +12 native,
50 formal jobs total, each gated by its own two-update smoke. Queue ceiling72hours, run ceiling6hours.
Every formal endpoint computes all89, MMLU10/subject, WikiText2, plus test464 audit.
Source/file hashes frozen before jobs; dependency is completed exp014 aligned C1 batch03.
Finite loss/gradient, source loss/gradient parity, native masking, fixed reference, layer uniqueness,
split hash, source immutability, storage and timeout checks apply. Smoke failures block their formal run.
Storage gate never falls back from native to LoRA. Resource failures remain failed/not-run, not completed.
Native full-weight endpoints save lossless zlib-compressed XOR deltas against exact base weights,
with per-tensor base/final/file hashes and roundtrip tests. KLUE saves visited native weight slices.
Delta storage can still exhaust disk; a space gate stops the affected run rather than dropping its checkpoint.

Observed: final local suite43 passed (32 complete-reproduction tests plus11 existing regression tests);
earlier cloud CPU suites24/26 passed; final cloud/source and real-model outcomes recorded in dispatch/run logs.
Verdict: untested for performance.
Cause guess: baseline implementation/configuration/evaluation differences under audit.
Next action: real-model smoke, retain all failures, then run validated jobs and merge measured rows
into the existing comparison tables. Final paper-level adequacy still requires development-only
hyperparameter selection, independent confirmation and multi-seed results.
Assets: immutable source loss bridge, native KLUE implementation, tests, training/evaluation queue.

