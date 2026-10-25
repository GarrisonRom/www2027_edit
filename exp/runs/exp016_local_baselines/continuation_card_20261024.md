# Frozen continuation of local baseline comparison

Hypothesis, expected pattern and null control: inherited unchanged from card.md;
C1 may improve association-covered recall but GA already wins all89 in the first batch.
Frozen primary metric: all89 path macro recall lower better; covered75, F0/F1 residuals
lower better; independent retain higher better. No post-result endpoint switch.
Baselines to finish: DPOmis, DPOrej, CEU, PDU, WGA, SatImp, UNDIAL.
Budget: seed42, Qwen2.5-1.5B, F0=33, first33 retain, LoRA r8 alpha16,
114updates lr5e-5; each own two-update smoke, per-job6h, continuation24h.
Original six successful endpoints remain unchanged. Old queue stopped on shared source
hash change; source additions concerned other experiments. Continuation uses the EXACT
previous source snapshot and hash-verified independent input copies, not current shared code.
Sanity: verify source/input hashes before every child, finite updates, smoke gates;
PDU warmup and CEU skipped-answer caveats persist. MMLU/PPL remain pending.
Observed: first batch six completed, including KLUE86updates normal threshold stop.
Interpretation: C1 is not the best all89 method; coverage advantage remains to be compared.
Verdict: pending completion; no significance or Pareto-optimum claim.
Cause guess: source management interruption, not an algorithmic training failure.
Next action: isolated continuation, progressive comparison table with raw source paths.
Assets salvaged: six completed endpoints, all frozen sources and inputs.
