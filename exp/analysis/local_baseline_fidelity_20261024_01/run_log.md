# Read-only fidelity audit

Date:2026-10-24 Asia/Shanghai. Training endpoints unchanged.
Command: D:/Anaconda/envs/cuda126_env/python.exe -B exp/train/audit_local_baseline_fidelity.py
--output exp/analysis/local_baseline_fidelity_20261024_01.
Inputs/sources: immutable exp016 initial manifest and continuation isolated input copies;
actual hashes checked by audit.json. Tokenizer: E:/12models/modelscope/Qwen2.5-1.5B-Instruct.
Environment: environment.txt. No optimizer/training or endpoint selection performed.
Evidence: audit.json, prior/continuation configs and raw diagnostics/behavior.json.
Independent tests: test_baseline_fidelity_audit.py plus complete reproduction and remote
baseline/RMU regression suites,45passed; pytest basetemp tmp/pytest_fidelity_20261024_b.
Two initial failures were new independent-test indexing mistakes, repaired before final suite;
not failures of frozen training code. Source fetched for paper review:
tmp/faithun_2502_19207v2_audit.html (arXiv2502.19207v2, Section4/B.1/B.3).
Scope: local exp016 only, no certification of remote or historical baseline artifacts.
Verdict: formula checks pass in tested cases; paper-level reproduction not established.
Detailed findings: docs/本地baseline复现审计_20261024.md.
