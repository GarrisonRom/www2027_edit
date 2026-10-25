# ModelScope capability dataset download receipt

Date2026-10-25 Asia/Shanghai. Download COMPLETE, not capability evaluation complete.
Root:D:/GSU_datasets/modelscope_eval_20261025_02.
Source:modelscope/mmlu and modelscope/wikitext; raw API revisions, file sizes and SHA256
preserved in root/manifest.json. Raw sources kept under raw/, verified Parquet under prepared/.
No remote dataset Python code executed. Archives inspected without general-purpose extraction;
links and traversal paths rejected. Local helpers:local_capability_data.py,capability_data.json.

MMLU data.tar166184960bytes SHA256bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b.
57subjects, dev285, validation1531, test14042. Original CSV question and option order preserved;
A/B/C/D mapped to0/1/2/3 for existing harness fieldsquestion/choices/answer.
ModelScope old mmlu.py uses pandas.read_csv with inferred header, losing first question in
each split; metadata reflects this row loss in all171subject/splits. We parse HEADERLESS
original CSV and retain every question, then check canonical split totals. Original source
files and first-run validation failure at modelscop_eval_20261025_01 retained for audit.
13source rows have duplicate option text; preserved, not silently removed or reordered.

WikiText-2-raw-v1 zip4721645bytes SHA256
ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11.
train36718,validation3760,test4358. Match source text generator: nonempty raw lines keep
newline, whitespace-only lines become empty strings. Evaluation corpus joins text fields
with two newlines, matching existing PPL evaluator. Test corpus1294336characters, SHA256
696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83.

174prepared Parquet files. Every MMLU subject hash verified; WikiText3splits hash verified.
Offline datasets.load_dataset(parquet) succeeds on abstract_algebra(dev5,validation11,test100)
and full WikiText test corpus read. Archive traversal tests6passed. Python interpreter:
D:/Anaconda/envs/cuda126_env/python.exe. Initial downloaded archives reused after parser
repair; no duplicate network redownload of166MB tar.

Limits: ModelScope raw source checksums certified, byte-equivalence to the cloud's pinned
HuggingFace revision not independently certified. Report ModelScope source/version and text
hash, do not falsely attach the old HF revision to these files. Existing immutable evaluation
queue snapshot is unchanged by this download; its capability stages still require an explicit
local-source-aware continuation. Training checkpoints and completed scoring are untouched.
