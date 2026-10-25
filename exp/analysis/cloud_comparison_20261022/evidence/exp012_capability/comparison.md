# Cloud capability and forgetting audit

MMLU-10/subject is a balanced subject-coverage pilot subset, not full MMLU.
Token PPL is compared within each tokenizer/model; do not rank absolute Qwen vs Llama PPL.
All-hop2 path-macro is recomputed on all89 paths for each baseline; C2 reuses its frozen training-run evaluation.

| Model | Method | All-hop2 path recall ↓ | Covered-hop2 accuracy ↓ | F0 accuracy ↓ | Heldout retain ↑ | MMLU macro ↑ | ΔMMLU pp ↑ | WikiText token PPL ↓ | PPL ratio ↓ | ΔNLL ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-3B-Instruct | GA [F0] | 0.3034 | 0.0133 | 0.2424 | 0.2121 | 0.6526 | 0.00 | 3407.0173 | 354.1480 | 5.8697 |
| Llama-3.2-3B-Instruct | GA [F0F1] | 0.2772 | 0.0000 | 0.2424 | 0.2121 | 0.6474 | -0.53 | 871.7635 | 90.6169 | 4.5066 |
| Llama-3.2-3B-Instruct | GSU-C2 [F0] | 0.3034 | 0.0133 | 0.2424 | 0.2121 | 0.6579 | 0.53 | 1033.6389 | 107.4433 | 4.6770 |
| Llama-3.2-3B-Instruct | GradDiff [F0] | 0.6067 | 0.8400 | 0.4242 | 0.6667 | 0.6509 | -0.18 | 9.8389 | 1.0227 | 0.0225 |
| Llama-3.2-3B-Instruct | NPO [F0] | 0.7154 | 0.9200 | 0.7273 | 0.8788 | 0.6526 | 0.00 | 9.6588 | 1.0040 | 0.0040 |
| Llama-3.2-3B-Instruct | Original [F0] | 0.7041 | 0.9200 | 0.8788 | 0.8788 | 0.6526 | 0.00 | 9.6203 | 1.0000 | 0.0000 |
| Llama-3.2-3B-Instruct | RMU [F0] | 0.6966 | 0.9067 | 0.8485 | 0.9091 | 0.6491 | -0.35 | 9.6836 | 1.0066 | 0.0066 |
| Llama-3.2-3B-Instruct | SimNPO [F0] | 0.7154 | 0.9200 | 0.7576 | 0.8788 | 0.6509 | -0.18 | 9.6963 | 1.0079 | 0.0079 |
| Qwen3.5-4B | GA [F0] | 0.3820 | 0.1600 | 0.3030 | 0.2424 | 0.6930 | -4.39 | 565697.9218 | 66898.1589 | 11.1109 |
| Qwen3.5-4B | GA [F0F1] | 0.3745 | 0.1200 | 0.3030 | 0.2424 | 0.6912 | -4.56 | 233689.9052 | 27635.6405 | 10.2269 |
| Qwen3.5-4B | GSU-C2 [F0] | 0.3858 | 0.1600 | 0.3030 | 0.2424 | 0.7000 | -3.68 | 663393.6673 | 78451.4372 | 11.2702 |
| Qwen3.5-4B | GradDiff [F0] | 0.6554 | 0.6133 | 0.3636 | 0.4848 | 0.7404 | 0.35 | 10.9926 | 1.3000 | 0.2623 |
| Qwen3.5-4B | NPO [F0] | 0.6667 | 0.8667 | 0.4242 | 0.6970 | 0.7404 | 0.35 | 8.4632 | 1.0008 | 0.0008 |
| Qwen3.5-4B | Original [F0] | 0.6517 | 0.8667 | 0.7273 | 0.6970 | 0.7368 | 0.00 | 8.4561 | 1.0000 | 0.0000 |
| Qwen3.5-4B | RMU [F0] | 0.6929 | 0.8800 | 0.5455 | 0.6364 | 0.7368 | 0.00 | 8.6368 | 1.0214 | 0.0211 |
| Qwen3.5-4B | SimNPO [F0] | 0.7154 | 0.9867 | 0.7576 | 0.7879 | 0.7509 | 1.40 | 8.4870 | 1.0037 | 0.0036 |
