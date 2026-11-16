# Same-run native-cuDNN bucket and MLP comparisons

Latency is the median of three trial medians; memory is maximum peak allocation over ranks/trials.
Positive memory_reduction_vs_tp means less allocation; latency_ratio_vs_tp > 1 means slower.

| block | batch | seq_len | method | effective_bucket | latency_ms | peak_alloc_MiB | latency_ratio_vs_tp | memory_reduction_vs_tp |
|---|---|---|---|---|---|---|---|---|
| attn | 1 | 8192 | tp | - | 0.439 | 152 | 1 | 0 |
| attn | 1 | 8192 | tsp | 8 | 1.41 | 184 | 3.21 | -0.21 |
| attn | 1 | 8192 | tsp | 4 | 2.2 | 144 | 5.01 | 0.0526 |
| attn | 1 | 8192 | tsp | 2 | 3.59 | 124 | 8.19 | 0.184 |
| attn | 1 | 8192 | tsp | 1 | 6.59 | 114 | 15 | 0.25 |
| attn | 1 | 32768 | tp | - | 2.06 | 536 | 1 | 0 |
| attn | 1 | 32768 | tsp | 8 | 3.13 | 616 | 1.52 | -0.149 |
| attn | 1 | 32768 | tsp | 4 | 3.41 | 456 | 1.65 | 0.149 |
| attn | 1 | 32768 | tsp | 2 | 5.17 | 376 | 2.51 | 0.298 |
| attn | 1 | 32768 | tsp | 1 | 9.82 | 336 | 4.76 | 0.373 |
| attn | 4 | 8192 | tp | - | 1.11 | 536 | 1 | 0 |
| attn | 4 | 8192 | tsp | 8 | 1.95 | 616 | 1.75 | -0.149 |
| attn | 4 | 8192 | tsp | 4 | 2.17 | 456 | 1.95 | 0.149 |
| attn | 4 | 8192 | tsp | 2 | 3.64 | 376 | 3.27 | 0.298 |
| attn | 4 | 8192 | tsp | 1 | 6.61 | 336 | 5.93 | 0.373 |
| mlp | 1 | 8192 | tp | - | 0.644 | 248 | 1 | 0 |
| mlp | 1 | 8192 | tsp | - | 0.669 | 200 | 1.04 | 0.193 |
| mlp | 1 | 32768 | tp | - | 2 | 824 | 1 | 0 |
| mlp | 1 | 32768 | tsp | - | 1.92 | 488 | 0.961 | 0.408 |
| mlp | 4 | 8192 | tp | - | 2 | 824 | 1 | 0 |
| mlp | 4 | 8192 | tsp | - | 1.88 | 488 | 0.94 | 0.408 |
