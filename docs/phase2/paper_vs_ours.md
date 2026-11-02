## R3 and R4: split points (Table II) and splitting overhead (Table IV)
| model | paper found / used | ours found / used | paper overhead | ours p50 | ours p99 |
|---|---|---|---|---|---|
| alexnet | 10 / 10 | 19 / 10 | +7.5% | +8.7% | +78.5% |
| inception_v4 | 23 / 6 | 27 / 6 | +5.3% | -1.5% | +7.8% |
| resnet18 | 11 / 11 | 21 / 11 | +48.0% | +9.9% | +27.4% |
| vgg19 | 21 / 5 | 43 / 5 | +73.0% | +2.5% | +33.4% |

Report the p50 overhead: the p99 figure adds up p99s of different sessions, which is noisy.
Ours use the paper's number of cuts for each model (10, 5, 11 and 6).

## R2: schedulability against utilisation (Fig. 4)
Paper: up to 96% against 10% for the baseline (Jetson Orin, 12 tasks, U=0.9).
Ours: share of task sets with zero deadline misses. 'proposed' = accepted by the analysis AND zero misses when run.

**alexnet_n3.csv** (alexnet.onnx, 3 tasks, heuristic)

| U | sets | baseline | analysis accepts | proposed (validated) |
|---|---|---|---|---|
| 0.6 | 50 | 58% | 100% | 82% |
| 0.7 | 50 | 8% | 100% | 22% |
| 0.8 | 50 | 18% | 100% | 44% |
| 0.9 | 5 | 20% | 100% | 60% |

**alexnet_n3_u09.csv** (alexnet.onnx, 3 tasks, heuristic)

| U | sets | baseline | analysis accepts | proposed (validated) |
|---|---|---|---|---|
| 0.9 | 5 | 20% | 100% | 40% |

**alexnet_n3_h5_u06.csv** (alexnet.onnx, 3 tasks, heuristic)

| U | sets | baseline | analysis accepts | proposed (validated) |
|---|---|---|---|---|
| 0.6 | 20 | 65% | 100% | 85% |

**alexnet_n3_o5_u06.csv** (alexnet.onnx, 3 tasks, optimal)

| U | sets | baseline | analysis accepts | proposed (validated) |
|---|---|---|---|---|
| 0.6 | 20 | 55% | 100% | 50% |

**alexnet_n3_h5_u07.csv** (alexnet.onnx, 3 tasks, heuristic)

| U | sets | baseline | analysis accepts | proposed (validated) |
|---|---|---|---|---|
| 0.7 | 20 | 35% | 100% | 65% |

**alexnet_n3_o5_u07.csv** (alexnet.onnx, 3 tasks, optimal)

| U | sets | baseline | analysis accepts | proposed (validated) |
|---|---|---|---|---|
| 0.7 | 20 | 40% | 100% | 65% |

**alexnet_n3_h5_u08.csv** (alexnet.onnx, 3 tasks, heuristic)

| U | sets | baseline | analysis accepts | proposed (validated) |
|---|---|---|---|---|
| 0.8 | 20 | 10% | 100% | 30% |

**alexnet_n3_o5_u08.csv** (alexnet.onnx, 3 tasks, optimal)

| U | sets | baseline | analysis accepts | proposed (validated) |
|---|---|---|---|---|
| 0.8 | 20 | 20% | 100% | 35% |

**alexnet_n3_h5_u09.csv** (alexnet.onnx, 3 tasks, heuristic)

| U | sets | baseline | analysis accepts | proposed (validated) |
|---|---|---|---|---|
| 0.9 | 20 | 20% | 85% | 35% |

**alexnet_n3_o5_u09.csv** (alexnet.onnx, 3 tasks, optimal)

| U | sets | baseline | analysis accepts | proposed (validated) |
|---|---|---|---|---|
| 0.9 | 20 | 25% | 85% | 25% |

**yolo_n3_B_v2.csv** (yolov8n.onnx, 3 tasks, heuristic, multi-tensor cuts)

| U | sets | baseline | analysis accepts | proposed (validated) |
|---|---|---|---|---|
| 0.6 | 10 | 20% | 100% | 90% |
| 0.7 | 10 | 10% | 100% | 80% |
| 0.8 | 10 | 0% | 100% | 60% |
| 0.9 | 10 | 20% | 90% | 70% |

**mixed_quick_test.csv** (mixed, 4 tasks, heuristic)

| U | sets | baseline | analysis accepts | proposed (validated) |
|---|---|---|---|---|
| 0.6 | 2 | 50% | 100% | 50% |
| 0.7 | 2 | 50% | 100% | 100% |
| 0.8 | 2 | 0% | 100% | 50% |
| 0.9 | 2 | 0% | 100% | 100% |

## R5: optimiser runtime (Fig. 5)
Paper: the heuristic is up to 38% faster than the optimal search.

| file | method | median optimiser s | median profiling s | median configs |
|---|---|---|---|---|
| alexnet_n3.csv | heuristic | 0.00 | 0.00 | 2.00 |
| alexnet_n3_u09.csv | heuristic | 0.00 | 0.00 | 9.00 |
| alexnet_n3_h5_u06.csv | heuristic | 0.00 | 0.00 | 2.00 |
| alexnet_n3_o5_u06.csv | optimal | 0.00 | 0.00 | 2.00 |
| alexnet_n3_h5_u07.csv | heuristic | 0.00 | 0.00 | 2.00 |
| alexnet_n3_o5_u07.csv | optimal | 0.00 | 0.00 | 2.00 |
| alexnet_n3_h5_u08.csv | heuristic | 0.00 | 0.00 | 4.50 |
| alexnet_n3_o5_u08.csv | optimal | 0.00 | 0.00 | 8.00 |
| alexnet_n3_h5_u09.csv | heuristic | 0.00 | 0.00 | 7.00 |
| alexnet_n3_o5_u09.csv | optimal | 0.00 | 0.00 | 14.00 |
| yolo_n3_B_v2.csv | heuristic | 0.00 | 0.00 | 2.00 |
| mixed_quick_test.csv | heuristic | 0.00 | 0.00 | 4.00 |

## R1, R6, R7, R8
| id | result | paper | ours | note |
|---|---|---|---|---|
| R1 | Concurrent against sequential makespan (Fig. 1) | similar makespans, sometimes worse; kernel concurrency as low as 1% | not run yet (makespan_test.py) |  |
| R6 | GPU memory, proposed against baseline (Table V) | -1% to +2% | one fully split AlexNet (8 chunk sessions) 530 MB against 544 MB for one whole-model session: -2.6% | per task; our tasks with the same split share one chunk set, which saves more |
| R7 | Start-up time | at most 11.1 s (Jetson TX2) | 0.59 s for the 8-chunk set; 0.72 s for one whole-model session; 2.28 s for 4 whole-model sessions | different hardware, so only the trend compares |
| R8 | Dispatch cost per chunk | 8 to 13 microseconds | gap between chunks: p50 0.08 to 0.11 ms, p99 0.18 to 0.35 ms (Phase 1) | about 10x the paper's: Python plus ONNX Runtime calls on Windows |

