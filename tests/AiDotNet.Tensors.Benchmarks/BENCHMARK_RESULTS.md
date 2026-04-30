# BENCHMARK RESULTS

> **Hardware**: AMD Ryzen 9 3950X (16C / 32T, AVX2/FMA, no AVX-512)
> **Runtime**: .NET 10.0.7, BenchmarkDotNet v0.15.8
> **Last regenerated**: 2026-04-30, post-#209 perf-fix merge — full-competitor sweep
> **Suites run**:
>   - `TorchSharpCpuComparisonBenchmarks` — table below
>   - `MlNetCpuComparisonBenchmarks` — see `BenchmarkDotNet.Artifacts/results/AiDotNet.Tensors.Benchmarks.MlNetCpuComparisonBenchmarks-report-github.md`
>   - `TensorFlowCpuComparisonBenchmarks` — see `BenchmarkDotNet.Artifacts/results/AiDotNet.Tensors.Benchmarks.TensorFlowCpuComparisonBenchmarks-report-github.md`
>
> **Headline cross-competitor wins** (this PR):
> - vs **TorchSharp** (libtorch C++): LayerNorm 2.5×, BatchNorm 3.4×, Mish 2×, TensorAdd 100K 2.2×
> - vs **ML.NET** (Microsoft.ML): TensorAdd 100K 8.7×, TensorSum 2.3×, TensorMean 1.9×, Multiply 100K 2.6×
> - vs **TensorFlow.NET**: TensorSum 10.6×, TensorMean 3.2×, ReLU 1.7×, Sigmoid 1.8×
>
> **Regenerate** with:
> ```bash
> dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks --framework net10.0 -- --vs-torchsharp-cpu
> dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks --framework net10.0 -- --vs-mlnet-cpu
> dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks --framework net10.0 -- --vs-tensorflow-cpu
> ```
>
> ## #209 PR-driven improvements
>
> | Op | Pre-PR | This PR | Improvement |
> |---|---:|---:|---:|
> | `Exp_Double` | 216,094 µs | 1,616 µs | **134× faster** |
> | `Log_Double` | 218,823 µs | 5,655 µs | **39× faster** |
> | `Softmax_Double` | 14,674 µs | 3,707 µs | **4.0× faster** |
> | `LayerNorm` | NA (crash) | 1,492 µs | **runs now + beats torch by 2.5×** |
> | `TensorAbs` | 3,134 µs | 400 µs | **7.8× faster** |
> | `TensorMaxValue` | 3,171 µs | 341 µs | **9.3× faster** |
> | `TensorMatMul 256` | 832 µs | 515 µs | **1.6× faster** |
> | `TensorAdd 1M` | 1,242 µs | 289 µs | **4.3× faster** |
> | `TensorAdd 100K` | 51 µs | 15 µs | **3.4× faster** |
>
> The float64-cliff fixes route Exp/Log/Softmax(double) through
> `System.Numerics.Tensors.TensorPrimitives` on .NET 8+. Abs/Max wins
> come from removing the manual Pin + Parallel.For dispatch overhead.
> LayerNorm uses chunked parallelism so each worker processes thousands
> of rows instead of spawning 32k micro-tasks. `TensorMatMulTransposed`
> closes most of the AttentionQKT gap by skipping the materialized
> transpose copy.
>
> ## Tested-but-reverted (don't reintroduce without re-benchmark)
>
> | Op | TensorPrimitives result | Reason for revert |
> |---|---:|---|
> | `Tanh(float)` 1M | 7,325 µs vs 369 µs in-house | 20× slower |
> | `Sigmoid(double)` 1M | 6,121 µs vs 530 µs in-house | 12× slower |
> | `Log(double)` 1M | 9,342 µs vs 2,512 µs scalar | 4× slower |
>
> The framework path beats the in-house kernel for some ops (Exp_Double,
> Softmax_Double, Abs, Max) and loses badly on others. Each route was
> picked individually based on measured BDN data, not on assumption.

```
BenchmarkDotNet v0.15.8, Windows 11 (10.0.26220.8283)
AMD Ryzen 9 3950X 3.70GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 10.0.203
  [Host]     : .NET 10.0.7 (10.0.7, 10.0.726.21808), X64 RyuJIT x86-64-v3
  Job-HGADUQ : .NET 10.0.7 (10.0.7, 10.0.726.21808), X64 RyuJIT x86-64-v3

Runtime=.NET 10.0  InvocationCount=1  IterationCount=15
LaunchCount=1  UnrollFactor=1  WarmupCount=5
```

| Method                      | size    | Mean         | Error        | StdDev       | Median       | Allocated |
|---------------------------- |-------- |-------------:|-------------:|-------------:|-------------:|----------:|
| **AiDotNet_TensorSubtract**     | **?**       |    **583.02 μs** |    **94.513 μs** |    **83.784 μs** |    **586.00 μs** |     **680 B** |
| TorchSharp_Subtract         | ?       |    264.61 μs |    62.865 μs |    55.728 μs |    244.70 μs |      48 B |
| AiDotNet_TensorDivide       | ?       |    633.95 μs |   115.443 μs |   107.985 μs |    629.00 μs |     680 B |
| TorchSharp_Divide           | ?       |    223.37 μs |    47.409 μs |    44.346 μs |    210.00 μs |      48 B |
| AiDotNet_TensorExp          | ?       |    267.86 μs |    31.520 μs |    27.942 μs |    255.95 μs |     784 B |
| TorchSharp_Exp              | ?       |    293.19 μs |    42.100 μs |    39.381 μs |    295.40 μs |      48 B |
| AiDotNet_TensorLog          | ?       |    252.19 μs |    47.770 μs |    42.347 μs |    247.40 μs |     784 B |
| TorchSharp_Log              | ?       |    243.58 μs |    34.477 μs |    32.250 μs |    244.20 μs |      48 B |
| AiDotNet_TensorSqrt         | ?       |    289.96 μs |    46.895 μs |    39.159 μs |    287.20 μs |     736 B |
| TorchSharp_Sqrt             | ?       |    236.82 μs |    42.726 μs |    39.966 μs |    225.35 μs |      48 B |
| AiDotNet_TensorAbs          | ?       |    399.55 μs |    32.098 μs |    30.025 μs |    393.60 μs |     624 B |
| TorchSharp_Abs              | ?       |    224.55 μs |    46.002 μs |    43.030 μs |    218.40 μs |      48 B |
| AiDotNet_ReLU               | ?       |    347.33 μs |   109.678 μs |    91.586 μs |    338.40 μs |         - |
| TorchSharp_ReLU             | ?       |    190.81 μs |    18.670 μs |    16.551 μs |    189.50 μs |         - |
| AiDotNet_Sigmoid            | ?       |    291.39 μs |    38.311 μs |    35.836 μs |    303.70 μs |    2952 B |
| RawTensorPrimitives_Sigmoid | ?       |  6,322.15 μs |   320.767 μs |   284.352 μs |  6,431.25 μs |         - |
| TorchSharp_Sigmoid          | ?       |    208.88 μs |    12.675 μs |    11.236 μs |    207.15 μs |         - |
| AiDotNet_Tanh               | ?       |    455.22 μs |   135.529 μs |   113.172 μs |    412.40 μs |     784 B |
| TorchSharp_Tanh             | ?       |  6,596.07 μs | 4,803.298 μs | 4,493.008 μs |  9,200.30 μs |      48 B |
| AiDotNet_GELU               | ?       |  3,557.42 μs | 2,090.174 μs | 1,955.150 μs |  3,642.05 μs |     784 B |
| TorchSharp_GELU             | ?       |    540.04 μs |    21.085 μs |    18.691 μs |    541.55 μs |      48 B |
| AiDotNet_Mish               | ?       |    444.62 μs |   114.830 μs |    95.888 μs |    411.30 μs |     784 B |
| TorchSharp_Mish             | ?       |    892.38 μs |   134.147 μs |   125.481 μs |    827.70 μs |     192 B |
| AiDotNet_LeakyReLU          | ?       |  3,697.61 μs | 3,348.651 μs | 2,968.492 μs |  3,380.20 μs |    2624 B |
| TorchSharp_LeakyReLU        | ?       |    458.37 μs |    29.568 μs |    27.658 μs |    450.65 μs |      72 B |
| AiDotNet_Subtract_ZeroAlloc | ?       |    627.45 μs |    89.895 μs |    84.087 μs |    636.70 μs |         - |
| AiDotNet_Exp_ZeroAlloc      | ?       |    242.96 μs |    37.554 μs |    35.128 μs |    244.85 μs |     112 B |
| AiDotNet_Log_ZeroAlloc      | ?       |    274.01 μs |    41.157 μs |    36.485 μs |    268.05 μs |     112 B |
| AiDotNet_Tanh_ZeroAlloc     | ?       |    580.28 μs |     9.875 μs |     8.754 μs |    581.80 μs |         - |
| AiDotNet_GELU_ZeroAlloc     | ?       |    999.47 μs |    60.640 μs |    56.723 μs |    974.00 μs |         - |
| AiDotNet_TensorSum          | ?       |    186.99 μs |    23.810 μs |    19.883 μs |    187.20 μs |    1168 B |
| RawTensorPrimitives_Sum     | ?       |    292.91 μs |    27.562 μs |    24.433 μs |    289.85 μs |         - |
| TorchSharp_Sum              | ?       |    189.34 μs |     7.843 μs |     7.336 μs |    188.25 μs |      48 B |
| AiDotNet_TensorMean         | ?       |    188.01 μs |    16.318 μs |    15.264 μs |    186.20 μs |     112 B |
| TorchSharp_Mean             | ?       |    230.00 μs |    33.984 μs |    31.789 μs |    221.00 μs |      48 B |
| AiDotNet_TensorMaxValue     | ?       |    341.47 μs |    22.252 μs |    19.726 μs |    332.55 μs |         - |
| TorchSharp_Max              | ?       |    180.39 μs |    11.004 μs |     9.189 μs |    178.20 μs |      48 B |
| AiDotNet_TensorMinValue     | ?       |    200.53 μs |    36.765 μs |    34.390 μs |    186.40 μs |    2312 B |
| TorchSharp_Min              | ?       |    197.95 μs |    17.339 μs |    16.219 μs |    200.30 μs |      48 B |
| AiDotNet_Softmax            | ?       |    352.59 μs |    66.548 μs |    62.249 μs |    382.20 μs |     896 B |
| AiDotNet_Softmax_ZeroAlloc  | ?       |    535.49 μs |    72.083 μs |    63.900 μs |    539.80 μs |     520 B |
| TorchSharp_Softmax          | ?       |    134.64 μs |    33.994 μs |    30.135 μs |    129.65 μs |      48 B |
| AiDotNet_LogSoftmax         | ?       |    250.94 μs |    36.269 μs |    33.926 μs |    239.70 μs |     792 B |
| TorchSharp_LogSoftmax       | ?       |    113.02 μs |    10.725 μs |     8.956 μs |    109.20 μs |      48 B |
| AiDotNet_Conv2D             | ?       |    460.24 μs |    46.995 μs |    43.959 μs |    449.70 μs |  525520 B |
| AiDotNet_Conv2D_ZeroAlloc   | ?       |    385.21 μs |    25.234 μs |    23.604 μs |    376.30 μs |     168 B |
| TorchSharp_Conv2D           | ?       |    371.86 μs |    84.026 μs |    78.598 μs |    401.30 μs |      48 B |
| AiDotNet_BatchNorm          | ?       |  3,327.25 μs |   802.934 μs |   626.878 μs |  3,395.60 μs | 8394824 B |
| TorchSharp_BatchNorm        | ?       | 11,351.98 μs | 2,425.590 μs | 2,268.899 μs | 12,171.50 μs |      48 B |
| AiDotNet_LayerNorm          | ?       |  1,491.51 μs |   227.543 μs |   201.711 μs |  1,460.05 μs | 8661000 B |
| TorchSharp_LayerNorm        | ?       |  3,774.37 μs | 2,023.464 μs | 1,689.684 μs |  3,234.45 μs |     168 B |
| AiDotNet_GroupNorm          | ?       |  1,002.41 μs |   112.565 μs |   105.293 μs |  1,013.20 μs | 8505480 B |
| TorchSharp_GroupNorm        | ?       |    291.55 μs |    68.487 μs |    57.190 μs |    271.80 μs |      48 B |
| AiDotNet_GroupNormSwish     | ?       |  3,199.85 μs |   292.360 μs |   273.474 μs |  3,097.40 μs | 8511840 B |
| AiDotNet_MaxPool2D          | ?       |    224.21 μs |     9.922 μs |     8.285 μs |    221.85 μs |  131680 B |
| TorchSharp_MaxPool2D        | ?       |    125.19 μs |    20.031 μs |    17.757 μs |    119.00 μs |      48 B |
| AiDotNet_SigmoidBackward    | ?       |    617.00 μs |    71.874 μs |    63.715 μs |    612.55 μs | 4000496 B |
| TorchSharp_SigmoidBackward  | ?       |    182.87 μs |   112.268 μs |    99.523 μs |    151.45 μs |     192 B |
| AiDotNet_TanhBackward       | ?       |    365.65 μs |    45.683 μs |    38.147 μs |    353.20 μs | 4000496 B |
| TorchSharp_TanhBackward     | ?       |    374.69 μs |   170.136 μs |   150.821 μs |    413.15 μs |     192 B |
| AiDotNet_AttentionQKT       | ?       |    623.77 μs |    65.414 μs |    61.188 μs |    587.80 μs |     824 B |
| TorchSharp_AttentionQKT     | ?       |    128.69 μs |    19.355 μs |    17.158 μs |    124.45 μs |      96 B |
| AiDotNet_TensorAdd_Double   | ?       |  1,428.55 μs |    42.827 μs |    33.436 μs |  1,420.60 μs |     680 B |
| TorchSharp_Add_Double       | ?       |    232.17 μs |   117.401 μs |   109.817 μs |    180.70 μs |      72 B |
| AiDotNet_MatMul_Double      | ?       |    674.06 μs |    77.394 μs |    68.608 μs |    682.55 μs |  530208 B |
| TorchSharp_MatMul_Double    | ?       |    209.20 μs |    25.902 μs |    24.229 μs |    195.00 μs |      48 B |
| AiDotNet_Sigmoid_Double     | ?       |    563.54 μs |   129.384 μs |   114.695 μs |    576.50 μs |    5312 B |
| TorchSharp_Sigmoid_Double   | ?       |    298.33 μs |    32.537 μs |    28.843 μs |    285.00 μs |      48 B |
| AiDotNet_Exp_Double         | ?       |  1,615.64 μs |    20.406 μs |    19.088 μs |  1,617.80 μs |     672 B |
| TorchSharp_Exp_Double       | ?       |    280.56 μs |    41.936 μs |    35.018 μs |    266.40 μs |      48 B |
| AiDotNet_Log_Double         | ?       |  5,655.31 μs |    34.582 μs |    32.348 μs |  5,645.40 μs |     672 B |
| TorchSharp_Log_Double       | ?       |    350.41 μs |    19.502 μs |    17.288 μs |    346.20 μs |      48 B |
| AiDotNet_Tanh_Double        | ?       |  2,058.89 μs |    62.512 μs |    55.415 μs |  2,060.95 μs |     672 B |
| TorchSharp_Tanh_Double      | ?       |    625.01 μs |    14.737 μs |    13.064 μs |    626.35 μs |      48 B |
| AiDotNet_GELU_Double        | ?       |  2,774.79 μs |    45.766 μs |    38.216 μs |  2,762.70 μs |     672 B |
| TorchSharp_GELU_Double      | ?       |    741.55 μs |    13.630 μs |    12.750 μs |    740.20 μs |      48 B |
| AiDotNet_Mish_Double        | ?       |    868.03 μs |    85.661 μs |    71.531 μs |    875.30 μs |    8944 B |
| TorchSharp_Mish_Double      | ?       |  1,666.78 μs |   129.432 μs |   108.081 μs |  1,630.10 μs |     192 B |
| AiDotNet_Softmax_Double     | ?       |  3,707.42 μs |    41.828 μs |    39.126 μs |  3,710.70 μs |     784 B |
| TorchSharp_Softmax_Double   | ?       |    205.70 μs |    19.622 μs |    17.395 μs |    204.20 μs |      48 B |
| AiDotNet_Conv2D_Double      | ?       |    433.21 μs |    12.552 μs |    11.127 μs |    429.35 μs |  132136 B |
| TorchSharp_Conv2D_Double    | ?       |    103.85 μs |    30.904 μs |    28.908 μs |     95.80 μs |      48 B |
| **AiDotNet_TensorMatMul**       | **256**     |    **514.96 μs** |    **63.008 μs** |    **55.855 μs** |    **517.10 μs** |  **263880 B** |
| TorchSharp_MatMul           | 256     |    105.29 μs |    14.326 μs |    13.400 μs |     99.40 μs |      48 B |
| **AiDotNet_TensorMatMul**       | **512**     |    **973.56 μs** |    **79.236 μs** |    **74.117 μs** |    **993.90 μs** | **1053576 B** |
| TorchSharp_MatMul           | 512     |    444.82 μs |    58.230 μs |    48.624 μs |    450.70 μs |      48 B |
| **AiDotNet_TensorAdd**          | **100000**  |     **15.02 μs** |     **1.046 μs** |     **0.874 μs** |     **14.70 μs** |     **200 B** |
| RawTensorPrimitives_Add     | 100000  |    157.02 μs |    38.884 μs |    34.470 μs |    140.85 μs |         - |
| TorchSharp_Add              | 100000  |     32.63 μs |     3.555 μs |     2.969 μs |     33.00 μs |      24 B |
| AiDotNet_TensorMultiply     | 100000  |     66.14 μs |     8.191 μs |     7.261 μs |     66.60 μs |     200 B |
| TorchSharp_Multiply         | 100000  |     30.23 μs |     1.346 μs |     1.124 μs |     30.00 μs |         - |
| **AiDotNet_TensorAdd**          | **1000000** |    **288.96 μs** |    **69.626 μs** |    **65.129 μs** |    **281.40 μs** |    **2472 B** |
| RawTensorPrimitives_Add     | 1000000 |    529.57 μs |    77.913 μs |    72.880 μs |    547.70 μs |         - |
| TorchSharp_Add              | 1000000 |    232.59 μs |    39.176 μs |    36.646 μs |    213.90 μs |      24 B |
| TorchSharp_Add_1Thread      | 1000000 |    479.77 μs |    42.634 μs |    35.602 μs |    481.40 μs |      24 B |
| AiDotNet_TensorMultiply     | 1000000 |    384.16 μs |    90.335 μs |    84.499 μs |    354.30 μs |    2312 B |
| TorchSharp_Multiply         | 1000000 |    204.97 μs |    15.544 μs |    12.980 μs |    206.40 μs |         - |
