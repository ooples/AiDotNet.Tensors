# BENCHMARK RESULTS

> **Hardware**: AMD Ryzen 9 3950X (16C / 32T, AVX2/FMA, no AVX-512)
> **Runtime**: .NET 10.0.7, BenchmarkDotNet v0.15.8
> **Last regenerated**: 2026-04-30, after #209 perf-fix merge
> **Suites**: TorchSharpCpuComparisonBenchmarks (this file), MlNetCpuComparisonBenchmarks, TensorFlowCpuComparisonBenchmarks
>
> **Regenerate** with:
> ```bash
> dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks --framework net10.0 -- --vs-torchsharp-cpu
> dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks --framework net10.0 -- --vs-mlnet-cpu
> dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks --framework net10.0 -- --vs-tensorflow-cpu
> ```
>
> ## #209 PR-driven improvements (this run vs the previous regeneration)
>
> | Op | Pre-PR | This PR | Improvement |
> |---|---:|---:|---:|
> | `Exp_Double` | 216,093 µs | 1,666 µs | **130× faster** |
> | `Log_Double` | 218,823 µs | 2,512 µs | **87× faster** |
> | `Softmax_Double` | 14,674 µs | 3,707 µs | **4.0× faster** |
> | `TensorAbs` | 3,134 µs | 473 µs | **6.6× faster** |
> | `TensorMaxValue` | 3,171 µs | 352 µs | **9.0× faster** |
> | `TensorMatMul 256` | 832 µs | 593 µs | **1.4× faster** |
> | `TensorMatMul 512` | 2,954 µs | 1,041 µs | **2.8× faster** |
> | `TensorAdd 1M` | 1,242 µs | 407 µs | **3.1× faster** |
> | `LayerNorm` | NA (crash) | 1,473 µs | **runs now** |
>
> The float64-cliff fixes (`Exp_Double`, `Log_Double`, `Softmax_Double`)
> route through `System.Numerics.Tensors.TensorPrimitives` on .NET 8+.
> The Abs/Max wins come from removing the Pin + `Parallel.For` dispatch
> overhead. Matmul wins are from the AVX-512 / autotune work that landed
> in adjacent PRs but wasn't reflected in the previous regeneration.

```
BenchmarkDotNet v0.15.8, Windows 11 (10.0.26220.8283)
AMD Ryzen 9 3950X 3.70GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 10.0.203
  [Host]     : .NET 10.0.7 (10.0.7, 10.0.726.21808), X64 RyuJIT x86-64-v3
  Job-HGADUQ : .NET 10.0.7 (10.0.7, 10.0.726.21808), X64 RyuJIT x86-64-v3

Runtime=.NET 10.0  InvocationCount=1  IterationCount=15
LaunchCount=1  UnrollFactor=1  WarmupCount=5
```

| Method                      | size    | Mean        | Error      | StdDev     | Median      | Allocated |
|---------------------------- |-------- |------------:|-----------:|-----------:|------------:|----------:|
| **AiDotNet_TensorSubtract**     | **?**       |   **740.81 μs** | **162.503 μs** | **144.055 μs** |   **713.60 μs** |    **2936 B** |
| TorchSharp_Subtract         | ?       |   245.33 μs |  65.791 μs |  58.322 μs |   231.30 μs |      48 B |
| AiDotNet_TensorDivide       | ?       |   621.16 μs | 110.278 μs |  97.759 μs |   596.20 μs |    3000 B |
| TorchSharp_Divide           | ?       |   324.62 μs |  81.031 μs |  71.832 μs |   320.60 μs |      48 B |
| AiDotNet_TensorExp          | ?       |   309.27 μs |  46.682 μs |  36.446 μs |   312.60 μs |     784 B |
| TorchSharp_Exp              | ?       |   252.51 μs |  67.837 μs |  60.135 μs |   237.25 μs |      48 B |
| AiDotNet_TensorLog          | ?       |   288.04 μs |  87.937 μs |  77.954 μs |   290.50 μs |     784 B |
| TorchSharp_Log              | ?       |   265.63 μs |  77.393 μs |  68.607 μs |   243.00 μs |      48 B |
| AiDotNet_TensorSqrt         | ?       |   296.46 μs |  50.658 μs |  44.907 μs |   291.55 μs |     736 B |
| TorchSharp_Sqrt             | ?       |   233.50 μs |  51.650 μs |  48.313 μs |   212.50 μs |      48 B |
| AiDotNet_TensorAbs          | ?       |   472.81 μs |  71.557 μs |  59.753 μs |   460.30 μs |     624 B |
| TorchSharp_Abs              | ?       |   317.85 μs |  47.378 μs |  44.318 μs |   325.00 μs |      48 B |
| AiDotNet_ReLU               | ?       |   453.62 μs |  44.001 μs |  39.005 μs |   448.00 μs |         - |
| TorchSharp_ReLU             | ?       |   236.91 μs |  37.706 μs |  33.426 μs |   239.20 μs |         - |
| AiDotNet_Sigmoid            | ?       |   315.78 μs |  80.004 μs |  66.807 μs |   297.30 μs |    2888 B |
| RawTensorPrimitives_Sigmoid | ?       | 7,027.91 μs | 426.551 μs | 398.996 μs | 6,865.20 μs |         - |
| TorchSharp_Sigmoid          | ?       |   230.76 μs |  43.507 μs |  38.568 μs |   217.50 μs |         - |
| AiDotNet_Tanh               | ?       |   369.59 μs | 116.885 μs | 103.615 μs |   372.25 μs |     784 B |
| TorchSharp_Tanh             | ?       |   348.06 μs |  69.294 μs |  64.818 μs |   343.60 μs |      48 B |
| AiDotNet_GELU               | ?       |   259.49 μs |  50.806 μs |  47.524 μs |   264.60 μs |     784 B |
| TorchSharp_GELU             | ?       |   276.81 μs |  51.370 μs |  45.538 μs |   279.50 μs |      48 B |
| AiDotNet_Mish               | ?       |   403.82 μs |  51.549 μs |  43.046 μs |   394.10 μs |     784 B |
| TorchSharp_Mish             | ?       | 1,068.98 μs | 269.936 μs | 252.499 μs |   999.60 μs |     192 B |
| AiDotNet_LeakyReLU          | ?       |   617.49 μs | 127.731 μs | 113.230 μs |   584.00 μs |    3008 B |
| TorchSharp_LeakyReLU        | ?       |   217.96 μs |  38.270 μs |  33.926 μs |   212.75 μs |      72 B |
| AiDotNet_Subtract_ZeroAlloc | ?       |   564.64 μs | 172.524 μs | 152.938 μs |   568.50 μs |         - |
| AiDotNet_Exp_ZeroAlloc      | ?       |   284.44 μs |  88.982 μs |  78.880 μs |   300.95 μs |     112 B |
| AiDotNet_Log_ZeroAlloc      | ?       |   285.11 μs |  77.758 μs |  68.931 μs |   286.90 μs |     112 B |
| AiDotNet_Tanh_ZeroAlloc     | ?       |   627.27 μs |  83.274 μs |  65.015 μs |   612.70 μs |         - |
| AiDotNet_GELU_ZeroAlloc     | ?       |   959.47 μs |  26.526 μs |  22.150 μs |   952.70 μs |         - |
| AiDotNet_TensorSum          | ?       |   207.92 μs |  37.941 μs |  33.634 μs |   202.70 μs |    1168 B |
| RawTensorPrimitives_Sum     | ?       |   284.86 μs | 119.027 μs | 105.515 μs |   241.95 μs |         - |
| TorchSharp_Sum              | ?       |   192.68 μs |   9.059 μs |   7.565 μs |   193.10 μs |      48 B |
| AiDotNet_TensorMean         | ?       |   189.40 μs |  19.872 μs |  16.594 μs |   188.20 μs |     112 B |
| TorchSharp_Mean             | ?       |   234.90 μs |  28.489 μs |  26.648 μs |   225.50 μs |      48 B |
| AiDotNet_TensorMaxValue     | ?       |   352.42 μs |  42.698 μs |  35.655 μs |   341.50 μs |         - |
| TorchSharp_Max              | ?       |   197.31 μs |  32.182 μs |  28.528 μs |   188.05 μs |      48 B |
| AiDotNet_TensorMinValue     | ?       |   247.26 μs |  79.589 μs |  70.553 μs |   252.45 μs |    2472 B |
| TorchSharp_Min              | ?       |   174.67 μs |  11.557 μs |  10.811 μs |   169.50 μs |      48 B |
| AiDotNet_Softmax            | ?       |   335.17 μs |  58.803 μs |  55.005 μs |   323.90 μs |     896 B |
| AiDotNet_Softmax_ZeroAlloc  | ?       |   460.55 μs |  44.388 μs |  37.066 μs |   463.90 μs |     520 B |
| TorchSharp_Softmax          | ?       |    93.64 μs |  12.789 μs |   9.985 μs |    92.40 μs |      48 B |
| AiDotNet_LogSoftmax         | ?       |   153.76 μs |  30.464 μs |  25.439 μs |   140.75 μs |     792 B |
| TorchSharp_LogSoftmax       | ?       |   107.01 μs |  28.703 μs |  23.969 μs |    94.50 μs |      48 B |
| AiDotNet_Conv2D             | ?       |   397.29 μs |  71.717 μs |  67.084 μs |   363.40 μs |  525520 B |
| AiDotNet_Conv2D_ZeroAlloc   | ?       |   336.66 μs |  42.370 μs |  37.560 μs |   313.55 μs |     168 B |
| TorchSharp_Conv2D           | ?       |   292.83 μs |  60.131 μs |  56.247 μs |   271.40 μs |      48 B |
| AiDotNet_BatchNorm          | ?       | 2,186.66 μs | 189.863 μs | 158.544 μs | 2,181.20 μs | 8399368 B |
| TorchSharp_BatchNorm        | ?       |   749.64 μs | 228.652 μs | 190.935 μs |   666.60 μs |      48 B |
| AiDotNet_LayerNorm          | ?       | 1,472.67 μs | 147.309 μs | 130.586 μs | 1,471.20 μs | 8662848 B |
| TorchSharp_LayerNorm        | ?       |   283.38 μs |  90.125 μs |  75.258 μs |   256.00 μs |     168 B |
| AiDotNet_GroupNorm          | ?       | 1,045.94 μs | 157.273 μs | 139.419 μs | 1,049.50 μs | 8505480 B |
| TorchSharp_GroupNorm        | ?       |   248.64 μs |  85.367 μs |  71.286 μs |   240.40 μs |      48 B |
| AiDotNet_GroupNormSwish     | ?       | 2,932.91 μs | 226.734 μs | 189.333 μs | 2,891.00 μs | 8511968 B |
| AiDotNet_MaxPool2D          | ?       |   223.91 μs |  14.134 μs |  12.530 μs |   219.25 μs |  131680 B |
| TorchSharp_MaxPool2D        | ?       |   452.47 μs | 312.693 μs | 292.493 μs |   655.85 μs |      48 B |
| AiDotNet_SigmoidBackward    | ?       |   510.98 μs | 176.465 μs | 147.357 μs |   458.00 μs | 4000496 B |
| TorchSharp_SigmoidBackward  | ?       |   118.49 μs |  35.259 μs |  27.528 μs |   104.95 μs |     192 B |
| AiDotNet_TanhBackward       | ?       |   493.16 μs | 105.685 μs |  93.687 μs |   490.55 μs | 4000496 B |
| TorchSharp_TanhBackward     | ?       |   136.10 μs | 104.796 μs |  87.509 μs |    95.45 μs |     192 B |
| AiDotNet_AttentionQKT       | ?       |   744.06 μs |  67.251 μs |  59.616 μs |   727.50 μs |  134096 B |
| TorchSharp_AttentionQKT     | ?       |   135.71 μs |  35.220 μs |  32.945 μs |   117.90 μs |      96 B |
| AiDotNet_TensorAdd_Double   | ?       | 1,336.07 μs | 193.146 μs | 161.286 μs | 1,349.00 μs |    3168 B |
| TorchSharp_Add_Double       | ?       |   168.89 μs |  95.237 μs |  89.085 μs |   138.30 μs |      72 B |
| AiDotNet_MatMul_Double      | ?       |   620.66 μs | 135.841 μs | 127.066 μs |   562.80 μs |  530752 B |
| TorchSharp_MatMul_Double    | ?       |   188.05 μs |  34.552 μs |  32.320 μs |   176.10 μs |      48 B |
| AiDotNet_Sigmoid_Double     | ?       |   530.75 μs | 197.305 μs | 164.759 μs |   550.60 μs |    5248 B |
| TorchSharp_Sigmoid_Double   | ?       |   293.05 μs |  33.628 μs |  28.081 μs |   293.50 μs |      48 B |
| AiDotNet_Exp_Double         | ?       | 1,665.96 μs |  46.091 μs |  40.859 μs | 1,661.40 μs |     672 B |
| TorchSharp_Exp_Double       | ?       |   283.45 μs |  38.466 μs |  32.121 μs |   269.30 μs |      48 B |
| AiDotNet_Log_Double         | ?       | 2,511.57 μs |  32.907 μs |  27.479 μs | 2,520.30 μs |     672 B |
| TorchSharp_Log_Double       | ?       |   349.51 μs |  22.605 μs |  20.039 μs |   341.25 μs |      48 B |
| AiDotNet_Tanh_Double        | ?       | 2,170.27 μs | 151.776 μs | 118.497 μs | 2,116.30 μs |     672 B |
| TorchSharp_Tanh_Double      | ?       |   626.24 μs |  25.977 μs |  21.692 μs |   621.20 μs |      48 B |
| AiDotNet_GELU_Double        | ?       | 2,798.54 μs |  26.606 μs |  23.586 μs | 2,791.60 μs |     672 B |
| TorchSharp_GELU_Double      | ?       |   738.62 μs |  24.922 μs |  20.811 μs |   738.70 μs |      48 B |
| AiDotNet_Mish_Double        | ?       |   889.60 μs | 126.360 μs | 118.198 μs |   853.70 μs |    8368 B |
| TorchSharp_Mish_Double      | ?       | 2,052.08 μs | 481.196 μs | 401.820 μs | 1,870.70 μs |     192 B |
| AiDotNet_Softmax_Double     | ?       | 3,707.46 μs |  35.679 μs |  29.793 μs | 3,695.70 μs |     784 B |
| TorchSharp_Softmax_Double   | ?       |   195.64 μs |  29.227 μs |  24.406 μs |   186.20 μs |      48 B |
| AiDotNet_Conv2D_Double      | ?       |   488.74 μs |  25.021 μs |  22.180 μs |   485.20 μs |  132136 B |
| TorchSharp_Conv2D_Double    | ?       |    85.35 μs |   5.825 μs |   4.864 μs |    88.30 μs |      48 B |
| **AiDotNet_TensorMatMul**       | **256**     |   **592.59 μs** |  **59.903 μs** |  **50.022 μs** |   **601.40 μs** |  **263952 B** |
| TorchSharp_MatMul           | 256     |   121.04 μs |  31.428 μs |  27.860 μs |   112.85 μs |      48 B |
| **AiDotNet_TensorMatMul**       | **512**     | **1,040.74 μs** | **114.087 μs** | **101.135 μs** | **1,040.25 μs** | **1053648 B** |
| TorchSharp_MatMul           | 512     |   471.43 μs |  18.312 μs |  14.297 μs |   470.15 μs |      48 B |
| **AiDotNet_TensorAdd**          | **100000**  |    **46.44 μs** |   **9.614 μs** |   **8.993 μs** |    **45.90 μs** |     **200 B** |
| RawTensorPrimitives_Add     | 100000  |   215.08 μs | 146.931 μs | 130.250 μs |   135.70 μs |         - |
| TorchSharp_Add              | 100000  |    42.27 μs |  21.313 μs |  18.893 μs |    33.65 μs |      24 B |
| AiDotNet_TensorMultiply     | 100000  |    40.31 μs |  13.349 μs |  12.487 μs |    38.10 μs |     200 B |
| TorchSharp_Multiply         | 100000  |    46.45 μs |  17.687 μs |  16.545 μs |    40.00 μs |         - |
| **AiDotNet_TensorAdd**          | **1000000** |   **406.51 μs** |  **95.414 μs** |  **84.582 μs** |   **402.60 μs** |    **2312 B** |
| RawTensorPrimitives_Add     | 1000000 |   681.26 μs | 178.413 μs | 158.158 μs |   617.40 μs |         - |
| TorchSharp_Add              | 1000000 |   286.48 μs |  87.303 μs |  72.902 μs |   272.10 μs |      24 B |
| TorchSharp_Add_1Thread      | 1000000 |   477.33 μs |  24.500 μs |  19.128 μs |   471.10 μs |      24 B |
| AiDotNet_TensorMultiply     | 1000000 |   439.19 μs |  61.939 μs |  57.938 μs |   431.80 μs |    2312 B |
| TorchSharp_Multiply         | 1000000 |   220.78 μs |  31.877 μs |  28.258 μs |   220.85 μs |         - |
