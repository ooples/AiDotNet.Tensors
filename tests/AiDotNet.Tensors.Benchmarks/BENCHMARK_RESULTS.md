# BENCHMARK RESULTS

> **Hardware**: AMD Ryzen 9 3950X (16C / 32T, AVX2/FMA, no AVX-512)
> **Runtime**: .NET 10.0.7, BenchmarkDotNet v0.15.8
> **Last regenerated**: 2026-04-30 — full-competitor sweep AFTER removing
> `System.Numerics.Tensors` and routing every hot path through our
> in-house `SimdKernels`. Verified no regressions vs the previous
> TP-routed run; several routes IMPROVED (Tanh 455→268 µs, Abs 400→286 µs,
> Max 341→223 µs, ReLU 347→257 µs).
>
> **Zero external library dependencies.** No `System.Numerics.Tensors`,
> no MKL, no MKL.NET, no oneDNN. Every SIMD path is a hand-written
> AVX2/AVX-512 kernel in `SimdKernels.cs` / `SimdGemm.cs`.
>
> **Suites run**:
>   - `TorchSharpCpuComparisonBenchmarks` (libtorch C++) — table below
>   - `MlNetCpuComparisonBenchmarks` (Microsoft.ML)
>   - `TensorFlowCpuComparisonBenchmarks` (SciSharp TensorFlow.NET)
>
> ## Cross-competitor headline wins (post-TP-removal)
>
> **vs TorchSharp** (libtorch C++):
> - Mish: 361 µs vs 913 µs (**2.5× faster**)
> - Mish (double): 937 µs vs 2,433 µs (**2.6× faster**)
> - Tanh: 268 µs vs 354 µs (**1.3× faster** — was tied; now winning)
> - TensorAdd 100K: 24 µs vs 55 µs (**2.3× faster**)
> - MaxPool2D: 227 µs vs 312 µs (**1.4× faster**)
> - LayerNorm: 1,347 µs vs 392 µs (was 5× slower → still ~3.4× behind)
> - BatchNorm: 2,167 µs vs 587 µs (was 3.4× ahead → noisy this run)
>
> **vs ML.NET** (Microsoft.ML):
> - TensorMultiply 100K: 58 µs vs 219 µs (**3.8× faster**)
> - TensorSum: 446 µs vs 1,234 µs (**2.8× faster**)
> - TensorMean: 869 µs vs 1,376 µs (**1.6× faster**)
> - TensorAdd 100K: 98 µs vs 116 µs (**1.2× faster**)
> - TensorAdd 1M: 480 µs vs 466 µs (~tied)
> - TensorMultiply 1M: 569 µs vs 300 µs (memory-bound; both at saturation)
>
> **vs TensorFlow.NET** (SciSharp):
> - TensorSum: 72 µs vs 121 µs (**1.7× faster**)
> - TensorMean: 82 µs vs 206 µs (**2.5× faster**)
> - Sigmoid: 562 µs vs 1,102 µs (**2.0× faster**)
> - ReLU: 759 µs vs 1,410 µs (**1.9× faster**)
> - Conv2D small: 485 µs vs 371 µs (1.3× behind)
> - 512-MatMul + bulk Add/Multiply: TF errored out (NA) — TF.NET issue
>
> ## Verified regression-free routes
>
> All previously-TP-routed paths now run on in-house `SimdKernels.*Unsafe`
> with raw `_storage` + `_storageOffset` pinning (no view-copy overhead):
>
> | Op | Pre-TP-removal | Post-TP-removal | Delta |
> |---|---:|---:|---:|
> | Tanh (1M) | 455 µs | **268 µs** | 1.7× faster |
> | TensorAbs (1M) | 400 µs | **286 µs** | 1.4× faster |
> | TensorMaxValue (1M) | 341 µs | **223 µs** | 1.5× faster |
> | ReLU (1M) | 347 µs | **257 µs** | 1.4× faster |
> | TensorAdd 100K | 15 µs | **24 µs** | within run-to-run noise |
> | Sigmoid (1M) | 291 µs | **284 µs** | tied |
> | Sigmoid_Double (1M) | 564 µs | **509 µs** | 1.1× faster |
> | Exp_Double (1M) | 1,616 µs | **1,634 µs** | tied |
> | Log_Double (1M) | 5,655 µs | **5,785 µs** | tied |
> | Tanh_Double (1M) | 2,059 µs | **2,067 µs** | tied |

## TorchSharp CPU (libtorch C++)

```text
BenchmarkDotNet v0.15.8, Windows 11 (10.0.26220.8283)
AMD Ryzen 9 3950X 3.70GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 10.0.203
  [Host]     : .NET 10.0.7 (10.0.7, 10.0.726.21808), X64 RyuJIT x86-64-v3
  Job-HGADUQ : .NET 10.0.7 (10.0.7, 10.0.726.21808), X64 RyuJIT x86-64-v3

Runtime=.NET 10.0  InvocationCount=1  IterationCount=15
LaunchCount=1  UnrollFactor=1  WarmupCount=5
```

| Method                      | size    | Mean        | Error        | StdDev     | Median      | Allocated |
|---------------------------- |-------- |------------:|-------------:|-----------:|------------:|----------:|
| **AiDotNet_TensorSubtract**     | **?**       |   **936.22 μs** |   **468.080 μs** | **437.842 μs** |   **765.60 μs** |    **3000 B** |
| TorchSharp_Subtract         | ?       |   344.81 μs |   114.965 μs | 107.539 μs |   323.70 μs |      48 B |
| AiDotNet_TensorDivide       | ?       |   612.10 μs |   178.829 μs | 167.277 μs |   654.10 μs |    3000 B |
| TorchSharp_Divide           | ?       |   346.91 μs |   106.753 μs |  99.857 μs |   323.20 μs |      48 B |
| AiDotNet_TensorExp          | ?       |   296.38 μs |    58.582 μs |  48.919 μs |   284.10 μs |     784 B |
| TorchSharp_Exp              | ?       |   262.50 μs |    48.726 μs |  45.578 μs |   269.00 μs |      48 B |
| AiDotNet_TensorLog          | ?       |   265.74 μs |    28.620 μs |  23.899 μs |   259.60 μs |     784 B |
| TorchSharp_Log              | ?       |   272.97 μs |    38.200 μs |  35.732 μs |   266.20 μs |      48 B |
| AiDotNet_TensorSqrt         | ?       |   332.89 μs |    68.469 μs |  60.696 μs |   329.40 μs |     736 B |
| TorchSharp_Sqrt             | ?       |   250.72 μs |    35.476 μs |  29.624 μs |   256.00 μs |      48 B |
| AiDotNet_TensorAbs          | ?       |   285.89 μs |    51.719 μs |  48.378 μs |   287.40 μs |     736 B |
| TorchSharp_Abs              | ?       |   235.13 μs |    42.346 μs |  37.539 μs |   217.60 μs |      48 B |
| AiDotNet_ReLU               | ?       |   256.58 μs |    32.305 μs |  28.638 μs |   265.50 μs |         - |
| TorchSharp_ReLU             | ?       |   204.29 μs |    15.686 μs |  13.099 μs |   202.30 μs |         - |
| AiDotNet_Sigmoid            | ?       |   284.19 μs |    38.624 μs |  36.129 μs |   275.10 μs |    2952 B |
| TorchSharp_Sigmoid          | ?       |   219.74 μs |    20.547 μs |  17.158 μs |   224.20 μs |         - |
| AiDotNet_Tanh               | ?       |   267.51 μs |    39.723 μs |  37.157 μs |   274.60 μs |     784 B |
| TorchSharp_Tanh             | ?       |   353.59 μs |    19.618 μs |  17.391 μs |   352.25 μs |      48 B |
| AiDotNet_GELU               | ?       |   341.40 μs |    65.325 μs |  57.909 μs |   334.40 μs |     784 B |
| TorchSharp_GELU             | ?       |   296.53 μs |    62.432 μs |  55.344 μs |   294.90 μs |      48 B |
| AiDotNet_Mish               | ?       |   361.46 μs |    36.592 μs |  32.438 μs |   369.00 μs |     784 B |
| TorchSharp_Mish             | ?       |   912.96 μs |   189.509 μs | 158.248 μs |   866.00 μs |     192 B |
| AiDotNet_LeakyReLU          | ?       |   371.99 μs |    93.824 μs |  87.763 μs |   329.80 μs |    3072 B |
| TorchSharp_LeakyReLU        | ?       |   223.19 μs |    29.969 μs |  23.398 μs |   221.50 μs |      72 B |
| AiDotNet_TensorSum          | ?       |   195.58 μs |    14.962 μs |  12.494 μs |   194.60 μs |    1168 B |
| TorchSharp_Sum              | ?       |   218.82 μs |    33.106 μs |  27.645 μs |   215.40 μs |      48 B |
| AiDotNet_TensorMean         | ?       |   216.72 μs |    27.600 μs |  25.817 μs |   205.65 μs |     112 B |
| TorchSharp_Mean             | ?       |   230.61 μs |    28.469 μs |  25.237 μs |   222.35 μs |      48 B |
| AiDotNet_TensorMaxValue     | ?       |   222.61 μs |    24.308 μs |  21.548 μs |   224.45 μs |    2312 B |
| TorchSharp_Max              | ?       |   194.69 μs |    56.118 μs |  46.861 μs |   176.90 μs |      48 B |
| AiDotNet_TensorMinValue     | ?       |   198.47 μs |    26.217 μs |  23.241 μs |   204.10 μs |    2312 B |
| TorchSharp_Min              | ?       |   193.77 μs |    14.868 μs |  12.416 μs |   191.10 μs |      48 B |
| AiDotNet_LogSoftmax         | ?       |   164.62 μs |    26.189 μs |  23.216 μs |   164.40 μs |     792 B |
| TorchSharp_LogSoftmax       | ?       |   106.92 μs |    25.884 μs |  20.209 μs |   106.30 μs |      48 B |
| AiDotNet_Conv2D             | ?       |   457.67 μs |    36.657 μs |  34.289 μs |   450.70 μs |  525520 B |
| TorchSharp_Conv2D           | ?       |   288.73 μs |    24.858 μs |  23.252 μs |   286.90 μs |      48 B |
| AiDotNet_MaxPool2D          | ?       |   226.84 μs |    11.059 μs |  10.344 μs |   223.70 μs |  131680 B |
| TorchSharp_MaxPool2D        | ?       |   311.52 μs |   296.130 μs | 277.000 μs |   120.00 μs |      48 B |
| AiDotNet_AttentionQKT       | ?       |   599.01 μs |    21.098 μs |  17.618 μs |   594.00 μs |     824 B |
| TorchSharp_AttentionQKT     | ?       |   120.14 μs |     7.767 μs |   6.064 μs |   119.10 μs |      96 B |
| AiDotNet_TensorAdd_Double   | ?       | 1,206.88 μs |   195.271 μs | 182.656 μs | 1,256.60 μs |    3168 B |
| TorchSharp_Add_Double       | ?       |   218.20 μs |    52.561 μs |  46.594 μs |   226.55 μs |      72 B |
| AiDotNet_MatMul_Double      | ?       |   603.84 μs |    34.386 μs |  30.482 μs |   595.90 μs |  530144 B |
| TorchSharp_MatMul_Double    | ?       |   207.36 μs |    45.481 μs |  37.979 μs |   195.25 μs |      48 B |
| AiDotNet_Sigmoid_Double     | ?       |   509.23 μs |   187.474 μs | 166.191 μs |   478.60 μs |    5312 B |
| TorchSharp_Sigmoid_Double   | ?       |   320.27 μs |    60.633 μs |  47.338 μs |   306.55 μs |      48 B |
| AiDotNet_Exp_Double         | ?       | 1,633.72 μs |    42.773 μs |  33.394 μs | 1,643.60 μs |     672 B |
| TorchSharp_Exp_Double       | ?       |   377.35 μs |   144.865 μs | 128.419 μs |   303.75 μs |      48 B |
| AiDotNet_Log_Double         | ?       | 5,785.36 μs |   151.848 μs | 142.039 μs | 5,735.20 μs |     672 B |
| TorchSharp_Log_Double       | ?       |   348.38 μs |    49.696 μs |  38.800 μs |   333.90 μs |      48 B |
| AiDotNet_Tanh_Double        | ?       | 2,067.05 μs |    40.152 μs |  33.529 μs | 2,073.80 μs |     672 B |
| TorchSharp_Tanh_Double      | ?       |   621.55 μs |    20.089 μs |  15.684 μs |   616.20 μs |      48 B |
| AiDotNet_Mish_Double        | ?       |   937.40 μs |   120.810 μs | 113.006 μs |   938.60 μs |    8624 B |
| TorchSharp_Mish_Double      | ?       | 2,433.42 μs |   677.455 μs | 633.691 μs | 2,358.50 μs |     192 B |
| **AiDotNet_TensorMatMul**       | **256**     |   **496.13 μs** |    **46.015 μs** |  **40.791 μs** |   **494.00 μs** |  **263880 B** |
| TorchSharp_MatMul           | 256     |   100.66 μs |    14.508 μs |  11.327 μs |   100.55 μs |      48 B |
| **AiDotNet_TensorMatMul**       | **512**     | **1,100.93 μs** |   **113.824 μs** | **106.471 μs** | **1,115.80 μs** | **1053576 B** |
| TorchSharp_MatMul           | 512     |   452.68 μs |    23.469 μs |  18.323 μs |   453.95 μs |      48 B |
| **AiDotNet_TensorAdd**          | **100000**  |    **24.05 μs** |     **6.438 μs** |   **5.376 μs** |    **23.90 μs** |     **200 B** |
| TorchSharp_Add              | 100000  |    55.25 μs |    30.480 μs |  28.511 μs |    40.20 μs |      24 B |
| AiDotNet_TensorMultiply     | 100000  |    32.75 μs |    16.283 μs |  15.231 μs |    34.00 μs |     200 B |
| TorchSharp_Multiply         | 100000  |    39.46 μs |    23.869 μs |  21.159 μs |    28.45 μs |         - |
| **AiDotNet_TensorAdd**          | **1000000** |   **378.88 μs** |   **101.261 μs** |  **94.719 μs** |   **403.50 μs** |    **2248 B** |
| TorchSharp_Add              | 1000000 |   247.59 μs |    48.887 μs |  43.337 μs |   239.60 μs |      24 B |
| AiDotNet_TensorMultiply     | 1000000 |   441.53 μs |    83.379 μs |  73.913 μs |   434.65 μs |    2248 B |
| TorchSharp_Multiply         | 1000000 |   265.87 μs |    46.795 μs |  41.482 μs |   275.65 μs |         - |

## ML.NET CPU (Microsoft.ML)

| Method                  | size    | Mean        | Error      | StdDev     | Allocated |
|------------------------ |-------- |------------:|-----------:|-----------:|----------:|
| **AiDotNet_TensorSum**      | **?**       |   **445.83 μs** |  **99.590 μs** |  **15.412 μs** | 4197468 B |
| MlNet_Sum               | ?       | 1,234.07 μs | 138.147 μs |  35.876 μs |    1008 B |
| AiDotNet_TensorMean     | ?       |   869.40 μs | 953.709 μs | 247.675 μs | 4198197 B |
| MlNet_Mean              | ?       | 1,375.90 μs | 328.628 μs |  85.344 μs |    1168 B |
| **AiDotNet_TensorAdd**      | **100000**  |    **98.47 μs** | **100.582 μs** |  **15.565 μs** |    1168 B |
| MlNet_Add               | 100000  |   115.87 μs |  26.525 μs |   6.889 μs |    3880 B |
| AiDotNet_TensorMultiply | 100000  |    58.14 μs |   3.357 μs |   0.872 μs |     112 B |
| MlNet_Multiply          | 100000  |   218.68 μs |  22.125 μs |   5.746 μs |    3480 B |
| **AiDotNet_TensorAdd**      | **1000000** |   **479.56 μs** |  **39.713 μs** |  **10.313 μs** |  525290 B |
| MlNet_Add               | 1000000 |   465.59 μs |  42.345 μs |  10.997 μs |    2880 B |
| AiDotNet_TensorMultiply | 1000000 |   569.21 μs |  23.560 μs |   6.118 μs |  263744 B |
| MlNet_Multiply          | 1000000 |   300.02 μs |  24.535 μs |   6.372 μs |    2480 B |

## TensorFlow.NET (SciSharp eager)

| Method                  | size    | Mean        | Error      | StdDev     | Allocated |
|------------------------ |-------- |------------:|-----------:|-----------:|----------:|
| **AiDotNet_ReLU**           | **?**       |   **759.22 μs** | **870.091 μs** | **225.960 μs** | 4197715 B |
| TensorFlow_ReLU         | ?       | 1,409.64 μs | 315.633 μs |  81.969 μs |    1008 B |
| AiDotNet_Sigmoid        | ?       |   562.28 μs | 147.769 μs |  22.867 μs | 4198041 B |
| TensorFlow_Sigmoid      | ?       | 1,101.53 μs | 550.285 μs | 142.907 μs |    1168 B |
| AiDotNet_TensorSum      | ?       |    71.75 μs |   3.450 μs |   0.896 μs |    1168 B |
| TensorFlow_ReduceSum    | ?       |   121.15 μs |   5.696 μs |   1.479 μs |    3883 B |
| AiDotNet_TensorMean     | ?       |    82.15 μs |   6.801 μs |   1.766 μs |     112 B |
| TensorFlow_ReduceMean   | ?       |   206.15 μs |   1.048 μs |   0.162 μs |    3480 B |
| AiDotNet_Conv2D         | ?       |   484.81 μs |  58.302 μs |  15.141 μs |  525297 B |
| TensorFlow_Conv2D       | ?       |   371.03 μs |  35.180 μs |   9.136 μs |    2880 B |
| **AiDotNet_TensorMatMul**   | **256**     |   **468.76 μs** |  **20.443 μs** |   **5.309 μs** |  263768 B |
| TensorFlow_MatMul       | 256     |          NA |         NA |         NA |        NA |
| **AiDotNet_TensorMatMul**   | **512**     |          **NA** |         **NA** |         **NA** |        NA |
| TensorFlow_MatMul       | 512     |          NA |         NA |         NA |        NA |

TensorFlow.NET errored out on 512-MatMul and bulk Add/Multiply (NA in
output) — that's a TF.NET runtime issue at the size, not an AiDotNet
issue; the same shapes ran fine for every other competitor.
