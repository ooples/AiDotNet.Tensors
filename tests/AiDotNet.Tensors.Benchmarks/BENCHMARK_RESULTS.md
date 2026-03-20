```

BenchmarkDotNet v0.15.8, Windows 11 (10.0.26220.7961)
AMD Ryzen 9 3950X 3.70GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 10.0.103
  [Host]     : .NET 10.0.3 (10.0.3, 10.0.326.7603), X64 RyuJIT x86-64-v3
  Job-HGADUQ : .NET 10.0.3 (10.0.3, 10.0.326.7603), X64 RyuJIT x86-64-v3

Runtime=.NET 10.0  InvocationCount=1  IterationCount=15  
LaunchCount=1  UnrollFactor=1  WarmupCount=5  

```
| Method                      | size    | Mean        | Error     | StdDev    | Median      | Allocated  |
|---------------------------- |-------- |------------:|----------:|----------:|------------:|-----------:|
| **AiDotNet_TensorSubtract**     | **?**       |   **614.91 μs** | **124.24 μs** | **116.21 μs** |   **624.20 μs** |  **4196720 B** |
| TorchSharp_Subtract         | ?       |   317.25 μs |  46.89 μs |  41.57 μs |   317.00 μs |       48 B |
| AiDotNet_TensorDivide       | ?       |   590.24 μs | 156.61 μs | 146.49 μs |   628.30 μs |  4196720 B |
| TorchSharp_Divide           | ?       |   277.40 μs |  39.55 μs |  33.03 μs |   282.30 μs |       48 B |
| AiDotNet_TensorExp          | ?       |   621.07 μs |  98.43 μs |  92.07 μs |   597.50 μs |  4196752 B |
| TorchSharp_Exp              | ?       |   360.34 μs |  61.61 μs |  57.63 μs |   362.50 μs |       48 B |
| AiDotNet_TensorLog          | ?       |   836.19 μs |  96.64 μs |  85.67 μs |   850.15 μs |  4196752 B |
| TorchSharp_Log              | ?       |   307.95 μs |  56.24 μs |  52.61 μs |   321.80 μs |       48 B |
| AiDotNet_TensorSqrt         | ?       |   409.81 μs |  90.47 μs |  84.63 μs |   423.10 μs |  4196688 B |
| TorchSharp_Sqrt             | ?       |   273.55 μs |  73.61 μs |  61.47 μs |   293.20 μs |       48 B |
| AiDotNet_TensorAbs          | ?       |   440.31 μs |  79.59 μs |  74.45 μs |   429.90 μs |  4196688 B |
| TorchSharp_Abs              | ?       |   301.23 μs |  49.41 μs |  43.80 μs |   309.80 μs |       48 B |
| AiDotNet_ReLU               | ?       |   282.32 μs |  48.96 μs |  45.80 μs |   278.20 μs |       32 B |
| TorchSharp_ReLU             | ?       |   211.17 μs |  18.47 μs |  17.28 μs |   213.20 μs |          - |
| AiDotNet_Sigmoid            | ?       |   232.13 μs |  32.03 μs |  26.75 μs |   227.50 μs |     2920 B |
| RawTensorPrimitives_Sigmoid | ?       | 7,203.47 μs | 468.30 μs | 438.04 μs | 7,119.20 μs |          - |
| TorchSharp_Sigmoid          | ?       |   251.18 μs |  29.56 μs |  27.65 μs |   246.50 μs |          - |
| AiDotNet_Tanh               | ?       |   502.56 μs | 109.59 μs |  97.15 μs |   487.30 μs |  4196624 B |
| TorchSharp_Tanh             | ?       |   360.60 μs |  19.42 μs |  15.16 μs |   358.55 μs |       48 B |
| AiDotNet_GELU               | ?       |   437.61 μs |  68.62 μs |  64.18 μs |   435.40 μs |  4196688 B |
| TorchSharp_GELU             | ?       |   283.63 μs |  50.23 μs |  46.99 μs |   258.30 μs |       48 B |
| AiDotNet_Mish               | ?       | 1,107.17 μs |  63.27 μs |  56.09 μs | 1,110.40 μs |  4196688 B |
| TorchSharp_Mish             | ?       | 1,030.69 μs | 141.57 μs | 132.43 μs | 1,066.20 μs |      192 B |
| AiDotNet_LeakyReLU          | ?       |   410.78 μs |  81.31 μs |  76.06 μs |   390.60 μs |  4196688 B |
| TorchSharp_LeakyReLU        | ?       |   257.66 μs |  64.64 μs |  60.47 μs |   233.90 μs |       72 B |
| AiDotNet_TensorSum          | ?       |   247.94 μs |  25.82 μs |  22.89 μs |   248.65 μs |      208 B |
| RawTensorPrimitives_Sum     | ?       |   247.41 μs |  44.06 μs |  39.06 μs |   232.05 μs |          - |
| TorchSharp_Sum              | ?       |   241.20 μs |  22.54 μs |  19.99 μs |   245.35 μs |       48 B |
| AiDotNet_TensorMean         | ?       |   258.29 μs |  35.00 μs |  32.74 μs |   249.40 μs |      208 B |
| TorchSharp_Mean             | ?       |   268.90 μs |  52.10 μs |  48.73 μs |   281.25 μs |       48 B |
| AiDotNet_TensorMaxValue     | ?       |   253.44 μs |  37.78 μs |  33.49 μs |   245.80 μs |     2672 B |
| TorchSharp_Max              | ?       |   224.09 μs |  31.02 μs |  29.01 μs |   213.20 μs |       48 B |
| AiDotNet_TensorMinValue     | ?       |   254.04 μs |  38.66 μs |  36.16 μs |   260.50 μs |     2312 B |
| TorchSharp_Min              | ?       |   196.84 μs |  29.92 μs |  26.52 μs |   188.05 μs |       48 B |
| AiDotNet_Softmax            | ?       |   857.24 μs | 209.88 μs | 196.32 μs |   775.30 μs |  2104640 B |
| TorchSharp_Softmax          | ?       |    93.17 μs |  14.91 μs |  13.22 μs |    88.20 μs |       48 B |
| AiDotNet_LogSoftmax         | ?       |   943.51 μs | 151.96 μs | 142.14 μs |   933.60 μs |  2104680 B |
| TorchSharp_LogSoftmax       | ?       |   127.63 μs |  32.38 μs |  30.29 μs |   128.70 μs |       48 B |
| AiDotNet_Conv2D             | ?       |   476.33 μs |  30.31 μs |  28.35 μs |   483.00 μs |   524608 B |
| AiDotNet_Conv2D_ZeroAlloc   | ?       |   419.42 μs |  12.37 μs |  10.33 μs |   418.50 μs |      168 B |
| TorchSharp_Conv2D           | ?       |   345.45 μs | 113.24 μs | 105.92 μs |   307.10 μs |       48 B |
| AiDotNet_BatchNorm          | ?       | 1,163.31 μs | 340.79 μs | 318.77 μs | 1,035.30 μs |  8395768 B |
| TorchSharp_BatchNorm        | ?       |   870.92 μs | 184.33 μs | 172.43 μs |   870.60 μs |       48 B |
| AiDotNet_LayerNorm          | ?       |          NA |        NA |        NA |          NA |         NA |
| TorchSharp_LayerNorm        | ?       |          NA |        NA |        NA |          NA |         NA |
| AiDotNet_MaxPool2D          | ?       |   259.12 μs |  39.05 μs |  36.53 μs |   267.70 μs |   131296 B |
| TorchSharp_MaxPool2D        | ?       |   345.91 μs | 268.19 μs | 250.86 μs |   203.10 μs |       48 B |
| AiDotNet_SigmoidBackward    | ?       |   733.33 μs | 249.15 μs | 233.05 μs |   664.10 μs |  4000112 B |
| TorchSharp_SigmoidBackward  | ?       |   539.28 μs | 282.92 μs | 264.64 μs |   498.50 μs |      192 B |
| AiDotNet_TanhBackward       | ?       |   521.12 μs |  26.67 μs |  20.82 μs |   525.25 μs |  4000112 B |
| TorchSharp_TanhBackward     | ?       |   541.32 μs | 309.24 μs | 289.26 μs |   485.00 μs |      192 B |
| AiDotNet_AttentionQKT       | ?       |   326.25 μs |  49.70 μs |  46.49 μs |   333.70 μs |  1179992 B |
| TorchSharp_AttentionQKT     | ?       |   168.40 μs |  49.10 μs |  45.92 μs |   162.70 μs |       96 B |
| AiDotNet_TensorAdd_Double   | ?       | 1,350.95 μs | 270.43 μs | 252.96 μs | 1,324.50 μs |  8391192 B |
| TorchSharp_Add_Double       | ?       |   321.41 μs | 241.81 μs | 226.19 μs |   298.80 μs |       72 B |
| AiDotNet_MatMul_Double      | ?       |   214.59 μs |  40.91 μs |  36.26 μs |   216.75 μs |   524488 B |
| TorchSharp_MatMul_Double    | ?       |   211.34 μs |  23.20 μs |  20.56 μs |   215.75 μs |       48 B |
| AiDotNet_Sigmoid_Double     | ?       | 2,405.09 μs | 325.38 μs | 304.36 μs | 2,493.60 μs | 16393896 B |
| TorchSharp_Sigmoid_Double   | ?       |   346.92 μs |  50.32 μs |  42.02 μs |   331.80 μs |       48 B |
| **AiDotNet_TensorMatMul**       | **256**     |   **147.24 μs** |  **32.21 μs** |  **28.55 μs** |   **153.15 μs** |   **262344 B** |
| TorchSharp_MatMul           | 256     |   169.00 μs |  31.32 μs |  29.30 μs |   168.70 μs |       48 B |
| **AiDotNet_TensorMatMul**       | **512**     |   **909.33 μs** | **248.09 μs** | **232.07 μs** |   **840.50 μs** |  **1048776 B** |
| TorchSharp_MatMul           | 512     |   708.44 μs | 160.64 μs | 150.27 μs |   697.40 μs |       48 B |
| **AiDotNet_TensorAdd**          | **100000**  |   **134.36 μs** |  **60.64 μs** |  **56.72 μs** |   **127.40 μs** |      **192 B** |
| RawTensorPrimitives_Add     | 100000  |   199.19 μs | 133.67 μs | 118.49 μs |   129.00 μs |          - |
| TorchSharp_Add              | 100000  |    89.88 μs |  26.37 μs |  24.67 μs |    98.80 μs |       24 B |
| AiDotNet_TensorMultiply     | 100000  |    69.68 μs |  40.02 μs |  31.24 μs |    59.70 μs |      192 B |
| TorchSharp_Multiply         | 100000  |    66.48 μs |  26.90 μs |  22.46 μs |    61.50 μs |          - |
| **AiDotNet_TensorAdd**          | **1000000** |   **695.99 μs** | **151.98 μs** | **126.91 μs** |   **669.70 μs** |     **2296 B** |
| RawTensorPrimitives_Add     | 1000000 |   781.07 μs | 254.71 μs | 212.70 μs |   753.60 μs |          - |
| TorchSharp_Add              | 1000000 |   328.23 μs |  77.27 μs |  64.53 μs |   327.10 μs |       24 B |
| TorchSharp_Add_1Thread      | 1000000 |   729.76 μs | 198.04 μs | 175.56 μs |   696.35 μs |       24 B |
| AiDotNet_TensorMultiply     | 1000000 |   597.36 μs | 223.65 μs | 209.21 μs |   600.50 μs |     2296 B |
| TorchSharp_Multiply         | 1000000 |   294.10 μs |  87.04 μs |  72.68 μs |   262.80 μs |          - |

Benchmarks with issues:
  TorchSharpCpuComparisonBenchmarks.AiDotNet_LayerNorm: Job-HGADUQ(Runtime=.NET 10.0, InvocationCount=1, IterationCount=15, LaunchCount=1, UnrollFactor=1, WarmupCount=5)
  TorchSharpCpuComparisonBenchmarks.TorchSharp_LayerNorm: Job-HGADUQ(Runtime=.NET 10.0, InvocationCount=1, IterationCount=15, LaunchCount=1, UnrollFactor=1, WarmupCount=5)
