| Method                      | size    | Mean         | Error      | StdDev     | Allocated  |
|---------------------------- |-------- |-------------:|-----------:|-----------:|-----------:|
| AiDotNet_TensorSubtract     | ?       |    512.47 us | 109.831 us | 102.736 us |  4196736 B |
| TorchSharp_Subtract         | ?       |    373.32 us |  34.978 us |  31.007 us |       48 B |
| AiDotNet_TensorDivide       | ?       |    595.46 us | 114.393 us | 107.003 us |  4196736 B |
| TorchSharp_Divide           | ?       |    396.46 us |  48.904 us |  43.352 us |       48 B |
| AiDotNet_TensorExp          | ?       |    588.46 us |  81.587 us |  76.316 us |  4196688 B |
| TorchSharp_Exp              | ?       |    372.31 us |  27.692 us |  25.903 us |       48 B |
| AiDotNet_TensorLog          | ?       |    893.48 us | 103.153 us |  96.489 us |  4196688 B |
| TorchSharp_Log              | ?       |    328.31 us |   7.346 us |   6.512 us |       48 B |
| AiDotNet_TensorSqrt         | ?       |    481.95 us |  78.463 us |  73.395 us |  4196752 B |
| TorchSharp_Sqrt             | ?       |    278.51 us |  24.061 us |  22.507 us |       48 B |
| AiDotNet_TensorAbs          | ?       |    476.96 us | 131.843 us | 123.326 us |  4196752 B |
| TorchSharp_Abs              | ?       |    318.63 us |   7.454 us |   6.973 us |       48 B |
| AiDotNet_ReLU               | ?       |    235.40 us |   9.316 us |   8.258 us |       32 B |
| TorchSharp_ReLU             | ?       |    155.83 us |   3.596 us |   3.003 us |          - |
| AiDotNet_Sigmoid            | ?       |    295.93 us |  16.250 us |  14.406 us |     2952 B |
| RawTensorPrimitives_Sigmoid | ?       |  5,890.03 us | 126.912 us | 105.977 us |          - |
| TorchSharp_Sigmoid          | ?       |    199.09 us |  21.961 us |  20.542 us |          - |
| AiDotNet_Tanh               | ?       |    731.61 us | 135.016 us | 126.294 us |  4196688 B |
| TorchSharp_Tanh             | ?       |    354.45 us |  18.561 us |  17.362 us |       48 B |
| AiDotNet_GELU               | ?       |    522.09 us | 118.286 us | 110.644 us |  4196688 B |
| TorchSharp_GELU             | ?       |    320.27 us |  13.837 us |  11.555 us |       48 B |
| AiDotNet_Mish               | ?       |  1,713.52 us | 102.245 us |  95.640 us |  4196752 B |
| TorchSharp_Mish             | ?       |    725.07 us |  15.754 us |  13.156 us |      192 B |
| AiDotNet_LeakyReLU          | ?       |    589.68 us | 261.620 us | 231.920 us |  4196688 B |
| TorchSharp_LeakyReLU        | ?       |    300.36 us |  23.114 us |  20.490 us |       72 B |
| AiDotNet_TensorSum          | ?       |    190.30 us |  19.398 us |  18.145 us |      208 B |
| RawTensorPrimitives_Sum     | ?       |    202.43 us |  10.337 us |   9.164 us |          - |
| TorchSharp_Sum              | ?       |    232.78 us |  18.593 us |  16.483 us |       48 B |
| AiDotNet_TensorMean         | ?       |    182.05 us |  14.006 us |  13.101 us |      208 B |
| TorchSharp_Mean             | ?       |    184.95 us |   4.443 us |   3.710 us |       48 B |
| AiDotNet_TensorMaxValue     | ?       |    187.07 us |   8.192 us |   7.262 us |     2312 B |
| TorchSharp_Max              | ?       |    157.92 us |   4.003 us |   3.126 us |       48 B |
| AiDotNet_TensorMinValue     | ?       |    220.05 us |  16.046 us |  15.010 us |     2312 B |
| TorchSharp_Min              | ?       |    163.50 us |   6.543 us |   5.800 us |       48 B |
| AiDotNet_Softmax            | ?       |    760.22 us | 135.764 us | 126.994 us |  2104480 B |
| AiDotNet_Softmax_ZeroAlloc  | ?       |    826.23 us | 147.768 us | 138.223 us |     6096 B |
| TorchSharp_Softmax          | ?       |     74.95 us |  10.652 us |   9.443 us |       48 B |
| AiDotNet_LogSoftmax         | ?       |    791.89 us | 117.597 us | 104.247 us |  2103384 B |
| TorchSharp_LogSoftmax       | ?       |    100.57 us |   3.120 us |   2.606 us |       48 B |
| AiDotNet_Conv2D             | ?       |    431.11 us |  21.296 us |  18.878 us |   524608 B |
| AiDotNet_Conv2D_ZeroAlloc   | ?       |    393.61 us |  15.509 us |  14.507 us |      168 B |
| TorchSharp_Conv2D           | ?       |    352.92 us |  48.450 us |  45.320 us |       48 B |
| AiDotNet_BatchNorm          | ?       |  1,114.19 us | 304.735 us | 285.050 us |  8396344 B |
| TorchSharp_BatchNorm        | ?       |    550.69 us | 114.869 us | 107.448 us |       48 B |
| AiDotNet_LayerNorm          | ?       |           NA |         NA |         NA |         NA |
| TorchSharp_LayerNorm        | ?       |           NA |         NA |         NA |         NA |
| AiDotNet_GroupNorm          | ?       |  9,246.40 us | 414.154 us | 345.837 us | 16803840 B |
| TorchSharp_GroupNorm        | ?       |    208.11 us |  40.269 us |  37.668 us |       48 B |
| AiDotNet_GroupNormSwish     | ?       | 15,326.72 us | 530.138 us | 495.891 us | 16812552 B |
| AiDotNet_MaxPool2D          | ?       |    228.55 us |   4.244 us |   3.544 us |   131296 B |
| TorchSharp_MaxPool2D        | ?       |    102.32 us |  12.853 us |  10.035 us |       48 B |
| AiDotNet_SigmoidBackward    | ?       |    364.51 us | 107.013 us |  94.865 us |  4000112 B |
| TorchSharp_SigmoidBackward  | ?       |    378.26 us |  31.385 us |  27.822 us |      192 B |
| AiDotNet_TanhBackward       | ?       |    335.07 us |  52.525 us |  43.861 us |  4000112 B |
| TorchSharp_TanhBackward     | ?       |    485.27 us | 123.885 us | 115.882 us |      192 B |
| AiDotNet_AttentionQKT       | ?       |    411.89 us |  13.103 us |  12.256 us |  1179992 B |
| TorchSharp_AttentionQKT     | ?       |    105.62 us |   3.032 us |   2.532 us |       96 B |
| AiDotNet_TensorAdd_Double   | ?       |  1,091.29 us | 140.381 us | 131.313 us |  8391272 B |
| TorchSharp_Add_Double       | ?       |    172.88 us |  28.468 us |  23.772 us |       72 B |
| AiDotNet_MatMul_Double      | ?       |    295.16 us |  48.343 us |  45.220 us |   524488 B |
| TorchSharp_MatMul_Double    | ?       |    286.71 us |  29.322 us |  27.428 us |       48 B |
| AiDotNet_Sigmoid_Double     | ?       |  1,333.13 us |  79.916 us |  74.754 us |  8393808 B |
| TorchSharp_Sigmoid_Double   | ?       |    338.20 us |  68.540 us |  64.112 us |       48 B |
| AiDotNet_Exp_Double         | ?       |  1,431.99 us | 162.060 us | 151.591 us |  8395392 B |
| TorchSharp_Exp_Double       | ?       |    264.67 us |   5.912 us |   5.241 us |       48 B |
| AiDotNet_Log_Double         | ?       |  1,621.26 us | 357.739 us | 317.126 us |  8395224 B |
| TorchSharp_Log_Double       | ?       |    192.21 us |   3.198 us |   2.670 us |       48 B |
| AiDotNet_Tanh_Double        | ?       |  1,084.81 us |  76.292 us |  59.564 us |  8394976 B |
| TorchSharp_Tanh_Double      | ?       |    610.71 us |   9.479 us |   7.916 us |       48 B |
| AiDotNet_GELU_Double        | ?       |  1,197.59 us | 109.999 us |  91.854 us |  8395616 B |
| TorchSharp_GELU_Double      | ?       |    628.86 us | 145.960 us | 136.531 us |       48 B |
| AiDotNet_Mish_Double        | ?       |  1,718.22 us |  46.119 us |  40.883 us |  8396232 B |
| TorchSharp_Mish_Double      | ?       |  1,656.19 us | 336.226 us | 314.506 us |      192 B |
| AiDotNet_Softmax_Double     | ?       |  8,845.62 us | 791.784 us | 661.176 us |  8410728 B |
| TorchSharp_Softmax_Double   | ?       |    186.91 us |   9.788 us |   9.156 us |       48 B |
| AiDotNet_Conv2D_Double      | ?       |     98.55 us |   2.544 us |   2.125 us |   131224 B |
| TorchSharp_Conv2D_Double    | ?       |     77.54 us |   2.773 us |   2.165 us |       48 B |
| AiDotNet_TensorMatMul       | 256     |    111.82 us |   2.759 us |   2.304 us |   262344 B |
| TorchSharp_MatMul           | 256     |    102.24 us |  16.177 us |  14.341 us |       48 B |
| AiDotNet_TensorMatMul       | 512     |    640.10 us |  45.721 us |  42.768 us |  1048776 B |
| TorchSharp_MatMul           | 512     |    339.28 us |   9.681 us |   8.084 us |       48 B |
| AiDotNet_TensorAdd          | 100000  |     15.75 us |   1.791 us |   1.495 us |      200 B |
| RawTensorPrimitives_Add     | 100000  |    144.89 us |  54.744 us |  48.529 us |          - |
| TorchSharp_Add              | 100000  |     25.15 us |   2.330 us |   1.945 us |       24 B |
| AiDotNet_TensorMultiply     | 100000  |     19.12 us |   2.522 us |   2.106 us |      200 B |
| TorchSharp_Multiply         | 100000  |     32.33 us |   1.737 us |   1.451 us |          - |
| AiDotNet_TensorAdd          | 1000000 |    229.68 us |  19.558 us |  15.270 us |     2312 B |
| RawTensorPrimitives_Add     | 1000000 |    385.82 us |  14.002 us |  10.932 us |          - |
| TorchSharp_Add              | 1000000 |    172.66 us |   7.704 us |   6.433 us |       24 B |
| TorchSharp_Add_1Thread      | 1000000 |    339.01 us |  34.764 us |  30.817 us |       24 B |
| AiDotNet_TensorMultiply     | 1000000 |    337.51 us |  68.622 us |  64.189 us |     2312 B |
| TorchSharp_Multiply         | 1000000 |    180.86 us |  33.495 us |  31.331 us |          - |
