| Method                      | size    | Mean         | Error        | StdDev       | Median       | Allocated |
|---------------------------- |-------- |-------------:|-------------:|-------------:|-------------:|----------:|
| AiDotNet_TensorSubtract     | ?       |    963.11 us |   114.009 us |   101.066 us |    963.10 us | 4196672 B |
| TorchSharp_Subtract         | ?       |    368.87 us |    45.019 us |    35.148 us |    369.25 us |      48 B |
| AiDotNet_TensorDivide       | ?       |    939.49 us |   195.222 us |   173.059 us |    866.20 us | 4196736 B |
| TorchSharp_Divide           | ?       |    499.67 us |   182.880 us |   171.066 us |    479.40 us |      48 B |
| AiDotNet_TensorExp          | ?       |    954.11 us |   260.122 us |   243.318 us |  1,016.00 us | 4196688 B |
| TorchSharp_Exp              | ?       |    352.99 us |    95.760 us |    84.888 us |    325.85 us |      48 B |
| AiDotNet_TensorLog          | ?       |  1,676.69 us |   883.341 us |   783.059 us |  1,379.20 us | 4196752 B |
| TorchSharp_Log              | ?       |    388.68 us |    42.845 us |    35.777 us |    397.80 us |      48 B |
| AiDotNet_TensorSqrt         | ?       |    819.71 us |   388.218 us |   363.140 us |    862.30 us | 4196688 B |
| TorchSharp_Sqrt             | ?       |    347.46 us |   127.593 us |   119.351 us |    313.30 us |      48 B |
| AiDotNet_TensorAbs          | ?       |  1,086.14 us |   271.697 us |   240.852 us |  1,028.60 us | 4196848 B |
| TorchSharp_Abs              | ?       |    364.79 us |    92.687 us |    82.165 us |    361.35 us |      48 B |
| AiDotNet_ReLU               | ?       |    395.77 us |    75.122 us |    66.594 us |    382.60 us |      32 B |
| TorchSharp_ReLU             | ?       |    235.86 us |    40.196 us |    37.599 us |    225.65 us |         - |
| AiDotNet_Sigmoid            | ?       |    343.00 us |    41.666 us |    32.530 us |    345.45 us |    2952 B |
| RawTensorPrimitives_Sigmoid | ?       |  8,085.61 us | 1,423.338 us | 1,331.392 us |  7,978.80 us |         - |
| TorchSharp_Sigmoid          | ?       |    256.05 us |    51.525 us |    43.026 us |    261.00 us |         - |
| AiDotNet_Tanh               | ?       |  1,306.83 us |   339.878 us |   301.293 us |  1,281.45 us | 4196688 B |
| TorchSharp_Tanh             | ?       | 10,311.17 us | 3,460.692 us | 3,237.134 us | 11,344.50 us |      48 B |
| AiDotNet_GELU               | ?       |  3,307.60 us | 1,842.759 us | 1,723.718 us |  2,994.00 us | 4196304 B |
| TorchSharp_GELU             | ?       |    379.93 us |   100.172 us |    93.701 us |    382.10 us |      48 B |
| AiDotNet_Mish               | ?       |  1,817.59 us |   198.352 us |   185.539 us |  1,832.90 us | 4196688 B |
| TorchSharp_Mish             | ?       |    839.90 us |   132.227 us |   123.685 us |    780.00 us |     192 B |
| AiDotNet_LeakyReLU          | ?       |    426.49 us |    83.286 us |    77.905 us |    418.20 us | 4196688 B |
| TorchSharp_LeakyReLU        | ?       |    264.54 us |    32.889 us |    30.764 us |    266.90 us |      72 B |
| AiDotNet_Subtract_ZeroAlloc | ?       |    870.06 us |    86.164 us |    76.382 us |    848.40 us |         - |
| AiDotNet_Exp_ZeroAlloc      | ?       |    698.31 us |    33.059 us |    29.306 us |    697.20 us |         - |
| AiDotNet_Log_ZeroAlloc      | ?       |  1,374.49 us |    92.900 us |    86.899 us |  1,427.85 us |         - |
| AiDotNet_Tanh_ZeroAlloc     | ?       |    963.34 us |    71.060 us |    66.470 us |    929.60 us |         - |
| AiDotNet_GELU_ZeroAlloc     | ?       |    626.37 us |   320.320 us |   299.628 us |    466.80 us |    6864 B |
| AiDotNet_TensorSum          | ?       |    227.25 us |    19.281 us |    17.092 us |    226.95 us |     208 B |
| RawTensorPrimitives_Sum     | ?       |    419.77 us |    31.741 us |    29.691 us |    419.90 us |         - |
| TorchSharp_Sum              | ?       |    346.36 us |    22.709 us |    18.963 us |    337.00 us |      48 B |
| AiDotNet_TensorMean         | ?       |    279.45 us |    23.340 us |    21.832 us |    277.40 us |     208 B |
| TorchSharp_Mean             | ?       |  6,095.90 us | 6,846.389 us | 6,069.146 us |  4,338.45 us |      48 B |
| AiDotNet_TensorMaxValue     | ?       |    285.88 us |    41.769 us |    39.071 us |    286.90 us |    2312 B |
| TorchSharp_Max              | ?       |    216.30 us |    24.803 us |    23.201 us |    223.30 us |      48 B |
| AiDotNet_TensorMinValue     | ?       |    256.69 us |    47.669 us |    44.590 us |    249.90 us |    2312 B |
| TorchSharp_Min              | ?       |    204.38 us |    40.625 us |    38.001 us |    197.70 us |      48 B |
| AiDotNet_Softmax            | ?       |    620.62 us |    68.962 us |    64.507 us |    618.80 us | 2097424 B |
| AiDotNet_Softmax_ZeroAlloc  | ?       |    728.43 us |    66.775 us |    59.195 us |    731.25 us |     200 B |
| TorchSharp_Softmax          | ?       |    107.34 us |    17.655 us |    15.651 us |    100.30 us |      48 B |
| AiDotNet_LogSoftmax         | ?       |  1,023.89 us |   223.688 us |   209.238 us |  1,094.40 us | 2103960 B |
| TorchSharp_LogSoftmax       | ?       |    110.65 us |    21.908 us |    20.493 us |    101.60 us |      48 B |
| AiDotNet_Conv2D             | ?       |    484.74 us |    23.307 us |    20.661 us |    483.45 us |  524608 B |
| AiDotNet_Conv2D_ZeroAlloc   | ?       |    420.23 us |    15.168 us |    12.666 us |    423.60 us |     168 B |
| TorchSharp_Conv2D           | ?       |    296.32 us |    95.622 us |    89.445 us |    239.90 us |      48 B |
| AiDotNet_BatchNorm          | ?       |  1,378.85 us |   234.823 us |   219.653 us |  1,290.70 us | 8396000 B |
| TorchSharp_BatchNorm        | ?       |  1,659.75 us |   139.929 us |   130.889 us |  1,703.30 us |      48 B |
| AiDotNet_LayerNorm          | ?       |           NA |           NA |           NA |           NA |        NA |
| TorchSharp_LayerNorm        | ?       |           NA |           NA |           NA |           NA |        NA |
| AiDotNet_GroupNorm          | ?       |  2,322.28 us |   325.089 us |   288.183 us |  2,249.70 us | 8405696 B |
| TorchSharp_GroupNorm        | ?       |    288.21 us |    78.845 us |    69.894 us |    247.40 us |      48 B |
| AiDotNet_GroupNormSwish     | ?       |  4,555.20 us |   303.233 us |   283.645 us |  4,475.30 us | 8414264 B |
| AiDotNet_MaxPool2D          | ?       |    213.99 us |     5.568 us |     4.936 us |    213.05 us |  131296 B |
| TorchSharp_MaxPool2D        | ?       |    150.47 us |     9.501 us |     7.934 us |    150.60 us |      48 B |
| AiDotNet_SigmoidBackward    | ?       |    386.97 us |    84.779 us |    70.794 us |    361.30 us | 4000112 B |
| TorchSharp_SigmoidBackward  | ?       |    238.26 us |    66.769 us |    55.755 us |    228.00 us |     192 B |
| AiDotNet_TanhBackward       | ?       |    553.47 us |    31.304 us |    29.282 us |    551.20 us | 4000112 B |
| TorchSharp_TanhBackward     | ?       |    209.26 us |   111.047 us |    98.440 us |    181.40 us |     192 B |
| AiDotNet_AttentionQKT       | ?       |    416.39 us |    19.243 us |    17.058 us |    422.25 us | 1179992 B |
| TorchSharp_AttentionQKT     | ?       |    123.18 us |     6.348 us |     5.301 us |    122.80 us |      96 B |
| AiDotNet_TensorAdd_Double   | ?       |  1,353.04 us |   142.978 us |   133.741 us |  1,376.70 us | 8391336 B |
| TorchSharp_Add_Double       | ?       |    510.04 us |   168.339 us |   149.228 us |    487.40 us |      72 B |
| AiDotNet_MatMul_Double      | ?       |    299.95 us |    40.300 us |    37.696 us |    296.50 us |  524488 B |
| TorchSharp_MatMul_Double    | ?       |    195.27 us |    31.404 us |    29.375 us |    181.20 us |      48 B |
| AiDotNet_Sigmoid_Double     | ?       |  1,311.06 us |    83.969 us |    78.545 us |  1,296.30 us | 8393744 B |
| TorchSharp_Sigmoid_Double   | ?       |    413.00 us |   116.917 us |   109.364 us |    406.40 us |      48 B |
| AiDotNet_Exp_Double         | ?       |  1,538.17 us |   241.658 us |   226.047 us |  1,549.80 us | 8395136 B |
| TorchSharp_Exp_Double       | ?       |    378.09 us |   123.738 us |   115.745 us |    328.70 us |      48 B |
| AiDotNet_Log_Double         | ?       |  1,726.77 us |   291.913 us |   273.055 us |  1,732.30 us | 8395456 B |
| TorchSharp_Log_Double       | ?       |    391.55 us |    41.435 us |    34.600 us |    376.00 us |      48 B |
| AiDotNet_Tanh_Double        | ?       |  1,238.44 us |   100.095 us |    88.732 us |  1,230.00 us | 8395224 B |
| TorchSharp_Tanh_Double      | ?       |    614.65 us |    15.523 us |    12.962 us |    609.80 us |      48 B |
| AiDotNet_GELU_Double        | ?       |  1,415.73 us |   282.501 us |   264.251 us |  1,280.10 us | 8395200 B |
| TorchSharp_GELU_Double      | ?       |    730.83 us |     7.766 us |     6.485 us |    727.80 us |      48 B |
| AiDotNet_Mish_Double        | ?       |  1,841.77 us |   120.021 us |   112.267 us |  1,830.30 us | 8397632 B |
| TorchSharp_Mish_Double      | ?       |  1,952.91 us |   293.711 us |   260.367 us |  1,870.25 us |     192 B |
| AiDotNet_Softmax_Double     | ?       |  8,835.82 us |   217.516 us |   181.636 us |  8,802.00 us | 8411016 B |
| TorchSharp_Softmax_Double   | ?       |    215.77 us |    30.168 us |    28.219 us |    208.30 us |      48 B |
| AiDotNet_Conv2D_Double      | ?       |    107.25 us |    10.670 us |     8.330 us |    110.00 us |  131224 B |
| TorchSharp_Conv2D_Double    | ?       |     84.87 us |     5.601 us |     4.677 us |     84.20 us |      48 B |
| AiDotNet_TensorMatMul       | 256     |    123.66 us |    29.822 us |    26.437 us |    119.80 us |  262344 B |
| TorchSharp_MatMul           | 256     |    128.10 us |    27.526 us |    24.401 us |    127.55 us |      48 B |
| AiDotNet_TensorMatMul       | 512     |    687.96 us |    77.334 us |    72.339 us |    673.00 us | 1048776 B |
| TorchSharp_MatMul           | 512     |    648.52 us |    41.855 us |    37.103 us |    639.20 us |      48 B |
| AiDotNet_TensorAdd          | 100000  |     53.46 us |    13.784 us |    12.219 us |     52.10 us |     200 B |
| RawTensorPrimitives_Add     | 100000  |    148.31 us |    57.316 us |    50.809 us |    134.70 us |         - |
| TorchSharp_Add              | 100000  |     47.14 us |    18.767 us |    17.554 us |     49.20 us |      24 B |
| AiDotNet_TensorMultiply     | 100000  |     43.29 us |    33.835 us |    31.649 us |     26.90 us |     200 B |
| TorchSharp_Multiply         | 100000  |     47.27 us |    19.033 us |    16.872 us |     43.35 us |         - |
| AiDotNet_TensorAdd          | 1000000 |    344.56 us |    91.506 us |    85.595 us |    329.20 us |    2312 B |
| RawTensorPrimitives_Add     | 1000000 |    659.63 us |   147.022 us |   137.524 us |    687.30 us |         - |
| TorchSharp_Add              | 1000000 |  1,985.44 us | 3,161.557 us | 2,802.638 us |    484.25 us |      24 B |
| TorchSharp_Add_1Thread      | 1000000 |    467.61 us |    78.083 us |    73.039 us |    465.30 us |      24 B |
| AiDotNet_TensorMultiply     | 1000000 |    353.84 us |    82.704 us |    77.361 us |    348.60 us |    2312 B |
| TorchSharp_Multiply         | 1000000 |    220.78 us |    35.187 us |    32.914 us |    208.80 us |         - |
