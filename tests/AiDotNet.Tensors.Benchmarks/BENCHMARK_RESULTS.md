| Method                      | size    | Mean          | Error         | StdDev       | Median        | Allocated |
|---------------------------- |-------- |--------------:|--------------:|-------------:|--------------:|----------:|
| AiDotNet_TensorSubtract     | ?       |   2,308.34 us |    733.195 us |   572.431 us |   2,126.15 us |    2376 B |
| TorchSharp_Subtract         | ?       |   5,771.19 us |  3,061.937 us | 2,714.328 us |   4,735.90 us |      48 B |
| AiDotNet_TensorDivide       | ?       |   3,398.71 us |  2,114.895 us | 1,766.033 us |   2,982.30 us |    2440 B |
| TorchSharp_Divide           | ?       |   4,698.55 us |  2,589.511 us | 2,295.534 us |   4,619.40 us |      48 B |
| AiDotNet_TensorExp          | ?       |   2,721.59 us |  1,501.704 us | 1,331.222 us |   2,164.25 us |    2392 B |
| TorchSharp_Exp              | ?       |   2,180.01 us |  2,752.376 us | 2,298.358 us |   1,632.50 us |      48 B |
| AiDotNet_TensorLog          | ?       |   1,960.54 us |    493.558 us |   461.674 us |   1,794.50 us |    2328 B |
| TorchSharp_Log              | ?       |   3,020.87 us |  1,712.646 us | 1,430.137 us |   2,911.70 us |      48 B |
| AiDotNet_TensorSqrt         | ?       |   1,861.96 us |    628.577 us |   524.890 us |   1,839.80 us |    2392 B |
| TorchSharp_Sqrt             | ?       |   2,098.88 us |  3,302.459 us | 2,757.703 us |     412.60 us |      48 B |
| AiDotNet_TensorAbs          | ?       |   3,134.29 us |  1,827.561 us | 1,709.501 us |   2,920.10 us |    2008 B |
| TorchSharp_Abs              | ?       |     356.14 us |    100.428 us |    83.862 us |     336.60 us |      48 B |
| AiDotNet_ReLU               | ?       |     420.52 us |     36.998 us |    34.608 us |     422.40 us |      32 B |
| TorchSharp_ReLU             | ?       |   4,045.11 us |  3,213.732 us | 3,006.127 us |   2,776.20 us |         - |
| AiDotNet_Sigmoid            | ?       |   4,412.92 us |  4,515.046 us | 4,223.377 us |   2,087.50 us |    2344 B |
| RawTensorPrimitives_Sigmoid | ?       |   9,277.62 us |  1,759.032 us | 1,645.400 us |  10,109.25 us |         - |
| TorchSharp_Sigmoid          | ?       |   2,056.33 us |  1,760.198 us | 1,646.491 us |   1,871.30 us |         - |
| AiDotNet_Tanh               | ?       |   1,687.85 us |    471.493 us |   417.967 us |   1,504.30 us |    2328 B |
| TorchSharp_Tanh             | ?       |   4,949.86 us |  4,211.915 us | 3,939.828 us |   4,472.45 us |      48 B |
| AiDotNet_GELU               | ?       |   1,892.08 us |    558.747 us |   466.579 us |   1,823.60 us |    2392 B |
| TorchSharp_GELU             | ?       |   3,822.06 us |  3,322.702 us | 2,774.606 us |   3,318.25 us |      48 B |
| AiDotNet_Mish               | ?       |   2,349.04 us |    312.698 us |   244.134 us |   2,253.45 us |    2008 B |
| TorchSharp_Mish             | ?       |   5,195.42 us |  5,326.203 us | 4,982.134 us |   3,263.90 us |     192 B |
| AiDotNet_LeakyReLU          | ?       |   1,833.22 us |    585.937 us |   457.461 us |   1,710.55 us |    2392 B |
| TorchSharp_LeakyReLU        | ?       |   3,702.12 us |  3,237.697 us | 3,028.544 us |   3,144.55 us |      72 B |
| AiDotNet_Subtract_ZeroAlloc | ?       |   1,352.78 us |     66.572 us |    55.591 us |   1,374.90 us |         - |
| AiDotNet_Exp_ZeroAlloc      | ?       |   2,871.79 us |  2,903.094 us | 2,573.517 us |   2,073.00 us |         - |
| AiDotNet_Log_ZeroAlloc      | ?       |     911.88 us |    516.171 us |   402.992 us |     842.85 us |         - |
| AiDotNet_Tanh_ZeroAlloc     | ?       |   1,150.30 us |     55.923 us |    49.574 us |   1,157.15 us |         - |
| AiDotNet_GELU_ZeroAlloc     | ?       |   1,250.42 us |    396.669 us |   309.693 us |   1,187.40 us |    6720 B |
| AiDotNet_TensorSum          | ?       |     249.82 us |     14.722 us |    13.050 us |     252.00 us |     208 B |
| RawTensorPrimitives_Sum     | ?       |     404.93 us |     29.875 us |    27.945 us |     409.80 us |         - |
| TorchSharp_Sum              | ?       |   3,133.34 us |  2,021.334 us | 1,791.860 us |   2,935.00 us |      48 B |
| AiDotNet_TensorMean         | ?       |     257.64 us |     24.943 us |    22.111 us |     261.95 us |     208 B |
| TorchSharp_Mean             | ?       |   7,811.44 us |  3,250.397 us | 3,040.424 us |   7,473.00 us |      48 B |
| AiDotNet_TensorMaxValue     | ?       |   3,171.39 us |  2,844.549 us | 2,660.793 us |   2,731.90 us |    3232 B |
| TorchSharp_Max              | ?       |     515.42 us |    546.143 us |   426.393 us |     285.20 us |      48 B |
| AiDotNet_TensorMinValue     | ?       |   2,280.06 us |  2,222.613 us | 1,970.288 us |   1,737.00 us |    1768 B |
| TorchSharp_Min              | ?       |   5,492.17 us |  4,287.609 us | 4,010.632 us |   5,224.60 us |      48 B |
| AiDotNet_Softmax            | ?       |   1,560.28 us |     77.356 us |    64.596 us |   1,551.80 us | 2097456 B |
| AiDotNet_Softmax_ZeroAlloc  | ?       |   2,795.10 us |    207.777 us |   194.355 us |   2,871.35 us | 2097408 B |
| TorchSharp_Softmax          | ?       |   5,813.15 us |  4,888.135 us | 4,572.364 us |   4,858.80 us |      48 B |
| AiDotNet_LogSoftmax         | ?       |   3,198.42 us |    679.848 us |   530.781 us |   3,016.95 us | 2105464 B |
| TorchSharp_LogSoftmax       | ?       |   2,917.67 us |  2,822.186 us | 2,356.653 us |   2,396.15 us |      48 B |
| AiDotNet_Conv2D             | ?       |     525.42 us |     41.830 us |    37.081 us |     520.25 us |  524608 B |
| AiDotNet_Conv2D_ZeroAlloc   | ?       |     483.78 us |     49.268 us |    46.086 us |     461.10 us |     168 B |
| TorchSharp_Conv2D           | ?       |   8,575.16 us | 10,242.577 us | 9,580.913 us |   4,356.90 us |      48 B |
| AiDotNet_BatchNorm          | ?       |   3,229.96 us |    479.151 us |   424.755 us |   3,370.55 us | 8392992 B |
| TorchSharp_BatchNorm        | ?       |  17,180.01 us |  5,806.294 us | 5,147.129 us |  15,518.10 us |      48 B |
| AiDotNet_LayerNorm          | ?       |            NA |            NA |           NA |            NA |        NA |
| TorchSharp_LayerNorm        | ?       |            NA |            NA |           NA |            NA |        NA |
| AiDotNet_GroupNorm          | ?       |   6,345.21 us |    549.090 us |   458.515 us |   6,232.60 us | 8405728 B |
| TorchSharp_GroupNorm        | ?       |   1,783.79 us |  1,663.087 us | 1,388.753 us |   1,205.60 us |      48 B |
| AiDotNet_GroupNormSwish     | ?       |  12,309.98 us |  3,306.548 us | 2,761.118 us |  10,942.60 us | 8412200 B |
| AiDotNet_MaxPool2D          | ?       |     325.21 us |     13.007 us |    12.167 us |     324.60 us |  131296 B |
| TorchSharp_MaxPool2D        | ?       |  10,924.10 us |  6,848.785 us | 6,406.358 us |  11,209.50 us |      48 B |
| AiDotNet_SigmoidBackward    | ?       |   1,006.67 us |    159.541 us |   133.224 us |     964.75 us | 4000112 B |
| TorchSharp_SigmoidBackward  | ?       |   1,133.82 us |    444.225 us |   346.822 us |   1,162.75 us |     192 B |
| AiDotNet_TanhBackward       | ?       |     984.29 us |    116.327 us |   103.121 us |     980.45 us | 4000112 B |
| TorchSharp_TanhBackward     | ?       |   6,518.21 us |  8,673.756 us | 8,113.437 us |   1,212.20 us |     192 B |
| AiDotNet_AttentionQKT       | ?       |   1,018.31 us |    323.928 us |   252.902 us |     956.45 us | 1180024 B |
| TorchSharp_AttentionQKT     | ?       |     235.22 us |     44.790 us |    37.402 us |     228.80 us |      96 B |
| AiDotNet_TensorAdd_Double   | ?       |   3,675.74 us |    645.300 us |   603.614 us |   3,357.10 us |    2608 B |
| TorchSharp_Add_Double       | ?       |   3,493.08 us |  2,492.442 us | 2,209.485 us |   3,200.00 us |      72 B |
| AiDotNet_MatMul_Double      | ?       |     938.16 us |  1,198.410 us | 1,000.727 us |     475.20 us |  524488 B |
| TorchSharp_MatMul_Double    | ?       |   4,785.15 us |  4,406.562 us | 4,121.900 us |   4,228.80 us |      48 B |
| AiDotNet_Sigmoid_Double     | ?       |   4,000.65 us |  1,488.052 us | 1,391.925 us |   3,776.50 us |    3352 B |
| TorchSharp_Sigmoid_Double   | ?       |   4,667.17 us |  2,647.460 us | 2,346.904 us |   4,620.70 us |      48 B |
| AiDotNet_Exp_Double         | ?       | 216,093.74 us |  4,141.557 us | 3,671.382 us | 216,512.65 us |    8824 B |
| TorchSharp_Exp_Double       | ?       |   3,570.97 us |  2,122.021 us | 1,881.117 us |   3,467.35 us |      48 B |
| AiDotNet_Log_Double         | ?       | 218,823.29 us |  7,593.615 us | 7,103.072 us | 215,978.10 us |    8720 B |
| TorchSharp_Log_Double       | ?       |   3,093.57 us |  1,207.053 us | 1,070.021 us |   2,878.50 us |      48 B |
| AiDotNet_Tanh_Double        | ?       |   3,235.41 us |    766.791 us |   598.660 us |   3,266.10 us |    3752 B |
| TorchSharp_Tanh_Double      | ?       |   1,007.89 us |    536.499 us |   418.863 us |     867.10 us |      48 B |
| AiDotNet_GELU_Double        | ?       |   3,129.05 us |    964.773 us |   805.629 us |   2,985.30 us |    3776 B |
| TorchSharp_GELU_Double      | ?       |   1,113.23 us |    317.412 us |   296.908 us |   1,019.00 us |      48 B |
| AiDotNet_Mish_Double        | ?       |   3,286.34 us |    699.470 us |   546.100 us |   3,249.05 us |    6536 B |
| TorchSharp_Mish_Double      | ?       |   6,919.77 us |  5,040.400 us | 4,714.794 us |   5,199.10 us |     192 B |
| AiDotNet_Softmax_Double     | ?       |  14,673.92 us |  3,179.937 us | 2,818.931 us |  14,977.75 us | 8407504 B |
| TorchSharp_Softmax_Double   | ?       |     340.91 us |    121.252 us |   107.487 us |     319.20 us |      48 B |
| AiDotNet_Conv2D_Double      | ?       |   5,126.84 us |  4,746.677 us | 4,440.045 us |   4,348.20 us |  131224 B |
| TorchSharp_Conv2D_Double    | ?       |   4,886.60 us |  6,152.217 us | 5,754.788 us |   1,021.55 us |      48 B |
| AiDotNet_TensorMatMul       | 256     |     832.05 us |  1,216.180 us | 1,015.565 us |     342.20 us |  262344 B |
| TorchSharp_MatMul           | 256     |     309.05 us |     22.569 us |    18.846 us |     309.40 us |      48 B |
| AiDotNet_TensorMatMul       | 512     |   2,954.12 us |  2,554.959 us | 2,133.506 us |   1,748.35 us | 1048808 B |
| TorchSharp_MatMul           | 512     |   1,116.64 us |    207.927 us |   194.495 us |   1,008.90 us |      48 B |
| AiDotNet_TensorAdd          | 100000  |      51.01 us |      9.775 us |     9.144 us |      47.70 us |     200 B |
| RawTensorPrimitives_Add     | 100000  |     197.71 us |     66.365 us |    58.831 us |     181.45 us |         - |
| TorchSharp_Add              | 100000  |   4,263.03 us |  4,833.916 us | 4,521.648 us |   2,939.35 us |      24 B |
| AiDotNet_TensorMultiply     | 100000  |      39.82 us |      3.792 us |     3.167 us |      39.30 us |     200 B |
| TorchSharp_Multiply         | 100000  |   1,132.33 us |  1,352.956 us | 1,265.556 us |     377.50 us |         - |
| AiDotNet_TensorAdd          | 1000000 |   1,241.90 us |    974.030 us |   760.459 us |   1,028.50 us |    2512 B |
| RawTensorPrimitives_Add     | 1000000 |     953.17 us |     89.154 us |    79.033 us |     948.00 us |         - |
| TorchSharp_Add              | 1000000 |     774.25 us |    880.223 us |   735.026 us |     421.10 us |      24 B |
| TorchSharp_Add_1Thread      | 1000000 |     696.37 us |     49.581 us |    43.952 us |     686.35 us |      24 B |
| AiDotNet_TensorMultiply     | 1000000 |   2,072.63 us |  2,251.242 us | 2,105.813 us |     852.10 us |    2248 B |
| TorchSharp_Multiply         | 1000000 |   4,999.92 us |  5,605.950 us | 5,243.810 us |   1,961.00 us |         - |
