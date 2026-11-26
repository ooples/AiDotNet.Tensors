// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The bias epilogue of the fused linear on paths the 2D float tests do not reach: a rank-3 float input (one GEMM over
/// the collapsed batch) and the double destination-passing form. Each is checked against a naive x·W + b.
/// </summary>
public class FusedLinearBiasEpilogueTests
{
    private const int K = 6, N = 5;

    private static double X(int i) => ((i * 5 + 3) % 11 - 5) / 4.0;
    private static double W(int i) => ((i * 3 + 1) % 7 - 3) / 8.0;
    private static double B(int j) => 0.25 * j - 0.5;

    private static double Naive(int row, int col)
    {
        double s = B(col);
        for (int k = 0; k < K; k++) s += X(row * K + k) * W(k * N + col);
        return s;
    }

    [Fact]
    public void A_rank_three_float_input_adds_the_bias_to_every_row_of_every_batch()
    {
        const int batch = 2, rows = 3;
        var x = new Tensor<float>(new[] { batch, rows, K });
        for (int i = 0; i < x.Length; i++) x[i] = (float)X(i);
        var w = new Tensor<float>(new[] { K, N });
        for (int i = 0; i < w.Length; i++) w[i] = (float)W(i);
        var b = new Tensor<float>(new[] { N });
        for (int j = 0; j < N; j++) b[j] = (float)B(j);

        var y = new CpuEngine().FusedLinear(x, w, b, FusedActivationType.None);

        Assert.Equal(new[] { batch, rows, N }, y.Shape.ToArray());
        for (int r = 0; r < batch * rows; r++)
            for (int c = 0; c < N; c++)
                Assert.Equal(Naive(r, c), y[r * N + c], 5);
    }

    [Fact]
    public void A_double_linear_into_a_destination_adds_the_bias()
    {
        const int rows = 4;
        var x = new Tensor<double>(new[] { rows, K });
        for (int i = 0; i < x.Length; i++) x[i] = X(i);
        var w = new Tensor<double>(new[] { K, N });
        for (int i = 0; i < w.Length; i++) w[i] = W(i);
        var b = new Tensor<double>(new[] { N });
        for (int j = 0; j < N; j++) b[j] = B(j);
        var y = new Tensor<double>(new[] { rows, N });

        new CpuEngine().FusedLinearInto(y, x, w, b, FusedActivationType.None);

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < N; c++)
                Assert.Equal(Naive(r, c), y[r * N + c], 12);
    }
}
