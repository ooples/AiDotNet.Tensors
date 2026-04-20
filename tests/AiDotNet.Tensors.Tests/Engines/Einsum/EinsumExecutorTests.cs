using System.Linq;
using AiDotNet.Tensors.Engines.Einsum;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class EinsumExecutorTests
{
    private static Tensor<float> Run(string equation, params Tensor<float>[] operands)
    {
        var eq = EinsumEquation.Parse(equation);
        var shapes = operands.Select(o => o.Shape.ToArray()).ToArray();
        var binding = EinsumShapeBinding.Bind(eq, shapes);
        var path = EinsumPathOptimizer.Greedy(binding);
        return EinsumExecutor.Execute(binding, path, operands);
    }

    private static Tensor<float> Seq(params int[] shape)
    {
        var t = new Tensor<float>(shape);
        int total = 1; foreach (var d in shape) total *= d;
        for (int i = 0; i < total; i++) t.AsSpan(); // force materialisation cost negligible
        var span = new float[total];
        for (int i = 0; i < total; i++) span[i] = i + 1; // 1,2,3,...
        var idx = new int[shape.Length];
        for (int i = 0; i < total; i++)
        {
            t[idx] = span[i];
            // increment idx
            int k = shape.Length - 1;
            while (k >= 0)
            {
                idx[k]++;
                if (idx[k] < shape[k]) break;
                idx[k] = 0;
                k--;
            }
        }
        return t;
    }

    private static bool Close(float a, float b, float eps = 1e-4f)
        => System.MathF.Abs(a - b) <= eps * (1f + System.MathF.Abs(a) + System.MathF.Abs(b));

    // --- Single-operand reshapes ---------------------------------------

    [Fact]
    public void Transpose_ij_to_ji()
    {
        var A = Seq(2, 3);
        var R = Run("ij->ji", A);
        Assert.Equal(new[] { 3, 2 }, R.Shape.ToArray());
        // A[i,j] in row-major = i*3 + j + 1; R[j,i] should match A[i,j]
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(A[i, j], R[j, i]);
    }

    [Fact]
    public void SumReduction_ij_to_scalar()
    {
        var A = Seq(3, 4); // 1..12, sum = 78
        var R = Run("ij->", A);
        Assert.Empty(R.Shape.ToArray());
        Assert.Equal(78f, R[System.Array.Empty<int>()]);
    }

    [Fact]
    public void RowSum_ij_to_i()
    {
        var A = Seq(2, 3); // row 0: 1+2+3=6; row 1: 4+5+6=15
        var R = Run("ij->i", A);
        Assert.Equal(new[] { 2 }, R.Shape.ToArray());
        Assert.Equal(6f, R[0]);
        Assert.Equal(15f, R[1]);
    }

    [Fact]
    public void Trace_ii_to_scalar()
    {
        // 3x3 matrix, trace = 1 + 5 + 9 = 15
        var A = Seq(3, 3);
        var R = Run("ii->", A);
        Assert.Empty(R.Shape.ToArray());
        Assert.Equal(15f, R[System.Array.Empty<int>()]);
    }

    [Fact]
    public void Diagonal_ii_to_i()
    {
        var A = Seq(3, 3);
        var R = Run("ii->i", A);
        Assert.Equal(new[] { 3 }, R.Shape.ToArray());
        Assert.Equal(A[0, 0], R[0]);
        Assert.Equal(A[1, 1], R[1]);
        Assert.Equal(A[2, 2], R[2]);
    }

    // --- Two-operand core patterns -------------------------------------

    [Fact]
    public void Matmul_ij_jk_to_ik()
    {
        var A = Seq(2, 3);
        var B = Seq(3, 4);
        var R = Run("ij,jk->ik", A, B);
        Assert.Equal(new[] { 2, 4 }, R.Shape.ToArray());
        // Reference manual matmul
        for (int i = 0; i < 2; i++)
            for (int k = 0; k < 4; k++)
            {
                float expected = 0;
                for (int j = 0; j < 3; j++) expected += A[i, j] * B[j, k];
                Assert.True(Close(expected, R[i, k]),
                    $"mismatch at [{i},{k}]: expected {expected}, got {R[i, k]}");
            }
    }

    [Fact]
    public void OuterProduct_i_j_to_ij()
    {
        var a = Seq(3);
        var b = Seq(4);
        var R = Run("i,j->ij", a, b);
        Assert.Equal(new[] { 3, 4 }, R.Shape.ToArray());
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                Assert.Equal(a[i] * b[j], R[i, j]);
    }

    [Fact]
    public void BatchMatmul_bij_bjk_to_bik()
    {
        var A = Seq(2, 3, 4);
        var B = Seq(2, 4, 5);
        var R = Run("bij,bjk->bik", A, B);
        Assert.Equal(new[] { 2, 3, 5 }, R.Shape.ToArray());
        for (int b = 0; b < 2; b++)
            for (int i = 0; i < 3; i++)
                for (int k = 0; k < 5; k++)
                {
                    float expected = 0;
                    for (int j = 0; j < 4; j++) expected += A[b, i, j] * B[b, j, k];
                    Assert.True(Close(expected, R[b, i, k]));
                }
    }

    [Fact]
    public void ImplicitOutput_MatchesExplicit()
    {
        var A = Seq(2, 3);
        var B = Seq(3, 4);
        var R1 = Run("ij,jk->ik", A, B);
        var R2 = Run("ij,jk", A, B);
        Assert.Equal(R1.Shape.ToArray(), R2.Shape.ToArray());
        for (int i = 0; i < R1.Length; i++)
            Assert.Equal(R1.AsSpan()[i], R2.AsSpan()[i]);
    }

    // --- Three-operand paths -------------------------------------------

    [Fact]
    public void ChainContraction_ab_bc_cd_to_ad()
    {
        var A = Seq(2, 3);
        var B = Seq(3, 4);
        var C = Seq(4, 5);
        var R = Run("ab,bc,cd->ad", A, B, C);
        Assert.Equal(new[] { 2, 5 }, R.Shape.ToArray());
        // Reference: triple contraction
        for (int i = 0; i < 2; i++)
            for (int l = 0; l < 5; l++)
            {
                float expected = 0;
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 4; k++)
                        expected += A[i, j] * B[j, k] * C[k, l];
                Assert.True(Close(expected, R[i, l]),
                    $"mismatch at [{i},{l}]: expected {expected}, got {R[i, l]}");
            }
    }

    // --- Ellipsis ------------------------------------------------------

    [Fact]
    public void EllipsisBatchedMatmul()
    {
        var A = Seq(2, 3, 4);  // batch=2, i=3, j=4
        var B = Seq(2, 4, 5);  // batch=2, j=4, k=5
        var R = Run("...ij,...jk->...ik", A, B);
        Assert.Equal(new[] { 2, 3, 5 }, R.Shape.ToArray());
        for (int b = 0; b < 2; b++)
            for (int i = 0; i < 3; i++)
                for (int k = 0; k < 5; k++)
                {
                    float expected = 0;
                    for (int j = 0; j < 4; j++) expected += A[b, i, j] * B[b, j, k];
                    Assert.True(Close(expected, R[b, i, k]));
                }
    }

    [Fact]
    public void EllipsisBroadcastsSingleton()
    {
        var A = Seq(1, 3, 4);
        var B = Seq(2, 4, 5);
        var R = Run("...ij,...jk->...ik", A, B);
        Assert.Equal(new[] { 2, 3, 5 }, R.Shape.ToArray());
        // A broadcasts on batch; row 0 of output = row 1 of output (batches
        // are degenerate in A).
        for (int i = 0; i < 3; i++)
            for (int k = 0; k < 5; k++)
            {
                float b0 = 0, b1 = 0;
                for (int j = 0; j < 4; j++)
                {
                    b0 += A[0, i, j] * B[0, j, k];
                    b1 += A[0, i, j] * B[1, j, k];
                }
                Assert.True(Close(b0, R[0, i, k]));
                Assert.True(Close(b1, R[1, i, k]));
            }
    }

    // --- Attention-style -----------------------------------------------

    [Fact]
    public void AttentionScores_bhqd_bhkd_to_bhqk()
    {
        var A = Seq(1, 1, 2, 3);
        var B = Seq(1, 1, 4, 3);
        var R = Run("bhqd,bhkd->bhqk", A, B);
        Assert.Equal(new[] { 1, 1, 2, 4 }, R.Shape.ToArray());
        for (int q = 0; q < 2; q++)
            for (int k = 0; k < 4; k++)
            {
                float expected = 0;
                for (int d = 0; d < 3; d++) expected += A[0, 0, q, d] * B[0, 0, k, d];
                Assert.True(Close(expected, R[0, 0, q, k]));
            }
    }
}
