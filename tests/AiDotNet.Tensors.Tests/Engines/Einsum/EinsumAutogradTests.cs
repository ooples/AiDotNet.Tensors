using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class EinsumAutogradTests
{
    private static Tensor<float> Seq(params int[] shape)
    {
        var t = new Tensor<float>(shape);
        int total = 1; foreach (var d in shape) total *= d;
        var idx = new int[shape.Length];
        for (int i = 0; i < total; i++)
        {
            t[idx] = (i + 1) * 0.1f;
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

    private static bool Close(float a, float b, float eps = 1e-2f)
        => System.MathF.Abs(a - b) <= eps * (1f + System.MathF.Abs(a) + System.MathF.Abs(b));

    // Finite-difference check for a 2-operand einsum. Returns true if the
    // analytical gradient from the tape matches the numerical gradient.
    private static (Tensor<float> gradA, Tensor<float> gradB) Run(string equation, Tensor<float> A, Tensor<float> B)
    {
        var engine = new CpuEngine();
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
var output = engine.TensorEinsum(equation, A, B);
        var loss = engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { A, B });
        return (grads[A], grads[B]);
    }

    [Fact]
    public void Matmul_GradientMatchesReverseRule()
    {
        // Forward: C = A @ B, L = sum(C)
        // dL/dA = ones(C) @ B^T, dL/dB = A^T @ ones(C)
        var A = Seq(2, 3);
        var B = Seq(3, 4);
        var (gA, gB) = Run("ij,jk->ik", A, B);

        // Analytical reference: for L = sum(A@B), dL/dA[i,j] = sum_k B[j,k];
        // dL/dB[j,k] = sum_i A[i,j].
        Assert.Equal(new[] { 2, 3 }, gA.Shape.ToArray());
        Assert.Equal(new[] { 3, 4 }, gB.Shape.ToArray());
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
            {
                float expected = 0;
                for (int k = 0; k < 4; k++) expected += B[j, k];
                Assert.True(Close(expected, gA[i, j]),
                    $"gradA[{i},{j}] expected {expected}, got {gA[i, j]}");
            }
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 4; k++)
            {
                float expected = 0;
                for (int i = 0; i < 2; i++) expected += A[i, j];
                Assert.True(Close(expected, gB[j, k]),
                    $"gradB[{j},{k}] expected {expected}, got {gB[j, k]}");
            }
    }

    [Fact]
    public void Matmul_AgreesWithFiniteDifferences()
    {
        var A = Seq(2, 3);
        var B = Seq(3, 2);
        var (gA, _) = Run("ij,jk->ik", A, B);

        // Check one element of gA against central-difference approximation.
        var engine = new CpuEngine();
        float eps = 1e-3f;

        float Loss(Tensor<float> a, Tensor<float> b)
        {
            var c = engine.TensorEinsum("ij,jk->ik", a, b);
            float s = 0;
            for (int i = 0; i < c.Length; i++) s += c.AsSpan()[i];
            return s;
        }

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
            {
                var plus = (Tensor<float>)A.Clone();
                plus[i, j] += eps;
                var minus = (Tensor<float>)A.Clone();
                minus[i, j] -= eps;
                float numGrad = (Loss(plus, B) - Loss(minus, B)) / (2f * eps);
                Assert.True(Close(numGrad, gA[i, j], 1e-2f),
                    $"finite-diff gA[{i},{j}]: num={numGrad}, tape={gA[i, j]}");
            }
    }

    [Fact]
    public void BatchedMatmul_GradientFlows()
    {
        var A = Seq(2, 3, 4);
        var B = Seq(2, 4, 5);
        var engine = new CpuEngine();
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
var output = engine.TensorEinsum("bij,bjk->bik", A, B);
        var loss = engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { A, B });
        Assert.Equal(A.Shape.ToArray(), grads[A].Shape.ToArray());
        Assert.Equal(B.Shape.ToArray(), grads[B].Shape.ToArray());
    }

    [Fact]
    public void EllipsisBatchedMatmul_GradientFlows()
    {
        // Forces the general path (no hardcoded fast-path for this equation).
        var A = Seq(2, 3, 4);
        var B = Seq(2, 4, 5);
        var engine = new CpuEngine();
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
var output = engine.TensorEinsum("...ij,...jk->...ik", A, B);
        var loss = engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { A, B });
        Assert.Equal(A.Shape.ToArray(), grads[A].Shape.ToArray());
        Assert.Equal(B.Shape.ToArray(), grads[B].Shape.ToArray());
    }

    [Fact]
    public void DiagonalEquation_TapeEntry_HasNoBackward_SoBackwardNoOps()
    {
        // "ii->i" is a diagonal-operand equation. v1 backward doesn't support
        // this: the record step is skipped, so ComputeGradients produces
        // zeros for the input (no path to propagate).
        var A = Seq(3, 3);
        var engine = new CpuEngine();
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var output = engine.TensorEinsum("ii->i", A);
        var loss = engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { A });
        // No recording ⇒ no gradient propagated ⇒ either no entry or zero.
        if (grads.TryGetValue(A, out var gA))
        {
            foreach (var v in gA.AsSpan().ToArray()) Assert.Equal(0f, v);
        }
    }

    [Fact]
    public void ImplicitOutput_RecordsBackward()
    {
        var A = Seq(2, 3);
        var B = Seq(3, 4);
        var engine = new CpuEngine();
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
var output = engine.TensorEinsum("ij,jk", A, B);
        var loss = engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { A, B });
        Assert.True(grads.ContainsKey(A));
        Assert.True(grads.ContainsKey(B));
    }
}
