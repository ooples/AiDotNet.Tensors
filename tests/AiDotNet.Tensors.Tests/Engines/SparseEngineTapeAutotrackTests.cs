// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The ISparseEngine tape-aware Tensor&lt;T&gt; operations must AUTOMATICALLY record
/// themselves on the active autodiff tape — mirroring every dense IEngine op — so a
/// consumer (e.g. SparseLinearLayer&lt;T&gt;) gets gradients into a sparse trainable
/// parameter just by calling the engine method, with no manual SparseAutograd plumbing.
/// Without a tape active they must be a plain forward. These tests lock in that contract
/// at the interface boundary (the underlying Record backward math is covered separately in
/// SparseCompletenessTests).
/// </summary>
public class SparseEngineTapeAutotrackTests
{
    private readonly CpuEngine _engine = new();
    private readonly ISparseEngine _sparse = CpuSparseEngine.Instance;

    // [[1,0,2,0],[0,3,0,4],[5,0,6,0],[0,7,0,8]]
    private static SparseTensor<float> SmallA() =>
        SparseTensor<float>.FromDense(MakeDense(new float[,]
        {
            { 1, 0, 2, 0 },
            { 0, 3, 0, 4 },
            { 5, 0, 6, 0 },
            { 0, 7, 0, 8 },
        }));

    private static Tensor<float> MakeDense(float[,] data)
    {
        int r = data.GetLength(0), c = data.GetLength(1);
        var t = new Tensor<float>(new[] { r, c });
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                t[i, j] = data[i, j];
        return t;
    }

    private static bool AnyNonZero(Tensor<float> t)
    {
        var span = t.AsSpan();
        for (int i = 0; i < span.Length; i++)
            if (span[i] != 0f) return true;
        return false;
    }

    [Fact]
    public void SparseMatMul_WithActiveTape_AutoRecordsAndGradientsReachBothOperands()
    {
        var a = SmallA();                                  // 4x4 sparse (the "weight")
        var b = MakeDense(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 } }); // 4x2 dense

        using var tape = new GradientTape<float>();
        var output = _sparse.SparseMatMul(a, b);           // auto-records on the tape
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new Tensor<float>[] { a, b });

        // The dense operand always tracks; the sparse operand — the whole point — must too.
        Assert.True(grads.ContainsKey(b), "dense operand b received no gradient");
        Assert.True(AnyNonZero(grads[b]), "dense operand b gradient is all-zero");
        Assert.True(grads.ContainsKey(a), "sparse operand a received no gradient (tape not tracked)");
        Assert.True(AnyNonZero(grads[a]), "sparse operand a gradient is all-zero");
    }

    [Fact]
    public void SparseMatMul_WithoutTape_IsPlainForwardMatchingDense()
    {
        var a = SmallA();
        var b = MakeDense(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 } });

        // No GradientTape in scope -> zero-overhead plain forward.
        var output = _sparse.SparseMatMul(a, b);
        var expected = _engine.TensorMatMul(a.ToDense(), b);

        Assert.Equal(expected._shape, output._shape);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected.AsSpan()[i], output.AsSpan()[i], 3);
    }

    [Fact]
    public void SparseAddMM_WithActiveTape_AutoRecordsGradients()
    {
        var a = SmallA();                                  // 4x4
        var b = MakeDense(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 } }); // 4x2
        var c = MakeDense(new float[,] { { 0, 1 }, { 1, 0 }, { 1, 1 }, { 0, 0 } }); // 4x2

        using var tape = new GradientTape<float>();
        var output = _sparse.SparseAddMM(c, a, b, alpha: 1f, beta: 1f);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new Tensor<float>[] { b, c });

        Assert.True(grads.ContainsKey(b) && AnyNonZero(grads[b]), "b gradient missing/zero");
        Assert.True(grads.ContainsKey(c) && AnyNonZero(grads[c]), "c gradient missing/zero");
    }

    [Fact]
    public void SparseSum_WithActiveTape_AutoRecordsGradient()
    {
        var a = SmallA();
        using var tape = new GradientTape<float>();
        var sum = _sparse.SparseSum(a);
        var grads = tape.ComputeGradients(sum, new Tensor<float>[] { a });

        // d(sum)/dx = 1 everywhere -> gradient exists and is non-zero.
        Assert.True(grads.ContainsKey(a), "sparse sum did not record on the tape");
        Assert.True(AnyNonZero(grads[a]), "sparse sum gradient is all-zero");
    }

    [Fact]
    public void SparseSoftmax_WithActiveTape_AutoRecordsFiniteGradient()
    {
        var a = SmallA();
        using var tape = new GradientTape<float>();
        var sm = _sparse.SparseSoftmax(a);
        var loss = _engine.ReduceSum(sm, null);
        var grads = tape.ComputeGradients(loss, new Tensor<float>[] { a });

        Assert.True(grads.ContainsKey(a), "sparse softmax did not record on the tape");
        var span = grads[a].AsSpan();
        for (int i = 0; i < span.Length; i++)
            Assert.False(float.IsNaN(span[i]) || float.IsInfinity(span[i]),
                $"sparse softmax gradient non-finite at {i}: {span[i]}");
    }
}
