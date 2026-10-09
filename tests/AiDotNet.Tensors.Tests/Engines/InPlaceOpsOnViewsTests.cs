using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// An in-place op handed a non-contiguous view must write through that view. Several used to make a
/// contiguous copy, update the copy, and return, leaving the caller's tensor untouched without an
/// error (#653 in-place work).
/// </summary>
public class InPlaceOpsOnViewsTests
{
    private readonly CpuEngine _engine = new();

    private static Tensor<float> Base(int seed)
    {
        var t = new Tensor<float>(new[] { 6, 5 });
        for (int i = 0; i < t.Length; i++) t[i] = (float)(Math.Sin(seed + i) * 2.0);
        return t;
    }

    // Absolute tolerance: in-place and out-of-place kernels differ by an ulp on some targets (Mish on
    // net471), while an unchanged view misses by ~0.3. Rounded-precision Assert.Equal flips at a digit edge.
    /// <summary>Applies <paramref name="inPlace"/> to a transposed view of a fresh base tensor and
    /// <paramref name="outOfPlace"/> to a contiguous copy of the same view, then compares.</summary>
    private void AssertWritesThroughView(Action<Tensor<float>> inPlace, Func<Tensor<float>, Tensor<float>> outOfPlace)
    {
        var storage = Base(1);
        var view = storage.Transpose();
        Assert.False(view.IsContiguous);
        var expected = outOfPlace(view.Contiguous());

        inPlace(view);

        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 6; j++)
                Assert.True(Math.Abs(expected[i, j] - storage[j, i]) < 1e-5f,
                    $"[{i},{j}] expected {expected[i, j]} but the view holds {storage[j, i]}");
    }

    [Fact] public void Swish() => AssertWritesThroughView(t => _engine.SwishInPlace(t), t => _engine.Swish(t));
    [Fact] public void GELU() => AssertWritesThroughView(t => _engine.GELUInPlace(t), t => _engine.GELU(t));
    [Fact] public void Tanh() => AssertWritesThroughView(t => _engine.TanhInPlace(t), t => _engine.Tanh(t));
    [Fact] public void Mish() => AssertWritesThroughView(t => _engine.MishInPlace(t), t => _engine.Mish(t));
    [Fact] public void Sigmoid() => AssertWritesThroughView(t => _engine.SigmoidInPlace(t), t => _engine.Sigmoid(t));
    [Fact] public void ReLU() => AssertWritesThroughView(t => _engine.ReLUInPlace(t), t => _engine.ReLU(t));
    [Fact] public void LeakyReLU() => AssertWritesThroughView(t => _engine.LeakyReLUInPlace(t, 0.1f), t => _engine.LeakyReLU(t, 0.1f));

    // In-place arithmetic does not write through a view: it saves the input for the tape and bumps the
    // version before mutating, so it rejects a non-contiguous target outright. Loud, unlike the silent
    // no-op the activations had; pinned so it cannot quietly turn into one.
    [Fact]
    public void TensorMultiply_RejectsANonContiguousTarget()
    {
        var other = Base(2).Transpose().Contiguous();
        var view = Base(1).Transpose();
        Assert.Throws<InvalidOperationException>(() => _engine.TensorMultiplyInPlace(view, other));
    }

    [Fact]
    public void TensorSubtract_RejectsANonContiguousTarget()
    {
        var other = Base(3).Transpose().Contiguous();
        var view = Base(1).Transpose();
        Assert.Throws<InvalidOperationException>(() => _engine.TensorSubtractInPlace(view, other));
    }

    [Fact]
    public void TensorAdd_RejectsANonContiguousTarget()
    {
        var other = Base(4).Transpose().Contiguous();
        var view = Base(1).Transpose();
        Assert.Throws<InvalidOperationException>(() => _engine.TensorAddInPlace(view, other));
    }
    [Fact]
    public void TensorBroadcastAdd()
    {
        var row = new Tensor<float>(new[] { 1, 6 });
        for (int i = 0; i < 6; i++) row[i] = 0.25f * i;
        AssertWritesThroughView(t => _engine.TensorBroadcastAddInPlace(t, row), t => _engine.TensorAdd(t, row));
    }
}
