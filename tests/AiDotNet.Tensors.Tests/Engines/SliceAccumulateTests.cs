using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tests the slice-accumulate pattern: GetSliceAlongDimension → TensorAdd → SetSlice.
/// This pattern is used in conv1d backward to accumulate gradients at specific time positions.
/// Verifies that non-contiguous slice views work correctly in this pattern.
/// </summary>
public class SliceAccumulateTests
{
    private readonly CpuEngine _engine = new();

    [Fact]
    public void SliceAccumulate_MultipleContributions_AccumulatesCorrectly()
    {
        // Simulate conv1d backward: accumulate into dInput[srcT] from multiple t values
        int batch = 2, seqLen = 4, dim = 3;
        var dInput = new Tensor<double>(new[] { batch, seqLen, dim });

        // Simulate two contributions to srcT=1
        var contrib1 = new Tensor<double>(new[] { batch, dim });
        var contrib2 = new Tensor<double>(new[] { batch, dim });
        for (int i = 0; i < contrib1.Length; i++) contrib1[i] = 1.0 + i;
        for (int i = 0; i < contrib2.Length; i++) contrib2[i] = 10.0 + i;

        // First accumulation at srcT=1
        var slice1 = dInput.GetSliceAlongDimension(1, 1);
        var added1 = _engine.TensorAdd(slice1, contrib1);
        dInput.SetSlice(1, 1, added1);

        // Second accumulation at srcT=1
        var slice2 = dInput.GetSliceAlongDimension(1, 1);
        var added2 = _engine.TensorAdd(slice2, contrib2);
        dInput.SetSlice(1, 1, added2);

        // Verify accumulated correctly: should be contrib1 + contrib2
        for (int b = 0; b < batch; b++)
        {
            for (int d = 0; d < dim; d++)
            {
                double expected = contrib1[new[] { b, d }] + contrib2[new[] { b, d }];
                double actual = dInput[new[] { b, 1, d }];
                Assert.Equal(expected, actual, 1e-10);
            }
        }

        // Verify other time positions are still zero
        for (int t = 0; t < seqLen; t++)
        {
            if (t == 1) continue;
            for (int b = 0; b < batch; b++)
                for (int d = 0; d < dim; d++)
                    Assert.Equal(0.0, dInput[new[] { b, t, d }], 1e-10);
        }
    }

    [Fact]
    public void BroadcastMultiply_WithNonContiguousSlice_ProducesCorrectResult()
    {
        // GetSliceAlongDimension on dim 1 returns non-contiguous view
        int batch = 2, seqLen = 3, dim = 4;
        var tensor = new Tensor<double>(new[] { batch, seqLen, dim });
        var rng = new Random(42);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = rng.NextDouble();

        var weight = new Tensor<double>(new[] { 1, dim });
        for (int d = 0; d < dim; d++) weight[new[] { 0, d }] = 0.5 + d * 0.1;

        // Slice and broadcast multiply
        var slice = tensor.GetSliceAlongDimension(1, 1); // [batch, dim], non-contiguous
        var result = _engine.TensorBroadcastMultiply(slice, weight);

        // Verify
        for (int b = 0; b < batch; b++)
        {
            for (int d = 0; d < dim; d++)
            {
                double expected = tensor[new[] { b, 1, d }] * weight[new[] { 0, d }];
                Assert.Equal(expected, result[new[] { b, d }], 1e-10);
            }
        }
    }

    [Fact]
    public void GetSliceAlongDimension_UnderActiveTape_ScattersGradientToSlicedPositionOnly()
    {
        // The fix records a SliceAxisBackward node when a GradientTape is active so
        // gradients flow back to ONLY the sliced position (zeros elsewhere) instead of
        // leaking through storage aliasing. loss = sum(input[:, sliceIndex, :]) =>
        // d loss / d input = 1 at seq==sliceIndex, 0 everywhere else.
        int batch = 2, seqLen = 3, dim = 4;
        var input = new Tensor<double>(new[] { batch, seqLen, dim });
        var rng = new Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() + 0.1;

        const int sliceIndex = 1;
        using var tape = new GradientTape<double>();
        var slice = input.GetSliceAlongDimension(sliceIndex, 1); // [batch, dim] view at seq==1
        var loss = _engine.ReduceSum(slice, null);
        var grads = tape.ComputeGradients(loss, new[] { input });

        Assert.True(grads.ContainsKey(input), "Gradient was not recorded for the sliced input (taped path missing).");
        var g = grads[input];
        Assert.Equal(input.Shape.ToArray(), g.Shape.ToArray());

        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqLen; t++)
                for (int d = 0; d < dim; d++)
                {
                    double expected = (t == sliceIndex) ? 1.0 : 0.0;
                    Assert.Equal(expected, g[new[] { b, t, d }], 1e-10);
                }
    }

    [Fact]
    public void Slice_UnderActiveTape_RecordsBackwardLikeGetSliceAlongDimension()
    {
        // Slice(index) (axis 0) and GetSlice(index) must follow the same taped path as
        // GetSliceAlongDimension(index, 0); otherwise equivalent public APIs behave
        // differently under autodiff. loss = sum(input[sliceIndex]) =>
        // d loss / d input = 1 at row==sliceIndex, 0 elsewhere.
        int batch = 3, dim = 4;
        var input = new Tensor<double>(new[] { batch, dim });
        var rng = new Random(11);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() + 0.1;

        const int sliceIndex = 2;
        using var tape = new GradientTape<double>();
        var slice = input.Slice(sliceIndex); // axis-0 slice must route through the taped path
        var loss = _engine.ReduceSum(slice, null);
        var grads = tape.ComputeGradients(loss, new[] { input });

        Assert.True(grads.ContainsKey(input), "Slice(int) dropped the gradient — it is not routing through the taped path.");
        var g = grads[input];
        for (int b = 0; b < batch; b++)
            for (int d = 0; d < dim; d++)
            {
                double expected = (b == sliceIndex) ? 1.0 : 0.0;
                Assert.Equal(expected, g[new[] { b, d }], 1e-10);
            }
    }
}
