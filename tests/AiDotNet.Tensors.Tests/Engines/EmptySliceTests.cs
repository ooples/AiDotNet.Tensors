// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// An empty slice (zero elements on some axis) is valid wherever it starts in [0, size], the end of the axis included,
/// as x[:, n:n] is in PyTorch and NumPy. Concatenate's backward slices each input's gradient back out of the output's,
/// so before this a concatenation with a zero-width input (a CTGAN-family conditional block over data with no discrete
/// column) threw "Start index n out of range for axis 1 with size n" in the backward pass.
/// </summary>
// The GPU test sets the process-wide fallback switch, so it runs with the other DirectGpu tests, one at a time.
[Collection("DirectGpuSerial")]
public class EmptySliceTests
{
    [Fact]
    public void Slice_ZeroWidthAtTheEndOfAnAxis_IsEmpty()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });

        var tail = engine.TensorSlice(x, new[] { 0, 3 }, new[] { 2, 0 });
        var middle = engine.TensorSlice(x, new[] { 0, 1 }, new[] { 2, 0 });

        Assert.Equal(new[] { 2, 0 }, tail.Shape.ToArray());
        Assert.Equal(0, tail.Length);
        Assert.Equal(new[] { 2, 0 }, middle.Shape.ToArray());
    }

    [Fact]
    public void Slice_EmptyBoxOutsideTheTensor_Throws()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 2, 3 });

        Assert.Throws<ArgumentOutOfRangeException>(() => engine.TensorSlice(x, new[] { 0, 4 }, new[] { 2, 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => engine.TensorSlice(x, new[] { 1, 0 }, new[] { 2, 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => engine.TensorSlice(x, new[] { 0, 0 }, new[] { 2, -1 }));
    }

    // start + length overflows int for these lengths, so a sum-based bounds check would accept them.
    [Fact]
    public void Slice_LengthThatOverflowsTheEndIndex_Throws()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 2, 3 });

        Assert.Throws<ArgumentOutOfRangeException>(() => engine.TensorSlice(x, new[] { 1, 0 }, new[] { int.MaxValue, 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => engine.TensorSlice(x, new[] { 1, 1 }, new[] { int.MaxValue, 1 }));
    }

    /// <summary>
    /// The same step compiled: the zero-width input and the empty slice are recorded with the rest of the graph, and
    /// the plan's gradient matches the tape's on every step.
    /// </summary>
    [Fact]
    public void CompiledTrainingPlan_WithAZeroWidthInput_MatchesTheTape()
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var a = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
            var weights = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
            var bias = new Tensor<float>(new float[] { 1, -1 }, new[] { 2, 1 });
            Tensor<float> Forward(IEngine e)
            {
                var h = e.TensorMultiply(a, a);
                var joined = e.TensorConcatenate(new[] { h, e.TensorSlice(h, new[] { 0, 3 }, new[] { 2, 0 }) }, axis: 1);
                var condition = e.TensorSlice(joined, new[] { 0, 3 }, new[] { 2, 0 });
                var withBias = e.TensorConcatenate(new[] { condition, bias }, axis: 1);
                return e.TensorAdd(e.ReduceSum(e.TensorMultiply(joined, weights), null), e.ReduceSum(withBias, null));
            }
            float[] tape;
            using (var t = new GradientTape<float>())
                tape = t.ComputeGradients(Forward(new CpuEngine()), new[] { a })[a].GetFlattenedData();

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward(new CpuEngine());
                plan = scope.CompileTraining(new[] { a });
            }
            using (plan)
            {
                for (int step = 0; step < 2; step++)
                {
                    plan.Step();
                    Assert.Equal(tape, plan.Gradients[0].ToArray());
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    [Fact]
    public void ConcatenateBackward_WithAZeroWidthInput_RoutesTheGradientToTheOthers()
        => AssertConcatenateBackward(new CpuEngine());

    [SkippableFact]
    public void ConcatenateBackward_WithAZeroWidthInput_OnTheGpuEngine()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        // A CPU fallback would pass the value checks while hiding a missing GPU path, so fallbacks throw here and
        // the probe must record none.
        bool savedThrowOnFallback = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        try
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
            GpuLaunchProbe.Reset();
            AssertConcatenateBackward(gpu);
            Assert.Empty(GpuLaunchProbe.Fallbacks);
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = savedThrowOnFallback;
        }
    }

    private static void AssertConcatenateBackward(IEngine engine)
    {
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var empty = new Tensor<float>(new[] { 2, 0 });
        var weights = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });

        using var tape = new GradientTape<float>();
        var joined = engine.TensorConcatenate(new[] { a, empty }, axis: 1);
        // The conditional block is also read back out of the joined input, as the GAN generator steps do.
        var condition = engine.TensorSlice(joined, new[] { 0, 3 }, new[] { 2, 0 });
        var loss = engine.ReduceSum(engine.TensorMultiply(joined, weights), null);
        var grads = tape.ComputeGradients(loss, new[] { a });

        Assert.Equal(new[] { 2, 0 }, condition.Shape.ToArray());
        Assert.Equal(new[] { 2, 3 }, joined.Shape.ToArray());
        Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, grads[a].ToArray());
    }
}
