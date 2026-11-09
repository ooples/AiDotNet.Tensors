using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// GPU TensorScatter and TensorTopK in training: forward and gradients must match the CPU engine, and the device path
/// must serve the call (fallbacks throw). Repeated indices are included: torch.scatter keeps the last write, which the
/// device resolves with an amax scatter of write positions, and only that write's source element gets the gradient.
/// They SKIP when no backend is present rather than passing vacuously.
/// </summary>
[Collection("DirectGpuSerial")]
public class ScatterTopKGpuParityTests
{
    private static float[] Gen(int n, int seed) =>
        Enumerable.Range(0, n).Select(i => (float)Math.Sin(0.7 * (i + 1) + 1.9 * seed)).ToArray();

    // Indices along the axis with repeats inside a fibre, so overwritten writes are exercised.
    private static Tensor<int> RepeatingIndices(int[] shape, int ax, int count)
    {
        var idxShape = (int[])shape.Clone();
        idxShape[ax] = count;
        int n = idxShape.Aggregate(1, (a, b) => a * b);
        return new Tensor<int>(Enumerable.Range(0, n).Select(i => (i * 7 + i / 3) % shape[ax] / 2).ToArray(), idxShape);
    }

    private static (float[] y, float[] gDst, float[] gSrc) RunScatter(IEngine eng, int[] shape, int axis, int count)
    {
        int ax = axis < 0 ? shape.Length + axis : axis;
        var idx = RepeatingIndices(shape, ax, count);
        int n = shape.Aggregate(1, (a, b) => a * b);
        var dst = new Tensor<float>(shape, new Vector<float>(Gen(n, 1)));
        var src = new Tensor<float>(idx.Shape.ToArray(), new Vector<float>(Gen(idx.Length, 2)));
        var w = new Tensor<float>(shape, new Vector<float>(Gen(n, 3)));
        using var tape = new GradientTape<float>();
        var y = eng.TensorScatter(dst, idx, src, axis);
        var g = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, w), null, keepDims: false), new[] { dst, src });
        return (y.ToArray(), g[dst].ToArray(), g[src].ToArray());
    }

    private static T OnDevice<T>(Func<T> run)
    {
        bool saved = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        try
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
            GpuLaunchProbe.Reset();
            var result = run();
            Assert.True(GpuLaunchProbe.Count > 0, "No GPU work was launched.");
            Assert.Empty(GpuLaunchProbe.Fallbacks);
            return result;
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = saved;
        }
    }

    [SkippableTheory]
    [InlineData(new[] { 3, 8 }, 1, 6)]
    [InlineData(new[] { 9 }, 0, 7)]
    [InlineData(new[] { 6, 5 }, 0, 4)]
    [InlineData(new[] { 2, 7, 4 }, 1, 5)]
    [InlineData(new[] { 2, 3, 6 }, -1, 4)]
    public void Scatter_with_repeated_indices_trains_on_the_device(int[] shape, int axis, int count)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        var expected = RunScatter(new CpuEngine(), shape, axis, count);
        var actual = OnDevice(() => RunScatter(gpu, shape, axis, count));

        Assert.Equal(expected.y, actual.y);
        Assert.Equal(expected.gDst, actual.gDst);
        Assert.Equal(expected.gSrc, actual.gSrc);
    }

    [SkippableFact]
    public void TopK_gradient_matches_the_cpu_on_the_device()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        (float[] values, float[] grad) Run(IEngine eng)
        {
            var x = new Tensor<float>(new[] { 4, 9 }, new Vector<float>(Gen(36, 4)));
            var w = new Tensor<float>(new[] { 4, 3 }, new Vector<float>(Gen(12, 5)));
            using var tape = new GradientTape<float>();
            var values = eng.TensorTopK(x, 3, 1, out _);
            var g = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(values, w), null, keepDims: false), new[] { x });
            return (values.ToArray(), g[x].ToArray());
        }

        var expected = Run(new CpuEngine());
        var actual = OnDevice(() => Run(gpu));
        Assert.Equal(expected.values, actual.values);
        Assert.Equal(expected.grad, actual.grad);
    }

    [SkippableFact]
    public void SearchSorted_kernel_orders_nan_like_the_cpu()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        var cpu = new CpuEngine();
        // torch.searchsorted moves right on every comparison with a NaN; the kernels used to move left, so a NaN query
        // returned 0 instead of the sequence length.
        foreach (var seq in new[] { new[] { 1f, 2f, float.NaN }, new[] { 0f, 3f, 4f, 9f }, new[] { float.NaN, float.NaN } })
        {
            var sequence = new Tensor<float>(seq, new[] { seq.Length });
            var values = new Tensor<float>(new[] { float.NaN, 2f, 5f, -1f, 3f, float.PositiveInfinity }, new[] { 6 });
            foreach (bool right in new[] { false, true })
            {
                var expected = cpu.TensorSearchSorted(sequence, values, right).ToArray();
                var actual = OnDevice(() => gpu.TensorSearchSorted(sequence, values, right).ToArray());
                Assert.Equal(expected, actual);
            }
        }
    }
}
