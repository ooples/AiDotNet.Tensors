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

    /// <summary>
    /// Runs one phase (a forward or a backward) and, on the device, proves that phase itself launched GPU work and fell
    /// back nowhere. A whole-run count cannot: the loss ops launch work even when a backward runs on the host. With
    /// <paramref name="noReadbacks"/> the phase may not read a buffer back either, which a host backward must do to see
    /// the device gradient.
    /// </summary>
    private static T Phase<T>(IEngine eng, string name, Func<T> run, bool noReadbacks)
    {
        if (eng is not DirectGpuTensorEngine) return run();
        long launches = GpuLaunchProbe.Count, readbacks = GpuLaunchProbe.Readbacks;
        int fallbacks = GpuLaunchProbe.Fallbacks.Length;
        var result = run();
        Assert.True(GpuLaunchProbe.Count > launches, $"{name} launched no GPU work.");
        Assert.Equal(fallbacks, GpuLaunchProbe.Fallbacks.Length);
        if (noReadbacks)
            Assert.True(GpuLaunchProbe.Readbacks == readbacks,
                $"{name} read {GpuLaunchProbe.Readbacks - readbacks} buffers back to the host.");
        return result;
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
        var y = Phase(eng, "The scatter forward", () => eng.TensorScatter(dst, idx, src, axis), noReadbacks: true);
        var loss = eng.ReduceSum(eng.TensorMultiply(y, w), null, keepDims: false);
        var g = Phase(eng, "The scatter backward", () => tape.ComputeGradients(loss, new[] { dst, src }), noReadbacks: true);
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
            // The top-k forward hands its indices to the host, so only its backward is held to no readbacks.
            var values = Phase(eng, "The top-k forward", () => eng.TensorTopK(x, 3, 1, out _), noReadbacks: false);
            var loss = eng.ReduceSum(eng.TensorMultiply(values, w), null, keepDims: false);
            var g = Phase(eng, "The top-k backward", () => tape.ComputeGradients(loss, new[] { x }), noReadbacks: true);
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

    [SkippableFact]
    public void Scatter_and_take_along_dim_read_raw_int32_resident_indices()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        int[] shape = { 3, 8 };
        var host = RepeatingIndices(shape, 1, 6);
        // An int32 op on the device leaves its result as raw int32 storage, the form argmax indices take.
        var resident = gpu.TensorBitwiseAnd(host, host);
        Assert.True(resident._gpuBufferContainsRawInt32, "The positive control lost its raw int32 storage.");

        (float[] y, float[] gDst, float[] gSrc, float[] taken) Run(IEngine eng, Tensor<int> idx)
        {
            var dst = new Tensor<float>(shape, new Vector<float>(Gen(24, 1)));
            var src = new Tensor<float>(idx.Shape.ToArray(), new Vector<float>(Gen(idx.Length, 2)));
            var w = new Tensor<float>(shape, new Vector<float>(Gen(24, 3)));
            using var tape = new GradientTape<float>();
            var y = eng.TensorScatter(dst, idx, src, 1);
            var g = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, w), null, keepDims: false), new[] { dst, src });
            var taken = eng.TensorTakeAlongDim(w, idx, 1).ToArray();
            return (y.ToArray(), g[dst].ToArray(), g[src].ToArray(), taken);
        }

        var expected = Run(new CpuEngine(), host);
        var actual = OnDevice(() => Run(gpu, resident));
        Assert.Equal(expected.y, actual.y);
        Assert.Equal(expected.gDst, actual.gDst);
        Assert.Equal(expected.gSrc, actual.gSrc);
        Assert.Equal(expected.taken, actual.taken);
    }

    [SkippableFact]
    public void Scatter_backward_uses_the_indices_of_the_forward()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        var dst = new Tensor<float>(new[] { 1, 3 }, new Vector<float>(new[] { 1f, 2f, 3f }));
        var idx = new Tensor<int>(new[] { 0, 1 }, new[] { 1, 2 });
        var src = new Tensor<float>(new[] { 1, 2 }, new Vector<float>(new[] { 5f, 7f }));
        var w = new Tensor<float>(new[] { 1, 3 }, new Vector<float>(new[] { 2f, 3f, 4f }));
        var g = OnDevice(() =>
        {
            using var tape = new GradientTape<float>();
            var y = ((IEngine)gpu).TensorScatter(dst, idx, src, 1);
            // The caller reuses its index tensor before the backward runs.
            idx[0, 0] = 1;
            idx[0, 1] = 0;
            return tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(y, w), null, keepDims: false), new[] { dst, src });
        });
        Assert.Equal(new[] { 2f, 3f }, g[src].ToArray());
        Assert.Equal(new[] { 0f, 0f, 4f }, g[dst].ToArray());
    }

    [SkippableFact]
    public void Taped_index_select_along_a_middle_axis_trains_on_the_device()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        int[] shape = { 2, 5, 3 };
        // A repeated index: its gradient slices accumulate into the one source slice.
        var idx = new Tensor<int>(new[] { 4, 1, 4, 0 }, new[] { 4 });
        (float[] y, float[] g, long forwardReadbacks) Run(IEngine eng, bool upload)
        {
            var host = new Tensor<float>(shape, new Vector<float>(Gen(30, 6)));
            var x = upload && eng is DirectGpuTensorEngine d ? d.UploadToGpu(host, AiDotNet.Tensors.Engines.Gpu.GpuTensorRole.General) : host;
            var w = new Tensor<float>(new[] { 2, 4, 3 }, new Vector<float>(Gen(24, 7)));
            using var tape = new GradientTape<float>();
            GpuLaunchProbe.Reset();
            var y = eng.TensorIndexSelect(x, idx, 1);
            long readbacks = GpuLaunchProbe.Readbacks;
            var grads = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, w), null, keepDims: false), new[] { x });
            return (y.ToArray(), grads[x].ToArray(), readbacks);
        }

        var expected = Run(new CpuEngine(), upload: false);
        var actual = OnDevice(() => Run(gpu, upload: true));
        Assert.True(actual.forwardReadbacks == 0, $"The taped index select read {actual.forwardReadbacks} buffers back.");
        Assert.Equal(expected.y, actual.y);
        Assert.Equal(expected.g, actual.g);
    }

    [Fact]
    public void Index_select_backward_uses_the_indices_of_the_forward()
    {
        var eng = new CpuEngine();
        var x = new Tensor<float>(new[] { 3, 2 }, new Vector<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }));
        var idx = new Tensor<int>(new[] { 2, 0 }, new[] { 2 });
        var w = new Tensor<float>(new[] { 2, 2 }, new Vector<float>(new[] { 1f, 2f, 3f, 4f }));
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> g;
        using (var tape = new GradientTape<float>())
        {
            var y = eng.TensorIndexSelect(x, idx, 0);
            idx[0] = 1;
            g = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, w), null, keepDims: false), new[] { x });
        }
        Assert.Equal(new[] { 3f, 4f, 0f, 0f, 1f, 2f }, g[x].ToArray());
    }

    [SkippableFact]
    public void Empty_scatter_returns_the_destination_and_passes_its_gradient()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        (float[] y, float[] g, int sourceGradLength) Run(IEngine eng)
        {
            var dst = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
            var src = new Tensor<float>(new[] { 0 });
            var idx = new Tensor<int>(new[] { 0 });
            var w = new Tensor<float>(new[] { 4f, 5f, 6f }, new[] { 3 });
            using var tape = new GradientTape<float>();
            var y = eng.TensorScatter(dst, idx, src, 0);
            var g = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, w), null, keepDims: false), new[] { dst, src });
            return (y.ToArray(), g[dst].ToArray(), g[src].Length);
        }

        var expected = Run(new CpuEngine());
        var actual = OnDevice(() => Run(gpu));
        Assert.Equal(new[] { 1f, 2f, 3f }, actual.y);
        Assert.Equal(expected.y, actual.y);
        Assert.Equal(expected.g, actual.g);
        Assert.Equal(0, actual.sourceGradLength);
    }
}
