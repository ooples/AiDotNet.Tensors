using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The slice backward runs on the device for a GPU engine: a zero tensor of the input's shape with the upstream
/// gradient written at the slice origin. The host version downloaded the whole upstream gradient - a sync per step,
/// and inside a whole-step CUDA-graph capture an illegal operation (CUDA 900) that aborted the capture.
/// </summary>
[Collection("DirectGpuSerial")]
public class SliceBackwardDeviceTests
{
    [SkippableTheory]
    [InlineData(new[] { 3, 5 }, new[] { 1, 1 }, new[] { 2, 3 })]
    [InlineData(new[] { 2, 3, 4 }, new[] { 1, 0, 2 }, new[] { 1, 3, 2 })]
    [InlineData(new[] { 4, 6 }, new[] { 0, 0 }, new[] { 4, 6 })]
    public void Slice_gradient_on_the_gpu_scatters_into_zeros_at_the_origin(int[] shape, int[] start, int[] size)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception) { }
        Skip.IfNot(gpu is not null && gpu.IsGpuAvailable, "GPU backend did not resolve.");
        var prior = AiDotNetEngine.Current;
        try
        {
            IEngine engine = gpu!;
            AiDotNetEngine.Current = gpu!;
            int n = 1; foreach (var d in shape) n *= d;
            int m = 1; foreach (var d in size) m *= d;
            var x = new Tensor<float>(shape);
            for (int i = 0; i < n; i++) x[i] = 0.1f * i - 1f;
            var w = new Tensor<float>(size);
            for (int i = 0; i < m; i++) w[i] = 0.5f + 0.25f * i;
            x.Gpu();

            Tensor<float> gx;
            using (var tape = new GradientTape<float>())
            {
                var slice = engine.TensorSlice(x, start, size);
                var loss = engine.ReduceSum(engine.TensorMultiply(slice, w), null);
                gx = tape.ComputeGradients(loss)[x];
            }

            var expected = new float[n];
            for (int i = 0; i < m; i++)
            {
                int rem = i, flat = 0, stride = 1;
                for (int d = shape.Length - 1; d >= 0; d--)
                {
                    int idx = rem % size[d]; rem /= size[d];
                    flat += (start[d] + idx) * stride;
                    stride *= shape[d];
                }
                expected[flat] = w[i];
            }
            var actual = gx.ToArray();
            for (int i = 0; i < n; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-6f, $"dx[{i}] = {actual[i]}, expected {expected[i]}");
        }
        finally
        {
            AiDotNetEngine.Current = prior;
            gpu?.Dispose();
        }
    }
}
