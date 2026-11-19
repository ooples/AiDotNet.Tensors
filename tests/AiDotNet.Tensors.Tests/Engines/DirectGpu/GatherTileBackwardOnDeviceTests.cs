#if NET6_0_OR_GREATER
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Gather's backward along a non-zero axis and Tile's backward take device routes on a GPU engine (an index-add and a
/// reshape-and-sum) instead of host loops; each must give the CPU engine's gradient, and the gather route must stay
/// on the device (no host array in its result before a read).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GatherTileBackwardOnDeviceTests
{
    private static DirectGpuTensorEngine? TryGpu()
    {
        try
        {
            var gpu = new DirectGpuTensorEngine();
            if (gpu.IsGpuAvailable) return gpu;
            gpu.Dispose();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        return null;
    }

    private static Tensor<float> Ramp(int[] shape, float scale)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = ((i * 7) % 23 - 11) * scale;
        return new Tensor<float>(data, shape);
    }

    private static float[] RunGather(IEngine engine, bool onDevice, Tensor<int> indices, int axis)
    {
        var source = Ramp([2, 5, 3], 0.1f);
        var grad = Ramp([2, indices.Length, 3], 0.05f);
        if (onDevice) { source.Gpu(); grad.Gpu(); }
        var grads = new Dictionary<Tensor<float>, Tensor<float>>();
        BackwardFunctions<float>.GatherBackward(grad, [source], grad, [indices, axis], engine, grads);
        return grads[source].ToArray();
    }

    [SkippableFact]
    public void Gather_backward_on_a_later_axis_matches_the_cpu_engine_with_repeated_indices()
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU backend did not resolve.");
        using (gpu)
        {
            var previous = AiDotNetEngine.Current;
            try
            {
                var indices = new Tensor<int>(new[] { 4, 0, 4, 2, 4, 1 }, [6]);
                AiDotNetEngine.Current = new CpuEngine();
                var expected = RunGather(AiDotNetEngine.Current, onDevice: false, indices, axis: 1);
                AiDotNetEngine.Current = gpu!;
                var actual = RunGather(gpu!, onDevice: true, indices, axis: 1);
                Assert.Equal(expected.Length, actual.Length);
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f, $"[{i}] gpu {actual[i]} != cpu {expected[i]}");
                // Never-selected slice 3 gets exactly zero.
                for (int b = 0; b < 2; b++)
                    for (int c = 0; c < 3; c++) Assert.Equal(0f, actual[(b * 5 + 3) * 3 + c]);
            }
            finally { AiDotNetEngine.Current = previous; }
        }
    }

    private static float[] RunTile(IEngine engine, bool onDevice)
    {
        var input = Ramp([2, 3, 4], 0.1f);
        var grad = Ramp([6, 3, 8], 0.05f); // multiples [3, 1, 2]
        if (onDevice) { input.Gpu(); grad.Gpu(); }
        var grads = new Dictionary<Tensor<float>, Tensor<float>>();
        BackwardFunctions<float>.TileBackward(grad, [input], grad, [new[] { 3, 1, 2 }], engine, grads);
        return grads[input].ToArray();
    }

    [SkippableFact]
    public void Tile_backward_by_device_reduction_matches_the_cpu_engine()
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU backend did not resolve.");
        using (gpu)
        {
            var previous = AiDotNetEngine.Current;
            try
            {
                AiDotNetEngine.Current = new CpuEngine();
                var expected = RunTile(AiDotNetEngine.Current, onDevice: false);
                AiDotNetEngine.Current = gpu!;
                var actual = RunTile(gpu!, onDevice: true);
                Assert.Equal(expected.Length, actual.Length);
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f, $"[{i}] gpu {actual[i]} != cpu {expected[i]}");
            }
            finally { AiDotNetEngine.Current = previous; }
        }
    }
}
#endif
