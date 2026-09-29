using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A tensor built with <c>FromMemory</c> shares the caller's array. The GPU engine keeps a device copy of a weight
/// and refreshes it only when the tensor records a write, so a write through the array must be followed by
/// <c>MarkModified</c>. The head-to-head harness's raw-array SGD did not, and its CNN trained on stale weights.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class FromMemoryMarkModifiedTests
{
    private const int Batch = 8, In = 16, Out = 4;
    private const float Tolerance = 1e-5f;

    private static float[] Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() - 0.5);
        return d;
    }

    [SkippableFact]
    public void WriteThroughTheArray_ThenMarkModified_IsSeenByTheGpuEngine()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");

        var x = new Tensor<float>(Rand(Batch * In, 1), new[] { Batch, In });
        var weights = Rand(In * Out, 2);
        var w = Tensor<float>.FromMemory(weights, new[] { In, Out });
        gpu.FusedLinear(x, w, null, FusedActivationType.None).ToArray();   // the engine now caches a device copy of w

        for (int i = 0; i < weights.Length; i++) weights[i] = -2f * weights[i] + 0.25f;
        w.MarkModified();
        var after = gpu.FusedLinear(x, w, null, FusedActivationType.None).ToArray();

        var expected = new CpuEngine().TensorMatMul(x, new Tensor<float>((float[])weights.Clone(), new[] { In, Out })).ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - after[i]) <= Tolerance * Math.Max(1f, Math.Abs(expected[i])),
                $"[{i}]: GPU {after[i]} after the write, expected {expected[i]}");
    }
}
