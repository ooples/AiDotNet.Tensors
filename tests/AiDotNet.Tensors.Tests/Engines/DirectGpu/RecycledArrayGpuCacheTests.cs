#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The GPU activation cache is keyed by a tensor's host array. A pooled array returned to the pool and
/// rented again belongs to a new tensor whose fresh storage can report the same GPU-cache version the old
/// entry was recorded at, so the upload used to hand the new tensor the previous owner's device data.
/// </summary>
[Collection("DirectGpuSerial")]
public class RecycledArrayGpuCacheTests
{
    [SkippableFact]
    public void RecycledPooledArray_IsNotServedThePreviousOwnersDeviceCopy()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException or TypeInitializationException) { }
        Skip.IfNot(gpu is not null && gpu.IsGpuAvailable, "No DirectGpu backend available.");

        using (gpu)
        {
            IEngine engine = gpu!;
            // Large enough to be pooled (exact-size thread-local cache), so the second rent gets the same array.
            int[] shape = { 64, 1024 };
            var weight = new Tensor<float>(new[] { 1024, 16 });
            var bias = new Tensor<float>(new[] { 16 });
            var rng = new Random(7);
            var w = weight.AsWritableSpan();
            for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() - 0.5);
            bias.AsWritableSpan().Fill(0.25f);

            var first = TensorAllocator.RentUninitialized<float>(shape);
            Assert.True(first.PooledArray is not null, "precondition: the rent was not pooled");
            float[] recycled = first.GetDataArray();
            first.AsWritableSpan().Fill(1f);

            // FusedLinearReLU uploads its input through the caching upload: the device copy is cached under the
            // input's array.
            _ = engine.FusedLinearReLU(first, weight, bias).GetDataArray();
            TensorPool.Return(first);

            var reused = TensorAllocator.RentUninitialized<float>(shape);
            Assert.Same(recycled, reused.GetDataArray()); // precondition: the pool handed the same array back
            var r = reused.AsWritableSpan();
            for (int i = 0; i < r.Length; i++) r[i] = (float)(rng.NextDouble() * 2 - 1);

            var gpuOut = engine.FusedLinearReLU(reused, weight, bias).GetDataArray();
            var cpuOut = new CpuEngine().FusedLinearReLU(reused, weight, bias).GetDataArray();
            double maxDiff = 0;
            for (int i = 0; i < cpuOut.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(cpuOut[i] - gpuOut[i]));
            Assert.True(maxDiff < 1e-3, $"GPU used a stale device copy of the recycled input: max |gpu - cpu| = {maxDiff}");
        }
    }
}
#endif
