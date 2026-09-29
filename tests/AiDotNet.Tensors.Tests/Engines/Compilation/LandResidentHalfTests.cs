using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// TryLandResidentHalf moves an op result r into the stable node output o and must release every Half copy of r,
/// or each activation is held twice for the rest of the step. r's copy can be cached under its DataVector (a deferred
/// result - the key the lookup checks first) as well as its backing array; only the backing-array key was released.
/// </summary>
[Collection("DirectGpuSerial")]
public class LandResidentHalfTests
{
    private static readonly MethodInfo CacheActivation = typeof(DirectGpuTensorEngine)
        .GetMethod("CacheActivation", BindingFlags.NonPublic | BindingFlags.Instance,
            null, new[] { typeof(object), typeof(IGpuBuffer), typeof(int[]), typeof(IDirectGpuBackend), typeof(bool), typeof(int), typeof(int) }, null)!;

    [SkippableFact]
    public void Landing_releases_the_results_half_copy_under_either_key()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend.");
        Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "TryLandResidentHalf is CUDA-only.");
        var backend = gpu.GetBackend()!;
        const int n = 64;
        var values = new float[n];
        for (int i = 0; i < n; i++) values[i] = i * 0.25f - 3f;

        IGpuBuffer HalfOf(float[] data)
        {
            using var f = backend.AllocateBuffer(data);
            var h = backend.AllocateByteBuffer(n * 2);
            backend.ConvertToFp16(f, h, n);
            return h;
        }

        var r = new Tensor<Half>(new[] { n });
        var o = new Tensor<Half>(new[] { n });
        gpu.ClearActivationCache();
        long baseline = gpu.CurrentActivationCacheBytes;
        CacheActivation.Invoke(gpu, new object?[] { r.DataVector, HalfOf(values), new[] { n }, backend, true, n, 0 });
        CacheActivation.Invoke(gpu, new object?[] { r.GetBackingArrayForCacheLookupUnsafe()!, HalfOf(new float[n]), new[] { n }, backend, true, n, 0 });
        Assert.Equal(baseline + 2L * n * 2, gpu.CurrentActivationCacheBytes);

        Assert.True(gpu.TryLandResidentHalf(r, o));

        Assert.Equal(baseline + (long)n * 2, gpu.CurrentActivationCacheBytes);   // o's entry only
        var landed = o.ToArray();
        for (int i = 0; i < n; i++) Assert.Equal(values[i], (float)landed[i], 3);
        gpu.ClearActivationCache();
    }
}
