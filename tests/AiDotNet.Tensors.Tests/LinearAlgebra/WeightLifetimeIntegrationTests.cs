// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Issue #276 follow-up gaps: end-to-end Tensor.Lifetime + WeightRegistry
/// dispatch, integrating streaming pool / offload allocator into a real
/// model-author workflow.
///
/// <para>WeightRegistry is process-wide static state — Configure / Reset /
/// Register all share global slots. xUnit parallelizes test classes by
/// default; <c>[Collection("WeightRegistry")]</c> serializes this class
/// against any other class that touches the registry to prevent
/// interleaved-test corruption.</para>
/// </summary>
[Collection("WeightRegistry")]
public class WeightLifetimeIntegrationTests
{
    [Fact]
    public void Tensor_DefaultLifetime_IsDefault()
    {
        var t = new Tensor<float>(new[] { 4 });
        Assert.Equal(WeightLifetime.Default, t.Lifetime);
        Assert.Equal(-1L, t.StreamingPoolHandle);
        Assert.Equal(IntPtr.Zero, t.OffloadDevicePointer);
    }

    [Fact]
    public void RegisterWeight_Streaming_RoutesToStreamingPool()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-life-{Guid.NewGuid():N}");
        try
        {
            WeightRegistry.Configure(new GpuOffloadOptions
            {
                StreamingBackingStorePath = dir,
                StreamingPoolMaxResidentBytes = 1024L * 1024,
            });

            var t = new Tensor<float>(new[] { 64 });
            for (int i = 0; i < 64; i++) t[i] = i;
            t.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(t);

            Assert.True(t.StreamingPoolHandle >= 0);
            Assert.True(WeightRegistry.StreamingPool.ResidentBytes >= 64 * sizeof(float));

            WeightRegistry.UnregisterWeight(t);
            Assert.Equal(-1L, t.StreamingPoolHandle);
        }
        finally
        {
            WeightRegistry.Reset();
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void RegisterWeight_GpuOffload_NoBackend_FallsBackToDefault()
    {
        // No allocator configured → registration silently demotes to Default
        // so consumers can opt in without crashing on hosts that lack the
        // matching GPU runtime.
        WeightRegistry.Reset();
        var t = new Tensor<float>(new[] { 16 });
        t.Lifetime = WeightLifetime.GpuOffload;
        WeightRegistry.RegisterWeight(t);
        Assert.Equal(WeightLifetime.Default, t.Lifetime);
        Assert.Equal(IntPtr.Zero, t.OffloadDevicePointer);
    }
}
