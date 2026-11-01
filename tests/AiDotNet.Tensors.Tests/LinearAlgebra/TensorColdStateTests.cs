// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// The rarely-set tensor state (streaming, offload, uniform fill) lives in a side object allocated on first use: a
/// fresh tensor reads the documented defaults, values round-trip, and ordinary tensors stay small.
/// </summary>
public class TensorColdStateTests
{
    [Fact]
    public void A_fresh_tensor_reads_the_defaults()
    {
        var t = new Tensor<float>(new[] { 2, 3 });
        Assert.Equal(-1, t.StreamingPoolHandle);
        Assert.Equal(-1, t.OffloadRegistryHandle);
        Assert.Equal(0, t.StreamingReservedBytes);
        Assert.False(t.StreamingDropDeferred);
        Assert.Equal(0, t.StreamingStoreEncoding);
        Assert.Equal(IntPtr.Zero, t.OffloadDevicePointer);
        Assert.Equal(IntPtr.Zero, t.OffloadHostPointer);
        Assert.Null(t.OffloadOpaqueHandle);
        Assert.Equal(0, t.OffloadByteCount);
        Assert.Null(t.UniformFillValue);
        Assert.Null(t.StreamingInt8);
        Assert.Null(t.StreamingInt4);
    }

    [Fact]
    public void Values_round_trip_and_reset()
    {
        var t = new Tensor<float>(new[] { 4 });
        var handle = new object();
        t.StreamingPoolHandle = 7;
        t.StreamingReservedBytes = 4096;
        t.StreamingDropDeferred = true;
        t.StreamingStoreEncoding = 1;
        t.OffloadDevicePointer = new IntPtr(0x1000);
        t.OffloadHostPointer = new IntPtr(0x2000);
        t.OffloadOpaqueHandle = handle;
        t.OffloadByteCount = 16;
        t.OffloadRegistryHandle = 3;
        t.UniformFillValue = 2.5;

        Assert.Equal(7, t.StreamingPoolHandle);
        Assert.Equal(4096, t.StreamingReservedBytes);
        Assert.True(t.StreamingDropDeferred);
        Assert.Equal(1, t.StreamingStoreEncoding);
        Assert.Equal(new IntPtr(0x1000), t.OffloadDevicePointer);
        Assert.Equal(new IntPtr(0x2000), t.OffloadHostPointer);
        Assert.Same(handle, t.OffloadOpaqueHandle);
        Assert.Equal(16, t.OffloadByteCount);
        Assert.Equal(3, t.OffloadRegistryHandle);
        Assert.Equal(2.5, t.UniformFillValue);

        t.StreamingPoolHandle = -1;
        t.OffloadRegistryHandle = -1;
        t.UniformFillValue = null;
        Assert.Equal(-1, t.StreamingPoolHandle);
        Assert.Equal(-1, t.OffloadRegistryHandle);
        Assert.Null(t.UniformFillValue);
        // The others are untouched by resetting these.
        Assert.Equal(4096, t.StreamingReservedBytes);
    }

    [Fact]
    public void A_write_invalidates_the_uniform_fill_marker()
    {
        var t = new Tensor<float>(new[] { 3 });
        t.UniformFillValue = 1.0;
        t[0] = 5f;
        Assert.Null(t.UniformFillValue);
    }

#if NET
    // Every tensor an op returns pays its size in allocation and GC work. A GPU-resident [8, 64] float tensor
    // (no host array) allocated 608 bytes with these fields inline and 488 with the side object (net10.0, x64).
    [Fact]
    public void A_gpu_resident_tensor_does_not_carry_the_cold_fields()
    {
        var shape = new[] { 8, 64 };
        for (int i = 0; i < 3; i++) GC.KeepAlive(Tensor<float>.CreateGpuResident(shape));
        long before = GC.GetAllocatedBytesForCurrentThread();
        var t = Tensor<float>.CreateGpuResident(shape);
        long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
        GC.KeepAlive(t);
        Assert.True(bytes <= 520, $"a GPU-resident tensor allocated {bytes} bytes");
    }
#endif
}
