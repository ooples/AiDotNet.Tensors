// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NETCOREAPP
using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The driver and management-library counter queries on a backend that exposes neither: each answers false with zeroed
/// outputs (the Metal, WebGPU and OpenCL backends take this path), and a null backend is an argument error.
/// </summary>
public class DeviceCounterQueryTests
{
    [Fact]
    public void A_backend_without_driver_counters_reports_none_and_zeroes_the_outputs()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());

        Assert.False(GpuMemoryStats.TryGetDeviceMemory(backend, out long free, out long total));
        Assert.Equal(0, free);
        Assert.Equal(0, total);
        Assert.False(GpuUtilization.TryRead(backend, out int gpu, out int memory));
        Assert.Equal(0, gpu);
        Assert.Equal(0, memory);
        Assert.False(GpuUtilization.TryReadDeviceMemory(backend, out long used, out long size));
        Assert.Equal(0, used);
        Assert.Equal(0, size);
    }

    [Fact]
    public void A_null_backend_is_an_argument_error()
    {
        Assert.Throws<ArgumentNullException>(() => GpuMemoryStats.TryGetDeviceMemory(null!, out _, out _));
        Assert.Throws<ArgumentNullException>(() => GpuUtilization.TryRead(null!, out _, out _));
        Assert.Throws<ArgumentNullException>(() => GpuUtilization.TryReadDeviceMemory(null!, out _, out _));
    }
}
#endif
