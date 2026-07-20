using System;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests;

/// <summary>
/// Unit tests for the central device-dispatch policy. Devices are set directly via the internal setter
/// (InternalsVisibleTo) so the resolution logic is exercised without a real multi-GPU box.
/// </summary>
public sealed class DeviceDispatchTests
{
    private static Tensor<float> OnDevice(TensorDevice device, int index = 0)
    {
        var t = new Tensor<float>(new[] { 2, 2 }); // non-empty so it isn't skipped as "deferred"
        t.Device = device;
        t._gpuDeviceIndex = index;
        return t;
    }

    private static Tensor<float> Cpu() => new Tensor<float>(new[] { 2, 2 });

    [Fact]
    public void AllCpu_ResolvesToCpu()
    {
        var r = DeviceDispatch.Resolve(Cpu(), Cpu());
        Assert.Equal(TensorDevice.CPU, r.Device);
        Assert.False(r.IsGpu);
    }

    [Fact]
    public void AllSameGpu_ResolvesToThatGpu()
    {
        var r = DeviceDispatch.Resolve(OnDevice(TensorDevice.CUDA, 1), OnDevice(TensorDevice.CUDA, 1));
        Assert.Equal(TensorDevice.CUDA, r.Device);
        Assert.Equal(1, r.Index);
        Assert.True(r.IsGpu);
    }

    [Fact]
    public void GpuPlusCpu_Permissive_ResolvesToGpu()
    {
        var previous = DeviceDispatch.Mode;
        try
        {
            DeviceDispatch.Mode = DeviceDispatchMode.Permissive;
            var r = DeviceDispatch.Resolve(OnDevice(TensorDevice.OpenCL), Cpu());
            Assert.Equal(TensorDevice.OpenCL, r.Device);
            Assert.True(r.IsGpu);
        }
        finally { DeviceDispatch.Mode = previous; }
    }

    [Fact]
    public void GpuPlusCpu_Strict_Throws()
    {
        var previous = DeviceDispatch.Mode;
        try
        {
            DeviceDispatch.Mode = DeviceDispatchMode.Strict;
            var ex = Assert.Throws<DeviceMismatchException>(
                () => DeviceDispatch.Resolve(OnDevice(TensorDevice.CUDA), Cpu()));
            Assert.Equal(TensorDevice.CUDA, ex.Expected.Type);
            Assert.Equal(TensorDevice.CPU, ex.Actual.Type);
        }
        finally { DeviceDispatch.Mode = previous; }
    }

    [Fact]
    public void DifferentGpuTypes_AlwaysThrows()
    {
        var previous = DeviceDispatch.Mode;
        try
        {
            DeviceDispatch.Mode = DeviceDispatchMode.Permissive; // cross-GPU is an error even when permissive
            Assert.Throws<DeviceMismatchException>(
                () => DeviceDispatch.Resolve(OnDevice(TensorDevice.CUDA), OnDevice(TensorDevice.OpenCL)));
        }
        finally { DeviceDispatch.Mode = previous; }
    }

    [Fact]
    public void DifferentGpuIndices_AlwaysThrows()
    {
        Assert.Throws<DeviceMismatchException>(
            () => DeviceDispatch.Resolve(OnDevice(TensorDevice.CUDA, 0), OnDevice(TensorDevice.CUDA, 1)));
    }

    [Fact]
    public void NullAndDeferredOperands_AreIgnored()
    {
        var deferred = new Tensor<float>(new[] { 0 }); // zero-length => carries no device
        var r = DeviceDispatch.Resolve(null!, deferred, OnDevice(TensorDevice.Metal, 2));
        Assert.Equal(TensorDevice.Metal, r.Device);
        Assert.Equal(2, r.Index);
    }

    [Fact]
    public void NoOperands_ResolvesToCpu()
    {
        var r = DeviceDispatch.Resolve<float>();
        Assert.Equal(TensorDevice.CPU, r.Device);
    }

    [Fact]
    public void EnforceStrict_Permissive_IsCompleteNoOp()
    {
        var previous = DeviceDispatch.Mode;
        try
        {
            DeviceDispatch.Mode = DeviceDispatchMode.Permissive;
            // Even genuinely-mismatched operands do not throw while permissive: the guard returns immediately.
            DeviceDispatch.EnforceStrict(OnDevice(TensorDevice.CUDA), Cpu());
            DeviceDispatch.EnforceStrict(OnDevice(TensorDevice.CUDA, 0), OnDevice(TensorDevice.OpenCL));
        }
        finally { DeviceDispatch.Mode = previous; }
    }

    [Fact]
    public void EnforceStrict_Strict_ThrowsOnMismatch_PassesOnSameDevice()
    {
        var previous = DeviceDispatch.Mode;
        try
        {
            DeviceDispatch.Mode = DeviceDispatchMode.Strict;
            Assert.Throws<DeviceMismatchException>(
                () => DeviceDispatch.EnforceStrict(OnDevice(TensorDevice.CUDA), Cpu()));
            // Same device (and all-CPU) pass cleanly.
            DeviceDispatch.EnforceStrict(OnDevice(TensorDevice.CUDA, 1), OnDevice(TensorDevice.CUDA, 1));
            DeviceDispatch.EnforceStrict(Cpu(), Cpu());
        }
        finally { DeviceDispatch.Mode = previous; }
    }
}
