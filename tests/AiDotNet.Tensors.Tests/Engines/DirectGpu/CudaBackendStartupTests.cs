using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A machine with a CUDA driver, a device and NVRTC must get a working CUDA backend. The failure this guards was
/// silent: on a CUDA 13 toolkit NVRTC rejected the conv engine's generic lambdas as host functions in JIT mode, the
/// constructor caught the compile error, and the backend reported itself unavailable -- so every CUDA test on that
/// machine skipped as "No GPU device" instead of failing.
/// </summary>
[Collection("DirectGpuSerial")]
public class CudaBackendStartupTests
{
    [SkippableFact]
    public void Cuda_backend_starts_wherever_cuda_and_nvrtc_are_present()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "No CUDA driver or NVRTC.");
        Skip.IfNot(AiDotNet.Tensors.Engines.CuBlasNative.cuInit(0) == CudaResult.Success
            && AiDotNet.Tensors.Engines.CuBlasNative.cuDeviceGetCount(out int devices) == CudaResult.Success && devices > 0,
            "No CUDA device.");
        using var backend = new CudaBackend();
        Assert.True(backend.IsAvailable,
            "CUDA and NVRTC are present but the CUDA backend did not start; its constructor traces the compile error " +
            "(attach a System.Diagnostics.Trace listener to see it).");
    }

    [SkippableFact]
    public void Nvrtc_13_selects_the_device_as_the_default_execution_space()
    {
        Skip.IfNot(NvrtcNativeBindings.IsAvailable, "No NVRTC.");
        Assert.Equal(NvrtcResult.Success, NvrtcNativeBindings.nvrtcVersion(out int major, out _));
        if (major >= 13)
            Assert.Contains("-default-device", NvrtcNativeBindings.JitCompatibilityOptions);
        else
            Assert.Empty(NvrtcNativeBindings.JitCompatibilityOptions);
    }
}