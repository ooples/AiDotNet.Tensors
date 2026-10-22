using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1519: HIP's device-side finiteness guard (the checks are in <see cref="DeviceFiniteGuardTestBase"/>; HIP sums in
/// double, so there is no float-to-double finish). The optimizer module is also compiled as CUDA C++ under NVRTC (the
/// dialect has nothing HIP-specific), so the guarded kernels are checked on any CUDA box.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class HipDeviceFiniteGuardTests : DeviceFiniteGuardTestBase
{
    protected override bool HasFinishKernel => false;

    protected override BackendLease Lease()
    {
        Skip.IfNot(HipBackend.IsHipAvailable, "HIP is not available.");
        var backend = new HipBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            Skip.If(true, "HIP backend did not initialize.");
        }
        return new BackendLease(backend, owned: true);
    }

    [SkippableFact]
    public void Optimizer_module_compiles_as_cuda_cxx()
    {
        Skip.IfNot(NvrtcNativeBindings.IsAvailable, "NVRTC is not available.");
        IntPtr program = IntPtr.Zero;
        Assert.Equal(NvrtcResult.Success, NvrtcNativeBindings.nvrtcCreateProgram(ref program,
            HipOptimizerKernels.GetSource(), "hip_optimizer", 0, IntPtr.Zero, IntPtr.Zero));
        try
        {
            var result = NvrtcNativeBindings.nvrtcCompileProgram(program, 1, new[] { "--gpu-architecture=compute_75" });
            Assert.True(result == NvrtcResult.Success, $"the HIP optimizer module did not compile ({result})");
        }
        finally
        {
            NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
        }
        foreach (var name in new[] { "sgd_update_guarded", "adam_update_guarded", "adamw_update_guarded" })
            Assert.Contains(name, HipOptimizerKernels.GetKernelNames());
    }
}