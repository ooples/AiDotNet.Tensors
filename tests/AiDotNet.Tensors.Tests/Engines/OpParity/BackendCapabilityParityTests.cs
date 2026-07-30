#if !NETFRAMEWORK
using System;
using System.Linq;
using System.Reflection;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Cross-backend capability-parity guard (#775). Full GPU residency means every op-family capability
/// interface is implemented on-device by every backend — otherwise the engine's synchronous capability
/// dispatch CPU-falls-back for that family on the backends that lack it. This asserts the op-family
/// interfaces are implemented by all six backends (OpenCL / CUDA / HIP / Metal / Vulkan / WebGPU).
///
/// WebGPU implements these by blocking on its async launchers (sync-over-async), so a regression that
/// drops one of the sync interfaces — e.g. reverting a WebGpuBackend partial to async-only — fails here.
/// </summary>
public sealed class BackendCapabilityParityTests
{
    private const string Ns = "AiDotNet.Tensors.Engines.DirectGpu.";

    private static readonly string[] Backends =
    {
        "OpenCL.OpenClBackend", "CUDA.CudaBackend", "HIP.HipBackend",
        "Metal.MetalBackend", "Vulkan.VulkanBackend", "WebGpu.WebGpuBackend",
    };

    // Op-family capability interfaces that must be resident on EVERY backend. (Backend-specific interfaces
    // like IFftBackend [Metal/Vulkan], IMultiTensorGpuOptimizerBackend [CUDA], IGpuBatchExecution [Vulkan],
    // and the OpenCL-only composite IExtendedConvKernels / scaffold IScatterRowsKernels are intentionally
    // NOT listed — those families are served on other backends via the 9 IExtendedConv sub-interfaces and
    // IResidentIndexBackend, which are also asserted below.)
    public static readonly string[] UniversalFamilies =
    {
        "IAudioBackend", "IDetectionBackend", "IGeometryBackend", "ILinalgBackend",
        "IParity210Backend", "IRoiBackend",
        // extended-conv geometry families + GNN scatter (ported to all 6 backends earlier in #775)
        "IPool3DKernels", "IConv3DBackwardKernels", "IDepthwiseConv2DBackwardKernels",
        "ITrilinearInterpolationKernels", "IConvTranspose3DKernels", "ISpiralConvKernels",
        "IAdaptiveMaxPool2DKernels", "IGaussianSplatKernels", "IResidentIndexBackend",
    };

    public static TheoryData<string> Families()
    {
        var data = new TheoryData<string>();
        foreach (var f in UniversalFamilies) data.Add(f);
        return data;
    }

    [Theory]
    [MemberData(nameof(Families))]
    public void EveryBackend_ImplementsOpFamily(string familyInterface)
    {
        var asm = typeof(AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend).Assembly;
        var missing = Backends
            .Where(b =>
            {
                var t = asm.GetType(Ns + b);
                return t is null || !t.GetInterfaces().Any(i => i.Name == familyInterface);
            })
            .ToArray();

        Assert.True(missing.Length == 0,
            $"{familyInterface} is not implemented (GPU-resident) on: {string.Join(", ", missing)}. " +
            $"That op-family CPU-falls-back there. Implement the sync interface (WebGPU can block on its " +
            $"*Async launcher, per WebGpuBackend.*.cs); do not remove the family from this guard.");
    }
}
#endif
