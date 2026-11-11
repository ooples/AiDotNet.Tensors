// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// The <c>.target</c> for PTX ISA 7.1 modules on a compute-capability 8.0+ device. ISA 7.1 defines <c>sm_80</c> and
/// <c>sm_86</c> but not later targets (<c>sm_87</c>, <c>sm_89</c>, <c>sm_90</c> and newer need a later ISA), so those
/// devices get <c>sm_80</c>: PTX for <c>sm_XY</c> JIT-compiles for every <c>sm_MN</c> with MN &gt;= XY.
/// </summary>
/// <remarks>
/// <c>sm_80</c> PTX cannot run on an older device, so a pre-Ampere capability is refused rather than given a module
/// the driver would reject at link time.
/// </remarks>
internal static class PtxIsa71Target
{
    internal static string For(int ccMajor, int ccMinor)
    {
        if (ccMajor < 8)
            throw new System.PlatformNotSupportedException(
                "This tensor-core module targets sm_80 and needs compute capability 8.0+.");
        return ccMajor == 8 && ccMinor == 6 ? "sm_86" : "sm_80";
    }
}
