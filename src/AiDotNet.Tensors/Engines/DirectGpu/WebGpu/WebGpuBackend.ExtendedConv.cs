#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #775: WebGPU implementations of the per-family extended-conv capability interfaces. Added
// incrementally (interface + WGSL kernels) so a partially-ported backend opts into exactly the families
// it can run; the engine routes the rest to the CPU. WGSL kernels live in WebGpuExtendedConvKernels; this
// partial only wires the dispatch via Dispatch{N}BufferAsync. Scalar params are packed into a float[]
// uniform block (ints reinterpreted as their float bit pattern, floats stored directly, padded to a
// 16-byte multiple to satisfy WGSL uniform alignment).
public sealed partial class WebGpuBackend : ITrilinearInterpolationKernels
{
    private static float Bits(int value) => BitConverter.Int32BitsToSingle(value);

    public void TrilinearInterpolate(IGpuBuffer grid, IGpuBuffer positions, IGpuBuffer output,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(p * c);
        if (total <= 0) return;
        Dispatch3BufferAsync("ExtTrilinear", WebGpuExtendedConvKernels.TrilinearInterpolate, "main",
            grid, positions, output,
            [Bits(d), Bits(h), Bits(w), Bits(c), Bits(p), upperEps, 0f, 0f], total).GetAwaiter().GetResult();
    }

    public void TrilinearInterpolateBackward(IGpuBuffer gradOutput, IGpuBuffer positions, IGpuBuffer gradGrid,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(d * h * w * c);
        if (total <= 0) return;
        Dispatch3BufferAsync("ExtTrilinearBackward", WebGpuExtendedConvKernels.TrilinearInterpolateBackward, "main",
            gradOutput, positions, gradGrid,
            [Bits(d), Bits(h), Bits(w), Bits(c), Bits(p), upperEps, 0f, 0f], total).GetAwaiter().GetResult();
    }
}
#endif
