namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #775: Vulkan implementations of the per-family extended-conv capability interfaces. Added
// incrementally (interface + GLSL kernels) so a partially-ported backend opts into exactly the families
// it can run; the engine routes the rest to the CPU. GLSL kernels live in VulkanExtendedConvKernels;
// this partial only wires the dispatch via GlslDispatchN (int params -> push constants; a float param is
// passed as its raw bit pattern via the shared FloatBits helper and declared `float` in the push block).
public sealed partial class VulkanBackend : ITrilinearInterpolationKernels
{
    public void TrilinearInterpolate(IGpuBuffer grid, IGpuBuffer positions, IGpuBuffer output,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(p * c);
        if (total <= 0) return;
        GlslDispatchN(VulkanExtendedConvKernels.TrilinearInterpolate, total,
            [grid, positions, output],
            [(uint)d, (uint)h, (uint)w, (uint)c, (uint)p, FloatBits(upperEps)]);
    }

    public void TrilinearInterpolateBackward(IGpuBuffer gradOutput, IGpuBuffer positions, IGpuBuffer gradGrid,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(d * h * w * c);
        if (total <= 0) return;
        GlslDispatchN(VulkanExtendedConvKernels.TrilinearInterpolateBackward, total,
            [gradOutput, positions, gradGrid],
            [(uint)d, (uint)h, (uint)w, (uint)c, (uint)p, FloatBits(upperEps)]);
    }
}
