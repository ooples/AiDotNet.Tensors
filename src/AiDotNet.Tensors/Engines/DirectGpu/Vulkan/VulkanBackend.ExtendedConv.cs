namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #775: Vulkan implementations of the per-family extended-conv capability interfaces. Added
// incrementally (interface + GLSL kernels) so a partially-ported backend opts into exactly the families
// it can run; the engine routes the rest to the CPU. GLSL kernels live in VulkanExtendedConvKernels;
// this partial only wires the dispatch via GlslDispatchN (int params -> push constants; a float param is
// passed as its raw bit pattern via the shared FloatBits helper and declared `float` in the push block).
public sealed partial class VulkanBackend : ITrilinearInterpolationKernels, IConvTranspose3DKernels
{
    private static uint[] Conv3DPush(int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        [(uint)n, (uint)inC, (uint)iD, (uint)iH, (uint)iW, (uint)outC, (uint)outD, (uint)outH, (uint)outW,
         (uint)kD, (uint)kH, (uint)kW, (uint)strideD, (uint)strideH, (uint)strideW, (uint)padD, (uint)padH, (uint)padW];

    public void ConvTranspose3D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        int total = checked(n * outC * outD * outH * outW);
        if (total <= 0) return;
        GlslDispatchN(VulkanExtendedConvKernels.ConvTranspose3D, total, [input, weights, output],
            Conv3DPush(n, inC, iD, iH, iW, outC, outD, outH, outW, kD, kH, kW, strideD, strideH, strideW, padD, padH, padW));
    }

    public void ConvTranspose3DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        int total = checked(n * inC * iD * iH * iW);
        if (total <= 0) return;
        GlslDispatchN(VulkanExtendedConvKernels.ConvTranspose3DBackwardInput, total, [gradOutput, weights, gradInput],
            Conv3DPush(n, inC, iD, iH, iW, outC, outD, outH, outW, kD, kH, kW, strideD, strideH, strideW, padD, padH, padW));
    }

    public void ConvTranspose3DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        int total = checked(inC * outC * kD * kH * kW);
        if (total <= 0) return;
        GlslDispatchN(VulkanExtendedConvKernels.ConvTranspose3DBackwardWeights, total, [gradOutput, input, gradWeights],
            Conv3DPush(n, inC, iD, iH, iW, outC, outD, outH, outW, kD, kH, kW, strideD, strideH, strideW, padD, padH, padW));
    }

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
