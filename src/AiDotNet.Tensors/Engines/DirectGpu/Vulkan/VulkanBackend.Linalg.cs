// Copyright (c) AiDotNet. All rights reserved.
// Vulkan dispatchers for the torch.linalg decomposition kernels (#211 moat #2).
// Reuses the GLSL compute-pipeline machinery already in place for Parity-210.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : ILinalgBackend
{
    public void LinalgCholesky(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer info,
        int batchCount, int n, bool upper)
    {
        if (batchCount <= 0 || n <= 0) return;
        // Workgroup size is 256 (compile-time in the GLSL); we dispatch one
        // workgroup per batch, so total threads = batchCount * 256.
        uint[] pc = { (uint)batchCount, (uint)n, (uint)(upper ? 1 : 0) };
        GlslBinaryOp3(VulkanLinalgKernels.Cholesky, input, output, info,
            dispatchSize: batchCount * 256, pc, sizeof(uint) * 3u);
    }

    public void LinalgLuFactor(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer pivots,
        int batchCount, int m, int n)
    {
        if (batchCount <= 0 || m <= 0 || n <= 0) return;
        uint[] pc = { (uint)batchCount, (uint)m, (uint)n };
        GlslBinaryOp3(VulkanLinalgKernels.LuFactor, input, output, pivots,
            dispatchSize: batchCount * 256, pc, sizeof(uint) * 3u);
    }

    public void LinalgQrReduced(
        IGpuBuffer input, IGpuBuffer q, IGpuBuffer r,
        int batchCount, int m, int n)
    {
        if (batchCount <= 0 || m <= 0 || n <= 0) return;
        uint[] pc = { (uint)batchCount, (uint)m, (uint)n };
        GlslBinaryOp3(VulkanLinalgKernels.QrReduced, input, q, r,
            dispatchSize: batchCount * 256, pc, sizeof(uint) * 3u);
    }

    public void LinalgEigh(
        IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
        int batchCount, int n)
    {
        if (batchCount <= 0 || n <= 0) return;
        uint[] pc = { (uint)batchCount, (uint)n };
        GlslBinaryOp3(VulkanLinalgKernels.Eigh, input, eigenvalues, eigenvectors,
            dispatchSize: batchCount * 256, pc, sizeof(uint) * 2u);
    }

    /// <summary>
    /// 3-buffer GLSL dispatch — wraps <see cref="GlslBinaryOp"/> but names the
    /// intent (linalg decompositions are "1 input, 2 outputs" rather than
    /// "2 inputs, 1 output" like binary ops). Same pipeline geometry.
    /// </summary>
    private void GlslBinaryOp3(string glslSource, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C,
        int dispatchSize, uint[] pushConstants, uint pushConstantSize)
        => GlslBinaryOp(glslSource, A, B, C, dispatchSize, pushConstants, pushConstantSize);
}
