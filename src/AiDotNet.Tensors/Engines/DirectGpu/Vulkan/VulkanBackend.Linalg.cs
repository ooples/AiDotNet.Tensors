// Copyright (c) AiDotNet. All rights reserved.
// Vulkan dispatchers for the torch.linalg decomposition kernels (#211 moat #2).
// Reuses the GLSL compute-pipeline machinery already in place for Parity-210.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : ILinalgBackend
{
    /// <inheritdoc/>
    /// <remarks>The GLSL kernels compile at dispatch through the runtime compiler, so readiness is that compiler.</remarks>
    public bool LinalgKernelsReady => IsGlslCompilerAvailable;

    /// <inheritdoc/>
    /// <remarks>The Eigh kernel provisions 64 x 64 workgroup arrays.</remarks>
    public int LinalgEighMaxN => 64;

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

    /// <summary>Householder, as the CPU and torch.linalg.qr: Gram-Schmidt would force R's diagonal positive (#1142).</summary>
    public void LinalgQrReduced(
        IGpuBuffer input, IGpuBuffer q, IGpuBuffer r,
        int batchCount, int m, int n)
        => PortableExtendedLinalg.QrReduced(this, LaunchSolver, input, q, r, batchCount, m, n);

    public void LinalgEigh(
        IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
        int batchCount, int n)
    {
        if (batchCount <= 0 || n <= 0) return;
        // VulkanLinalgKernels.Eigh declares `shared float Ash[64*64]; shared
        // float Vsh[64*64];` — anything beyond 64 would index out of bounds.
        // The engine-level dispatcher already caps at 64, but enforce again
        // here so a misuse surfaces as an explicit exception rather than an
        // undefined-behavior kernel launch.
        if (n > 64)
            throw new ArgumentOutOfRangeException(nameof(n),
                $"Vulkan Eigh kernel supports n ≤ 64 (got n = {n}). " +
                "Route to CPU via DirectGpuTensorEngine.TryGpuEigh for larger sizes.");
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
