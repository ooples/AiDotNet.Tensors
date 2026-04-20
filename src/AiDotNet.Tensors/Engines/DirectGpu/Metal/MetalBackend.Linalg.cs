// Copyright (c) AiDotNet. All rights reserved.
// Metal dispatchers for the torch.linalg decomposition kernels (#211 moat #2).

using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : ILinalgBackend
{
    private const string LinalgLibName = "Linalg";

    private MetalPipelineState GetLinalgPipeline(string kernelName)
    {
        if (_linalgLibrary == IntPtr.Zero)
            throw new InvalidOperationException(
                "Metal Linalg library was not compiled. Falling back to CPU reference.");
        return GetPipeline(LinalgLibName, _linalgLibrary, kernelName);
    }

    /// <summary>
    /// Pick threadgroup size that accommodates the matrix dimension. Metal
    /// caps at 1024 threads per threadgroup; for n &gt; 1024 the linalg path
    /// falls back to CPU via <see cref="DirectGpuTensorEngine"/>.
    /// </summary>
    private static uint LinalgThreadsPerGroup(int n)
    {
        int v = 1; while (v < n) v <<= 1;
        return (uint)Math.Min(1024, Math.Max(32, v));
    }

    public void LinalgCholesky(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer info,
        int batchCount, int n, bool upper)
    {
        ThrowIfDisposed();
        if (batchCount <= 0 || n <= 0) return;
        if (input is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer outBuf || info is not MetalGpuBuffer infoBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        var pipeline = GetLinalgPipeline("parity211_cholesky");
        uint tpg = LinalgThreadsPerGroup(n);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBuffer(infoBuf, 2);
        encoder.SetBytes(batchCount, 3);
        encoder.SetBytes(n, 4);
        encoder.SetBytes(upper ? 1 : 0, 5);
        encoder.DispatchThreadgroups(
            new MTLSize((uint)batchCount, 1, 1),
            new MTLSize(tpg, 1, 1));
    }

    public void LinalgLuFactor(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer pivots,
        int batchCount, int m, int n)
    {
        ThrowIfDisposed();
        if (batchCount <= 0 || m <= 0 || n <= 0) return;
        if (input is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer outBuf || pivots is not MetalGpuBuffer pivBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        var pipeline = GetLinalgPipeline("parity211_lu_factor");
        uint tpg = LinalgThreadsPerGroup(Math.Max(m, n));
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBuffer(pivBuf, 2);
        encoder.SetBytes(batchCount, 3);
        encoder.SetBytes(m, 4);
        encoder.SetBytes(n, 5);
        encoder.DispatchThreadgroups(
            new MTLSize((uint)batchCount, 1, 1),
            new MTLSize(tpg, 1, 1));
    }

    public void LinalgQrReduced(
        IGpuBuffer input, IGpuBuffer q, IGpuBuffer r,
        int batchCount, int m, int n)
    {
        ThrowIfDisposed();
        if (batchCount <= 0 || m <= 0 || n <= 0) return;
        if (input is not MetalGpuBuffer inBuf || q is not MetalGpuBuffer qBuf || r is not MetalGpuBuffer rBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        var pipeline = GetLinalgPipeline("parity211_qr_reduced");
        uint tpg = LinalgThreadsPerGroup(Math.Max(m, n));
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(qBuf, 1);
        encoder.SetBuffer(rBuf, 2);
        encoder.SetBytes(batchCount, 3);
        encoder.SetBytes(m, 4);
        encoder.SetBytes(n, 5);
        encoder.DispatchThreadgroups(
            new MTLSize((uint)batchCount, 1, 1),
            new MTLSize(tpg, 1, 1));
    }

    public void LinalgEigh(
        IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
        int batchCount, int n)
    {
        ThrowIfDisposed();
        if (batchCount <= 0 || n <= 0) return;
        if (input is not MetalGpuBuffer inBuf || eigenvalues is not MetalGpuBuffer wBuf || eigenvectors is not MetalGpuBuffer vBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        var pipeline = GetLinalgPipeline("parity211_eigh");
        uint tpg = LinalgThreadsPerGroup(n);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(wBuf, 1);
        encoder.SetBuffer(vBuf, 2);
        encoder.SetBytes(batchCount, 3);
        encoder.SetBytes(n, 4);
        // Threadgroup memory: 2 · n · n floats (working A + V).
        encoder.SetThreadgroupMemoryLength((uint)(2 * n * n * sizeof(float)), 0);
        encoder.DispatchThreadgroups(
            new MTLSize((uint)batchCount, 1, 1),
            new MTLSize(tpg, 1, 1));
    }
}
