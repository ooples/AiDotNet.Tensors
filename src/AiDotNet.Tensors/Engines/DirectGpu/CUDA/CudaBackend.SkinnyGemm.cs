using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Row-major FP32 GEMMs for narrow products (see CudaSkinnyGemmKernels): Gemm, MatMulTransposed and MatMulTransposedA
// try these before cuBLAS. Strict fp32 and deterministic (fixed summation order, no atomics on values), so they serve
// GpuDeterminism as well as the default mode.
public sealed partial class CudaBackend : IGemmBiasActivationBackend
{
    // AIDOTNET_SKINNY_GEMM=0: every GEMM goes to cuBLAS as before.
    private static readonly bool SkinnyGemmEnabled =
        Environment.GetEnvironmentVariable("AIDOTNET_SKINNY_GEMM") != "0";

    private const int SkinnyTileRows = 32, SkinnyTileColumns = 64, SkinnyBlockThreads = 64;
    // skinny_gemm_tn: four 64-thread groups per block, each reducing 16-row chunks of the block's split.
    private const int SkinnyTransposedABlockThreads = 256, SkinnyTransposedAPassRows = 64;

    // The narrow dimensions these kernels take. Past them a product has enough output tiles for cuBLAS's own tiling.
    private const int SkinnyMaxNarrow = 256;

    // Per-thread arrival counters of skinny_gemm_tn, one per output tile: (256/32) * (256/64) tiles at most.
    private const int SkinnyCounterSlots = 64;

    /// <summary>
    /// <c>C[M,N] = alpha · A[M,K] · op(B) + beta · C</c> (op(B) = <c>B[K,N]</c>, or <c>B[N,K]ᵀ</c> when
    /// <paramref name="transB"/>) through skinny_gemm, for a tall A (M ≥ 256) and N, K ≤ 256. False -- nothing
    /// launched -- for any other shape, an unaligned operand or a missing kernel; the caller then runs cuBLAS.
    /// </summary>
    internal unsafe bool TrySkinnyGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, bool transB,
        float alpha, float beta, IGpuBuffer? bias = null, bool relu = false)
    {
        if (bias is not null && (bias.Size < N || (bias.Handle.ToInt64() & 15) != 0)) return false;
        if (!SkinnyGemmEnabled || M < 256 || M > 65535 * SkinnyTileRows || N > SkinnyMaxNarrow || K > SkinnyMaxNarrow
            || (N & 3) != 0 || (K & 3) != 0 || !SkinnyAligned(A, B, C)
            || !_kernelCache.TryGetValue("skinny_gemm", out var kernel))
            return false;
        using var _ = PushContext();
        IntPtr a = A.Handle, b = B.Handle, c = C.Handle;
        int m = M, n = N, k = K, tb = transB ? 1 : 0, re = relu ? 1 : 0;
        float al = alpha, be = beta;
        IntPtr bi = bias?.Handle ?? IntPtr.Zero;
        void** args = stackalloc void*[11];
        args[0] = &a; args[1] = &b; args[2] = &c; args[3] = &m; args[4] = &n; args[5] = &k; args[6] = &tb;
        args[7] = &al; args[8] = &be; args[9] = &bi; args[10] = &re;
        LaunchKernel2D(kernel, (uint)((N + SkinnyTileColumns - 1) / SkinnyTileColumns),
            (uint)((M + SkinnyTileRows - 1) / SkinnyTileRows), 1, SkinnyBlockThreads, 1, args);
        return true;
    }

    /// <summary>
    /// <c>C[M,N] = alpha · A[K,M]ᵀ · B[K,N] + beta · C</c> through skinny_gemm_tn, for a long reduction (K ≥ 512)
    /// into a small output (M, N ≤ 256): a weight gradient. Its split partials go in the calling thread's cuBLAS
    /// workspace, which cuBLAS and cuBLASLt use only in this stream's order. False -- nothing launched -- when the
    /// shape, alignment, workspace or kernel does not qualify; the caller then runs cuBLAS.
    /// </summary>
    // wgrad_tile_tn takes reductions from this many rows up to skinny_gemm_tn's (512).
    private const int WgradTileMinReduce = 64;

    /// <summary>
    /// <c>C[M,N] = alpha * A[K,M]ᵀ * B[K,N] + beta * C</c> through wgrad_tile_tn for a reduction of 64..511 rows with
    /// M, N &lt;= 256 (multiples of 4, 16-byte aligned): one launch where cuBLASLt ran a GEMM and a split-K reduce.
    /// False -- nothing launched -- for any other shape.
    /// </summary>
    internal unsafe bool TryWgradTile(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha, float beta)
    {
        if (!SkinnyGemmEnabled || K < WgradTileMinReduce || K >= 512 || M <= 0 || N <= 0
            || M > SkinnyMaxNarrow || N > SkinnyMaxNarrow || (M & 3) != 0 || (N & 3) != 0 || !SkinnyAligned(A, B, C)
            || !_kernelCache.TryGetValue("wgrad_tile_tn", out var wgradKernel))
            return false;
        using var _ = PushContext();
        IntPtr a = A.Handle, b = B.Handle, c = C.Handle;
        int m = M, n = N, k = K;
        float al = alpha, be = beta;
        void** args = stackalloc void*[8];
        args[0] = &a; args[1] = &b; args[2] = &c; args[3] = &m; args[4] = &n; args[5] = &k; args[6] = &al; args[7] = &be;
        LaunchKernel2D(wgradKernel, (uint)((N + 15) / 16), (uint)((M + 15) / 16), 1, 256, 1, args);
        return true;
    }

    internal unsafe bool TrySkinnyGemmTransposedA(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha, float beta)
    {
        if (K < 512) return TryWgradTile(A, B, C, M, N, K, alpha, beta);
        if (!SkinnyGemmEnabled || K < 512 || M > SkinnyMaxNarrow || N > SkinnyMaxNarrow
            || (M & 3) != 0 || (N & 3) != 0 || !SkinnyAligned(A, B, C)
            || !_kernelCache.TryGetValue("skinny_gemm_tn", out var kernel))
            return false;
        if (!CurrentCublasHasWorkspace || _threadCublas?.Value is not { Workspace: { } workspace, SkinnyCounters: { } counters })
            return false;
        int tilesN = (N + SkinnyTileColumns - 1) / SkinnyTileColumns;
        int tilesM = (M + SkinnyTileRows - 1) / SkinnyTileRows;
        int tiles = tilesM * tilesN;
        if (tiles > SkinnyCounterSlots) return false;
        int rowsPerSplit = SkinnyRowsPerSplit(K, tiles);
        int splits = (K + rowsPerSplit - 1) / rowsPerSplit;
        if ((long)splits * tiles * SkinnyTileRows * SkinnyTileColumns > CublasWorkspaceFloats) return false;

        using var _ = PushContext();
        IntPtr a = A.Handle, b = B.Handle, c = C.Handle, p = workspace.Handle, cnt = counters.Handle;
        int m = M, n = N, k = K, rps = rowsPerSplit;
        float al = alpha, be = beta;
        void** args = stackalloc void*[11];
        args[0] = &a; args[1] = &b; args[2] = &c; args[3] = &p; args[4] = &cnt; args[5] = &m; args[6] = &n;
        args[7] = &k; args[8] = &rps; args[9] = &al; args[10] = &be;
        LaunchKernel2D(kernel, (uint)tilesN, (uint)tilesM, (uint)splits, SkinnyTransposedABlockThreads, 1, args);
        return true;
    }

    /// <summary>
    /// A linear layer's forward in one kernel: <c>C = act(A[M,K] · W[K,N] + bias)</c> with act ReLU or none, the bias and
    /// activation applied in skinny_gemm's epilogue (the GEMM, a bias-add pass and an activation pass otherwise). Equal
    /// bit for bit to the GEMM followed by BiasAdd and Relu. False -- nothing launched -- for another activation or a
    /// shape skinny_gemm does not take.
    /// </summary>
    bool IGemmBiasActivationBackend.TryGemmBiasActivation(IGpuBuffer A, IGpuBuffer W, IGpuBuffer bias, IGpuBuffer C,
        int M, int N, int K, FusedActivationType activation)
        => (activation == FusedActivationType.None || activation == FusedActivationType.ReLU)
            && TrySkinnyGemm(A, W, C, M, N, K, transB: false, 1f, 0f, bias, activation == FusedActivationType.ReLU);

    // Rows each split of skinny_gemm_tn reduces: ~256 (four passes per group) and at most 64 blocks in all, a whole
    // number of the block's 64-row passes. Measured on a GTX 1660 Ti at K = 2048 for 64x64, 64x128 and 128x64 outputs,
    // 256-row splits beat 128 and 512: shorter splits multiply the partials the last block sums, longer ones leave SMs idle.
    internal static int SkinnyRowsPerSplit(int K, int tiles)
    {
        int splits = Math.Max(1, Math.Min(K / 256, 64 / Math.Max(1, tiles)));
        int rows = (K + splits - 1) / splits;
        return (rows + SkinnyTransposedAPassRows - 1) / SkinnyTransposedAPassRows * SkinnyTransposedAPassRows;
    }

    /// <summary>
    /// The compiled resource use of a loaded kernel -- registers per thread, local (spill) bytes per thread, and how
    /// many blocks of <paramref name="blockThreads"/> threads one SM holds at once -- or null when no kernel of that
    /// name is loaded. A kernel whose accumulators spilled to local memory, or that fits one block per SM, runs
    /// several times slower with no other symptom.
    /// </summary>
    internal (int Registers, int LocalBytes, int BlocksPerSm)? KernelResourceUsage(string kernelName,
        int blockThreads = SkinnyBlockThreads)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel)) return null;
        using var _ = PushContext();
        if (CudaNativeBindings.cuFuncGetAttribute(out int registers, CudaFunctionAttribute.NumRegisters, kernel)
                != CudaResult.Success
            || CudaNativeBindings.cuFuncGetAttribute(out int local, CudaFunctionAttribute.LocalSizeBytes, kernel)
                != CudaResult.Success
            || CudaNativeBindings.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(out int blocks, kernel, blockThreads, 0, 0)
                != CudaResult.Success)
            return null;
        return (registers, local, blocks);
    }

    private static bool SkinnyAligned(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C) =>
        ((A.Handle.ToInt64() | B.Handle.ToInt64() | C.Handle.ToInt64()) & 15) == 0;
}
