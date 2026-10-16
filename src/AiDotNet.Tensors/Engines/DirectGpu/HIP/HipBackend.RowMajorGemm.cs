using System;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend
{
    private IntPtr _rowMajorGemmModule;
    private bool _rowMajorGemmTried;
    private IGpuBuffer? _rowMajorSplitKWorkspace;

    /// <summary>Compiles the row-major SGEMM tiles (#377) once. Returns whether they are usable.</summary>
    internal bool EnsureRowMajorGemm()
    {
        lock (_clblastLock)
        {
            if (_rowMajorGemmTried) return _rowMajorGemmModule != IntPtr.Zero;
            _rowMajorGemmTried = true;
            if (!IsAvailable) return false;
            IntPtr module = IntPtr.Zero;
            if (CompileKernelModule(HipRowMajorSgemmKernels.BuildSource(), "rowmajor_sgemm", ref module,
                    HipRowMajorSgemmKernels.KernelNames(), useFastMath: false) is not null)
            {
                if (module != IntPtr.Zero) HipNativeBindings.hipModuleUnload(module);
                return false;
            }
            _rowMajorGemmModule = module;
            return true;
        }
    }

    /// <summary>
    /// Row-major C[M,N] = alpha * A[M,K] * B[K,N] + beta * C with <paramref name="tile"/>'s kernel, no packing. With
    /// <paramref name="splitK"/> &gt; 1 the K range is cut into that many 16-aligned pieces computed by separate blocks into
    /// a workspace and summed in a fixed order (deterministic), which fills the device on shapes with few output tiles.
    /// Synchronous, like <see cref="Gemm"/>.
    /// </summary>
    internal unsafe bool TryExecuteRowMajorGemm(HipRowMajorSgemmTile tile, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, float alpha, float beta, int splitK = 1)
    {
        if (!tile.IsValid || !tile.Supports(M, N, K) || splitK < 1 || !EnsureRowMajorGemm()) return false;
        if (!_kernelCache.TryGetValue(tile.KernelName, out IntPtr kernel)) return false;
        int kChunk = splitK == 1 ? K : ((K + splitK - 1) / splitK + HipRowMajorSgemmTile.BlockK - 1) / HipRowMajorSgemmTile.BlockK * HipRowMajorSgemmTile.BlockK;
        int splits = (K + kChunk - 1) / kChunk;
        long count = (long)M * N;
        uint gridX = (uint)((N + tile.BlockN - 1) / tile.BlockN), gridY = (uint)((M + tile.BlockM - 1) / tile.BlockM);
        IntPtr pa = A.Handle, pb = B.Handle;
        void** args = stackalloc void*[10];
        args[0] = &pa; args[1] = &pb; args[3] = &M; args[4] = &N; args[5] = &K; args[8] = &kChunk;
        if (splits == 1)
        {
            IntPtr pc = C.Handle;
            ulong cStride = 0;
            args[2] = &pc; args[6] = &alpha; args[7] = &beta; args[9] = &cStride;
            LaunchKernel3D(kernel, gridX, gridY, 1, HipRowMajorSgemmTile.Threads, 1, 1, args);
        }
        else
        {
            if (count * splits > int.MaxValue || !_kernelCache.TryGetValue(HipRowMajorSgemmKernels.SplitKReduceName, out IntPtr reduce))
                return false;
            lock (_clblastLock)
            {
                var workspace = ClBlastScratch(ref _rowMajorSplitKWorkspace, count * splits);
                IntPtr pw = workspace.Handle;
                float one = 1f, zero = 0f;
                ulong cStride = (ulong)count;
                args[2] = &pw; args[6] = &one; args[7] = &zero; args[9] = &cStride;
                LaunchKernel3D(kernel, gridX, gridY, (uint)splits, HipRowMajorSgemmTile.Threads, 1, 1, args);
                IntPtr pc = C.Handle;
                int countI = (int)count;
                void** rargs = stackalloc void*[6];
                rargs[0] = &pw; rargs[1] = &pc; rargs[2] = &countI; rargs[3] = &splits; rargs[4] = &alpha; rargs[5] = &beta;
                LaunchKernel(reduce, (uint)((countI + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, rargs);
            }
        }
        HipNativeBindings.CheckError(HipNativeBindings.hipStreamSynchronize(_stream), "hipStreamSynchronize (row-major SGEMM)");
        Interlocked.Increment(ref _rowMajorGemmRuns);
        return true;
    }

    private long _rowMajorGemmRuns;

    /// <summary>How many GEMMs the row-major kernel has run (lets tests prove the routing).</summary>
    internal long RowMajorGemmRuns => Interlocked.Read(ref _rowMajorGemmRuns);

    /// <summary>
    /// Whether the routing table below was measured on this device's architecture: RDNA1 (gfx1010 / gfx1012). Other
    /// architectures keep the vendor-then-CLBlast order until they are measured.
    /// </summary>
    /// <summary>
    /// Whether Gemm consults the row-major routing table at all (default true). Tests of the CLBlast indirect and
    /// vendor routes turn it off so they keep exercising those routes on shapes the table now sends elsewhere.
    /// </summary>
    internal bool UseRowMajorGemmTable { get; set; } = true;

    internal bool RowMajorGemmTableApplies =>
        _architectureTarget.StartsWith("gfx1010", StringComparison.OrdinalIgnoreCase) ||
        _architectureTarget.StartsWith("gfx1012", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// The row-major tile and split-K for shape (M, N, K), or null where hipBLAS (when loaded) or the CLBlast indirect
    /// kernel measured faster. Measured on an RX 5500 XT (gfx1012), best row-major time over hipBLAS's (lower is ours
    /// faster): 128^3 0.84, 256^3 0.57, 512^3 0.76, 256x256x2304 0.37, 1024x1024x4096 0.69, 200x500x300 0.82,
    /// 32x1024x1024 0.89; hipBLAS wins 1024^3 (1.48), 2048^3 (1.43), 1024x4096x1024 (1.47) and the skinny
    /// 64x4096x1024 / 4096x64x1024 (1.16 / 1.15). Against the CLBlast indirect kernel (no hipBLAS), the row-major
    /// kernel wins the skinny shapes (0.335 vs 0.351 ms, 0.459 vs 0.875 ms) and loses the large balanced ones
    /// (2048^3 6.66 vs 6.04 ms).
    /// </summary>
    internal (HipRowMajorSgemmTile Tile, int SplitK)? ChooseRowMajorGemm(int M, int N, int K, bool vendorAvailable)
        => ChooseRowMajorGemm(M, N, K, () => vendorAvailable);

    /// <summary>
    /// <see cref="ChooseRowMajorGemm(int, int, int, bool)"/> with the vendor check deferred: hipBLAS is loaded (its
    /// Tensile libraries cost hundreds of milliseconds) only for the skinny shapes whose route depends on it.
    /// </summary>
    internal (HipRowMajorSgemmTile Tile, int SplitK)? ChooseRowMajorGemm(int M, int N, int K, Func<bool> vendorAvailable)
    {
        if (M <= 0 || N <= 0 || K <= 0) return null;
        long work = (long)M * N * K;
        bool small = work <= 512L * 512 * 512;
        bool deepK = K >= 2L * Math.Max(M, N);
        bool skinny = (M <= 64 && N >= 512) || (N <= 64 && M >= 512);
        bool take = small || deepK || (M <= 32 && N >= 512) || (skinny && !vendorAvailable());
        if (!take) return null;

        bool vec = K % 4 == 0 && N % 4 == 0;
        HipRowMajorSgemmTile tile = !vec
            ? new HipRowMajorSgemmTile(64, 64, 4, 4)
            : M <= 64 || N < 128 ? new HipRowMajorSgemmTile(64, 64, 4, 4, true) : new HipRowMajorSgemmTile(64, 128, 4, 8, true);
        // Split K until the device has about two blocks per compute unit, keeping each piece at least 128 deep.
        long blocks = (long)((M + tile.BlockM - 1) / tile.BlockM) * ((N + tile.BlockN - 1) / tile.BlockN);
        int target = Math.Max(1, 2 * _deviceProps.MultiProcessorCount);
        int split = 1;
        while (split < 4 && blocks * split < target && K / (split * 2) >= 128) split *= 2;
        return (tile, split);
    }

    private void DisposeRowMajorGemm()
    {
        lock (_clblastLock)
        {
            if (_rowMajorGemmModule != IntPtr.Zero) { HipNativeBindings.hipModuleUnload(_rowMajorGemmModule); _rowMajorGemmModule = IntPtr.Zero; }
            _rowMajorSplitKWorkspace?.Dispose(); _rowMajorSplitKWorkspace = null;
        }
    }
}
