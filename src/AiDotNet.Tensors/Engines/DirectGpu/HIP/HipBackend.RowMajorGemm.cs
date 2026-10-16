using System;
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
        return true;
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
