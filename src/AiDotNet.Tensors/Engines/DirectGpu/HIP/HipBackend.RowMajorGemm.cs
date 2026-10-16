using System;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend
{
    private IntPtr _rowMajorGemmModule;
    private bool _rowMajorGemmTried;

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
    /// Row-major C[M,N] = alpha * A[M,K] * B[K,N] + beta * C with <paramref name="tile"/>'s kernel, no packing.
    /// Synchronous, like <see cref="Gemm"/>.
    /// </summary>
    internal unsafe bool TryExecuteRowMajorGemm(HipRowMajorSgemmTile tile, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, float alpha, float beta)
    {
        if (!tile.IsValid || !tile.Supports(M, N, K) || !EnsureRowMajorGemm()) return false;
        if (!_kernelCache.TryGetValue(tile.KernelName, out IntPtr kernel)) return false;
        IntPtr pa = A.Handle, pb = B.Handle, pc = C.Handle;
        void** args = stackalloc void*[8];
        args[0] = &pa; args[1] = &pb; args[2] = &pc; args[3] = &M; args[4] = &N; args[5] = &K; args[6] = &alpha; args[7] = &beta;
        uint gridX = (uint)((N + tile.BlockN - 1) / tile.BlockN), gridY = (uint)((M + tile.BlockM - 1) / tile.BlockM);
        LaunchKernel2D(kernel, gridX, gridY, HipRowMajorSgemmTile.Threads, 1, args);
        HipNativeBindings.CheckError(HipNativeBindings.hipStreamSynchronize(_stream), "hipStreamSynchronize (row-major SGEMM)");
        return true;
    }

    private void DisposeRowMajorGemm()
    {
        lock (_clblastLock)
        {
            if (_rowMajorGemmModule != IntPtr.Zero) { HipNativeBindings.hipModuleUnload(_rowMajorGemmModule); _rowMajorGemmModule = IntPtr.Zero; }
        }
    }
}
