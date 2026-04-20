// Copyright (c) AiDotNet. All rights reserved.
// CUDA dispatchers for the torch.linalg decomposition kernels (#211 moat #2).

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : ILinalgBackend
{
    private IntPtr ResolveLinalgKernel(string name)
    {
        if (_linalgModule == IntPtr.Zero)
            throw new InvalidOperationException(
                "Linalg CUDA module was not compiled (older toolkit or NVRTC failure). Falling back to CPU reference.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {name}");
        return kernel;
    }

    /// <summary>
    /// Choose a block size that fits the matrix dimension. The kernels spawn
    /// one block per batch slice and use intra-block cooperation, so block size
    /// bounds the max <c>n</c> we can handle on the GPU path (n ≤ 1024).
    /// Larger matrices fall back to the CPU reference.
    /// </summary>
    private static uint LinalgBlockSize(int n) => (uint)Math.Min(1024, Math.Max(32, RoundUpToPow2(n)));

    private static int RoundUpToPow2(int n)
    {
        int v = 1;
        while (v < n) v <<= 1;
        return v;
    }

    public unsafe void LinalgCholesky(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer info,
        int batchCount, int n, bool upper)
    {
        if (batchCount <= 0 || n <= 0) return;
        var kernel = ResolveLinalgKernel("parity211_cholesky");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle; IntPtr infoPtr = info.Handle;
        int b = batchCount, nn = n, up = upper ? 1 : 0;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &infoPtr;
        args[3] = &b; args[4] = &nn; args[5] = &up;
        LaunchKernel(kernel, (uint)batchCount, LinalgBlockSize(n), args);
    }

    public unsafe void LinalgLuFactor(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer pivots,
        int batchCount, int m, int n)
    {
        if (batchCount <= 0 || m <= 0 || n <= 0) return;
        var kernel = ResolveLinalgKernel("parity211_lu_factor");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle; IntPtr pivPtr = pivots.Handle;
        int b = batchCount, mm = m, nn = n;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &pivPtr;
        args[3] = &b; args[4] = &mm; args[5] = &nn;
        LaunchKernel(kernel, (uint)batchCount, LinalgBlockSize(Math.Max(m, n)), args);
    }

    public unsafe void LinalgQrReduced(
        IGpuBuffer input, IGpuBuffer q, IGpuBuffer r,
        int batchCount, int m, int n)
    {
        if (batchCount <= 0 || m <= 0 || n <= 0) return;
        var kernel = ResolveLinalgKernel("parity211_qr_reduced");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle; IntPtr qPtr = q.Handle; IntPtr rPtr = r.Handle;
        int b = batchCount, mm = m, nn = n;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &qPtr; args[2] = &rPtr;
        args[3] = &b; args[4] = &mm; args[5] = &nn;
        LaunchKernel(kernel, (uint)batchCount, LinalgBlockSize(Math.Max(m, n)), args);
    }

    public unsafe void LinalgEigh(
        IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
        int batchCount, int n)
    {
        if (batchCount <= 0 || n <= 0) return;
        // The kernel allocates 2·n·n floats of dynamic shared memory (Ash +
        // Vsh scratch). CUDA caps dynamic shared memory per block at 48 KB
        // by default, which bounds n at ⌊√(48 KB / 2 / 4 B)⌋ ≈ 77. We clamp
        // conservatively to n ≤ 64 so the budget stays safely inside every
        // compute capability. DirectGpuTensorEngine already gates larger
        // values to CPU, but enforce again here so direct callers surface a
        // clear error instead of a silent shared-memory overflow.
        if (n > 64)
            throw new ArgumentOutOfRangeException(nameof(n),
                $"CUDA Eigh kernel supports n ≤ 64 (got n = {n}); route to CPU " +
                "via DirectGpuTensorEngine.TryGpuEigh for larger sizes.");
        var kernel = ResolveLinalgKernel("parity211_eigh");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle; IntPtr wPtr = eigenvalues.Handle; IntPtr vPtr = eigenvectors.Handle;
        int b = batchCount, nn = n;
        uint sharedBytes = (uint)(2 * n * n * sizeof(float));
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &wPtr; args[2] = &vPtr;
        args[3] = &b; args[4] = &nn;
        LaunchKernelWithSharedMem(kernel, (uint)batchCount, LinalgBlockSize(n), sharedBytes, args);
    }
}
