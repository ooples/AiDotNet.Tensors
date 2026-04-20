// Copyright (c) AiDotNet. All rights reserved.
// HIP dispatchers for the torch.linalg decomposition kernels (#211 moat #2).
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : ILinalgBackend
{
    private IntPtr ResolveLinalgKernel(string name)
    {
        if (_linalgModule == IntPtr.Zero)
            throw new InvalidOperationException(
                "Linalg HIP module was not compiled (hipRTC rejected source?). Falling back to CPU reference.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"HIP kernel not found: {name}");
        return kernel;
    }

    private static uint LinalgBlockSize(int n)
    {
        int v = 1; while (v < n) v <<= 1;
        return (uint)Math.Min(1024, Math.Max(32, v));
    }

    public unsafe void LinalgCholesky(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer info,
        int batchCount, int n, bool upper)
    {
        if (batchCount <= 0 || n <= 0) return;
        var kernel = ResolveLinalgKernel("parity211_cholesky");
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle; IntPtr infoPtr = info.Handle;
        int b = batchCount, nn = n, up = upper ? 1 : 0;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &infoPtr;
        args[3] = &b; args[4] = &nn; args[5] = &up;
        LaunchKernel(kernel, (uint)batchCount, LinalgBlockSize(n), args);
        Synchronize();
    }

    public unsafe void LinalgLuFactor(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer pivots,
        int batchCount, int m, int n)
    {
        if (batchCount <= 0 || m <= 0 || n <= 0) return;
        var kernel = ResolveLinalgKernel("parity211_lu_factor");
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle; IntPtr pivPtr = pivots.Handle;
        int b = batchCount, mm = m, nn = n;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &pivPtr;
        args[3] = &b; args[4] = &mm; args[5] = &nn;
        LaunchKernel(kernel, (uint)batchCount, LinalgBlockSize(Math.Max(m, n)), args);
        Synchronize();
    }

    public unsafe void LinalgQrReduced(
        IGpuBuffer input, IGpuBuffer q, IGpuBuffer r,
        int batchCount, int m, int n)
    {
        if (batchCount <= 0 || m <= 0 || n <= 0) return;
        var kernel = ResolveLinalgKernel("parity211_qr_reduced");
        IntPtr inPtr = input.Handle; IntPtr qPtr = q.Handle; IntPtr rPtr = r.Handle;
        int b = batchCount, mm = m, nn = n;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &qPtr; args[2] = &rPtr;
        args[3] = &b; args[4] = &mm; args[5] = &nn;
        LaunchKernel(kernel, (uint)batchCount, LinalgBlockSize(Math.Max(m, n)), args);
        Synchronize();
    }

    public unsafe void LinalgEigh(
        IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
        int batchCount, int n)
    {
        if (batchCount <= 0 || n <= 0) return;
        var kernel = ResolveLinalgKernel("parity211_eigh");
        IntPtr inPtr = input.Handle; IntPtr wPtr = eigenvalues.Handle; IntPtr vPtr = eigenvectors.Handle;
        int b = batchCount, nn = n;
        uint sharedBytes = (uint)(2 * n * n * sizeof(float));
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &wPtr; args[2] = &vPtr;
        args[3] = &b; args[4] = &nn;
        LaunchKernelWithSharedMem(kernel, (uint)batchCount, LinalgBlockSize(n), sharedBytes, args);
        Synchronize();
    }
}
