// Copyright (c) AiDotNet. All rights reserved.
// OpenCL dispatchers for the torch.linalg decomposition kernels (#211 moat #2).
#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : ILinalgBackend
    {
        private DirectOpenClKernel GetLinalgKernel(string name)
        {
            if (!_kernelCache.TryGetValue(name, out var kernel))
                throw new InvalidOperationException(
                    $"OpenCL Linalg kernel not found: {name}. Module may have failed to compile.");
            return kernel;
        }

        private static int LinalgLocal(int n)
        {
            int v = 1; while (v < n) v <<= 1;
            return Math.Min(1024, Math.Max(32, v));
        }

        public void LinalgCholesky(
            IGpuBuffer input, IGpuBuffer output, IGpuBuffer info,
            int batchCount, int n, bool upper)
        {
            if (batchCount <= 0 || n <= 0) return;
            var k = GetLinalgKernel("parity211_cholesky");
            k.SetArg(0, BufHandle(input));
            k.SetArg(1, BufHandle(output));
            k.SetArg(2, BufHandle(info));
            k.SetArg(3, batchCount);
            k.SetArg(4, n);
            k.SetArg(5, upper ? 1 : 0);
            int local = LinalgLocal(n);
            k.Execute1D(batchCount * local, local);
        }

        public void LinalgLuFactor(
            IGpuBuffer input, IGpuBuffer output, IGpuBuffer pivots,
            int batchCount, int m, int n)
        {
            if (batchCount <= 0 || m <= 0 || n <= 0) return;
            var k = GetLinalgKernel("parity211_lu_factor");
            k.SetArg(0, BufHandle(input));
            k.SetArg(1, BufHandle(output));
            k.SetArg(2, BufHandle(pivots));
            k.SetArg(3, batchCount);
            k.SetArg(4, m);
            k.SetArg(5, n);
            int local = LinalgLocal(Math.Max(m, n));
            k.Execute1D(batchCount * local, local);
        }

        public void LinalgQrReduced(
            IGpuBuffer input, IGpuBuffer q, IGpuBuffer r,
            int batchCount, int m, int n)
        {
            if (batchCount <= 0 || m <= 0 || n <= 0) return;
            var k = GetLinalgKernel("parity211_qr_reduced");
            k.SetArg(0, BufHandle(input));
            k.SetArg(1, BufHandle(q));
            k.SetArg(2, BufHandle(r));
            k.SetArg(3, batchCount);
            k.SetArg(4, m);
            k.SetArg(5, n);
            int local = LinalgLocal(Math.Max(m, n));
            k.Execute1D(batchCount * local, local);
        }

        public void LinalgEigh(
            IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
            int batchCount, int n)
        {
            if (batchCount <= 0 || n <= 0) return;
            var k = GetLinalgKernel("parity211_eigh");
            k.SetArg(0, BufHandle(input));
            k.SetArg(1, BufHandle(eigenvalues));
            k.SetArg(2, BufHandle(eigenvectors));
            k.SetArg(3, batchCount);
            k.SetArg(4, n);
            // Local memory: 2·n·n floats (working A + V).
            k.SetLocalArg(5, 2 * n * n * sizeof(float));
            int local = LinalgLocal(n);
            k.Execute1D(batchCount * local, local);
        }
    }
}
#endif
