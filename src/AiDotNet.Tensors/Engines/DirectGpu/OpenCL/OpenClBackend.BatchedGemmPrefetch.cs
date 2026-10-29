// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend
    {
        // The kernel's two __local float[32][65] tiles.
        private const long Rb64PrefetchLocalBytes = 2L * 32 * 65 * sizeof(float);

        /// <summary>
        /// C[b] = A[b] * B[b] through gemm_batched_rb64p (#1432): rb64's 64 x 64 blocks with a 32-deep K step, float4 A
        /// loads and the next tile prefetched into registers. False, with nothing launched, when the kernel did not build
        /// or the device's local memory cannot hold its tiles (an unreported size, zero, is taken as enough, as the
        /// other local-memory checks do); the caller then runs rb64.
        /// </summary>
        internal bool TryBatchedGemmRb64Prefetch(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k, int batchCount)
        {
            ulong localBytes = _context?.LocalMemSize ?? 0;
            if (m <= 0 || n <= 0 || batchCount <= 0
                || (localBytes != 0 && localBytes < (ulong)Rb64PrefetchLocalBytes)
                || !_kernelCache.TryGetValue(Kernels.BatchedGemmKernels.Rb64Prefetch, out var kernel))
                return false;
            kernel.SetArg(0, BufferHandle(a)); kernel.SetArg(1, BufferHandle(b)); kernel.SetArg(2, BufferHandle(c));
            kernel.SetArg(3, m); kernel.SetArg(4, n); kernel.SetArg(5, k);
            kernel.Execute3D(RoundUpToMultiple(n, 64) / 4, RoundUpToMultiple(m, 64) / 4, batchCount, 16, 16, 1);
            return true;
        }
    }
}