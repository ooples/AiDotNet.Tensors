// Copyright (c) AiDotNet. All rights reserved.
// HIP kernel source — mirror of CudaLinalgKernels. HIP's C++ dialect compiles
// CUDA-style `__global__` kernels directly through hipRTC, so the kernel bodies
// are reused verbatim apart from the shared-memory declaration syntax.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels
{
    internal static class HipLinalgKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "parity211_cholesky",
            "parity211_lu_factor",
            "parity211_qr_reduced",
            "parity211_eigh",
        };

        // Reuse the CUDA source — hipRTC accepts the same program text when it
        // only touches __global__ / __syncthreads / __shared__ / atomicExch.
        public static string GetSource() => CUDA.Kernels.CudaLinalgKernels.GetSource();
    }
}
