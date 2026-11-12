// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Capability: the jagged-rows grouped GEMM behind <see cref="CpuEngine.TensorGroupedMatMul{T}"/> (torch grouped_mm with a
/// 2-D a and a 3-D b, the expert GEMMs of a mixture-of-experts layer) as device work, each group at its own row count.
/// Rows <c>[ends[g-1], ends[g])</c> of a belong to group g; rows past the last offset produce zero.
/// </summary>
/// <remarks>
/// CUDA and HIP run one vendor-BLAS GEMM per non-empty group on buffer views (free at any offset there,
/// <see cref="ViewGroupedGemm"/>). OpenCL, Vulkan, Metal and WebGPU run <see cref="GroupedGemmKernelTemplate"/>'s kernels
/// (one launch over every group, <see cref="PortableGroupedGemm"/>). A false return launches nothing and the caller runs
/// the host op.
/// </remarks>
internal interface IGroupedGemmBackend
{
    /// <summary>
    /// <c>y[r, :] = a[r, :] op(b[g(r)]) + bias</c> over a [M, K] and b [G, K, N] (or [G, N, K] with
    /// <paramref name="transposeB"/>): every element of y [M, N] is written. <paramref name="bias"/> is [N], or [G, N]
    /// with <paramref name="biasPerGroup"/>; rows past the last offset get no bias.
    /// </summary>
    bool TryGroupedRowsGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer y, int[] ends, int m, int n, int k, bool transposeB,
        IGpuBuffer? bias, bool biasPerGroup);

    /// <summary><c>w[g] = a_g^T c_g</c> over a [M, K] and c [M, N]: every element of w [G, K, N] is written (an empty
    /// group's block as zero).</summary>
    bool TryGroupedRowsWeightGrad(IGpuBuffer a, IGpuBuffer c, IGpuBuffer w, int[] ends, int m, int k, int n);
}
