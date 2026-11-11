// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// The jagged-rows grouped GEMM (IGroupedGemmBackend): one vendor-BLAS GEMM per non-empty group on buffer views, which
// this backend creates at any offset (ViewGroupedGemm).
public sealed partial class HipBackend : IGroupedGemmBackend
{
    bool IGroupedGemmBackend.TryGroupedRowsGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer y, int[] ends, int m, int n, int k,
        bool transposeB, IGpuBuffer? bias, bool biasPerGroup)
        => ViewGroupedGemm.TryForward(this, a, b, y, ends, m, n, k, transposeB, bias, biasPerGroup);

    bool IGroupedGemmBackend.TryGroupedRowsWeightGrad(IGpuBuffer a, IGpuBuffer c, IGpuBuffer w, int[] ends, int m, int k, int n)
        => ViewGroupedGemm.TryWeightGrad(this, a, c, w, ends, m, k, n);
}
