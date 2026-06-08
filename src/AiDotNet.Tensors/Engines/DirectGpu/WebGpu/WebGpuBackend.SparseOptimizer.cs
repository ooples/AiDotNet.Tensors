// Copyright (c) AiDotNet. All rights reserved.
// WebGpu backend - sparse optimizer scatter-update stubs (PR #567).

#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    private static NotSupportedException SparseWebGpuNotImpl(string op) =>
        new NotSupportedException(
            $"WebGpu backend does not yet ship a native sparse '{op}' kernel; " +
            "caller should fall back to the CPU sparse optimizer path.");

    public void SparseSgdUpdate(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float weightDecay)
        => throw SparseWebGpuNotImpl("sgd_update");
    public void SparseSgdMomentumUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
        => throw SparseWebGpuNotImpl("sgd_momentum_update");
    public void SparseAdamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => throw SparseWebGpuNotImpl("adam_update");
    public void SparseAdamWUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => throw SparseWebGpuNotImpl("adamw_update");
    public void SparseRmspropUpdate(IGpuBuffer param, IGpuBuffer squaredAvg, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float rho, float epsilon, float weightDecay)
        => throw SparseWebGpuNotImpl("rmsprop_update");
    public void SparseAdagradUpdate(IGpuBuffer param, IGpuBuffer accumulatedGrad, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float epsilon, float weightDecay)
        => throw SparseWebGpuNotImpl("adagrad_update");
    public void SparseNagUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
        => throw SparseWebGpuNotImpl("nag_update");
    public void SparseAdadeltaUpdate(IGpuBuffer param, IGpuBuffer accumGrad, IGpuBuffer accumUpdate, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float rho, float epsilon, float weightDecay)
        => throw SparseWebGpuNotImpl("adadelta_update");
    public void SparseAmsgradUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => throw SparseWebGpuNotImpl("amsgrad_update");
    public void SparseAdamaxUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer u, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => throw SparseWebGpuNotImpl("adamax_update");
    public void SparseLionUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float weightDecay)
        => throw SparseWebGpuNotImpl("lion_update");
    public void SparseNadamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => throw SparseWebGpuNotImpl("nadam_update");
    public void SparseFtrlUpdate(IGpuBuffer param, IGpuBuffer z, IGpuBuffer n, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Reg, float l2Reg, float beta)
        => throw SparseWebGpuNotImpl("ftrl_update");
    public void SparseProximalL1Update(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Strength)
        => throw SparseWebGpuNotImpl("proximal_l1_update");
}
#endif
