// Copyright (c) AiDotNet. All rights reserved.
// HIP backend - sparse optimizer scatter-update stubs (PR #567).
//
// Sparse optimizer kernels are CUDA-only in this PR — once equivalent HIP
// rocBLAS / Composable Kernels are wired (mirroring the existing dense
// AdamUpdate etc.), each method below should dispatch the matching HIP
// kernel. Until then, callers (GpuOptimizer.TryXStepSparse) catch
// NotSupportedException and fall back to the CPU sparse path.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend
{
    private static NotSupportedException SparseNotImpl(string op) =>
        new NotSupportedException(
            $"HIP backend does not yet ship a native sparse '{op}' kernel; " +
            "caller should fall back to the CPU sparse optimizer path.");

    public void SparseSgdUpdate(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float weightDecay)
        => throw SparseNotImpl("sgd_update");
    public void SparseSgdMomentumUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
        => throw SparseNotImpl("sgd_momentum_update");
    public void SparseAdamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => throw SparseNotImpl("adam_update");
    public void SparseAdamWUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => throw SparseNotImpl("adamw_update");
    public void SparseRmspropUpdate(IGpuBuffer param, IGpuBuffer squaredAvg, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float rho, float epsilon, float weightDecay)
        => throw SparseNotImpl("rmsprop_update");
    public void SparseAdagradUpdate(IGpuBuffer param, IGpuBuffer accumulatedGrad, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float epsilon, float weightDecay)
        => throw SparseNotImpl("adagrad_update");
    public void SparseNagUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
        => throw SparseNotImpl("nag_update");
    public void SparseAdadeltaUpdate(IGpuBuffer param, IGpuBuffer accumGrad, IGpuBuffer accumUpdate, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float rho, float epsilon, float weightDecay)
        => throw SparseNotImpl("adadelta_update");
    public void SparseAmsgradUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => throw SparseNotImpl("amsgrad_update");
    public void SparseAdamaxUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer u, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => throw SparseNotImpl("adamax_update");
    public void SparseLionUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float weightDecay)
        => throw SparseNotImpl("lion_update");
    public void SparseNadamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => throw SparseNotImpl("nadam_update");
    public void SparseFtrlUpdate(IGpuBuffer param, IGpuBuffer z, IGpuBuffer n, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Reg, float l2Reg, float beta)
        => throw SparseNotImpl("ftrl_update");
    public void SparseProximalL1Update(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Strength)
        => throw SparseNotImpl("proximal_l1_update");
}
