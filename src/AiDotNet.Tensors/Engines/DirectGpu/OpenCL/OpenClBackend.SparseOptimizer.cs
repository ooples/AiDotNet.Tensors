// Copyright (c) AiDotNet. All rights reserved.
// OpenCL backend - sparse optimizer scatter-update stubs (PR #567).

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend
    {
        private static NotSupportedException SparseOpenClNotImpl(string op) =>
            new NotSupportedException(
                $"OpenCL backend does not yet ship a native sparse '{op}' kernel; " +
                "caller should fall back to the CPU sparse optimizer path.");

        public void SparseSgdUpdate(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float weightDecay)
            => throw SparseOpenClNotImpl("sgd_update");
        public void SparseSgdMomentumUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
            => throw SparseOpenClNotImpl("sgd_momentum_update");
        public void SparseAdamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
            => throw SparseOpenClNotImpl("adam_update");
        public void SparseAdamWUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
            => throw SparseOpenClNotImpl("adamw_update");
        public void SparseRmspropUpdate(IGpuBuffer param, IGpuBuffer squaredAvg, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float rho, float epsilon, float weightDecay)
            => throw SparseOpenClNotImpl("rmsprop_update");
        public void SparseAdagradUpdate(IGpuBuffer param, IGpuBuffer accumulatedGrad, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float epsilon, float weightDecay)
            => throw SparseOpenClNotImpl("adagrad_update");
        public void SparseNagUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
            => throw SparseOpenClNotImpl("nag_update");
        public void SparseAdadeltaUpdate(IGpuBuffer param, IGpuBuffer accumGrad, IGpuBuffer accumUpdate, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float rho, float epsilon, float weightDecay)
            => throw SparseOpenClNotImpl("adadelta_update");
        public void SparseAmsgradUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
            => throw SparseOpenClNotImpl("amsgrad_update");
        public void SparseAdamaxUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer u, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
            => throw SparseOpenClNotImpl("adamax_update");
        public void SparseLionUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float weightDecay)
            => throw SparseOpenClNotImpl("lion_update");
        public void SparseNadamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
            => throw SparseOpenClNotImpl("nadam_update");
        public void SparseFtrlUpdate(IGpuBuffer param, IGpuBuffer z, IGpuBuffer n, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Reg, float l2Reg, float beta)
            => throw SparseOpenClNotImpl("ftrl_update");
        public void SparseProximalL1Update(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Strength)
            => throw SparseOpenClNotImpl("proximal_l1_update");
    }
}
