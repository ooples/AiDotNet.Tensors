// Copyright (c) AiDotNet. All rights reserved.
// Vulkan backend - deterministic resident sparse optimizer scatter updates.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    private void DispatchSparseOptimizer(IGpuBuffer param, IGpuBuffer? state1, IGpuBuffer? state2,
        IGpuBuffer? state3, IGpuBuffer sparseIndices, IGpuBuffer sparseValues,
        int nnz, int operation, int step, float learningRate, float p0, float p1, float p2, float p3)
    {
        EnsureInitialized();
        if (nnz < 0 || nnz > sparseIndices.Size || nnz > sparseValues.Size)
            throw new ArgumentOutOfRangeException(nameof(nnz), "nnz must fit both sparse input buffers.");
        if ((state1 is not null && state1.Size < param.Size) ||
            (state2 is not null && state2.Size < param.Size) ||
            (state3 is not null && state3.Size < param.Size))
            throw new ArgumentException("Sparse optimizer state buffers must be at least as large as the parameter buffer.");
        if (nnz == 0) return;

        using var state1Dummy = state1 is null ? AllocateBuffer(1) : null;
        using var state2Dummy = state2 is null ? AllocateBuffer(1) : null;
        using var state3Dummy = state3 is null ? AllocateBuffer(1) : null;
        GlslNaryOp(VulkanOptimizerKernels.Sparse,
            new[] { param, state1 ?? state1Dummy!, state2 ?? state2Dummy!, state3 ?? state3Dummy!, sparseIndices, sparseValues },
            1, new[] { (uint)param.Size, (uint)nnz, (uint)operation, (uint)step, FloatBits(learningRate),
                FloatBits(p0), FloatBits(p1), FloatBits(p2), FloatBits(p3) });
    }

    public void SparseSgdUpdate(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues,
        int nnz, float learningRate, float weightDecay)
        => DispatchSparseOptimizer(param, null, null, null, sparseIndices, sparseValues,
            nnz, 0, 0, learningRate, weightDecay, 0f, 0f, 0f);

    public void SparseSgdMomentumUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices,
        IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
        => DispatchSparseOptimizer(param, velocity, null, null, sparseIndices, sparseValues,
            nnz, 1, 0, learningRate, momentum, weightDecay, 0f, 0f);

    public void SparseAdamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices,
        IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
    {
        ValidateSparseBiasCorrected(step, epsilon);
        DispatchSparseOptimizer(param, m, v, null, sparseIndices, sparseValues,
            nnz, 2, step, learningRate, beta1, beta2, epsilon, weightDecay);
    }

    public void SparseAdamWUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices,
        IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
    {
        ValidateSparseBiasCorrected(step, epsilon);
        DispatchSparseOptimizer(param, m, v, null, sparseIndices, sparseValues,
            nnz, 3, step, learningRate, beta1, beta2, epsilon, weightDecay);
    }

    public void SparseRmspropUpdate(IGpuBuffer param, IGpuBuffer squaredAvg, IGpuBuffer sparseIndices,
        IGpuBuffer sparseValues, int nnz, float learningRate, float rho, float epsilon, float weightDecay)
    {
        ValidateSparseEpsilon(epsilon);
        DispatchSparseOptimizer(param, squaredAvg, null, null, sparseIndices, sparseValues,
            nnz, 4, 0, learningRate, rho, epsilon, weightDecay, 0f);
    }

    public void SparseAdagradUpdate(IGpuBuffer param, IGpuBuffer accumulatedGrad, IGpuBuffer sparseIndices,
        IGpuBuffer sparseValues, int nnz, float learningRate, float epsilon, float weightDecay)
    {
        ValidateSparseEpsilon(epsilon);
        DispatchSparseOptimizer(param, accumulatedGrad, null, null, sparseIndices, sparseValues,
            nnz, 5, 0, learningRate, epsilon, weightDecay, 0f, 0f);
    }

    public void SparseNagUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices,
        IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
        => DispatchSparseOptimizer(param, velocity, null, null, sparseIndices, sparseValues,
            nnz, 6, 0, learningRate, momentum, weightDecay, 0f, 0f);

    public void SparseAdadeltaUpdate(IGpuBuffer param, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float rho, float epsilon, float weightDecay)
    {
        ValidateSparseEpsilon(epsilon);
        DispatchSparseOptimizer(param, accumGrad, accumUpdate, null, sparseIndices, sparseValues,
            nnz, 7, 0, 0f, rho, epsilon, weightDecay, 0f);
    }

    public void SparseAmsgradUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1,
        float beta2, float epsilon, float weightDecay, int step)
    {
        ValidateSparseBiasCorrected(step, epsilon);
        DispatchSparseOptimizer(param, m, v, vMax, sparseIndices, sparseValues,
            nnz, 8, step, learningRate, beta1, beta2, epsilon, weightDecay);
    }

    public void SparseAdamaxUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer u, IGpuBuffer sparseIndices,
        IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
    {
        ValidateSparseBiasCorrected(step, epsilon);
        DispatchSparseOptimizer(param, m, u, null, sparseIndices, sparseValues,
            nnz, 9, step, learningRate, beta1, beta2, epsilon, weightDecay);
    }

    public void SparseLionUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer sparseIndices,
        IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float weightDecay)
        => DispatchSparseOptimizer(param, m, null, null, sparseIndices, sparseValues,
            nnz, 10, 0, learningRate, beta1, beta2, weightDecay, 0f);

    public void SparseNadamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices,
        IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
    {
        ValidateSparseBiasCorrected(step, epsilon);
        DispatchSparseOptimizer(param, m, v, null, sparseIndices, sparseValues,
            nnz, 11, step, learningRate, beta1, beta2, epsilon, weightDecay);
    }

    public void SparseFtrlUpdate(IGpuBuffer param, IGpuBuffer z, IGpuBuffer n, IGpuBuffer sparseIndices,
        IGpuBuffer sparseValues, int nnz, float learningRate, float l1Reg, float l2Reg, float beta)
        => DispatchSparseOptimizer(param, z, n, null, sparseIndices, sparseValues,
            nnz, 12, 0, learningRate, l1Reg, l2Reg, beta, 0f);

    public void SparseProximalL1Update(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues,
        int nnz, float learningRate, float l1Strength)
        => DispatchSparseOptimizer(param, null, null, null, sparseIndices, sparseValues,
            nnz, 13, 0, learningRate, l1Strength, 0f, 0f, 0f);

    private static void ValidateSparseBiasCorrected(int step, float epsilon)
    {
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        ValidateSparseEpsilon(epsilon);
    }

    private static void ValidateSparseEpsilon(float epsilon)
    {
        if (epsilon <= 0f) throw new ArgumentOutOfRangeException(nameof(epsilon));
    }
}
