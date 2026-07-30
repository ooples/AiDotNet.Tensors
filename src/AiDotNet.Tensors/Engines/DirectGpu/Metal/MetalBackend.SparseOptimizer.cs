// Copyright (c) AiDotNet. All rights reserved.
// Metal backend - sparse optimizer scatter updates.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    private void DispatchSparseOptimizerMetal(uint operation, IGpuBuffer param,
        IGpuBuffer? state1, IGpuBuffer? state2, IGpuBuffer? state3,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, int step,
        float learningRate, float a = 0f, float b = 0f, float c = 0f, float d = 0f)
    {
        ThrowIfDisposed();
        if (nnz < 0 || nnz > sparseIndices.Size || nnz > sparseValues.Size)
            throw new ArgumentOutOfRangeException(nameof(nnz), "nnz exceeds a sparse input buffer.");
        if ((state1 is not null && state1.Size < param.Size)
            || (state2 is not null && state2.Size < param.Size)
            || (state3 is not null && state3.Size < param.Size))
            throw new ArgumentException("Sparse optimizer state buffer is shorter than the parameter buffer.");
        if (nnz == 0) return;

        using var dummy1 = state1 is null ? AllocateBuffer(1) : null;
        using var dummy2 = state2 is null ? AllocateBuffer(1) : null;
        using var dummy3 = state3 is null ? AllocateBuffer(1) : null;
        DispatchResidentMetal("sparse_optimizer_serial", 1,
            new[] { param, state1 ?? dummy1!, state2 ?? dummy2!, state3 ?? dummy3!, sparseIndices, sparseValues },
            (uint)param.Size, (uint)nnz, operation, (uint)step,
            unchecked((uint)SingleToInt32BitsCompat(learningRate)),
            unchecked((uint)SingleToInt32BitsCompat(a)),
            unchecked((uint)SingleToInt32BitsCompat(b)),
            unchecked((uint)SingleToInt32BitsCompat(c)),
            unchecked((uint)SingleToInt32BitsCompat(d)));
    }

    private static void ValidateSparseBiasCorrection(int step, float epsilon)
    {
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        if (epsilon <= 0f) throw new ArgumentOutOfRangeException(nameof(epsilon));
    }

    public void SparseSgdUpdate(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues,
        int nnz, float learningRate, float weightDecay)
        => DispatchSparseOptimizerMetal(0u, param, null, null, null, sparseIndices, sparseValues,
            nnz, 0, learningRate, weightDecay);

    public void SparseSgdMomentumUpdate(IGpuBuffer param, IGpuBuffer velocity,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float momentum, float weightDecay)
        => DispatchSparseOptimizerMetal(1u, param, velocity, null, null, sparseIndices, sparseValues,
            nnz, 0, learningRate, momentum, weightDecay);

    public void SparseAdamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        ValidateSparseBiasCorrection(step, epsilon);
        DispatchSparseOptimizerMetal(2u, param, m, v, null, sparseIndices, sparseValues,
            nnz, step, learningRate, beta1, beta2, epsilon, weightDecay);
    }

    public void SparseAdamWUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        ValidateSparseBiasCorrection(step, epsilon);
        DispatchSparseOptimizerMetal(3u, param, m, v, null, sparseIndices, sparseValues,
            nnz, step, learningRate, beta1, beta2, epsilon, weightDecay);
    }

    public void SparseRmspropUpdate(IGpuBuffer param, IGpuBuffer squaredAvg,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float rho, float epsilon, float weightDecay)
    {
        if (epsilon <= 0f) throw new ArgumentOutOfRangeException(nameof(epsilon));
        DispatchSparseOptimizerMetal(4u, param, squaredAvg, null, null, sparseIndices, sparseValues,
            nnz, 0, learningRate, rho, epsilon, weightDecay);
    }

    public void SparseAdagradUpdate(IGpuBuffer param, IGpuBuffer accumulatedGrad,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float epsilon, float weightDecay)
    {
        if (epsilon <= 0f) throw new ArgumentOutOfRangeException(nameof(epsilon));
        DispatchSparseOptimizerMetal(5u, param, accumulatedGrad, null, null, sparseIndices, sparseValues,
            nnz, 0, learningRate, epsilon, weightDecay);
    }

    public void SparseNagUpdate(IGpuBuffer param, IGpuBuffer velocity,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float momentum, float weightDecay)
        => DispatchSparseOptimizerMetal(6u, param, velocity, null, null, sparseIndices, sparseValues,
            nnz, 0, learningRate, momentum, weightDecay);

    public void SparseAdadeltaUpdate(IGpuBuffer param, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float rho, float epsilon, float weightDecay)
    {
        if (epsilon <= 0f) throw new ArgumentOutOfRangeException(nameof(epsilon));
        DispatchSparseOptimizerMetal(7u, param, accumGrad, accumUpdate, null, sparseIndices, sparseValues,
            nnz, 0, 0f, rho, epsilon, weightDecay);
    }

    public void SparseAmsgradUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        ValidateSparseBiasCorrection(step, epsilon);
        DispatchSparseOptimizerMetal(8u, param, m, v, vMax, sparseIndices, sparseValues,
            nnz, step, learningRate, beta1, beta2, epsilon, weightDecay);
    }

    public void SparseAdamaxUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer u,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        ValidateSparseBiasCorrection(step, epsilon);
        DispatchSparseOptimizerMetal(9u, param, m, u, null, sparseIndices, sparseValues,
            nnz, step, learningRate, beta1, beta2, epsilon, weightDecay);
    }

    public void SparseLionUpdate(IGpuBuffer param, IGpuBuffer m,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float weightDecay)
        => DispatchSparseOptimizerMetal(10u, param, m, null, null, sparseIndices, sparseValues,
            nnz, 0, learningRate, beta1, beta2, weightDecay);

    public void SparseNadamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        ValidateSparseBiasCorrection(step, epsilon);
        DispatchSparseOptimizerMetal(11u, param, m, v, null, sparseIndices, sparseValues,
            nnz, step, learningRate, beta1, beta2, epsilon, weightDecay);
    }

    public void SparseFtrlUpdate(IGpuBuffer param, IGpuBuffer z, IGpuBuffer n,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float l1Reg, float l2Reg, float beta)
        => DispatchSparseOptimizerMetal(12u, param, z, n, null, sparseIndices, sparseValues,
            nnz, 0, learningRate, l1Reg, l2Reg, beta);

    public void SparseProximalL1Update(IGpuBuffer param,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float l1Strength)
        => DispatchSparseOptimizerMetal(13u, param, null, null, null, sparseIndices, sparseValues,
            nnz, 0, learningRate, l1Strength);
}
