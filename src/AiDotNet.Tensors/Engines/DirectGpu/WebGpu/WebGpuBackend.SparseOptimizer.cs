// Copyright (c) AiDotNet. All rights reserved.
// WebGPU backend - native sparse optimizer scatter updates.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    public void SparseSgdUpdate(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float weightDecay)
        => DispatchSparse("sparse_sgd_update", param, sparseIndices, sparseValues, SharedDummyBuffer, SharedDummyBuffer, SharedDummyBuffer,
            MakeOptimizerUniforms(nnz, learningRate, 0, 0, 0, weightDecay, 0), nnz);

    public void SparseSgdMomentumUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
        => DispatchSparse("sparse_sgd_momentum_update", param, sparseIndices, sparseValues, velocity, SharedDummyBuffer, SharedDummyBuffer,
            MakeOptimizerUniforms(nnz, learningRate, momentum, 0, 0, weightDecay, 0), nnz);

    public void SparseAdamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => DispatchSparseBiasCorrected("sparse_adam_update", param, sparseIndices, sparseValues, m, v, SharedDummyBuffer,
            nnz, learningRate, beta1, beta2, epsilon, weightDecay, step);

    public void SparseAdamWUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => DispatchSparseBiasCorrected("sparse_adamw_update", param, sparseIndices, sparseValues, m, v, SharedDummyBuffer,
            nnz, learningRate, beta1, beta2, epsilon, weightDecay, step);

    public void SparseRmspropUpdate(IGpuBuffer param, IGpuBuffer squaredAvg, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float rho, float epsilon, float weightDecay)
    {
        EnsureEpsilon(epsilon);
        DispatchSparse("sparse_rmsprop_update", param, sparseIndices, sparseValues, squaredAvg, SharedDummyBuffer, SharedDummyBuffer,
            MakeOptimizerUniforms(nnz, learningRate, rho, 0, epsilon, weightDecay, 0), nnz);
    }

    public void SparseAdagradUpdate(IGpuBuffer param, IGpuBuffer accumulatedGrad, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float epsilon, float weightDecay)
    {
        EnsureEpsilon(epsilon);
        DispatchSparse("sparse_adagrad_update", param, sparseIndices, sparseValues, accumulatedGrad, SharedDummyBuffer, SharedDummyBuffer,
            MakeOptimizerUniforms(nnz, learningRate, 0, 0, epsilon, weightDecay, 0), nnz);
    }

    public void SparseNagUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
        => DispatchSparse("sparse_nag_update", param, sparseIndices, sparseValues, velocity, SharedDummyBuffer, SharedDummyBuffer,
            MakeOptimizerUniforms(nnz, learningRate, momentum, 0, 0, weightDecay, 0), nnz);

    public void SparseAdadeltaUpdate(IGpuBuffer param, IGpuBuffer accumGrad, IGpuBuffer accumUpdate, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float rho, float epsilon, float weightDecay)
    {
        EnsureEpsilon(epsilon);
        DispatchSparse("sparse_adadelta_update", param, sparseIndices, sparseValues, accumGrad, accumUpdate, SharedDummyBuffer,
            MakeOptimizerUniforms(nnz, 0, rho, 0, epsilon, weightDecay, 0), nnz);
    }

    public void SparseAmsgradUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => DispatchSparseBiasCorrected("sparse_amsgrad_update", param, sparseIndices, sparseValues, m, v, vMax,
            nnz, learningRate, beta1, beta2, epsilon, weightDecay, step);

    public void SparseAdamaxUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer u, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => DispatchSparseBiasCorrected("sparse_adamax_update", param, sparseIndices, sparseValues, m, u, SharedDummyBuffer,
            nnz, learningRate, beta1, beta2, epsilon, weightDecay, step);

    public void SparseLionUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float weightDecay)
        => DispatchSparse("sparse_lion_update", param, sparseIndices, sparseValues, m, SharedDummyBuffer, SharedDummyBuffer,
            MakeOptimizerUniforms(nnz, learningRate, beta1, beta2, 0, weightDecay, 0), nnz);

    public void SparseNadamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => DispatchSparseBiasCorrected("sparse_nadam_update", param, sparseIndices, sparseValues, m, v, SharedDummyBuffer,
            nnz, learningRate, beta1, beta2, epsilon, weightDecay, step);

    public void SparseFtrlUpdate(IGpuBuffer param, IGpuBuffer z, IGpuBuffer n, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Reg, float l2Reg, float beta)
        => DispatchSparse("sparse_ftrl_update", param, sparseIndices, sparseValues, z, n, SharedDummyBuffer,
            MakeOptimizerUniforms(nnz, learningRate, l1Reg, l2Reg, 0, 0, 0, beta), nnz);

    public void SparseProximalL1Update(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Strength)
        => DispatchSparse("sparse_proximal_l1_update", param, sparseIndices, sparseValues, SharedDummyBuffer, SharedDummyBuffer, SharedDummyBuffer,
            MakeOptimizerUniforms(nnz, learningRate, l1Strength, 0, 0, 0, 0), nnz);

    private void DispatchSparseBiasCorrected(string kernelName,
        IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues,
        IGpuBuffer state1, IGpuBuffer state2, IGpuBuffer state3,
        int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        EnsureEpsilon(epsilon);
        DispatchSparse(kernelName, param, sparseIndices, sparseValues, state1, state2, state3,
            MakeOptimizerUniforms(nnz, learningRate, beta1, beta2, epsilon, weightDecay, step), nnz);
    }

    private void DispatchSparse(string kernelName,
        IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues,
        IGpuBuffer state1, IGpuBuffer state2, IGpuBuffer state3,
        float[] uniforms, int nnz)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (state1 is null) throw new ArgumentNullException(nameof(state1));
        if (state2 is null) throw new ArgumentNullException(nameof(state2));
        if (state3 is null) throw new ArgumentNullException(nameof(state3));
        // The kernel indexes state[i] for i in [0, param.Size); undersized state buffers would read/
        // write out of bounds on the device (crash / device loss). Validate coverage up front.
        if (state1.Size < param.Size) throw new ArgumentException($"state1 buffer holds {state1.Size} elements but param needs {param.Size}.", nameof(state1));
        if (state2.Size < param.Size) throw new ArgumentException($"state2 buffer holds {state2.Size} elements but param needs {param.Size}.", nameof(state2));
        if (state3.Size < param.Size) throw new ArgumentException($"state3 buffer holds {state3.Size} elements but param needs {param.Size}.", nameof(state3));
        if (nnz == 0) return;
        uniforms[8] = BitConverter.Int32BitsToSingle(param.Size);
        Dispatch6BufferAsync("SparseOptimizer", WebGpuKernels.SparseOptimizerSource, kernelName,
            param, sparseIndices, sparseValues, state1, state2, state3, uniforms, nnz).GetAwaiter().GetResult();
    }

    private static void EnsureSparseArgs(IGpuBuffer? param, IGpuBuffer? sparseIndices, IGpuBuffer? sparseValues, int nnz)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (sparseIndices is null) throw new ArgumentNullException(nameof(sparseIndices));
        if (sparseValues is null) throw new ArgumentNullException(nameof(sparseValues));
        if (nnz < 0) throw new ArgumentOutOfRangeException(nameof(nnz));
        // The kernel reads sparseIndices[k]/sparseValues[k] for k in [0, nnz); reject undersized
        // buffers before dispatch so a bad caller gets a clear error instead of a device OOB access.
        if (sparseIndices.Size < nnz)
            throw new ArgumentException($"sparseIndices buffer holds {sparseIndices.Size} elements but nnz={nnz}.", nameof(sparseIndices));
        if (sparseValues.Size < nnz)
            throw new ArgumentException($"sparseValues buffer holds {sparseValues.Size} elements but nnz={nnz}.", nameof(sparseValues));
    }

    private static void EnsureEpsilon(float epsilon)
    {
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
    }
}
#endif
