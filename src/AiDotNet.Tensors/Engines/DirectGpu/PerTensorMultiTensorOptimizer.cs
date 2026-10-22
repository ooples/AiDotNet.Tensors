// Copyright (c) AiDotNet. All rights reserved.
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IMultiTensorGpuOptimizerBackend"/> for backends without a fused multi-tensor kernel (#1142): the same
/// validation as CUDA's fused path, then each tensor through the backend's own native Adam/AdamW kernel. Every
/// launch stays on the device; the fused kernel only saves the per-tensor launch overhead.
/// </summary>
internal static class PerTensorMultiTensorOptimizer
{
    public static void Update(IDirectGpuBackend backend, bool decoupledWeightDecay,
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments, IReadOnlyList<IGpuBuffer> secondMoments,
        IReadOnlyList<int> sizes, float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
    {
        ValidateAdam(parameters, gradients, firstMoments, secondMoments, sizes, epsilon, step);
        int n = parameters.Count;
        for (int t = 0; t < n; t++)
        {
            if (decoupledWeightDecay)
                backend.AdamWUpdate(parameters[t], gradients[t], firstMoments[t], secondMoments[t],
                    learningRate, beta1, beta2, epsilon, weightDecay, step, sizes[t]);
            else
                backend.AdamUpdate(parameters[t], gradients[t], firstMoments[t], secondMoments[t],
                    learningRate, beta1, beta2, epsilon, weightDecay, step, sizes[t]);
        }
    }

    /// <summary>The argument checks every Adam/AdamW <c>MultiTensorUpdate</c> runs before launching anything.</summary>
    public static void ValidateAdam(IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments, IReadOnlyList<IGpuBuffer> secondMoments, IReadOnlyList<int> sizes,
        float epsilon, int step)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        if (firstMoments is null) throw new ArgumentNullException(nameof(firstMoments));
        if (secondMoments is null) throw new ArgumentNullException(nameof(secondMoments));
        if (sizes is null) throw new ArgumentNullException(nameof(sizes));
        int n = parameters.Count;
        if (gradients.Count != n || firstMoments.Count != n || secondMoments.Count != n || sizes.Count != n)
            throw new ArgumentException("Multi-tensor optimizer buffer lists must all have the same length.");
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step), "Step must be at least 1.");
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        for (int t = 0; t < n; t++)
            if (sizes[t] <= 0) throw new ArgumentOutOfRangeException(nameof(sizes), "Every tensor size must be positive.");
    }

    /// <summary>
    /// The guarded update for a backend whose kernels cannot read the guard: the guard is read on the host (a
    /// download, which waits for the queue) and the update is skipped when it is not finite.
    /// </summary>
    public static void UpdateGuarded(IDirectGpuBackend backend, bool decoupledWeightDecay,
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments, IReadOnlyList<IGpuBuffer> secondMoments,
        IReadOnlyList<int> sizes, float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step, IGpuBuffer finiteGuard)
    {
        if (finiteGuard is null) throw new ArgumentNullException(nameof(finiteGuard));
        double guard = ReadGuard(backend, finiteGuard);
        if (double.IsNaN(guard) || double.IsInfinity(guard)) return;
        Update(backend, decoupledWeightDecay, parameters, gradients, firstMoments, secondMoments, sizes,
            learningRate, beta1, beta2, epsilon, weightDecay, step);
    }


    /// <summary>Plain SGD over a tensor list through the backend's own per-tensor <c>SgdUpdate</c> kernel.</summary>
    public static void Sgd(IDirectGpuBackend backend,
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes,
        float learningRate, float weightDecay)
    {
        ValidateSgd(parameters, gradients, sizes);
        for (int t = 0; t < parameters.Count; t++)
            backend.SgdUpdate(parameters[t], gradients[t], learningRate, weightDecay, sizes[t]);
    }

    /// <summary>
    /// The guard's double, from the first two float slots of a float buffer (how the plans allocate it). Not
    /// <c>DownloadByteBuffer</c>: OpenCL accepts only its byte-buffer type there, so the guarded fallbacks threw on it.
    /// </summary>
    private static double ReadGuard(IDirectGpuBackend backend, IGpuBuffer finiteGuard)
    {
        float[] slots = backend.DownloadBuffer(finiteGuard);
        if (slots.Length < 2) throw new ArgumentException("The double guard needs two float slots.", nameof(finiteGuard));
        var bytes = new byte[sizeof(double)];
        Buffer.BlockCopy(slots, 0, bytes, 0, sizeof(double));
        return BitConverter.ToDouble(bytes, 0);
    }

    /// <summary><see cref="Sgd"/> skipped when the device-side finiteness guard is not finite (read back here).</summary>
    public static void SgdGuarded(IDirectGpuBackend backend,
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes,
        float learningRate, float weightDecay, IGpuBuffer finiteGuard)
    {
        if (finiteGuard is null) throw new ArgumentNullException(nameof(finiteGuard));
        ValidateSgd(parameters, gradients, sizes);
        double guard = ReadGuard(backend, finiteGuard);
        if (double.IsNaN(guard) || double.IsInfinity(guard)) return;
        Sgd(backend, parameters, gradients, sizes, learningRate, weightDecay);
    }

    /// <summary>The argument checks every <c>SgdMultiTensorUpdate</c> runs before launching anything.</summary>
    public static void ValidateSgd(
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        if (sizes is null) throw new ArgumentNullException(nameof(sizes));
        int n = parameters.Count;
        if (gradients.Count != n || sizes.Count != n)
            throw new ArgumentException("Multi-tensor optimizer buffer lists must all have the same length.");
        for (int t = 0; t < n; t++)
        {
            if (parameters[t] is null || gradients[t] is null)
                throw new ArgumentNullException(nameof(parameters), "Every parameter and gradient buffer is required.");
            if (sizes[t] <= 0) throw new ArgumentOutOfRangeException(nameof(sizes), "Every tensor size must be positive.");
            if (parameters[t].Size < sizes[t] || gradients[t].Size < sizes[t])
                throw new ArgumentException("A parameter or gradient buffer is smaller than its size.", nameof(sizes));
        }
    }
}
