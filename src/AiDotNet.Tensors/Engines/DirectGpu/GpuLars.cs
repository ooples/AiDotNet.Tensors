// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The LARS update on any GPU backend, built from primitives every backend implements.
/// </summary>
/// <remarks>
/// <para>
/// LARS scales each layer's step by the local rate <c>eta ||w|| / (||g|| + beta ||w|| + epsilon)</c>, which needs two
/// whole-tensor norms. The per-backend kernels disagreed: CUDA and HIP ignored the trust coefficient (plain momentum
/// SGD), OpenCL multiplied the step by it, and Metal, Vulkan and WebGPU each placed the learning rate differently. This
/// computes the norms between phases and runs the same algorithm as the CPU optimizer on every backend.
/// </para>
/// <para><b>Reference:</b> Y. You, I. Gitman and B. Ginsburg, "Large Batch Training of Convolutional Networks",
/// arXiv:1708.03888, 2017, Algorithm 1.</para>
/// </remarks>
internal static class GpuLars
{
    /// <summary>
    /// <c>v = momentum v + lr_local (g + weightDecay w)</c>, then <c>w = w - v</c>, where
    /// <c>lr_local = learningRate trustCoeff ||w|| / (||g|| + weightDecay ||w|| + epsilon)</c>, or the learning rate
    /// itself where either norm is below <paramref name="epsilon"/>.
    /// </summary>
    internal static void Step(
        IDirectGpuBackend backend, IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, float trustCoeff, float epsilon, int size)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (velocity is null) throw new ArgumentNullException(nameof(velocity));
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (epsilon < 0f) throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must not be negative.");

        float paramNorm = backend.L2Norm(param, size);
        float gradNorm = backend.L2Norm(gradient, size);
        // Same rule as the CPU optimizer: a zero norm keeps the learning rate even at epsilon 0, where the ratio is 0 / 0.
        float localRate = paramNorm < epsilon || gradNorm < epsilon || !(paramNorm > 0f) || !(gradNorm > 0f)
            ? learningRate
            : learningRate * (trustCoeff * paramNorm / (gradNorm + weightDecay * paramNorm + epsilon));

        // v = momentum v + localRate (g + weightDecay w) ;  w = w - v
        using var update = backend.AllocateBuffer(size);
        backend.AddScaled(gradient, param, update, 1f, weightDecay, size);
        backend.AddScaled(velocity, update, velocity, momentum, localRate, size);
        backend.AddScaled(param, velocity, param, 1f, -1f, size);
    }
}
