// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Issue #336 — GPU-resident optimizer-step entry points. Consumers
/// (AiDotNet's <c>AdamOptimizer</c> / <c>SgdOptimizer</c> /
/// <c>AdamWOptimizer</c>) tag their <c>_tapeM</c> / <c>_tapeV</c> /
/// parameter tensors with <see cref="WeightLifetime.GpuPinned"/> via
/// <see cref="Helpers.TensorAllocator.RentPinnedOnGpu{T}"/> then call
/// one of these methods. The GPU kernel reads + writes the pinned
/// tensors directly via their device pointers — no per-step
/// cuMemcpyHtoD / DtoH round-trip.
/// </summary>
/// <remarks>
/// Path:
/// <list type="number">
/// <item>Caller tags param / grad / m / v with <see cref="WeightLifetime.GpuPinned"/>.</item>
/// <item><see cref="WeightRegistry.RegisterWeight{T}"/> allocates pinned
/// host memory + DMA-maps to GPU; populates <see cref="TensorBase{T}.OffloadDevicePointer"/>.</item>
/// <item>This helper calls <see cref="TensorBase{T}.TryGetGpuBuffer"/>
/// on each tensor; gets non-owning <see cref="IGpuBuffer"/> views.</item>
/// <item>Dispatches to <see cref="IDirectGpuBackend.AdamUpdate"/> /
/// <see cref="IDirectGpuBackend.SgdUpdate"/> on the active GPU backend.</item>
/// </list>
/// Returns true when the GPU path ran. Returns false (no-op) when any
/// argument fails the GPU-residency contract — caller should fall back
/// to CPU optimizer.
/// </remarks>
public static class GpuOptimizer
{
    /// <summary>
    /// Adam optimizer step on GPU-resident tensors. All four tensors
    /// (<paramref name="param"/>, <paramref name="grad"/>,
    /// <paramref name="m"/>, <paramref name="v"/>) must be
    /// <see cref="WeightLifetime.GpuPinned"/> /
    /// <see cref="WeightLifetime.GpuOffload"/> and reachable from the
    /// active GPU backend.
    /// </summary>
    /// <returns>True when the kernel ran. False when any precondition
    /// fails — caller should fall back to a CPU Adam step.</returns>
    public static bool TryAdamStep(
        Tensor<float> param, Tensor<float> grad,
        Tensor<float> m, Tensor<float> v,
        float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));

        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine gpuEngine)) return false;
        var backend = gpuEngine.GetBackend();
        if (backend is null) return false;

        var pBuf = param.TryGetGpuBuffer();
        var gBuf = grad.TryGetGpuBuffer();
        var mBuf = m.TryGetGpuBuffer();
        var vBuf = v.TryGetGpuBuffer();
        if (pBuf is null || gBuf is null || mBuf is null || vBuf is null) return false;

        backend.AdamUpdate(pBuf, gBuf, mBuf, vBuf,
            learningRate, beta1, beta2, epsilon, weightDecay, step, param.Length);
        return true;
    }

    /// <summary>
    /// AdamW optimizer step on GPU-resident tensors. Same contract as
    /// <see cref="TryAdamStep"/>; AdamW applies decoupled weight decay
    /// (decays the parameter directly rather than the gradient).
    /// </summary>
    public static bool TryAdamWStep(
        Tensor<float> param, Tensor<float> grad,
        Tensor<float> m, Tensor<float> v,
        float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
    {
        // The CUDA AdamUpdate kernel already takes weightDecay; passing
        // a non-zero value makes it AdamW. Some backends ship a separate
        // adamw_update kernel; for now we route both through AdamUpdate
        // and rely on the kernel implementation to apply the decay
        // correctly (the existing adam_step kernel handles this).
        return TryAdamStep(param, grad, m, v, learningRate, beta1, beta2,
            epsilon, weightDecay, step);
    }

    /// <summary>
    /// SGD optimizer step on GPU-resident tensors. Simpler signature
    /// than Adam — no momentum buffer needed for the plain SGD path.
    /// (SGD with momentum should use <see cref="TryAdamStep"/> with
    /// beta1=momentum, beta2=0, and a single moment buffer or use a
    /// dedicated SgdMomentumUpdate when the backend ships one.)
    /// </summary>
    public static bool TrySgdStep(
        Tensor<float> param, Tensor<float> grad, float learningRate)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));

        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine gpuEngine)) return false;
        var backend = gpuEngine.GetBackend();
        if (backend is null) return false;

        var pBuf = param.TryGetGpuBuffer();
        var gBuf = grad.TryGetGpuBuffer();
        if (pBuf is null || gBuf is null) return false;

        backend.SgdUpdate(pBuf, gBuf, learningRate, weightDecay: 0f, param.Length);
        return true;
    }
}
