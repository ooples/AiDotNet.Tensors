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

    /// <summary>
    /// Proximal gradient (ISTA) step with L1 soft-threshold prox, in place on the GPU.
    /// CUDA-only (proximal_l1_update lives on CudaBackend); returns false on any
    /// other backend or when param/grad aren't GPU-resident, so the caller falls
    /// back to the CPU path.
    /// </summary>
    public static bool TryProximalL1Step(Tensor<float> p, Tensor<float> g, float lr, float l1Strength)
    {
        if (p is null) throw new ArgumentNullException(nameof(p));
        if (g is null) throw new ArgumentNullException(nameof(g));
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false;
        if (!(e.GetBackend() is DirectGpu.CUDA.CudaBackend cb)) return false;
        var pb = p.TryGetGpuBuffer(); var gb = g.TryGetGpuBuffer();
        if (pb is null || gb is null) return false;
        cb.ProximalL1Update(pb, gb, lr, l1Strength, p.Length);
        return true;
    }

    /// <summary>
    /// Allocates + initializes GPU-resident 8-bit Adam state (int8 m/v + per-block
    /// double scales) for a parameter of <paramref name="length"/> elements. m starts
    /// at the signed-zero byte (128), v at 0, scales at 1e-10 (matching the CPU
    /// Adam8BitOptimizer's m_0=v_0=0 init). The caller keeps the four buffers
    /// resident across steps and frees them with <see cref="FreeGpuBuffer"/>.
    /// Returns false (buffers null) on any non-CUDA backend.
    /// </summary>
    public static bool TryAllocAdam8BitState(int length, int blockSize,
        out DirectGpu.IGpuBuffer? mQ, out DirectGpu.IGpuBuffer? vQ,
        out DirectGpu.IGpuBuffer? mScales, out DirectGpu.IGpuBuffer? vScales)
    {
        mQ = vQ = mScales = vScales = null;
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false;
        if (!(e.GetBackend() is DirectGpu.CUDA.CudaBackend cb)) return false;
        int nb = (length + blockSize - 1) / blockSize;
        mQ = cb.AllocateByteBuffer(length);
        vQ = cb.AllocateByteBuffer(length);
        mScales = cb.AllocateByteBuffer(nb * 8);
        vScales = cb.AllocateByteBuffer(nb * 8);
        var im = new byte[length]; for (int i = 0; i < length; i++) im[i] = 128; // signed-zero
        var iv = new byte[length];                                              // unsigned-zero (already 0)
        var isc = new byte[nb * 8]; for (int b = 0; b < nb; b++) BitConverter.GetBytes(1e-10).CopyTo(isc, b * 8);
        cb.UploadBytes(mQ, im); cb.UploadBytes(vQ, iv);
        cb.UploadBytes(mScales, isc); cb.UploadBytes(vScales, (byte[])isc.Clone());
        return true;
    }

    /// <summary>Frees a GPU buffer allocated by <see cref="TryAllocAdam8BitState"/> (safe on null / non-CUDA).</summary>
    public static void FreeGpuBuffer(DirectGpu.IGpuBuffer? buffer)
    {
        if (buffer is IDisposable d) d.Dispose();
    }

    /// <summary>
    /// GPU-resident 8-bit Adam step: dequantize → Adam → param update → requantize,
    /// in place on the GPU (no host download of moments). CUDA-only; returns false
    /// (→ CPU fallback) when param/grad aren't GPU-resident or any state buffer is null.
    /// </summary>
    public static bool TryAdam8BitStep(Tensor<float> p, Tensor<float> g,
        DirectGpu.IGpuBuffer? mQ, DirectGpu.IGpuBuffer? vQ, DirectGpu.IGpuBuffer? mScales, DirectGpu.IGpuBuffer? vScales,
        float lr, float beta1, float beta2, float epsilon, float biasCorrection1, float biasCorrection2, int blockSize)
    {
        if (p is null || g is null) return false;
        if (mQ is null || vQ is null || mScales is null || vScales is null) return false;
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false;
        if (!(e.GetBackend() is DirectGpu.CUDA.CudaBackend cb)) return false;
        var pb = p.TryGetGpuBuffer(); var gb = g.TryGetGpuBuffer();
        if (pb is null || gb is null) return false;
        int nb = (p.Length + blockSize - 1) / blockSize;
        cb.Adam8BitUpdate(pb, gb, mQ, vQ, mScales, vScales,
            lr, beta1, beta2, epsilon, 1f - beta1, 1f - beta2, biasCorrection1, biasCorrection2, blockSize, p.Length, nb);
        return true;
    }

    // ---- GPU-resident steps for the rest of the backend optimizer kernels (same contract as TryAdamStep:
    //      returns false if not on a GPU engine or any tensor isn't GPU-resident, so the caller falls back to CPU).
    //      All buffers must be GPU-resident; the kernel updates param + state IN PLACE with no host download. ----

    public static bool TrySgdMomentumStep(Tensor<float> p, Tensor<float> g, Tensor<float> velocity, float lr, float momentum, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; var b = e.GetBackend(); if (b is null) return false;
        var pb = p?.TryGetGpuBuffer(); var gb = g?.TryGetGpuBuffer(); var vb = velocity?.TryGetGpuBuffer();
        if (pb is null || gb is null || vb is null) return false;
        b.SgdMomentumUpdate(pb, gb, vb, lr, momentum, weightDecay, p!.Length); return true;
    }

    public static bool TryRmspropStep(Tensor<float> p, Tensor<float> g, Tensor<float> squaredAvg, float lr, float rho, float epsilon, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; var b = e.GetBackend(); if (b is null) return false;
        var pb = p?.TryGetGpuBuffer(); var gb = g?.TryGetGpuBuffer(); var sb = squaredAvg?.TryGetGpuBuffer();
        if (pb is null || gb is null || sb is null) return false;
        b.RmspropUpdate(pb, gb, sb, lr, rho, epsilon, weightDecay, p!.Length); return true;
    }

    public static bool TryAdagradStep(Tensor<float> p, Tensor<float> g, Tensor<float> accum, float lr, float epsilon, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; var b = e.GetBackend(); if (b is null) return false;
        var pb = p?.TryGetGpuBuffer(); var gb = g?.TryGetGpuBuffer(); var ab = accum?.TryGetGpuBuffer();
        if (pb is null || gb is null || ab is null) return false;
        b.AdagradUpdate(pb, gb, ab, lr, epsilon, weightDecay, p!.Length); return true;
    }

    public static bool TryNagStep(Tensor<float> p, Tensor<float> g, Tensor<float> velocity, float lr, float momentum, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; var b = e.GetBackend(); if (b is null) return false;
        var pb = p?.TryGetGpuBuffer(); var gb = g?.TryGetGpuBuffer(); var vb = velocity?.TryGetGpuBuffer();
        if (pb is null || gb is null || vb is null) return false;
        b.NagUpdate(pb, gb, vb, lr, momentum, weightDecay, p!.Length); return true;
    }

    public static bool TryLarsStep(Tensor<float> p, Tensor<float> g, Tensor<float> velocity, float lr, float momentum, float weightDecay, float trustCoeff)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; var b = e.GetBackend(); if (b is null) return false;
        var pb = p?.TryGetGpuBuffer(); var gb = g?.TryGetGpuBuffer(); var vb = velocity?.TryGetGpuBuffer();
        if (pb is null || gb is null || vb is null) return false;
        b.LarsUpdate(pb, gb, vb, lr, momentum, weightDecay, trustCoeff, p!.Length); return true;
    }

    public static bool TryLambStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, Tensor<float> v, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; var b = e.GetBackend(); if (b is null) return false;
        var pb = p?.TryGetGpuBuffer(); var gb = g?.TryGetGpuBuffer(); var mb = m?.TryGetGpuBuffer(); var vb = v?.TryGetGpuBuffer();
        if (pb is null || gb is null || mb is null || vb is null) return false;
        b.LambUpdate(pb, gb, mb, vb, lr, beta1, beta2, epsilon, weightDecay, step, p!.Length); return true;
    }

    public static bool TryAdadeltaStep(Tensor<float> p, Tensor<float> g, Tensor<float> accumGrad, Tensor<float> accumUpdate, float rho, float epsilon, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; var b = e.GetBackend(); if (b is null) return false;
        var pb = p?.TryGetGpuBuffer(); var gb = g?.TryGetGpuBuffer(); var ag = accumGrad?.TryGetGpuBuffer(); var au = accumUpdate?.TryGetGpuBuffer();
        if (pb is null || gb is null || ag is null || au is null) return false;
        b.AdadeltaUpdate(pb, gb, ag, au, rho, epsilon, weightDecay, p!.Length); return true;
    }

    public static bool TryAmsgradStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, Tensor<float> v, Tensor<float> vMax, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; var b = e.GetBackend(); if (b is null) return false;
        var pb = p?.TryGetGpuBuffer(); var gb = g?.TryGetGpuBuffer(); var mb = m?.TryGetGpuBuffer(); var vb = v?.TryGetGpuBuffer(); var xb = vMax?.TryGetGpuBuffer();
        if (pb is null || gb is null || mb is null || vb is null || xb is null) return false;
        b.AmsgradUpdate(pb, gb, mb, vb, xb, lr, beta1, beta2, epsilon, weightDecay, step, p!.Length); return true;
    }

    public static bool TryAdamaxStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, Tensor<float> u, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; var b = e.GetBackend(); if (b is null) return false;
        var pb = p?.TryGetGpuBuffer(); var gb = g?.TryGetGpuBuffer(); var mb = m?.TryGetGpuBuffer(); var ub = u?.TryGetGpuBuffer();
        if (pb is null || gb is null || mb is null || ub is null) return false;
        b.AdamaxUpdate(pb, gb, mb, ub, lr, beta1, beta2, epsilon, weightDecay, step, p!.Length); return true;
    }

    public static bool TryLionStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, float lr, float beta1, float beta2, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; var b = e.GetBackend(); if (b is null) return false;
        var pb = p?.TryGetGpuBuffer(); var gb = g?.TryGetGpuBuffer(); var mb = m?.TryGetGpuBuffer();
        if (pb is null || gb is null || mb is null) return false;
        b.LionUpdate(pb, gb, mb, lr, beta1, beta2, weightDecay, p!.Length); return true;
    }

    public static bool TryNadamStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, Tensor<float> v, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; var b = e.GetBackend(); if (b is null) return false;
        var pb = p?.TryGetGpuBuffer(); var gb = g?.TryGetGpuBuffer(); var mb = m?.TryGetGpuBuffer(); var vb = v?.TryGetGpuBuffer();
        if (pb is null || gb is null || mb is null || vb is null) return false;
        b.NadamUpdate(pb, gb, mb, vb, lr, beta1, beta2, epsilon, weightDecay, step, p!.Length); return true;
    }

    public static bool TryFtrlStep(Tensor<float> p, Tensor<float> g, Tensor<float> z, Tensor<float> n, float lr, float l1Reg, float l2Reg, float beta)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; var b = e.GetBackend(); if (b is null) return false;
        var pb = p?.TryGetGpuBuffer(); var gb = g?.TryGetGpuBuffer(); var zb = z?.TryGetGpuBuffer(); var nb = n?.TryGetGpuBuffer();
        if (pb is null || gb is null || zb is null || nb is null) return false;
        b.FtrlUpdate(pb, gb, zb, nb, lr, l1Reg, l2Reg, beta, p!.Length); return true;
    }
}
