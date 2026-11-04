// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Issue #336 — GPU-resident optimizer-step entry points. Consumers
/// (Adam, SGD, AdamW, RMSprop, Adagrad, LARS, LAMB, AdaDelta, AMSGrad,
/// AdaMax, Lion, Nadam, FTRL, and sparse variants) tag their optimizer
/// state and parameter tensors with <see cref="WeightLifetime.GpuPinned"/>
/// via <see cref="Helpers.TensorAllocator.RentPinnedOnGpu{T}"/> then call
/// one of these methods. The GPU kernel reads and writes the pinned tensors
/// directly via their device pointers, avoiding a per-step host/device
/// round-trip.
/// </summary>
/// <remarks>
/// Path:
/// <list type="number">
/// <item>Caller tags param / grad / m / v with <see cref="WeightLifetime.GpuPinned"/>.</item>
/// <item><see cref="WeightRegistry.RegisterWeight{T}"/> allocates pinned
/// host memory + DMA-maps to GPU; populates <see cref="TensorBase{T}.OffloadDevicePointer"/>.</item>
/// <item>This helper calls <see cref="TensorBase{T}.TryGetGpuBuffer"/>
/// on each tensor and gets non-owning <see cref="IGpuBuffer"/> views.</item>
/// <item>Dispatches to the matching <see cref="IDirectGpuBackend"/> optimizer
/// update method on the active GPU backend.</item>
/// </list>
/// Returns true when the GPU path ran. Returns false (no-op) when any
/// argument fails the GPU-residency contract — caller should fall back
/// to CPU optimizer.
/// </remarks>
public static class GpuOptimizer
{
    private static GpuSyncPoint CreateWriteSyncPoint(IDirectGpuBackend backend)
        => backend is IAsyncGpuBackend asyncBackend
            ? asyncBackend.CreateSyncPoint()
            : GpuSyncPoint.CreateComplete();

    private static void MarkGpuUpdated(IDirectGpuBackend backend, Tensor<float>? tensor)
    {
        if (tensor is null) return;
        tensor.MarkModified(CreateWriteSyncPoint(backend));
        tensor._gpuBufferVersion = tensor.GpuCacheVersion;
    }

    /// <summary>
    /// Creates a zero-filled optimizer-state tensor (e.g. Adam's first/second moments) that lives on the GPU when the
    /// current engine is a GPU engine, so the fused optimizer kernels update it in place with no host traffic; a
    /// host read downloads it lazily. It is not a step activation, so tape-scoped release never frees it. On a CPU
    /// engine it is an ordinary zero tensor.
    /// </summary>
    public static Tensor<float> CreateStateTensor(int[] shape)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (AiDotNetEngine.Current is DirectGpuTensorEngine gpuEngine
            && gpuEngine.TryCreateDeviceStateTensor(shape, out var state))
            return state;
        return new Tensor<float>(shape);
    }

    /// <summary>
    /// Marks a parameter updated in place on the device. A parameter with a host backing array becomes
    /// device-authoritative: its persistent weight entry follows the new version and a host read downloads it lazily.
    /// </summary>
    private static void MarkParameterUpdated(DirectGpuTensorEngine engine, IDirectGpuBackend backend, Tensor<float> param, IGpuBuffer buffer)
    {
        if (param.GetBackingArrayForCacheLookupUnsafe() is not null)
            engine.CommitParameterUpdatedOnDevice(param, buffer, backend, CreateWriteSyncPoint(backend));
        else
            MarkGpuUpdated(backend, param);
    }

    private static void MarkGpuUpdated(IDirectGpuBackend backend, Tensor<float>? first, Tensor<float>? second)
    {
        MarkGpuUpdated(backend, first);
        MarkGpuUpdated(backend, second);
    }

    private static void MarkGpuUpdated(IDirectGpuBackend backend, Tensor<float>? first, Tensor<float>? second, Tensor<float>? third)
    {
        MarkGpuUpdated(backend, first);
        MarkGpuUpdated(backend, second);
        MarkGpuUpdated(backend, third);
    }

    private static void MarkGpuUpdated(IDirectGpuBackend backend, Tensor<float>? first, Tensor<float>? second, Tensor<float>? third, Tensor<float>? fourth)
    {
        MarkGpuUpdated(backend, first);
        MarkGpuUpdated(backend, second);
        MarkGpuUpdated(backend, third);
        MarkGpuUpdated(backend, fourth);
    }

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

        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine gpuEngine)) return false; using var __gpuOp = gpuEngine.EnterGpuOp();
        var backend = gpuEngine.GetBackend();
        if (backend is null) return false;

        var gBuf = grad.TryGetGpuBuffer();
        var mBuf = m.TryGetGpuBuffer();
        var vBuf = v.TryGetGpuBuffer();
        if (gBuf is null || mBuf is null || vBuf is null) return false;

        if (!TryUpdateParameter(gpuEngine, backend, param, pBuf => backend.AdamUpdate(pBuf, gBuf, mBuf, vBuf,
                learningRate, beta1, beta2, epsilon, weightDecay, step, param.Length), "TryAdamStep"))
            return false;
        MarkGpuUpdated(backend, m, v);
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
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));

        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine gpuEngine)) return false; using var __gpuOp = gpuEngine.EnterGpuOp();
        var backend = gpuEngine.GetBackend();
        if (backend is null) return false;

        var gBuf = grad.TryGetGpuBuffer();
        var mBuf = m.TryGetGpuBuffer();
        var vBuf = v.TryGetGpuBuffer();
        // Each decline is recorded: the caller silently falls back to op-by-op AdamW on the host copies, which is
        // correct but downloads the parameter, gradient and moments every step.
        if (gBuf is null) { GpuLaunchProbe.OnFallback("TryAdamWStep-gradient-not-device-resident", null); return false; }
        if (mBuf is null || vBuf is null) { GpuLaunchProbe.OnFallback("TryAdamWStep-moments-not-device-resident", null); return false; }
        // A host-authoritative parameter (eager training) is moved onto the device once and kept there, as PyTorch
        // keeps parameters on the GPU; its host copy is downloaded only when read.
        // A ParameterBuffer view is always updated through its flat array's device buffer -- the one a host read
        // downloads and forwards read -- never through a buffer some other path bound to this tensor object.
        if (!TryUpdateParameter(gpuEngine, backend, param, pBuf => backend.AdamWUpdate(pBuf, gBuf, mBuf, vBuf,
                learningRate, beta1, beta2, epsilon, weightDecay, step, param.Length), "TryAdamWStep"))
            return false;
        // AdamW mutates param, m and v on the GPU — mark all three current so a later read
        // doesn't serve stale host data (TryUpdateParameter marks the parameter).
        MarkGpuUpdated(backend, m, v);
        GpuLaunchProbe.OnAdamWDispatch();
        return true;
    }

    /// <summary>
    /// Runs a fused in-place update on <paramref name="param"/>'s device buffer and marks the parameter updated. A
    /// host-authoritative parameter (eager training) is moved onto the device once and kept there, as PyTorch keeps
    /// parameters on the GPU; its host copy is downloaded only when read. A ParameterBuffer view is always updated
    /// through its flat array's device buffer -- the one a host read downloads and forwards read -- never through a
    /// buffer some other path bound to this tensor object; a view the backend cannot address as a sub-buffer is
    /// updated in a staged device copy and written back. Every fused step goes through here: outside AdamW they took
    /// only a parameter already bound to a device buffer, so an eager model's parameters (host-authoritative, or
    /// views) always declined and the step ran on the host. Returns <c>false</c> (recorded as a fallback) when the
    /// parameter cannot be placed.
    /// </summary>
    private static bool TryUpdateParameter(DirectGpuTensorEngine engine, IDirectGpuBackend backend, Tensor<float> param,
        Action<IGpuBuffer> update, string site)
    {
        // Privatize a copy-on-write share (a DeepCopy snapshot) first: the in-place device write would reach it; the
        // detach also drops the binding to the shared device buffer.
        param.EnsureOwnedForDeviceWrite();
        var pBuf = DirectGpuTensorEngine.IsFlatParameterView(param)
            ? engine.AcquireParameterBuffer(param, out _)
            : param.TryGetGpuBuffer() ?? engine.AcquireParameterBuffer(param, out _);
        if (pBuf is null && engine.TryStageFlatParameter(param) is { } staged)
        {
            // The staging buffer is returned even when the update or the copy back throws.
            using (staged)
            {
                update(staged.Staged);
                backend.Copy(staged.Staged, 0, staged.FlatBuffer, staged.Offset, param.Length);
            }
            MarkParameterUpdated(engine, backend, param, staged.FlatBuffer);
            return true;
        }
        if (pBuf is null) { GpuLaunchProbe.OnFallback(site + "-parameter-not-placeable", null); return false; }
        update(pBuf);
        MarkParameterUpdated(engine, backend, param, pBuf);
        return true;
    }

    /// <summary>
    /// SGD optimizer step on GPU-resident tensors. Simpler signature
    /// than Adam: no momentum buffer is needed for the plain SGD path.
    /// SGD with momentum should use <see cref="TrySgdMomentumStep"/>.
    /// </summary>
    public static bool TrySgdStep(
        Tensor<float> param, Tensor<float> grad, float learningRate)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));

        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine gpuEngine)) return false; using var __gpuOp = gpuEngine.EnterGpuOp();
        var backend = gpuEngine.GetBackend();
        if (backend is null) return false;

        var gBuf = grad.TryGetGpuBuffer();
        if (gBuf is null) return false;

        return TryUpdateParameter(gpuEngine, backend, param,
            pBuf => backend.SgdUpdate(pBuf, gBuf, learningRate, weightDecay: 0f, param.Length), "TrySgdStep");
    }

    /// <summary>
    /// Plain SGD over a whole parameter list on the GPU, as <c>torch.optim.SGD</c> with <c>foreach</c>: one fused launch
    /// on a backend with a multi-tensor kernel, otherwise each tensor through the backend's own SGD kernel. The result
    /// equals calling <see cref="TrySgdStep"/> per tensor. Nothing is updated unless every parameter and gradient is
    /// GPU-resident, so a <c>false</c> leaves the whole list unchanged for the caller's CPU path.
    /// </summary>
    public static bool TrySgdStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients, float learningRate)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        if (parameters.Count != gradients.Count)
            throw new ArgumentException("Every parameter needs a gradient.", nameof(gradients));
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine gpuEngine)) return false; using var __gpuOp = gpuEngine.EnterGpuOp();
        var backend = gpuEngine.GetBackend();
        if (backend is null) return false;

        int n = parameters.Count;
        var parameterBuffers = new IGpuBuffer[n];
        var gradientBuffers = new IGpuBuffer[n];
        var sizes = new int[n];
        for (int i = 0; i < n; i++)
        {
            var param = parameters[i] ?? throw new ArgumentNullException(nameof(parameters));
            var grad = gradients[i] ?? throw new ArgumentNullException(nameof(gradients));
            if (param.Length != grad.Length)
                throw new ArgumentException($"Parameter {i} and its gradient differ in length.", nameof(gradients));
            param.EnsureOwnedForDeviceWrite();   // see TryUpdateParameter
            var pBuf = param.TryGetGpuBuffer();
            var gBuf = grad.TryGetGpuBuffer();
            if (pBuf is null || gBuf is null) return false;
            parameterBuffers[i] = pBuf;
            gradientBuffers[i] = gBuf;
            sizes[i] = param.Length;
        }

        if (backend is IMultiTensorGpuOptimizerBackend multiTensor)
            multiTensor.SgdMultiTensorUpdate(parameterBuffers, gradientBuffers, sizes, learningRate, weightDecay: 0f);
        else
            for (int i = 0; i < n; i++)
                backend.SgdUpdate(parameterBuffers[i], gradientBuffers[i], learningRate, weightDecay: 0f, sizes[i]);
        for (int i = 0; i < n; i++)
            MarkGpuUpdated(backend, parameters[i]);
        return true;
    }

    /// <summary>
    /// PyTorch's <c>torch.nn.utils.clip_grad_norm_</c> (foreach) on the device: the total L2 norm of every gradient,
    /// the coefficient <c>min(1, maxNorm / (norm + 1e-6))</c> and the in-place scale of every gradient, with no host
    /// read. A non-finite norm leaves the gradients unscaled (the kernels' coefficient is 1), so a poisoned step keeps
    /// its evidence. Returns null when the current engine is not a GPU engine, its backend has no multi-tensor kernels,
    /// a gradient is not device-resident, or <paramref name="maxNorm"/> exceeds <see cref="float.MaxValue"/>; the caller
    /// then clips on the host.
    /// </summary>
    /// <param name="gradients">The gradients, scaled in place.</param>
    /// <param name="maxNorm">The norm bound (positive).</param>
    /// <returns>The total norm, read back to the host only when a caller asks for it.</returns>
    public static DeviceGradientNorm? TryClipGradNorm(IReadOnlyList<Tensor<float>> gradients, double maxNorm)
    {
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        if (!(maxNorm > 0) || double.IsInfinity(maxNorm)) throw new ArgumentOutOfRangeException(nameof(maxNorm), "The norm bound must be positive and finite.");
        // The device kernels take the bound as a float: past float.MaxValue it would narrow to infinity and the
        // coefficient to 1, leaving a norm above the bound unclipped. Such a bound is clipped on the host, in double.
        if (maxNorm > float.MaxValue) return null;
        return DeviceSumOfSquares(gradients, (float)maxNorm);
    }

    /// <summary>
    /// The total squared L2 norm of device-resident gradients, reduced on the device (one multi-tensor launch on CUDA
    /// and HIP) and read back only when the result is read: one scalar, where a host scan downloads every gradient.
    /// A non-finite result means some element is NaN or infinite. Returns null under the same conditions as
    /// <see cref="TryClipGradNorm"/>.
    /// </summary>
    /// <remarks>The reduction accumulates squares in float. Squared magnitudes below the backend's float
    /// subnormal range can underflow to zero (and devices that flush subnormals may underflow earlier).
    /// A zero result therefore does not distinguish exactly zero gradients from underflowed tiny gradients.</remarks>
    public static DeviceGradientNorm? TryGradientSumOfSquares(IReadOnlyList<Tensor<float>> gradients)
    {
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        return DeviceSumOfSquares(gradients, clipTo: null);
    }

    private static DeviceGradientNorm? DeviceSumOfSquares(IReadOnlyList<Tensor<float>> gradients, float? clipTo)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine gpuEngine)) return null;
        using var __gpuOp = gpuEngine.EnterGpuOp();
        var backend = gpuEngine.GetBackend();
        if (backend is not IMultiTensorKernels multi) return null;

        var buffers = new List<IGpuBuffer>(gradients.Count);
        var sizes = new List<int>(gradients.Count);
        var owners = new List<Tensor<float>>(gradients.Count);
        for (int i = 0; i < gradients.Count; i++)
        {
            var g = gradients[i];
            if (g is null || g.Length == 0) continue;
            // Only a current device copy: the gradient's value on the device (a pending host download), or a buffer
            // uploaded at the gradient's present version. Anything else is host-authoritative.
            var buf = g.TryGetGpuBuffer();
            if (buf is null || !(g.HasPendingGpuData || g._gpuBufferVersion == g.GpuCacheVersion)) return null;
            buffers.Add(buf);
            sizes.Add(g.Length);
            owners.Add(g);
        }

        var sumOfSquares = backend.AllocateBuffer(2);
        if (buffers.Count == 0)
        {
            backend.Fill(sumOfSquares, 0f, 2);
            return new DeviceGradientNorm(multi, sumOfSquares, null);
        }
        IGpuBuffer? scale = null;
        try
        {
            multi.MultiTensorSumOfSquares(buffers, sizes, sumOfSquares);
            if (clipTo is { } maxNorm)
            {
                scale = backend.AllocateBuffer(1);
                multi.ClipScaleFromSumOfSquares(sumOfSquares, maxNorm, scale);
                multi.MultiTensorScaleByDeviceScalar(buffers, sizes, scale);
            }
        }
        catch
        {
            sumOfSquares.Dispose();
            scale?.Dispose();
            throw;
        }
        if (scale is not null)
        {
            // The device buffers now hold the clipped values: they are each gradient's authoritative copy.
            for (int i = 0; i < owners.Count; i++)
                gpuEngine.BindResidentBuffer(owners[i], buffers[i], backend);
        }
        // The handle keeps the scratch buffers alive past the queued launches that read them.
        return new DeviceGradientNorm(multi, sumOfSquares, scale);
    }

    /// <summary>
    /// Proximal gradient (ISTA) step with L1 soft-threshold prox, in place on the GPU.
    /// Returns false when param/grad aren't GPU-resident so the caller falls
    /// back to the CPU path.
    /// </summary>
    public static bool TryProximalL1Step(Tensor<float> p, Tensor<float> g, float lr, float l1Strength)
    {
        if (p is null) throw new ArgumentNullException(nameof(p));
        if (g is null) throw new ArgumentNullException(nameof(g));
        if (p.Length != g.Length)
            throw new ArgumentException("param and grad must have the same length.", nameof(g));
        if (float.IsNaN(lr) || float.IsInfinity(lr) || lr < 0f)
            throw new ArgumentOutOfRangeException(nameof(lr), lr, "Learning rate must be finite and non-negative.");
        if (float.IsNaN(l1Strength) || float.IsInfinity(l1Strength) || l1Strength < 0f)
            throw new ArgumentOutOfRangeException(nameof(l1Strength), l1Strength, "L1 strength must be finite and non-negative.");
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp();
        var backend = e.GetBackend();
        if (backend is null) return false;
        var pb = p.TryGetGpuBuffer(); var gb = g.TryGetGpuBuffer();
        if (pb is null || gb is null) return false;
        backend.ProximalL1Update(pb, gb, lr, l1Strength, p.Length);
        MarkGpuUpdated(backend, p);
        return true;
    }

    /// <summary>
    /// Allocates GPU-resident 8-bit Adam state (int8 m/v + per-block float
    /// scales) for a parameter of <paramref name="length"/> elements. The first
    /// Adam8Bit step treats m/v as zero state, matching the CPU Adam8BitOptimizer's
    /// m_0=v_0=0 contract. The caller keeps the four buffers resident across
    /// steps and frees them with <see cref="FreeGpuBuffer"/>. Returns false
    /// when the active GPU backend does not implement compressed optimizer
    /// moments.
    /// </summary>
    public static bool TryAllocAdam8BitState(int length, int blockSize,
        out DirectGpu.IGpuBuffer? mQ, out DirectGpu.IGpuBuffer? vQ,
        out DirectGpu.IGpuBuffer? mScales, out DirectGpu.IGpuBuffer? vScales)
    {
        mQ = vQ = mScales = vScales = null;
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp();
        if (e.GetBackend() is not ICompressedMomentGpuOptimizerBackend cb) return false;
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
        int nb = (length + blockSize - 1) / blockSize;

        // Allocate into locals first. If a later AllocateByteBuffer / AllocateBuffer
        // throws (e.g. device OOM), free whatever was already allocated so the GPU
        // buffers don't leak, leave the out-params null (no dangling references),
        // and let the exception surface — masking an OOM as a `false` "fall back to
        // CPU" would silently hide a real device failure.
        DirectGpu.IGpuBuffer? lmQ = null, lvQ = null, lmScales = null, lvScales = null;
        try
        {
            lmQ = cb.AllocateByteBuffer(length);
            lvQ = cb.AllocateByteBuffer(length);
            lmScales = cb.AllocateBuffer(new float[nb]);
            lvScales = cb.AllocateBuffer(new float[nb]);
        }
        catch
        {
            FreeGpuBuffer(lmQ); FreeGpuBuffer(lvQ); FreeGpuBuffer(lmScales); FreeGpuBuffer(lvScales);
            throw;
        }
        mQ = lmQ; vQ = lvQ; mScales = lmScales; vScales = lvScales;
        return true;
    }

    /// <summary>Frees a GPU buffer allocated by <see cref="TryAllocAdam8BitState"/> (safe on null).</summary>
    public static void FreeGpuBuffer(DirectGpu.IGpuBuffer? buffer)
    {
        if (buffer is IDisposable d) d.Dispose();
    }

    private static bool TryUpdateCompressedParameter(DirectGpuTensorEngine engine, IDirectGpuBackend backend,
        Tensor<float> param, Action<IGpuBuffer> update)
    {
        param.EnsureOwnedForDeviceWrite();   // see TryUpdateParameter
        bool flat = DirectGpuTensorEngine.IsFlatParameterView(param);
        var buffer = flat ? engine.AcquireParameterBuffer(param, out _) : param.TryGetGpuBuffer();
        if (buffer is null && flat && engine.TryStageFlatParameter(param) is { } staged)
        {
            using (staged)
            {
                update(staged.Staged);
                backend.Copy(staged.Staged, 0, staged.FlatBuffer, staged.Offset, param.Length);
            }
            MarkParameterUpdated(engine, backend, param, staged.FlatBuffer);
            return true;
        }
        if (buffer is null) return false;
        update(buffer);
        MarkParameterUpdated(engine, backend, param, buffer);
        return true;
    }


    /// <summary>
    /// GPU-resident 8-bit Adam step: dequantize, update and requantize without host moment downloads.
    /// Returns false when parameters or gradients are not resident, state is missing, or the backend lacks support.
    /// </summary>
    public static bool TryAdam8BitStep(Tensor<float> p, Tensor<float> g,
        DirectGpu.IGpuBuffer? mQ, DirectGpu.IGpuBuffer? vQ, DirectGpu.IGpuBuffer? mScales, DirectGpu.IGpuBuffer? vScales,
        float lr, float beta1, float beta2, float epsilon, float biasCorrection1, float biasCorrection2, int blockSize)
    {
        if (p is null) throw new ArgumentNullException(nameof(p));
        if (g is null) throw new ArgumentNullException(nameof(g));
        if (p.Length != g.Length)
            throw new ArgumentException("param and grad must have the same length.", nameof(g));
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
        // The four state buffers are optional inputs: a null one means the caller
        // hasn't allocated GPU-resident 8-bit Adam state, so signal CPU fallback
        // (matching the documented contract) rather than throwing.
        if (mQ is null || vQ is null || mScales is null || vScales is null) return false;
        CompressedMomentHyperparameters.Validate(lr, beta1, beta2, epsilon, 0f);
        CompressedMomentHyperparameters.ValidateInt8Factors(1f - beta1, 1f - beta2, biasCorrection1, biasCorrection2);
        int stateBlocks = HostCompressedMomentAdam.BlockCount(p.Length, blockSize);
        CompressedMomentBufferValidation.RequireCapacity(mQ, p.Length, nameof(mQ));
        CompressedMomentBufferValidation.RequireCapacity(vQ, p.Length, nameof(vQ));
        CompressedMomentBufferValidation.RequireCapacity(mScales, (long)stateBlocks * sizeof(float), nameof(mScales));
        CompressedMomentBufferValidation.RequireCapacity(vScales, (long)stateBlocks * sizeof(float), nameof(vScales));
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp();
        if (e.GetBackend() is not ICompressedMomentGpuOptimizerBackend cb) return false;
        var gb = g.TryGetGpuBuffer();
        if (gb is null) return false;
        int nb = HostCompressedMomentAdam.BlockCount(p.Length, blockSize);
        return TryUpdateCompressedParameter(e, cb, p, pb => cb.Adam8BitUpdate(pb, gb, mQ, vQ, mScales, vScales,
            lr, beta1, beta2, epsilon, 1f - beta1, 1f - beta2, biasCorrection1, biasCorrection2, blockSize, p.Length, nb));
    }

    /// <summary>
    /// <see cref="TryAdam8BitStep(Tensor{float}, Tensor{float}, DirectGpu.IGpuBuffer?, DirectGpu.IGpuBuffer?, DirectGpu.IGpuBuffer?, DirectGpu.IGpuBuffer?, float, float, float, float, float, float, int)"/>
    /// with weight decay inside the same kernel: L2 decay added to the gradient (Adam, <c>torch.optim.Adam</c>'s
    /// <c>weight_decay</c>) or AdamW's decoupled decay (<c>param *= 1 - lr * weightDecay</c> first, as
    /// <c>torch.optim.AdamW</c> and bitsandbytes' <c>AdamW8bit</c>). Returns false (CPU fallback) on the same
    /// conditions, and when the backend has no weight-decay int8 kernel and <paramref name="weightDecay"/> is not 0.
    /// </summary>
    public static bool TryAdam8BitStep(Tensor<float> p, Tensor<float> g,
        DirectGpu.IGpuBuffer? mQ, DirectGpu.IGpuBuffer? vQ, DirectGpu.IGpuBuffer? mScales, DirectGpu.IGpuBuffer? vScales,
        float lr, float beta1, float beta2, float epsilon, float biasCorrection1, float biasCorrection2, int blockSize,
        float weightDecay, bool decoupledWeightDecay)
    {
        if (p is null) throw new ArgumentNullException(nameof(p));
        if (g is null) throw new ArgumentNullException(nameof(g));
        if (p.Length != g.Length)
            throw new ArgumentException("param and grad must have the same length.", nameof(g));
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (mQ is null || vQ is null || mScales is null || vScales is null) return false;
        CompressedMomentHyperparameters.Validate(lr, beta1, beta2, epsilon, weightDecay);
        CompressedMomentHyperparameters.ValidateInt8Factors(1f - beta1, 1f - beta2, biasCorrection1, biasCorrection2);
        int stateBlocks = HostCompressedMomentAdam.BlockCount(p.Length, blockSize);
        CompressedMomentBufferValidation.RequireCapacity(mQ, p.Length, nameof(mQ));
        CompressedMomentBufferValidation.RequireCapacity(vQ, p.Length, nameof(vQ));
        CompressedMomentBufferValidation.RequireCapacity(mScales, (long)stateBlocks * sizeof(float), nameof(mScales));
        CompressedMomentBufferValidation.RequireCapacity(vScales, (long)stateBlocks * sizeof(float), nameof(vScales));
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp();
        var backend = e.GetBackend();
        if (backend is not IAdam8BitWeightDecayBackend db)
            return weightDecay == 0f
                && TryAdam8BitStep(p, g, mQ, vQ, mScales, vScales, lr, beta1, beta2, epsilon, biasCorrection1, biasCorrection2, blockSize);
        var gb = g.TryGetGpuBuffer();
        if (gb is null) return false;
        int nb = HostCompressedMomentAdam.BlockCount(p.Length, blockSize);
        return TryUpdateCompressedParameter(e, db, p, pb => db.Adam8BitUpdate(pb, gb, mQ, vQ, mScales, vScales,
            lr, beta1, beta2, epsilon, 1f - beta1, 1f - beta2, biasCorrection1, biasCorrection2, blockSize, p.Length, nb,
            weightDecay, decoupledWeightDecay));
    }

    /// <summary>
    /// Allocates device bfloat16 Adam moment state for <paramref name="length"/> elements: <paramref name="m"/> and
    /// <paramref name="v"/> are two bytes per element (half of fp32 moments) for
    /// <see cref="TryAdamBf16Step"/>. Returns false, with both null, when the active engine is not a direct GPU engine
    /// or its backend has no compressed-moment kernels. A device allocation failure is thrown, not masked.
    /// </summary>
    public static bool TryAllocBf16MomentState(int length, out DirectGpu.IGpuBuffer? m, out DirectGpu.IGpuBuffer? v)
    {
        m = v = null;
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp();
        if (e.GetBackend() is not ICompressedMomentGpuOptimizerBackend cb) return false;
        DirectGpu.IGpuBuffer? lm = null, lv = null;
        try
        {
            lm = cb.AllocateByteBuffer(length * sizeof(ushort));
            lv = cb.AllocateByteBuffer(length * sizeof(ushort));
        }
        catch
        {
            FreeGpuBuffer(lm); FreeGpuBuffer(lv);
            throw;
        }
        m = lm; v = lv;
        return true;
    }

    /// <summary>
    /// GPU-resident Adam (<paramref name="decoupledWeightDecay"/> false: L2 decay into the gradient) or AdamW
    /// (true: decoupled decay) step with bfloat16 moments from <see cref="TryAllocBf16MomentState"/>: the update math
    /// is fp32 and the new moments are stored round-to-nearest-even. <paramref name="step"/> is the 1-based step the
    /// bias corrections use; step 1 ignores the stored moments. Returns false (CPU fallback) when param/grad are not
    /// GPU-resident, a moment buffer is null, or the backend has no compressed-moment kernels.
    /// </summary>
    public static bool TryAdamBf16Step(Tensor<float> p, Tensor<float> g, DirectGpu.IGpuBuffer? m, DirectGpu.IGpuBuffer? v,
        float lr, float beta1, float beta2, float epsilon, float weightDecay, int step, bool decoupledWeightDecay)
    {
        if (p is null) throw new ArgumentNullException(nameof(p));
        if (g is null) throw new ArgumentNullException(nameof(g));
        if (p.Length != g.Length)
            throw new ArgumentException("param and grad must have the same length.", nameof(g));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step), step, "The Adam step is 1-based.");
        if (m is null || v is null) return false;
        CompressedMomentHyperparameters.Validate(lr, beta1, beta2, epsilon, weightDecay);
        CompressedMomentBufferValidation.RequireCapacity(m, (long)p.Length * sizeof(ushort), nameof(m));
        CompressedMomentBufferValidation.RequireCapacity(v, (long)p.Length * sizeof(ushort), nameof(v));
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp();
        if (e.GetBackend() is not ICompressedMomentGpuOptimizerBackend cb) return false;
        var gb = g.TryGetGpuBuffer();
        if (gb is null) return false;
        return TryUpdateCompressedParameter(e, cb, p, pb =>
        {
            if (decoupledWeightDecay)
                cb.AdamWUpdateBf16(pb, gb, m, v, lr, beta1, beta2, epsilon, weightDecay, step, p.Length);
            else
                cb.AdamUpdateBf16(pb, gb, m, v, lr, beta1, beta2, epsilon, weightDecay, step, p.Length);
        });
    }

    // ---- GPU-resident steps for the rest of the backend optimizer kernels (same contract as TryAdamStep:
    //      returns false if not on a GPU engine or any tensor isn't GPU-resident, so the caller falls back to CPU).
    //      All buffers must be GPU-resident; the kernel updates param + state IN PLACE with no host download. ----

    /// <summary>Plain SGD with L2-coupled weight decay (<c>grad + weightDecay·param</c>), in place on the GPU.</summary>
    internal static bool TrySgdStep(Tensor<float> p, Tensor<float> g, float lr, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g.TryGetGpuBuffer();
        if (gb is null) return false;
        return TryUpdateParameter(e, b, p, pb => b.SgdUpdate(pb, gb, lr, weightDecay, p.Length), "TrySgdStep");
    }

    /// <summary>
    /// Downloads a device-resident tensor's elements, or returns null when it has no device buffer. The backend may
    /// round a buffer up, so exactly <c>tensor.Length</c> elements are returned.
    /// </summary>
    internal static float[]? TryDownload(Tensor<float> tensor)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return null; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return null;
        var buffer = tensor.TryGetGpuBuffer();
        if (buffer is null) return null;
        var full = b.DownloadBuffer(buffer);
        if (full.Length == tensor.Length) return full;
        var exact = new float[tensor.Length];
        Array.Copy(full, exact, exact.Length);
        return exact;
    }

    /// <summary>Overwrites a device-resident tensor's elements with <paramref name="data"/> and marks it current.</summary>
    internal static bool TryUpload(Tensor<float> tensor, float[] data)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (data.Length < tensor.Length)
            throw new ArgumentException($"data has {data.Length} elements but the tensor has {tensor.Length}.", nameof(data));
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var buffer = tensor.TryGetGpuBuffer();
        if (buffer is null) return false;
        using (var staged = b.AllocateBuffer(data))
            b.Copy(staged, buffer, tensor.Length);
        MarkGpuUpdated(b, tensor);
        return true;
    }

    public static bool TrySgdMomentumStep(Tensor<float> p, Tensor<float> g, Tensor<float> velocity, float lr, float momentum, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g?.TryGetGpuBuffer(); var vb = velocity?.TryGetGpuBuffer();
        if (p is null || gb is null || vb is null) return false;
        if (!TryUpdateParameter(e, b, p, pb => b.SgdMomentumUpdate(pb, gb, vb, lr, momentum, weightDecay, p.Length), "TrySgdMomentumStep")) return false;
        MarkGpuUpdated(b, velocity);
        return true;
    }

    public static bool TryRmspropStep(Tensor<float> p, Tensor<float> g, Tensor<float> squaredAvg, float lr, float rho, float epsilon, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g?.TryGetGpuBuffer(); var sb = squaredAvg?.TryGetGpuBuffer();
        if (p is null || gb is null || sb is null) return false;
        if (!TryUpdateParameter(e, b, p, pb => b.RmspropUpdate(pb, gb, sb, lr, rho, epsilon, weightDecay, p.Length), "TryRmspropStep")) return false;
        MarkGpuUpdated(b, squaredAvg);
        return true;
    }

    public static bool TryAdagradStep(Tensor<float> p, Tensor<float> g, Tensor<float> accum, float lr, float epsilon, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g?.TryGetGpuBuffer(); var ab = accum?.TryGetGpuBuffer();
        if (p is null || gb is null || ab is null) return false;
        if (!TryUpdateParameter(e, b, p, pb => b.AdagradUpdate(pb, gb, ab, lr, epsilon, weightDecay, p.Length), "TryAdagradStep")) return false;
        MarkGpuUpdated(b, accum);
        return true;
    }

    public static bool TryNagStep(Tensor<float> p, Tensor<float> g, Tensor<float> velocity, float lr, float momentum, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g?.TryGetGpuBuffer(); var vb = velocity?.TryGetGpuBuffer();
        if (p is null || gb is null || vb is null) return false;
        if (!TryUpdateParameter(e, b, p, pb => b.NagUpdate(pb, gb, vb, lr, momentum, weightDecay, p.Length), "TryNagStep")) return false;
        MarkGpuUpdated(b, velocity);
        return true;
    }

    public static bool TryLarsStep(Tensor<float> p, Tensor<float> g, Tensor<float> velocity, float lr, float momentum, float weightDecay, float trustCoeff)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g?.TryGetGpuBuffer(); var vb = velocity?.TryGetGpuBuffer();
        if (p is null || gb is null || vb is null) return false;
        if (!TryUpdateParameter(e, b, p, pb => b.LarsUpdate(pb, gb, vb, lr, momentum, weightDecay, trustCoeff, p.Length), "TryLarsStep")) return false;
        MarkGpuUpdated(b, velocity);
        return true;
    }

    public static bool TryLambStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, Tensor<float> v, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step,
        float maxTrustRatio = 0f, bool biasCorrection = true)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g?.TryGetGpuBuffer(); var mb = m?.TryGetGpuBuffer(); var vb = v?.TryGetGpuBuffer();
        if (p is null || gb is null || mb is null || vb is null) return false;
        if (!TryUpdateParameter(e, b, p, pb => b.LambUpdate(pb, gb, mb, vb, lr, beta1, beta2, epsilon, weightDecay, step, p.Length, maxTrustRatio, biasCorrection), "TryLambStep")) return false;
        MarkGpuUpdated(b, m, v);
        return true;
    }

    public static bool TryAdadeltaStep(Tensor<float> p, Tensor<float> g, Tensor<float> accumGrad, Tensor<float> accumUpdate, float rho, float epsilon, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g?.TryGetGpuBuffer(); var ag = accumGrad?.TryGetGpuBuffer(); var au = accumUpdate?.TryGetGpuBuffer();
        if (p is null || gb is null || ag is null || au is null) return false;
        if (!TryUpdateParameter(e, b, p, pb => b.AdadeltaUpdate(pb, gb, ag, au, rho, epsilon, weightDecay, p.Length), "TryAdadeltaStep")) return false;
        MarkGpuUpdated(b, accumGrad, accumUpdate);
        return true;
    }

    public static bool TryAmsgradStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, Tensor<float> v, Tensor<float> vMax, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g?.TryGetGpuBuffer(); var mb = m?.TryGetGpuBuffer(); var vb = v?.TryGetGpuBuffer(); var xb = vMax?.TryGetGpuBuffer();
        if (p is null || gb is null || mb is null || vb is null || xb is null) return false;
        if (!TryUpdateParameter(e, b, p, pb => b.AmsgradUpdate(pb, gb, mb, vb, xb, lr, beta1, beta2, epsilon, weightDecay, step, p.Length), "TryAmsgradStep")) return false;
        MarkGpuUpdated(b, m, v, vMax);
        return true;
    }

    public static bool TryAdamaxStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, Tensor<float> u, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g?.TryGetGpuBuffer(); var mb = m?.TryGetGpuBuffer(); var ub = u?.TryGetGpuBuffer();
        if (p is null || gb is null || mb is null || ub is null) return false;
        if (!TryUpdateParameter(e, b, p, pb => b.AdamaxUpdate(pb, gb, mb, ub, lr, beta1, beta2, epsilon, weightDecay, step, p.Length), "TryAdamaxStep")) return false;
        MarkGpuUpdated(b, m, u);
        return true;
    }

    public static bool TryLionStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, float lr, float beta1, float beta2, float weightDecay)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g?.TryGetGpuBuffer(); var mb = m?.TryGetGpuBuffer();
        if (p is null || gb is null || mb is null) return false;
        if (!TryUpdateParameter(e, b, p, pb => b.LionUpdate(pb, gb, mb, lr, beta1, beta2, weightDecay, p.Length), "TryLionStep")) return false;
        MarkGpuUpdated(b, m);
        return true;
    }

    public static bool TryNadamStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, Tensor<float> v, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g?.TryGetGpuBuffer(); var mb = m?.TryGetGpuBuffer(); var vb = v?.TryGetGpuBuffer();
        if (p is null || gb is null || mb is null || vb is null) return false;
        if (!TryUpdateParameter(e, b, p, pb => b.NadamUpdate(pb, gb, mb, vb, lr, beta1, beta2, epsilon, weightDecay, step, p.Length), "TryNadamStep")) return false;
        MarkGpuUpdated(b, m, v);
        return true;
    }

    public static bool TryFtrlStep(Tensor<float> p, Tensor<float> g, Tensor<float> z, Tensor<float> n, float lr, float l1Reg, float l2Reg, float beta)
    {
        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var gb = g?.TryGetGpuBuffer(); var zb = z?.TryGetGpuBuffer(); var nb = n?.TryGetGpuBuffer();
        if (p is null || gb is null || zb is null || nb is null) return false;
        if (!TryUpdateParameter(e, b, p, pb => b.FtrlUpdate(pb, gb, zb, nb, lr, l1Reg, l2Reg, beta, p.Length), "TryFtrlStep")) return false;
        MarkGpuUpdated(b, z, n);
        return true;
    }

    // ==================================================================
    // SPARSE-AWARE GPU OPTIMIZER WRAPPERS — PR #567
    // ==================================================================
    //
    // Each wrapper below dispatches the matching native backend sparse optimizer
    // kernel. Grid dim = ceil(nnz / 256); each kernel
    // thread reads exactly one (idx[k], val[k]) and scatter-updates only
    // (param[idx], state[idx]). Memory traffic is O(nnz) — the other
    // (param.Length - nnz) entries are never read or written on native
    // scatter backends. Staged fallback backends preserve correctness but
    // may transfer full param/state buffers.
    //
    // sparseIndices and sparseValues must already be GPU-resident
    // (TryGetGpuBuffer must succeed for them). sparseIndices[0..nnz) must
    // be in range, unique, and pre-aggregated; native kernels intentionally
    // use non-atomic scatter writes and do not define duplicate-index order.
    // The dense state buffers (m, v, accum, ...) keep their full shape and
    // are mutated in place at the touched indices.
    //
    // Returns false on:
    //   * non-GPU engine (caller should run CPU sparse path)
    //   * any tensor missing its GPU buffer
    //   * backend NotSupportedException
    // ==================================================================

    private static bool TryPrepareSparse(
        Tensor<float> param, Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        out IDirectGpuBackend? backend, out IGpuBuffer? pBuf, out IGpuBuffer? idxBuf, out IGpuBuffer? valBuf,
        out GpuOpGate.Lease lease)
    {
        backend = null; pBuf = null; idxBuf = null; valBuf = null; lease = default;
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (sparseIndices is null) throw new ArgumentNullException(nameof(sparseIndices));
        if (sparseValues is null) throw new ArgumentNullException(nameof(sparseValues));
        if (nnz < 0) throw new ArgumentOutOfRangeException(nameof(nnz));
        if (nnz > sparseIndices.Length || nnz > sparseValues.Length)
            throw new ArgumentException("nnz exceeds the supplied sparse buffers.", nameof(nnz));

        if (!(AiDotNetEngine.Current is DirectGpuTensorEngine engine)) return false;
        // The caller dispatches on the backend after this returns, so the lease goes to the caller (released here
        // only when preparation fails).
        lease = engine.EnterGpuOp();
        bool prepared = false;
        try
        {
            backend = engine.GetBackend();
            if (backend is null) return false;
            param.EnsureOwnedForDeviceWrite();   // see TryUpdateParameter
            pBuf = param.TryGetGpuBuffer();
            idxBuf = sparseIndices.TryGetGpuBuffer();
            valBuf = sparseValues.TryGetGpuBuffer();
            if (pBuf is null || idxBuf is null || valBuf is null) return false;

            // Cheap, always-on device-buffer bounds guards: the kernels read indices[k] AND values[k] for
            // k < nnz, so both buffers must hold at least nnz elements or the dispatch reads out of bounds.
            if (idxBuf.Size < nnz)
                throw new ArgumentException($"sparseIndices buffer holds {idxBuf.Size} elements but nnz={nnz}.", nameof(sparseIndices));
            if (valBuf.Size < nnz)
                throw new ArgumentException($"sparseValues buffer holds {valBuf.Size} elements but nnz={nnz}.", nameof(sparseValues));

            EnsureUniqueSparseDispatchIndices(backend, idxBuf, nnz, param.Length);
            prepared = true;
            return true;
        }
        finally
        {
            if (!prepared)
            {
                lease.Dispose();
                lease = default;
            }
        }
    }

    /// <summary>
    /// Opt-in gate for the O(nnz) host-side duplicate-index verification in
    /// <see cref="EnsureUniqueSparseDispatchIndices"/>. The native sparse kernels are non-atomic and
    /// require unique, pre-aggregated indices per dispatch (see the <c>IDirectGpuBackend</c> sparse
    /// optimizer contract), but downloading the index buffer and building a <see cref="HashSet{T}"/> on
    /// every step forces a host round-trip that can serialize the GPU queue and defeats the O(nnz)
    /// on-device benefit. Off by default (release/native fast path — callers own the uniqueness
    /// invariant); enable via <c>AIDOTNET_VALIDATE_SPARSE_INDICES=1</c> for debugging or staged fallback.
    /// </summary>
    internal static bool ValidateSparseIndexUniqueness { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_VALIDATE_SPARSE_INDICES") == "1";

    internal static void EnsureUniqueSparseDispatchIndices(IDirectGpuBackend backend, IGpuBuffer sparseIndices, int nnz, int paramLength)
    {
        if (nnz == 0) return;
        // Always-on cheap bounds guard (no host transfer).
        if (sparseIndices.Size < nnz)
            throw new ArgumentException($"sparseIndices buffer holds {sparseIndices.Size} elements but nnz={nnz}.", nameof(sparseIndices));

        // The duplicate/uniqueness scan below needs a DownloadBuffer round-trip + a HashSet — O(nnz)
        // off-device work on every sparse step. Skip it on the fast path; the native kernels' unique-index
        // requirement is part of the backend contract and callers pre-aggregate. Opt in for verification.
        if (!ValidateSparseIndexUniqueness) return;

        var rawIndices = backend.DownloadBuffer(sparseIndices);
        var seen = new HashSet<int>();
        for (int k = 0; k < nnz; k++)
        {
            int index;
            try
            {
                index = SparseOptimizerReference.DecodeIndex(rawIndices[k], paramLength);
            }
            catch (ArgumentOutOfRangeException)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(sparseIndices),
                    rawIndices[k],
                    $"sparseIndices[{k}] is outside [0, {paramLength}).");
            }

            if (!seen.Add(index))
                throw new ArgumentException(
                    $"sparseIndices contains duplicate index {index} at position {k}; GPU sparse optimizer indices must be unique and pre-aggregated before dispatch.",
                    nameof(sparseIndices));
        }
    }

    /// <summary>Adam GPU step driven by a sparse gradient — dispatches the native
    /// sparse_adam_update kernel (one thread per nnz, scatter-update). Returns false
    /// when any tensor isn't GPU-resident or the backend has no native sparse kernel.</summary>
    public static bool TryAdamStepSparse(
        Tensor<float> param, Tensor<float> m, Tensor<float> v,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
    {
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        var mb = m.TryGetGpuBuffer(); var vbState = v.TryGetGpuBuffer();
        if (mb is null || vbState is null) return false;
        try { b!.SparseAdamUpdate(pb!, mb, vbState, iBuf!, vBuf!, nnz, learningRate, beta1, beta2, epsilon, weightDecay, step); MarkGpuUpdated(b, param, m, v); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>AdamW GPU step driven by a sparse gradient (native sparse_adamw_update kernel).</summary>
    public static bool TryAdamWStepSparse(
        Tensor<float> param, Tensor<float> m, Tensor<float> v,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
    {
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        var mb = m.TryGetGpuBuffer(); var vbState = v.TryGetGpuBuffer();
        if (mb is null || vbState is null) return false;
        try { b!.SparseAdamWUpdate(pb!, mb, vbState, iBuf!, vBuf!, nnz, learningRate, beta1, beta2, epsilon, weightDecay, step); MarkGpuUpdated(b, param, m, v); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>SGD GPU step driven by a sparse gradient (native sparse_sgd_update kernel).</summary>
    public static bool TrySgdStepSparse(
        Tensor<float> param,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float learningRate, float weightDecay = 0f)
    {
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        try { b!.SparseSgdUpdate(pb!, iBuf!, vBuf!, nnz, learningRate, weightDecay); MarkGpuUpdated(b, param); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>SGD-momentum GPU step driven by a sparse gradient.</summary>
    public static bool TrySgdMomentumStepSparse(
        Tensor<float> param, Tensor<float> velocity,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float learningRate, float momentum, float weightDecay)
    {
        if (velocity is null) throw new ArgumentNullException(nameof(velocity));
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        var velBuf = velocity.TryGetGpuBuffer();
        if (velBuf is null) return false;
        try { b!.SparseSgdMomentumUpdate(pb!, velBuf, iBuf!, vBuf!, nnz, learningRate, momentum, weightDecay); MarkGpuUpdated(b, param, velocity); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>RMSProp GPU step driven by a sparse gradient.</summary>
    public static bool TryRmspropStepSparse(
        Tensor<float> param, Tensor<float> squaredAvg,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float rho, float epsilon, float weightDecay)
    {
        if (squaredAvg is null) throw new ArgumentNullException(nameof(squaredAvg));
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        var sb = squaredAvg.TryGetGpuBuffer();
        if (sb is null) return false;
        try { b!.SparseRmspropUpdate(pb!, sb, iBuf!, vBuf!, nnz, lr, rho, epsilon, weightDecay); MarkGpuUpdated(b, param, squaredAvg); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>Adagrad GPU step driven by a sparse gradient.</summary>
    public static bool TryAdagradStepSparse(
        Tensor<float> param, Tensor<float> accum,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float epsilon, float weightDecay)
    {
        if (accum is null) throw new ArgumentNullException(nameof(accum));
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        var ab = accum.TryGetGpuBuffer();
        if (ab is null) return false;
        try { b!.SparseAdagradUpdate(pb!, ab, iBuf!, vBuf!, nnz, lr, epsilon, weightDecay); MarkGpuUpdated(b, param, accum); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>NAG GPU step driven by a sparse gradient.</summary>
    public static bool TryNagStepSparse(
        Tensor<float> param, Tensor<float> velocity,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float momentum, float weightDecay)
    {
        if (velocity is null) throw new ArgumentNullException(nameof(velocity));
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        var velBuf = velocity.TryGetGpuBuffer();
        if (velBuf is null) return false;
        try { b!.SparseNagUpdate(pb!, velBuf, iBuf!, vBuf!, nnz, lr, momentum, weightDecay); MarkGpuUpdated(b, param, velocity); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>AdaDelta GPU step driven by a sparse gradient.</summary>
    public static bool TryAdadeltaStepSparse(
        Tensor<float> param, Tensor<float> accumGrad, Tensor<float> accumUpdate,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float rho, float epsilon, float weightDecay)
    {
        if (accumGrad is null) throw new ArgumentNullException(nameof(accumGrad));
        if (accumUpdate is null) throw new ArgumentNullException(nameof(accumUpdate));
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        var agb = accumGrad.TryGetGpuBuffer();
        var aub = accumUpdate.TryGetGpuBuffer();
        if (agb is null || aub is null) return false;
        try { b!.SparseAdadeltaUpdate(pb!, agb, aub, iBuf!, vBuf!, nnz, rho, epsilon, weightDecay); MarkGpuUpdated(b, param, accumGrad, accumUpdate); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>AMSGrad GPU step driven by a sparse gradient.</summary>
    public static bool TryAmsgradStepSparse(
        Tensor<float> param, Tensor<float> m, Tensor<float> v, Tensor<float> vMax,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (vMax is null) throw new ArgumentNullException(nameof(vMax));
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        var mb = m.TryGetGpuBuffer(); var vbState = v.TryGetGpuBuffer(); var vmb = vMax.TryGetGpuBuffer();
        if (mb is null || vbState is null || vmb is null) return false;
        try { b!.SparseAmsgradUpdate(pb!, mb, vbState, vmb, iBuf!, vBuf!, nnz, lr, beta1, beta2, epsilon, weightDecay, step); MarkGpuUpdated(b, param, m, v, vMax); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>Adamax GPU step driven by a sparse gradient.</summary>
    public static bool TryAdamaxStepSparse(
        Tensor<float> param, Tensor<float> m, Tensor<float> u,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (u is null) throw new ArgumentNullException(nameof(u));
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        var mb = m.TryGetGpuBuffer(); var ub = u.TryGetGpuBuffer();
        if (mb is null || ub is null) return false;
        try { b!.SparseAdamaxUpdate(pb!, mb, ub, iBuf!, vBuf!, nnz, lr, beta1, beta2, epsilon, weightDecay, step); MarkGpuUpdated(b, param, m, u); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>Lion GPU step driven by a sparse gradient.</summary>
    public static bool TryLionStepSparse(
        Tensor<float> param, Tensor<float> m,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float beta1, float beta2, float weightDecay)
    {
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        var mb = m.TryGetGpuBuffer();
        if (mb is null) return false;
        try { b!.SparseLionUpdate(pb!, mb, iBuf!, vBuf!, nnz, lr, beta1, beta2, weightDecay); MarkGpuUpdated(b, param, m); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>Nadam GPU step driven by a sparse gradient.</summary>
    public static bool TryNadamStepSparse(
        Tensor<float> param, Tensor<float> m, Tensor<float> v,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        var mb = m.TryGetGpuBuffer(); var vbState = v.TryGetGpuBuffer();
        if (mb is null || vbState is null) return false;
        try { b!.SparseNadamUpdate(pb!, mb, vbState, iBuf!, vBuf!, nnz, lr, beta1, beta2, epsilon, weightDecay, step); MarkGpuUpdated(b, param, m, v); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>FTRL GPU step driven by a sparse gradient.</summary>
    public static bool TryFtrlStepSparse(
        Tensor<float> param, Tensor<float> z, Tensor<float> n,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float l1Reg, float l2Reg, float beta)
    {
        if (z is null) throw new ArgumentNullException(nameof(z));
        if (n is null) throw new ArgumentNullException(nameof(n));
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        var zb = z.TryGetGpuBuffer(); var nb = n.TryGetGpuBuffer();
        if (zb is null || nb is null) return false;
        try { b!.SparseFtrlUpdate(pb!, zb, nb, iBuf!, vBuf!, nnz, lr, l1Reg, l2Reg, beta); MarkGpuUpdated(b, param, z, n); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>Proximal-L1 GPU step driven by a sparse gradient.</summary>
    public static bool TryProximalL1StepSparse(
        Tensor<float> param,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float l1Strength)
    {
        if (!TryPrepareSparse(param, sparseIndices, sparseValues, nnz, out var b, out var pb, out var iBuf, out var vBuf, out var lease)) return false;
        using var __gpuOp = lease;
        try { b!.SparseProximalL1Update(pb!, iBuf!, vBuf!, nnz, lr, l1Strength); MarkGpuUpdated(b, param); return true; }
        catch (NotSupportedException) { return false; }
    }
}
