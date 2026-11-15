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
        if (tensor._gpuBuffer is { } attachedBuf && attachedBuf.Handle != IntPtr.Zero && tensor.OffloadDevicePointer == IntPtr.Zero
            && tensor.GetBackingArrayForCacheLookupUnsafe() is { } backingArray)
        {
            Helpers.HostSync.Remove(backingArray);
            Helpers.ResidentHostMirror.Attach(tensor);
        }
        else if (tensor._gpuMaterializerKey is { } materializerKey)
        {
            Helpers.HostSync.Remove(materializerKey);
        }
        tensor.MarkModified(CreateWriteSyncPoint(backend));
        tensor._gpuBufferVersion = tensor.GpuCacheVersion;
    }

    /// <summary>
    /// Returns an already-bound device buffer only when its contents match <paramref name="tensor"/>'s current version
    /// (or resident-step capture is active) or come from a <c>GpuPinned</c>/<c>GpuOffload</c> mirror. Rejects buffers made
    /// stale by an intervening host mutation (e.g., host gradient clipping or checkpoint restore), even when the tensor's
    /// logical <see cref="TensorBase{T}.Device"/> still reports GPU.
    /// </summary>
    private static IGpuBuffer? TryGetCurrentGpuBuffer(
        DirectGpuTensorEngine engine, IDirectGpuBackend backend, Tensor<float> tensor)
    {
        if (tensor._gpuBuffer is null && tensor.OffloadDevicePointer != IntPtr.Zero)
            return tensor.TryGetGpuBuffer();
        if (tensor._gpuBackend is not null && !ReferenceEquals(tensor._gpuBackend, backend))
            return null;
        if (!(engine.ResidentStepActive || tensor._gpuBufferVersion == tensor.GpuCacheVersion))
            return null;
        var buf = tensor.TryGetGpuBuffer();
        return buf is not null && buf.Handle != IntPtr.Zero ? buf : null;
    }

    private static IGpuBuffer? TryGetCurrentGpuBufferInt(
        DirectGpuTensorEngine engine, IDirectGpuBackend backend, Tensor<int> tensor)
    {
        if (tensor._gpuBuffer is null && tensor.OffloadDevicePointer != IntPtr.Zero)
            return tensor.TryGetGpuBuffer();
        if (tensor._gpuBackend is not null && !ReferenceEquals(tensor._gpuBackend, backend))
            return null;
        if (!(engine.ResidentStepActive || tensor._gpuBufferVersion == tensor.GpuCacheVersion))
            return null;
        var buf = tensor.TryGetGpuBuffer();
        return buf is not null && buf.Handle != IntPtr.Zero ? buf : null;
    }

    /// <summary>
    /// Resolves a read-only float gradient/value buffer of at least <paramref name="requiredLength"/> elements. Reuses a
    /// current device buffer when available; otherwise uploads a contiguous host tensor into a stream-ordered transient
    /// scratch slice without mutating the source tensor's version or residency metadata.
    /// </summary>
    private static (IGpuBuffer Buffer, DirectGpuTensorEngine.TransientDeviceSlice Slice)? TryAcquireReadOnlyFloatBuffer(
        DirectGpuTensorEngine engine, IDirectGpuBackend backend, Tensor<float> tensor, int requiredLength,
        bool enforceSizeOnExisting = true)
    {
        if (tensor.Length < requiredLength) return null;
        var existing = TryGetCurrentGpuBuffer(engine, backend, tensor);
        if (existing is not null && (!enforceSizeOnExisting || existing.Size >= requiredLength))
            return (existing, default);
        if (engine.TryUploadTransientFloat(backend, tensor, requiredLength, out var transientSlice))
            return (transientSlice.Buffer, transientSlice);
        return null;
    }

    private static (IGpuBuffer Buffer, DirectGpuTensorEngine.TransientDeviceSlice Slice)? TryAcquireReadOnlyIntBuffer(
        DirectGpuTensorEngine engine, IDirectGpuBackend backend, Tensor<int> tensor, int requiredLength,
        bool enforceSizeOnExisting = true)
    {
        if (tensor.Length < requiredLength) return null;
        var existing = TryGetCurrentGpuBufferInt(engine, backend, tensor);
        if (existing is not null && (!enforceSizeOnExisting || existing.Size >= requiredLength))
            return (existing, default);
        if (engine.TryUploadTransientInt(backend, tensor, requiredLength, out var transientSlice))
            return (transientSlice.Buffer, transientSlice);
        return null;
    }

    /// <summary>
    /// Resolves a mutable optimizer state buffer (<c>m</c>, <c>v</c>, <c>velocity</c>, etc.) of at least
    /// <paramref name="requiredLength"/> elements. Reuses a current device buffer, syncs <c>GpuPinned</c> offload memory,
    /// or promotes/refreshes a contiguous host state tensor onto the device in place so step-1 execution and post-restore
    /// stepping work out-of-the-box without host fallbacks.
    /// </summary>
    private static IGpuBuffer? TryResolveStateBuffer(
        DirectGpuTensorEngine engine, IDirectGpuBackend backend, Tensor<float>? state, int requiredLength)
    {
        if (state is null || state.Length < requiredLength) return null;
        state.EnsureOwnedForDeviceWrite();
        if (state._gpuBuffer is null && state.OffloadDevicePointer != IntPtr.Zero)
        {
            var offload = state.TryGetGpuBuffer();
            return offload is not null && offload.Handle != IntPtr.Zero && offload.Size >= requiredLength
                ? offload
                : null;
        }
        var current = TryGetCurrentGpuBuffer(engine, backend, state);
        if (current is not null && current.Size >= requiredLength)
            return current;
        if (!state.IsContiguous || state._storageOffset != 0)
            return null;
        var hostData = state.GetCpuBackingForStridedRead(out int offset);
        float[] uploadData;
        if (hostData is not null)
        {
            if (offset != 0 || hostData.Length != state.Length)
                return null;
            uploadData = hostData;
        }
        else if (state._gpuBuffer is { } gpuBuf && gpuBuf.Handle != IntPtr.Zero && state._device != TensorDevice.CPU)
        {
            uploadData = state.AsSpan().ToArray();
        }
        else
        {
            return null;
        }
        try
        {
            if (state._gpuBuffer is { } existing && existing.Handle != IntPtr.Zero
                && ReferenceEquals(state._gpuBackend, backend)
                && existing.Size >= state.Length)
            {
                backend.UploadBufferInPlace(uploadData, existing);
                engine.BindResidentBuffer(state, existing, backend, isDeviceModified: false);
                return existing;
            }
            if (state._ownsGpuBuffer && state._gpuBuffer is { } oldOwned)
            {
                state._gpuBuffer = null;
                state._ownsGpuBuffer = false;
                oldOwned.Dispose();
            }
            var allocated = backend.AllocateBuffer(uploadData);
            state._ownsGpuBuffer = true;
            engine.BindResidentBuffer(state, allocated, backend, isDeviceModified: false);
            return allocated;
        }
        catch (NotSupportedException ex)
        {
            GpuLaunchProbe.OnFallback("TryResolveStateBuffer", ex);
            return null;
        }
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
        if (AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var gpuEngine)
            && gpuEngine.TryCreateDeviceStateTensor(shape) is { } state)
            return TensorScope.Detach(state);
        return TensorScope.Detach(new Tensor<float>(shape));
    }

    /// <summary>
    /// Marks a parameter updated in place on the device. A parameter with a host backing array becomes
    /// device-authoritative: its persistent weight entry follows the new version and a host read downloads it lazily.
    /// A GpuPinned/offload tensor keeps its own mirror protocol instead: binding the wrapped device pointer would make
    /// <see cref="TensorBase{T}.TryGetGpuBuffer"/> return it without first uploading host writes, so a write made
    /// after a step (an optimizer state loaded mid-training) never reached the next kernel.
    /// </summary>
    private static void MarkParameterUpdated(DirectGpuTensorEngine engine, IDirectGpuBackend backend, Tensor<float> param, IGpuBuffer buffer)
    {
        if (param._gpuBuffer is not null ||
            (param.OffloadDevicePointer == IntPtr.Zero && param.GetBackingArrayForCacheLookupUnsafe() is not null))
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
        return TryExecuteDense(param, grad, m, v, "TryAdamStep",
            (b, pBuf, gBuf, mBuf, vBuf) => b.AdamUpdate(
                pBuf, gBuf, mBuf, vBuf, learningRate, beta1, beta2, epsilon, weightDecay, step, param.Length));
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

        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var gpuEngine)) return false; using var __gpuOp = gpuEngine.EnterGpuOp();
        var backend = gpuEngine.GetBackend();
        if (backend is null) return false;
        if (param.Length != grad.Length || param.Length != m.Length || param.Length != v.Length)
            return false;

        if (TryResolveStateBuffer(gpuEngine, backend, m, param.Length) is not { } mBuf
            || TryResolveStateBuffer(gpuEngine, backend, v, param.Length) is not { } vBuf)
        {
            GpuLaunchProbe.OnFallback("TryAdamWStep-moments-not-device-resident", null);
            return false;
        }
        if (TryAcquireReadOnlyFloatBuffer(gpuEngine, backend, grad, param.Length) is not { } gradIn)
        {
            GpuLaunchProbe.OnFallback("TryAdamWStep-gradient-not-device-resident", null);
            return false;
        }
        // A host-authoritative parameter (eager training) is moved onto the device once and kept there, as PyTorch
        // keeps parameters on the GPU; its host copy is downloaded only when read.
        // A ParameterBuffer view is always updated through its flat array's device buffer -- the one a host read
        // downloads and forwards read -- never through a buffer some other path bound to this tensor object.
        using (gradIn.Slice)
        {
            var gBuf = gradIn.Buffer;
            if (!TryUpdateParameter(gpuEngine, backend, param, pBuf => backend.AdamWUpdate(pBuf, gBuf, mBuf, vBuf,
                    learningRate, beta1, beta2, epsilon, weightDecay, step, param.Length), "TryAdamWStep"))
                return false;
        }
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
        IGpuBuffer? pBuf;
        if (DirectGpuTensorEngine.IsFlatParameterView(param))
        {
            pBuf = engine.AcquireParameterBuffer(param, out _);
        }
        else if (param._gpuBuffer is null && param.OffloadDevicePointer != IntPtr.Zero)
        {
            // GpuPinned / GpuOffload mirror: TryGetGpuBuffer uploads any host writes into the offload pointer first.
            pBuf = param.TryGetGpuBuffer();
        }
        else
        {
            // AcquireParameterBuffer checks version freshness and ensures host-backed weights live in the persistent
            // weight cache (so IsDeviceAuthoritative is true and activation eviction cannot drop them); falls back to
            // a current pre-bound buffer for mock/external non-array-backed GPU tensors.
            pBuf = engine.AcquireParameterBuffer(param, out _) ?? TryGetCurrentGpuBuffer(engine, backend, param);
        }
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
        if (pBuf is null || pBuf.Size < param.Length) { GpuLaunchProbe.OnFallback(site + "-parameter-not-placeable", null); return false; }
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
        return TryExecuteDense(param, grad, "TrySgdStep",
            (b, pBuf, gBuf) => b.SgdUpdate(pBuf, gBuf, learningRate, weightDecay: 0f, param.Length));
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
        int n = parameters.Count;
        for (int i = 0; i < n; i++)
        {
            var pCheck = parameters[i] ?? throw new ArgumentNullException(nameof(parameters));
            var gCheck = gradients[i] ?? throw new ArgumentNullException(nameof(gradients));
            if (pCheck.Length != gCheck.Length)
                throw new ArgumentException($"Parameter {i} and its gradient differ in length.", nameof(gradients));
        }
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var gpuEngine)) return false; using var __gpuOp = gpuEngine.EnterGpuOp();
        var backend = gpuEngine.GetBackend();
        if (backend is null) return false;
        if (n == 0) return true;

        var parameterBuffers = new IGpuBuffer[n];
        var gradientBuffers = new IGpuBuffer[n];
        var sizes = new int[n];
        for (int i = 0; i < n; i++)
        {
            var param = parameters[i];
            var grad = gradients[i];
            var pBuf = TryGetCurrentGpuBuffer(gpuEngine, backend, param);
            var gBuf = TryGetCurrentGpuBuffer(gpuEngine, backend, grad);
            if (pBuf is null || pBuf.Size < param.Length || gBuf is null || gBuf.Size < grad.Length)
                return false;
            parameterBuffers[i] = pBuf;
            gradientBuffers[i] = gBuf;
            sizes[i] = param.Length;
        }
        for (int i = 0; i < n; i++)
        {
            var param = parameters[i];
            param.EnsureOwnedForDeviceWrite();   // see TryUpdateParameter
            var pBuf = TryGetCurrentGpuBuffer(gpuEngine, backend, param);
            if (pBuf is null || pBuf.Size < param.Length) return false;
            parameterBuffers[i] = pBuf;
        }

        if (backend is IMultiTensorGpuOptimizerBackend multiTensor)
            multiTensor.SgdMultiTensorUpdate(parameterBuffers, gradientBuffers, sizes, learningRate, weightDecay: 0f);
        else
            for (int i = 0; i < n; i++)
                backend.SgdUpdate(parameterBuffers[i], gradientBuffers[i], learningRate, weightDecay: 0f, sizes[i]);
        for (int i = 0; i < n; i++)
            MarkParameterUpdated(gpuEngine, backend, parameters[i], parameterBuffers[i]);
        return true;
    }

    /// <summary>
    /// PyTorch's <c>torch.nn.utils.clip_grad_norm_</c> (foreach) on the device: the total L2 norm of every gradient,
    /// the coefficient <c>min(1, maxNorm / (norm + 1e-6))</c> and the in-place scale of every gradient, with no host
    /// read. The norm is exact and overflow-free, and every result bit is the same on every backend
    /// (<see cref="DeterministicGradientNorm"/>). A non-finite norm leaves the gradients unscaled (the coefficient is 1),
    /// so a poisoned step keeps its evidence. Returns null when the current engine is not a GPU engine, its backend has no multi-tensor kernels,
    /// a gradient is not device-resident, or <paramref name="maxNorm"/> exceeds <see cref="float.MaxValue"/>; the caller
    /// then clips on the host.
    /// </summary>
    /// <param name="gradients">The gradients, scaled in place.</param>
    /// <param name="maxNorm">The norm bound (positive).</param>
    /// <returns>The total norm, read back to the host only when a caller asks for it.</returns>
    public static DeviceGradientNorm? TryClipGradNorm(IReadOnlyList<Tensor<float>> gradients, double maxNorm)
        => TryClipGradNorm(gradients, maxNorm, NonFiniteGradientPolicy.LeaveUnscaled);

    /// <summary>As <see cref="TryClipGradNorm(IReadOnlyList{Tensor{float}}, double)"/>, with
    /// <paramref name="nonFinite"/> choosing what a NaN or infinite element does to the gradients.</summary>
    public static DeviceGradientNorm? TryClipGradNorm(IReadOnlyList<Tensor<float>> gradients, double maxNorm,
        NonFiniteGradientPolicy nonFinite)
    {
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        if (!(maxNorm > 0) || double.IsInfinity(maxNorm)) throw new ArgumentOutOfRangeException(nameof(maxNorm), "The norm bound must be positive and finite.");
        if (!Enum.IsDefined(typeof(NonFiniteGradientPolicy), nonFinite))
            throw new ArgumentOutOfRangeException(nameof(nonFinite), nonFinite, "Undefined non-finite gradient policy.");
        // The device kernels take the bound as a float: past float.MaxValue it would narrow to infinity and the
        // coefficient to 1, leaving a norm above the bound unclipped. Such a bound is clipped on the host, in double.
        if (maxNorm > float.MaxValue) return null;
        return DeviceSumOfSquares(gradients, (float)maxNorm, nonFinite);
    }

    /// <summary>
    /// The total squared L2 norm of device-resident gradients, reduced on the device (one multi-tensor launch on CUDA
    /// and HIP) and read back only when the result is read: one scalar, where a host scan downloads every gradient.
    /// A non-finite result means some element is NaN or infinite. Returns null under the same conditions as
    /// <see cref="TryClipGradNorm"/>.
    /// </summary>
    /// <remarks>The sum is exact to <see cref="DeterministicGradientNorm"/>'s grid, in integers: no finite gradient
    /// overflows it or underflows to zero, and it has the same bits on every backend, the CPU included.</remarks>
    public static DeviceGradientNorm? TryGradientSumOfSquares(IReadOnlyList<Tensor<float>> gradients)
    {
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        return DeviceSumOfSquares(gradients, clipTo: null, NonFiniteGradientPolicy.LeaveUnscaled);
    }

    private static DeviceGradientNorm? DeviceSumOfSquares(IReadOnlyList<Tensor<float>> gradients, float? clipTo,
        NonFiniteGradientPolicy nonFinite)
    {
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var gpuEngine)) return null;
        using var __gpuOp = gpuEngine.EnterGpuOp();
        var backend = gpuEngine.GetBackend();
        if (backend is not IMultiTensorKernels multi) return null;

        var buffers = new List<IGpuBuffer>(gradients.Count);
        var sizes = new List<int>(gradients.Count);
        var owners = new List<Tensor<float>>(gradients.Count);
        var gradientIndex = new List<int>(gradients.Count);
        for (int i = 0; i < gradients.Count; i++)
        {
            var g = gradients[i];
            if (g is null || g.Length == 0) continue;
            // Only a current device copy: a buffer whose version matches the gradient's present version (or GpuPinned
            // offload mirror). Rejects device buffers made stale by a host write, even if Device still reports GPU.
            var buf = TryGetCurrentGpuBuffer(gpuEngine, backend, g);
            if (buf is null || buf.Size < g.Length) return null;
            buffers.Add(buf);
            sizes.Add(g.Length);
            owners.Add(g);
            gradientIndex.Add(i);
        }

        var sumOfSquares = backend.AllocateBuffer(DeterministicGradientNorm.RecordWords);
        IGpuBuffer? scale = null;
        try
        {
            if (clipTo is not null) scale = backend.AllocateBuffer(1);
            // The record and the coefficient (one launch on CUDA and HIP). Error is raised on the host below; on the
            // device it leaves the gradients unscaled.
            MultiTensorArgs.NormAndClipScale(multi, buffers, sizes, sumOfSquares, clipTo ?? 0f, scale,
                nonFinite == NonFiniteGradientPolicy.Error ? NonFiniteGradientPolicy.LeaveUnscaled : nonFinite);
            if (scale is not null)
                multi.MultiTensorClipScale(buffers, sizes, scale, nonFinite == NonFiniteGradientPolicy.ZeroNonFinite);
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
        var norm = new DeviceGradientNorm(multi, sumOfSquares, scale, gradientIndex.ToArray());
        try
        {
            if (clipTo is not null && nonFinite == NonFiniteGradientPolicy.Error && !norm.IsFinite)
                throw NonFiniteNormException(norm.FirstNonFiniteGradient, norm.NanCount, norm.InfinityCount);
            return norm;
        }
        catch
        {
            norm.Dispose();
            throw;
        }
    }

    /// <summary>The <see cref="NonFiniteGradientPolicy.Error"/> exception (PyTorch's error_if_nonfinite message, with
    /// the first non-finite gradient and the counts).</summary>
    internal static NonFiniteGradientException NonFiniteNormException(int firstGradient, long nanCount, long infinityCount) =>
        new NonFiniteGradientException(firstGradient, nanCount, infinityCount);

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
        return TryExecuteDense(p, g, "TryProximalL1Step",
            (b, pb, gb) => b.ProximalL1Update(pb, gb, lr, l1Strength, p.Length));
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
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var e)) return false; using var __gpuOp = e.EnterGpuOp();
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
            if (length > 0)
            {
                var mInitial = new byte[length];
                for (int i = 0; i < mInitial.Length; i++)
                    mInitial[i] = Compilation.DynamicQuantizationCodebook.SignedZeroIndex;
                cb.UploadByteBuffer(lmQ, mInitial);
                cb.UploadByteBuffer(lvQ, new byte[length]);
            }
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

    /// <summary>
    /// GPU-resident 8-bit Adam step: dequantize, update and requantize without host moment downloads.
    /// A host-authoritative parameter is placed on the device as every fused step does. Returns false when the gradient is
    /// not resident, the parameter cannot be placed, state is missing, or the backend lacks support.
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
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var e)) return false; using var __gpuOp = e.EnterGpuOp();
        if (e.GetBackend() is not ICompressedMomentGpuOptimizerBackend cb) return false;
        if (TryAcquireReadOnlyFloatBuffer(e, cb, g, p.Length) is not { } gradIn) return false;
        int nb = HostCompressedMomentAdam.BlockCount(p.Length, blockSize);
        using (gradIn.Slice)
        {
            var gb = gradIn.Buffer;
            return TryUpdateParameter(e, cb, p, pb => cb.Adam8BitUpdate(pb, gb, mQ, vQ, mScales, vScales,
                lr, beta1, beta2, epsilon, 1f - beta1, 1f - beta2, biasCorrection1, biasCorrection2, blockSize, p.Length, nb),
                "TryAdam8BitStep");
        }
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
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var e)) return false; using var __gpuOp = e.EnterGpuOp();
        var backend = e.GetBackend();
        if (backend is not IAdam8BitWeightDecayBackend db)
            return weightDecay == 0f
                && TryAdam8BitStep(p, g, mQ, vQ, mScales, vScales, lr, beta1, beta2, epsilon, biasCorrection1, biasCorrection2, blockSize);
        if (TryAcquireReadOnlyFloatBuffer(e, db, g, p.Length) is not { } gradIn) return false;
        int nb = HostCompressedMomentAdam.BlockCount(p.Length, blockSize);
        using (gradIn.Slice)
        {
            var gb = gradIn.Buffer;
            return TryUpdateParameter(e, db, p, pb => db.Adam8BitUpdate(pb, gb, mQ, vQ, mScales, vScales,
                lr, beta1, beta2, epsilon, 1f - beta1, 1f - beta2, biasCorrection1, biasCorrection2, blockSize, p.Length, nb,
                weightDecay, decoupledWeightDecay), "TryAdam8BitStep");
        }
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
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var e)) return false; using var __gpuOp = e.EnterGpuOp();
        if (e.GetBackend() is not ICompressedMomentGpuOptimizerBackend cb) return false;
        DirectGpu.IGpuBuffer? lm = null, lv = null;
        try
        {
            int byteLength = checked(length * sizeof(ushort));
            lm = cb.AllocateByteBuffer(byteLength);
            lv = cb.AllocateByteBuffer(byteLength);
            if (byteLength > 0)
            {
                var zeroBytes = new byte[byteLength];
                cb.UploadByteBuffer(lm, zeroBytes);
                cb.UploadByteBuffer(lv, zeroBytes);
            }
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
    /// bias corrections use; step 1 ignores the stored moments. A host-authoritative parameter is placed on the device as
    /// every fused step does. Returns false (CPU fallback) when the gradient is not GPU-resident, the parameter cannot be
    /// placed, a moment buffer is null, or the backend has no compressed-moment kernels.
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
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var e)) return false; using var __gpuOp = e.EnterGpuOp();
        if (e.GetBackend() is not ICompressedMomentGpuOptimizerBackend cb) return false;
        if (TryAcquireReadOnlyFloatBuffer(e, cb, g, p.Length) is not { } gradIn) return false;
        using (gradIn.Slice)
        {
            var gb = gradIn.Buffer;
            return TryUpdateParameter(e, cb, p, pb =>
            {
                if (decoupledWeightDecay)
                    cb.AdamWUpdateBf16(pb, gb, m, v, lr, beta1, beta2, epsilon, weightDecay, step, p.Length);
                else
                    cb.AdamUpdateBf16(pb, gb, m, v, lr, beta1, beta2, epsilon, weightDecay, step, p.Length);
            }, "TryAdamBf16Step");
        }
    }

    // ---- GPU-resident steps for the rest of the backend optimizer kernels (same contract as TryAdamStep:
    //      returns false if not on a GPU engine or any tensor cannot be resolved on the device, so the caller falls back to CPU).
    //      Parameters and state tensors are placed/kept GPU-resident and updated IN PLACE with no host download. ----

    private static bool TryExecuteDense(
        Tensor<float>? p, Tensor<float>? g, string site,
        Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer> kernel)
    {
        if (p is null || g is null || p.Length != g.Length) return false;
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var e)) return false;
        using var __gpuOp = e.EnterGpuOp();
        var b = e.GetBackend();
        if (b is null) return false;
        if (TryAcquireReadOnlyFloatBuffer(e, b, g, p.Length) is not { } gradIn) return false;
        using (gradIn.Slice)
        {
            var gb = gradIn.Buffer;
            return TryUpdateParameter(e, b, p, pb => kernel(b, pb, gb), site);
        }
    }

    private static bool TryExecuteDense(
        Tensor<float>? p, Tensor<float>? g, Tensor<float>? s1, string site,
        Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer> kernel)
    {
        if (p is null || g is null || s1 is null || p.Length != g.Length || p.Length != s1.Length) return false;
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var e)) return false;
        using var __gpuOp = e.EnterGpuOp();
        var b = e.GetBackend();
        if (b is null) return false;
        if (TryResolveStateBuffer(e, b, s1, p.Length) is not { } sb1) return false;
        if (TryAcquireReadOnlyFloatBuffer(e, b, g, p.Length) is not { } gradIn) return false;
        using (gradIn.Slice)
        {
            var gb = gradIn.Buffer;
            if (!TryUpdateParameter(e, b, p, pb => kernel(b, pb, gb, sb1), site)) return false;
        }
        MarkGpuUpdated(b, s1);
        return true;
    }

    private static bool TryExecuteDense(
        Tensor<float>? p, Tensor<float>? g, Tensor<float>? s1, Tensor<float>? s2, string site,
        Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, IGpuBuffer> kernel)
    {
        if (p is null || g is null || s1 is null || s2 is null || p.Length != g.Length || p.Length != s1.Length || p.Length != s2.Length) return false;
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var e)) return false;
        using var __gpuOp = e.EnterGpuOp();
        var b = e.GetBackend();
        if (b is null) return false;
        if (TryResolveStateBuffer(e, b, s1, p.Length) is not { } sb1) return false;
        if (TryResolveStateBuffer(e, b, s2, p.Length) is not { } sb2) return false;
        if (TryAcquireReadOnlyFloatBuffer(e, b, g, p.Length) is not { } gradIn) return false;
        using (gradIn.Slice)
        {
            var gb = gradIn.Buffer;
            if (!TryUpdateParameter(e, b, p, pb => kernel(b, pb, gb, sb1, sb2), site)) return false;
        }
        MarkGpuUpdated(b, s1, s2);
        return true;
    }

    private static bool TryExecuteDense(
        Tensor<float>? p, Tensor<float>? g, Tensor<float>? s1, Tensor<float>? s2, Tensor<float>? s3, string site,
        Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, IGpuBuffer, IGpuBuffer> kernel)
    {
        if (p is null || g is null || s1 is null || s2 is null || s3 is null
            || p.Length != g.Length || p.Length != s1.Length || p.Length != s2.Length || p.Length != s3.Length) return false;
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var e)) return false;
        using var __gpuOp = e.EnterGpuOp();
        var b = e.GetBackend();
        if (b is null) return false;
        if (TryResolveStateBuffer(e, b, s1, p.Length) is not { } sb1) return false;
        if (TryResolveStateBuffer(e, b, s2, p.Length) is not { } sb2) return false;
        if (TryResolveStateBuffer(e, b, s3, p.Length) is not { } sb3) return false;
        if (TryAcquireReadOnlyFloatBuffer(e, b, g, p.Length) is not { } gradIn) return false;
        using (gradIn.Slice)
        {
            var gb = gradIn.Buffer;
            if (!TryUpdateParameter(e, b, p, pb => kernel(b, pb, gb, sb1, sb2, sb3), site)) return false;
        }
        MarkGpuUpdated(b, s1, s2, s3);
        return true;
    }

    /// <summary>Plain SGD with L2-coupled weight decay (<c>grad + weightDecay·param</c>), in place on the GPU.</summary>
    internal static bool TrySgdStep(Tensor<float> p, Tensor<float> g, float lr, float weightDecay)
        => TryExecuteDense(p, g, "TrySgdStep", (b, pb, gb) => b.SgdUpdate(pb, gb, lr, weightDecay, p.Length));

    /// <summary>
    /// Downloads a device-resident tensor's elements, or returns null when it has no device buffer. The backend may
    /// round a buffer up, so exactly <c>tensor.Length</c> elements are returned.
    /// </summary>
    internal static float[]? TryDownload(Tensor<float> tensor)
    {
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var e)) return null; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return null;
        var buffer = TryGetCurrentGpuBuffer(e, b, tensor);
        if (buffer is null) return null;
        if (buffer.Size == tensor.Length)
        {
            var result = new float[tensor.Length];
            b.DownloadBuffer(buffer, result);
            return result;
        }
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
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var e)) return false; using var __gpuOp = e.EnterGpuOp(); var b = e.GetBackend(); if (b is null) return false;
        var buffer = tensor.TryGetGpuBuffer();
        if (buffer is null || buffer.Handle == IntPtr.Zero || buffer.Size < tensor.Length) return false;
        var source = data.Length == tensor.Length ? data : SubArray(data, tensor.Length);
        b.UploadBufferInPlace(source, buffer);
        MarkParameterUpdated(e, b, tensor, buffer);
        return true;
    }

    private static float[] SubArray(float[] source, int length)
    {
        var slice = new float[length];
        Array.Copy(source, 0, slice, 0, length);
        return slice;
    }

    public static bool TrySgdMomentumStep(Tensor<float> p, Tensor<float> g, Tensor<float> velocity, float lr, float momentum, float weightDecay)
        => TryExecuteDense(p, g, velocity, "TrySgdMomentumStep", (b, pb, gb, vb) => b.SgdMomentumUpdate(pb, gb, vb, lr, momentum, weightDecay, p.Length));

    public static bool TryRmspropStep(Tensor<float> p, Tensor<float> g, Tensor<float> squaredAvg, float lr, float rho, float epsilon, float weightDecay)
        => TryExecuteDense(p, g, squaredAvg, "TryRmspropStep", (b, pb, gb, sb) => b.RmspropUpdate(pb, gb, sb, lr, rho, epsilon, weightDecay, p.Length));

    public static bool TryAdagradStep(Tensor<float> p, Tensor<float> g, Tensor<float> accum, float lr, float epsilon, float weightDecay)
        => TryExecuteDense(p, g, accum, "TryAdagradStep", (b, pb, gb, ab) => b.AdagradUpdate(pb, gb, ab, lr, epsilon, weightDecay, p.Length));

    public static bool TryNagStep(Tensor<float> p, Tensor<float> g, Tensor<float> velocity, float lr, float momentum, float weightDecay)
        => TryExecuteDense(p, g, velocity, "TryNagStep", (b, pb, gb, vb) => b.NagUpdate(pb, gb, vb, lr, momentum, weightDecay, p.Length));

    public static bool TryLarsStep(Tensor<float> p, Tensor<float> g, Tensor<float> velocity, float lr, float momentum, float weightDecay, float trustCoeff)
        => TryExecuteDense(p, g, velocity, "TryLarsStep", (b, pb, gb, vb) => b.LarsUpdate(pb, gb, vb, lr, momentum, weightDecay, trustCoeff, p.Length));

    public static bool TryLambStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, Tensor<float> v, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step,
        float maxTrustRatio = 0f, bool biasCorrection = true)
        => TryExecuteDense(p, g, m, v, "TryLambStep", (b, pb, gb, mb, vb) => b.LambUpdate(pb, gb, mb, vb, lr, beta1, beta2, epsilon, weightDecay, step, p.Length, maxTrustRatio, biasCorrection));

    public static bool TryAdadeltaStep(Tensor<float> p, Tensor<float> g, Tensor<float> accumGrad, Tensor<float> accumUpdate, float rho, float epsilon, float weightDecay)
        => TryExecuteDense(p, g, accumGrad, accumUpdate, "TryAdadeltaStep", (b, pb, gb, ag, au) => b.AdadeltaUpdate(pb, gb, ag, au, rho, epsilon, weightDecay, p.Length));

    public static bool TryAmsgradStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, Tensor<float> v, Tensor<float> vMax, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => TryExecuteDense(p, g, m, v, vMax, "TryAmsgradStep", (b, pb, gb, mb, vb, xb) => b.AmsgradUpdate(pb, gb, mb, vb, xb, lr, beta1, beta2, epsilon, weightDecay, step, p.Length));

    public static bool TryAdamaxStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, Tensor<float> u, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => TryExecuteDense(p, g, m, u, "TryAdamaxStep", (b, pb, gb, mb, ub) => b.AdamaxUpdate(pb, gb, mb, ub, lr, beta1, beta2, epsilon, weightDecay, step, p.Length));

    public static bool TryLionStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, float lr, float beta1, float beta2, float weightDecay)
        => TryExecuteDense(p, g, m, "TryLionStep", (b, pb, gb, mb) => b.LionUpdate(pb, gb, mb, lr, beta1, beta2, weightDecay, p.Length));

    public static bool TryNadamStep(Tensor<float> p, Tensor<float> g, Tensor<float> m, Tensor<float> v, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => TryExecuteDense(p, g, m, v, "TryNadamStep", (b, pb, gb, mb, vb) => b.NadamUpdate(pb, gb, mb, vb, lr, beta1, beta2, epsilon, weightDecay, step, p.Length));

    public static bool TryFtrlStep(Tensor<float> p, Tensor<float> g, Tensor<float> z, Tensor<float> n, float lr, float l1Reg, float l2Reg, float beta)
        => TryExecuteDense(p, g, z, n, "TryFtrlStep", (b, pb, gb, zb, nb) => b.FtrlUpdate(pb, gb, zb, nb, lr, l1Reg, l2Reg, beta, p.Length));

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
    // sparseIndices and sparseValues are reused when GPU-resident or uploaded
    // transiently onto the device stream when host-resident. sparseIndices[0..nnz)
    // must be in range, unique, and pre-aggregated; native kernels intentionally
    // use non-atomic scatter writes and do not define duplicate-index order.
    // The dense state buffers (m, v, accum, ...) keep their full shape and
    // are mutated in place at the touched indices.
    //
    // Returns false on:
    //   * non-GPU engine (caller should run CPU sparse path)
    //   * any tensor that cannot be placed on the active GPU backend
    //   * backend NotSupportedException
    // ==================================================================

    private static bool TryExecuteSparse(
        Tensor<float> param, Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        string site,
        Func<DirectGpuTensorEngine, IDirectGpuBackend, IGpuBuffer, IGpuBuffer, Action<IGpuBuffer>?> resolveKernel,
        Action<IDirectGpuBackend> markStatesUpdated)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (sparseIndices is null) throw new ArgumentNullException(nameof(sparseIndices));
        if (sparseValues is null) throw new ArgumentNullException(nameof(sparseValues));
        if (nnz < 0) throw new ArgumentOutOfRangeException(nameof(nnz));
        if (nnz > sparseIndices.Length || nnz > sparseValues.Length)
            throw new ArgumentException("nnz exceeds the supplied sparse buffers.", nameof(nnz));

        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var engine)) return false;
        using var __gpuOp = engine.EnterGpuOp();
        var backend = engine.GetBackend();
        if (backend is null) return false;

        if (TryAcquireReadOnlyIntBuffer(engine, backend, sparseIndices, nnz, enforceSizeOnExisting: false) is not { } idxIn)
            return false;
        using (idxIn.Slice)
        {
            var idxBuf = idxIn.Buffer;
            if (TryAcquireReadOnlyFloatBuffer(engine, backend, sparseValues, nnz, enforceSizeOnExisting: false) is not { } valIn)
                return false;
            using (valIn.Slice)
            {
                var valBuf = valIn.Buffer;
                // Cheap, always-on device-buffer bounds guards: the kernels read indices[k] AND values[k] for
                // k < nnz, so both buffers must hold at least nnz elements or the dispatch reads out of bounds.
                if (idxBuf.Size < nnz)
                    throw new ArgumentException($"sparseIndices buffer holds {idxBuf.Size} elements but nnz={nnz}.", nameof(sparseIndices));
                if (valBuf.Size < nnz)
                    throw new ArgumentException($"sparseValues buffer holds {valBuf.Size} elements but nnz={nnz}.", nameof(sparseValues));

                EnsureUniqueSparseDispatchIndices(backend, idxBuf, nnz, param.Length);
                var kernel = resolveKernel(engine, backend, idxBuf, valBuf);
                if (kernel is null) return false;
                try
                {
                    if (!TryUpdateParameter(engine, backend, param, kernel, site))
                        return false;
                    markStatesUpdated(backend);
                    return true;
                }
                catch (NotSupportedException ex)
                {
                    GpuLaunchProbe.OnFallback(site, ex);
                    return false;
                }
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

    private static bool TryExecuteSparse(
        Tensor<float> param, Tensor<float> s1, string s1Name,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz, string site,
        Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, IGpuBuffer> kernel)
    {
        if (s1 is null) throw new ArgumentNullException(s1Name);
        return TryExecuteSparse(
            param, sparseIndices, sparseValues, nnz, site,
            (e, b, iBuf, vBuf) =>
            {
                if (s1.Length != param.Length) return null;
                if (TryResolveStateBuffer(e, b, s1, param.Length) is not { } sb1) return null;
                return pb => kernel(b, pb, sb1, iBuf, vBuf);
            },
            b => MarkGpuUpdated(b, s1));
    }

    private static bool TryExecuteSparse(
        Tensor<float> param, Tensor<float> s1, string s1Name, Tensor<float> s2, string s2Name,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz, string site,
        Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, IGpuBuffer, IGpuBuffer> kernel)
    {
        if (s1 is null) throw new ArgumentNullException(s1Name);
        if (s2 is null) throw new ArgumentNullException(s2Name);
        return TryExecuteSparse(
            param, sparseIndices, sparseValues, nnz, site,
            (e, b, iBuf, vBuf) =>
            {
                if (s1.Length != param.Length || s2.Length != param.Length) return null;
                if (TryResolveStateBuffer(e, b, s1, param.Length) is not { } sb1) return null;
                if (TryResolveStateBuffer(e, b, s2, param.Length) is not { } sb2) return null;
                return pb => kernel(b, pb, sb1, sb2, iBuf, vBuf);
            },
            b => MarkGpuUpdated(b, s1, s2));
    }

    private static bool TryExecuteSparse(
        Tensor<float> param, Tensor<float> s1, string s1Name, Tensor<float> s2, string s2Name, Tensor<float> s3, string s3Name,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz, string site,
        Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, IGpuBuffer, IGpuBuffer, IGpuBuffer> kernel)
    {
        if (s1 is null) throw new ArgumentNullException(s1Name);
        if (s2 is null) throw new ArgumentNullException(s2Name);
        if (s3 is null) throw new ArgumentNullException(s3Name);
        return TryExecuteSparse(
            param, sparseIndices, sparseValues, nnz, site,
            (e, b, iBuf, vBuf) =>
            {
                if (s1.Length != param.Length || s2.Length != param.Length || s3.Length != param.Length) return null;
                if (TryResolveStateBuffer(e, b, s1, param.Length) is not { } sb1) return null;
                if (TryResolveStateBuffer(e, b, s2, param.Length) is not { } sb2) return null;
                if (TryResolveStateBuffer(e, b, s3, param.Length) is not { } sb3) return null;
                return pb => kernel(b, pb, sb1, sb2, sb3, iBuf, vBuf);
            },
            b => MarkGpuUpdated(b, s1, s2, s3));
    }

    /// <summary>Adam GPU step driven by a sparse gradient — dispatches the native
    /// sparse_adam_update kernel (one thread per nnz, scatter-update). Returns false
    /// when any tensor cannot be placed on the GPU or the backend has no native sparse kernel.</summary>
    public static bool TryAdamStepSparse(
        Tensor<float> param, Tensor<float> m, Tensor<float> v,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
        => TryExecuteSparse(param, m, nameof(m), v, nameof(v), sparseIndices, sparseValues, nnz, "TryAdamStepSparse",
            (b, pb, mb, vb, iBuf, vBuf) => b.SparseAdamUpdate(pb, mb, vb, iBuf, vBuf, nnz, learningRate, beta1, beta2, epsilon, weightDecay, step));

    /// <summary>AdamW GPU step driven by a sparse gradient (native sparse_adamw_update kernel).</summary>
    public static bool TryAdamWStepSparse(
        Tensor<float> param, Tensor<float> m, Tensor<float> v,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
        => TryExecuteSparse(param, m, nameof(m), v, nameof(v), sparseIndices, sparseValues, nnz, "TryAdamWStepSparse",
            (b, pb, mb, vb, iBuf, vBuf) => b.SparseAdamWUpdate(pb, mb, vb, iBuf, vBuf, nnz, learningRate, beta1, beta2, epsilon, weightDecay, step));

    /// <summary>SGD GPU step driven by a sparse gradient (native sparse_sgd_update kernel).</summary>
    public static bool TrySgdStepSparse(
        Tensor<float> param,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float learningRate, float weightDecay = 0f)
        => TryExecuteSparse(param, sparseIndices, sparseValues, nnz, "TrySgdStepSparse",
            (_, b, iBuf, vBuf) => pb => b.SparseSgdUpdate(pb, iBuf, vBuf, nnz, learningRate, weightDecay),
            _ => { });

    /// <summary>SGD-momentum GPU step driven by a sparse gradient.</summary>
    public static bool TrySgdMomentumStepSparse(
        Tensor<float> param, Tensor<float> velocity,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float learningRate, float momentum, float weightDecay)
        => TryExecuteSparse(param, velocity, nameof(velocity), sparseIndices, sparseValues, nnz, "TrySgdMomentumStepSparse",
            (b, pb, velBuf, iBuf, vBuf) => b.SparseSgdMomentumUpdate(pb, velBuf, iBuf, vBuf, nnz, learningRate, momentum, weightDecay));

    /// <summary>RMSProp GPU step driven by a sparse gradient.</summary>
    public static bool TryRmspropStepSparse(
        Tensor<float> param, Tensor<float> squaredAvg,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float rho, float epsilon, float weightDecay)
        => TryExecuteSparse(param, squaredAvg, nameof(squaredAvg), sparseIndices, sparseValues, nnz, "TryRmspropStepSparse",
            (b, pb, sb, iBuf, vBuf) => b.SparseRmspropUpdate(pb, sb, iBuf, vBuf, nnz, lr, rho, epsilon, weightDecay));

    /// <summary>Adagrad GPU step driven by a sparse gradient.</summary>
    public static bool TryAdagradStepSparse(
        Tensor<float> param, Tensor<float> accum,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float epsilon, float weightDecay)
        => TryExecuteSparse(param, accum, nameof(accum), sparseIndices, sparseValues, nnz, "TryAdagradStepSparse",
            (b, pb, ab, iBuf, vBuf) => b.SparseAdagradUpdate(pb, ab, iBuf, vBuf, nnz, lr, epsilon, weightDecay));

    /// <summary>NAG GPU step driven by a sparse gradient.</summary>
    public static bool TryNagStepSparse(
        Tensor<float> param, Tensor<float> velocity,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float momentum, float weightDecay)
        => TryExecuteSparse(param, velocity, nameof(velocity), sparseIndices, sparseValues, nnz, "TryNagStepSparse",
            (b, pb, velBuf, iBuf, vBuf) => b.SparseNagUpdate(pb, velBuf, iBuf, vBuf, nnz, lr, momentum, weightDecay));

    /// <summary>AdaDelta GPU step driven by a sparse gradient.</summary>
    public static bool TryAdadeltaStepSparse(
        Tensor<float> param, Tensor<float> accumGrad, Tensor<float> accumUpdate,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float rho, float epsilon, float weightDecay)
        => TryExecuteSparse(param, accumGrad, nameof(accumGrad), accumUpdate, nameof(accumUpdate), sparseIndices, sparseValues, nnz, "TryAdadeltaStepSparse",
            (b, pb, agb, aub, iBuf, vBuf) => b.SparseAdadeltaUpdate(pb, agb, aub, iBuf, vBuf, nnz, rho, epsilon, weightDecay));

    /// <summary>AMSGrad GPU step driven by a sparse gradient.</summary>
    public static bool TryAmsgradStepSparse(
        Tensor<float> param, Tensor<float> m, Tensor<float> v, Tensor<float> vMax,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => TryExecuteSparse(param, m, nameof(m), v, nameof(v), vMax, nameof(vMax), sparseIndices, sparseValues, nnz, "TryAmsgradStepSparse",
            (b, pb, mb, vb, vmb, iBuf, vBuf) => b.SparseAmsgradUpdate(pb, mb, vb, vmb, iBuf, vBuf, nnz, lr, beta1, beta2, epsilon, weightDecay, step));

    /// <summary>Adamax GPU step driven by a sparse gradient.</summary>
    public static bool TryAdamaxStepSparse(
        Tensor<float> param, Tensor<float> m, Tensor<float> u,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => TryExecuteSparse(param, m, nameof(m), u, nameof(u), sparseIndices, sparseValues, nnz, "TryAdamaxStepSparse",
            (b, pb, mb, ub, iBuf, vBuf) => b.SparseAdamaxUpdate(pb, mb, ub, iBuf, vBuf, nnz, lr, beta1, beta2, epsilon, weightDecay, step));

    /// <summary>Lion GPU step driven by a sparse gradient.</summary>
    public static bool TryLionStepSparse(
        Tensor<float> param, Tensor<float> m,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float beta1, float beta2, float weightDecay)
        => TryExecuteSparse(param, m, nameof(m), sparseIndices, sparseValues, nnz, "TryLionStepSparse",
            (b, pb, mb, iBuf, vBuf) => b.SparseLionUpdate(pb, mb, iBuf, vBuf, nnz, lr, beta1, beta2, weightDecay));

    /// <summary>Nadam GPU step driven by a sparse gradient.</summary>
    public static bool TryNadamStepSparse(
        Tensor<float> param, Tensor<float> m, Tensor<float> v,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => TryExecuteSparse(param, m, nameof(m), v, nameof(v), sparseIndices, sparseValues, nnz, "TryNadamStepSparse",
            (b, pb, mb, vb, iBuf, vBuf) => b.SparseNadamUpdate(pb, mb, vb, iBuf, vBuf, nnz, lr, beta1, beta2, epsilon, weightDecay, step));

    /// <summary>FTRL GPU step driven by a sparse gradient.</summary>
    public static bool TryFtrlStepSparse(
        Tensor<float> param, Tensor<float> z, Tensor<float> n,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float l1Reg, float l2Reg, float beta)
        => TryExecuteSparse(param, z, nameof(z), n, nameof(n), sparseIndices, sparseValues, nnz, "TryFtrlStepSparse",
            (b, pb, zb, nb, iBuf, vBuf) => b.SparseFtrlUpdate(pb, zb, nb, iBuf, vBuf, nnz, lr, l1Reg, l2Reg, beta));

    /// <summary>Proximal-L1 GPU step driven by a sparse gradient.</summary>
    public static bool TryProximalL1StepSparse(
        Tensor<float> param,
        Tensor<int> sparseIndices, Tensor<float> sparseValues, int nnz,
        float lr, float l1Strength)
        => TryExecuteSparse(param, sparseIndices, sparseValues, nnz, "TryProximalL1StepSparse",
            (_, b, iBuf, vBuf) => pb => b.SparseProximalL1Update(pb, iBuf, vBuf, nnz, lr, l1Strength),
            _ => { });
}
