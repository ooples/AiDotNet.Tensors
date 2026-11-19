// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// The state write of a stateful compiled inference plan: copies <paramref name="next"/> into
    /// <paramref name="state"/>'s own device buffer, so the address a captured graph read the state from stays the
    /// address it is written back to, and marks <paramref name="state"/> device-resident (a host read downloads it).
    /// </summary>
    /// <remarks>
    /// The state must be a whole contiguous tensor. Its buffer is resolved through the engine's cache, so the first
    /// call (an eager or pre-residency run, never a capture) uploads the host value once; afterwards the state lives on
    /// the device and every later write is one device-to-device copy, which a stream capture records.
    /// </remarks>
    internal void AssignStateInPlace<T>(Tensor<T> state, Tensor<T> next)
    {
        using var lease = EnterGpuOp();
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (next is null) throw new ArgumentNullException(nameof(next));
        if (next.Length != state.Length)
            throw new ArgumentException(
                $"A state of {state.Length} elements cannot be assigned a value of {next.Length}.", nameof(next));
        if (!state.IsContiguous || IsPartialView(state))
            throw new ArgumentException("A plan state must be a whole contiguous tensor, not a view.", nameof(state));
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend))
        {
            ((Compilation.ISignatureTensor)state).CopyLeafFrom(next);
            return;
        }

        var destination = PersistentBufferFor(backend, state);
        using (var source = GetOrAllocateBuffer(backend, next))
            backend.Copy(source.Buffer, destination, state.Length);
        state.IncrementVersion();
        BindResidentBuffer(state, destination, backend);
    }

    /// <summary>
    /// Gives a captured plan's input slot a persistent device buffer holding its current host value and binds it,
    /// so the capture records reads of that address and every replay refreshes it in place. No-op when bound.
    /// </summary>
    internal IGpuBuffer? PinInputResident<T>(Tensor<T> input) => PinResident(input, deviceAuthoritative: true);

    /// <summary>
    /// Gives a leaf a captured plan reads (a constant, a weight) a persistent device buffer, so the capture records a
    /// read of a stable address instead of a transient upload. The host copy stays authoritative (nothing downloads).
    /// </summary>
    internal IGpuBuffer? PinLeafResident<T>(Tensor<T> leaf) => PinResident(leaf, deviceAuthoritative: false);

    private IGpuBuffer? PinResident<T>(Tensor<T> tensor, bool deviceAuthoritative)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous || IsPartialView(tensor) || !TryGetBackend(out var backend)) return null;
        if (tensor._gpuBuffer is { } bound && bound.Handle != IntPtr.Zero && bound.Size >= tensor.Length
            && ReferenceEquals(tensor._gpuBackend, backend) && tensor._gpuBufferVersion == tensor.GpuCacheVersion) return bound;
        var buffer = PersistentBufferFor(backend, tensor);
        BindResidentBuffer(tensor, buffer, backend, isDeviceModified: deviceAuthoritative);
        return buffer;
    }

    /// <summary>
    /// Undoes <see cref="PinInputResident{T}"/> when a plan goes back to eager execution: SetInputs writes only the host
    /// copy, and a pinned binding would keep serving the call that pinned it (measured: every eager call after an
    /// abandoned capture read that call's input).
    /// </summary>
    internal void UnpinInput<T>(Tensor<T> input)
    {
        if (input is not null && input._gpuBuffer is not null) InvalidateStaleBinding(input);
    }

    private void InvalidateStaleBinding<T>(Tensor<T> tensor)
    {
        var array = tensor.GetBackingArrayForCacheLookupUnsafe();
        if (array is not null) InvalidateActivationCacheEntry(array);
        tensor._gpuBuffer = null;
        tensor._gpuBackend = null;
        tensor._gpuBufferVersion = -1;
    }

    // The tensor's own device buffer: the bound one, else the cached copy of its host array, else a fresh upload
    // adopted into the activation cache under that array, so it lives as long as the tensor's storage.
    private IGpuBuffer PersistentBufferFor<T>(IDirectGpuBackend backend, Tensor<T> tensor)
    {
        if (tensor._gpuBuffer is { } bound && ReferenceEquals(tensor._gpuBackend, backend) && bound.Handle != IntPtr.Zero
            && bound.Size >= tensor.Length && tensor._gpuBufferVersion == tensor.GpuCacheVersion)
            return bound;
        using var owned = GetOrAllocateBuffer(backend, tensor);
        var buffer = owned.Buffer;
        if (owned.OwnsBuffer)
        {
            var key = tensor.GetBackingArrayForCacheLookupUnsafe()
                ?? throw new InvalidOperationException("The tensor has no host storage to own its device buffer.");
            if (!CacheActivation(cacheKey: key, buffer, tensor._shape, backend, hostVersion: tensor.GpuCacheVersion))
                throw new InvalidOperationException("The stateful plan could not retain its device buffer.");
            owned.RelinquishOwnership();
        }
        return buffer;
    }

    /// <summary>
    /// After a captured graph wrote <paramref name="state"/> into <paramref name="capturedBuffer"/> (the address the
    /// capture recorded), re-binds it there device-resident, so a host read downloads this replay's value (a replay runs
    /// no host bookkeeping, and host reads between replays may have moved the tensor's binding).
    /// </summary>
    internal void RearmStateAfterReplay<T>(Tensor<T> state, IGpuBuffer capturedBuffer)
    {
        if (TryGetBackend(out var backend))
        {
            state.IncrementVersion();
            BindResidentBuffer(state, capturedBuffer, backend);
        }
    }
}
