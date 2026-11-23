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
    /// <see cref="AssignStateInPlace"/> for several states at once: with <see cref="IBatchedCopyBackend"/> (CUDA) the copies are one launch (one graph node
    /// in a captured plan instead of one memcpy node per state). The states must be distinct tensors.
    /// </summary>
    internal void AssignStatesInPlace<T>(IReadOnlyList<Tensor<T>> states, IReadOnlyList<Tensor<T>> nexts)
    {
        if (states is null) throw new ArgumentNullException(nameof(states));
        if (nexts is null) throw new ArgumentNullException(nameof(nexts));
        if (nexts.Count != states.Count) throw new ArgumentException("One next value per state.", nameof(nexts));
        if (states.Count < 2 || typeof(T) != typeof(float))
        {
            for (int i = 0; i < states.Count; i++) AssignStateInPlace(states[i], nexts[i]);
            return;
        }
        using var lease = EnterGpuOp();
        if (!TryGetBackend(out var backend) || backend is not IBatchedCopyBackend batched)
        {
            for (int i = 0; i < states.Count; i++) AssignStateInPlace(states[i], nexts[i]);
            return;
        }
        var destinations = new IGpuBuffer[states.Count];
        var sources = new OwnedBuffer[states.Count];
        var lengths = new int[states.Count];
        bool ordered = false;
        try
        {
            for (int i = 0; i < states.Count; i++)
            {
                var (state, next) = (states[i], nexts[i]);
                if (state is null) throw new ArgumentNullException(nameof(states));
                if (next is null) throw new ArgumentNullException(nameof(nexts));
                if (next.Length != state.Length)
                    throw new ArgumentException(
                        $"A state of {state.Length} elements cannot be assigned a value of {next.Length}.", nameof(nexts));
                if (!state.IsContiguous || IsPartialView(state))
                    throw new ArgumentException("A plan state must be a whole contiguous tensor, not a view.", nameof(states));
                destinations[i] = PersistentBufferFor(backend, state);
                sources[i] = GetOrAllocateBuffer(backend, next);
                lengths[i] = state.Length;
            }
            var sourceBuffers = sources.Select(s => s.Buffer).ToArray();
            // One launch runs its copies concurrently, so it is only equivalent to the declared-order writes when no
            // copy touches another's buffer: two states sharing one backing array, or a next that is a view of another
            // state. Those take the ordered per-state path below. A partial view resolves to a fresh gathered buffer
            // whose address never matches its base state's, so a next sharing a state's host storage is checked by
            // storage too: the batch would read the state before its write, the ordered path after it.
            ordered = StateCopiesOverlap(sourceBuffers, destinations) || NextSharesStateStorage(states, nexts);
            if (!ordered) batched.CopyMany(sourceBuffers, destinations, lengths);
        }
        finally
        {
            foreach (var s in sources) s.Dispose();
        }
        if (ordered)
        {
            for (int i = 0; i < states.Count; i++) AssignStateInPlace(states[i], nexts[i]);
            return;
        }
        for (int i = 0; i < states.Count; i++)
        {
            states[i].IncrementVersion();
            BindResidentBuffer(states[i], destinations[i], backend);
        }
    }

    /// <summary>
    /// Whether some next value shares its backing storage with a state (a view of a state): its copy can then depend on
    /// whether that state's write has happened, so the writes keep their declared order.
    /// </summary>
    internal static bool NextSharesStateStorage<T>(IReadOnlyList<Tensor<T>> states, IReadOnlyList<Tensor<T>> nexts)
    {
        for (int i = 0; i < nexts.Count; i++)
        {
            var next = nexts[i];
            var nextArray = next.GetBackingArrayForCacheLookupUnsafe();
            for (int j = 0; j < states.Count; j++)
            {
                var state = states[j];
                if (ReferenceEquals(next.StorageIdentity, state.StorageIdentity)
                    || (nextArray is not null && ReferenceEquals(nextArray, state.GetBackingArrayForCacheLookupUnsafe())))
                    return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Whether a batch of state copies cannot run as one concurrent launch: two destinations are the same device buffer,
    /// or a source is another copy's destination (it would be read before or after that copy overwrites it).
    /// </summary>
    internal static bool StateCopiesOverlap(IReadOnlyList<IGpuBuffer> sources, IReadOnlyList<IGpuBuffer> destinations)
    {
        static bool Same(IGpuBuffer a, IGpuBuffer b)
            => ReferenceEquals(a, b) || (a.Handle != IntPtr.Zero && a.Handle == b.Handle);
        for (int i = 0; i < destinations.Count; i++)
            for (int j = 0; j < destinations.Count; j++)
            {
                if (i == j) continue;
                if (j > i && Same(destinations[i], destinations[j])) return true;
                if (Same(sources[i], destinations[j])) return true;
            }
        return false;
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
        if (!tensor.IsContiguous || IsPartialView(tensor)) return null;
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend)) return null;
        if (tensor._gpuBuffer is { } bound && bound.Handle != IntPtr.Zero && bound.Size >= tensor.Length
            && ReferenceEquals(tensor._gpuBackend, backend) && tensor._gpuBufferVersion == tensor.GpuCacheVersion) return bound;
        var buffer = PersistentBufferFor(backend, tensor);
        BindResidentBuffer(tensor, buffer, backend, isDeviceModified: deviceAuthoritative);
        return buffer;
    }

    /// <summary>
    /// A device copy of the first <paramref name="length"/> elements of <paramref name="buffer"/> (stream-ordered), so a
    /// pass that writes the buffer in place can be undone with <see cref="RestoreDeviceSnapshot"/>.
    /// </summary>
    internal IGpuBuffer TakeDeviceSnapshot(IGpuBuffer buffer, int length)
    {
        using var __gpuOp = EnterGpuOp();
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (!TryGetBackend(out var backend)) throw new InvalidOperationException("No GPU backend.");
        var snapshot = AllocateDeviceBufferUninitialized(backend, length);
        try
        {
            backend.Copy(buffer, snapshot, length);
        }
        catch
        {
            // The caller never receives the snapshot, so nothing else can release it.
            snapshot.Dispose();
            throw;
        }
        return snapshot;
    }

    /// <summary>Writes a <see cref="TakeDeviceSnapshot"/> back into <paramref name="buffer"/> and frees the snapshot.</summary>
    internal void RestoreDeviceSnapshot(IGpuBuffer snapshot, IGpuBuffer buffer, int length)
    {
        using var __gpuOp = EnterGpuOp();
        if (snapshot is null) throw new ArgumentNullException(nameof(snapshot));
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        // Freed on every path, including a failed copy: the caller handed the snapshot over.
        using var owned = snapshot;
        if (!TryGetBackend(out var backend)) throw new InvalidOperationException("No GPU backend.");
        backend.Copy(owned, buffer, length);
    }

    /// <summary>
    /// Downloads the first <paramref name="length"/> elements of a buffer a captured graph writes (a replayed plan's
    /// output), independent of any tensor's binding to it. Null when the buffer is gone.
    /// </summary>
    internal float[]? DownloadCapturedBuffer(IGpuBuffer buffer, int length)
    {
        using var __gpuOp = EnterGpuOp();
        if (buffer is null || buffer.Handle == IntPtr.Zero || buffer.Size < length || !TryGetBackend(out var backend)) return null;
        if (buffer.Size == length) return backend.DownloadBuffer(buffer);
        using var exact = backend.AllocateBuffer(length);
        backend.Copy(buffer, 0, exact, 0, length);
        return backend.DownloadBuffer(exact);
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
    /// <summary>
    /// Whether <paramref name="tensor"/>'s host copy may hold something its device buffer does not: false while its
    /// device download is still pending (nothing on the host has read or written it since the device wrote it), true
    /// once the host copy was made current (a host write after that need not advance the version).
    /// </summary>
    internal static bool HostMayDifferFromDevice<T>(Tensor<T> tensor)
    {
        if (tensor.GetBackingArrayForCacheLookupUnsafe() is not { } array) return false;
        return !Helpers.HostSync.IsPending(array) && !Helpers.HostSync.IsPending(tensor.DataVector);
    }

    internal void RearmStateAfterReplay<T>(Tensor<T> state, IGpuBuffer capturedBuffer)
    {
        using var __gpuOp = EnterGpuOp();
        if (TryGetBackend(out var backend))
        {
            state.IncrementVersion();
            BindResidentBuffer(state, capturedBuffer, backend);
        }
    }
}
