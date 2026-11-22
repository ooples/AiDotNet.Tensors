using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// What an engine does when an op meets a tensor whose GPU buffer belongs to a different backend than the one the op
/// runs on: for example a tensor moved with <c>Tensor.Gpu()</c> while an OpenCL engine was
/// <see cref="AiDotNetEngine.Current"/>, then trained by a CUDA engine.
/// </summary>
/// <remarks>
/// <para>A buffer from another backend is never a valid address for this backend's kernels (an OpenCL handle read as a
/// CUDA device pointer is an illegal-address fault that poisons the CUDA context for the rest of the process), so the
/// engine never hands it to a kernel. The policy decides whether it migrates the tensor or refuses.</para>
/// <para><b>For Beginners:</b> PyTorch stops with "Expected all tensors to be on the same device" and makes you call
/// <c>.to(device)</c>. The default here moves the tensor for you once and tells you it did, so mixed code keeps
/// working; choose <see cref="Throw"/> for PyTorch's stricter behaviour.</para>
/// </remarks>
public enum CrossBackendTensorPolicy
{
    /// <summary>Move the tensor to the op's backend once (it then stays there) and log one line per tensor and backend
    /// pair through <see cref="AiDotNetEngine.Logger"/>. The default.</summary>
    MigrateWarn = 0,

    /// <summary>Move the tensor to the op's backend once, silently.</summary>
    Migrate = 1,

    /// <summary>Throw <see cref="InvalidOperationException"/>: the caller must place the tensor on the engine's
    /// device itself (PyTorch's behaviour).</summary>
    Throw = 2,
}

/// <summary>
/// Resolves a tensor's GPU buffer against the backend of the engine op running on the calling thread
/// (<see cref="GpuOpGate.CurrentOpBackend"/>) and applies <see cref="AiDotNetEngine.CrossBackendPolicy"/>.
/// </summary>
internal static class CrossBackendTensors
{
    // Storage identities already reported, with the backend pairs reported for each (warn once per pair).
    private static readonly ConditionalWeakTable<object, HashSet<string>> s_reported = new();

    /// <summary>Number of tensors migrated between backends in this process (diagnostics and tests).</summary>
    internal static long MigrationCount => Interlocked.Read(ref s_migrations);
    private static long s_migrations;

    /// <summary>
    /// Whether <paramref name="owner"/> is a backend other than the calling op's, so the op must not use a buffer
    /// it placed. False outside an engine op or when either side is unknown.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsForeign(IDirectGpuBackend? owner)
    {
        var consumer = GpuOpGate.CurrentOpBackend;
        return owner is not null && consumer is not null && !ReferenceEquals(owner, consumer)
               && !SameAddressSpace(owner, consumer);
    }

    /// <summary>
    /// Whether a buffer <paramref name="owner"/> placed is a valid address for <paramref name="consumer"/>'s kernels
    /// with no synchronization: the same backend seen through any wrapper (a deferred scope's recording backend, a
    /// tracing wrapper), or two backends that share one address space AND one stream
    /// (<see cref="IGpuSharedAddressSpace"/>).
    /// </summary>
    internal static bool SameAddressSpace(IDirectGpuBackend owner, IDirectGpuBackend consumer)
    {
        var o = Unwrap(owner);
        var c = Unwrap(consumer);
        return ReferenceEquals(o, c) || (o is IGpuSharedAddressSpace shared && shared.SharesAddressSpaceWith(c));
    }

    /// <summary>The backend a wrapper (<see cref="Gpu.DelegatingGpuBackend"/> and its subclasses) forwards to.</summary>
    internal static IDirectGpuBackend Unwrap(IDirectGpuBackend backend)
    {
        while (backend is Gpu.DelegatingGpuBackend wrapper) backend = wrapper.InnerBackend;
        return backend;
    }

    /// <summary>
    /// Applies the policy to a tensor whose buffer belongs to <paramref name="owner"/> while an op on
    /// <paramref name="consumer"/> asked for it. Throw raises. Migrate reads the data back through its owner where
    /// the device copy is the newer one, under the owning engine's op lease so a concurrent Dispose cannot tear its
    /// context down mid-read, and drops the foreign binding; a resident tensor is then re-placed on the consumer as
    /// Tensor.Gpu() would, so it stays resident there. <paramref name="hostAuthoritative"/> is for an external input
    /// whose caller writes through the host and uploads it itself: its device copy is read back only while a device
    /// write is still pending, and it is not re-placed unless <paramref name="rePlaceFromHost"/> (a compiled plan's
    /// leaf, re-placed from its host copy so it stays resident).
    /// </summary>
    internal static void Resolve<T>(Tensor<T> tensor, IDirectGpuBackend owner, IDirectGpuBackend consumer,
        bool hostAuthoritative = false, bool rePlaceFromHost = false)
    {
        var policy = AiDotNetEngine.CrossBackendPolicy;
        if (policy == CrossBackendTensorPolicy.Throw)
            throw new InvalidOperationException(
                $"A tensor of shape [{Shape(tensor)}] is on the {Describe(owner)} backend, but the operation runs on "
                + $"{Describe(consumer)}. Move it to the engine's device first (Tensor.To, or Tensor.Gpu with that engine as "
                + "AiDotNetEngine.Current), or set AiDotNetEngine.CrossBackendPolicy to Migrate or MigrateWarn.");

        // One migration per storage at a time: two ops on two backends can both find a shared weight foreign, and the
        // second must not take the buffer the first just placed for its own backend as the foreign one and free it.
        // Inside the lock the binding is looked at again; a migration that already happened leaves nothing to do.
        lock (tensor.StorageIdentity)
        {
            var bound = tensor._gpuBackend;
            if (tensor._gpuBuffer is null || bound is null || SameAddressSpace(bound, consumer)) return;
            ResolveLocked(tensor, bound, consumer, hostAuthoritative, rePlaceFromHost, policy);
        }
    }

    private static void ResolveLocked<T>(Tensor<T> tensor, IDirectGpuBackend owner, IDirectGpuBackend consumer,
        bool hostAuthoritative, bool rePlaceFromHost, CrossBackendTensorPolicy policy)
    {
        long bytes = (long)tensor.Length * sizeof(float);
        bool wasResident = tensor.IsGpuResident;
        // The binding is dropped whatever the tensor's extent: a whole-storage tensor's binding is its storage's (every
        // alias of that storage moves with it), a partial view's is the view's own (its siblings keep theirs).
        //
        // The device copy is read back only where it holds what the host does not: a pending download (a device write
        // not yet brought home), or a resident tensor whose buffer is as new as its version. A buffer older than the
        // tensor's version is stale - the host was written since - and reading it would overwrite that newer host
        // data; an external input's host is authoritative unless a download is pending.
        bool pending = (tensor.GetBackingArrayForCacheLookupUnsafe() is { } array && Helpers.HostSync.IsPending(array))
                       || Helpers.HostSync.IsPending(tensor.DataVector);
        bool deviceCurrent = wasResident && tensor._gpuBufferVersion == tensor.GpuCacheVersion;
        // A backend whose owning engine was disposed has been freed: nothing on it can be read or freed. Its tensors'
        // host copies are what is left - current unless a device write was still pending, which is lost.
        bool retired = DirectGpuTensorEngine.IsRetiredBackend(Unwrap(owner));
        bool readBack = pending || (!hostAuthoritative && deviceCurrent && !retired);

        var ownerEngine = retired ? null : DirectGpuTensorEngine.EngineOwning(Unwrap(owner));
        using (var ownerLease = ownerEngine?.EnterGpuOp() ?? default)
        {
            bool ownerLive = !retired && (ownerEngine is null || ownerLease.Held);
            if (readBack && !ownerLive)
                throw new InvalidOperationException(
                    $"A tensor of shape [{Shape(tensor)}] is held on the {Describe(owner)} backend by an engine that is "
                    + (retired ? "disposed" : "being disposed") + ", with device values not yet brought to the host, so it "
                    + $"cannot be moved to the operation on {Describe(consumer)}.");
            // A storage the owning engine handed out as its result owns its buffer whichever alias asks.
            bool storageOwned = ownerLive && ownerEngine is not null && tensor.CoversWholeVector
                                && ownerEngine.OwnsDeviceResult(tensor.DataVector);
            // An owner mid-teardown (or gone) frees its own memory with its context: the buffer is detached, not freed.
            tensor.ReleaseForeignGpuBinding(readBack, free: ownerLive, storageOwned);
        }
        // Re-place it on the op's backend exactly as Tensor.Gpu() would have (version-tagged, host-mirrored), so a
        // resident parameter stays resident: left host-only, an on-device optimizer would update a buffer the forward
        // never reads and training would stall. Placed on the real backend, never on a recording wrapper. An external
        // input is not: its caller uploads the host data into a binding of its own next.
        if (wasResident && (!hostAuthoritative || rePlaceFromHost)) tensor.PlaceOn(Unwrap(consumer), DirectGpuTensorEngine.EngineOwning(Unwrap(consumer)));
        Interlocked.Increment(ref s_migrations);
        if (policy == CrossBackendTensorPolicy.MigrateWarn && FirstReport(tensor.StorageIdentity, owner, consumer))
            AiDotNetEngine.LogDiagnostic(
                $"[AiDotNet] Migrated a tensor of shape [{Shape(tensor)}] ({bytes:N0} bytes) from the {Describe(owner)} "
                + $"backend to {Describe(consumer)}: it was placed while another engine was AiDotNetEngine.Current. It now "
                + "lives on the new backend; place it there directly to avoid the copy, or set "
                + "AiDotNetEngine.CrossBackendPolicy = Throw to make this an error.");
    }

    /// <summary>
    /// Applies the policy when <paramref name="tensor"/> is bound on a backend other than <paramref name="consumer"/>:
    /// for engine paths that resolve a tensor against an explicit backend instead of through
    /// <see cref="TensorBase{T}.TryGetGpuBuffer"/>, so a foreign binding is migrated (or refused) there too rather
    /// than re-uploaded from the host on every op, or freed without reading back what its owner last wrote.
    /// </summary>
    internal static void ResolveIfForeign<T>(Tensor<T> tensor, IDirectGpuBackend consumer, bool hostAuthoritative = false,
        bool rePlaceFromHost = false)
    {
        var owner = tensor._gpuBackend;
        if (tensor._gpuBuffer is null || owner is null || SameAddressSpace(owner, consumer)) return;
        Resolve(tensor, owner, consumer, hostAuthoritative, rePlaceFromHost);
    }

    private static string Shape<T>(Tensor<T> tensor) => string.Join(", ", tensor.Shape.ToArray());

    private static bool FirstReport(object storage, IDirectGpuBackend owner, IDirectGpuBackend consumer)
    {
        string pair = Unwrap(owner).GetType().Name + "->" + Unwrap(consumer).GetType().Name;
        var seen = s_reported.GetValue(storage, static _ => new HashSet<string>(StringComparer.Ordinal));
        lock (seen) return seen.Add(pair);
    }

    private static string Describe(IDirectGpuBackend backend)
    {
        try { return $"{backend.BackendName} ({backend.DeviceName})"; }
        catch (Exception) { return backend.GetType().Name; }
    }
}
