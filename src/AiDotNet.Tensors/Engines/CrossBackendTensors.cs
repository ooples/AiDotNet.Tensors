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
        return owner is not null && consumer is not null && !ReferenceEquals(owner, consumer);
    }

    /// <summary>
    /// Applies the policy to a tensor whose buffer belongs to <paramref name="owner"/> while an op on
    /// <paramref name="consumer"/> asked for it. Migrate brings the data home through its owner and drops the foreign
    /// binding, so the caller's ordinary upload path places it on its own backend (once: the tensor then belongs
    /// there); Throw raises.
    /// </summary>
    internal static void Resolve<T>(Tensor<T> tensor, IDirectGpuBackend owner, IDirectGpuBackend consumer)
    {
        var policy = AiDotNetEngine.CrossBackendPolicy;
        if (policy == CrossBackendTensorPolicy.Throw)
            throw new InvalidOperationException(
                $"A tensor of shape [{string.Join(", ", tensor.Shape.ToArray())}] is on the {Describe(owner)} backend, but the "
                + $"operation runs on {Describe(consumer)}. Move it to the engine's device first (Tensor.To / Tensor.Gpu with that "
                + "engine as AiDotNetEngine.Current), or set AiDotNetEngine.CrossBackendPolicy to Migrate or MigrateWarn.");

        long bytes = (long)tensor.Length * sizeof(float);
        tensor.ReleaseForeignGpuBinding();
        Interlocked.Increment(ref s_migrations);
        if (policy == CrossBackendTensorPolicy.MigrateWarn && FirstReport(tensor.StorageIdentity, owner, consumer))
            AiDotNetEngine.LogDiagnostic(
                $"[AiDotNet] Migrated a tensor of shape [{string.Join(", ", tensor.Shape.ToArray())}] ({bytes:N0} bytes) from the "
                + $"{Describe(owner)} backend to {Describe(consumer)}: it was placed while another engine was "
                + "AiDotNetEngine.Current. It now lives on the new backend; place it there directly to avoid the copy, or set "
                + "AiDotNetEngine.CrossBackendPolicy = Throw to make this an error.");
    }

    /// <summary>
    /// Applies the policy when <paramref name="tensor"/> is bound on a backend other than <paramref name="consumer"/>:
    /// for engine paths that resolve a tensor against an explicit backend instead of through
    /// <see cref="TensorBase{T}.TryGetGpuBuffer"/>, so a foreign binding is migrated (or refused) there too rather
    /// than re-uploaded from the host on every op, or dropped without reading back what its owner last wrote.
    /// </summary>
    internal static void ResolveIfForeign<T>(Tensor<T> tensor, IDirectGpuBackend consumer)
    {
        var owner = tensor._gpuBackend;
        if (tensor._gpuBuffer is null || owner is null || ReferenceEquals(owner, consumer)) return;
        Resolve(tensor, owner, consumer);
    }

    private static bool FirstReport(object storage, IDirectGpuBackend owner, IDirectGpuBackend consumer)
    {
        string pair = owner.GetType().Name + "->" + consumer.GetType().Name;
        var seen = s_reported.GetValue(storage, static _ => new HashSet<string>(StringComparer.Ordinal));
        lock (seen) return seen.Add(pair);
    }

    private static string Describe(IDirectGpuBackend backend)
    {
        try { return $"{backend.BackendName} ({backend.DeviceName})"; }
        catch (Exception) { return backend.GetType().Name; }
    }
}
