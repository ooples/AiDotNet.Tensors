using System;
using System.Threading;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Internal helpers for the Layer 3 weight pre-pack cache. The
/// <see cref="WeightPackHandle"/> stores the packed buffer + a version
/// counter; this class provides the logic for deciding when to re-pack.
///
/// <para>
/// On each Gemm call referencing a handle:
/// </para>
/// <list type="number">
///   <item>If <c>handle.Version == handle.LastPackedVersion</c>, the cached buffer is current — skip the pack step.</item>
///   <item>If they differ, re-pack the weight into <c>handle.PackedBuffer</c> and set <c>LastPackedVersion = Version</c>.</item>
/// </list>
///
/// <para>
/// The optimizer-step path calls <see cref="WeightPackHandle.MarkDirty"/>
/// after mutating a weight, which atomically increments
/// <c>handle.Version</c>. The next Gemm call sees the version mismatch and
/// triggers a re-pack. Atomicity matters because multiple threads may read
/// Version concurrently with the optimizer's write.
/// </para>
/// </summary>
internal static class WeightPackCache
{
    /// <summary>
    /// Allocate a new <see cref="WeightPackHandle"/> with a freshly-rented
    /// pack buffer of the requested size. Sets Version=1 (initial state) and
    /// LastPackedVersion=0 (forces a pack on first use).
    /// </summary>
    /// <param name="packedBytes">Size of the pack buffer in bytes.</param>
    /// <param name="key">Cache key (Mc, Kc, TransA, PackingMode, ElemType).</param>
    /// <param name="isForA">True for pre-packed A, false for pre-packed B.</param>
    internal static WeightPackHandle Allocate(
        int packedBytes,
        (int Mc, int Kc, bool TransA, PackingMode Mode, Type ElemType) key,
        bool isForA)
    {
        if (packedBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(packedBytes), "Pack buffer size must be positive.");

        var buffer = new byte[packedBytes];
        return new WeightPackHandle(buffer, key, isForA);
    }

    /// <summary>
    /// True if the handle's cached packed buffer is current — i.e., no
    /// MarkDirty since the last pack. Callers use this to decide whether to
    /// invoke the pack routine.
    /// </summary>
    internal static bool IsCacheCurrent(WeightPackHandle handle)
    {
        // Both fields atomic so net471 / x86 doesn't tear the 64-bit read.
        long current = Interlocked.Read(ref handle.Version);
        long packed = handle.ReadLastPackedVersion();
        return current == packed;
    }

    /// <summary>
    /// Mark the handle's cache as current after a successful pack. Callers
    /// MUST snapshot <c>Interlocked.Read(ref handle.Version)</c> BEFORE
    /// starting the pack and pass it here. Reading Version after the pack
    /// would race with a concurrent <see cref="WeightPackHandle.MarkDirty"/>
    /// and mark stale packed data as current (CodeRabbit #366 race).
    /// </summary>
    /// <param name="handle">The weight pack handle.</param>
    /// <param name="packedVersion">The Version snapshot taken before packing.</param>
    internal static void MarkCacheCurrent(WeightPackHandle handle, long packedVersion)
    {
        handle.WriteLastPackedVersion(packedVersion);
    }
}
