using System;
using System.Threading;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Handle to a pre-packed weight buffer (A or B side of a GEMM). Created by
/// <c>BlasManaged.PrePackA</c> / <c>PrePackB</c>; passed back into subsequent
/// <c>Gemm</c> calls via <see cref="BlasOptions{T}.PackedA"/> /
/// <see cref="BlasOptions{T}.PackedB"/>. Reuses the packed layout across
/// training iterations to amortize pack cost.
/// </summary>
/// <remarks>
/// The handle owns a byte buffer holding the packed weight in BLIS-style
/// stripe layout (see Section 5 of the design spec). The dispatcher honours
/// the cached pack if the handle's <c>Version</c> matches
/// <c>LastPackedVersion</c>; otherwise re-packs and updates
/// <c>LastPackedVersion</c> to the current <c>Version</c>.
///
/// <para>
/// <c>Version</c> is atomically incremented by <see cref="MarkDirty"/> each
/// time the underlying weight is mutated (e.g., from an optimizer step).
/// <c>LastPackedVersion</c> is written by the dispatch path after a
/// successful pack and records the <c>Version</c> at which the buffer was
/// last filled. When <c>Version == LastPackedVersion</c> the packed buffer
/// is current and the pack step can be skipped.
/// </para>
///
/// Optimizer-step paths must call <see cref="MarkDirty"/> after mutating the
/// underlying weight so the next Gemm call re-packs before use.
/// </remarks>
public sealed class WeightPackHandle : IDisposable
{
    internal byte[] PackedBuffer;
    internal long Version;
    /// <summary>
    /// The <see cref="Version"/> value at which <see cref="PackedBuffer"/> was
    /// last successfully written. Initialized to 0; <see cref="Version"/>
    /// starts at 1, so the very first Gemm call always triggers a pack.
    /// Written only from the dispatch/pack path (single-writer); read
    /// atomically by <see cref="WeightPackCache.IsCacheCurrent"/>.
    /// </summary>
    internal long LastPackedVersion;  // 0 = never packed; Version starts at 1
    internal (int Mc, int Kc, bool TransA, PackingMode Mode, Type ElemType) Key;
    internal bool IsForA;  // true = pre-packed A; false = pre-packed B

    internal WeightPackHandle(
        byte[] packedBuffer,
        (int Mc, int Kc, bool TransA, PackingMode Mode, Type ElemType) key,
        bool isForA)
    {
        PackedBuffer = packedBuffer;
        Version = 1;
        LastPackedVersion = 0;
        Key = key;
        IsForA = isForA;
    }

    /// <summary>
    /// Signal that the underlying weight has been mutated (e.g., by an optimizer
    /// step). The next Gemm call referencing this handle will re-pack the weight
    /// before use.
    /// </summary>
    public void MarkDirty() => Interlocked.Increment(ref Version);

    /// <summary>
    /// Release the packed buffer back to the pool. Idempotent.
    /// </summary>
    public void Dispose()
    {
        // Pool return is handled by WeightPackCache in Phase F (Task F3); nothing to do here yet.
    }
}
