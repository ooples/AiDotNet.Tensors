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
    /// Accessed atomically via <see cref="ReadLastPackedVersion"/> /
    /// <see cref="WriteLastPackedVersion"/> — torn 64-bit reads on 32-bit
    /// runtimes (net471 on x86) would otherwise produce stale cache-current
    /// decisions even though only one thread writes at a time.
    /// </summary>
    internal long LastPackedVersion;  // 0 = never packed; Version starts at 1
    internal (int Mc, int Kc, bool TransA, PackingMode Mode, Type ElemType) Key;
    internal bool IsForA;  // true = pre-packed A; false = pre-packed B

    /// <summary>
    /// Sub-E (#373) — full weight dimensions and the (Mc, Kc) tiling used during
    /// pack. <see cref="MultiPanelStride"/> tells consumers how many bytes to
    /// advance per (icIdx, pcIdx) tile in the flat <see cref="PackedBuffer"/>.
    /// All zero when the handle was packed via the legacy single-panel path
    /// (pre-Sub-E) — consumers gate on <c>MultiPanelStride &gt; 0</c> to detect
    /// the new layout.
    /// </summary>
    internal int FullM;
    internal int FullK;
    internal int TileMc;
    internal int TileKc;
    internal int NumIcBlocks;
    internal int NumPcBlocks;
    internal int MultiPanelStride;  // bytes per tile in PackedBuffer

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
    /// Atomic read of <see cref="LastPackedVersion"/>. Pairs with
    /// <see cref="WriteLastPackedVersion"/> to keep reads/writes torn-free on
    /// 32-bit runtimes (CodeRabbit #366 thread on this file).
    /// </summary>
    internal long ReadLastPackedVersion() => Interlocked.Read(ref LastPackedVersion);

    /// <summary>
    /// Atomic write of <see cref="LastPackedVersion"/>. Used by the pack path
    /// after a successful pack completes.
    /// </summary>
    internal void WriteLastPackedVersion(long version) => Interlocked.Exchange(ref LastPackedVersion, version);

    /// <summary>
    /// Sub-E (#373): get the byte slice for a specific (icIdx, pcIdx) tile when
    /// this handle is multi-panel. Returns empty span for legacy single-panel
    /// handles (consumers fall back to the old offset-0 path).
    /// </summary>
    internal Span<byte> GetTileSlice(int icIdx, int pcIdx)
    {
        if (MultiPanelStride <= 0) return Span<byte>.Empty;  // legacy single-panel
        if (icIdx < 0 || icIdx >= NumIcBlocks) return Span<byte>.Empty;
        if (pcIdx < 0 || pcIdx >= NumPcBlocks) return Span<byte>.Empty;
        int offset = (icIdx * NumPcBlocks + pcIdx) * MultiPanelStride;
        return PackedBuffer.AsSpan(offset, MultiPanelStride);
    }

    /// <summary>
    /// Returns true if this multi-panel handle's (TileMc, TileKc) match the
    /// caller's (mc, kc) blocking parameters. Required for the strategy to
    /// consume the pre-packed tiles directly.
    /// </summary>
    internal bool TilingMatches(int mc, int kc) =>
        MultiPanelStride > 0 && TileMc == mc && TileKc == kc;

    /// <summary>
    /// Release the packed buffer back to the pool. Idempotent.
    /// </summary>
    public void Dispose()
    {
        // Pool return is handled by WeightPackCache in Phase F (Task F3); nothing to do here yet.
    }
}
