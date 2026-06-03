namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Invalidation surface for the engine's identity-keyed inference weight
/// caches. Several CPU fast paths cache a derived form of a weight array
/// keyed by the array's OBJECT IDENTITY and never re-read its contents:
/// <list type="bullet">
///   <item><c>SimdGemm.SgemmWithCachedB</c> — persistent pre-packed B panels
///   (the FusedLinear / MLP / transformer-FFN inference GEMM path).</item>
///   <item><c>SimdGemm.SgemmWithInt8CachedB</c> — weight-only int8 packed B.</item>
///   <item><c>CpuEngine</c>'s Conv2D small-N fast path — pre-transposed
///   kernel arrays.</item>
/// </list>
/// Mutating a weight array IN PLACE (an optimizer step, a
/// <c>SetParameters</c>/<c>WithParameters</c>-style bulk load, manual tensor
/// writes) therefore leaves those caches stale: subsequent inference would
/// silently compute with the OLD weights. Callers that mutate weights in
/// place must call <see cref="InvalidateAll"/> afterwards; the next use of
/// each weight re-derives its cached form from the live contents.
/// </summary>
/// <remarks>
/// <para>
/// Invalidation is a single global epoch bump rather than per-array removal:
/// the GEMM caches keep per-thread MRU slots that other threads cannot reach,
/// and an epoch comparison on every hit path covers them all. The cost of a
/// hit-path check is one interlocked read; the cost of an invalidation is a
/// re-pack of every live weight on its next use — inherent, since the weights
/// changed.
/// </para>
/// <para>
/// This mirrors the GPU-side contract (<c>DirectGpuTensorEngine.
/// InvalidateAllWeightCaches</c>): both engines key weight-derived caches by
/// reference identity, so both need an explicit flush after in-place updates.
/// </para>
/// </remarks>
public static class InferenceWeightCache
{
    /// <summary>
    /// Invalidate every cached derived weight form (packed GEMM panels,
    /// int8 packs, transposed conv kernels). Call after any in-place weight
    /// mutation so the next inference re-derives from live weight contents.
    /// Thread-safe; O(1).
    /// </summary>
    public static void InvalidateAll() => Simd.SimdGemm.InvalidateCachedWeights();
}
