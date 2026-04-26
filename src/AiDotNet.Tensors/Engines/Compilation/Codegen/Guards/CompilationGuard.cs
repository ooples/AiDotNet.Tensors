// Copyright (c) AiDotNet. All rights reserved.
// Codegen compilation guards — sub-microsecond runtime checks that
// decide whether a cached compiled kernel is still valid for the
// current call's shapes / dtypes / graph structure.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Guards;

/// <summary>
/// A packed, bit-comparable fingerprint of a compiled kernel's
/// preconditions. Captures the graph content hash + the concrete
/// dtype + a shape-bucket identifier so the runtime can decide in
/// &lt; 100 ns whether a cached kernel is valid for the incoming
/// call.
/// </summary>
/// <remarks>
/// <para><b>Why 64 bits:</b></para>
/// <para>
/// The guard fires on every compiled-kernel call. A 64-bit integer
/// comparison is a single CPU instruction; any richer structure
/// (multiple fields, span checks) would show up in the profile as
/// guard overhead. Collisions are resolved by a slow-path full
/// equality check on the cached graph hash + shape tuple.
/// </para>
/// <para><b>Composition:</b></para>
/// <para>
/// <c>(graphContentHash ^ dtypeBits ^ shapeBucketBits)</c> with the
/// three components spread across different bit ranges to minimise
/// collision probability. FNV-1a remains the underlying hash; the
/// packed form is a deterministic projection of it.
/// </para>
/// </remarks>
public readonly struct CompilationGuard : IEquatable<CompilationGuard>
{
    /// <summary>The packed 64-bit fingerprint.</summary>
    public long Value { get; }

    /// <summary>
    /// Constructs a guard from its individual components.
    /// </summary>
    /// <param name="graphContentHash">Content hash from
    /// <see cref="CodegenGraph.ComputeContentHash"/>.</param>
    /// <param name="dtype">The dtype the kernel was specialised for.</param>
    /// <param name="shapeBucket">Shape-bucket identifier — see
    /// <see cref="ShapeBucket.Compute"/>.</param>
    public CompilationGuard(long graphContentHash, CodegenElementType dtype, long shapeBucket)
    {
        // Rotate the graph hash into the high 32 bits, dtype into the
        // middle, shape bucket into the low 32 — keeps the three axes
        // independent in the fingerprint space.
        long dtypeBits = ((long)dtype & 0xFFFF) << 32;
        long shapeBits = shapeBucket & 0xFFFFFFFF;
        long rotatedGraph = (graphContentHash << 32) | ((long)(graphContentHash >> 32) & 0xFFFFFFFFL);
        Value = rotatedGraph ^ dtypeBits ^ shapeBits;
    }

    /// <inheritdoc/>
    public bool Equals(CompilationGuard other) => Value == other.Value;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is CompilationGuard g && Equals(g);

    /// <inheritdoc/>
    public override int GetHashCode() => Value.GetHashCode();

    /// <inheritdoc/>
    public static bool operator ==(CompilationGuard a, CompilationGuard b) => a.Value == b.Value;

    /// <inheritdoc/>
    public static bool operator !=(CompilationGuard a, CompilationGuard b) => a.Value != b.Value;

    /// <inheritdoc/>
    public override string ToString()
        => $"CompilationGuard(0x{Value:X16})";
}

/// <summary>
/// Shape-bucket scheme — groups concrete shapes into equivalence
/// classes so a single compiled kernel can serve a range of batch
/// sizes (8 → 128) without recompiling.
/// </summary>
/// <remarks>
/// <para><b>Bucket policy:</b></para>
/// <para>
/// The default policy rounds each dimension to the next power of
/// two; callers can substitute their own
/// <see cref="IShapeBucketPolicy"/> via
/// <see cref="CodegenGuardRegistry.SetPolicy"/>. The default is the
/// torch.compile "<c>dynamic=False, cache_by_shape=bucket</c>"
/// equivalent.
/// </para>
/// </remarks>
public static class ShapeBucket
{
    /// <summary>
    /// Computes the shape-bucket identifier for <paramref name="shape"/>
    /// under the supplied policy (or the registry default if
    /// <paramref name="policy"/> is null).
    /// </summary>
    public static long Compute(int[] shape, IShapeBucketPolicy? policy = null)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        var activePolicy = policy ?? CodegenGuardRegistry.Policy;
        const long FnvOffset = unchecked((long)0xCBF29CE484222325UL);
        const long FnvPrime = 0x00000100000001B3L;
        long h = FnvOffset;
        for (int i = 0; i < shape.Length; i++)
        {
            int bucketed = activePolicy.BucketizeDimension(i, shape[i]);
            h = unchecked((h ^ bucketed) * FnvPrime);
        }
        return h;
    }
}

/// <summary>
/// Decides how a concrete shape dimension collapses into a bucket.
/// Callers that want per-axis recompile control (e.g. bucket batch
/// tightly but let spatial dims vary freely) supply a custom policy
/// via <see cref="CodegenGuardRegistry.SetPolicy"/>.
/// </summary>
public interface IShapeBucketPolicy
{
    /// <summary>
    /// Returns the bucket id for dimension <paramref name="dimIndex"/>
    /// with concrete value <paramref name="dimValue"/>. Two dims that
    /// return the same id are considered equivalent for cache reuse.
    /// </summary>
    int BucketizeDimension(int dimIndex, int dimValue);
}

/// <summary>
/// Default policy: next-power-of-two bucketing. Matches the
/// behaviour of <c>torch.compile</c> with dynamic=False but
/// <c>cache_size_limit</c> relaxed.
/// </summary>
public sealed class PowerOfTwoBucketPolicy : IShapeBucketPolicy
{
    /// <inheritdoc/>
    public int BucketizeDimension(int dimIndex, int dimValue)
    {
        if (dimValue <= 1) return 1;
        int v = dimValue - 1;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }
}

/// <summary>
/// Identity policy: no bucketing, every shape is its own class. The
/// strictest option — any shape change triggers a recompile. Useful
/// for small-kernel paths where a recompile is cheap and bucketing
/// would hurt perf.
/// </summary>
public sealed class ExactShapePolicy : IShapeBucketPolicy
{
    /// <inheritdoc/>
    public int BucketizeDimension(int dimIndex, int dimValue) => dimValue;
}
