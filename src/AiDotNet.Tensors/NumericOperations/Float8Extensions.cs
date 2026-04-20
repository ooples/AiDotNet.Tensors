using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Conversion helpers for the FP8 types. Per issue #197 spec, consumers
/// need <c>float</c> / <c>double</c> / <c>Half</c> round-trip paths that
/// do not require manually instantiating the struct's explicit operator.
/// </summary>
public static class Float8Extensions
{
    // ──────────────── Float8E4M3 ────────────────

    /// <summary>Convert <paramref name="v"/> (float) to E4M3 with saturating overflow.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float8E4M3 ToE4M3(this float v) => Float8E4M3.FromFloat(v);

    /// <summary>Convert <paramref name="v"/> (double) to E4M3.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float8E4M3 ToE4M3(this double v) => Float8E4M3.FromFloat((float)v);

    /// <summary>Convert <paramref name="v"/> (Half) to E4M3.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float8E4M3 ToE4M3(this Half v) => Float8E4M3.FromFloat((float)v);

    /// <summary>Convert E4M3 to Half.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Half ToHalf(this Float8E4M3 v) => (Half)v.ToFloat();

    /// <summary>Convert E4M3 to double.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ToDouble(this Float8E4M3 v) => (double)v.ToFloat();

    /// <summary>Bulk-convert a <c>float[]</c> span into an E4M3 output span.</summary>
    public static void ToE4M3Array(ReadOnlySpan<float> src, Span<Float8E4M3> dst)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException("Destination too small.", nameof(dst));
        for (int i = 0; i < src.Length; i++) dst[i] = Float8E4M3.FromFloat(src[i]);
    }

    /// <summary>Bulk-convert an E4M3 span into a <c>float[]</c> output span.</summary>
    public static void ToFloatArray(ReadOnlySpan<Float8E4M3> src, Span<float> dst)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException("Destination too small.", nameof(dst));
        for (int i = 0; i < src.Length; i++) dst[i] = src[i].ToFloat();
    }

    // ──────────────── Float8E5M2 ────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float8E5M2 ToE5M2(this float v) => Float8E5M2.FromFloat(v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float8E5M2 ToE5M2(this double v) => Float8E5M2.FromFloat((float)v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float8E5M2 ToE5M2(this Half v) => Float8E5M2.FromFloat((float)v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Half ToHalf(this Float8E5M2 v) => (Half)v.ToFloat();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ToDouble(this Float8E5M2 v) => (double)v.ToFloat();

    public static void ToE5M2Array(ReadOnlySpan<float> src, Span<Float8E5M2> dst)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException("Destination too small.", nameof(dst));
        for (int i = 0; i < src.Length; i++) dst[i] = Float8E5M2.FromFloat(src[i]);
    }

    public static void ToFloatArray(ReadOnlySpan<Float8E5M2> src, Span<float> dst)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException("Destination too small.", nameof(dst));
        for (int i = 0; i < src.Length; i++) dst[i] = src[i].ToFloat();
    }
}
