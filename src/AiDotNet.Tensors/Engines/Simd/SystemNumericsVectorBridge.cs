// audit-2026-05 phase 5: BCL-portable SIMD bridge for net471.
//
// AiDotNet.Tensors targets net10.0;net471. The net10.0 build uses
// System.Runtime.Intrinsics (AVX2 / AVX-512 directly via Avx, Avx512F, Sse,
// AdvSimd). System.Runtime.Intrinsics is unavailable pre-net5, so the net471
// build used to fall through to a per-element scalar loop after every
// `#if NET5_0_OR_GREATER` block — trading ~8× throughput for source-compatibility.
//
// The BCL has shipped `System.Numerics.Vector<T>` since .NET 4.6 (RyuJIT
// auto-vectorizes it to SSE2 / AVX where the host CPU + JIT support it).
// Lane width is JIT-determined at runtime (4 lanes for SSE2-only, 8 lanes
// for AVX), accessed via the static `Vector<T>.Count` property.
//
// Phase 5 wires this bridge into every hot path in SimdKernels.cs (and
// downstream callers) via `#if NET5_0_OR_GREATER ... #else bridge call; #endif`
// branches. The net10 path is unchanged in every wiring; net471 silently
// gains AVX2 throughput.
//
// API SHAPE: spans in, span out. Internally uses
// `MemoryMarshal.Cast<T, Vector<T>>` to reinterpret the spans as
// Vector<T> spans (Vector<T> is blittable), then loops element-wise.
// Same pattern that SimdGemm.cs uses at line 1329 — proven on net471.

#if !NET5_0_OR_GREATER

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// BCL-portable SIMD primitives for the net471 fallback path. Every method
/// uses <see cref="Vector{T}"/> which RyuJIT auto-vectorizes to SSE2 / AVX2 on
/// supporting hardware.
/// </summary>
internal static class SystemNumericsVectorBridge
{
    public static int FloatLaneCount => Vector<float>.Count;
    public static int DoubleLaneCount => Vector<double>.Count;
    public static bool IsHardwareAccelerated => Vector.IsHardwareAccelerated;

    // ====================================================================
    // FLOAT — binary element-wise
    // ====================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void VectorAdd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);

        var aVecs = MemoryMarshal.Cast<float, Vector<float>>(a.Slice(0, simdElements));
        var bVecs = MemoryMarshal.Cast<float, Vector<float>>(b.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] + bVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] + b[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void VectorSubtract(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);

        var aVecs = MemoryMarshal.Cast<float, Vector<float>>(a.Slice(0, simdElements));
        var bVecs = MemoryMarshal.Cast<float, Vector<float>>(b.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] - bVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] - b[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void VectorMultiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);

        var aVecs = MemoryMarshal.Cast<float, Vector<float>>(a.Slice(0, simdElements));
        var bVecs = MemoryMarshal.Cast<float, Vector<float>>(b.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] * bVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] * b[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void VectorDivide(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);

        var aVecs = MemoryMarshal.Cast<float, Vector<float>>(a.Slice(0, simdElements));
        var bVecs = MemoryMarshal.Cast<float, Vector<float>>(b.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] / bVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] / b[i];
    }

    // ====================================================================
    // FLOAT — scalar broadcast
    // ====================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddScalar(ReadOnlySpan<float> a, float scalar, Span<float> result)
    {
        if (a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);
        var sVec = new Vector<float>(scalar);

        var aVecs = MemoryMarshal.Cast<float, Vector<float>>(a.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] + sVec;

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] + scalar;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SubtractScalar(ReadOnlySpan<float> a, float scalar, Span<float> result)
    {
        if (a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);
        var sVec = new Vector<float>(scalar);

        var aVecs = MemoryMarshal.Cast<float, Vector<float>>(a.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] - sVec;

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] - scalar;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MultiplyScalar(ReadOnlySpan<float> a, float scalar, Span<float> result)
    {
        if (a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);
        var sVec = new Vector<float>(scalar);

        var aVecs = MemoryMarshal.Cast<float, Vector<float>>(a.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] * sVec;

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] * scalar;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DivideScalar(ReadOnlySpan<float> a, float scalar, Span<float> result)
    {
        if (a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        // Pre-multiply by the reciprocal — matches the SimdKernels scalar fallback
        // for inv-scalar broadcast (preserves IEEE semantics within ulp ≤ 1).
        float inv = 1f / scalar;
        MultiplyScalar(a, inv, result);
    }

    /// <summary>r[i] = a[i] + scalar * b[i] — SAXPY-style fused mul-add.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ScalarMultiplyAdd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, float scalar, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);
        var sVec = new Vector<float>(scalar);

        var aVecs = MemoryMarshal.Cast<float, Vector<float>>(a.Slice(0, simdElements));
        var bVecs = MemoryMarshal.Cast<float, Vector<float>>(b.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] + sVec * bVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] + scalar * b[i];
    }

    /// <summary>Alias of the foundation primitive — kept for symmetry with SimdKernels.DotProduct.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Saxpy(float alpha, ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> result)
    {
        if (x.Length != y.Length || x.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);
        var alphaVec = new Vector<float>(alpha);

        var xVecs = MemoryMarshal.Cast<float, Vector<float>>(x.Slice(0, simdElements));
        var yVecs = MemoryMarshal.Cast<float, Vector<float>>(y.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < xVecs.Length; v++)
            rVecs[v] = alphaVec * xVecs[v] + yVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = alpha * x[i] + y[i];
    }

    // ====================================================================
    // FLOAT — unary element-wise
    // ====================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Sqrt(ReadOnlySpan<float> src, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);

        var sVecs = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
            rVecs[v] = Vector.SquareRoot(sVecs[v]);

        for (int i = simdElements; i < length; i++)
            result[i] = (float)Math.Sqrt(src[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Abs(ReadOnlySpan<float> src, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);

        var sVecs = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
            rVecs[v] = Vector.Abs(sVecs[v]);

        for (int i = simdElements; i < length; i++)
            result[i] = Math.Abs(src[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Negate(ReadOnlySpan<float> src, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);

        var sVecs = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
            rVecs[v] = -sVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = -src[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Clamp(ReadOnlySpan<float> src, float min, float max, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);
        var minVec = new Vector<float>(min);
        var maxVec = new Vector<float>(max);

        var sVecs = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
            rVecs[v] = Vector.Min(maxVec, Vector.Max(minVec, sVecs[v]));

        for (int i = simdElements; i < length; i++)
            result[i] = Math.Min(max, Math.Max(min, src[i]));
    }

    // ====================================================================
    // FLOAT — activations
    // ====================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ReLU(ReadOnlySpan<float> src, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);
        var zero = Vector<float>.Zero;

        var sVecs = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
            rVecs[v] = Vector.Max(zero, sVecs[v]);

        for (int i = simdElements; i < length; i++)
            result[i] = Math.Max(0f, src[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void LeakyReLU(ReadOnlySpan<float> src, float alpha, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);
        var zero = Vector<float>.Zero;
        var alphaVec = new Vector<float>(alpha);

        var sVecs = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
        {
            var x = sVecs[v];
            var mask = Vector.GreaterThan(x, zero);   // all-ones lanes where x>0
            rVecs[v] = Vector.ConditionalSelect(mask, x, alphaVec * x);
        }

        for (int i = simdElements; i < length; i++)
            result[i] = src[i] > 0f ? src[i] : alpha * src[i];
    }

    // ====================================================================
    // FLOAT — reductions
    // ====================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(ReadOnlySpan<float> src)
    {
        int length = src.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);

        var accum = Vector<float>.Zero;
        var sVecs = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
            accum += sVecs[v];

        float sum = Vector.Dot(accum, Vector<float>.One);
        for (int i = simdElements; i < length; i++)
            sum += src[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(ReadOnlySpan<float> src)
    {
        if (src.Length == 0)
            throw new ArgumentException("Cannot compute Max of empty span.");

        int length = src.Length;
        int lanes = Vector<float>.Count;
        if (length < lanes)
        {
            float m = src[0];
            for (int i = 1; i < length; i++)
                if (src[i] > m) m = src[i];
            return m;
        }

        int simdElements = length - (length % lanes);
        var sVecs = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simdElements));

        var accum = sVecs[0];
        for (int v = 1; v < sVecs.Length; v++)
            accum = Vector.Max(accum, sVecs[v]);

        float max = accum[0];
        for (int j = 1; j < lanes; j++)
            if (accum[j] > max) max = accum[j];

        for (int i = simdElements; i < length; i++)
            if (src[i] > max) max = src[i];
        return max;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(ReadOnlySpan<float> src)
    {
        if (src.Length == 0)
            throw new ArgumentException("Cannot compute Min of empty span.");

        int length = src.Length;
        int lanes = Vector<float>.Count;
        if (length < lanes)
        {
            float m = src[0];
            for (int i = 1; i < length; i++)
                if (src[i] < m) m = src[i];
            return m;
        }

        int simdElements = length - (length % lanes);
        var sVecs = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simdElements));

        var accum = sVecs[0];
        for (int v = 1; v < sVecs.Length; v++)
            accum = Vector.Min(accum, sVecs[v]);

        float min = accum[0];
        for (int j = 1; j < lanes; j++)
            if (accum[j] < min) min = accum[j];

        for (int i = simdElements; i < length; i++)
            if (src[i] < min) min = src[i];
        return min;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Input spans must have the same length.");

        int length = a.Length;
        int lanes = Vector<float>.Count;
        int simdElements = length - (length % lanes);
        var accum = Vector<float>.Zero;

        var aVecs = MemoryMarshal.Cast<float, Vector<float>>(a.Slice(0, simdElements));
        var bVecs = MemoryMarshal.Cast<float, Vector<float>>(b.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            accum += aVecs[v] * bVecs[v];

        float sum = Vector.Dot(accum, Vector<float>.One);
        for (int i = simdElements; i < length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    // ====================================================================
    // DOUBLE — binary element-wise
    // ====================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void VectorAdd(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);

        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Slice(0, simdElements));
        var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] + bVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] + b[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void VectorSubtract(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);

        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Slice(0, simdElements));
        var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] - bVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] - b[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void VectorMultiply(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);

        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Slice(0, simdElements));
        var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] * bVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] * b[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void VectorDivide(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);

        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Slice(0, simdElements));
        var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] / bVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] / b[i];
    }

    // ====================================================================
    // DOUBLE — scalar broadcast
    // ====================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddScalar(ReadOnlySpan<double> a, double scalar, Span<double> result)
    {
        if (a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);
        var sVec = new Vector<double>(scalar);

        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] + sVec;

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] + scalar;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SubtractScalar(ReadOnlySpan<double> a, double scalar, Span<double> result)
    {
        if (a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);
        var sVec = new Vector<double>(scalar);

        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] - sVec;

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] - scalar;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MultiplyScalar(ReadOnlySpan<double> a, double scalar, Span<double> result)
    {
        if (a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);
        var sVec = new Vector<double>(scalar);

        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] * sVec;

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] * scalar;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DivideScalar(ReadOnlySpan<double> a, double scalar, Span<double> result)
    {
        if (a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        double inv = 1.0 / scalar;
        MultiplyScalar(a, inv, result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ScalarMultiplyAdd(ReadOnlySpan<double> a, ReadOnlySpan<double> b, double scalar, Span<double> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);
        var sVec = new Vector<double>(scalar);

        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Slice(0, simdElements));
        var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            rVecs[v] = aVecs[v] + sVec * bVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = a[i] + scalar * b[i];
    }

    // ====================================================================
    // DOUBLE — unary element-wise
    // ====================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Sqrt(ReadOnlySpan<double> src, Span<double> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);

        var sVecs = MemoryMarshal.Cast<double, Vector<double>>(src.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
            rVecs[v] = Vector.SquareRoot(sVecs[v]);

        for (int i = simdElements; i < length; i++)
            result[i] = Math.Sqrt(src[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Abs(ReadOnlySpan<double> src, Span<double> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);

        var sVecs = MemoryMarshal.Cast<double, Vector<double>>(src.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
            rVecs[v] = Vector.Abs(sVecs[v]);

        for (int i = simdElements; i < length; i++)
            result[i] = Math.Abs(src[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Negate(ReadOnlySpan<double> src, Span<double> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);

        var sVecs = MemoryMarshal.Cast<double, Vector<double>>(src.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
            rVecs[v] = -sVecs[v];

        for (int i = simdElements; i < length; i++)
            result[i] = -src[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Clamp(ReadOnlySpan<double> src, double min, double max, Span<double> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);
        var minVec = new Vector<double>(min);
        var maxVec = new Vector<double>(max);

        var sVecs = MemoryMarshal.Cast<double, Vector<double>>(src.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
            rVecs[v] = Vector.Min(maxVec, Vector.Max(minVec, sVecs[v]));

        for (int i = simdElements; i < length; i++)
            result[i] = Math.Min(max, Math.Max(min, src[i]));
    }

    // ====================================================================
    // DOUBLE — activations
    // ====================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ReLU(ReadOnlySpan<double> src, Span<double> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);
        var zero = Vector<double>.Zero;

        var sVecs = MemoryMarshal.Cast<double, Vector<double>>(src.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
            rVecs[v] = Vector.Max(zero, sVecs[v]);

        for (int i = simdElements; i < length; i++)
            result[i] = Math.Max(0.0, src[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void LeakyReLU(ReadOnlySpan<double> src, double alpha, Span<double> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        int length = result.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);
        var zero = Vector<double>.Zero;
        var alphaVec = new Vector<double>(alpha);

        var sVecs = MemoryMarshal.Cast<double, Vector<double>>(src.Slice(0, simdElements));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
        {
            var x = sVecs[v];
            var mask = Vector.GreaterThan(x, zero);
            rVecs[v] = Vector.ConditionalSelect(mask, x, alphaVec * x);
        }

        for (int i = simdElements; i < length; i++)
            result[i] = src[i] > 0.0 ? src[i] : alpha * src[i];
    }

    // ====================================================================
    // DOUBLE — reductions
    // ====================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Sum(ReadOnlySpan<double> src)
    {
        int length = src.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);

        var accum = Vector<double>.Zero;
        var sVecs = MemoryMarshal.Cast<double, Vector<double>>(src.Slice(0, simdElements));

        for (int v = 0; v < sVecs.Length; v++)
            accum += sVecs[v];

        double sum = Vector.Dot(accum, Vector<double>.One);
        for (int i = simdElements; i < length; i++)
            sum += src[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Max(ReadOnlySpan<double> src)
    {
        if (src.Length == 0)
            throw new ArgumentException("Cannot compute Max of empty span.");

        int length = src.Length;
        int lanes = Vector<double>.Count;
        if (length < lanes)
        {
            double m = src[0];
            for (int i = 1; i < length; i++)
                if (src[i] > m) m = src[i];
            return m;
        }

        int simdElements = length - (length % lanes);
        var sVecs = MemoryMarshal.Cast<double, Vector<double>>(src.Slice(0, simdElements));

        var accum = sVecs[0];
        for (int v = 1; v < sVecs.Length; v++)
            accum = Vector.Max(accum, sVecs[v]);

        double max = accum[0];
        for (int j = 1; j < lanes; j++)
            if (accum[j] > max) max = accum[j];

        for (int i = simdElements; i < length; i++)
            if (src[i] > max) max = src[i];
        return max;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Min(ReadOnlySpan<double> src)
    {
        if (src.Length == 0)
            throw new ArgumentException("Cannot compute Min of empty span.");

        int length = src.Length;
        int lanes = Vector<double>.Count;
        if (length < lanes)
        {
            double m = src[0];
            for (int i = 1; i < length; i++)
                if (src[i] < m) m = src[i];
            return m;
        }

        int simdElements = length - (length % lanes);
        var sVecs = MemoryMarshal.Cast<double, Vector<double>>(src.Slice(0, simdElements));

        var accum = sVecs[0];
        for (int v = 1; v < sVecs.Length; v++)
            accum = Vector.Min(accum, sVecs[v]);

        double min = accum[0];
        for (int j = 1; j < lanes; j++)
            if (accum[j] < min) min = accum[j];

        for (int i = simdElements; i < length; i++)
            if (src[i] < min) min = src[i];
        return min;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Input spans must have the same length.");

        int length = a.Length;
        int lanes = Vector<double>.Count;
        int simdElements = length - (length % lanes);
        var accum = Vector<double>.Zero;

        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Slice(0, simdElements));
        var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Slice(0, simdElements));

        for (int v = 0; v < aVecs.Length; v++)
            accum += aVecs[v] * bVecs[v];

        double sum = Vector.Dot(accum, Vector<double>.One);
        for (int i = simdElements; i < length; i++)
            sum += a[i] * b[i];
        return sum;
    }
}

#endif
