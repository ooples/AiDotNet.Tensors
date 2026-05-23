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

    // ====================================================================
    // TRANSCENDENTALS — Cephes-style polynomial exp / log on BCL Vector<float>
    //
    // These are faithful ports of SimdKernels.FastExp256 / FastLog256 from the
    // net10.0 path. The only structural difference is that net471's BCL
    // Vector<int> has no left/right-shift operator (Vector.ShiftLeft is .NET 7+),
    // so the IEEE exponent manipulation uses integer MULTIPLY by 2^23 (== <<23)
    // and integer DIVIDE by 2^23 (== >>23, valid here because the operand is
    // always non-negative after the denormal clamp). The bit-reinterpret between
    // int and float bit patterns uses Vector.AsVectorSingle / Vector.AsVectorInt32,
    // both of which ship in the netstandard2.0 System.Numerics.Vectors that
    // net471 references.
    //
    // Accuracy matches the net10 polynomial's class (~1e-4 relative for exp,
    // ~1e-6 for log) — RyuJIT lacks hardware FMA on Vector<T> so the mul-add
    // chains carry one extra rounding step vs the net10 Fma path, which is well
    // within the activation tolerance.
    // ====================================================================

    private const int FloatExpBias = 127;
    private const int FloatMantissaBits = 23;
    private const int TwoPow23 = 1 << FloatMantissaBits; // 8388608

    /// <summary>Cephes-style exp(x) for one BCL <see cref="Vector{Single}"/> lane-block.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastExp(Vector<float> x)
    {
        // Clamp to the safe range (exp(-87.3) ~ 1e-38, exp(88.7) ~ 3.4e38).
        x = Vector.Min(new Vector<float>(88.7228f), Vector.Max(new Vector<float>(-87.3365f), x));

        var log2e = new Vector<float>(1.44269504088896341f); // 1/ln(2)
        var ln2hi = new Vector<float>(0.693359375f);
        var ln2lo = new Vector<float>(-2.12194440e-4f);
        var half = new Vector<float>(0.5f);
        var one = new Vector<float>(1.0f);

        // n = floor(x*log2e + 0.5) — round-to-nearest (ties up), matching Cephes.
        var z = x * log2e + half;
        var trunc = Vector.ConvertToSingle(Vector.ConvertToInt32(z)); // truncate toward zero
        // floor correction: where truncation overshot (z negative, frac != 0), subtract 1.
        var overshot = Vector.GreaterThan(trunc, z);
        var n = trunc - Vector.ConditionalSelect(overshot, one, Vector<float>.Zero);

        // r = x - n*ln2 (hi/lo split). No hardware FMA: plain mul-add.
        var r = x - n * ln2hi;
        r = r - n * ln2lo;

        // exp(r) via degree-6 Taylor poly (same coefficients as FastExp256), Horner form.
        var c2 = new Vector<float>(0.5f);
        var c3 = new Vector<float>(0.166666666666f);
        var c4 = new Vector<float>(0.041666666666f);
        var c5 = new Vector<float>(0.008333333333f);
        var c6 = new Vector<float>(0.001388888888f);

        var poly = c6;
        poly = poly * r + c5;
        poly = poly * r + c4;
        poly = poly * r + c3;
        poly = poly * r + c2;
        poly = poly * r + one;   // c1 = 1
        poly = poly * r + one;   // c0 = 1

        // 2^n via IEEE-754 exponent injection: ((n + 127) << 23) reinterpreted as float.
        // BCL net471 has no Vector<int> shift, so << 23 == * 2^23.
        var nInt = Vector.ConvertToInt32(n);
        var biased = (nInt + new Vector<int>(FloatExpBias)) * new Vector<int>(TwoPow23);
        var scale = Vector.AsVectorSingle(biased);

        return poly * scale;
    }

    /// <summary>Cephes-style natural log(x) for one BCL <see cref="Vector{Single}"/> lane-block.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastLog(Vector<float> x)
    {
        var one = new Vector<float>(1.0f);
        var zero = Vector<float>.Zero;

        var zeroMask = Vector.Equals(x, zero);
        var negativeMask = Vector.LessThan(x, zero);
        var infMask = Vector.Equals(x, new Vector<float>(float.PositiveInfinity));
        var nanMask = Vector.OnesComplement(Vector.Equals(x, x)); // NaN != NaN → all-ones where NaN

        // Clamp denormals to smallest normal positive for mantissa extraction.
        x = Vector.Max(x, new Vector<float>(1.17549435e-38f));

        // Extract exponent: e = (bits >> 23) - 127. Operand non-negative ⇒ divide == arithmetic shift.
        var xi = Vector.AsVectorInt32(x);
        var exponent = xi / new Vector<int>(TwoPow23) - new Vector<int>(FloatExpBias);
        var e = Vector.ConvertToSingle(exponent);

        // Mantissa in [1,2): (bits & 0x007FFFFF) | 0x3F800000.
        var mantissaBits = (xi & new Vector<int>(0x007FFFFF)) | new Vector<int>(0x3F800000);
        var m = Vector.AsVectorSingle(mantissaBits);

        // Range-shift to [sqrt(2)/2, sqrt(2)] for better conditioning.
        var needAdjust = Vector.GreaterThan(m, new Vector<float>(1.4142135623730951f));
        m = Vector.ConditionalSelect(needAdjust, m * new Vector<float>(0.5f), m);
        e = Vector.ConditionalSelect(needAdjust, e + one, e);

        var f = m - one;

        // Cephes minimax poly for log(1+f).
        var p0 = new Vector<float>(7.0376836292e-2f);
        var p1 = new Vector<float>(-1.1514610310e-1f);
        var p2 = new Vector<float>(1.1676998740e-1f);
        var p3 = new Vector<float>(-1.2420140846e-1f);
        var p4 = new Vector<float>(1.4249322787e-1f);
        var p5 = new Vector<float>(-1.6668057665e-1f);
        var p6 = new Vector<float>(2.0000714765e-1f);
        var p7 = new Vector<float>(-2.4999993993e-1f);
        var p8 = new Vector<float>(3.3333331174e-1f);

        var f2 = f * f;
        var poly = p0;
        poly = poly * f + p1;
        poly = poly * f + p2;
        poly = poly * f + p3;
        poly = poly * f + p4;
        poly = poly * f + p5;
        poly = poly * f + p6;
        poly = poly * f + p7;
        poly = poly * f + p8;
        poly = poly * (f2 * f);

        var ln2 = new Vector<float>(0.6931471805599453f);
        var result = e * ln2 + f;
        result = result + poly;
        result = result - new Vector<float>(0.5f) * f2;

        // Restore IEEE special values.
        result = Vector.ConditionalSelect(zeroMask, new Vector<float>(float.NegativeInfinity), result);
        result = Vector.ConditionalSelect(infMask, new Vector<float>(float.PositiveInfinity), result);
        result = Vector.ConditionalSelect(negativeMask, new Vector<float>(float.NaN), result);
        result = Vector.ConditionalSelect(nanMask, new Vector<float>(float.NaN), result);
        return result;
    }

    /// <summary>sigmoid(x) = 1/(1+exp(-x)) for one BCL <see cref="Vector{Single}"/> block.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastSigmoid(Vector<float> x)
    {
        var one = new Vector<float>(1.0f);
        return one / (one + FastExp(-x));
    }

    /// <summary>tanh(x) = 2*sigmoid(2x) - 1 for one BCL <see cref="Vector{Single}"/> block.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastTanh(Vector<float> x)
    {
        var one = new Vector<float>(1.0f);
        var two = new Vector<float>(2.0f);
        return two / (one + FastExp(-two * x)) - one;
    }

    // ---- span-level transcendental ops -------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Exp(ReadOnlySpan<float> src, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");
        int length = result.Length, lanes = Vector<float>.Count;
        int simd = length - (length % lanes);
        var s = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simd));
        var r = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simd));
        for (int v = 0; v < s.Length; v++) r[v] = FastExp(s[v]);
        for (int i = simd; i < length; i++) result[i] = (float)Math.Exp(src[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Log(ReadOnlySpan<float> src, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");
        int length = result.Length, lanes = Vector<float>.Count;
        int simd = length - (length % lanes);
        var s = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simd));
        var r = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simd));
        for (int v = 0; v < s.Length; v++) r[v] = FastLog(s[v]);
        for (int i = simd; i < length; i++) result[i] = (float)Math.Log(src[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Sigmoid(ReadOnlySpan<float> src, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");
        int length = result.Length, lanes = Vector<float>.Count;
        int simd = length - (length % lanes);
        var one = new Vector<float>(1.0f);
        var s = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simd));
        var r = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simd));
        for (int v = 0; v < s.Length; v++) r[v] = one / (one + FastExp(-s[v]));
        for (int i = simd; i < length; i++) result[i] = 1f / (1f + (float)Math.Exp(-src[i]));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Tanh(ReadOnlySpan<float> src, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");
        int length = result.Length, lanes = Vector<float>.Count;
        int simd = length - (length % lanes);
        var one = new Vector<float>(1.0f);
        var two = new Vector<float>(2.0f);
        var s = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simd));
        var r = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simd));
        // tanh(x) = 2*sigmoid(2x) - 1
        for (int v = 0; v < s.Length; v++) r[v] = two / (one + FastExp(-two * s[v])) - one;
        for (int i = simd; i < length; i++) result[i] = (float)Math.Tanh(src[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Swish(ReadOnlySpan<float> src, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");
        int length = result.Length, lanes = Vector<float>.Count;
        int simd = length - (length % lanes);
        var one = new Vector<float>(1.0f);
        var s = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simd));
        var r = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simd));
        for (int v = 0; v < s.Length; v++) r[v] = s[v] * (one / (one + FastExp(-s[v])));
        for (int i = simd; i < length; i++) result[i] = src[i] * (1f / (1f + (float)Math.Exp(-src[i])));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ELU(ReadOnlySpan<float> src, float alpha, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");
        int length = result.Length, lanes = Vector<float>.Count;
        int simd = length - (length % lanes);
        var one = new Vector<float>(1.0f);
        var zero = Vector<float>.Zero;
        var alphaVec = new Vector<float>(alpha);
        var s = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simd));
        var r = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simd));
        for (int v = 0; v < s.Length; v++)
        {
            var x = s[v];
            var neg = alphaVec * (FastExp(x) - one);
            r[v] = Vector.ConditionalSelect(Vector.GreaterThan(x, zero), x, neg);
        }
        for (int i = simd; i < length; i++)
            result[i] = src[i] > 0f ? src[i] : alpha * ((float)Math.Exp(src[i]) - 1f);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void GELU(ReadOnlySpan<float> src, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");
        int length = result.Length, lanes = Vector<float>.Count;
        int simd = length - (length % lanes);
        var half = new Vector<float>(0.5f);
        var one = new Vector<float>(1.0f);
        var two = new Vector<float>(2.0f);
        var sqrt2OverPi = new Vector<float>(0.7978845608028654f);
        var coeff = new Vector<float>(0.044715f);
        var s = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simd));
        var r = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simd));
        for (int v = 0; v < s.Length; v++)
        {
            var x = s[v];
            var inner = sqrt2OverPi * (x + coeff * x * x * x);
            // tanh(inner) = 2*sigmoid(2*inner) - 1
            var t = two / (one + FastExp(-two * inner)) - one;
            r[v] = half * x * (one + t);
        }
        for (int i = simd; i < length; i++)
        {
            float x = src[i];
            float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            result[i] = 0.5f * x * (1f + (float)Math.Tanh(inner));
        }
    }
}

#endif
