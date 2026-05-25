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
            // Propagate NaN (any NaN in input -> NaN result), matching NumPy/PyTorch
            // max and giving a lane-/position-independent, deterministic contract.
            // Scalar `x > m` skips NaNs, so detect them explicitly.
            float m = src[0];
            bool nan = float.IsNaN(m);
            for (int i = 1; i < length; i++)
            {
                float x = src[i];
                if (float.IsNaN(x)) nan = true;
                else if (x > m) m = x;
            }
            return nan ? float.NaN : m;
        }

        int simdElements = length - (length % lanes);
        var sVecs = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simdElements));

        var accum = sVecs[0];
        // Vector.Equals(v, v) is all-ones for non-NaN lanes and 0 for NaN lanes, so
        // AND-accumulating it across the raw inputs detects NaN independently of which
        // lane Vector.Max happens to keep (Vector.Max's NaN behavior is unspecified).
        var notNan = Vector.Equals(sVecs[0], sVecs[0]);
        for (int v = 1; v < sVecs.Length; v++)
        {
            var cur = sVecs[v];
            accum = Vector.Max(accum, cur);
            notNan = Vector.BitwiseAnd(notNan, Vector.Equals(cur, cur));
        }

        // A 0 lane in notNan (reinterpreted as int) means a NaN was seen somewhere.
        bool anyNaN = !Vector.EqualsAll(Vector.AsVectorInt32(notNan), new Vector<int>(-1));

        float max = accum[0];
        for (int j = 1; j < lanes; j++)
            if (accum[j] > max) max = accum[j];

        for (int i = simdElements; i < length; i++)
        {
            float x = src[i];
            if (float.IsNaN(x)) anyNaN = true;
            else if (x > max) max = x;
        }
        return anyNaN ? float.NaN : max;
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
            // Propagate NaN (any NaN -> NaN), matching NumPy/PyTorch min. Scalar
            // `x < m` skips NaNs, so detect them explicitly for a deterministic result.
            float m = src[0];
            bool nan = float.IsNaN(m);
            for (int i = 1; i < length; i++)
            {
                float x = src[i];
                if (float.IsNaN(x)) nan = true;
                else if (x < m) m = x;
            }
            return nan ? float.NaN : m;
        }

        int simdElements = length - (length % lanes);
        var sVecs = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simdElements));

        var accum = sVecs[0];
        // notNan AND-accumulator detects NaN independently of which lane Vector.Min keeps.
        var notNan = Vector.Equals(sVecs[0], sVecs[0]);
        for (int v = 1; v < sVecs.Length; v++)
        {
            var cur = sVecs[v];
            accum = Vector.Min(accum, cur);
            notNan = Vector.BitwiseAnd(notNan, Vector.Equals(cur, cur));
        }

        bool anyNaN = !Vector.EqualsAll(Vector.AsVectorInt32(notNan), new Vector<int>(-1));

        float min = accum[0];
        for (int j = 1; j < lanes; j++)
            if (accum[j] < min) min = accum[j];

        for (int i = simdElements; i < length; i++)
        {
            float x = src[i];
            if (float.IsNaN(x)) anyNaN = true;
            else if (x < min) min = x;
        }
        return anyNaN ? float.NaN : min;
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
            // Propagate NaN (any NaN -> NaN), matching NumPy/PyTorch max.
            double m = src[0];
            bool nan = double.IsNaN(m);
            for (int i = 1; i < length; i++)
            {
                double x = src[i];
                if (double.IsNaN(x)) nan = true;
                else if (x > m) m = x;
            }
            return nan ? double.NaN : m;
        }

        int simdElements = length - (length % lanes);
        var sVecs = MemoryMarshal.Cast<double, Vector<double>>(src.Slice(0, simdElements));

        var accum = sVecs[0];
        // notNan AND-accumulator detects NaN independently of which lane Vector.Max keeps.
        var notNan = Vector.Equals(sVecs[0], sVecs[0]);
        for (int v = 1; v < sVecs.Length; v++)
        {
            var cur = sVecs[v];
            accum = Vector.Max(accum, cur);
            notNan = Vector.BitwiseAnd(notNan, Vector.Equals(cur, cur));
        }

        bool anyNaN = !Vector.EqualsAll(Vector.AsVectorInt64(notNan), new Vector<long>(-1L));

        double max = accum[0];
        for (int j = 1; j < lanes; j++)
            if (accum[j] > max) max = accum[j];

        for (int i = simdElements; i < length; i++)
        {
            double x = src[i];
            if (double.IsNaN(x)) anyNaN = true;
            else if (x > max) max = x;
        }
        return anyNaN ? double.NaN : max;
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
            // Propagate NaN (any NaN -> NaN), matching NumPy/PyTorch min.
            double m = src[0];
            bool nan = double.IsNaN(m);
            for (int i = 1; i < length; i++)
            {
                double x = src[i];
                if (double.IsNaN(x)) nan = true;
                else if (x < m) m = x;
            }
            return nan ? double.NaN : m;
        }

        int simdElements = length - (length % lanes);
        var sVecs = MemoryMarshal.Cast<double, Vector<double>>(src.Slice(0, simdElements));

        var accum = sVecs[0];
        // notNan AND-accumulator detects NaN independently of which lane Vector.Min keeps.
        var notNan = Vector.Equals(sVecs[0], sVecs[0]);
        for (int v = 1; v < sVecs.Length; v++)
        {
            var cur = sVecs[v];
            accum = Vector.Min(accum, cur);
            notNan = Vector.BitwiseAnd(notNan, Vector.Equals(cur, cur));
        }

        bool anyNaN = !Vector.EqualsAll(Vector.AsVectorInt64(notNan), new Vector<long>(-1L));

        double min = accum[0];
        for (int j = 1; j < lanes; j++)
            if (accum[j] < min) min = accum[j];

        for (int i = simdElements; i < length; i++)
        {
            double x = src[i];
            if (double.IsNaN(x)) anyNaN = true;
            else if (x < min) min = x;
        }
        return anyNaN ? double.NaN : min;
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

    /// <summary>
    /// Cephes-style sin(x) for one BCL <see cref="Vector{Single}"/> block — faithful port of
    /// SimdKernels.FastSin256 (2/π range reduction + quadrant selection + odd/even minimax).
    /// Accurate for bounded arguments; callers route huge-magnitude lanes to scalar Math.Sin.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastSin(Vector<float> x)
    {
        var twoOverPi = new Vector<float>(0.6366197723675814f);
        var piOver2Hi = new Vector<float>(1.5707963267341256f);
        var piOver2Lo = new Vector<float>(6.077100506303966e-11f);
        var one = new Vector<float>(1.0f);

        // sign(x) bit, then work with |x|.
        var signBit = Vector.AsVectorInt32(x) & new Vector<int>(unchecked((int)0x80000000));
        x = Vector.Abs(x);

        // j = round(x * 2/pi) via floor(t + 0.5).
        var t = x * twoOverPi + new Vector<float>(0.5f);
        var jf = Vector.ConvertToSingle(Vector.ConvertToInt32(t));
        jf -= Vector.ConditionalSelect(Vector.GreaterThan(jf, t), one, Vector<float>.Zero);
        var jInt = Vector.ConvertToInt32(jf);

        // r = x - j*pi/2 (hi/lo split).
        var r = x - jf * piOver2Hi;
        r = r - jf * piOver2Lo;

        var quadrant = jInt & new Vector<int>(3);
        var needNeg = Vector.GreaterThan(quadrant, new Vector<int>(1));            // all-ones where j%4 >= 2
        var negMask = needNeg & new Vector<int>(unchecked((int)0x80000000));
        var useCosPoly = Vector.Equals(jInt & new Vector<int>(1), new Vector<int>(1));

        var r2 = r * r;

        // sin (odd): r - r^3/6 + r^5/120 - r^7/5040
        var sinP = new Vector<float>(-1.9515295891e-4f);
        sinP = sinP * r2 + new Vector<float>(8.3321608736e-3f);
        sinP = sinP * r2 + new Vector<float>(-1.6666654611e-1f);
        sinP = sinP * (r2 * r) + r;

        // cos (even): 1 - r^2/2 + r^4/24 - r^6/720
        var cosP = new Vector<float>(-1.3888377460e-3f);
        cosP = cosP * r2 + new Vector<float>(4.1666638908e-2f);
        cosP = cosP * r2 + new Vector<float>(-0.5f);
        cosP = cosP * r2 + one;

        // Select sin/cos poly by reinterpreting the int mask, then apply sign flips.
        var result = Vector.ConditionalSelect(Vector.AsVectorSingle(useCosPoly), cosP, sinP);
        var flipBits = negMask ^ signBit;
        return Vector.AsVectorSingle(Vector.AsVectorInt32(result) ^ flipBits);
    }

    /// <summary>cos(x) = sin(x + π/2) for one BCL <see cref="Vector{Single}"/> block.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastCos(Vector<float> x)
        => FastSin(x + new Vector<float>(1.5707963267948966f));

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

    /// <summary>
    /// exp(x) for one BCL <see cref="Vector{Double}"/> block — double-precision
    /// counterpart of <see cref="FastExp"/>. Range-reduces x = n·ln2 + r and
    /// evaluates a degree-7 Taylor of exp(r) on r ∈ [-ln2/2, ln2/2] (error
    /// ~5e-8, well inside the fast-poly accuracy class). 2^n is injected via the
    /// IEEE-754 exponent field: ((n + 1023) · 2^52) reinterpreted as double —
    /// integer multiply (not shift) so it works on the pre-.NET7 BCL Vector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<double> FastExpDouble(Vector<double> x)
    {
        x = Vector.Min(new Vector<double>(709.0), Vector.Max(new Vector<double>(-708.0), x));

        var log2e = new Vector<double>(1.4426950408889634);
        var ln2hi = new Vector<double>(0.6931471803691238);   // hi/lo split of ln2
        var ln2lo = new Vector<double>(1.9082149292705877e-10);
        var half = new Vector<double>(0.5);
        var one = new Vector<double>(1.0);

        // n = round-to-nearest(x*log2e) via truncate + floor-correction.
        var z = x * log2e + half;
        var trunc = Vector.ConvertToDouble(Vector.ConvertToInt64(z));
        var overshot = Vector.GreaterThan(trunc, z);
        var n = trunc - Vector.ConditionalSelect(overshot, one, Vector<double>.Zero);

        // r = x - n*ln2 (hi/lo).
        var r = x - n * ln2hi;
        r = r - n * ln2lo;

        // exp(r), degree-7 Taylor (Horner). c_k = 1/k!.
        var poly = new Vector<double>(1.0 / 5040.0);
        poly = poly * r + new Vector<double>(1.0 / 720.0);
        poly = poly * r + new Vector<double>(1.0 / 120.0);
        poly = poly * r + new Vector<double>(1.0 / 24.0);
        poly = poly * r + new Vector<double>(1.0 / 6.0);
        poly = poly * r + half;
        poly = poly * r + one;
        poly = poly * r + one;

        // 2^n via exponent injection: (n + 1023) << 52 == (n + 1023) * 2^52.
        var nLong = Vector.ConvertToInt64(n);
        var biased = (nLong + new Vector<long>(1023L)) * new Vector<long>(1L << 52);
        var scale = Vector.AsVectorDouble(biased);

        return poly * scale;
    }

    /// <summary>tanh(x) for one BCL <see cref="Vector{Double}"/> block.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<double> FastTanhDouble(Vector<double> x)
    {
        var one = new Vector<double>(1.0);
        var two = new Vector<double>(2.0);
        return two / (one + FastExpDouble(-two * x)) - one;
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

    // Above this |x| the float ulp exceeds π/2, so 2/π range reduction loses all
    // precision — those lanes fall back to libm Math.Sin/Cos.
    private const float TrigReductionLimit = 105414f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Sin(ReadOnlySpan<float> src, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");
        int length = result.Length, lanes = Vector<float>.Count;
        int simd = length - (length % lanes);
        var limit = new Vector<float>(TrigReductionLimit);
        var s = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simd));
        var r = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simd));
        for (int v = 0; v < s.Length; v++)
        {
            var x = s[v];
            // If any lane is out of the reduction range, do the whole block in scalar.
            if (Vector.GreaterThanAny(Vector.Abs(x), limit))
            {
                int baseIdx = v * lanes;
                for (int l = 0; l < lanes; l++) result[baseIdx + l] = (float)Math.Sin(src[baseIdx + l]);
            }
            else
            {
                r[v] = FastSin(x);
            }
        }
        for (int i = simd; i < length; i++) result[i] = (float)Math.Sin(src[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Cos(ReadOnlySpan<float> src, Span<float> result)
    {
        if (src.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");
        int length = result.Length, lanes = Vector<float>.Count;
        int simd = length - (length % lanes);
        // FastCos(x) evaluates FastSin(x + π/2), so the SHIFTED argument is what must
        // stay inside the 2/π reduction window. Tighten the cutoff by π/2 (use the
        // larger |·| margin) so inputs in the top π/2 band fall back to libm rather
        // than feeding FastSin an out-of-range argument.
        var limit = new Vector<float>(TrigReductionLimit - 1.5707964f);
        var s = MemoryMarshal.Cast<float, Vector<float>>(src.Slice(0, simd));
        var r = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simd));
        for (int v = 0; v < s.Length; v++)
        {
            var x = s[v];
            if (Vector.GreaterThanAny(Vector.Abs(x), limit))
            {
                int baseIdx = v * lanes;
                for (int l = 0; l < lanes; l++) result[baseIdx + l] = (float)Math.Cos(src[baseIdx + l]);
            }
            else
            {
                r[v] = FastCos(x);
            }
        }
        for (int i = simd; i < length; i++) result[i] = (float)Math.Cos(src[i]);
    }

    /// <summary>
    /// pow(x, exponent) = exp(exponent * log(x)) — the SIMD fast path is only valid for NORMAL,
    /// FINITE, POSITIVE bases. exponent == 0 (libm returns 1 for every base, incl. ±∞ and NaN),
    /// a non-finite exponent, and any block containing a non-positive / subnormal / non-finite base
    /// all route to libm <see cref="Math.Pow(double, double)"/> so the result matches MathF.Pow.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Pow(ReadOnlySpan<float> baseValues, float exponent, Span<float> result)
    {
        if (baseValues.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");
        int length = result.Length;

        // pow(x, 0) == 1 for ALL x (including ±∞ and NaN). The fast path would give
        // exp(0 * log(+∞)) = exp(0 * +∞) = exp(NaN) = NaN, so special-case it.
        if (exponent == 0f)
        {
            result.Slice(0, length).Fill(1f);
            return;
        }
        // A non-finite exponent makes exp(y·log x) unreliable — defer the whole span to libm.
        if (float.IsNaN(exponent) || float.IsInfinity(exponent))
        {
            for (int i = 0; i < length; i++) result[i] = (float)Math.Pow(baseValues[i], exponent);
            return;
        }

        int lanes = Vector<float>.Count;
        int simd = length - (length % lanes);
        var expVec = new Vector<float>(exponent);
        var minNormal = new Vector<float>(1.17549435e-38f); // smallest normal positive float
        var maxFinite = new Vector<float>(float.MaxValue);
        var s = MemoryMarshal.Cast<float, Vector<float>>(baseValues.Slice(0, simd));
        var r = MemoryMarshal.Cast<float, Vector<float>>(result.Slice(0, simd));
        for (int v = 0; v < s.Length; v++)
        {
            var b = s[v];
            // FastLog clamps subnormals up to MinNormal and FastExp clamps infinities, so the
            // fast path is wrong for non-positive, subnormal, or non-finite bases. Detect any
            // such lane and route the whole block to libm:
            //   b < MinNormal   → catches <= 0 AND positive subnormals
            //   b > MaxFinite   → catches +∞
            //   b != b          → catches NaN
            var bad = Vector.BitwiseOr(
                Vector.BitwiseOr(Vector.LessThan(b, minNormal), Vector.GreaterThan(b, maxFinite)),
                Vector.OnesComplement(Vector.Equals(b, b)));
            if (!Vector.EqualsAll(Vector.AsVectorInt32(bad), Vector<int>.Zero))
            {
                int baseIdx = v * lanes;
                for (int l = 0; l < lanes; l++) result[baseIdx + l] = (float)Math.Pow(baseValues[baseIdx + l], exponent);
            }
            else
            {
                r[v] = FastExp(expVec * FastLog(b));
            }
        }
        for (int i = simd; i < length; i++) result[i] = (float)Math.Pow(baseValues[i], exponent);
    }
}

#endif
