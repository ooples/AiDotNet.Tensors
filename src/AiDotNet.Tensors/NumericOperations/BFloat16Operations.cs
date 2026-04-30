// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Buffers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// <see cref="INumericOperations{T}"/> for <see cref="BFloat16"/>. Mirrors
/// <see cref="HalfOperations"/>: scalar ops round-trip through <see cref="float"/>;
/// vectorized ops convert to a pooled float buffer, dispatch through SimdKernels,
/// and convert back. Acceleration is "Cpu = true" — GPU paths in CUDA / ROCm /
/// MPS / OpenCL / Vulkan / WebGPU all surface their own native bfloat16 ops via
/// the per-backend kernel dispatch and read this type's <see cref="BFloat16.RawValue"/>
/// directly when staging a host buffer.
/// </summary>
public class BFloat16Operations : INumericOperations<BFloat16>
{
    public BFloat16 Zero => BFloat16.Zero;
    public BFloat16 One => BFloat16.One;
    public BFloat16 MinValue => BFloat16.MinValue;
    public BFloat16 MaxValue => BFloat16.MaxValue;
    public int PrecisionBits => 16;

    public BFloat16 Add(BFloat16 a, BFloat16 b) => BFloat16.FromFloat((float)a + (float)b);
    public BFloat16 Subtract(BFloat16 a, BFloat16 b) => BFloat16.FromFloat((float)a - (float)b);
    public BFloat16 Multiply(BFloat16 a, BFloat16 b) => BFloat16.FromFloat((float)a * (float)b);
    public BFloat16 Divide(BFloat16 a, BFloat16 b) => BFloat16.FromFloat((float)a / (float)b);
    public BFloat16 Negate(BFloat16 a) => -a;
    public BFloat16 Sqrt(BFloat16 v) => BFloat16.FromFloat((float)Math.Sqrt((float)v));
    public BFloat16 FromDouble(double v) => BFloat16.FromFloat((float)v);
    public int ToInt32(BFloat16 v) => (int)(float)v;
    public bool GreaterThan(BFloat16 a, BFloat16 b) => a > b;
    public bool LessThan(BFloat16 a, BFloat16 b) => a < b;
    public BFloat16 Abs(BFloat16 v) => BFloat16.FromFloat(Math.Abs((float)v));
    public BFloat16 Square(BFloat16 v) => BFloat16.FromFloat((float)v * (float)v);
    public BFloat16 Exp(BFloat16 v) => BFloat16.FromFloat((float)Math.Exp((float)v));
    public bool Equals(BFloat16 a, BFloat16 b) => a == b;
    public int Compare(BFloat16 a, BFloat16 b) => a.CompareTo(b);
    public BFloat16 Power(BFloat16 b, BFloat16 e) => BFloat16.FromFloat((float)Math.Pow((float)b, (float)e));
    public BFloat16 Log(BFloat16 v) => BFloat16.FromFloat((float)Math.Log((float)v));
    public bool GreaterThanOrEquals(BFloat16 a, BFloat16 b) => a >= b;
    public bool LessThanOrEquals(BFloat16 a, BFloat16 b) => a <= b;
    public BFloat16 Round(BFloat16 v) => BFloat16.FromFloat((float)Math.Round((float)v));
    public BFloat16 Floor(BFloat16 v) => BFloat16.FromFloat((float)Math.Floor((float)v));
    public BFloat16 Ceiling(BFloat16 v) => BFloat16.FromFloat((float)Math.Ceiling((float)v));
    public BFloat16 Frac(BFloat16 v) => BFloat16.FromFloat((float)v - (float)Math.Floor((float)v));
    public BFloat16 Sin(BFloat16 v) => BFloat16.FromFloat((float)Math.Sin((float)v));
    public BFloat16 Cos(BFloat16 v) => BFloat16.FromFloat((float)Math.Cos((float)v));
    public bool IsNaN(BFloat16 v) => BFloat16.IsNaN(v);
    public bool IsInfinity(BFloat16 v) => BFloat16.IsInfinity(v);
    public BFloat16 SignOrZero(BFloat16 v)
    {
        if (BFloat16.IsNaN(v)) return v;
        if (v > Zero) return One;
        if (v < Zero) return BFloat16.FromFloat(-1f);
        return Zero;
    }
    public float ToFloat(BFloat16 v) => (float)v;
    public BFloat16 FromFloat(float v) => BFloat16.FromFloat(v);
    public Half ToHalf(BFloat16 v) => (Half)(float)v;
    public BFloat16 FromHalf(Half v) => BFloat16.FromFloat((float)v);
    public double ToDouble(BFloat16 v) => (double)v;
    public bool SupportsCpuAcceleration => true;
    public bool SupportsGpuAcceleration => true;

    // ── Vectorized round-trip via pooled float buffer ──────────────

    private static void ToFloatBuf(ReadOnlySpan<BFloat16> src, float[] dst)
    {
        for (int i = 0; i < src.Length; i++) dst[i] = (float)src[i];
    }

    private static void FromFloatBuf(float[] src, Span<BFloat16> dst, int len)
    {
        for (int i = 0; i < len; i++) dst[i] = BFloat16.FromFloat(src[i]);
    }

    private static void RequireBinaryLengths(int xLen, int yLen, int dstLen, string op)
    {
        if (xLen != yLen || xLen != dstLen)
            throw new ArgumentException(
                $"{op} requires equal-length spans: x={xLen}, y={yLen}, dst={dstLen}.");
    }

    private static void RequireUnaryLengths(int xLen, int dstLen, string op)
    {
        if (xLen != dstLen)
            throw new ArgumentException(
                $"{op} requires equal-length spans: x={xLen}, dst={dstLen}.");
    }

    private void Lift(
        ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y, Span<BFloat16> dst,
        Action<float[], float[], float[], int> kernel)
    {
        int n = x.Length;
        // Pooled buffers are rented at length n but only the first n entries
        // are populated. With mismatched spans, the kernel would consume
        // stale tail values from the pool — silent wrong results.
        RequireBinaryLengths(n, y.Length, dst.Length, nameof(Lift));
        var xf = ArrayPool<float>.Shared.Rent(n);
        var yf = ArrayPool<float>.Shared.Rent(n);
        var df = ArrayPool<float>.Shared.Rent(n);
        try
        {
            ToFloatBuf(x, xf); ToFloatBuf(y, yf);
            kernel(xf, yf, df, n);
            FromFloatBuf(df, dst, n);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(xf);
            ArrayPool<float>.Shared.Return(yf);
            ArrayPool<float>.Shared.Return(df);
        }
    }

    private void Lift1(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst, Action<float[], float[], int> kernel)
    {
        int n = x.Length;
        RequireUnaryLengths(n, dst.Length, nameof(Lift1));
        var xf = ArrayPool<float>.Shared.Rent(n);
        var df = ArrayPool<float>.Shared.Rent(n);
        try
        {
            ToFloatBuf(x, xf);
            kernel(xf, df, n);
            FromFloatBuf(df, dst, n);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(xf);
            ArrayPool<float>.Shared.Return(df);
        }
    }

    public void Add(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y, Span<BFloat16> dst) =>
        Lift(x, y, dst, (a, b, c, n) => Engines.Simd.SimdKernels.VectorAdd(a.AsSpan(0, n), b.AsSpan(0, n), c.AsSpan(0, n)));
    public void Subtract(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y, Span<BFloat16> dst) =>
        Lift(x, y, dst, (a, b, c, n) => Engines.Simd.SimdKernels.VectorSubtract(a.AsSpan(0, n), b.AsSpan(0, n), c.AsSpan(0, n)));
    public void Multiply(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y, Span<BFloat16> dst) =>
        Lift(x, y, dst, (a, b, c, n) => Engines.Simd.SimdKernels.VectorMultiply(a.AsSpan(0, n), b.AsSpan(0, n), c.AsSpan(0, n)));
    public void Divide(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y, Span<BFloat16> dst) =>
        Lift(x, y, dst, (a, b, c, n) => Engines.Simd.SimdKernels.VectorDivide(a.AsSpan(0, n), b.AsSpan(0, n), c.AsSpan(0, n)));

    public BFloat16 Dot(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y)
    {
        int n = x.Length;
        if (y.Length != n)
            throw new ArgumentException($"Dot requires equal-length spans: x={n}, y={y.Length}.");
        var xf = ArrayPool<float>.Shared.Rent(n);
        var yf = ArrayPool<float>.Shared.Rent(n);
        try
        {
            ToFloatBuf(x, xf); ToFloatBuf(y, yf);
            return BFloat16.FromFloat(Engines.Simd.SimdKernels.DotProduct(xf.AsSpan(0, n), yf.AsSpan(0, n)));
        }
        finally { ArrayPool<float>.Shared.Return(xf); ArrayPool<float>.Shared.Return(yf); }
    }

    private BFloat16 LiftReduce(ReadOnlySpan<BFloat16> x, Func<float[], int, float> kernel)
    {
        int n = x.Length;
        var xf = ArrayPool<float>.Shared.Rent(n);
        try { ToFloatBuf(x, xf); return BFloat16.FromFloat(kernel(xf, n)); }
        finally { ArrayPool<float>.Shared.Return(xf); }
    }

    public BFloat16 Sum(ReadOnlySpan<BFloat16> x) =>
        LiftReduce(x, (a, n) => Engines.Simd.SimdKernels.Sum(a.AsSpan(0, n)));
    public BFloat16 Max(ReadOnlySpan<BFloat16> x) =>
        LiftReduce(x, (a, n) => Engines.Simd.SimdKernels.Max(a.AsSpan(0, n)));
    public BFloat16 Min(ReadOnlySpan<BFloat16> x) =>
        LiftReduce(x, (a, n) => Engines.Simd.SimdKernels.Min(a.AsSpan(0, n)));

    public void Exp(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst) =>
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.Exp(a.AsSpan(0, n), c.AsSpan(0, n)));
    public void Log(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst) =>
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.Log(a.AsSpan(0, n), c.AsSpan(0, n)));
    public void Tanh(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst) =>
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.Tanh(a.AsSpan(0, n), c.AsSpan(0, n)));
    public void Sigmoid(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst) =>
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.Sigmoid(a.AsSpan(0, n), c.AsSpan(0, n)));
    public void Log2(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst) =>
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.Log2(a.AsSpan(0, n), c.AsSpan(0, n)));
    public void SoftMax(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst) =>
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.SoftMax(a.AsSpan(0, n), c.AsSpan(0, n)));
    public void Sqrt(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst) =>
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.Sqrt(a.AsSpan(0, n), c.AsSpan(0, n)));
    public void Abs(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst) =>
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.Abs(a.AsSpan(0, n), c.AsSpan(0, n)));
    public void Negate(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst) =>
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.Negate(a.AsSpan(0, n), c.AsSpan(0, n)));

    public BFloat16 CosineSimilarity(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y)
    {
        int n = x.Length;
        if (y.Length != n)
            throw new ArgumentException($"CosineSimilarity requires equal-length spans: x={n}, y={y.Length}.");
        var xf = ArrayPool<float>.Shared.Rent(n);
        var yf = ArrayPool<float>.Shared.Rent(n);
        try
        {
            ToFloatBuf(x, xf); ToFloatBuf(y, yf);
            return BFloat16.FromFloat(Engines.Simd.SimdKernels.CosineSimilarity(xf.AsSpan(0, n), yf.AsSpan(0, n)));
        }
        finally { ArrayPool<float>.Shared.Return(xf); ArrayPool<float>.Shared.Return(yf); }
    }

    public void Fill(Span<BFloat16> dst, BFloat16 v) => dst.Fill(v);

    public void MultiplyScalar(ReadOnlySpan<BFloat16> x, BFloat16 scalar, Span<BFloat16> dst)
    {
        float s = (float)scalar;
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.MultiplyScalar(a.AsSpan(0, n), s, c.AsSpan(0, n)));
    }
    public void DivideScalar(ReadOnlySpan<BFloat16> x, BFloat16 scalar, Span<BFloat16> dst)
    {
        float s = (float)scalar;
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.DivideScalar(a.AsSpan(0, n), s, c.AsSpan(0, n)));
    }
    public void AddScalar(ReadOnlySpan<BFloat16> x, BFloat16 scalar, Span<BFloat16> dst)
    {
        float s = (float)scalar;
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.AddScalar(a.AsSpan(0, n), s, c.AsSpan(0, n)));
    }
    public void SubtractScalar(ReadOnlySpan<BFloat16> x, BFloat16 scalar, Span<BFloat16> dst)
    {
        float s = (float)scalar;
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.SubtractScalar(a.AsSpan(0, n), s, c.AsSpan(0, n)));
    }

    public void Clip(ReadOnlySpan<BFloat16> x, BFloat16 min, BFloat16 max, Span<BFloat16> dst)
    {
        float lo = (float)min, hi = (float)max;
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.Clamp(a.AsSpan(0, n), lo, hi, c.AsSpan(0, n)));
    }

    public void Pow(ReadOnlySpan<BFloat16> x, BFloat16 power, Span<BFloat16> dst)
    {
        float p = (float)power;
        Lift1(x, dst, (a, c, n) => Engines.Simd.SimdKernels.Pow(a.AsSpan(0, n), p, c.AsSpan(0, n)));
    }

    public void Copy(ReadOnlySpan<BFloat16> source, Span<BFloat16> destination) => source.CopyTo(destination);

    public void Floor(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst)
    {
        for (int i = 0; i < x.Length; i++) dst[i] = BFloat16.FromFloat((float)Math.Floor((float)x[i]));
    }
    public void Ceiling(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst)
    {
        for (int i = 0; i < x.Length; i++) dst[i] = BFloat16.FromFloat((float)Math.Ceiling((float)x[i]));
    }
    public void Frac(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst)
    {
        for (int i = 0; i < x.Length; i++) dst[i] = BFloat16.FromFloat((float)x[i] - (float)Math.Floor((float)x[i]));
    }
    public void Sin(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst)
    {
        for (int i = 0; i < x.Length; i++) dst[i] = BFloat16.FromFloat((float)Math.Sin((float)x[i]));
    }
    public void Cos(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst)
    {
        for (int i = 0; i < x.Length; i++) dst[i] = BFloat16.FromFloat((float)Math.Cos((float)x[i]));
    }

    public void MultiplyAdd(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y, BFloat16 scalar, Span<BFloat16> dst)
        => VectorizedOperationsFallback.MultiplyAdd(this, x, y, scalar, dst);

    public void ToFloatSpan(ReadOnlySpan<BFloat16> src, Span<float> dst)
    {
        for (int i = 0; i < src.Length; i++) dst[i] = (float)src[i];
    }
    public void FromFloatSpan(ReadOnlySpan<float> src, Span<BFloat16> dst)
    {
        for (int i = 0; i < src.Length; i++) dst[i] = BFloat16.FromFloat(src[i]);
    }
    public void ToHalfSpan(ReadOnlySpan<BFloat16> src, Span<Half> dst)
    {
        for (int i = 0; i < src.Length; i++) dst[i] = (Half)(float)src[i];
    }
    public void FromHalfSpan(ReadOnlySpan<Half> src, Span<BFloat16> dst)
    {
        for (int i = 0; i < src.Length; i++) dst[i] = BFloat16.FromFloat((float)src[i]);
    }

    public void LeakyReLU(ReadOnlySpan<BFloat16> x, BFloat16 alpha, Span<BFloat16> dst)
        => VectorizedOperationsFallback.LeakyReLU(this, x, alpha, dst);
    public void GELU(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst)
        => VectorizedOperationsFallback.GELU(this, x, dst);
    public void Mish(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst)
        => VectorizedOperationsFallback.Mish(this, x, dst);
    public void Swish(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst)
        => VectorizedOperationsFallback.Swish(this, x, dst);
    public void ELU(ReadOnlySpan<BFloat16> x, BFloat16 alpha, Span<BFloat16> dst)
        => VectorizedOperationsFallback.ELU(this, x, alpha, dst);
    public void ReLU(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst)
        => VectorizedOperationsFallback.ReLU(this, x, dst);
}
