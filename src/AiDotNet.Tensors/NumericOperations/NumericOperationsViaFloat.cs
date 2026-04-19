using System.Buffers;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Shared base for narrow-width numeric types (FP8 E4M3 / E5M2, and
/// future mini-floats) that implement <see cref="INumericOperations{T}"/>
/// via <c>float</c> upcast. Every scalar op upcasts the inputs, does the
/// math in <c>float</c>, and re-quantizes the output. Every vectorized
/// op rents a float scratch buffer from <see cref="ArrayPool{T}"/>,
/// routes the work through SIMD <see cref="SimdKernels"/>, and writes
/// the quantized result back.
///
/// <para>Concrete subclasses provide two methods — <see cref="ToFloatImpl"/>
/// and <see cref="FromFloatImpl"/> — plus <see cref="Zero"/>,
/// <see cref="One"/>, and the NaN/Inf predicates when they diverge from
/// the float fallback.</para>
///
/// <para>Why float and not double: FP8 is purely a storage format that
/// upsizes to FP16 / FP32 for compute. Going through double would be
/// precision-overkill and slower (SIMD lanes are fewer per instruction).</para>
/// </summary>
public abstract class NumericOperationsViaFloat<T> : INumericOperations<T>
    where T : unmanaged
{
    /// <summary>Upcast <paramref name="value"/> to float.</summary>
    protected abstract float ToFloatImpl(T value);

    /// <summary>Quantize <paramref name="value"/> from float to T.</summary>
    protected abstract T FromFloatImpl(float value);

    public abstract T Zero { get; }
    public abstract T One { get; }
    public abstract T MinValue { get; }
    public abstract T MaxValue { get; }
    public abstract int PrecisionBits { get; }

    // ──────────── scalar ops (all via float) ────────────

    public virtual T Add(T a, T b)      => FromFloatImpl(ToFloatImpl(a) + ToFloatImpl(b));
    public virtual T Subtract(T a, T b) => FromFloatImpl(ToFloatImpl(a) - ToFloatImpl(b));
    public virtual T Multiply(T a, T b) => FromFloatImpl(ToFloatImpl(a) * ToFloatImpl(b));
    public virtual T Divide(T a, T b)   => FromFloatImpl(ToFloatImpl(a) / ToFloatImpl(b));
    public virtual T Negate(T a)        => FromFloatImpl(-ToFloatImpl(a));
    public virtual T Sqrt(T value)      => FromFloatImpl((float)Math.Sqrt(ToFloatImpl(value)));
    public virtual T FromDouble(double value) => FromFloatImpl((float)value);
    public virtual int ToInt32(T value)        => (int)ToFloatImpl(value);
    public virtual bool GreaterThan(T a, T b)  => ToFloatImpl(a) >  ToFloatImpl(b);
    public virtual bool LessThan(T a, T b)     => ToFloatImpl(a) <  ToFloatImpl(b);
    public virtual T Abs(T value)              => FromFloatImpl(Math.Abs(ToFloatImpl(value)));
    public virtual T Square(T value) { float f = ToFloatImpl(value); return FromFloatImpl(f * f); }
    public virtual T Exp(T value)              => FromFloatImpl((float)Math.Exp(ToFloatImpl(value)));
    public virtual bool Equals(T a, T b)       => ToFloatImpl(a) == ToFloatImpl(b);
    public virtual int Compare(T a, T b)       => ToFloatImpl(a).CompareTo(ToFloatImpl(b));
    public virtual T Power(T baseValue, T exponent)
        => FromFloatImpl((float)Math.Pow(ToFloatImpl(baseValue), ToFloatImpl(exponent)));
    public virtual T Log(T value)              => FromFloatImpl((float)Math.Log(ToFloatImpl(value)));
    public virtual bool GreaterThanOrEquals(T a, T b) => ToFloatImpl(a) >= ToFloatImpl(b);
    public virtual bool LessThanOrEquals(T a, T b)    => ToFloatImpl(a) <= ToFloatImpl(b);
    public virtual T Round(T value)            => FromFloatImpl((float)Math.Round(ToFloatImpl(value)));
    public virtual T Floor(T value)            => FromFloatImpl((float)Math.Floor(ToFloatImpl(value)));
    public virtual T Ceiling(T value)          => FromFloatImpl((float)Math.Ceiling(ToFloatImpl(value)));
    public virtual T Frac(T value)             { float f = ToFloatImpl(value); return FromFloatImpl(f - (float)Math.Floor(f)); }
    public virtual T Sin(T value)              => FromFloatImpl((float)Math.Sin(ToFloatImpl(value)));
    public virtual T Cos(T value)              => FromFloatImpl((float)Math.Cos(ToFloatImpl(value)));
    public virtual bool IsNaN(T value)         => float.IsNaN(ToFloatImpl(value));
    public virtual bool IsInfinity(T value)    => float.IsInfinity(ToFloatImpl(value));
    public virtual T SignOrZero(T value)
    {
        float f = ToFloatImpl(value);
        if (float.IsNaN(f)) return value;
        if (f > 0f) return One;
        if (f < 0f) return FromFloatImpl(-1f);
        return Zero;
    }
    public virtual float ToFloat(T value)      => ToFloatImpl(value);
    public virtual T FromFloat(float value)    => FromFloatImpl(value);
    public virtual Half ToHalf(T value)        => (Half)ToFloatImpl(value);
    public virtual T FromHalf(Half value)      => FromFloatImpl((float)value);
    public virtual double ToDouble(T value)    => ToFloatImpl(value);

    public virtual bool SupportsCpuAcceleration => true;
    public virtual bool SupportsGpuAcceleration => false;

    // ──────────── vectorized ops (via float SIMD + pool) ────────────

    /// <summary>
    /// Convert a span of T to float using <see cref="ToFloatImpl"/>.
    /// Subclasses can override with a bulk converter (e.g. SIMD gather)
    /// but the per-element fallback works correctly for any type.
    /// </summary>
    protected virtual void ToFloatSpanImpl(ReadOnlySpan<T> src, Span<float> dst)
    {
        for (int i = 0; i < src.Length; i++) dst[i] = ToFloatImpl(src[i]);
    }

    /// <summary>Inverse of <see cref="ToFloatSpanImpl"/>.</summary>
    protected virtual void FromFloatSpanImpl(ReadOnlySpan<float> src, Span<T> dst)
    {
        for (int i = 0; i < src.Length; i++) dst[i] = FromFloatImpl(src[i]);
    }

    // Binary ops: rent two float scratch buffers, SIMD-compute, re-pack.
    public virtual void Add(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => BinaryViaFloat(x, y, destination, SimdKernels.VectorAdd);
    public virtual void Subtract(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => BinaryViaFloat(x, y, destination, SimdKernels.VectorSubtract);
    public virtual void Multiply(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => BinaryViaFloat(x, y, destination, SimdKernels.VectorMultiply);
    public virtual void Divide(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => BinaryViaFloat(x, y, destination, SimdKernels.VectorDivide);

    private delegate void BinaryFloatKernel(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> dst);
    private void BinaryViaFloat(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination, BinaryFloatKernel kernel)
    {
        // Pooled scratch buffers are sized to x.Length and indexed with
        // AsSpan(0, len). Without these checks a short y would leave the
        // tail of yf filled with stale ArrayPool contents and the SIMD
        // kernel would compute garbage; a short destination would surface
        // as IndexOutOfRangeException instead of a documented ArgumentException.
        if (y.Length != x.Length)
            throw new ArgumentException(
                $"y length {y.Length} must match x length {x.Length}.", nameof(y));
        if (destination.Length < x.Length)
            throw new ArgumentException(
                $"destination length {destination.Length} must be >= x length {x.Length}.",
                nameof(destination));

        int len = x.Length;
        float[] xf = ArrayPool<float>.Shared.Rent(len);
        float[] yf = ArrayPool<float>.Shared.Rent(len);
        float[] df = ArrayPool<float>.Shared.Rent(len);
        try
        {
            ToFloatSpanImpl(x, xf.AsSpan(0, len));
            ToFloatSpanImpl(y, yf.AsSpan(0, len));
            kernel(xf.AsSpan(0, len), yf.AsSpan(0, len), df.AsSpan(0, len));
            FromFloatSpanImpl(df.AsSpan(0, len), destination);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(xf);
            ArrayPool<float>.Shared.Return(yf);
            ArrayPool<float>.Shared.Return(df);
        }
    }

    // Reduction ops: rent one float scratch, compute, quantize result.
    public virtual T Dot(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        // Same rationale as BinaryViaFloat — rented scratch is sized to
        // x.Length, so a short y would read stale pool data and a long y
        // would silently ignore the overflow. Validate before renting.
        if (y.Length != x.Length)
            throw new ArgumentException(
                $"y length {y.Length} must match x length {x.Length}.", nameof(y));

        int len = x.Length;
        float[] xf = ArrayPool<float>.Shared.Rent(len);
        float[] yf = ArrayPool<float>.Shared.Rent(len);
        try
        {
            ToFloatSpanImpl(x, xf.AsSpan(0, len));
            ToFloatSpanImpl(y, yf.AsSpan(0, len));
            return FromFloatImpl(SimdKernels.DotProduct(xf.AsSpan(0, len), yf.AsSpan(0, len)));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(xf);
            ArrayPool<float>.Shared.Return(yf);
        }
    }

    public virtual T Sum(ReadOnlySpan<T> x) => UnaryReduction(x, SimdKernels.Sum);
    public virtual T Max(ReadOnlySpan<T> x) => UnaryReduction(x, SimdKernels.Max);
    public virtual T Min(ReadOnlySpan<T> x) => UnaryReduction(x, SimdKernels.Min);

    private delegate float UnaryReductionKernel(ReadOnlySpan<float> x);
    private T UnaryReduction(ReadOnlySpan<T> x, UnaryReductionKernel kernel)
    {
        int len = x.Length;
        float[] xf = ArrayPool<float>.Shared.Rent(len);
        try
        {
            ToFloatSpanImpl(x, xf.AsSpan(0, len));
            return FromFloatImpl(kernel(xf.AsSpan(0, len)));
        }
        finally { ArrayPool<float>.Shared.Return(xf); }
    }

    // Unary elementwise ops: convert in, kernel, convert out.
    // Unary ops: route through SimdKernels where we have a float kernel;
    // scalar fallback for the rest.
    public virtual void Exp(ReadOnlySpan<T> x, Span<T> destination)     => UnaryViaFloat(x, destination, (ReadOnlySpan<float> xf, Span<float> df) => SimdKernels.Exp(xf, df));
    public virtual void Log(ReadOnlySpan<T> x, Span<T> destination)     => UnaryViaFloatScalar(x, destination, (float f) => (float)Math.Log(f));
    public virtual void Log2(ReadOnlySpan<T> x, Span<T> destination)    => UnaryViaFloat(x, destination, (ReadOnlySpan<float> xf, Span<float> df) => SimdKernels.Log2(xf, df));
    public virtual void Tanh(ReadOnlySpan<T> x, Span<T> destination)    => UnaryViaFloat(x, destination, (ReadOnlySpan<float> xf, Span<float> df) => SimdKernels.Tanh(xf, df));
    public virtual void Sigmoid(ReadOnlySpan<T> x, Span<T> destination) => UnaryViaFloat(x, destination, (ReadOnlySpan<float> xf, Span<float> df) => SimdKernels.Sigmoid(xf, df));
    public virtual void SoftMax(ReadOnlySpan<T> x, Span<T> destination) => UnaryViaFloat(x, destination, (ReadOnlySpan<float> xf, Span<float> df) => SimdKernels.SoftMax(xf, df));
    public virtual void Sqrt(ReadOnlySpan<T> x, Span<T> destination)    => UnaryViaFloatScalar(x, destination, (float f) => (float)Math.Sqrt(f));
    public virtual void Abs(ReadOnlySpan<T> x, Span<T> destination)     => UnaryViaFloatScalar(x, destination, Math.Abs);
    public virtual void Negate(ReadOnlySpan<T> x, Span<T> destination)  => UnaryViaFloatScalar(x, destination, (float f) => -f);
    public virtual void Floor(ReadOnlySpan<T> x, Span<T> destination)   => UnaryViaFloatScalar(x, destination, (float f) => (float)Math.Floor(f));
    public virtual void Ceiling(ReadOnlySpan<T> x, Span<T> destination) => UnaryViaFloatScalar(x, destination, (float f) => (float)Math.Ceiling(f));
    public virtual void Frac(ReadOnlySpan<T> x, Span<T> destination)    => UnaryViaFloatScalar(x, destination, (float f) => f - (float)Math.Floor(f));
    public virtual void Sin(ReadOnlySpan<T> x, Span<T> destination)     => UnaryViaFloatScalar(x, destination, (float f) => (float)Math.Sin(f));
    public virtual void Cos(ReadOnlySpan<T> x, Span<T> destination)     => UnaryViaFloatScalar(x, destination, (float f) => (float)Math.Cos(f));
    public virtual void ReLU(ReadOnlySpan<T> x, Span<T> destination)    => UnaryViaFloat(x, destination, (ReadOnlySpan<float> xf, Span<float> df) => SimdKernels.ReLU(xf, df));
    public virtual void GELU(ReadOnlySpan<T> x, Span<T> destination)    => UnaryViaFloat(x, destination, (ReadOnlySpan<float> xf, Span<float> df) => SimdKernels.GELU(xf, df));
    public virtual void Mish(ReadOnlySpan<T> x, Span<T> destination)    => UnaryViaFloatScalar(x, destination, MishScalar);
    public virtual void Swish(ReadOnlySpan<T> x, Span<T> destination)   => UnaryViaFloatScalar(x, destination, SwishScalar);

    private delegate void UnaryKernel(ReadOnlySpan<float> x, Span<float> dst);
    private void UnaryViaFloat(ReadOnlySpan<T> x, Span<T> destination, UnaryKernel kernel)
    {
        // FromFloatSpanImpl is indexed up to x.Length; short destination
        // would surface as IndexOutOfRangeException with no hint to the
        // caller. Fail fast with the documented ArgumentException.
        if (destination.Length < x.Length)
            throw new ArgumentException(
                $"destination length {destination.Length} must be >= x length {x.Length}.",
                nameof(destination));

        int len = x.Length;
        float[] xf = ArrayPool<float>.Shared.Rent(len);
        float[] df = ArrayPool<float>.Shared.Rent(len);
        try
        {
            ToFloatSpanImpl(x, xf.AsSpan(0, len));
            kernel(xf.AsSpan(0, len), df.AsSpan(0, len));
            FromFloatSpanImpl(df.AsSpan(0, len), destination);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(xf);
            ArrayPool<float>.Shared.Return(df);
        }
    }

    private void UnaryViaFloatScalar(ReadOnlySpan<T> x, Span<T> destination, Func<float, float> fn)
    {
        if (destination.Length < x.Length)
            throw new ArgumentException(
                $"destination length {destination.Length} must be >= x length {x.Length}.",
                nameof(destination));

        int len = x.Length;
        float[] xf = ArrayPool<float>.Shared.Rent(len);
        float[] df = ArrayPool<float>.Shared.Rent(len);
        try
        {
            ToFloatSpanImpl(x, xf.AsSpan(0, len));
            for (int i = 0; i < len; i++) df[i] = fn(xf[i]);
            FromFloatSpanImpl(df.AsSpan(0, len), destination);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(xf);
            ArrayPool<float>.Shared.Return(df);
        }
    }

    private static float MishScalar(float x)
    {
        // mish(x) = x * tanh(softplus(x))
        float sp = (float)Math.Log(1 + Math.Exp(x));
        return x * (float)Math.Tanh(sp);
    }

    private static float SwishScalar(float x) => x / (1f + (float)Math.Exp(-x));

    public virtual T CosineSimilarity(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        if (y.Length != x.Length)
            throw new ArgumentException(
                $"y length {y.Length} must match x length {x.Length}.", nameof(y));

        int len = x.Length;
        float[] xf = ArrayPool<float>.Shared.Rent(len);
        float[] yf = ArrayPool<float>.Shared.Rent(len);
        try
        {
            ToFloatSpanImpl(x, xf.AsSpan(0, len));
            ToFloatSpanImpl(y, yf.AsSpan(0, len));
            float dot = SimdKernels.DotProduct(xf.AsSpan(0, len), yf.AsSpan(0, len));
            float nx = 0f, ny = 0f;
            for (int i = 0; i < len; i++) { nx += xf[i] * xf[i]; ny += yf[i] * yf[i]; }
            float denom = (float)(Math.Sqrt(nx) * Math.Sqrt(ny));
            return FromFloatImpl(denom == 0f ? 0f : dot / denom);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(xf);
            ArrayPool<float>.Shared.Return(yf);
        }
    }

    public virtual void Fill(Span<T> destination, T value) => destination.Fill(value);

    // Scalar-ops: x [op] scalar.
    public virtual void MultiplyScalar(ReadOnlySpan<T> x, T scalar, Span<T> destination)
    {
        float s = ToFloatImpl(scalar);
        UnaryViaFloatScalar(x, destination, (float f) => f * s);
    }
    public virtual void DivideScalar(ReadOnlySpan<T> x, T scalar, Span<T> destination)
    {
        float s = ToFloatImpl(scalar);
        UnaryViaFloatScalar(x, destination, (float f) => f / s);
    }
    public virtual void AddScalar(ReadOnlySpan<T> x, T scalar, Span<T> destination)
    {
        float s = ToFloatImpl(scalar);
        UnaryViaFloatScalar(x, destination, (float f) => f + s);
    }
    public virtual void SubtractScalar(ReadOnlySpan<T> x, T scalar, Span<T> destination)
    {
        float s = ToFloatImpl(scalar);
        UnaryViaFloatScalar(x, destination, (float f) => f - s);
    }

    public virtual void Clip(ReadOnlySpan<T> x, T min, T max, Span<T> destination)
    {
        float mn = ToFloatImpl(min), mx = ToFloatImpl(max);
        UnaryViaFloatScalar(x, destination, (float f) => f < mn ? mn : (f > mx ? mx : f));
    }

    public virtual void Pow(ReadOnlySpan<T> x, T power, Span<T> destination)
    {
        float p = ToFloatImpl(power);
        UnaryViaFloatScalar(x, destination, (float f) => (float)Math.Pow(f, p));
    }

    public virtual void Copy(ReadOnlySpan<T> source, Span<T> destination) => source.CopyTo(destination);

    public virtual void MultiplyAdd(ReadOnlySpan<T> x, ReadOnlySpan<T> y, T scalar, Span<T> destination)
    {
        int len = x.Length;
        float s = ToFloatImpl(scalar);
        float[] xf = ArrayPool<float>.Shared.Rent(len);
        float[] yf = ArrayPool<float>.Shared.Rent(len);
        float[] df = ArrayPool<float>.Shared.Rent(len);
        try
        {
            ToFloatSpanImpl(x, xf.AsSpan(0, len));
            ToFloatSpanImpl(y, yf.AsSpan(0, len));
            for (int i = 0; i < len; i++) df[i] = xf[i] + yf[i] * s;
            FromFloatSpanImpl(df.AsSpan(0, len), destination);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(xf);
            ArrayPool<float>.Shared.Return(yf);
            ArrayPool<float>.Shared.Return(df);
        }
    }

    public virtual void ToFloatSpan(ReadOnlySpan<T> source, Span<float> destination)
    {
        if (destination.Length < source.Length)
            throw new ArgumentException(
                $"destination length {destination.Length} must be >= source length {source.Length}.",
                nameof(destination));
        ToFloatSpanImpl(source, destination);
    }

    public virtual void FromFloatSpan(ReadOnlySpan<float> source, Span<T> destination)
    {
        if (destination.Length < source.Length)
            throw new ArgumentException(
                $"destination length {destination.Length} must be >= source length {source.Length}.",
                nameof(destination));
        FromFloatSpanImpl(source, destination);
    }

    public virtual void ToHalfSpan(ReadOnlySpan<T> source, Span<Half> destination)
    {
        if (destination.Length < source.Length)
            throw new ArgumentException(
                $"destination length {destination.Length} must be >= source length {source.Length}.",
                nameof(destination));
        for (int i = 0; i < source.Length; i++) destination[i] = (Half)ToFloatImpl(source[i]);
    }

    public virtual void FromHalfSpan(ReadOnlySpan<Half> source, Span<T> destination)
    {
        if (destination.Length < source.Length)
            throw new ArgumentException(
                $"destination length {destination.Length} must be >= source length {source.Length}.",
                nameof(destination));
        for (int i = 0; i < source.Length; i++) destination[i] = FromFloatImpl((float)source[i]);
    }

    public virtual void LeakyReLU(ReadOnlySpan<T> x, T alpha, Span<T> destination)
    {
        float a = ToFloatImpl(alpha);
        UnaryViaFloatScalar(x, destination, (float f) => f >= 0f ? f : f * a);
    }

    public virtual void ELU(ReadOnlySpan<T> x, T alpha, Span<T> destination)
    {
        float a = ToFloatImpl(alpha);
        UnaryViaFloatScalar(x, destination, (float f) => f >= 0f ? f : a * ((float)Math.Exp(f) - 1f));
    }
}
