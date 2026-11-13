using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Data-loading sources for any <see cref="IEngine"/>: raw bytes, numeric text, and a uniform fill from the caller's
/// <see cref="Random"/>, each producing the whole tensor in one call instead of a per-element loop.
/// </summary>
/// <remarks>
/// Every built-in engine derives from <see cref="CpuEngine"/>, so virtual dispatch reaches the device kernels where the
/// engine has them (DirectGpuTensorEngine: the bytes upload once and decode or parse on the device). Keeping these off
/// <see cref="IEngine"/> avoids an ABI break for external engines compiled against an earlier package; a third-party
/// engine is served by the host implementation.
/// </remarks>
public static class TensorDataEngineExtensions
{
    private static readonly CpuEngine HostEngine = new();

    private static CpuEngine BuiltIn(IEngine engine)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        return engine.As<CpuEngine>() ?? HostEngine;
    }

    /// <summary>
    /// Decodes a raw buffer of <paramref name="sourceType"/> elements into a tensor of <typeparamref name="T"/>, in one
    /// pass and without a typed host intermediate.
    /// </summary>
    /// <typeparam name="T">The numeric type of the result.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="data">The buffer: exactly <c>product(shape) * width(sourceType)</c> bytes, elements row-major.</param>
    /// <param name="sourceType">The element type stored in <paramref name="data"/>.</param>
    /// <param name="shape">The result's shape.</param>
    /// <param name="littleEndian">The byte order of each stored element (true for x86/ARM files, NumPy's default
    /// <c>&lt;</c>).</param>
    /// <returns>The decoded tensor.</returns>
    /// <remarks>
    /// <para><b>PyTorch / NumPy:</b> <c>torch.frombuffer(buf, dtype=src).view(shape).to(dtype)</c>,
    /// <c>numpy.frombuffer(buf, dtype=src).reshape(shape).astype(dtype)</c> (big-endian: <c>dtype('&gt;f8')</c>).</para>
    /// <para>Conversions round to nearest even (an int64 or float64 into float32 rounds once, a float64 too small for a
    /// float32 subnormal becomes a signed zero, one too large an infinity); a floating source converts to an integer
    /// <typeparamref name="T"/> by truncation toward zero, and into <c>int</c> or <c>long</c> it saturates at the type's range
    /// with NaN as 0 on every target framework; an integer source narrows into <c>int</c> by two's-complement wrap. NaN
    /// and infinity carry over to floating types. On
    /// DirectGpuTensorEngine a float32 result is decoded on the device on all six backends.</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The buffer's length does not match the shape, or a dimension is negative.</exception>
    public static Tensor<T> TensorFromBytes<T>(this IEngine engine, ReadOnlySpan<byte> data, NumericStorageType sourceType,
        int[] shape, bool littleEndian = true)
        => BuiltIn(engine).TensorFromBytes<T>(data, sourceType, shape, littleEndian);

    /// <summary>
    /// Parses numeric text tokens into a tensor; a token that does not parse becomes <paramref name="missingValue"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the result.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="tokens">One token per element, row-major (a null token is missing).</param>
    /// <param name="shape">The result's shape; its element count must equal the token count.</param>
    /// <param name="styles">The number styles each token may use (as <c>double.Parse</c> takes them).</param>
    /// <param name="provider">The culture's number format (<see cref="CultureInfo.InvariantCulture"/> for
    /// machine-written data).</param>
    /// <param name="missingValue">The value of a token that does not parse.</param>
    /// <returns>The parsed tensor.</returns>
    /// <remarks>
    /// <para><b>PyTorch / NumPy / pandas:</b> <c>numpy.genfromtxt(..., filling_values=missing)</c>,
    /// <c>pandas.to_numeric(tokens, errors='coerce').fillna(missing)</c>.</para>
    /// <para>Each token parses exactly as <typeparamref name="T"/>'s own <c>TryParse(token, styles, provider)</c> does
    /// (<c>double.TryParse</c> then the numeric conversion for other types). On DirectGpuTensorEngine a float32 result
    /// with invariant-style formatting is parsed on the device.</para>
    /// </remarks>
    public static Tensor<T> TensorParse<T>(this IEngine engine, IReadOnlyList<string?> tokens, int[] shape, NumberStyles styles,
        IFormatProvider provider, T missingValue)
        => BuiltIn(engine).TensorParse(tokens, shape, styles, provider, missingValue);

    /// <summary>
    /// Parses delimited UTF-8 numeric text (CSV / TSV / whitespace-free columns) into a tensor; a field that does not
    /// parse becomes <paramref name="missingValue"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the result.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="utf8Text">The text. Fields are separated by <paramref name="delimiter"/> or a line feed; a line feed
    /// at the very end starts no field; a carriage return before a line feed belongs to the line ending; an empty field is
    /// missing. Nothing is quoted or skipped (headers and comments are the caller's to drop).</param>
    /// <param name="delimiter">The field separator byte (',' for CSV, '\t' for TSV, ' ' for single-space columns).</param>
    /// <param name="shape">The result's shape; its element count must equal the field count.</param>
    /// <param name="styles">The number styles each field may use.</param>
    /// <param name="provider">The culture's number format.</param>
    /// <param name="missingValue">The value of a field that does not parse.</param>
    /// <returns>The parsed tensor.</returns>
    /// <remarks>
    /// <para><b>PyTorch / NumPy / pandas:</b> <c>numpy.loadtxt(f, delimiter=d)</c> /
    /// <c>numpy.genfromtxt(f, delimiter=d, filling_values=missing)</c>, <c>pandas.read_csv(f, header=None).to_numpy()</c>.</para>
    /// <para>Each field parses exactly as <typeparamref name="T"/>'s <c>TryParse(field, styles, provider)</c> does. For a
    /// float32 result under invariant-style formatting (decimal point '.', signs '-' and '+', no thousands, currency or
    /// parentheses) an exactly-rounded fast path decides almost every field without the framework parser; on
    /// DirectGpuTensorEngine that path runs on the device: a separator scan, a prefix sum over the field boundaries, and
    /// one thread per field. Fields it cannot decide exactly (NaN / infinity symbols, very long mantissas whose rounding
    /// the dropped digits could change, decimal exponents beyond +-9) are parsed on the host.</para>
    /// </remarks>
    public static Tensor<T> TensorParse<T>(this IEngine engine, ReadOnlySpan<byte> utf8Text, byte delimiter, int[] shape,
        NumberStyles styles, IFormatProvider provider, T missingValue)
        => BuiltIn(engine).TensorParse(utf8Text, delimiter, shape, styles, provider, missingValue);

    /// <summary>
    /// A tensor of uniform draws in [<paramref name="min"/>, <paramref name="max"/>) from the caller's
    /// <paramref name="generator"/>: element i (row-major) is <c>min + (max - min) * generator.NextDouble()</c>, the i-th
    /// draw, computed in double and converted to <typeparamref name="T"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the result.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="generator">The stream to draw from; it advances by exactly one NextDouble per element, as a scalar
    /// loop would, so a seeded caller's later draws are unchanged.</param>
    /// <param name="shape">The result's shape.</param>
    /// <param name="min">The inclusive lower bound.</param>
    /// <param name="max">The exclusive upper bound.</param>
    /// <returns>The filled tensor.</returns>
    /// <remarks>
    /// <para><b>PyTorch:</b> <c>torch.empty(shape).uniform_(min, max, generator=g)</c>. The draw is inherently
    /// sequential (one shared stream), so it runs on the host; DirectGpuTensorEngine then uploads a float32 result once
    /// and returns it device-resident. For a counter-based device stream use the seeded
    /// <c>IEngine.TensorRandomUniformRange(shape, min, max, seed)</c>.</para>
    /// </remarks>
    public static Tensor<T> TensorRandomUniformRange<T>(this IEngine engine, Random generator, int[] shape, double min,
        double max)
        => BuiltIn(engine).TensorRandomUniformFromGenerator<T>(generator, shape, min, max);

    /// <summary>
    /// Fills <paramref name="destination"/> (logical row-major order) with <c>min + (max - min) * generator.NextDouble()</c>,
    /// one draw per element, as <see cref="TensorRandomUniformRange{T}(IEngine, Random, int[], double, double)"/> does.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="generator">The stream to draw from.</param>
    /// <param name="destination">The tensor to fill in place.</param>
    /// <param name="min">The inclusive lower bound.</param>
    /// <param name="max">The exclusive upper bound.</param>
    /// <remarks><b>PyTorch:</b> <c>t.uniform_(min, max, generator=g)</c>.</remarks>
    public static void TensorRandomUniformRangeInto<T>(this IEngine engine, Random generator, Tensor<T> destination,
        double min, double max)
        => BuiltIn(engine).TensorRandomUniformFromGeneratorInto(generator, destination, min, max);

    // ------------------------------------------------------------------ distribution draws (caller's Random)

    /// <summary>
    /// A tensor of <paramref name="sampler"/>(<paramref name="generator"/>) draws, one call per element in row-major
    /// order, converted to <typeparamref name="T"/> in one bulk pass.
    /// </summary>
    /// <typeparam name="T">The numeric type of the result.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="generator">The stream the sampler draws from.</param>
    /// <param name="shape">The result's shape.</param>
    /// <param name="sampler">The scalar sampler the loop called (for example <c>r =&gt; r.NextGaussian()</c>); called
    /// exactly once per element, in order, so a seeded caller's values and later draws are unchanged.</param>
    /// <returns>The filled tensor.</returns>
    /// <remarks>
    /// <para><b>PyTorch:</b> <c>torch.empty(shape).normal_(generator=g)</c> with the caller's own sampler. The sampler
    /// is an arbitrary delegate over one sequential stream, so it runs on the host; DirectGpuTensorEngine uploads a
    /// float32 result once and returns it device-resident. Prefer
    /// <see cref="TensorNormalBoxMullerFromRng{T}"/> when the sampler is a Box-Muller form: its transform runs on the
    /// device.</para>
    /// </remarks>
    public static Tensor<T> TensorNormalFromRng<T>(this IEngine engine, Random generator, int[] shape,
        Func<Random, double> sampler)
        => BuiltIn(engine).TensorNormalFromRng<T>(generator, shape, sampler);

    /// <summary>
    /// Normal draws <c>mean + stdDev * z</c> with <c>z</c> a Box-Muller normal from the caller's
    /// <paramref name="generator"/> under <paramref name="variant"/>'s convention, one pair per element in row-major
    /// order.
    /// </summary>
    /// <typeparam name="T">The numeric type of the result.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="generator">The stream to draw from; it advances exactly as the scalar loop did (two NextDouble per
    /// element, more only under a redraw rule).</param>
    /// <param name="shape">The result's shape.</param>
    /// <param name="variant">How u1 and u2 are drawn and which branch is kept, matching the loop being replaced.</param>
    /// <param name="mean">The mean.</param>
    /// <param name="stdDev">The standard deviation.</param>
    /// <returns>The filled tensor.</returns>
    /// <remarks>
    /// <para><b>PyTorch:</b> <c>torch.empty(shape).normal_(mean, std, generator=g)</c> (a different generator and
    /// transform; this reproduces a System.Random Box-Muller loop). The value is computed in double as
    /// <c>mean + stdDev * (Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2))</c> and converted to
    /// <typeparamref name="T"/>, so a loop written in that order gets identical values; a loop that multiplies in another
    /// order (<c>scale * sqrt(..) * cos(..)</c>) can differ in the last bit.</para>
    /// <para>The draws are host-side by necessity; on DirectGpuTensorEngine a float32 result's uniforms upload once and
    /// the transform runs as a device kernel on all six backends, in float32 (agreeing with the host to float32
    /// rounding of the log, sqrt and trig).</para>
    /// </remarks>
    public static Tensor<T> TensorNormalBoxMullerFromRng<T>(this IEngine engine, Random generator, int[] shape,
        BoxMullerVariant variant, double mean = 0.0, double stdDev = 1.0)
        => BuiltIn(engine).TensorNormalBoxMullerFromRng<T>(generator, shape, variant, mean, stdDev);

    /// <summary>
    /// Clamped normal draws: <c>clamp(mean + stdDev * z, low, high)</c> per element, z as in
    /// <see cref="TensorNormalBoxMullerFromRng{T}"/>, one pair per element (no rejection: a value outside the bounds is
    /// clamped, as the scalar loops being replaced do).
    /// </summary>
    /// <typeparam name="T">The numeric type of the result.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="generator">The stream to draw from.</param>
    /// <param name="shape">The result's shape.</param>
    /// <param name="variant">The Box-Muller convention.</param>
    /// <param name="mean">The mean.</param>
    /// <param name="stdDev">The standard deviation.</param>
    /// <param name="low">The lower clamp bound.</param>
    /// <param name="high">The upper clamp bound (at least <paramref name="low"/>).</param>
    /// <returns>The filled tensor.</returns>
    /// <remarks><b>PyTorch:</b> <c>torch.nn.init.trunc_normal_</c>'s clamp form, <c>normal_(mean, std).clamp_(low, high)</c>.
    /// The transform and the clamp run as one device kernel on DirectGpuTensorEngine.</remarks>
    public static Tensor<T> TensorTruncatedNormalFromRng<T>(this IEngine engine, Random generator, int[] shape,
        BoxMullerVariant variant, double mean, double stdDev, double low, double high)
        => BuiltIn(engine).TensorTruncatedNormalFromRng<T>(generator, shape, variant, mean, stdDev, low, high);

    /// <summary>
    /// Gamma(<paramref name="shape"/>, <paramref name="scale"/>[i]) draws in row-major order by Marsaglia and Tsang's
    /// method, consuming the caller's <paramref name="generator"/> exactly as the scalar sampler does.
    /// </summary>
    /// <typeparam name="T">The numeric type of the result.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="generator">The stream to draw from.</param>
    /// <param name="shape">The shared shape parameter (finite, positive).</param>
    /// <param name="scale">The per-element scale; the result has its shape.</param>
    /// <param name="normal">The Box-Muller convention of the sampler's normals.</param>
    /// <returns>The draws.</returns>
    /// <remarks>
    /// <para><b>PyTorch:</b> <c>torch.distributions.Gamma(shape, 1 / scale).sample(generator=g)</c>. Shape below one draws
    /// its boost uniform first, then <c>Gamma(shape + 1, scale) * u^(1 / shape)</c>; otherwise
    /// <c>d = shape - 1/3</c>, <c>c = 1 / sqrt(9 d)</c>, normals x redrawn while <c>v = 1 + c x &lt;= 0</c>, <c>v = v^3</c>,
    /// one uniform u, accepted when <c>u &lt; 1 - 0.0331 x^4</c> or <c>ln u &lt; x^2 / 2 + d (1 - v + ln v)</c>, giving
    /// <c>d v scale</c>. Beta(a, b) is <c>g_a / (g_a + g_b)</c>.</para>
    /// <para>A rejection sampler's draw count depends on its draws, so the whole sample runs on the host;
    /// DirectGpuTensorEngine uploads a float32 result once and returns it device-resident.</para>
    /// </remarks>
    public static Tensor<T> TensorGammaFromRng<T>(this IEngine engine, Random generator, double shape, Tensor<T> scale,
        BoxMullerVariant normal)
        => BuiltIn(engine).TensorGammaFromRng(generator, shape, scale, normal);

    /// <summary>
    /// Gamma(<paramref name="shape"/>[i], <paramref name="scale"/>[i]) draws in row-major order; see
    /// <see cref="TensorGammaFromRng{T}(IEngine, Random, double, Tensor{T}, BoxMullerVariant)"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the result.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="generator">The stream to draw from.</param>
    /// <param name="shape">The per-element shape parameters (finite, positive).</param>
    /// <param name="scale">The per-element scales, the same shape.</param>
    /// <param name="normal">The Box-Muller convention of the sampler's normals.</param>
    /// <returns>The draws.</returns>
    /// <remarks><b>PyTorch:</b> <c>torch.distributions.Gamma(shape, 1 / scale).sample(generator=g)</c>.</remarks>
    public static Tensor<T> TensorGammaFromRng<T>(this IEngine engine, Random generator, Tensor<T> shape, Tensor<T> scale,
        BoxMullerVariant normal)
        => BuiltIn(engine).TensorGammaFromRng(generator, shape, scale, normal);

    /// <summary>
    /// Bernoulli draws: element i is 1 when the i-th <c>NextDouble()</c> is on <paramref name="successWhen"/>'s side of
    /// <paramref name="probabilities"/>[i] (compared in double), otherwise 0.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="generator">The stream to draw from: one NextDouble per element, row-major.</param>
    /// <param name="probabilities">The per-element thresholds (success probabilities under the default comparison).</param>
    /// <param name="successWhen">Below (<c>u &lt; p</c>, the Bernoulli convention) or above (<c>u &gt; p</c>).</param>
    /// <returns>The 0/1 draws, shaped like <paramref name="probabilities"/>.</returns>
    /// <remarks><b>PyTorch:</b> <c>torch.bernoulli(p, generator=g)</c>. The draws are host-side; on DirectGpuTensorEngine
    /// the comparison runs as a device kernel on all six backends, exactly (each double draw travels as a float and its
    /// signed residual, which decide its order against a float threshold without rounding).</remarks>
    public static Tensor<T> TensorBernoulliFromRng<T>(this IEngine engine, Random generator, Tensor<T> probabilities,
        UniformDrawComparison successWhen)
        => BuiltIn(engine).TensorBernoulliFromRng(generator, probabilities, successWhen);

    /// <summary>
    /// Bernoulli draws against one threshold: element i is 1 when the i-th <c>NextDouble()</c> is on
    /// <paramref name="successWhen"/>'s side of <paramref name="threshold"/>, otherwise 0.
    /// </summary>
    /// <typeparam name="T">The numeric type of the result.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="generator">The stream to draw from: one NextDouble per element, row-major.</param>
    /// <param name="shape">The result's shape.</param>
    /// <param name="threshold">The threshold (the success probability under the default comparison).</param>
    /// <param name="successWhen">Below or above the threshold.</param>
    /// <returns>The 0/1 draws.</returns>
    /// <remarks><b>PyTorch:</b> <c>torch.bernoulli(torch.full(shape, p), generator=g)</c>. On DirectGpuTensorEngine the
    /// comparison is a device kernel when the threshold is exactly a float32 value (0.5, 0.25, ...); another threshold is
    /// compared on the host so the result stays exact.</remarks>
    public static Tensor<T> TensorBernoulliFromRng<T>(this IEngine engine, Random generator, int[] shape, double threshold,
        UniformDrawComparison successWhen)
        => BuiltIn(engine).TensorBernoulliFromRng<T>(generator, shape, threshold, successWhen);

    /// <summary>
    /// Rademacher (+1 / -1) draws, one <c>NextDouble()</c> per element in row-major order, the sign decided by
    /// <paramref name="rule"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the result.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="generator">The stream to draw from.</param>
    /// <param name="shape">The result's shape.</param>
    /// <param name="rule">How a draw of exactly 0.5 resolves: -1 below one half (else +1), or +1 above one half (else
    /// -1).</param>
    /// <returns>The +/-1 draws.</returns>
    /// <remarks><b>PyTorch:</b> <c>torch.randint(0, 2, shape, generator=g) * 2 - 1</c>. The sign runs as the same device
    /// threshold kernel as <see cref="TensorBernoulliFromRng{T}(IEngine, Random, int[], double, UniformDrawComparison)"/>.</remarks>
    public static Tensor<T> TensorRademacherFromRng<T>(this IEngine engine, Random generator, int[] shape,
        RademacherRule rule = RademacherRule.NegativeWhenDrawBelowHalf)
        => BuiltIn(engine).TensorRademacherFromRng<T>(generator, shape, rule);

    // ------------------------------------------------------------------ data sinks

    /// <summary>
    /// Encodes the tensor's elements, row-major, as a raw <paramref name="targetType"/> buffer: the inverse of
    /// <see cref="TensorFromBytes{T}"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="input">The tensor.</param>
    /// <param name="targetType">The stored element type (Float64 for <c>BinaryWriter.Write(double)</c>-compatible
    /// bytes).</param>
    /// <param name="littleEndian">The byte order of each stored element.</param>
    /// <returns><c>input.Length * width(targetType)</c> bytes.</returns>
    /// <remarks>
    /// <para><b>PyTorch / NumPy:</b> <c>t.numpy().astype(dtype).tobytes()</c>, <c>ndarray.astype('&gt;f8').tobytes()</c>
    /// for big-endian.</para>
    /// <para>Floating targets round to nearest even (binary16 directly from the source value); integer targets truncate
    /// toward zero and saturate at the type's range with NaN as 0, the same rule TensorFromBytes decodes with, so a float
    /// round trip is exact. <c>int</c> and <c>long</c> tensors encode from their exact integers. On
    /// DirectGpuTensorEngine a float32 tensor is encoded on the device (all six backends) and only the packed bytes are
    /// read back.</para>
    /// </remarks>
    public static byte[] TensorToBytes<T>(this IEngine engine, Tensor<T> input, NumericStorageType targetType,
        bool littleEndian = true)
        => BuiltIn(engine).TensorToBytes(input, targetType, littleEndian);

    /// <summary>
    /// Converts every element to <typeparamref name="TOut"/>. Into an integral type (int, long, short, sbyte, byte,
    /// ushort, uint, ulong) each value is first rounded under <paramref name="mode"/>, and a value the type cannot hold
    /// (NaN included) saturates (NaN as 0) or throws per <paramref name="overflow"/>; into any other type it converts
    /// as the element types' own conversion does (nearest).
    /// </summary>
    /// <typeparam name="TIn">The source element type.</typeparam>
    /// <typeparam name="TOut">The result element type.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="input">The tensor to convert.</param>
    /// <param name="mode">The rounding rule for an integral target (ignored otherwise).</param>
    /// <param name="overflow">What an out-of-range integral result does.</param>
    /// <returns>The converted tensor, the same shape.</returns>
    /// <remarks>
    /// <para><b>PyTorch / NumPy:</b> <c>t.to(dtype)</c> (truncation), <c>torch.round(t).to(dtype)</c>,
    /// <c>numpy.rint(a).astype(np.int32)</c>, <c>ndarray.astype(dtype)</c>. Integer-to-integer conversions are exact
    /// before the range check. Not differentiable: an integral result has a zero gradient almost everywhere.</para>
    /// <para>On DirectGpuTensorEngine a float32 source into an integral target is rounded and range-limited on the
    /// device (all six backends); only the narrowing into the integral array happens on the host.</para>
    /// </remarks>
    public static Tensor<TOut> TensorCast<TIn, TOut>(this IEngine engine, Tensor<TIn> input,
        RoundingMode mode = RoundingMode.TowardZero, CastOverflowBehavior overflow = CastOverflowBehavior.Saturate)
        => BuiltIn(engine).TensorCast<TIn, TOut>(input, mode, overflow);

    /// <summary>
    /// Each element's double value as text, <c>value.ToString(format, provider)</c>: the inverse of
    /// <see cref="TensorParse{T}(IEngine, IReadOnlyList{string}, int[], NumberStyles, IFormatProvider, T)"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="input">The tensor.</param>
    /// <param name="provider">The culture's number format (<see cref="CultureInfo.InvariantCulture"/> for
    /// machine-read text).</param>
    /// <param name="format">A standard or custom numeric format string; null for the default (shortest round-trip on
    /// .NET Core 3.0 and later).</param>
    /// <returns>One string per element, row-major.</returns>
    /// <remarks>
    /// <para><b>NumPy / pandas:</b> <c>a.astype(str)</c>, <c>Series.astype(str)</c>.</para>
    /// <para>The text is formatted on the host, in parallel chunks, by the framework formatter itself (its culture
    /// handling and round-trip guarantees have no device equivalent); a device-resident tensor is read back once.</para>
    /// </remarks>
    public static string[] TensorFormat<T>(this IEngine engine, Tensor<T> input, IFormatProvider provider,
        string? format = null)
        => BuiltIn(engine).TensorFormat(input, provider, format);

    /// <summary>
    /// The elements formatted as in <see cref="TensorFormat{T}(IEngine, Tensor{T}, IFormatProvider, string)"/> and joined
    /// by <paramref name="separator"/>, in row-major order (a state or cache key).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="input">The tensor.</param>
    /// <param name="format">The numeric format string, or null.</param>
    /// <param name="provider">The culture's number format.</param>
    /// <param name="separator">The text between consecutive elements.</param>
    /// <returns>The joined text.</returns>
    /// <remarks><b>Python:</b> <c>separator.join(format(v, spec) for v in a.ravel())</c>.</remarks>
    public static string TensorFormatJoined<T>(this IEngine engine, Tensor<T> input, string? format, IFormatProvider provider,
        string separator)
        => string.Join(separator ?? throw new ArgumentNullException(nameof(separator)),
            BuiltIn(engine).TensorFormat(input, provider, format));

    // ------------------------------------------------------------------ rows

    /// <summary>
    /// The rows of <paramref name="table"/> (<c>[vocab, ...]</c>) at the 64-bit ids <paramref name="rows"/>: the result
    /// has <paramref name="rows"/>' shape followed by the table's row shape; an id outside <c>[0, vocab)</c> gives a
    /// zero row.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="table">The table, rows along its first axis.</param>
    /// <param name="rows">The row ids.</param>
    /// <returns>The gathered rows.</returns>
    /// <remarks>
    /// <para><b>PyTorch:</b> <c>table[rows]</c> / <c>torch.index_select(table, 0, rows)</c> /
    /// <c>F.embedding(rows, table)</c>, with out-of-range ids masked to zero rows instead of raising (the sparse
    /// optimizers' skip rule). Unlike <see cref="IEngine.TensorIndexSelect{T}"/> the ids are 64-bit and need not be in
    /// range.</para>
    /// <para>Differentiable in <paramref name="table"/>: its gradient index-adds each output row's gradient at its id
    /// (a repeated id accumulates). On DirectGpuTensorEngine a float32 gather runs on the device, one thread per output
    /// element.</para>
    /// </remarks>
    public static Tensor<T> TensorGatherRows<T>(this IEngine engine, Tensor<T> table, Tensor<long> rows)
        => BuiltIn(engine).TensorGatherRows(table, rows);

    /// <summary>
    /// Writes <paramref name="source"/>'s rows into <paramref name="table"/> in place at the 64-bit ids
    /// <paramref name="rows"/>, in O(rows * rowSize): an id outside <c>[0, vocab)</c> is skipped, and of repeated ids the
    /// last one's row wins (the order a sequential loop writes them in).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="table">The table to update, rows along its first axis.</param>
    /// <param name="rows">The row ids.</param>
    /// <param name="source">The new rows: <paramref name="rows"/>' shape followed by the table's row shape.</param>
    /// <remarks>
    /// <para><b>PyTorch:</b> <c>table.index_copy_(0, rows, source)</c> / <c>table[rows] = source</c> with out-of-range
    /// ids dropped. In place, so not recorded on the tape. On DirectGpuTensorEngine a device-resident float32 table is
    /// updated in its own buffer (the duplicate resolution decided on the host, the copy one thread per element).</para>
    /// </remarks>
    public static void TensorScatterRowsInPlace<T>(this IEngine engine, Tensor<T> table, Tensor<long> rows, Tensor<T> source)
        => BuiltIn(engine).TensorScatterRowsInPlace(table, rows, source);

    // ------------------------------------------------------------------ linear filter

    /// <summary>
    /// The linear (IIR / FIR) filter along the last axis, <c>scipy.signal.lfilter(b, a, x, axis=-1, zi=zi)[0]</c>:
    /// <c>a[0] y[n] = sum_k b[k] x[n-k] - sum_{k&gt;=1} a[k] y[n-k]</c>, every leading index an independent signal.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="b">The numerator (feed-forward) coefficients, 1-D.</param>
    /// <param name="a">The denominator (feedback) coefficients, 1-D, <c>a[0] != 0</c>; <c>[1]</c> makes an FIR
    /// filter.</param>
    /// <param name="x">The signals, time along the last axis.</param>
    /// <param name="zi">Optional initial state, <c>x</c>'s leading shape then <c>max(len(a), len(b)) - 1</c>: scipy's
    /// direct-form-II-transposed state; null starts at rest.</param>
    /// <returns>y, shaped like x.</returns>
    /// <remarks>
    /// <para><b>SciPy:</b> <c>scipy.signal.lfilter</c> (direct form II transposed, coefficients normalized by
    /// <c>a[0]</c>); the ARMA residual recursion <c>e[t] = x[t] - sum_j theta[j] e[t-1-j]</c> is
    /// <c>lfilter([1], [1, theta...], x)</c>. <b>PyTorch:</b> <c>torchaudio.functional.lfilter(x, a, b, clamp=False)</c>.</para>
    /// <para>Differentiable in b, a, x and zi (the backward is the same filter on the time-reversed gradient). The
    /// recursion is sequential in time, so DirectGpuTensorEngine runs one device thread per signal with the recurrence
    /// inside the kernel, on all six backends.</para>
    /// </remarks>
    public static Tensor<T> TensorLinearFilter<T>(this IEngine engine, Tensor<T> b, Tensor<T> a, Tensor<T> x,
        Tensor<T>? zi = null)
        => BuiltIn(engine).TensorLinearFilter(b, a, x, zi);

    /// <summary>
    /// <see cref="TensorLinearFilter{T}"/> with a guard on every output before it feeds the recursion: NaN becomes 0
    /// and the value is clamped to <c>[-maxAbs, maxAbs]</c> (a recursive forecast that must not run away).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="b">The numerator coefficients.</param>
    /// <param name="a">The denominator coefficients.</param>
    /// <param name="x">The signals, time along the last axis.</param>
    /// <param name="zi">Optional initial state, as for <see cref="TensorLinearFilter{T}"/>.</param>
    /// <param name="maxAbs">The clamp bound (non-negative; positive infinity guards only NaN).</param>
    /// <returns>The guarded outputs.</returns>
    /// <remarks><b>SciPy:</b> <c>lfilter</c> with <c>np.clip(np.nan_to_num(y[n]), -maxAbs, maxAbs)</c> applied inside the
    /// loop. The clamp makes the map piecewise, so it is not recorded on the tape. One device thread per signal on
    /// DirectGpuTensorEngine.</remarks>
    public static Tensor<T> TensorLinearFilterGuarded<T>(this IEngine engine, Tensor<T> b, Tensor<T> a, Tensor<T> x,
        Tensor<T>? zi, double maxAbs)
        => BuiltIn(engine).TensorLinearFilterGuarded(b, a, x, zi, maxAbs);
}
