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
}
