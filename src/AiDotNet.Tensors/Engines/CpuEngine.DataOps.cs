using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// Data-loading sources: raw bytes (torch.frombuffer), numeric text (numpy.loadtxt / pandas.read_csv), and a uniform
// fill drawn from the caller's System.Random. None has a tensor input, so none records a tape node.
public partial class CpuEngine
{
    /// <summary>Elements per parallel chunk of a decode or parse.</summary>
    private const int DataOpChunk = 16 * 1024;

    // ------------------------------------------------------------------ TensorFromBytes

    /// <summary>
    /// Decodes a raw buffer of <paramref name="sourceType"/> elements into a tensor of <typeparamref name="T"/>; see
    /// <see cref="TensorDataEngineExtensions.TensorFromBytes{T}"/>.
    /// </summary>
    public virtual Tensor<T> TensorFromBytes<T>(ReadOnlySpan<byte> data, NumericStorageType sourceType, int[] shape,
        bool littleEndian = true)
    {
        int count = ValidateFromBytes(data.Length, sourceType, shape);
        var result = new Tensor<T>((int[])shape.Clone());
        if (count == 0) return result;
        unsafe
        {
            fixed (byte* p = data)
                DecodeBytesInto(result.GetDataArray(), (IntPtr)p, count, sourceType, littleEndian);
        }
        return result;
    }

    /// <summary>The element count of <paramref name="shape"/>, checked against the buffer's byte length.</summary>
    internal static int ValidateFromBytes(int byteLength, NumericStorageType sourceType, int[] shape)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        int width = sourceType.GetByteWidth();
        long count = ElementCountOf(shape);
        if (count * width != byteLength)
            throw new ArgumentException(
                $"A {sourceType} tensor of shape [{string.Join(", ", shape)}] needs {count * width} bytes; the buffer holds {byteLength}.",
                nameof(shape));
        return (int)count;
    }

    internal static long ElementCountOf(int[] shape)
    {
        long count = 1;
        foreach (int d in shape)
        {
            if (d < 0) throw new ArgumentException($"Shape dimensions must be non-negative; got {d}.", nameof(shape));
            count *= d;
            if (count > int.MaxValue) throw new ArgumentException("The shape holds more than int.MaxValue elements.", nameof(shape));
        }
        return count;
    }

    private static unsafe void DecodeBytesInto<T>(T[] destination, IntPtr source, int count, NumericStorageType type,
        bool littleEndian)
    {
        int width = type.GetByteWidth();
        int chunks = (count + DataOpChunk - 1) / DataOpChunk;
        double[]? staging = typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(long)
                            || typeof(T) == typeof(int)
            ? null
            : new double[count];
        CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)count * width, chunk =>
        {
            byte* p = (byte*)source;
            int from = chunk * DataOpChunk, to = Math.Min(count, from + DataOpChunk);
            if (destination is float[] f)
            {
                for (int i = from; i < to; i++) f[i] = DecodeAsSingle(p + (long)i * width, type, littleEndian);
            }
            else if (destination is double[] d)
            {
                for (int i = from; i < to; i++) d[i] = DecodeAsDouble(p + (long)i * width, type, littleEndian);
            }
            else if (destination is long[] l)
            {
                for (int i = from; i < to; i++) l[i] = DecodeAsInt64(p + (long)i * width, type, littleEndian);
            }
            else if (destination is int[] n)
            {
                for (int i = from; i < to; i++) n[i] = unchecked((int)DecodeAsInt64(p + (long)i * width, type, littleEndian));
            }
            else if (staging is not null)
            {
                for (int i = from; i < to; i++) staging[i] = DecodeAsDouble(p + (long)i * width, type, littleEndian);
            }
        }, deterministicSafe: true);
        if (staging is not null) ConvertFromDoubles(staging, destination);
    }

    /// <summary>Any other element type: one numeric-operations conversion per value from the double decode.</summary>
    [ScalarKernel("Generic element types convert through INumericOperations one value at a time; float, double, int and long decode on typed paths.")]
    private static void ConvertFromDoubles<T>(double[] source, T[] destination)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < source.Length; i++) destination[i] = ops.FromDouble(source[i]);
    }

    // The element's bits, as an unsigned integer of its width, in the buffer's byte order.
    private static unsafe ulong ReadRaw(byte* p, int width, bool littleEndian)
    {
        var span = new ReadOnlySpan<byte>(p, width);
        return width switch
        {
            1 => span[0],
            2 => littleEndian ? BinaryPrimitives.ReadUInt16LittleEndian(span) : BinaryPrimitives.ReadUInt16BigEndian(span),
            4 => littleEndian ? BinaryPrimitives.ReadUInt32LittleEndian(span) : BinaryPrimitives.ReadUInt32BigEndian(span),
            _ => littleEndian ? BinaryPrimitives.ReadUInt64LittleEndian(span) : BinaryPrimitives.ReadUInt64BigEndian(span),
        };
    }

    private static unsafe float DecodeAsSingle(byte* p, NumericStorageType type, bool littleEndian)
    {
        ulong raw = ReadRaw(p, type.GetByteWidth(), littleEndian);
        switch (type)
        {
            case NumericStorageType.Float16: { uint bits = HalfBitsToSingleBits((uint)raw); return *(float*)&bits; }
            case NumericStorageType.Float32: { uint bits = (uint)raw; return *(float*)&bits; }
            case NumericStorageType.Float64: return (float)(*(double*)&raw);
            case NumericStorageType.Int64: return (float)unchecked((long)raw);
            case NumericStorageType.UInt32: return (uint)raw;
            default: return DecodeIntegerValue(raw, type);
        }
    }

    private static unsafe double DecodeAsDouble(byte* p, NumericStorageType type, bool littleEndian)
    {
        ulong raw = ReadRaw(p, type.GetByteWidth(), littleEndian);
        switch (type)
        {
            case NumericStorageType.Float16: { uint bits = HalfBitsToSingleBits((uint)raw); return *(float*)&bits; }
            case NumericStorageType.Float32: { uint bits = (uint)raw; return *(float*)&bits; }
            case NumericStorageType.Float64: return *((double*)&raw);
            default: return DecodeAsInt64Raw(raw, type);
        }
    }

    private static unsafe long DecodeAsInt64(byte* p, NumericStorageType type, bool littleEndian)
    {
        ulong raw = ReadRaw(p, type.GetByteWidth(), littleEndian);
        switch (type)
        {
            // A floating source truncates toward zero, as torch's .to(int64) does.
            case NumericStorageType.Float16:
            case NumericStorageType.Float32:
            case NumericStorageType.Float64:
                return (long)DecodeAsDouble(p, type, littleEndian);
            default:
                return DecodeAsInt64Raw(raw, type);
        }
    }

    private static long DecodeAsInt64Raw(ulong raw, NumericStorageType type) => type switch
    {
        NumericStorageType.Int8 => unchecked((sbyte)raw),
        NumericStorageType.UInt8 => (byte)raw,
        NumericStorageType.Int16 => unchecked((short)raw),
        NumericStorageType.UInt16 => (ushort)raw,
        NumericStorageType.Int32 => unchecked((int)raw),
        NumericStorageType.UInt32 => (uint)raw,
        _ => unchecked((long)raw),
    };

    // 8- and 16-bit integers are exact in a float; int32 rounds once, to nearest even.
    private static float DecodeIntegerValue(ulong raw, NumericStorageType type) => DecodeAsInt64Raw(raw, type);

    /// <summary>The float32 bits of a binary16 value (exact; NaN payloads kept), as the device kernels convert it.</summary>
    internal static uint HalfBitsToSingleBits(uint h)
    {
        uint sign = (h & 0x8000u) << 16;
        uint e = (h >> 10) & 0x1Fu;
        uint m = h & 0x3FFu;
        if (e == 0x1Fu) return sign | 0x7F800000u | (m << 13);
        if (e == 0u)
        {
            if (m == 0u) return sign;
            int p = 31 - LeadingZeros(m);
            return sign | ((uint)(p + 103) << 23) | ((m << (23 - p)) & 0x7FFFFFu);
        }
        return sign | ((e + 112u) << 23) | (m << 13);
    }

    private static int LeadingZeros(uint x)
    {
        int n = 0;
        if (x == 0) return 32;
        while ((x & 0x80000000u) == 0) { x <<= 1; n++; }
        return n;
    }

    // ------------------------------------------------------------------ TensorParse

    /// <summary>
    /// Parses numeric text tokens into a tensor; see
    /// <see cref="TensorDataEngineExtensions.TensorParse{T}(IEngine, IReadOnlyList{string}, int[], NumberStyles, IFormatProvider, T)"/>.
    /// </summary>
    public virtual Tensor<T> TensorParse<T>(IReadOnlyList<string?> tokens, int[] shape, NumberStyles styles,
        IFormatProvider provider, T missingValue)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (provider is null) throw new ArgumentNullException(nameof(provider));
        int count = ValidateParseShape(tokens.Count, shape);
        var result = new Tensor<T>((int[])shape.Clone());
        if (count == 0) return result;
        var destination = result.GetDataArray();
        int chunks = (count + DataOpChunk - 1) / DataOpChunk;
        CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)count * 32, chunk =>
        {
            int from = chunk * DataOpChunk, to = Math.Min(count, from + DataOpChunk);
            for (int i = from; i < to; i++)
                destination[i] = ParseToken(tokens[i], styles, provider, missingValue);
        }, deterministicSafe: true);
        return result;
    }

    /// <summary>
    /// Parses delimited UTF-8 numeric text into a tensor; see
    /// <see cref="TensorDataEngineExtensions.TensorParse{T}(IEngine, ReadOnlySpan{byte}, byte, int[], NumberStyles, IFormatProvider, T)"/>.
    /// </summary>
    public virtual Tensor<T> TensorParse<T>(ReadOnlySpan<byte> utf8Text, byte delimiter, int[] shape, NumberStyles styles,
        IFormatProvider provider, T missingValue)
    {
        if (provider is null) throw new ArgumentNullException(nameof(provider));
        var bounds = SplitFieldBounds(utf8Text, delimiter, out int fields);
        int count = ValidateParseShape(fields, shape);
        var result = new Tensor<T>((int[])shape.Clone());
        if (count == 0) return result;
        var destination = result.GetDataArray();
        int fastMask = NumericTextFastPath.StyleMask(styles, provider);
        unsafe
        {
            fixed (byte* p = utf8Text)
            {
                IntPtr text = (IntPtr)p;
                int chunks = (count + DataOpChunk - 1) / DataOpChunk;
                CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)utf8Text.Length * 4, chunk =>
                {
                    int from = chunk * DataOpChunk, to = Math.Min(count, from + DataOpChunk);
                    var field = (byte*)text;
                    for (int i = from; i < to; i++)
                    {
                        int start = bounds[2 * i];
                        destination[i] = ParseUtf8Field(field + start, bounds[2 * i + 1] - start, styles, provider,
                            missingValue, fastMask);
                    }
                }, deterministicSafe: true);
            }
        }
        return result;
    }

    internal static int ValidateParseShape(int fields, int[] shape)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        long count = ElementCountOf(shape);
        if (count != fields)
            throw new ArgumentException(
                $"The text holds {fields} numeric fields; shape [{string.Join(", ", shape)}] needs {count}.", nameof(shape));
        return fields;
    }

    /// <summary>
    /// The [start, end) byte range of every field, two ints per field. Fields are separated by
    /// <paramref name="delimiter"/> or a line feed; one line feed at the very end starts no field, a carriage return
    /// before a line feed (or before that final one) belongs to the line ending, and an empty text has no fields.
    /// The device kernels split by the same rules.
    /// </summary>
    internal static int[] SplitFieldBounds(ReadOnlySpan<byte> text, byte delimiter, out int fields)
    {
        int length = text.Length;
        bool trailingNewline = length > 0 && text[length - 1] == (byte)'\n';
        if (trailingNewline) length--;
        if (length == 0)
        {
            fields = 0;
            return Array.Empty<int>();
        }
        var body = text.Slice(0, length);
        int separators = 0;
        for (int at = 0; ;)
        {
            int k = body.Slice(at).IndexOfAny(delimiter, (byte)'\n');
            if (k < 0) break;
            separators++;
            at += k + 1;
        }
        fields = separators + 1;
        var bounds = new int[2 * fields];
        int f = 0, startOfField = 0;
        for (int at = 0; ;)
        {
            int k = body.Slice(at).IndexOfAny(delimiter, (byte)'\n');
            if (k < 0) break;
            int pos = at + k;
            int end = pos;
            if (body[pos] == (byte)'\n' && delimiter != (byte)'\r' && pos > 0 && body[pos - 1] == (byte)'\r') end = pos - 1;
            bounds[2 * f] = startOfField;
            bounds[2 * f + 1] = end;
            f++;
            startOfField = pos + 1;
            at = pos + 1;
        }
        int last = length;
        if (trailingNewline && delimiter != (byte)'\r' && body[length - 1] == (byte)'\r') last = length - 1;
        bounds[2 * f] = startOfField;
        bounds[2 * f + 1] = last;
        return bounds;
    }

    private static unsafe T ParseUtf8Field<T>(byte* field, int length, NumberStyles styles, IFormatProvider provider,
        T missingValue, int fastMask)
    {
        var span = new ReadOnlySpan<byte>(field, length);
        if (typeof(T) == typeof(float) && fastMask >= 0)
        {
            int status = NumericTextFastPath.TryParseSingleBits(span, fastMask, out uint bits);
            if (status == NumericTextFastPath.Parsed) { float parsed = *(float*)&bits; return (T)(object)parsed; }
            if (status == NumericTextFastPath.Empty) return missingValue;
        }
#if NET8_0_OR_GREATER
        if (typeof(T) == typeof(float))
            return float.TryParse(span, styles, provider, out float f) ? (T)(object)f : missingValue;
        if (typeof(T) == typeof(double))
            return double.TryParse(span, styles, provider, out double d) ? (T)(object)d : missingValue;
#endif
        return ParseToken(Encoding.UTF8.GetString(field, length), styles, provider, missingValue);
    }

    /// <summary>One token as <typeparamref name="T"/>: the type's own TryParse for the built-in numeric types, the double
    /// parse converted otherwise; <paramref name="missingValue"/> when it does not parse.</summary>
    internal static T ParseToken<T>(string? token, NumberStyles styles, IFormatProvider provider, T missingValue)
    {
        if (token is null) return missingValue;
        if (typeof(T) == typeof(float))
            return float.TryParse(token, styles, provider, out float f) ? (T)(object)f : missingValue;
        if (typeof(T) == typeof(double))
            return double.TryParse(token, styles, provider, out double d) ? (T)(object)d : missingValue;
        if (typeof(T) == typeof(decimal))
            return decimal.TryParse(token, styles, provider, out decimal m) ? (T)(object)m : missingValue;
        if (typeof(T) == typeof(int))
            return int.TryParse(token, styles, provider, out int i) ? (T)(object)i : missingValue;
        if (typeof(T) == typeof(long))
            return long.TryParse(token, styles, provider, out long l) ? (T)(object)l : missingValue;
        if (typeof(T) == typeof(short))
            return short.TryParse(token, styles, provider, out short s) ? (T)(object)s : missingValue;
        if (typeof(T) == typeof(byte))
            return byte.TryParse(token, styles, provider, out byte b) ? (T)(object)b : missingValue;
        if (typeof(T) == typeof(sbyte))
            return sbyte.TryParse(token, styles, provider, out sbyte sb) ? (T)(object)sb : missingValue;
        if (typeof(T) == typeof(uint))
            return uint.TryParse(token, styles, provider, out uint u) ? (T)(object)u : missingValue;
        if (typeof(T) == typeof(ulong))
            return ulong.TryParse(token, styles, provider, out ulong ul) ? (T)(object)ul : missingValue;
        if (typeof(T) == typeof(ushort))
            return ushort.TryParse(token, styles, provider, out ushort us) ? (T)(object)us : missingValue;
        return double.TryParse(token, styles, provider, out double v)
            ? MathHelper.GetNumericOperations<T>().FromDouble(v)
            : missingValue;
    }

    // ------------------------------------------------------------------ TensorRandomUniformRange (caller's Random)

    /// <summary>
    /// A tensor of <c>min + (max - min) * generator.NextDouble()</c> draws in row-major order; see
    /// <see cref="TensorDataEngineExtensions.TensorRandomUniformRange{T}(IEngine, Random, int[], double, double)"/>.
    /// </summary>
    public virtual Tensor<T> TensorRandomUniformFromGenerator<T>(Random generator, int[] shape, double min, double max)
    {
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        ElementCountOf(shape);
        var result = new Tensor<T>((int[])shape.Clone());
        FillUniformFromGenerator(generator, result.GetDataArray().AsSpan(0, result.Length), min, max);
        return result;
    }

    /// <summary>
    /// Fills <paramref name="destination"/> in logical row-major order with <c>min + (max - min) * generator.NextDouble()</c>;
    /// see <see cref="TensorDataEngineExtensions.TensorRandomUniformRangeInto{T}(IEngine, Random, Tensor{T}, double, double)"/>.
    /// </summary>
    public virtual void TensorRandomUniformFromGeneratorInto<T>(Random generator, Tensor<T> destination, double min, double max)
    {
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (destination.IsSparse)
            throw new InvalidOperationException("Random initialization into SparseTensor is not supported.");
        if (destination.IsContiguous)
        {
            using var lease = destination.LeaseWritable();
            FillUniformFromGenerator(generator, lease.Span, min, max);
            return;
        }
        // A strided view: draw in logical order, then place each value at its storage slot.
        var draws = new T[destination.Length];
        FillUniformFromGenerator(generator, draws, min, max);
        var storage = destination.RawWritableStorageSpan;
        for (int i = 0; i < draws.Length; i++) storage[destination.LogicalToStorageIndex(i)] = draws[i];
    }

    /// <summary>
    /// <c>min + (max - min) * u</c> per element in order, u = <paramref name="generator"/>.NextDouble(), computed in double
    /// and converted to <typeparamref name="T"/>: the scalar loop's values and its consumption of the stream.
    /// </summary>
    [ScalarKernel("System.Random is a sequential stream: the draws must come one at a time, in order, so a seeded caller sees the values its scalar loop produced.")]
    internal static void FillUniformFromGenerator<T>(Random generator, Span<T> destination, double min, double max)
    {
        double range = max - min;
        if (typeof(T) == typeof(float))
        {
            for (int i = 0; i < destination.Length; i++)
                System.Runtime.CompilerServices.Unsafe.As<T, float>(ref destination[i]) = (float)(min + range * generator.NextDouble());
            return;
        }
        if (typeof(T) == typeof(double))
        {
            for (int i = 0; i < destination.Length; i++)
                System.Runtime.CompilerServices.Unsafe.As<T, double>(ref destination[i]) = min + range * generator.NextDouble();
            return;
        }
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < destination.Length; i++) destination[i] = ops.FromDouble(min + range * generator.NextDouble());
    }
}

/// <summary>
/// The exactly-rounded float32 fast path of numeric text parsing, the host twin of
/// <see cref="DirectGpu.DataDecodeKernelTemplate"/>'s parse kernel (the same grammar, the same integer arithmetic, so the
/// same fields are decided and to the same bits). A field it does not decide goes to <c>float.Parse</c>.
/// </summary>
internal static class NumericTextFastPath
{
    internal const int Parsed = 0, Undecided = 1, Empty = 2;

    private const NumberStyles Supported = NumberStyles.AllowLeadingWhite | NumberStyles.AllowTrailingWhite
        | NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent;

    /// <summary>
    /// The kernel's style bits for <paramref name="styles"/> under <paramref name="provider"/>, or -1 when the fast path
    /// does not apply: a style it does not implement (thousands, currency, parentheses, trailing sign, hex), a culture
    /// whose decimal point or signs are not '.', '-' and '+', or a NaN / infinity symbol spelled only with characters
    /// the fast grammar accepts (it could then be taken for a number).
    /// </summary>
    internal static int StyleMask(NumberStyles styles, IFormatProvider provider)
    {
        if ((styles & ~Supported) != 0) return -1;
        var info = NumberFormatInfo.GetInstance(provider);
        if (info.NumberDecimalSeparator != "." || info.NegativeSign != "-" || info.PositiveSign != "+") return -1;
        if (!HasForeignChar(info.NaNSymbol) || !HasForeignChar(info.PositiveInfinitySymbol)
            || !HasForeignChar(info.NegativeInfinitySymbol))
            return -1;
        int mask = 0;
        if ((styles & NumberStyles.AllowLeadingWhite) != 0) mask |= DirectGpu.DataDecodeKernelTemplate.StyleLeadingWhite;
        if ((styles & NumberStyles.AllowTrailingWhite) != 0) mask |= DirectGpu.DataDecodeKernelTemplate.StyleTrailingWhite;
        if ((styles & NumberStyles.AllowLeadingSign) != 0) mask |= DirectGpu.DataDecodeKernelTemplate.StyleLeadingSign;
        if ((styles & NumberStyles.AllowDecimalPoint) != 0) mask |= DirectGpu.DataDecodeKernelTemplate.StyleDecimalPoint;
        if ((styles & NumberStyles.AllowExponent) != 0) mask |= DirectGpu.DataDecodeKernelTemplate.StyleExponent;
        return mask;
    }

    private static bool HasForeignChar(string symbol)
    {
        if (symbol.Length == 0) return false;
        foreach (char c in symbol)
            if (!(c >= '0' && c <= '9') && c != '.' && c != '+' && c != '-' && c != 'e' && c != 'E' && !IsWhite(c))
                return true;
        return false;
    }

    private static bool IsWhite(uint b) => b == 32u || (b >= 9u && b <= 13u);

    /// <summary>
    /// <see cref="Parsed"/> with the float32 bits of the field, <see cref="Empty"/> for an empty field, or
    /// <see cref="Undecided"/> for a field the fast path leaves to <c>float.Parse</c>.
    /// </summary>
    internal static int TryParseSingleBits(ReadOnlySpan<byte> s, int styleMask, out uint bits)
    {
        bits = 0;
        int e = s.Length;
        if (e == 0) return Empty;
        int q = 0;
        while (q < e && IsWhite(s[q])) q++;
        if (q > 0 && (styleMask & 1) == 0) return Undecided;
        int r = e;
        while (r > q && IsWhite(s[r - 1])) r--;
        if (r < e && (styleMask & 2) == 0) return Undecided;
        if (q == r) return Undecided;
        bool neg = false;
        if (s[q] == (byte)'-' || s[q] == (byte)'+')
        {
            if ((styleMask & 4) == 0) return Undecided;
            neg = s[q] == (byte)'-';
            q++;
        }
        uint sig = 0;
        int nsig = 0, scale = 0, digits = 0;
        bool dropped = false;
        while (q < r && s[q] >= (byte)'0' && s[q] <= (byte)'9')
        {
            uint d = (uint)(s[q] - '0');
            digits++;
            if (sig == 0 && d == 0) { }
            else if (nsig < 9) { sig = sig * 10 + d; nsig++; }
            else { scale++; if (d != 0) dropped = true; }
            q++;
        }
        if (q < r && s[q] == (byte)'.')
        {
            if ((styleMask & 8) == 0) return Undecided;
            q++;
            while (q < r && s[q] >= (byte)'0' && s[q] <= (byte)'9')
            {
                uint d = (uint)(s[q] - '0');
                digits++;
                if (sig == 0 && d == 0) scale--;
                else if (nsig < 9) { sig = sig * 10 + d; nsig++; scale--; }
                else if (d != 0) dropped = true;
                q++;
            }
        }
        if (digits == 0) return Undecided;
        int ex = 0;
        if (q < r && (s[q] == (byte)'e' || s[q] == (byte)'E'))
        {
            if ((styleMask & 16) == 0) return Undecided;
            q++;
            bool eneg = false;
            if (q < r && (s[q] == (byte)'-' || s[q] == (byte)'+')) { eneg = s[q] == (byte)'-'; q++; }
            int ed = 0;
            while (q < r && s[q] >= (byte)'0' && s[q] <= (byte)'9')
            {
                if (ex < 100000) ex = ex * 10 + (s[q] - '0');
                ed++;
                q++;
            }
            if (ed == 0) return Undecided;
            if (eneg) ex = -ex;
        }
        if (q != r) return Undecided;
        if (sig == 0)
        {
            if (neg) return Undecided;   // "-0": .NET Framework and .NET Core disagree on its sign
            bits = 0;
            return Parsed;
        }
        long e10 = (long)scale + ex;
        uint numLo = sig, numHi = dropped ? sig + 1 : sig, den = 1;
        if (e10 > 9 || e10 < -9) return Undecided;
        if (e10 >= 0)
        {
            uint p10 = 1;
            for (int t = 0; t < e10; t++) p10 *= 10;
            if (numHi > 2147483647u / p10) return Undecided;
            numLo *= p10;
            numHi *= p10;
        }
        else
        {
            for (long t = 0; t < -e10; t++) den *= 10;
        }
        uint lo = DivToSingleBits(numLo, den, dropped);
        if (dropped && DivToSingleBits(numHi, den, true) != lo) return Undecided;
        bits = neg ? lo | 0x80000000u : lo;
        return Parsed;
    }

    /// <summary>The float32 bits of num / den (1 &lt;= num &lt; 2^31, den &lt;= 10^9) rounded to nearest even; a sticky
    /// input marks a value strictly above num / den by less than one unit of num.</summary>
    internal static uint DivToSingleBits(uint num, uint den, bool stickyIn)
    {
        uint q = num / den;
        uint r = num - q * den;
        uint m = 0;
        int ex2 = 0, nb = 0;
        bool sticky = stickyIn;
        if (q != 0)
        {
            int p = 31 - LeadingZeros(q);
            if (p >= 24)
            {
                int s = p - 24;
                m = q >> s;
                if ((q & ((1u << s) - 1u)) != 0) sticky = true;
                ex2 = s;
                nb = 25;
            }
            else
            {
                m = q;
                nb = p + 1;
            }
        }
        while (nb < 25)
        {
            r <<= 1;
            uint bit = 0;
            if (r >= den) { bit = 1; r -= den; }
            m = (m << 1) | bit;
            ex2--;
            if (m != 0) nb++;
        }
        if (r != 0) sticky = true;
        uint rb = m & 1u;
        m >>= 1;
        ex2++;
        if (rb != 0 && (sticky || (m & 1u) != 0)) m++;
        if (m == 1u << 24) { m = 1u << 23; ex2++; }
        return ((uint)(ex2 + 150) << 23) | (m & 0x7FFFFFu);
    }

    private static int LeadingZeros(uint x)
    {
        int n = 0;
        if (x == 0) return 32;
        while ((x & 0x80000000u) == 0) { x <<= 1; n++; }
        return n;
    }
}
