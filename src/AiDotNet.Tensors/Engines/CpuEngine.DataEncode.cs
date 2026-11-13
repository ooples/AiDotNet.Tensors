using System;
using System.Buffers.Binary;
using System.Globalization;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// Data sinks, the inverses of the data-loading sources: raw bytes (tensor.numpy().astype(dtype).tobytes()), an element
// type conversion with an explicit rounding rule (tensor.to(dtype) / ndarray.astype), and invariant text. None is
// differentiable (the bytes and strings leave the tensor world; an integer cast has a zero gradient almost
// everywhere), so none records a tape node.
public partial class CpuEngine
{
    // ------------------------------------------------------------------ TensorToBytes

    /// <summary>
    /// Encodes the tensor's elements (row-major) as a raw <paramref name="targetType"/> buffer; see
    /// <see cref="TensorDataEngineExtensions.TensorToBytes{T}"/>.
    /// </summary>
    public virtual byte[] TensorToBytes<T>(Tensor<T> input, NumericStorageType targetType, bool littleEndian = true)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        int width = targetType.GetByteWidth();
        int count = input.Length;
        if ((long)count * width > int.MaxValue)
            throw new ArgumentException("The encoded buffer would exceed int.MaxValue bytes.", nameof(input));
        var bytes = new byte[count * width];
        if (count == 0) return bytes;
        // float -> Float32 copies the bits: a round trip through double would quiet signaling NaNs.
        if (input is Tensor<float> floats && targetType == NumericStorageType.Float32)
            EncodeSingleBits(ReadAll(floats), bytes, littleEndian);
        else if (input is Tensor<long> longs) EncodeIntegers(ReadAll(longs), bytes, targetType, littleEndian);
        else if (input is Tensor<int> ints) EncodeIntegers(Widen(ReadAll(ints)), bytes, targetType, littleEndian);
        else if (input is Tensor<ulong> ulongs) EncodeUnsigned(ReadAll(ulongs), bytes, targetType, littleEndian);
        else if (input is Tensor<decimal> decimals && !IsFloatingStorage(targetType))
            EncodeIntegers(TruncateDecimals(ReadAll(decimals)), bytes, targetType, littleEndian);
        else if (IsIntegral(typeof(T))) EncodeIntegers(WidenIntegral(ReadAll(input)), bytes, targetType, littleEndian);
        else EncodeDoubles(RoIAlignPlan.ToDoubles(input), bytes, targetType, littleEndian);
        return bytes;
    }

    internal static TValue[] ReadAll<TValue>(Tensor<TValue> tensor)
    {
        using var lease = tensor.Lease();
        return lease.Span.ToArray();
    }

    private static void EncodeSingleBits(float[] values, byte[] bytes, bool littleEndian)
    {
        var bits = new uint[values.Length];
        Buffer.BlockCopy(values, 0, bits, 0, values.Length * sizeof(float));
        int chunks = (values.Length + DataOpChunk - 1) / DataOpChunk;
        CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)values.Length * 4, chunk =>
        {
            int from = chunk * DataOpChunk, to = Math.Min(values.Length, from + DataOpChunk);
            for (int i = from; i < to; i++)
                WriteRaw(bytes, i * 4, 4, bits[i], littleEndian);
        }, deterministicSafe: true);
    }

    private static long[] Widen(int[] values)
    {
        var result = new long[values.Length];
        for (int i = 0; i < values.Length; i++) result[i] = values[i];
        return result;
    }

    private static bool IsFloatingStorage(NumericStorageType type) =>
        type == NumericStorageType.Float64 || type == NumericStorageType.Float32 || type == NumericStorageType.Float16;

    // Decimal values truncated toward zero in decimal arithmetic (1.9999999999999999m is 1, where a double would read
    // 2.0), then clamped to long's range; EncodeInteger saturates each to the narrower target.
    private static long[] TruncateDecimals(decimal[] values)
    {
        var result = new long[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            decimal t = decimal.Truncate(values[i]);
            result[i] = t <= long.MinValue ? long.MinValue : t >= long.MaxValue ? long.MaxValue : (long)t;
        }
        return result;
    }

    // short / sbyte / byte / ushort / uint widened exactly (each fits a long); ulong takes EncodeUnsigned.
    private static long[] WidenIntegral<TValue>(TValue[] values)
    {
        var result = new long[values.Length];
        for (int i = 0; i < values.Length; i++) result[i] = (long)IntegerValue(values[i]);
        return result;
    }

    // ulong values: those within long's range take the exact integer encoding; larger ones saturate every integer
    // target (each maximum is at most long.MaxValue) and round once from the exact value to a floating target.
    private static void EncodeUnsigned(ulong[] values, byte[] bytes, NumericStorageType type, bool littleEndian)
    {
        int width = type.GetByteWidth();
        bool floating = IsFloatingStorage(type);
        int chunks = (values.Length + DataOpChunk - 1) / DataOpChunk;
        CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)values.Length * width, chunk =>
        {
            int from = chunk * DataOpChunk, to = Math.Min(values.Length, from + DataOpChunk);
            for (int i = from; i < to; i++)
            {
                ulong v = values[i];
                ulong raw = v <= long.MaxValue ? EncodeInteger((long)v, type)
                    : floating ? EncodeDouble(RoundUnsignedToSignificand(v, type), type)
                    : EncodeInteger(long.MaxValue, type);
                WriteRaw(bytes, i * width, width, raw, littleEndian);
            }
        }, deterministicSafe: true);
    }

    // v (at least 2^63) rounded once, to nearest even, to the target's significand width, as an exact double: going
    // through (double)v first would round twice (2^63 + 2^39 + 1 becomes a float32 midpoint and then rounds down).
    // Every such value is far past binary16's range, so Float16 takes the binary64 width and overflows to infinity.
    private static double RoundUnsignedToSignificand(ulong v, NumericStorageType type)
    {
        int significandBits = type == NumericStorageType.Float32 ? SingleSignificandBits : DoubleMantissaBits + 1;
        int shift = DoubleBits - significandBits;
        ulong q = RoundShiftRight(v, shift);
        return q * Math.Pow(2.0, shift);
    }

    private const int SingleSignificandBits = 24;

    /// <summary>Each value encoded as <paramref name="type"/> into its slot of <paramref name="bytes"/>, chunks in
    /// parallel.</summary>
    internal static void EncodeDoubles(double[] values, byte[] bytes, NumericStorageType type, bool littleEndian)
    {
        int width = type.GetByteWidth();
        int chunks = (values.Length + DataOpChunk - 1) / DataOpChunk;
        CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)values.Length * width, chunk =>
        {
            int from = chunk * DataOpChunk, to = Math.Min(values.Length, from + DataOpChunk);
            for (int i = from; i < to; i++) WriteRaw(bytes, i * width, width, EncodeDouble(values[i], type), littleEndian);
        }, deterministicSafe: true);
    }

    private static void EncodeIntegers(long[] values, byte[] bytes, NumericStorageType type, bool littleEndian)
    {
        int width = type.GetByteWidth();
        int chunks = (values.Length + DataOpChunk - 1) / DataOpChunk;
        CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)values.Length * width, chunk =>
        {
            int from = chunk * DataOpChunk, to = Math.Min(values.Length, from + DataOpChunk);
            for (int i = from; i < to; i++) WriteRaw(bytes, i * width, width, EncodeInteger(values[i], type), littleEndian);
        }, deterministicSafe: true);
    }

    // The element's low `width` bytes in the requested order.
    private static void WriteRaw(byte[] bytes, int at, int width, ulong raw, bool littleEndian)
    {
        for (int j = 0; j < width; j++)
        {
            byte b = (byte)(raw >> (8 * j));
            bytes[littleEndian ? at + j : at + width - 1 - j] = b;
        }
    }

    /// <summary>
    /// The storage bits of <paramref name="value"/> as <paramref name="type"/>: floating targets round to nearest even
    /// (binary16 directly from the double, no double rounding through binary32), integer targets truncate toward zero
    /// and saturate at the type's range with NaN as 0 (the inverse of TensorFromBytes's integer decode).
    /// </summary>
    internal static ulong EncodeDouble(double value, NumericStorageType type)
    {
        switch (type)
        {
            case NumericStorageType.Float64:
                return unchecked((ulong)BitConverter.DoubleToInt64Bits(value));
            case NumericStorageType.Float32:
                return SingleBits((float)value);
            case NumericStorageType.Float16:
                return DoubleToHalfBits(value);
            default:
                if (double.IsNaN(value)) return 0;
                double t = Math.Truncate(value);
                return type switch
                {
                    NumericStorageType.Int8 => unchecked((ulong)(long)Clamp(t, sbyte.MinValue, sbyte.MaxValue)),
                    NumericStorageType.UInt8 => (ulong)Clamp(t, 0, byte.MaxValue),
                    NumericStorageType.Int16 => unchecked((ulong)(long)Clamp(t, short.MinValue, short.MaxValue)),
                    NumericStorageType.UInt16 => (ulong)Clamp(t, 0, ushort.MaxValue),
                    NumericStorageType.Int32 => unchecked((ulong)(long)Clamp(t, int.MinValue, int.MaxValue)),
                    NumericStorageType.UInt32 => (ulong)Clamp(t, 0, uint.MaxValue),
                    NumericStorageType.Int64 => unchecked((ulong)SaturateToInt64(t)),
                    _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown numeric storage type."),
                };
        }
    }

    private static double Clamp(double v, double lo, double hi) => v < lo ? lo : v > hi ? hi : v;

    private static ulong EncodeInteger(long value, NumericStorageType type) => type switch
    {
        NumericStorageType.Float64 => unchecked((ulong)BitConverter.DoubleToInt64Bits(value)),
        NumericStorageType.Float32 => SingleBits(value),
        NumericStorageType.Float16 => DoubleToHalfBits(value),
        NumericStorageType.Int8 => unchecked((ulong)Math.Min(Math.Max(value, sbyte.MinValue), sbyte.MaxValue)),
        NumericStorageType.UInt8 => (ulong)Math.Min(Math.Max(value, 0), byte.MaxValue),
        NumericStorageType.Int16 => unchecked((ulong)Math.Min(Math.Max(value, short.MinValue), short.MaxValue)),
        NumericStorageType.UInt16 => (ulong)Math.Min(Math.Max(value, 0), ushort.MaxValue),
        NumericStorageType.Int32 => unchecked((ulong)Math.Min(Math.Max(value, int.MinValue), int.MaxValue)),
        NumericStorageType.UInt32 => (ulong)Math.Min(Math.Max(value, 0), uint.MaxValue),
        NumericStorageType.Int64 => unchecked((ulong)value),
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown numeric storage type."),
    };

    private static unsafe ulong SingleBits(float value) => *(uint*)&value;

    /// <summary>
    /// The binary16 bits of <paramref name="value"/>, rounded to nearest even from the double itself (subnormal results
    /// included; past 65504 + half an ulp an infinity). A NaN keeps its sign and the top of its payload, quieted.
    /// </summary>
    internal static ulong DoubleToHalfBits(double value)
    {
        ulong bits = unchecked((ulong)BitConverter.DoubleToInt64Bits(value));
        ulong sign = (bits >> (DoubleBits - HalfBits)) & HalfSignMask;
        int e = (int)((bits >> DoubleMantissaBits) & DoubleExponentMask);
        ulong m = bits & DoubleMantissaMask;
        if (e == DoubleExponentMask)
            return m != 0 ? sign | HalfQuietNaN | ((m >> MantissaShift) & HalfMantissaMask) : sign | HalfInfinity;
        if (e == 0) return sign;
        int ue = e - DoubleExponentBias;
        if (ue > HalfMaxExponent) return sign | HalfInfinity;
        if (ue >= HalfMinExponent)
        {
            ulong hm = RoundShiftRight(m, MantissaShift);
            if (hm == HalfMantissaCarry)
            {
                hm = 0;
                ue++;
                if (ue > HalfMaxExponent) return sign | HalfInfinity;
            }
            return sign | ((ulong)(ue + HalfExponentBias) << HalfMantissaBits) | hm;
        }
        // A subnormal (or zero) result in units of 2^(HalfMinExponent - HalfMantissaBits) = 2^-24, from the full
        // significand (implicit 1 included); a carry into 2^HalfMantissaBits is exactly the smallest normal's bits.
        int subnormalShift = DoubleMantissaBits + HalfMinExponent - HalfMantissaBits - ue;
        return sign | RoundShiftRight(m | (1UL << DoubleMantissaBits), subnormalShift);
    }

    // IEEE-754 binary64 and binary16 field layouts.
    private const int DoubleBits = 64;
    private const int DoubleMantissaBits = 52;
    private const int DoubleExponentMask = 0x7FF;
    private const ulong DoubleMantissaMask = (1UL << DoubleMantissaBits) - 1;
    private const int DoubleExponentBias = 1023;
    private const int HalfBits = 16;
    private const int HalfMantissaBits = 10;
    private const int HalfExponentBias = 15;
    private const int HalfMaxExponent = 15;    // the largest unbiased exponent of a finite half
    private const int HalfMinExponent = -14;   // the smallest unbiased exponent of a normal half
    private const int MantissaShift = DoubleMantissaBits - HalfMantissaBits;
    private const ulong HalfSignMask = 1UL << (HalfBits - 1);
    private const ulong HalfMantissaMask = (1UL << HalfMantissaBits) - 1;
    private const ulong HalfMantissaCarry = 1UL << HalfMantissaBits;
    private const ulong HalfInfinity = 0x1FUL << HalfMantissaBits;
    private const ulong HalfQuietNaN = HalfInfinity | (1UL << (HalfMantissaBits - 1));

    // x >> s rounded to nearest, ties to even (s >= 1).
    private static ulong RoundShiftRight(ulong x, int s)
    {
        if (s >= 64) return s == 64 && x > (1UL << 63) ? 1UL : 0UL;
        ulong q = x >> s;
        ulong rem = x & ((1UL << s) - 1);
        ulong half = 1UL << (s - 1);
        if (rem > half || (rem == half && (q & 1UL) != 0)) q++;
        return q;
    }

    // ------------------------------------------------------------------ TensorCast

    /// <summary>
    /// Converts every element to <typeparamref name="TOut"/>; see
    /// <see cref="TensorDataEngineExtensions.TensorCast{TIn, TOut}"/>.
    /// </summary>
    public virtual Tensor<TOut> TensorCast<TIn, TOut>(Tensor<TIn> input, RoundingMode mode = RoundingMode.TowardZero,
        CastOverflowBehavior overflow = CastOverflowBehavior.Saturate)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        ValidateRounding(mode, overflow);
        var result = new Tensor<TOut>((int[])input._shape.Clone());
        int count = input.Length;
        if (count == 0) return result;
        var destination = result.GetDataArray();
        if (IsIntegral(typeof(TOut)))
        {
            if (IsIntegral(typeof(TIn)))
            {
                var source = ReadAll(input);
                for (int i = 0; i < count; i++) StoreInteger(destination, i, IntegerValue(source[i]), overflow);
            }
            else if (input is Tensor<decimal> decimals)
            {
                // Rounded in decimal, then stored exactly: through double, 9007199254740993m would become ...992 and
                // (decimal)long.MaxValue would round up to 2^63 and read as an overflow.
                var source = ReadAll(decimals);
                for (int i = 0; i < count; i++) StoreInteger(destination, i, RoundDecimal(source[i], mode), overflow);
            }
            else
            {
                var source = RoIAlignPlan.ToDoubles(input);
                CastDoublesToIntegers(source, destination, mode, overflow);
            }
            return result;
        }
        if (destination is decimal[] decimalDestination && IsIntegral(typeof(TIn)))
        {
            // Every 64-bit integer is exact in decimal; FromDouble would round to 15 significant digits.
            var source = ReadAll(input);
            for (int i = 0; i < count; i++) decimalDestination[i] = IntegerValue(source[i]);
            return result;
        }
        if (typeof(TIn) == typeof(TOut))
        {
            Array.Copy(ReadAll(input), destination, count);
            // A floating identity cast is a copy, and a copy is differentiable: record it, or a tape would lose every
            // path through it. (Casts between different element types cannot share a typed tape.)
            DifferentiableOps.RecordIfActive("TensorCast", result, new[] { (Tensor<TOut>)(object)input },
                IdentityCastBackward<TOut>, Array.Empty<object>());
            return result;
        }
        ConvertNonIntegral(ReadAll(input), destination);
        return result;
    }

    // d(identity)/dx = 1: the output gradient passes straight to the input.
    internal static void IdentityCastBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (DifferentiableOps.IsGradientRequired(inputs[0]))
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gradOutput, engine);
    }

    internal static void ValidateRounding(RoundingMode mode, CastOverflowBehavior overflow)
    {
        if (mode < RoundingMode.TowardZero || mode > RoundingMode.TowardPositiveInfinity)
            throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unknown rounding mode.");
        if (overflow != CastOverflowBehavior.Saturate && overflow != CastOverflowBehavior.Throw)
            throw new ArgumentOutOfRangeException(nameof(overflow), overflow, "Unknown overflow behavior.");
    }

    /// <summary>Whether <paramref name="type"/> is one of the eight integral element types TensorCast rounds into.</summary>
    internal static bool IsIntegral(Type type) =>
        type == typeof(int) || type == typeof(long) || type == typeof(short) || type == typeof(sbyte)
        || type == typeof(byte) || type == typeof(ushort) || type == typeof(uint) || type == typeof(ulong);

    // An integral element's exact value (decimal holds every 64-bit integer).
    private static decimal IntegerValue<TValue>(TValue value) => value switch
    {
        int v => v,
        long v => v,
        short v => v,
        sbyte v => v,
        byte v => v,
        ushort v => v,
        uint v => v,
        ulong v => v,
        _ => throw new NotSupportedException($"{typeof(TValue).Name} is not an integral element type."),
    };

    /// <summary><paramref name="value"/> rounded to an integer under <paramref name="mode"/>, in decimal arithmetic.</summary>
    internal static decimal RoundDecimal(decimal value, RoundingMode mode) => mode switch
    {
        RoundingMode.TowardZero => decimal.Truncate(value),
        RoundingMode.ToNearestEven => Math.Round(value, MidpointRounding.ToEven),
        RoundingMode.ToNearestAwayFromZero => Math.Round(value, MidpointRounding.AwayFromZero),
        RoundingMode.TowardNegativeInfinity => decimal.Floor(value),
        _ => decimal.Ceiling(value),
    };

    /// <summary><paramref name="value"/> rounded to an integer under <paramref name="mode"/> (NaN and infinities pass
    /// through).</summary>
    internal static double RoundToIntegral(double value, RoundingMode mode) => mode switch
    {
        RoundingMode.TowardZero => Math.Truncate(value),
        RoundingMode.ToNearestEven => Math.Round(value, MidpointRounding.ToEven),
        RoundingMode.ToNearestAwayFromZero => Math.Round(value, MidpointRounding.AwayFromZero),
        RoundingMode.TowardNegativeInfinity => Math.Floor(value),
        _ => Math.Ceiling(value),
    };

    private static void CastDoublesToIntegers<TOut>(double[] source, TOut[] destination, RoundingMode mode,
        CastOverflowBehavior overflow)
    {
        int chunks = (source.Length + DataOpChunk - 1) / DataOpChunk;
        CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)source.Length * 8, chunk =>
        {
            int from = chunk * DataOpChunk, to = Math.Min(source.Length, from + DataOpChunk);
            for (int i = from; i < to; i++) StoreRounded(destination, i, RoundToIntegral(source[i], mode), overflow);
        }, deterministicSafe: true);
    }

    // The integral bounds of each target as exact doubles: [min, maxExclusive) (2^63 and 2^64 are exact, their
    // predecessors are not).
    internal static (double Min, double MaxExclusive) IntegralRange(Type type)
    {
        if (type == typeof(int)) return (int.MinValue, 2147483648.0);
        if (type == typeof(long)) return (-9223372036854775808.0, 9223372036854775808.0);
        if (type == typeof(short)) return (short.MinValue, 32768.0);
        if (type == typeof(sbyte)) return (sbyte.MinValue, 128.0);
        if (type == typeof(byte)) return (0.0, 256.0);
        if (type == typeof(ushort)) return (0.0, 65536.0);
        if (type == typeof(uint)) return (0.0, 4294967296.0);
        return (0.0, 18446744073709551616.0);
    }

    /// <summary>An already-rounded double stored into the integral array at <paramref name="i"/>: saturated (NaN as 0)
    /// or thrown on overflow.</summary>
    internal static void StoreRounded<TOut>(TOut[] destination, int i, double rounded, CastOverflowBehavior overflow)
    {
        var (min, maxExclusive) = IntegralRange(typeof(TOut));
        if (double.IsNaN(rounded) || rounded < min || rounded >= maxExclusive)
        {
            if (overflow == CastOverflowBehavior.Throw)
                throw new OverflowException($"{rounded.ToString("R", CultureInfo.InvariantCulture)} is outside the range of {typeof(TOut).Name}.");
            if (double.IsNaN(rounded)) rounded = 0.0;
            else if (rounded < min) rounded = min;
            else
            {
                StoreMax(destination, i);
                return;
            }
        }
        // Dispatch on TOut itself, not on the array: the CLR lets a byte[] pass as sbyte[] (and ushort[] as short[],
        // uint[] as int[], ulong[] as long[]), so an array type pattern can pick the wrong conversion.
        object array = destination;
        if (typeof(TOut) == typeof(int)) ((int[])array)[i] = (int)rounded;
        else if (typeof(TOut) == typeof(long)) ((long[])array)[i] = (long)rounded;
        else if (typeof(TOut) == typeof(short)) ((short[])array)[i] = (short)rounded;
        else if (typeof(TOut) == typeof(sbyte)) ((sbyte[])array)[i] = (sbyte)rounded;
        else if (typeof(TOut) == typeof(byte)) ((byte[])array)[i] = (byte)rounded;
        else if (typeof(TOut) == typeof(ushort)) ((ushort[])array)[i] = (ushort)rounded;
        else if (typeof(TOut) == typeof(uint)) ((uint[])array)[i] = (uint)rounded;
        else if (typeof(TOut) == typeof(ulong)) ((ulong[])array)[i] = (ulong)rounded;
    }

    private static void StoreMax<TOut>(TOut[] destination, int i)
    {
        // Dispatch on TOut itself, not on the array: the CLR lets a byte[] pass as sbyte[] (and ushort[] as short[],
        // uint[] as int[], ulong[] as long[]), so an array type pattern can pick the wrong conversion.
        object array = destination;
        if (typeof(TOut) == typeof(int)) ((int[])array)[i] = int.MaxValue;
        else if (typeof(TOut) == typeof(long)) ((long[])array)[i] = long.MaxValue;
        else if (typeof(TOut) == typeof(short)) ((short[])array)[i] = short.MaxValue;
        else if (typeof(TOut) == typeof(sbyte)) ((sbyte[])array)[i] = sbyte.MaxValue;
        else if (typeof(TOut) == typeof(byte)) ((byte[])array)[i] = byte.MaxValue;
        else if (typeof(TOut) == typeof(ushort)) ((ushort[])array)[i] = ushort.MaxValue;
        else if (typeof(TOut) == typeof(uint)) ((uint[])array)[i] = uint.MaxValue;
        else if (typeof(TOut) == typeof(ulong)) ((ulong[])array)[i] = ulong.MaxValue;
    }

    private static void StoreInteger<TOut>(TOut[] destination, int i, decimal value, CastOverflowBehavior overflow)
    {
        decimal min = typeof(TOut) == typeof(long) ? long.MinValue : (decimal)IntegralRange(typeof(TOut)).Min;
        decimal max = typeof(TOut) == typeof(long) ? long.MaxValue
            : typeof(TOut) == typeof(ulong) ? ulong.MaxValue
            : (decimal)IntegralRange(typeof(TOut)).MaxExclusive - 1m;
        if (value < min || value > max)
        {
            if (overflow == CastOverflowBehavior.Throw)
                throw new OverflowException($"{value.ToString(CultureInfo.InvariantCulture)} is outside the range of {typeof(TOut).Name}.");
            value = value < min ? min : max;
        }
        // Dispatch on TOut itself, not on the array: the CLR lets a byte[] pass as sbyte[] (and ushort[] as short[],
        // uint[] as int[], ulong[] as long[]), so an array type pattern can pick the wrong conversion.
        object array = destination;
        if (typeof(TOut) == typeof(int)) ((int[])array)[i] = (int)value;
        else if (typeof(TOut) == typeof(long)) ((long[])array)[i] = (long)value;
        else if (typeof(TOut) == typeof(short)) ((short[])array)[i] = (short)value;
        else if (typeof(TOut) == typeof(sbyte)) ((sbyte[])array)[i] = (sbyte)value;
        else if (typeof(TOut) == typeof(byte)) ((byte[])array)[i] = (byte)value;
        else if (typeof(TOut) == typeof(ushort)) ((ushort[])array)[i] = (ushort)value;
        else if (typeof(TOut) == typeof(uint)) ((uint[])array)[i] = (uint)value;
        else if (typeof(TOut) == typeof(ulong)) ((ulong[])array)[i] = (ulong)value;
    }

    /// <summary>A conversion into a non-integral type: the source type's own float / Half / double conversion, otherwise
    /// through double.</summary>
    [ScalarKernel("A generic element-type conversion goes through INumericOperations one value at a time; float and double targets of float and double sources take typed paths.")]
    private static void ConvertNonIntegral<TIn, TOut>(TIn[] source, TOut[] destination)
    {
        if (source is float[] fs && destination is double[] fd)
        {
            for (int i = 0; i < fs.Length; i++) fd[i] = fs[i];
            return;
        }
        if (source is double[] ds && destination is float[] df)
        {
            for (int i = 0; i < ds.Length; i++) df[i] = (float)ds[i];
            return;
        }
        var inOps = MathHelper.GetNumericOperations<TIn>();
        var outOps = MathHelper.GetNumericOperations<TOut>();
        if (destination is float[] f)
        {
            for (int i = 0; i < source.Length; i++) f[i] = inOps.ToFloat(source[i]);
            return;
        }
        if (destination is double[] d)
        {
            for (int i = 0; i < source.Length; i++) d[i] = inOps.ToDouble(source[i]);
            return;
        }
        if (destination is Half[] h)
        {
            for (int i = 0; i < source.Length; i++) h[i] = inOps.ToHalf(source[i]);
            return;
        }
        for (int i = 0; i < source.Length; i++) destination[i] = outOps.FromDouble(inOps.ToDouble(source[i]));
    }

    // ------------------------------------------------------------------ TensorFormat

    /// <summary>Each element's double value as text; see
    /// <see cref="TensorDataEngineExtensions.TensorFormat{T}(IEngine, Tensor{T}, IFormatProvider, string)"/>.</summary>
    /// <remarks>Intentionally host-only on every engine, the GPU engines included (they inherit this method): the
    /// result is a managed <see cref="string"/> array built by .NET's culture-aware formatting, which no device kernel
    /// can produce, so a device-resident input is read back once and formatted here. There is no device follow-up to
    /// track: a GPU override could only perform the same download.</remarks>
    public virtual string[] TensorFormat<T>(Tensor<T> input, IFormatProvider provider, string? format = null)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (provider is null) throw new ArgumentNullException(nameof(provider));
        var values = RoIAlignPlan.ToDoubles(input);
        var text = new string[values.Length];
        int chunks = (values.Length + DataOpChunk - 1) / DataOpChunk;
        CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)values.Length * 32, chunk =>
        {
            int from = chunk * DataOpChunk, to = Math.Min(values.Length, from + DataOpChunk);
            for (int i = from; i < to; i++) text[i] = values[i].ToString(format, provider);
        }, deterministicSafe: true);
        return text;
    }
}
