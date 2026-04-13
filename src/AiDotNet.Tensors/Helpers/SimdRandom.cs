using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// SIMD-parallel random number generator using xoshiro256** algorithm.
/// Produces 4 doubles per AVX2 iteration (4x faster than scalar Random.NextDouble).
/// </summary>
/// <remarks>
/// <para>
/// xoshiro256** is a fast, high-quality PRNG suitable for ML workloads (weight
/// initialization, dropout masks, data augmentation). It is NOT cryptographically
/// secure — use <see cref="System.Security.Cryptography.RandomNumberGenerator"/>
/// for security-sensitive applications.
/// </para>
/// <para>
/// The SIMD implementation processes 4 independent xoshiro256** states in parallel
/// using Vector256&lt;ulong&gt;, producing 4 uniform doubles in [0, 1) per iteration.
/// </para>
/// </remarks>
public sealed class SimdRandom
{
    // 4 parallel xoshiro256** states (each state = 4 x ulong)
    private ulong[] _s0 = new ulong[4];
    private ulong[] _s1 = new ulong[4];
    private ulong[] _s2 = new ulong[4];
    private ulong[] _s3 = new ulong[4];

    /// <summary>
    /// Creates a new SIMD random generator with the given seed.
    /// </summary>
    public SimdRandom(int seed)
    {
        // Use SplitMix64 to initialize 4 independent states from a single seed
        ulong s = (ulong)seed;
        for (int lane = 0; lane < 4; lane++)
        {
            _s0[lane] = SplitMix64(ref s);
            _s1[lane] = SplitMix64(ref s);
            _s2[lane] = SplitMix64(ref s);
            _s3[lane] = SplitMix64(ref s);
        }
    }

    private static int _seedCounter = Environment.TickCount;

    /// <summary>
    /// Creates a new SIMD random generator with a unique random seed.
    /// Each call produces a different seed via atomic increment.
    /// </summary>
    public SimdRandom() : this(System.Threading.Interlocked.Increment(ref _seedCounter)) { }

    /// <summary>
    /// Fills a span with uniform random doubles in [0, 1).
    /// Uses AVX2 SIMD when available for 4x throughput.
    /// </summary>
    public void NextDoubles(Span<double> destination)
    {
        int length = destination.Length;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && length >= 4)
        {
            var vs0 = Vector256.Create(_s0[0], _s0[1], _s0[2], _s0[3]);
            var vs1 = Vector256.Create(_s1[0], _s1[1], _s1[2], _s1[3]);
            var vs2 = Vector256.Create(_s2[0], _s2[1], _s2[2], _s2[3]);
            var vs3 = Vector256.Create(_s3[0], _s3[1], _s3[2], _s3[3]);

            // Constant for converting ulong to double [0, 1)
            // We use the upper 52 bits of the 64-bit result, mapped to [1.0, 2.0) then subtract 1.0
            var one = Vector256.Create(1.0);
            var exponentMask = Vector256.Create(0x3FF0000000000000UL); // IEEE 754 exponent for 1.0
            var mantissaMask = Vector256.Create(0x000FFFFFFFFFFFFFUL); // 52-bit mantissa

            int simdLength = i + ((length - i) & ~3);
            for (; i < simdLength; i += 4)
            {
                // xoshiro256** result: rotl(s1 * 5, 7) * 9
                var s1x5 = Multiply(vs1, Vector256.Create(5UL));
                var result = Multiply(RotateLeft7(s1x5), Vector256.Create(9UL));

                // Convert to double [0, 1): shift right 12 to use upper 52 bits (matches scalar path),
                // then set exponent to 1.0, subtract 1.0
                var shifted = Avx2.ShiftRightLogical(result, 12);
                var bits = Avx2.Or(Avx2.And(shifted, mantissaMask), exponentMask);
                var doubles = Avx.Subtract(bits.AsDouble(), one);

                // Store 4 doubles
                Unsafe.WriteUnaligned(
                    ref Unsafe.As<double, byte>(ref destination[i]),
                    doubles);

                // Advance state: xoshiro256** step
                var t = Avx2.ShiftLeftLogical(vs1, 17);
                vs2 = Avx2.Xor(vs2, vs0);
                vs3 = Avx2.Xor(vs3, vs1);
                vs1 = Avx2.Xor(vs1, vs2);
                vs0 = Avx2.Xor(vs0, vs3);
                vs2 = Avx2.Xor(vs2, t);
                vs3 = RotateLeft45(vs3);
            }

            // Store state back
            for (int lane = 0; lane < 4; lane++)
            {
                _s0[lane] = vs0.GetElement(lane);
                _s1[lane] = vs1.GetElement(lane);
                _s2[lane] = vs2.GetElement(lane);
                _s3[lane] = vs3.GetElement(lane);
            }
        }
#endif

        // Scalar tail
        for (; i < length; i++)
        {
            destination[i] = NextDoubleScalar();
        }
    }

    /// <summary>
    /// Fills a span with uniform random floats in [0, 1).
    /// </summary>
    public void NextFloats(Span<float> destination)
    {
        int length = destination.Length;
        int i = 0;

#if NET5_0_OR_GREATER
        // Process 4 at a time: generate 4 doubles then convert to float
        if (Avx2.IsSupported && length >= 4)
        {
            Span<double> temp = stackalloc double[4];
            int simdLength = i + ((length - i) & ~3);
            for (; i < simdLength; i += 4)
            {
                NextDoubles(temp);
                destination[i] = Math.Min((float)temp[0], 0.99999994f);
                destination[i + 1] = Math.Min((float)temp[1], 0.99999994f);
                destination[i + 2] = Math.Min((float)temp[2], 0.99999994f);
                destination[i + 3] = Math.Min((float)temp[3], 0.99999994f);
            }
        }
#endif

        for (; i < length; i++)
        {
            destination[i] = Math.Min((float)NextDoubleScalar(), 0.99999994f);
        }
    }

    /// <summary>
    /// Fills a span with normally distributed doubles (mean=0, stddev=1)
    /// using the Box-Muller transform.
    /// </summary>
    public void NextNormalDoubles(Span<double> destination)
    {
        int length = destination.Length;
        int i = 0;

        // Box-Muller produces pairs
        Span<double> u = stackalloc double[2];
        for (; i + 1 < length; i += 2)
        {
            NextDoubles(u);
            // Clamp u[0] away from 0 to avoid log(0)
            double u1 = Math.Max(u[0], double.Epsilon);
            double u2 = u[1];
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            destination[i] = r * Math.Cos(theta);
            destination[i + 1] = r * Math.Sin(theta);
        }

        // Handle odd length
        if (i < length)
        {
            NextDoubles(u);
            double u1 = Math.Max(u[0], double.Epsilon);
            double u2 = u[1];
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            destination[i] = r * Math.Cos(2.0 * Math.PI * u2);
        }
    }

    /// <summary>Scalar xoshiro256** for the tail.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double NextDoubleScalar()
    {
        ulong result = RotateLeft64(_s1[0] * 5, 7) * 9;

        // Convert to [0, 1)
        double d = BitConverter.Int64BitsToDouble((long)((result >> 12) | 0x3FF0000000000000UL)) - 1.0;

        // Advance lane 0 state
        ulong t = _s1[0] << 17;
        _s2[0] ^= _s0[0];
        _s3[0] ^= _s1[0];
        _s1[0] ^= _s2[0];
        _s0[0] ^= _s3[0];
        _s2[0] ^= t;
        _s3[0] = RotateLeft64(_s3[0], 45);

        return d;
    }

    /// <summary>
    /// Fills a span with uniform random values in [0, 1), using type-specialized
    /// direct writes for float/double to avoid per-element NumOps.FromDouble overhead.
    /// Not cryptographically secure — uses xoshiro256** PRNG suitable for ML workloads.
    /// </summary>
    public void FillUniform<T>(Span<T> destination)
    {
        if (typeof(T) == typeof(double))
        {
            const int chunk = 256;
            var buf = System.Buffers.ArrayPool<double>.Shared.Rent(chunk);
            try
            {
                int i = 0;
                while (i < destination.Length)
                {
                    int n = Math.Min(chunk, destination.Length - i);
                    NextDoubles(buf.AsSpan(0, n));
                    for (int j = 0; j < n; j++)
                        destination[i + j] = Unsafe.As<double, T>(ref buf[j]);
                    i += n;
                }
            }
            finally { System.Buffers.ArrayPool<double>.Shared.Return(buf); }
        }
        else if (typeof(T) == typeof(float))
        {
            const int chunk = 256;
            var buf = System.Buffers.ArrayPool<float>.Shared.Rent(chunk);
            try
            {
                int i = 0;
                while (i < destination.Length)
                {
                    int n = Math.Min(chunk, destination.Length - i);
                    NextFloats(buf.AsSpan(0, n));
                    for (int j = 0; j < n; j++)
                        destination[i + j] = Unsafe.As<float, T>(ref buf[j]);
                    i += n;
                }
            }
            finally { System.Buffers.ArrayPool<float>.Shared.Return(buf); }
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            var temp = System.Buffers.ArrayPool<double>.Shared.Rent(Math.Min(256, destination.Length));
            try
            {
                int i = 0;
                while (i < destination.Length)
                {
                    int n = Math.Min(256, destination.Length - i);
                    NextDoubles(temp.AsSpan(0, n));
                    for (int j = 0; j < n; j++)
                        destination[i + j] = numOps.FromDouble(temp[j]);
                    i += n;
                }
            }
            finally { System.Buffers.ArrayPool<double>.Shared.Return(temp); }
        }
    }

    /// <summary>
    /// Fills a span with uniform random values in [min, max), with type-specialized
    /// fast paths for float/double that avoid per-element NumOps virtual dispatch.
    /// </summary>
    public void FillUniformRange<T>(Span<T> destination, double min, double max)
    {
        double range = max - min;
        if (typeof(T) == typeof(double))
        {
            const int chunk = 256;
            var buf = System.Buffers.ArrayPool<double>.Shared.Rent(chunk);
            try
            {
                int i = 0;
                while (i < destination.Length)
                {
                    int n = Math.Min(chunk, destination.Length - i);
                    NextDoubles(buf.AsSpan(0, n));
                    for (int j = 0; j < n; j++)
                    {
                        double val = buf[j] * range + min;
                        destination[i + j] = Unsafe.As<double, T>(ref val);
                    }
                    i += n;
                }
            }
            finally { System.Buffers.ArrayPool<double>.Shared.Return(buf); }
        }
        else if (typeof(T) == typeof(float))
        {
            const int chunk = 256;
            var buf = System.Buffers.ArrayPool<float>.Shared.Rent(chunk);
            try
            {
                int i = 0;
                while (i < destination.Length)
                {
                    int n = Math.Min(chunk, destination.Length - i);
                    NextFloats(buf.AsSpan(0, n));
                    for (int j = 0; j < n; j++)
                    {
                        float val = buf[j] * (float)range + (float)min;
                        destination[i + j] = Unsafe.As<float, T>(ref val);
                    }
                    i += n;
                }
            }
            finally { System.Buffers.ArrayPool<float>.Shared.Return(buf); }
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            var temp = System.Buffers.ArrayPool<double>.Shared.Rent(Math.Min(256, destination.Length));
            try
            {
                int i = 0;
                while (i < destination.Length)
                {
                    int n = Math.Min(256, destination.Length - i);
                    NextDoubles(temp.AsSpan(0, n));
                    for (int j = 0; j < n; j++)
                        destination[i + j] = numOps.FromDouble(temp[j] * range + min);
                    i += n;
                }
            }
            finally { System.Buffers.ArrayPool<double>.Shared.Return(temp); }
        }
    }

    /// <summary>
    /// Fills a span with cryptographically secure random doubles in [0, 1).
    /// Uses RandomNumberGenerator.Fill for bulk byte generation, then SIMD-converts
    /// to doubles. Much faster than per-element LockedRandom.NextDouble() for large fills.
    /// </summary>
    public static void SecureFillDoubles(Span<double> destination)
    {
        int byteCount = destination.Length * 8;
        var bytes = System.Buffers.ArrayPool<byte>.Shared.Rent(byteCount);
        try
        {
#if NET8_0_OR_GREATER
            System.Security.Cryptography.RandomNumberGenerator.Fill(bytes.AsSpan(0, byteCount));
#else
            using var rng = System.Security.Cryptography.RandomNumberGenerator.Create();
            rng.GetBytes(bytes, 0, byteCount);
#endif
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && destination.Length >= 4)
            {
                var one = Vector256.Create(1.0);
                var exponentMask = Vector256.Create(0x3FF0000000000000UL);
                var mantissaMask = Vector256.Create(0x000FFFFFFFFFFFFFUL);

                int simdLen = destination.Length & ~3;
                for (; i < simdLen; i += 4)
                {
                    // Load 4 x 8 bytes as 4 ulongs
                    var raw = Unsafe.ReadUnaligned<Vector256<ulong>>(
                        ref bytes[i * 8]);
                    var shifted = Avx2.ShiftRightLogical(raw, 12);
                    var bits = Avx2.Or(Avx2.And(shifted, mantissaMask), exponentMask);
                    var doubles = Avx.Subtract(bits.AsDouble(), one);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref destination[i]),
                        doubles);
                }
            }
#endif
            // Scalar tail
            for (; i < destination.Length; i++)
            {
                ulong raw = BitConverter.ToUInt64(bytes, i * 8);
                destination[i] = BitConverter.Int64BitsToDouble(
                    (long)((raw >> 12) | 0x3FF0000000000000UL)) - 1.0;
            }
        }
        finally { System.Buffers.ArrayPool<byte>.Shared.Return(bytes, clearArray: true); }
    }

    /// <summary>
    /// Fills a span with cryptographically secure random floats in [0, 1).
    /// </summary>
    public static void SecureFillFloats(Span<float> destination)
    {
        int byteCount = destination.Length * 4;
        var bytes = System.Buffers.ArrayPool<byte>.Shared.Rent(byteCount);
        try
        {
#if NET8_0_OR_GREATER
            System.Security.Cryptography.RandomNumberGenerator.Fill(bytes.AsSpan(0, byteCount));
#else
            using var rng = System.Security.Cryptography.RandomNumberGenerator.Create();
            rng.GetBytes(bytes, 0, byteCount);
#endif
            for (int i = 0; i < destination.Length; i++)
            {
                uint raw = BitConverter.ToUInt32(bytes, i * 4);
                // Use upper 23 bits mapped to [1.0f, 2.0f) then subtract 1.0f
                int bits = (int)((raw >> 9) | 0x3F800000U);
                destination[i] = Unsafe.As<int, float>(ref bits) - 1.0f;
            }
        }
        finally { System.Buffers.ArrayPool<byte>.Shared.Return(bytes, clearArray: true); }
    }

    /// <summary>
    /// Fills a span with cryptographically secure uniform random values in [0, 1).
    /// Type-specialized: float/double use bulk crypto RNG + SIMD conversion.
    /// Other types use batched crypto doubles + NumOps.FromDouble.
    /// </summary>
    public static void SecureFillUniform<T>(Span<T> destination)
    {
        if (typeof(T) == typeof(double))
        {
            const int chunk = 4096;
            var buf = System.Buffers.ArrayPool<double>.Shared.Rent(Math.Min(chunk, destination.Length));
            try
            {
                int i = 0;
                while (i < destination.Length)
                {
                    int n = Math.Min(buf.Length, destination.Length - i);
                    SecureFillDoubles(buf.AsSpan(0, n));
                    for (int j = 0; j < n; j++)
                        destination[i + j] = Unsafe.As<double, T>(ref buf[j]);
                    i += n;
                }
            }
            finally { System.Buffers.ArrayPool<double>.Shared.Return(buf, clearArray: true); }
        }
        else if (typeof(T) == typeof(float))
        {
            const int chunk = 4096;
            var buf = System.Buffers.ArrayPool<float>.Shared.Rent(Math.Min(chunk, destination.Length));
            try
            {
                int i = 0;
                while (i < destination.Length)
                {
                    int n = Math.Min(buf.Length, destination.Length - i);
                    SecureFillFloats(buf.AsSpan(0, n));
                    for (int j = 0; j < n; j++)
                        destination[i + j] = Unsafe.As<float, T>(ref buf[j]);
                    i += n;
                }
            }
            finally { System.Buffers.ArrayPool<float>.Shared.Return(buf, clearArray: true); }
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            const int chunk = 256;
            var buf = System.Buffers.ArrayPool<double>.Shared.Rent(chunk);
            try
            {
                int i = 0;
                while (i < destination.Length)
                {
                    int n = Math.Min(chunk, destination.Length - i);
                    SecureFillDoubles(buf.AsSpan(0, n));
                    for (int j = 0; j < n; j++)
                        destination[i + j] = numOps.FromDouble(buf[j]);
                    i += n;
                }
            }
            finally { System.Buffers.ArrayPool<double>.Shared.Return(buf, clearArray: true); }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong SplitMix64(ref ulong state)
    {
        ulong z = (state += 0x9E3779B97F4A7C15UL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong RotateLeft64(ulong x, int k) => (x << k) | (x >> (64 - k));

#if NET5_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<ulong> RotateLeft7(Vector256<ulong> x)
    {
        return Avx2.Or(Avx2.ShiftLeftLogical(x, 7), Avx2.ShiftRightLogical(x, 57));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<ulong> RotateLeft17(Vector256<ulong> x)
    {
        return Avx2.Or(Avx2.ShiftLeftLogical(x, 17), Avx2.ShiftRightLogical(x, 47));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<ulong> RotateLeft45(Vector256<ulong> x)
    {
        return Avx2.Or(Avx2.ShiftLeftLogical(x, 45), Avx2.ShiftRightLogical(x, 19));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<ulong> Multiply(Vector256<ulong> a, Vector256<ulong> b)
    {
        // AVX2 doesn't have 64-bit multiply, emulate with 32-bit parts
        // a * b = (a_lo * b_lo) + ((a_lo * b_hi + a_hi * b_lo) << 32)
        var aLo = a.AsUInt32();
        var bLo = b.AsUInt32();
        var mulLo = Avx2.Multiply(aLo, bLo); // 32x32 -> 64 (even elements)
        var aSwap = Avx2.Shuffle(aLo, 0xB1); // swap hi/lo 32-bit within 64-bit
        var mulHiA = Avx2.Multiply(aSwap, bLo);
        var bSwap = Avx2.Shuffle(bLo, 0xB1);
        var mulHiB = Avx2.Multiply(aLo, bSwap);
        var hi = Avx2.Add(mulHiA.AsUInt64(), mulHiB.AsUInt64());
        hi = Avx2.ShiftLeftLogical(hi, 32);
        return Avx2.Add(mulLo.AsUInt64(), hi);
    }
#endif
}
