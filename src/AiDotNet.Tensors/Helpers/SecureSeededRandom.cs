using System.Security.Cryptography;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// A deterministically-seeded, cryptographically-strong <see cref="Random"/>
/// based on counter-mode SHA-256. Same seed produces the same byte stream on
/// every platform / process / run, but unlike <see cref="System.Random"/>
/// (a linear-congruential generator with known statistical weaknesses) the
/// next output is computationally indistinguishable from uniform random
/// given any prior outputs — under the standard assumption that SHA-256 is
/// a one-way function (NIST SP 800-90A Hash_DRBG construction).
/// </summary>
/// <remarks>
/// <para>
/// Internal state is the 32-bit seed and a 64-bit counter. Each refill
/// hashes <c>(seedBytes || counter)</c> through SHA-256, yielding 32 bytes
/// of output and incrementing the counter. Output bytes are consumed
/// sequentially for <see cref="NextDouble"/> / <see cref="Next()"/> calls.
/// </para>
/// <para>
/// Use this when you need <b>both</b> reproducibility (test determinism,
/// weight-init reproducibility for Clone, regression baselines) and
/// cryptographically-strong output quality. For pure reproducibility where
/// security doesn't matter, <see cref="RandomHelper.CreateSeededRandom"/>
/// (System.Random-based, faster) is sufficient.
/// </para>
/// </remarks>
public sealed class SecureSeededRandom : Random
#if !NET6_0_OR_GREATER
    , IDisposable
#endif
{
    private readonly byte[] _seedBytes;
    private readonly byte[] _buffer = new byte[32]; // SHA-256 output size.
    private readonly byte[] _input = new byte[12];  // seed (4) + counter (8).
    private int _bufferPos = 32; // Force refill on first consume.
    private ulong _counter;
#if !NET6_0_OR_GREATER
    private readonly SHA256 _sha256 = SHA256.Create();
    private bool _disposed;
#endif

    /// <summary>
    /// Creates a new instance seeded with the supplied 32-bit value.
    /// </summary>
    public SecureSeededRandom(int seed)
    {
        _seedBytes = new byte[4];
        _seedBytes[0] = (byte)(seed >> 24);
        _seedBytes[1] = (byte)(seed >> 16);
        _seedBytes[2] = (byte)(seed >> 8);
        _seedBytes[3] = (byte)seed;
        _counter = 0;
    }

    private void Refill()
    {
        // Build input: seedBytes (4) || counter (8 BE).
        Buffer.BlockCopy(_seedBytes, 0, _input, 0, 4);
        WriteUInt64BigEndian(_input, 4, _counter++);
#if NET6_0_OR_GREATER
        SHA256.HashData(_input, _buffer);
#else
        // ComputeHash allocates; reuse via TransformBlock to keep allocations
        // out of the hot path on net471.
        _sha256.Initialize();
        _sha256.TransformFinalBlock(_input, 0, _input.Length);
        Buffer.BlockCopy(_sha256.Hash, 0, _buffer, 0, 32);
#endif
        _bufferPos = 0;
    }

    private static void WriteUInt64BigEndian(byte[] dst, int offset, ulong value)
    {
        dst[offset]     = (byte)(value >> 56);
        dst[offset + 1] = (byte)(value >> 48);
        dst[offset + 2] = (byte)(value >> 40);
        dst[offset + 3] = (byte)(value >> 32);
        dst[offset + 4] = (byte)(value >> 24);
        dst[offset + 5] = (byte)(value >> 16);
        dst[offset + 6] = (byte)(value >> 8);
        dst[offset + 7] = (byte)value;
    }

#if !NET6_0_OR_GREATER
    /// <summary>
    /// Disposes the underlying <see cref="SHA256"/> hasher on .NET Framework.
    /// On modern .NET this type holds no disposable state (SHA256.HashData is
    /// a stateless static).
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _sha256.Dispose();
    }
#endif

    private byte NextByte()
    {
        if (_bufferPos >= _buffer.Length) Refill();
        return _buffer[_bufferPos++];
    }

    private ulong NextUInt64()
    {
        ulong v = 0;
        for (int i = 0; i < 8; i++) v = (v << 8) | NextByte();
        return v;
    }

    /// <inheritdoc/>
    protected override double Sample()
    {
        // Use the top 53 bits of a uint64 — matches double's mantissa size,
        // gives a uniform sample in [0, 1) without bias.
        return (NextUInt64() >> 11) * (1.0 / (1UL << 53));
    }

    /// <inheritdoc/>
    public override double NextDouble() => Sample();

    /// <inheritdoc/>
    public override int Next()
    {
        // System.Random.Next() returns a non-negative Int32 < int.MaxValue.
        // Mask off the sign bit and reject the single exact-int.MaxValue value
        // to match that contract.
        int v;
        do
        {
            v = (int)(NextUInt64() & 0x7FFFFFFFu);
        }
        while (v == int.MaxValue);
        return v;
    }

    /// <inheritdoc/>
    public override int Next(int maxValue)
    {
        if (maxValue < 0)
            throw new ArgumentOutOfRangeException(nameof(maxValue),
                "maxValue must be non-negative.");
        if (maxValue == 0) return 0;

        // Unbiased rejection sampling: find the largest multiple of
        // `maxValue` that fits in uint64, reject samples above that range.
        ulong range = (ulong)maxValue;
        ulong limit = ulong.MaxValue - (ulong.MaxValue % range);
        ulong v;
        do { v = NextUInt64(); } while (v >= limit);
        return (int)(v % range);
    }

    /// <inheritdoc/>
    public override int Next(int minValue, int maxValue)
    {
        if (minValue > maxValue)
            throw new ArgumentOutOfRangeException(nameof(minValue),
                "minValue must be <= maxValue.");
        // Range can be up to UInt32 (e.g. Next(int.MinValue, int.MaxValue)
        // = 2^32 - 1). The previous Math.Min(range, int.MaxValue) silently
        // capped the upper half off, so Next(int.MinValue, int.MaxValue)
        // could never return values in (0, int.MaxValue). Use unsigned
        // rejection sampling over the full 32-bit range and add back to
        // minValue (the cast wraps correctly because the result fits in
        // [minValue, maxValue) by construction).
        uint range = (uint)((long)maxValue - minValue);
        if (range == 0) return minValue;
        uint limit = uint.MaxValue - (uint.MaxValue % range);
        uint v;
        do { v = (uint)(NextUInt64() >> 32); } while (v >= limit);
        return (int)((long)minValue + (long)(v % range));
    }

    /// <inheritdoc/>
    public override void NextBytes(byte[] buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        for (int i = 0; i < buffer.Length; i++) buffer[i] = NextByte();
    }

#if NET6_0_OR_GREATER
    /// <inheritdoc/>
    public override void NextBytes(Span<byte> buffer)
    {
        for (int i = 0; i < buffer.Length; i++) buffer[i] = NextByte();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// System.Random.NextInt64() returns a value in [0, long.MaxValue) (exclusive).
    /// The previous implementation could return long.MaxValue itself: masking off
    /// the sign bit of NextUInt64() yields all 63 lower bits, including the all-
    /// ones pattern that maps to long.MaxValue. Reject long.MaxValue via a small
    /// rejection-sampling loop so the distribution matches the contract.
    /// </remarks>
    public override long NextInt64()
    {
        // Probability of rejection is 1 / 2^63 — effectively never, but
        // rejection-sampling guarantees the contract.
        long v;
        do
        {
            v = (long)(NextUInt64() & 0x7FFF_FFFF_FFFF_FFFFul);
        } while (v == long.MaxValue);
        return v;
    }

    /// <inheritdoc/>
    public override long NextInt64(long maxValue)
    {
        if (maxValue < 0)
            throw new ArgumentOutOfRangeException(nameof(maxValue));
        if (maxValue == 0) return 0;
        ulong range = (ulong)maxValue;
        ulong limit = ulong.MaxValue - (ulong.MaxValue % range);
        ulong v;
        do { v = NextUInt64(); } while (v >= limit);
        return (long)(v % range);
    }

    /// <inheritdoc/>
    public override long NextInt64(long minValue, long maxValue)
    {
        if (minValue > maxValue)
            throw new ArgumentOutOfRangeException(nameof(minValue));
        // Use ulong arithmetic for the range to avoid signed overflow when
        // (maxValue - minValue) exceeds long.MaxValue. The widest case is
        // NextInt64(long.MinValue, long.MaxValue) where the true range is
        // 2^64 - 1, which doesn't fit in a signed long. The cast-and-
        // subtract-as-ulong wraps correctly for any minValue/maxValue
        // pair (since we already validated minValue <= maxValue above):
        //   * minValue=10, maxValue=20: range = 10
        //   * minValue=-5, maxValue=5:  range = 10 (unchecked wrap)
        //   * minValue=long.MinValue, maxValue=long.MaxValue:
        //     range = unchecked((2^63-1) - 2^63) = 2^64-1
        ulong range = unchecked((ulong)maxValue - (ulong)minValue);
        if (range == 0) return minValue;
        ulong limit = ulong.MaxValue - (ulong.MaxValue % range);
        ulong v;
        do { v = NextUInt64(); } while (v >= limit);
        // unchecked: minValue + (v % range) wraps correctly into the
        // [minValue, maxValue) interval thanks to ulong arithmetic.
        return unchecked((long)((ulong)minValue + (v % range)));
    }

    /// <inheritdoc/>
    public override float NextSingle()
    {
        // 24-bit precision matches Single's mantissa.
        return (NextUInt64() >> 40) * (1.0f / (1u << 24));
    }
#endif
}
