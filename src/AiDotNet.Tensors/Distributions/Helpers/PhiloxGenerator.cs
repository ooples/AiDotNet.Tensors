using System;

namespace AiDotNet.Tensors.Distributions.Helpers;

/// <summary>
/// Philox 4×32-10 counter-based RNG (Salmon et al., 2011) — a stateless, deterministic,
/// parallel-friendly random number generator that matches the algorithm <c>torch.Generator</c>
/// uses on CUDA.
///
/// The generator is uniquely identified by a 64-bit seed and a 64-bit per-call counter; the
/// 64-bit subkey can be set explicitly to derive independent streams for different batch rows.
/// This makes it possible to seed the random draws for sample <c>i</c> in a batch by hashing
/// the batch index — a guarantee PyTorch's host-side <c>torch.Generator</c> does not provide
/// without manual workarounds.
///
/// Performance: stateless, vectorisable, no contention; one Philox call produces 4 32-bit
/// uniforms per round. For our usage we expose a <see cref="Random"/>-shaped facade so all
/// distributions can use it interchangeably with <see cref="Random"/> via duck-typing on
/// <see cref="NextDouble"/>.
/// </summary>
public sealed class PhiloxGenerator : Random
{
    private ulong _counter;
    private readonly ulong _seed;
    private ulong _subKey;
    private uint _bufRemaining;
    private readonly uint[] _buffer = new uint[4];

    /// <summary>Build a Philox generator with the given 64-bit seed and per-stream subkey.</summary>
    public PhiloxGenerator(ulong seed, ulong subKey = 0UL)
    {
        _seed = seed;
        _subKey = subKey;
        _counter = 0;
        _bufRemaining = 0;
    }

    /// <summary>Override the per-stream subkey to derive an independent stream from the same seed.
    /// Counter is reset so the new stream starts at the beginning.</summary>
    public void SetSubKey(ulong subKey)
    {
        _subKey = subKey;
        _counter = 0;
        _bufRemaining = 0;
    }

    /// <summary>Hash <paramref name="batchIndex"/> into the subkey so that each batch row sees a
    /// distinct, deterministic stream from the same seed. Useful for batch-parallel sampling.</summary>
    public void SetBatchRow(int batchIndex)
    {
        // Murmur3 fmix64 hash — well-distributed, fast.
        ulong h = (ulong)batchIndex ^ _seed;
        h ^= h >> 33; h *= 0xff51afd7ed558ccdUL;
        h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53UL;
        h ^= h >> 33;
        SetSubKey(h);
    }

    /// <inheritdoc />
    public override int Next() => (int)(NextUInt32() & 0x7FFFFFFFu);

    /// <inheritdoc />
    public override int Next(int maxValue)
    {
        if (maxValue <= 0) throw new ArgumentOutOfRangeException(nameof(maxValue));
        return (int)(NextUInt32() % (uint)maxValue);
    }

    /// <inheritdoc />
    public override int Next(int minValue, int maxValue)
    {
        if (minValue >= maxValue) throw new ArgumentOutOfRangeException(nameof(minValue));
        long range = (long)maxValue - minValue;
        return (int)(minValue + (long)(NextUInt32() % (uint)range));
    }

    /// <inheritdoc />
    public override double NextDouble()
    {
        // 53-bit mantissa: combine two 32-bit uniforms.
        ulong hi = NextUInt32();
        ulong lo = NextUInt32();
        ulong bits = (hi << 21) | (lo >> 11);
        return bits * (1.0 / (1UL << 53));
    }

    /// <inheritdoc />
    public override void NextBytes(byte[] buffer)
    {
        for (int i = 0; i < buffer.Length; i++) buffer[i] = (byte)NextUInt32();
    }

    private uint NextUInt32()
    {
        if (_bufRemaining == 0)
        {
            FillBuffer();
            _bufRemaining = 4;
        }
        return _buffer[--_bufRemaining];
    }

    private void FillBuffer()
    {
        // Each Philox call: 4 input 32-bit words = (counter_low, counter_high, subkey_low, subkey_high).
        uint c0 = (uint)(_counter & 0xFFFFFFFFUL);
        uint c1 = (uint)(_counter >> 32);
        uint c2 = (uint)(_subKey & 0xFFFFFFFFUL);
        uint c3 = (uint)(_subKey >> 32);
        uint k0 = (uint)(_seed & 0xFFFFFFFFUL);
        uint k1 = (uint)(_seed >> 32);

        const uint M0 = 0xD2511F53u;
        const uint M1 = 0xCD9E8D57u;
        const uint W0 = 0x9E3779B9u;  // Weyl constants (golden ratio)
        const uint W1 = 0xBB67AE85u;

        for (int round = 0; round < 10; round++)
        {
            ulong p0 = (ulong)M0 * c0;
            ulong p1 = (ulong)M1 * c2;
            uint hi0 = (uint)(p0 >> 32);
            uint lo0 = (uint)p0;
            uint hi1 = (uint)(p1 >> 32);
            uint lo1 = (uint)p1;

            uint nc0 = hi1 ^ c1 ^ k0;
            uint nc1 = lo1;
            uint nc2 = hi0 ^ c3 ^ k1;
            uint nc3 = lo0;
            c0 = nc0; c1 = nc1; c2 = nc2; c3 = nc3;
            k0 += W0; k1 += W1;
        }

        _buffer[0] = c0; _buffer[1] = c1; _buffer[2] = c2; _buffer[3] = c3;
        _counter++;
    }
}
