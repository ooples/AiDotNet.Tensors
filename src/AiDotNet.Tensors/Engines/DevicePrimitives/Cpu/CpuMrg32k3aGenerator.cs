// Copyright (c) AiDotNet. All rights reserved.
using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;

/// <summary>
/// L'Ecuyer's MRG32k3a combined multiple recursive generator (period about 2^191), producing the same numbers as
/// cuRAND's <c>CURAND_RNG_PSEUDO_MRG32K3A</c> generator in its default ordering for the same seed and offset (#1204).
/// </summary>
/// <remarks>
/// <para><b>What "the same as cuRAND" means.</b> cuRAND does not emit one MRG32k3a stream. It runs
/// <see cref="StreamCount"/> streams, stream <c>j</c> being the seeded generator skipped ahead by <c>j * 2^76</c>
/// steps, and deals their outputs round-robin: output <c>k</c> is element <c>k / StreamCount</c> of stream
/// <c>k % StreamCount</c>. This class reproduces that layout, cuRAND's seeding (the <c>curand_init</c> in its public
/// <c>curand_kernel.h</c>), and its conversions: a float is <c>(float)(z * 2.3283065498378288e-10)</c>, and a normal
/// pair is the Box-Muller transform of two consecutive elements of one stream. Uniform outputs are bit-identical;
/// normal outputs go through <c>log</c>, <c>sin</c> and <c>cos</c>, which can differ from cuRAND's host math library
/// in the last bit.</para>
/// <para><b>Offset.</b> <see cref="Offset"/> is the absolute index of the next uniform output, as
/// <c>curandSetGeneratorOffset</c> defines it. Uniform calls advance it by the number of values produced. A normal
/// call advances individual streams by two elements each, which no single offset describes; after one,
/// <see cref="Offset"/> reports the start of the first row no stream has reached, so restoring it never replays a
/// value (it skips the unread remainder of the rows in between).</para>
/// <para><b>Subsequence.</b> cuRAND's generator API has no subsequence for MRG32k3a. Here subsequence <c>s</c> is
/// the whole generator skipped ahead by <c>s * 2^127</c> steps (L'Ecuyer's stream spacing, far beyond the
/// <c>81,920 * 2^76</c> steps one subsequence spans), so subsequence 0 is cuRAND's output and the others are
/// non-overlapping.</para>
/// </remarks>
public sealed class CpuMrg32k3aGenerator : IDeviceRng
{
    /// <summary>The number of interleaved streams in cuRAND's default MRG32k3a ordering.</summary>
    public const int StreamCount = 81920;

    private const ulong M1 = 4294967087UL, M2 = 4294944443UL;
    private const ulong A12 = 1403580UL, A13N = 810728UL, A21 = 527612UL, A23N = 1370589UL;
    private const double Norm = 2.3283065498378288e-10;      // MRG32K3A_NORM
    private const double BitsNorm = 1.000000048662;           // MRG32K3A_BITS_NORM
    private const float TwoPiFloat = 6.2831855f;              // CURAND_2PI
    private const int Words = 7;                              // s1[3], s2[3], steps taken

    // The one-step transition of each component, acting on the column (s[0], s[1], s[2]).
    private static readonly ulong[] Step1 = { 0, 1, 0, 0, 0, 1, M1 - A13N, A12, 0 };
    private static readonly ulong[] Step2 = { 0, 1, 0, 0, 0, 1, M2 - A23N, 0, A21 };
    private static readonly Lazy<ulong[]> Stream1 = new(() => PowerOfTwo(Step1, 76, M1));
    private static readonly Lazy<ulong[]> Stream2 = new(() => PowerOfTwo(Step2, 76, M2));
    private static readonly Lazy<ulong[]> Sub1 = new(() => PowerOfTwo(Step1, 127, M1));
    private static readonly Lazy<ulong[]> Sub2 = new(() => PowerOfTwo(Step2, 127, M2));

    private ulong _subsequence;
    private readonly ulong[] _seed1 = new ulong[3], _seed2 = new ulong[3];   // the seeded state after the subsequence skip

    // Position: streams are visited in order from _start. Order index o is stream (_start + o) % StreamCount.
    private int _start;
    private ulong _baseRow;          // the element streams _start.. begin at; streams before _start begin one later
    private int _order;              // order index of the stream that produces the next value
    private int _visited;            // order indices [0, _visited) have state
    private uint[] _state = Array.Empty<uint>();
    private readonly ulong[] _chain1 = new ulong[3], _chain2 = new ulong[3];  // initial state of the last stream given state
    private ulong _resetOffset, _uniformCount, _maxPosition;
    private bool _onlyUniform = true;

    /// <inheritdoc/>
    public DeviceRngAlgorithm Algorithm => DeviceRngAlgorithm.Mrg32K3A;

    /// <inheritdoc/>
    public ulong Seed { get; }

    /// <inheritdoc/>
    public ulong Offset
    {
        get => _onlyUniform ? _resetOffset + _uniformCount : checked(_maxPosition * StreamCount);
        set => Reset(value);
    }

    /// <inheritdoc/>
    public ulong Subsequence
    {
        get => _subsequence;
        set
        {
            ulong offset = Offset;
            _subsequence = value;
            SeedState();
            Reset(offset);
        }
    }

    /// <summary>Creates a generator at <paramref name="offset"/> of <paramref name="subsequence"/>.</summary>
    public CpuMrg32k3aGenerator(ulong seed, ulong subsequence = 0, ulong offset = 0)
    {
        Seed = seed;
        _subsequence = subsequence;
        SeedState();
        Reset(offset);
    }

    // ---------------------------------------------------------------- 3x3 modular arithmetic

    private static ulong[] Multiply(ulong[] a, ulong[] b, ulong m)
    {
        var r = new ulong[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                // Every entry is below 2^32, so a product fits in 64 bits and three reduced products do too.
                r[i * 3 + j] = (a[i * 3] * b[j] % m + a[i * 3 + 1] * b[3 + j] % m + a[i * 3 + 2] * b[6 + j] % m) % m;
        return r;
    }

    private static ulong[] PowerOfTwo(ulong[] a, int exponent, ulong m)
    {
        var r = a;
        for (int i = 0; i < exponent; i++) r = Multiply(r, r, m);
        return r;
    }

    private static ulong[] Power(ulong[] a, ulong exponent, ulong m)
    {
        ulong[] r = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        while (exponent != 0)
        {
            if ((exponent & 1) != 0) r = Multiply(r, a, m);
            a = Multiply(a, a, m);
            exponent >>= 1;
        }
        return r;
    }

    private static void Apply(ulong[] matrix, ulong[] v, ulong m, ulong[] result)
    {
        ulong r0 = (matrix[0] * v[0] % m + matrix[1] * v[1] % m + matrix[2] * v[2] % m) % m;
        ulong r1 = (matrix[3] * v[0] % m + matrix[4] * v[1] % m + matrix[5] * v[2] % m) % m;
        ulong r2 = (matrix[6] * v[0] % m + matrix[7] * v[1] % m + matrix[8] * v[2] % m) % m;
        result[0] = r0; result[1] = r1; result[2] = r2;
    }

    // ---------------------------------------------------------------- seeding and positioning

    // curand_init: all six words 12345, each multiplied by a word of the scrambled seed when the seed is not zero.
    private void SeedState()
    {
        for (int i = 0; i < 3; i++) { _seed1[i] = 12345; _seed2[i] = 12345; }
        if (Seed != 0)
        {
            ulong x1 = (uint)Seed ^ 0x55555555u;
            ulong x2 = (uint)(Seed >> 32) ^ 0xAAAAAAAAu;
            _seed1[0] = x1 * _seed1[0] % M1; _seed1[1] = x2 * _seed1[1] % M1; _seed1[2] = x1 * _seed1[2] % M1;
            _seed2[0] = x2 * _seed2[0] % M2; _seed2[1] = x1 * _seed2[1] % M2; _seed2[2] = x2 * _seed2[2] % M2;
        }
        if (_subsequence != 0)
        {
            Apply(Power(Sub1.Value, _subsequence, M1), _seed1, M1, _seed1);
            Apply(Power(Sub2.Value, _subsequence, M2), _seed2, M2, _seed2);
        }
    }

    private void Reset(ulong offset)
    {
        _start = (int)(offset % StreamCount);
        _baseRow = offset / StreamCount;
        _order = 0;
        _visited = 0;
        _resetOffset = offset;
        _uniformCount = 0;
        _maxPosition = _baseRow;
        _onlyUniform = true;
    }

    /// <summary>The slot of the stream that produces the next value, giving it state on its first use.</summary>
    private int CurrentSlot()
    {
        if (_order < _visited) return _order * Words;

        int stream = (_start + _order) % StreamCount;
        if (_visited == 0)
        {
            // The first stream: the seeded state, _baseRow elements in, then _start stream skips.
            Apply(Power(Step1, _baseRow, M1), _seed1, M1, _chain1);
            Apply(Power(Step2, _baseRow, M2), _seed2, M2, _chain2);
            if (stream != 0)
            {
                Apply(Power(Stream1.Value, (ulong)stream, M1), _chain1, M1, _chain1);
                Apply(Power(Stream2.Value, (ulong)stream, M2), _chain2, M2, _chain2);
            }
        }
        else if (stream == 0)
        {
            // Wrapped: the streams before _start have already given their element _baseRow to an earlier row.
            Apply(Power(Step1, _baseRow + 1, M1), _seed1, M1, _chain1);
            Apply(Power(Step2, _baseRow + 1, M2), _seed2, M2, _chain2);
        }
        else
        {
            Apply(Stream1.Value, _chain1, M1, _chain1);
            Apply(Stream2.Value, _chain2, M2, _chain2);
        }

        int slot = _visited * Words;
        if (slot + Words > _state.Length)
            Array.Resize(ref _state, Math.Min(StreamCount * Words, Math.Max(64 * Words, _state.Length * 2)));
        _state[slot] = (uint)_chain1[0]; _state[slot + 1] = (uint)_chain1[1]; _state[slot + 2] = (uint)_chain1[2];
        _state[slot + 3] = (uint)_chain2[0]; _state[slot + 4] = (uint)_chain2[1]; _state[slot + 5] = (uint)_chain2[2];
        _state[slot + 6] = 0;
        _visited++;
        return slot;
    }

    /// <summary>One step of the stream in <paramref name="slot"/>: its next z, in [1, m1].</summary>
    private ulong Step(int slot)
    {
        var s = _state;
        ulong p1 = (A12 * s[slot + 1] + A13N * (M1 - s[slot])) % M1;
        s[slot] = s[slot + 1]; s[slot + 1] = s[slot + 2]; s[slot + 2] = (uint)p1;
        ulong p2 = (A21 * s[slot + 5] + A23N * (M2 - s[slot + 3])) % M2;
        s[slot + 3] = s[slot + 4]; s[slot + 4] = s[slot + 5]; s[slot + 5] = (uint)p2;
        uint steps = ++s[slot + 6];
        ulong position = _baseRow + (ulong)(_order >= StreamCount - _start ? 1 : 0) + steps;
        if (position > _maxPosition) _maxPosition = position;
        return p1 > p2 ? p1 - p2 : p1 + M1 - p2;
    }

    private void NextStream()
    {
        if (++_order == StreamCount) _order = 0;
    }

    private ulong NextUniformZ()
    {
        ulong z = Step(CurrentSlot());
        NextStream();
        _uniformCount++;
        return z;
    }

    /// <summary>The next raw 32-bit output, as <c>curandGenerate</c> returns it.</summary>
    /// <remarks>
    /// z = m1 scales to exactly 2^32, which cuRAND's conversion wraps to 0; the cast through 64 bits does the same
    /// (a direct cast to uint saturates). It is reachable: the seed 0xAAAAAAAA55555555 scrambles to an all-zero
    /// state, which cuRAND, and so this generator, never leaves.
    /// </remarks>
    internal uint NextUInt32() => unchecked((uint)(ulong)(NextUniformZ() * BitsNorm));

    // ---------------------------------------------------------------- IDeviceRng

    /// <inheritdoc/>
    public void Uniform(Tensor<float> output)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        using var lease = output.LeaseWritable();
        var span = lease.Span;
        for (int i = 0; i < span.Length; i++) span[i] = (float)(NextUniformZ() * Norm);
    }

    /// <inheritdoc/>
    public void Uniform(Tensor<double> output)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        using var lease = output.LeaseWritable();
        var span = lease.Span;
        for (int i = 0; i < span.Length; i++) span[i] = NextUniformZ() * Norm;
    }

    /// <inheritdoc/>
    /// <remarks>Values come in pairs, each pair from two consecutive elements of one stream. An odd length uses
    /// the first value of a final pair and discards the second (cuRAND itself rejects an odd length).</remarks>
    public void Normal(Tensor<float> output, float mean = 0f, float stddev = 1f)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        using var lease = output.LeaseWritable();
        var span = lease.Span;
        if (span.Length > 0) _onlyUniform = false;
        for (int i = 0; i < span.Length; i += 2)
        {
            int slot = CurrentSlot();
            float x = (float)(Step(slot) * Norm);
            float y = (float)(Step(slot) * Norm) * TwoPiFloat;
            NextStream();
            float s = MathF.Sqrt(-2.0f * MathF.Log(x));
            span[i] = s * MathF.Sin(y) * stddev + mean;
            if (i + 1 < span.Length) span[i + 1] = s * MathF.Cos(y) * stddev + mean;
        }
    }

    /// <inheritdoc/>
    /// <remarks>See the float overload for how pairs and odd lengths are handled.</remarks>
    public void Normal(Tensor<double> output, double mean = 0, double stddev = 1)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        using var lease = output.LeaseWritable();
        var span = lease.Span;
        if (span.Length > 0) _onlyUniform = false;
        for (int i = 0; i < span.Length; i += 2)
        {
            int slot = CurrentSlot();
            double x = Step(slot) * Norm;
            double y = Step(slot) * Norm * 2.0;
            NextStream();
            double s = Math.Sqrt(-2.0 * Math.Log(x));
            span[i] = s * Math.Sin(y * Math.PI) * stddev + mean;
            if (i + 1 < span.Length) span[i + 1] = s * Math.Cos(y * Math.PI) * stddev + mean;
        }
    }

    /// <inheritdoc/>
    /// <remarks>One uniform float per element: 1 when it is below <paramref name="p"/>.</remarks>
    public void Bernoulli(Tensor<float> output, float p)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (float.IsNaN(p) || p < 0f || p > 1f)
            throw new ArgumentOutOfRangeException(nameof(p), $"Bernoulli probability must be in [0, 1]; got {p}.");
        using var lease = output.LeaseWritable();
        var span = lease.Span;
        if (p == 0f) { span.Fill(0f); return; }
        if (p == 1f) { span.Fill(1f); return; }
        for (int i = 0; i < span.Length; i++) span[i] = (float)(NextUniformZ() * Norm) < p ? 1f : 0f;
    }
}
