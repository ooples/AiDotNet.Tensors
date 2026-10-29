// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;

/// <summary>
/// The 32-bit Sobol quasi-random sequence (Joe-Kuo direction numbers), plain or scrambled (#1204).
/// </summary>
/// <remarks>
/// <para><b>Plain</b> (<see cref="DeviceRngAlgorithm.Sobol"/>) produces cuRAND's <c>CURAND_RNG_QUASI_SOBOL32</c>
/// output: point <c>i</c> of a dimension is the XOR of that dimension's direction vectors over the set bits of the
/// Gray code of <c>i</c>, starting from point 0 (which is all zeros). A call for <c>n</c> values with <c>D</c>
/// dimensions needs <c>n</c> to be a multiple of <c>D</c> and writes <c>n / D</c> points dimension by dimension: all
/// of dimension 0, then all of dimension 1, and so on, as cuRAND does. Uniform values are bit-identical to cuRAND's
/// (<c>x * 2^-32 + 2^-33</c> in float, <c>x * 2^-32 + 2^-32</c> in double). Normal values are the inverse normal
/// CDF of the point, which is what cuRAND computes; they agree to rounding, not bit for bit.</para>
/// <para><b>Scrambled</b> (<see cref="DeviceRngAlgorithm.ScrambledSobol"/>) is this library's own seeded scramble,
/// identical on every backend. It is not cuRAND's scrambled sequence, whose table exists only inside cuRAND. Each
/// dimension gets a random lower-triangular binary matrix with a unit diagonal, applied to its direction vectors,
/// and a random 32-bit shift XORed into every point (Matousek's linear matrix scramble with a digital shift, the
/// scheme PyTorch's and SciPy's scrambled Sobol engines use). Both come from Philox keyed by the seed and the
/// dimension, so a seed fixes the sequence. The scramble keeps the sequence's equidistribution.</para>
/// <para><b>Offset</b> is the index of the next point and advances by the number of points produced.
/// <b>Subsequence</b> is the first dimension used: a generator with subsequence <c>s</c> and <c>D</c> dimensions
/// produces dimensions <c>s .. s + D - 1</c>, so parallel consumers can take disjoint dimensions of one sequence.
/// <b>Seed</b> is used by the scramble only.</para>
/// </remarks>
public sealed class CpuSobolGenerator : IDeviceRng
{
    /// <summary>The number of dimensions available (cuRAND's limit).</summary>
    public const int MaxDimensions = SobolDirectionNumbers.MaxDimensions;

    private const float Inv32Float = 2.3283064e-10f;           // CURAND_2POW32_INV
    private const double Inv32Double = 2.3283064365386963e-10;  // CURAND_2POW32_INV_DOUBLE

    private readonly bool _scrambled;
    private ulong _subsequence;
    private uint[][]? _vectors;    // per dimension, 32 vectors
    private uint[]? _shift;        // per dimension

    /// <inheritdoc/>
    public DeviceRngAlgorithm Algorithm => _scrambled ? DeviceRngAlgorithm.ScrambledSobol : DeviceRngAlgorithm.Sobol;

    /// <inheritdoc/>
    public ulong Seed { get; }

    /// <summary>The number of dimensions each point has.</summary>
    public int Dimensions { get; }

    /// <inheritdoc/>
    public ulong Offset { get; set; }

    /// <inheritdoc/>
    public ulong Subsequence
    {
        get => _subsequence;
        set
        {
            CheckDimensions(Dimensions, value);
            _subsequence = value;
            _vectors = null;
            _shift = null;
        }
    }

    /// <summary>Creates a Sobol generator with <paramref name="dimensions"/> dimensions.</summary>
    /// <param name="dimensions">Dimensions per point, 1 to <see cref="MaxDimensions"/>.</param>
    /// <param name="scrambled">True for the seeded scramble.</param>
    /// <param name="seed">The scramble's seed. Ignored by the plain sequence.</param>
    /// <param name="subsequence">The first dimension used.</param>
    /// <param name="offset">The index of the first point.</param>
    public CpuSobolGenerator(int dimensions = 1, bool scrambled = false, ulong seed = 0, ulong subsequence = 0, ulong offset = 0)
    {
        CheckDimensions(dimensions, subsequence);
        Dimensions = dimensions;
        _scrambled = scrambled;
        Seed = seed;
        _subsequence = subsequence;
        Offset = offset;
    }

    private static void CheckDimensions(int dimensions, ulong firstDimension)
    {
        if (dimensions < 1 || dimensions > MaxDimensions)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, $"Sobol dimensions are 1 to {MaxDimensions}.");
        if (firstDimension > (ulong)(MaxDimensions - dimensions))
            throw new ArgumentOutOfRangeException(nameof(firstDimension), firstDimension,
                $"Dimensions {firstDimension} to {firstDimension + (ulong)dimensions - 1} are past the {MaxDimensions} available.");
    }

    private void EnsureTables()
    {
        if (_vectors is not null) return;
        var vectors = new uint[Dimensions][];
        var shift = new uint[Dimensions];
        for (int d = 0; d < Dimensions; d++)
        {
            int dimension = (int)_subsequence + d;
            uint[] v = SobolDirectionNumbers.Vectors(dimension);
            if (_scrambled) Scramble(v, dimension, out shift[d]);
            vectors[d] = v;
        }
        _shift = shift;
        _vectors = vectors;
    }

    /// <summary>
    /// Applies the dimension's scramble: v'_k = L v_k for a random lower-triangular L with a unit diagonal (row r
    /// of L decides which of the more significant bits of a vector feed output bit r), and a random shift.
    /// </summary>
    private void Scramble(uint[] v, int dimension, out uint shift)
    {
        // 33 words per dimension from Philox keyed by the seed, on the subsequence that is the dimension: one row
        // mask per output bit and the shift. Nine blocks of four give 36.
        var philox = new CpuPhiloxGenerator(Seed, subsequence: (ulong)dimension);
        var words = new uint[36];
        for (int block = 0; block < 9; block++)
        {
            var (r0, r1, r2, r3) = philox.NextBlock();
            words[block * 4] = r0; words[block * 4 + 1] = r1; words[block * 4 + 2] = r2; words[block * 4 + 3] = r3;
        }

        // Bit 31 is the most significant digit. Output bit (31 - r) takes input bit (31 - r) (the unit diagonal)
        // and a random subset of the more significant input bits (31 .. 32 - r).
        var rows = new uint[32];
        for (int r = 0; r < 32; r++)
        {
            uint above = r == 0 ? 0u : words[r] & ~(uint.MaxValue >> r);
            rows[r] = above | (1u << (31 - r));
        }
        for (int k = 0; k < 32; k++)
        {
            uint input = v[k], output = 0;
            for (int r = 0; r < 32; r++)
            {
                if ((Parity(input & rows[r]) & 1u) != 0) output |= 1u << (31 - r);
            }
            v[k] = output;
        }
        shift = words[32];
    }

    private static uint Parity(uint x)
    {
        x ^= x >> 16; x ^= x >> 8; x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
        return x & 1u;
    }

    /// <summary>The count of points a call for <paramref name="length"/> values produces.</summary>
    private int PointCount(int length)
    {
        if (length % Dimensions != 0)
            throw new ArgumentException(
                $"A Sobol call writes whole points dimension by dimension: the length ({length}) must be a multiple of the dimension count ({Dimensions}).");
        return length / Dimensions;
    }

    /// <summary>Point <paramref name="index"/> of one dimension: the XOR of its vectors over the index's Gray code.</summary>
    private static uint PointAt(uint[] v, uint shift, ulong index)
    {
        // The 32-bit sequence repeats after 2^32 points, as cuRAND's does.
        uint gray = (uint)index ^ ((uint)index >> 1);
        uint x = shift;
        for (int k = 0; gray != 0; k++, gray >>= 1)
        {
            if ((gray & 1u) != 0) x ^= v[k];
        }
        return x;
    }

    private static int TrailingZeros(uint value)
    {
        int count = 0;
        while ((value & 1u) == 0) { value >>= 1; count++; }
        return count;
    }

    private delegate void Emit(int index, uint x);

    /// <summary>Produces <paramref name="points"/> points and hands each value to <paramref name="emit"/> with its place in the output.</summary>
    private void Generate(int points, Emit emit)
    {
        if (points == 0) return;
        EnsureTables();
        for (int d = 0; d < Dimensions; d++)
        {
            uint[] v = _vectors![d];
            uint x = PointAt(v, _shift![d], Offset);
            int at = d * points;
            emit(at, x);
            for (int i = 1; i < points; i++)
            {
                // Consecutive Gray codes differ in one bit: the lowest zero bit of the previous index.
                uint next = (uint)(Offset + (ulong)i);
                x = next == 0 ? PointAt(v, _shift[d], 0) : x ^ v[TrailingZeros(next)];
                emit(at + i, x);
            }
        }
        Offset += (ulong)points;
    }

    /// <summary>The next points as raw 32-bit values, in the layout the other methods use.</summary>
    internal uint[] NextRaw(int length)
    {
        var raw = new uint[length];
        Generate(PointCount(length), (i, x) => raw[i] = x);
        return raw;
    }

    /// <inheritdoc/>
    public void Uniform(Tensor<float> output)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        var values = NextRaw(output.Length);
        using var lease = output.LeaseWritable();
        var span = lease.Span;
        for (int i = 0; i < span.Length; i++) span[i] = values[i] * Inv32Float + (Inv32Float / 2.0f);
    }

    /// <inheritdoc/>
    public void Uniform(Tensor<double> output)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        var values = NextRaw(output.Length);
        using var lease = output.LeaseWritable();
        var span = lease.Span;
        for (int i = 0; i < span.Length; i++) span[i] = values[i] * Inv32Double + Inv32Double;
    }

    // The inverse normal CDF of the point, mirrored about one half so the tail keeps its precision.
    private static double InverseNormal(uint x)
    {
        bool mirrored = x > 0x80000000u;
        if (mirrored) x = 0xFFFFFFFFu - x;
        double p = x * Inv32Double + (Inv32Double / 2.0);      // (0, 0.5]
        double z = SpecialFunctions.Ndtri(p);                   // <= 0
        return mirrored ? -z : z;
    }

    /// <inheritdoc/>
    /// <remarks>The inverse normal CDF of each point (not Box-Muller, which would destroy the sequence's structure).</remarks>
    public void Normal(Tensor<float> output, float mean = 0f, float stddev = 1f)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        var values = NextRaw(output.Length);
        using var lease = output.LeaseWritable();
        var span = lease.Span;
        for (int i = 0; i < span.Length; i++) span[i] = (float)InverseNormal(values[i]) * stddev + mean;
    }

    /// <inheritdoc/>
    /// <remarks>See the float overload.</remarks>
    public void Normal(Tensor<double> output, double mean = 0, double stddev = 1)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        var values = NextRaw(output.Length);
        using var lease = output.LeaseWritable();
        var span = lease.Span;
        for (int i = 0; i < span.Length; i++) span[i] = InverseNormal(values[i]) * stddev + mean;
    }

    /// <inheritdoc/>
    /// <remarks>1 where the point's uniform value is below <paramref name="p"/>.</remarks>
    public void Bernoulli(Tensor<float> output, float p)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (float.IsNaN(p) || p < 0f || p > 1f)
            throw new ArgumentOutOfRangeException(nameof(p), $"Bernoulli probability must be in [0, 1]; got {p}.");
        var values = NextRaw(output.Length);
        using var lease = output.LeaseWritable();
        var span = lease.Span;
        for (int i = 0; i < span.Length; i++) span[i] = values[i] * Inv32Float + (Inv32Float / 2.0f) < p ? 1f : 0f;
    }
}
