// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.IO;

namespace AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;

/// <summary>
/// The 32-bit Sobol direction vectors of Joe and Kuo ("new-joe-kuo-6", 2008), for dimensions 0 to
/// <see cref="MaxDimensions"/> - 1. These are the vectors cuRAND returns as <c>CURAND_DIRECTION_VECTORS_32_JOEKUO6</c>.
/// </summary>
/// <remarks>
/// The published data gives, per dimension, a primitive polynomial (degree <c>s</c>, coefficient word <c>a</c>) and
/// initial numbers <c>m_1..m_s</c>. It is embedded packed (<c>scripts/pack_joe_kuo.py</c>) and expanded on first use
/// into an index of where each dimension starts; a dimension's 32 vectors are built when asked for.
/// </remarks>
internal static class SobolDirectionNumbers
{
    /// <summary>The number of dimensions available: cuRAND's limit for its Sobol generators.</summary>
    public const int MaxDimensions = 20000;

    private const string ResourceSuffix = "joe-kuo-6.bin";
    private static readonly Lazy<(byte[] Data, int[] Starts)> Packed = new(Load);

    private static (byte[], int[]) Load()
    {
        var assembly = typeof(SobolDirectionNumbers).Assembly;
        string? name = null;
        foreach (string candidate in assembly.GetManifestResourceNames())
        {
            if (candidate.EndsWith(ResourceSuffix, StringComparison.Ordinal)) { name = candidate; break; }
        }
        if (name is null) throw new InvalidOperationException($"The embedded Sobol direction numbers ({ResourceSuffix}) are missing.");

        byte[] data;
        using (Stream stream = assembly.GetManifestResourceStream(name)!)
        using (var buffer = new MemoryStream())
        {
            stream.CopyTo(buffer);
            data = buffer.ToArray();
        }
        if (data.Length < 8 || data[0] != (byte)'J' || data[1] != (byte)'K' || data[2] != (byte)'6' || data[3] != 0)
            throw new InvalidDataException("The embedded Sobol direction numbers have an unknown header.");
        int stored = BitConverter.ToInt32(data, 4);
        if (stored != MaxDimensions - 1)
            throw new InvalidDataException($"The embedded Sobol direction numbers hold {stored + 1} dimensions; {MaxDimensions} expected.");

        // starts[d] is where dimension d + 1 (the file's dimension d + 2) begins.
        var starts = new int[stored];
        int position = 8;
        for (int d = 0; d < stored; d++)
        {
            starts[d] = position;
            int s = data[position++];
            for (int field = 0; field <= s; field++) ReadVarint(data, ref position);
        }
        if (position != data.Length) throw new InvalidDataException("The embedded Sobol direction numbers have trailing bytes.");
        return (data, starts);
    }

    private static uint ReadVarint(byte[] data, ref int position)
    {
        uint value = 0;
        int shift = 0;
        while (true)
        {
            byte b = data[position++];
            value |= (uint)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return value;
            shift += 7;
        }
    }

    /// <summary>
    /// The 32 direction vectors of <paramref name="dimension"/> (zero-based): vector <c>k</c> is the contribution of
    /// bit <c>k</c> of the point index's Gray code.
    /// </summary>
    public static uint[] Vectors(int dimension)
    {
        if (dimension < 0 || dimension >= MaxDimensions)
            throw new ArgumentOutOfRangeException(nameof(dimension), dimension, $"Sobol dimensions are 0 to {MaxDimensions - 1}.");
        var v = new uint[32];
        if (dimension == 0)
        {
            for (int k = 0; k < 32; k++) v[k] = 1u << (31 - k);
            return v;
        }

        var (data, starts) = Packed.Value;
        int position = starts[dimension - 1];
        int s = data[position++];
        uint a = ReadVarint(data, ref position);
        for (int k = 0; k < s && k < 32; k++) v[k] = ReadVarint(data, ref position) << (31 - k);
        // v_k = v_{k-s} ^ (v_{k-s} >> s) ^ sum over the polynomial's middle coefficients a_j of v_{k-j}.
        for (int k = s; k < 32; k++)
        {
            uint value = v[k - s] ^ (v[k - s] >> s);
            for (int j = 1; j < s; j++)
            {
                if (((a >> (s - 1 - j)) & 1u) != 0) value ^= v[k - j];
            }
            v[k] = value;
        }
        return v;
    }
}
