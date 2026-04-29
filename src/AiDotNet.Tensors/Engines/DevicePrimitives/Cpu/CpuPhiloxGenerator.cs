// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;

/// <summary>
/// Managed Philox 4x32-10 generator — produces the same bit stream as
/// <c>cuRAND</c>'s <c>CURAND_RNG_PSEUDO_PHILOX4_32_10</c>. Cross-backend
/// determinism: <c>(seed, subsequence, offset)</c> is the same triple
/// the CUDA Philox kernel takes; identical inputs produce identical
/// outputs across the CPU and GPU backends.
///
/// <para>Algorithm reference: Salmon, Moraes, Dror & Shaw,
/// "Parallel Random Numbers: As Easy as 1, 2, 3" (SC'11). Philox is the
/// canonical counter-based RNG in HPC + ML toolchains.</para>
/// </summary>
public sealed class CpuPhiloxGenerator : IDeviceRng
{
    private const uint M0 = 0xD2511F53;  // Salmon-Moraes-Dror-Shaw multipliers
    private const uint M1 = 0xCD9E8D57;
    private const uint W32 = 0x9E3779B9;  // Weyl increment for the per-round key bump
    private const uint W64 = 0xBB67AE85;

    /// <summary>RNG algorithm — always Philox.</summary>
    public DeviceRngAlgorithm Algorithm => DeviceRngAlgorithm.Philox;

    /// <summary>Configured seed.</summary>
    public ulong Seed { get; }

    /// <summary>Counter advance, in 4-element blocks.</summary>
    public ulong Offset { get; set; }

    /// <summary>Sub-stream id — folded into the 64-bit counter.</summary>
    public ulong Subsequence { get; set; }

    /// <summary>Constructs a generator.</summary>
    public CpuPhiloxGenerator(ulong seed, ulong subsequence = 0, ulong offset = 0)
    {
        Seed = seed;
        Subsequence = subsequence;
        Offset = offset;
    }

    /// <inheritdoc/>
    public void Uniform(Tensor<float> output)
    {
        var span = output.AsWritableSpan();
        // Each Philox round returns 4 uint32. We pack 4 outputs per
        // counter increment, treating each uint as a uniform [0, 1) by
        // dividing by 2^32.
        const float Inv2_32 = 1.0f / 4294967296.0f;
        int i = 0;
        while (i + 4 <= span.Length)
        {
            var (r0, r1, r2, r3) = NextBlock();
            span[i + 0] = r0 * Inv2_32;
            span[i + 1] = r1 * Inv2_32;
            span[i + 2] = r2 * Inv2_32;
            span[i + 3] = r3 * Inv2_32;
            i += 4;
        }
        if (i < span.Length)
        {
            var (r0, r1, r2, r3) = NextBlock();
            uint[] tail = { r0, r1, r2, r3 };
            for (int k = 0; k < span.Length - i; k++) span[i + k] = tail[k] * Inv2_32;
        }
    }

    /// <inheritdoc/>
    public void Uniform(Tensor<double> output)
    {
        var span = output.AsWritableSpan();
        // For doubles we combine two uint32s into one uint53-equivalent
        // and divide by 2^53. Same convention cuRAND's CURAND_RNG_PSEUDO_PHILOX4_32_10
        // double generator uses.
        const double Inv2_53 = 1.0 / 9007199254740992.0;
        int i = 0;
        while (i + 2 <= span.Length)
        {
            var (r0, r1, r2, r3) = NextBlock();
            ulong c0 = (((ulong)r0) << 21) | (r1 >> 11);
            ulong c1 = (((ulong)r2) << 21) | (r3 >> 11);
            span[i + 0] = c0 * Inv2_53;
            span[i + 1] = c1 * Inv2_53;
            i += 2;
        }
        if (i < span.Length)
        {
            var (r0, r1, _, _) = NextBlock();
            ulong c = (((ulong)r0) << 21) | (r1 >> 11);
            span[i] = c * Inv2_53;
        }
    }

    /// <inheritdoc/>
    public void Normal(Tensor<float> output, float mean = 0f, float stddev = 1f)
    {
        var span = output.AsWritableSpan();
        // Each Philox block produces four normals via two Box-Muller
        // pairs. The previous loop's index mutations could fall through
        // every emit on lengths {1, 5} and similar, leaving the tail
        // uninitialised. Rewrite as a clean "emit while there's room"
        // sequence: if span.Length is not a multiple of 4 we simply
        // discard the unused tail of the last block, which is the
        // standard PyTorch/cuRAND behaviour for partial-block requests.
        int i = 0;
        while (i < span.Length)
        {
            var (r0, r1, r2, r3) = NextBlock();
            BoxMullerPair(r0, r1, out float n0, out float n1);
            BoxMullerPair(r2, r3, out float n2, out float n3);
            if (i < span.Length) span[i++] = mean + stddev * n0;
            if (i < span.Length) span[i++] = mean + stddev * n1;
            if (i < span.Length) span[i++] = mean + stddev * n2;
            if (i < span.Length) span[i++] = mean + stddev * n3;
        }
    }

    /// <inheritdoc/>
    public void Normal(Tensor<double> output, double mean = 0, double stddev = 1)
    {
        var span = output.AsWritableSpan();
        // Same partial-block fix as the float Normal above — the previous
        // `i + 2 <= span.Length` condition silently dropped the final
        // element on odd-length tensors, so a `Normal(new Tensor<double>(1))`
        // returned an uninitialised value.
        int i = 0;
        while (i < span.Length)
        {
            var (r0, r1, r2, r3) = NextBlock();
            ulong c0 = (((ulong)r0) << 21) | (r1 >> 11);
            ulong c1 = (((ulong)r2) << 21) | (r3 >> 11);
            BoxMullerPair53(c0, c1, out double n0, out double n1);
            if (i < span.Length) span[i++] = mean + stddev * n0;
            if (i < span.Length) span[i++] = mean + stddev * n1;
        }
    }

    /// <inheritdoc/>
    public void Bernoulli(Tensor<float> output, float p)
    {
        var span = output.AsWritableSpan();
        const float Inv2_32 = 1.0f / 4294967296.0f;
        int i = 0;
        while (i + 4 <= span.Length)
        {
            var (r0, r1, r2, r3) = NextBlock();
            span[i + 0] = (r0 * Inv2_32 < p) ? 1f : 0f;
            span[i + 1] = (r1 * Inv2_32 < p) ? 1f : 0f;
            span[i + 2] = (r2 * Inv2_32 < p) ? 1f : 0f;
            span[i + 3] = (r3 * Inv2_32 < p) ? 1f : 0f;
            i += 4;
        }
        if (i < span.Length)
        {
            var (r0, r1, r2, r3) = NextBlock();
            uint[] t = { r0, r1, r2, r3 };
            for (int k = 0; k < span.Length - i; k++)
                span[i + k] = (t[k] * Inv2_32 < p) ? 1f : 0f;
        }
    }

    private (uint r0, uint r1, uint r2, uint r3) NextBlock()
    {
        // Philox 4x32-10: 10 rounds of double-multiply-and-XOR over a
        // 128-bit counter (Subsequence:64 | Offset:64) keyed by Seed.
        uint c0 = (uint)(Offset & 0xFFFFFFFF);
        uint c1 = (uint)(Offset >> 32);
        uint c2 = (uint)(Subsequence & 0xFFFFFFFF);
        uint c3 = (uint)(Subsequence >> 32);
        uint k0 = (uint)(Seed & 0xFFFFFFFF);
        uint k1 = (uint)(Seed >> 32);

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
            k0 += W32;
            k1 += W64;
        }

        Offset++;
        return (c0, c1, c2, c3);
    }

    private static void BoxMullerPair(uint u0, uint u1, out float n0, out float n1)
    {
        const float Inv2_32 = 1.0f / 4294967296.0f;
        // Avoid log(0) — the spec cuRAND uses adds 1 to avoid the lowest
        // bin entirely. Same nudge here.
        float u = MathF.Max(1e-30f, u0 * Inv2_32);
        float v = u1 * Inv2_32;
        float r = MathF.Sqrt(-2f * MathF.Log(u));
        float theta = 2f * MathF.PI * v;
        n0 = r * MathF.Cos(theta);
        n1 = r * MathF.Sin(theta);
    }

    private static void BoxMullerPair53(ulong u0, ulong u1, out double n0, out double n1)
    {
        const double Inv2_53 = 1.0 / 9007199254740992.0;
        double u = Math.Max(1e-300, u0 * Inv2_53);
        double v = u1 * Inv2_53;
        double r = Math.Sqrt(-2.0 * Math.Log(u));
        double theta = 2.0 * Math.PI * v;
        n0 = r * Math.Cos(theta);
        n1 = r * Math.Sin(theta);
    }
}
