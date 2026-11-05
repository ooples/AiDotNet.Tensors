using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Asserts <see cref="PortableFp32_4x4"/> is BIT-IDENTICAL to <see cref="ScalarFp32_4x4"/>, the kernel it replaces on
/// x64 hosts without AVX2.
/// </summary>
/// <remarks>
/// <para>
/// Bit-identity, not a tolerance. The scalar kernel's tile, packing and double accumulation are shared with the M/N
/// tails and the streaming kernel on that path, and BlasMode.Deterministic promises the same bits whichever of them a
/// row lands in. A sibling that agreed only to 1e-7 would void that promise, so every assertion compares raw IEEE
/// payloads.
/// </para>
/// <para>
/// The kernel is reachable on every 128-bit SIMD host, so these run on AVX2 machines too, where production never
/// dispatches to it; the Cross-ISA nightly is what runs it inside a GEMM with AVX2 off.
/// </para>
/// </remarks>
public class PortableFp32MicrokernelBitExactTests
{
    private const int Mr = 4;
    private const int Nr = 4;

    private static float[] Filled(int count, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((rng.NextDouble() * 2.0) - 1.0) * scale;
        return values;
    }

    // BitConverter.SingleToInt32Bits is not in net462/net471.
    private static int Bits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

    private static void AssertBitIdentical(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            int e = Bits(expected[i]);
            int a = Bits(actual[i]);
            if (e != a)
            {
                Assert.Fail(
                    $"{what}: element {i} differs at the BIT level: scalar {expected[i]:R} (0x{e:X8}) vs "
                    + $"portable {actual[i]:R} (0x{a:X8}). The portable tier must be bit-identical to the "
                    + "scalar reference, not merely close.");
            }
        }
    }

    [Theory]
    [InlineData(1, 4)]
    [InlineData(2, 4)]
    [InlineData(7, 4)]
    [InlineData(64, 4)]
    [InlineData(129, 8)]      // ldc > Nr, so C rows are not adjacent
    [InlineData(512, 16)]     // a full Kc panel, where FP32 accumulation would already have drifted
    public void RunMatchesTheScalarReferenceBitForBit(int kc, int ldc)
    {
        if (!PortableFp32_4x4.IsSupported) return;   // net471 or no 128-bit SIMD: the scalar kernel runs

        var packedA = Filled(kc * Mr, seed: 11);
        var packedB = Filled(kc * Nr, seed: 12);
        // C is read-modify-write, so both kernels start from the SAME non-zero contents: starting from zero would
        // hide a mistake in how the existing value is widened.
        var initial = Filled(Mr * ldc, seed: 13);
        var expected = (float[])initial.Clone();
        var actual = (float[])initial.Clone();

        ScalarFp32_4x4.Run(packedA, packedB, expected, ldc, kc);
        PortableFp32_4x4.Run(packedA, packedB, actual, ldc, kc);

        AssertBitIdentical(expected, actual, $"Run(kc={kc}, ldc={ldc})");
    }

    [Theory]
    [InlineData(1, 4, 4)]
    [InlineData(5, 4, 4)]
    [InlineData(33, 9, 6)]     // ldb > Nr and ldc > Nr simultaneously
    [InlineData(300, 32, 16)]
    public void RunStridedBMatchesTheScalarReferenceBitForBit(int kc, int ldb, int ldc)
    {
        if (!PortableFp32_4x4.IsSupported) return;

        var packedA = Filled(kc * Mr, seed: 21);
        var b = Filled(kc * ldb, seed: 22);
        var initial = Filled(Mr * ldc, seed: 23);
        var expected = (float[])initial.Clone();
        var actual = (float[])initial.Clone();

        ScalarFp32_4x4.RunStridedB(packedA, b, ldb, expected, ldc, kc);
        PortableFp32_4x4.RunStridedB(packedA, b, ldb, actual, ldc, kc);

        AssertBitIdentical(expected, actual, $"RunStridedB(kc={kc}, ldb={ldb}, ldc={ldc})");
    }

    [Fact]
    public void WideRangesRoundAtTheWriteBackOnly()
    {
        // Products near float's range limits are exact in double, and the sum of large values of opposite sign
        // cancels there before rounding: a kernel that accumulated in float, or narrowed per step, would overflow to
        // Infinity or lose the small terms that the scalar kernel keeps. Also covers subnormal results.
        if (!PortableFp32_4x4.IsSupported) return;

        const int kc = 6;
        var packedA = new float[kc * Mr];
        var packedB = new float[kc * Nr];
        float[] aValues = { 3e38f, -3e38f, 1e-20f, 1.5f, -2.5e-30f, 7f };
        float[] bValues = { 2f, 2f, 1e-20f, 1e-30f, 4e-10f, -1e-38f };
        for (int k = 0; k < kc; k++)
        {
            for (int i = 0; i < Mr; i++) packedA[(k * Mr) + i] = aValues[k] * (i + 1);
            for (int j = 0; j < Nr; j++) packedB[(k * Nr) + j] = bValues[k] * (j + 1);
        }
        var initial = new float[Mr * Nr];
        var expected = (float[])initial.Clone();
        var actual = (float[])initial.Clone();

        ScalarFp32_4x4.Run(packedA, packedB, expected, Nr, kc);
        PortableFp32_4x4.Run(packedA, packedB, actual, Nr, kc);

        AssertBitIdentical(expected, actual, "wide ranges");
    }

    [Fact]
    public void SpecialValuesPropagateIdentically()
    {
        // The case an `a == 0` early-out would break: 0 * Infinity is NaN, not a skipped row, and -0.0 must stay
        // signed. A kernel that optimizes away zero multiplies passes random tests while diverging here.
        if (!PortableFp32_4x4.IsSupported) return;

        const int kc = 4;
        var packedA = new float[kc * Mr];
        var packedB = new float[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = 1f;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = 1f;
        packedA[0] = 0f;                           // 0 * Infinity
        packedB[1] = float.PositiveInfinity;
        packedA[Mr + 1] = -0f;                     // signed zero
        packedB[Nr + 2] = float.NaN;               // NaN propagation
        packedA[(2 * Mr) + 2] = float.NegativeInfinity;

        var initial = new float[Mr * Nr];
        for (int i = 0; i < initial.Length; i++) initial[i] = i % 2 == 0 ? 0f : -0f;
        var expected = (float[])initial.Clone();
        var actual = (float[])initial.Clone();

        ScalarFp32_4x4.Run(packedA, packedB, expected, Nr, kc);
        PortableFp32_4x4.Run(packedA, packedB, actual, Nr, kc);

        AssertBitIdentical(expected, actual, "special values");
    }
}
