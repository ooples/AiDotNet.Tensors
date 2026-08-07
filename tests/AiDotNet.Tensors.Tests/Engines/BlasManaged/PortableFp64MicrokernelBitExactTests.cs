using System;
using System.Numerics;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Asserts <see cref="PortableFp64_4x4"/> is BIT-IDENTICAL to <see cref="ScalarFp64_4x4"/>, the scalar
/// reference the other kernels are validated against.
/// </summary>
/// <remarks>
/// <para>
/// Bit-identity, not a tolerance, and that is the whole point. The scalar 4x4 kernel is the ground truth
/// for the packed FP64 path, and BlasMode.Deterministic promises bit-exact reproducibility. A SIMD sibling
/// that agreed only to 1e-15 would quietly void that promise, so every assertion here compares raw IEEE
/// payloads via BitConverter.DoubleToInt64Bits.
/// </para>
/// <para>
/// The property holds because vectorization is along the j axis: lane j accumulates exactly
/// <c>c_ij += a_i * b_j</c> over the same k values in the same order, with no horizontal reduction to
/// reassociate anything. These tests are what would catch a future change to k-axis vectorization or to an
/// FMA intrinsic, either of which would round differently while still looking correct.
/// </para>
/// </remarks>
public class PortableFp64MicrokernelBitExactTests
{
    private const int Mr = 4;
    private const int Nr = 4;

    private static double[] Filled(int count, int seed, double scale = 1.0)
    {
        var rng = new Random(seed);
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = ((rng.NextDouble() * 2.0) - 1.0) * scale;
        return values;
    }

    private static void AssertBitIdentical(double[] expected, double[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            long e = BitConverter.DoubleToInt64Bits(expected[i]);
            long a = BitConverter.DoubleToInt64Bits(actual[i]);
            if (e != a)
            {
                Assert.Fail(
                    $"{what}: element {i} differs at the BIT level — scalar {expected[i]:R} (0x{e:X16}) vs "
                    + $"portable {actual[i]:R} (0x{a:X16}). The portable tier must be bit-identical to the "
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
    [InlineData(256, 16)]
    public void RunMatchesTheScalarReferenceBitForBit(int kc, int ldc)
    {
        if (!PortableFp64_4x4.IsSupported) return;   // no four-lane SIMD here; the scalar path is used

        var packedA = Filled(kc * Mr, seed: 11);
        var packedB = Filled(kc * Nr, seed: 12);

        // C is read-modify-write, so both kernels must start from the SAME non-zero contents — starting
        // from zero would hide a mistake in how the existing value is loaded.
        var initial = Filled(Mr * ldc, seed: 13);
        var expected = (double[])initial.Clone();
        var actual = (double[])initial.Clone();

        ScalarFp64_4x4.Run(packedA, packedB, expected, ldc, kc);
        PortableFp64_4x4.Run(packedA, packedB, actual, ldc, kc);

        AssertBitIdentical(expected, actual, $"Run(kc={kc}, ldc={ldc})");
    }

    [Theory]
    [InlineData(1, 4, 4)]
    [InlineData(5, 4, 4)]
    [InlineData(33, 9, 6)]     // ldb > Nr and ldc > Nr simultaneously
    [InlineData(128, 32, 16)]
    public void RunStridedBMatchesTheScalarReferenceBitForBit(int kc, int ldb, int ldc)
    {
        if (!PortableFp64_4x4.IsSupported) return;

        var packedA = Filled(kc * Mr, seed: 21);
        var b = Filled(kc * ldb, seed: 22);

        var initial = Filled(Mr * ldc, seed: 23);
        var expected = (double[])initial.Clone();
        var actual = (double[])initial.Clone();

        ScalarFp64_4x4.RunStridedB(packedA, b, ldb, expected, ldc, kc);
        PortableFp64_4x4.RunStridedB(packedA, b, ldb, actual, ldc, kc);

        AssertBitIdentical(expected, actual, $"RunStridedB(kc={kc}, ldb={ldb}, ldc={ldc})");
    }

    [Fact]
    public void SpecialValuesPropagateIdentically()
    {
        // The case an `a == 0` early-out would break: 0 * Infinity is NaN, not a skipped row, and -0.0
        // must stay signed. A kernel that optimizes away zero multiplies still passes ordinary random
        // tests while diverging here.
        if (!PortableFp64_4x4.IsSupported) return;

        const int kc = 4;
        var packedA = new double[kc * Mr];
        var packedB = new double[kc * Nr];

        for (int i = 0; i < packedA.Length; i++) packedA[i] = 1.0;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = 1.0;

        packedA[0] = 0.0;                          // 0 * Infinity
        packedB[1] = double.PositiveInfinity;
        packedA[Mr + 1] = -0.0;                    // signed zero
        packedB[Nr + 2] = double.NaN;              // NaN propagation
        packedA[(2 * Mr) + 2] = double.NegativeInfinity;

        var initial = new double[Mr * Nr];
        for (int i = 0; i < initial.Length; i++) initial[i] = i % 2 == 0 ? 0.0 : -0.0;

        var expected = (double[])initial.Clone();
        var actual = (double[])initial.Clone();

        ScalarFp64_4x4.Run(packedA, packedB, expected, Nr, kc);
        PortableFp64_4x4.Run(packedA, packedB, actual, Nr, kc);

        AssertBitIdentical(expected, actual, "special values");
    }

    [Fact]
    public void TheSupportGateRequiresExactlyFourLanes()
    {
        // Documents the gate rather than asserting a machine property: two lanes would need two vectors
        // per C row, eight would read past the row into the next one, so the tier declines outside four.
        bool expected = Vector.IsHardwareAccelerated && Vector<double>.Count == Nr;
        Assert.Equal(expected, PortableFp64_4x4.IsSupported);
    }
}
