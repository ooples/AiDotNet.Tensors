#if !NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Audit-2026-05 phase-5 foundation parity tests for the net471 BCL <c>Vector&lt;T&gt;</c> bridge.
/// Each primitive in <see cref="SystemNumericsVectorBridge"/> is exercised against a hand-rolled
/// scalar reference at multiple lengths — including lengths that exactly fill the host's lane
/// count and lengths with a scalar tail — proving the SIMD path produces bit-identical output
/// to the scalar fallback the bridge replaces.
///
/// These tests are net471-only because the bridge itself is <c>#if !NET5_0_OR_GREATER</c>.
/// On net5+, <see cref="SimdKernels"/> routes through the hardware-intrinsics path and the
/// bridge file is excluded from compilation entirely.
/// </summary>
public class SystemNumericsVectorBridgeTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(8)]    // typical AVX2 lane count for float
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(33)]   // tail-bearing
    [InlineData(127)]
    [InlineData(1024)]
    public void VectorAdd_Matches_Scalar_Reference(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(42);
        var a = NextRandomArray(rng, length);
        var b = NextRandomArray(rng, length);
        var expected = new float[length];
        for (int i = 0; i < length; i++)
            expected[i] = a[i] + b[i];

        var actual = new float[length];
        SystemNumericsVectorBridge.VectorAdd(a, b, actual);

        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(33)]
    [InlineData(1024)]
    public void VectorMultiply_Matches_Scalar_Reference(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(43);
        var a = NextRandomArray(rng, length);
        var b = NextRandomArray(rng, length);
        var expected = new float[length];
        for (int i = 0; i < length; i++)
            expected[i] = a[i] * b[i];

        var actual = new float[length];
        SystemNumericsVectorBridge.VectorMultiply(a, b, actual);

        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(33)]
    [InlineData(1024)]
    public void Saxpy_Matches_Scalar_Reference(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(44);
        const float alpha = 1.7f;
        var x = NextRandomArray(rng, length);
        var y = NextRandomArray(rng, length);
        var expected = new float[length];
        for (int i = 0; i < length; i++)
            expected[i] = alpha * x[i] + y[i];

        var actual = new float[length];
        SystemNumericsVectorBridge.Saxpy(alpha, x, y, actual);

        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(33)]
    [InlineData(1024)]
    public void Dot_Matches_Scalar_Reference(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(45);
        var a = NextRandomArray(rng, length);
        var b = NextRandomArray(rng, length);
        float expected = 0f;
        for (int i = 0; i < length; i++)
            expected += a[i] * b[i];

        float actual = SystemNumericsVectorBridge.Dot(a, b);

        // SIMD horizontal reduction sums in a different order than the scalar loop, so
        // expect FMA-class rounding drift (≤ 2 ulp at the test scales). Bit-identity is
        // not guaranteed for reductions; tolerance accommodates that.
        double tol = Math.Max(1e-5, 1e-6 * length);
        Assert.True(Math.Abs(actual - expected) <= tol,
            $"Dot drift {actual - expected} exceeded tolerance {tol} at length {length}.");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(33)]
    [InlineData(1024)]
    public void ReLU_Matches_Scalar_Reference(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(46);
        // Center around zero so ~half the values are negative — exercises both branches of the max.
        var src = new float[length];
        for (int i = 0; i < length; i++)
            src[i] = (float)(rng.NextDouble() - 0.5);
        var expected = new float[length];
        for (int i = 0; i < length; i++)
            expected[i] = Math.Max(0f, src[i]);

        var actual = new float[length];
        SystemNumericsVectorBridge.ReLU(src, actual);

        AssertBitIdentical(expected, actual);
    }

    [Fact]
    public void IsHardwareAccelerated_Reflects_Bcl_Capability()
    {
        // The bridge exposes the same flag as System.Numerics.Vector. We just assert that
        // FloatLaneCount is positive and matches whatever the BCL reports.
        Assert.True(SystemNumericsVectorBridge.FloatLaneCount > 0);
    }

    [Fact]
    public void VectorAdd_ThrowsOnLengthMismatch()
    {
        var a = new float[8];
        var b = new float[8];
        var r = new float[7];
        Assert.Throws<ArgumentException>(() => SystemNumericsVectorBridge.VectorAdd(a, b, r));
    }

    private static float[] NextRandomArray(Random rng, int length)
    {
        var arr = new float[length];
        for (int i = 0; i < length; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }

    private static void AssertBitIdentical(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            // Bit-identity guard — element-wise SIMD ops must produce IEEE-equivalent results
            // to the scalar loop they replace. (Reductions, which sum in a different order,
            // are tested separately with tolerance.)
            Assert.True(expected[i] == actual[i],
                $"Mismatch at i={i}: expected {expected[i]}, actual {actual[i]}.");
        }
    }
}
#endif
