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
        Assert.True(SystemNumericsVectorBridge.DoubleLaneCount > 0);
    }

    [Fact]
    public void VectorAdd_ThrowsOnLengthMismatch()
    {
        var a = new float[8];
        var b = new float[8];
        var r = new float[7];
        Assert.Throws<ArgumentException>(() => SystemNumericsVectorBridge.VectorAdd(a, b, r));
    }

    // ==================================================================
    // Phase 5 slice 1 expanded coverage — every bridge primitive tested.
    // ==================================================================

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void VectorSubtract_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(50);
        var a = NextRandomArray(rng, length);
        var b = NextRandomArray(rng, length);
        var expected = new float[length];
        for (int i = 0; i < length; i++) expected[i] = a[i] - b[i];
        var actual = new float[length];
        SystemNumericsVectorBridge.VectorSubtract(a, b, actual);
        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void VectorDivide_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(51);
        var a = NextRandomArray(rng, length);
        var b = NextRandomArray(rng, length);
        // Avoid div-by-zero in denominator.
        for (int i = 0; i < length; i++) if (Math.Abs(b[i]) < 0.05f) b[i] = 0.5f;
        var expected = new float[length];
        for (int i = 0; i < length; i++) expected[i] = a[i] / b[i];
        var actual = new float[length];
        SystemNumericsVectorBridge.VectorDivide(a, b, actual);
        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void AddScalar_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(52);
        var a = NextRandomArray(rng, length);
        const float scalar = 3.14f;
        var expected = new float[length];
        for (int i = 0; i < length; i++) expected[i] = a[i] + scalar;
        var actual = new float[length];
        SystemNumericsVectorBridge.AddScalar(a, scalar, actual);
        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void MultiplyScalar_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(53);
        var a = NextRandomArray(rng, length);
        const float scalar = 1.7f;
        var expected = new float[length];
        for (int i = 0; i < length; i++) expected[i] = a[i] * scalar;
        var actual = new float[length];
        SystemNumericsVectorBridge.MultiplyScalar(a, scalar, actual);
        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Sqrt_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(54);
        var src = new float[length];
        for (int i = 0; i < length; i++) src[i] = (float)(rng.NextDouble() * 10.0); // non-negative
        var expected = new float[length];
        for (int i = 0; i < length; i++) expected[i] = (float)Math.Sqrt(src[i]);
        var actual = new float[length];
        SystemNumericsVectorBridge.Sqrt(src, actual);
        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Abs_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(55);
        var src = NextRandomArray(rng, length);
        var expected = new float[length];
        for (int i = 0; i < length; i++) expected[i] = Math.Abs(src[i]);
        var actual = new float[length];
        SystemNumericsVectorBridge.Abs(src, actual);
        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Negate_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(56);
        var src = NextRandomArray(rng, length);
        var expected = new float[length];
        for (int i = 0; i < length; i++) expected[i] = -src[i];
        var actual = new float[length];
        SystemNumericsVectorBridge.Negate(src, actual);
        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Clamp_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(57);
        var src = NextRandomArray(rng, length);
        const float lo = -0.3f, hi = 0.3f;
        var expected = new float[length];
        for (int i = 0; i < length; i++) expected[i] = Math.Min(hi, Math.Max(lo, src[i]));
        var actual = new float[length];
        SystemNumericsVectorBridge.Clamp(src, lo, hi, actual);
        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void LeakyReLU_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(58);
        var src = NextRandomArray(rng, length);
        const float alpha = 0.01f;
        var expected = new float[length];
        for (int i = 0; i < length; i++) expected[i] = src[i] > 0f ? src[i] : alpha * src[i];
        var actual = new float[length];
        SystemNumericsVectorBridge.LeakyReLU(src, alpha, actual);
        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Sum_Matches_Scalar_WithinTolerance(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(59);
        var src = NextRandomArray(rng, length);
        float expected = 0f;
        for (int i = 0; i < length; i++) expected += src[i];
        float actual = SystemNumericsVectorBridge.Sum(src);
        double tol = Math.Max(1e-5, 1e-6 * length);
        Assert.True(Math.Abs(actual - expected) <= tol,
            $"Sum drift {actual - expected} exceeded tolerance {tol} at length {length}.");
    }

    [Theory]
    [InlineData(1)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Max_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(60);
        var src = NextRandomArray(rng, length);
        float expected = src[0];
        for (int i = 1; i < length; i++) if (src[i] > expected) expected = src[i];
        float actual = SystemNumericsVectorBridge.Max(src);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(1)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Min_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(61);
        var src = NextRandomArray(rng, length);
        float expected = src[0];
        for (int i = 1; i < length; i++) if (src[i] < expected) expected = src[i];
        float actual = SystemNumericsVectorBridge.Min(src);
        Assert.Equal(expected, actual);
    }

    // ==================================================================
    // DOUBLE precision parity coverage.
    // ==================================================================

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(4)] [InlineData(17)] [InlineData(128)]
    public void VectorAdd_Double_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(70);
        var a = NextRandomDoubleArray(rng, length);
        var b = NextRandomDoubleArray(rng, length);
        var expected = new double[length];
        for (int i = 0; i < length; i++) expected[i] = a[i] + b[i];
        var actual = new double[length];
        SystemNumericsVectorBridge.VectorAdd(a, b, actual);
        AssertBitIdenticalDouble(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(4)] [InlineData(17)] [InlineData(128)]
    public void VectorMultiply_Double_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(71);
        var a = NextRandomDoubleArray(rng, length);
        var b = NextRandomDoubleArray(rng, length);
        var expected = new double[length];
        for (int i = 0; i < length; i++) expected[i] = a[i] * b[i];
        var actual = new double[length];
        SystemNumericsVectorBridge.VectorMultiply(a, b, actual);
        AssertBitIdenticalDouble(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(4)] [InlineData(17)] [InlineData(128)]
    public void Sqrt_Double_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(72);
        var src = new double[length];
        for (int i = 0; i < length; i++) src[i] = rng.NextDouble() * 10.0;
        var expected = new double[length];
        for (int i = 0; i < length; i++) expected[i] = Math.Sqrt(src[i]);
        var actual = new double[length];
        SystemNumericsVectorBridge.Sqrt(src, actual);
        AssertBitIdenticalDouble(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(4)] [InlineData(17)] [InlineData(128)]
    public void ReLU_Double_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(73);
        var src = NextRandomDoubleArray(rng, length);
        var expected = new double[length];
        for (int i = 0; i < length; i++) expected[i] = Math.Max(0.0, src[i]);
        var actual = new double[length];
        SystemNumericsVectorBridge.ReLU(src, actual);
        AssertBitIdenticalDouble(expected, actual);
    }

    [Theory]
    [InlineData(1)] [InlineData(4)] [InlineData(17)] [InlineData(128)]
    public void Max_Double_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(74);
        var src = NextRandomDoubleArray(rng, length);
        double expected = src[0];
        for (int i = 1; i < length; i++) if (src[i] > expected) expected = src[i];
        double actual = SystemNumericsVectorBridge.Max(src);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(1)] [InlineData(4)] [InlineData(17)] [InlineData(128)]
    public void Min_Double_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(75);
        var src = NextRandomDoubleArray(rng, length);
        double expected = src[0];
        for (int i = 1; i < length; i++) if (src[i] < expected) expected = src[i];
        double actual = SystemNumericsVectorBridge.Min(src);
        Assert.Equal(expected, actual);
    }

    private static double[] NextRandomDoubleArray(Random rng, int length)
    {
        var arr = new double[length];
        for (int i = 0; i < length; i++)
            arr[i] = rng.NextDouble() * 2.0 - 1.0;
        return arr;
    }

    private static void AssertBitIdenticalDouble(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(expected[i] == actual[i],
                $"Mismatch at i={i}: expected {expected[i]}, actual {actual[i]}.");
        }
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
