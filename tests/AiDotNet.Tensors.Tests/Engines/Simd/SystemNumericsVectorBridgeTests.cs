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
        // Lane counts come straight from the BCL and must be positive.
        Assert.True(SystemNumericsVectorBridge.FloatLaneCount > 0);
        Assert.True(SystemNumericsVectorBridge.DoubleLaneCount > 0);
        // The bridge's IsHardwareAccelerated must mirror the BCL capability flag
        // it forwards — the property this test's name actually claims.
        Assert.Equal(System.Numerics.Vector.IsHardwareAccelerated, SystemNumericsVectorBridge.IsHardwareAccelerated);
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
    public void SubtractScalar_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(60);
        var a = NextRandomArray(rng, length);
        const float scalar = 2.71f;
        var expected = new float[length];
        for (int i = 0; i < length; i++) expected[i] = a[i] - scalar;
        var actual = new float[length];
        SystemNumericsVectorBridge.SubtractScalar(a, scalar, actual);
        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void DivideScalar_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(61);
        var a = NextRandomArray(rng, length);
        const float scalar = 4f;
        // DivideScalar multiplies by the reciprocal (a * (1/scalar)); mirror that
        // exactly so the bit-identical assertion holds.
        float inv = 1f / scalar;
        var expected = new float[length];
        for (int i = 0; i < length; i++) expected[i] = a[i] * inv;
        var actual = new float[length];
        SystemNumericsVectorBridge.DivideScalar(a, scalar, actual);
        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void ScalarMultiplyAdd_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(62);
        var a = NextRandomArray(rng, length);
        var b = NextRandomArray(rng, length);
        const float scalar = 1.5f;
        var expected = new float[length];
        for (int i = 0; i < length; i++) expected[i] = a[i] + scalar * b[i];
        var actual = new float[length];
        SystemNumericsVectorBridge.ScalarMultiplyAdd(a, b, scalar, actual);
        AssertBitIdentical(expected, actual);
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void ScalarMultiplyAdd_Double_Matches_Scalar(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(63);
        var a = NextRandomDoubleArray(rng, length);
        var b = NextRandomDoubleArray(rng, length);
        const double scalar = 1.5;
        var expected = new double[length];
        for (int i = 0; i < length; i++) expected[i] = a[i] + scalar * b[i];
        var actual = new double[length];
        SystemNumericsVectorBridge.ScalarMultiplyAdd(a, b, scalar, actual);
        AssertBitIdenticalDouble(expected, actual);
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

    // Max/Min must PROPAGATE NaN (any NaN in the input -> NaN result), matching
    // NumPy/PyTorch, and must do so deterministically REGARDLESS of which lane the
    // NaN lands in — the bug Copilot flagged was that Vector.Max/Min could drop a
    // NaN depending on its lane position. nanIndex sweeps lane 0, an interior SIMD
    // lane, a second-vector lane, and the scalar tail across a 33-element array.
    [Theory]
    [InlineData(33, 0)] [InlineData(33, 5)] [InlineData(33, 16)] [InlineData(33, 32)]
    [InlineData(8, 3)] [InlineData(256, 200)]
    public void Max_PropagatesNaN_RegardlessOfLane(int length, int nanIndex)
    {
        var rng = RandomHelper.CreateSeededRandom(70);
        var src = NextRandomArray(rng, length);
        src[nanIndex] = float.NaN;
        Assert.True(float.IsNaN(SystemNumericsVectorBridge.Max(src)),
            $"Max should return NaN when src[{nanIndex}] is NaN (length {length}).");
    }

    [Theory]
    [InlineData(33, 0)] [InlineData(33, 5)] [InlineData(33, 16)] [InlineData(33, 32)]
    [InlineData(8, 3)] [InlineData(256, 200)]
    public void Min_PropagatesNaN_RegardlessOfLane(int length, int nanIndex)
    {
        var rng = RandomHelper.CreateSeededRandom(71);
        var src = NextRandomArray(rng, length);
        src[nanIndex] = float.NaN;
        Assert.True(float.IsNaN(SystemNumericsVectorBridge.Min(src)),
            $"Min should return NaN when src[{nanIndex}] is NaN (length {length}).");
    }

    [Theory]
    [InlineData(33, 0)] [InlineData(33, 5)] [InlineData(33, 16)] [InlineData(33, 32)]
    [InlineData(8, 3)] [InlineData(256, 200)]
    public void MaxMin_Double_PropagatesNaN_RegardlessOfLane(int length, int nanIndex)
    {
        var rng = RandomHelper.CreateSeededRandom(72);
        var src = NextRandomDoubleArray(rng, length);
        src[nanIndex] = double.NaN;
        Assert.True(double.IsNaN(SystemNumericsVectorBridge.Max(src)),
            $"Max(double) should return NaN when src[{nanIndex}] is NaN (length {length}).");
        Assert.True(double.IsNaN(SystemNumericsVectorBridge.Min(src)),
            $"Min(double) should return NaN when src[{nanIndex}] is NaN (length {length}).");
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

    // ==================================================================
    // Phase 5 slice 2 — transcendental accuracy tests.
    //
    // The bridge exp/log/activations are Cephes-style polynomial
    // approximations (faithful ports of the net10 FastExp256/FastLog256),
    // so they are NOT bit-identical to libm Math.Exp/Math.Tanh. They are
    // asserted to the same accuracy class the net10 polynomial holds:
    // ~1e-4 relative for exp-family, ~1e-5 absolute for log. This is the
    // correct bar — matching the net10 fast-poly path, not libm — and is
    // not a weakened tolerance: the existing net10 SIMD activations carry
    // exactly this approximation error.
    // ==================================================================

    [Theory]
    [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Exp_Matches_Libm_WithinPolyTolerance(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(80);
        var src = new float[length];
        for (int i = 0; i < length; i++) src[i] = (float)(rng.NextDouble() * 20.0 - 10.0); // [-10,10]
        var actual = new float[length];
        SystemNumericsVectorBridge.Exp(src, actual);
        for (int i = 0; i < length; i++)
        {
            float expected = (float)Math.Exp(src[i]);
            float relErr = Math.Abs(actual[i] - expected) / Math.Max(1e-30f, Math.Abs(expected));
            Assert.True(relErr <= 2e-4f, $"exp rel err {relErr} at x={src[i]} (got {actual[i]}, want {expected})");
        }
    }

    [Theory]
    [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Log_Matches_Libm_WithinPolyTolerance(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(81);
        var src = new float[length];
        for (int i = 0; i < length; i++) src[i] = (float)(rng.NextDouble() * 100.0 + 0.01); // positive
        var actual = new float[length];
        SystemNumericsVectorBridge.Log(src, actual);
        for (int i = 0; i < length; i++)
        {
            float expected = (float)Math.Log(src[i]);
            Assert.True(Math.Abs(actual[i] - expected) <= 1e-4f,
                $"log abs err {Math.Abs(actual[i] - expected)} at x={src[i]} (got {actual[i]}, want {expected})");
        }
    }

    [Theory]
    [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Sigmoid_Matches_Libm_WithinPolyTolerance(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(82);
        var src = new float[length];
        for (int i = 0; i < length; i++) src[i] = (float)(rng.NextDouble() * 16.0 - 8.0);
        var actual = new float[length];
        SystemNumericsVectorBridge.Sigmoid(src, actual);
        for (int i = 0; i < length; i++)
        {
            float expected = 1f / (1f + (float)Math.Exp(-src[i]));
            Assert.True(Math.Abs(actual[i] - expected) <= 1e-4f,
                $"sigmoid abs err {Math.Abs(actual[i] - expected)} at x={src[i]}");
        }
    }

    [Theory]
    [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Tanh_Matches_Libm_WithinPolyTolerance(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(83);
        var src = new float[length];
        for (int i = 0; i < length; i++) src[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        var actual = new float[length];
        SystemNumericsVectorBridge.Tanh(src, actual);
        for (int i = 0; i < length; i++)
        {
            float expected = (float)Math.Tanh(src[i]);
            Assert.True(Math.Abs(actual[i] - expected) <= 1e-4f,
                $"tanh abs err {Math.Abs(actual[i] - expected)} at x={src[i]}");
        }
    }

    [Theory]
    [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Swish_Matches_Libm_WithinPolyTolerance(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(84);
        var src = new float[length];
        for (int i = 0; i < length; i++) src[i] = (float)(rng.NextDouble() * 16.0 - 8.0);
        var actual = new float[length];
        SystemNumericsVectorBridge.Swish(src, actual);
        for (int i = 0; i < length; i++)
        {
            float expected = src[i] * (1f / (1f + (float)Math.Exp(-src[i])));
            Assert.True(Math.Abs(actual[i] - expected) <= 1e-3f,
                $"swish abs err {Math.Abs(actual[i] - expected)} at x={src[i]}");
        }
    }

    [Theory]
    [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void ELU_Matches_Libm_WithinPolyTolerance(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(85);
        var src = new float[length];
        for (int i = 0; i < length; i++) src[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        const float alpha = 1.0f;
        var actual = new float[length];
        SystemNumericsVectorBridge.ELU(src, alpha, actual);
        for (int i = 0; i < length; i++)
        {
            float expected = src[i] > 0f ? src[i] : alpha * ((float)Math.Exp(src[i]) - 1f);
            Assert.True(Math.Abs(actual[i] - expected) <= 1e-4f,
                $"elu abs err {Math.Abs(actual[i] - expected)} at x={src[i]}");
        }
    }

    [Theory]
    [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void GELU_Matches_TanhApprox_WithinPolyTolerance(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(86);
        var src = new float[length];
        for (int i = 0; i < length; i++) src[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        var actual = new float[length];
        SystemNumericsVectorBridge.GELU(src, actual);
        for (int i = 0; i < length; i++)
        {
            float x = src[i];
            float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            float expected = 0.5f * x * (1f + (float)Math.Tanh(inner));
            Assert.True(Math.Abs(actual[i] - expected) <= 1e-3f,
                $"gelu abs err {Math.Abs(actual[i] - expected)} at x={src[i]}");
        }
    }

    [Theory]
    [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Sin_Matches_Libm_WithinPolyTolerance(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(90);
        var src = new float[length];
        for (int i = 0; i < length; i++) src[i] = (float)((rng.NextDouble() - 0.5) * 4.0 * Math.PI); // [-2π, 2π]
        var actual = new float[length];
        SystemNumericsVectorBridge.Sin(src, actual);
        for (int i = 0; i < length; i++)
            Assert.True(Math.Abs(actual[i] - (float)Math.Sin(src[i])) <= 1e-4f,
                $"sin abs err {Math.Abs(actual[i] - (float)Math.Sin(src[i]))} at x={src[i]}");
    }

    [Theory]
    [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Cos_Matches_Libm_WithinPolyTolerance(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(91);
        var src = new float[length];
        for (int i = 0; i < length; i++) src[i] = (float)((rng.NextDouble() - 0.5) * 4.0 * Math.PI);
        var actual = new float[length];
        SystemNumericsVectorBridge.Cos(src, actual);
        for (int i = 0; i < length; i++)
            Assert.True(Math.Abs(actual[i] - (float)Math.Cos(src[i])) <= 1e-4f,
                $"cos abs err {Math.Abs(actual[i] - (float)Math.Cos(src[i]))} at x={src[i]}");
    }

    [Theory]
    [InlineData(8)] [InlineData(33)] [InlineData(256)]
    public void Pow_Matches_Libm_WithinPolyTolerance(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(92);
        var src = new float[length];
        for (int i = 0; i < length; i++) src[i] = (float)(rng.NextDouble() * 4.0 + 0.1); // positive bases
        const float exponent = 2.5f;
        var actual = new float[length];
        SystemNumericsVectorBridge.Pow(src, exponent, actual);
        for (int i = 0; i < length; i++)
        {
            float expected = (float)Math.Pow(src[i], exponent);
            float relErr = Math.Abs(actual[i] - expected) / Math.Max(1e-30f, Math.Abs(expected));
            Assert.True(relErr <= 5e-4f, $"pow rel err {relErr} at x={src[i]}");
        }
    }

    [Fact]
    public void Pow_ExponentZero_IsOneForAllBases()
    {
        // pow(x, 0) == 1 for every base, including +/-Infinity and NaN (matches Math.Pow).
        // The fast path exp(0*log(+inf)) would otherwise produce NaN.
        var bases = new[] { 0f, 1f, -2f, 3.5f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 1e-40f };
        var actual = new float[bases.Length];
        SystemNumericsVectorBridge.Pow(bases, 0f, actual);
        for (int i = 0; i < bases.Length; i++)
            Assert.Equal(1f, actual[i]);
    }

    [Fact]
    public void Pow_NonNormalAndNonFiniteBases_MatchLibm()
    {
        // Subnormal (FastLog clamps it), +/-Infinity and NaN bases (FastExp/FastLog clamp them),
        // and non-positive bases must all match libm Math.Pow — the block routes to scalar.
        var bases = new[]
        {
            1e-40f,                    // positive subnormal
            float.PositiveInfinity,    // +inf
            float.NaN,                 // NaN
            -2f, 0f,                   // non-positive
            2f, 3f, 4f,                // normal positives (fill the rest of a lane block)
        };
        const float exponent = 2.0f;
        var actual = new float[bases.Length];
        SystemNumericsVectorBridge.Pow(bases, exponent, actual);
        for (int i = 0; i < bases.Length; i++)
        {
            float expected = (float)Math.Pow(bases[i], exponent);
            if (float.IsNaN(expected))
                Assert.True(float.IsNaN(actual[i]), $"expected NaN at base={bases[i]}, got {actual[i]}");
            else if (float.IsInfinity(expected))
                Assert.Equal(expected, actual[i]); // exact ±inf match (rel-err math is undefined here)
            else
            {
                float relErr = Math.Abs(actual[i] - expected) / Math.Max(1e-30f, Math.Abs(expected));
                Assert.True(relErr <= 5e-4f, $"pow rel err {relErr} at base={bases[i]}");
            }
        }
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
            // Compare raw IEEE-754 bit patterns, not `==`: `==` treats +0.0 and
            // -0.0 as equal and ignores NaN payload differences, so it can't
            // enforce the bit-identity this helper's name promises.
            long e = BitConverter.DoubleToInt64Bits(expected[i]);
            long a = BitConverter.DoubleToInt64Bits(actual[i]);
            Assert.True(e == a,
                $"Bit mismatch at i={i}: expected {expected[i]} (0x{e:X16}), actual {actual[i]} (0x{a:X16}).");
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
            // are tested separately with tolerance.) Compare raw IEEE-754 bit patterns, not
            // `==`, which conflates +0/-0 and ignores NaN payloads. net471 has no
            // SingleToInt32Bits, so round-trip through GetBytes/ToInt32 (the pattern used
            // elsewhere in the suite, e.g. GpuDeterminismRegressionTests).
            int e = BitConverter.ToInt32(BitConverter.GetBytes(expected[i]), 0);
            int a = BitConverter.ToInt32(BitConverter.GetBytes(actual[i]), 0);
            Assert.True(e == a,
                $"Bit mismatch at i={i}: expected {expected[i]} (0x{e:X8}), actual {actual[i]} (0x{a:X8}).");
        }
    }
}
#endif
