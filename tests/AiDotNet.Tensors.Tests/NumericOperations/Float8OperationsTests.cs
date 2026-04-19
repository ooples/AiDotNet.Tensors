using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.NumericOperations;

/// <summary>
/// Tests that <see cref="Float8E4M3Operations"/> and
/// <see cref="Float8E5M2Operations"/> are reachable through
/// <see cref="MathHelper.GetNumericOperations{T}"/> and implement the
/// full <see cref="INumericOperations{T}"/> surface correctly.
/// </summary>
public class Float8OperationsTests
{
    [Fact]
    public void MathHelper_GetNumericOperations_ReturnsFp8Adapter()
    {
        var e4m3 = MathHelper.GetNumericOperations<Float8E4M3>();
        var e5m2 = MathHelper.GetNumericOperations<Float8E5M2>();
        Assert.IsType<Float8E4M3Operations>(e4m3);
        Assert.IsType<Float8E5M2Operations>(e5m2);
    }

    [Fact]
    public void E4M3_ScalarAdd_GoesThroughFloatAndRequantizes()
    {
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        var a = Float8E4M3.FromFloat(1.0f);
        var b = Float8E4M3.FromFloat(2.0f);
        var sum = ops.Add(a, b);
        Assert.Equal(3.0f, sum.ToFloat(), 1);
    }

    [Fact]
    public void E4M3_Multiply_Saturates_NotInf()
    {
        // E4M3 has no Inf encoding; overflow clamps to MaxFinite.
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        var big = Float8E4M3.MaxFinite;   // ≈ 448
        var prod = ops.Multiply(big, big); // 448² ≈ 200,000 → saturates
        Assert.Equal(Float8E4M3.MaxFinite.RawValue, prod.RawValue);
        Assert.False(ops.IsInfinity(prod));
    }

    [Fact]
    public void E4M3_IsInfinity_IsAlwaysFalse()
    {
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        Assert.False(ops.IsInfinity(Float8E4M3.MaxFinite));
        Assert.False(ops.IsInfinity(Float8E4M3.MinFinite));
        Assert.False(ops.IsInfinity(Float8E4M3.Zero));
    }

    [Fact]
    public void E4M3_NaN_Propagates()
    {
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        Assert.True(ops.IsNaN(Float8E4M3.NaN));
        // 1 + NaN = NaN (or at least not equal to 1)
        var r = ops.Add(Float8E4M3.FromFloat(1f), Float8E4M3.NaN);
        Assert.True(ops.IsNaN(r));
    }

    [Fact]
    public void E5M2_IsInfinity_PreservedFromEncoding()
    {
        var ops = MathHelper.GetNumericOperations<Float8E5M2>();
        Assert.True(ops.IsInfinity(Float8E5M2.PositiveInfinity));
        Assert.True(ops.IsInfinity(Float8E5M2.NegativeInfinity));
        Assert.False(ops.IsInfinity(Float8E5M2.Zero));
    }

    [Fact]
    public void VectorizedAdd_UsesFloatSimdAndRequantizes()
    {
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        var x = new Float8E4M3[] {
            Float8E4M3.FromFloat(1f), Float8E4M3.FromFloat(2f),
            Float8E4M3.FromFloat(3f), Float8E4M3.FromFloat(4f),
        };
        var y = new Float8E4M3[] {
            Float8E4M3.FromFloat(0.5f), Float8E4M3.FromFloat(0.5f),
            Float8E4M3.FromFloat(0.5f), Float8E4M3.FromFloat(0.5f),
        };
        var z = new Float8E4M3[4];
        ops.Add(x, y, z);
        // Every result equals x + 0.5 (rounded to FP8).
        Assert.Equal(1.5f, z[0].ToFloat(), 1);
        Assert.Equal(2.5f, z[1].ToFloat(), 1);
        Assert.Equal(3.5f, z[2].ToFloat(), 1);
        Assert.Equal(4.5f, z[3].ToFloat(), 1);
    }

    [Fact]
    public void VectorizedSum_UsesFloatReduction()
    {
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        var x = new Float8E4M3[] {
            Float8E4M3.FromFloat(1f), Float8E4M3.FromFloat(2f),
            Float8E4M3.FromFloat(3f), Float8E4M3.FromFloat(4f),
        };
        var sum = ops.Sum(x);
        Assert.Equal(10f, sum.ToFloat(), 0);
    }

    [Fact]
    public void VectorizedDot_Works()
    {
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        var x = new Float8E4M3[] { Float8E4M3.FromFloat(1f), Float8E4M3.FromFloat(2f), Float8E4M3.FromFloat(3f) };
        var y = new Float8E4M3[] { Float8E4M3.FromFloat(2f), Float8E4M3.FromFloat(2f), Float8E4M3.FromFloat(2f) };
        var dot = ops.Dot(x, y);
        Assert.Equal(12f, dot.ToFloat(), 0); // 1*2 + 2*2 + 3*2 = 12
    }

    [Fact]
    public void VectorizedReLU_ZerosNegative()
    {
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        var x = new Float8E4M3[] {
            Float8E4M3.FromFloat(1f), Float8E4M3.FromFloat(-2f),
            Float8E4M3.FromFloat(0f), Float8E4M3.FromFloat(3f),
        };
        var y = new Float8E4M3[4];
        ops.ReLU(x, y);
        Assert.Equal(1f, y[0].ToFloat(), 1);
        Assert.Equal(0f, y[1].ToFloat(), 1);
        Assert.Equal(0f, y[2].ToFloat(), 1);
        Assert.Equal(3f, y[3].ToFloat(), 1);
    }

    [Fact]
    public void SignOrZero_ReturnsCorrectSign()
    {
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        Assert.Equal(1f, ops.SignOrZero(Float8E4M3.FromFloat(0.5f)).ToFloat(), 1);
        Assert.Equal(-1f, ops.SignOrZero(Float8E4M3.FromFloat(-0.5f)).ToFloat(), 1);
        Assert.Equal(0f, ops.SignOrZero(Float8E4M3.Zero).ToFloat(), 1);
    }

    [Fact]
    public void FromDouble_Quantizes()
    {
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        var r = ops.FromDouble(3.14);
        Assert.InRange(r.ToFloat(), 3f, 3.3f); // rounded to nearest FP8 level
    }

    [Fact]
    public void ToFloat_ExactRoundtripForRepresentable()
    {
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        var two = Float8E4M3.FromFloat(2f);
        Assert.Equal(2f, ops.ToFloat(two), 1);
    }

    [Fact]
    public void PrecisionBits_DistinguishesFormats()
    {
        var e4m3 = MathHelper.GetNumericOperations<Float8E4M3>();
        var e5m2 = MathHelper.GetNumericOperations<Float8E5M2>();
        Assert.Equal(3, e4m3.PrecisionBits);  // E4M3: 3 mantissa bits
        Assert.Equal(2, e5m2.PrecisionBits);  // E5M2: 2 mantissa bits
    }

    [Fact]
    public void MinMaxValue_SymmetricRange()
    {
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        Assert.Equal(Float8E4M3.MaxFinite.RawValue, ops.MaxValue.RawValue);
        Assert.Equal(Float8E4M3.MinFinite.RawValue, ops.MinValue.RawValue);
    }

    [Fact]
    public void Compare_OrdersByFloatValue()
    {
        var ops = MathHelper.GetNumericOperations<Float8E4M3>();
        Assert.True(ops.Compare(Float8E4M3.FromFloat(1f), Float8E4M3.FromFloat(2f)) < 0);
        Assert.True(ops.Compare(Float8E4M3.FromFloat(2f), Float8E4M3.FromFloat(1f)) > 0);
        Assert.Equal(0, ops.Compare(Float8E4M3.Zero, Float8E4M3.Zero));
    }
}
