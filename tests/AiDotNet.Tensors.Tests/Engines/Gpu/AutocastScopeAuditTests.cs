using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #197 sub-feature 3 — the AutocastScope additions the first PR
/// cut missed: <see cref="AutocastScope.ShouldUseFP32"/>,
/// <see cref="AutocastScope.ShouldUseHigherPrecision"/>, and the static
/// <see cref="AutocastScope.CastToFP32"/> / <see cref="AutocastScope.CastToFP16"/>
/// helpers.
/// </summary>
public class AutocastScopeAuditTests
{
    [Fact]
    public void ShouldUseFP32_RoutesThroughPolicy()
    {
        var policy = LayerPrecisionPolicy.ForFP16(); // norm layers → FP32
        using var scope = new AutocastScope(PrecisionMode.Float16, policy);
        Assert.True(scope.ShouldUseFP32("layernorm"));
        Assert.False(scope.ShouldUseFP32("fc1"));
    }

    [Fact]
    public void ShouldUseFP32_NoPolicy_AlwaysFalse()
    {
        using var scope = new AutocastScope(PrecisionMode.Float16);
        Assert.False(scope.ShouldUseFP32("anything"));
    }

    [Fact]
    public void ShouldUseHigherPrecision_TrueForFP32LayerUnderFP16Scope()
    {
        var policy = LayerPrecisionPolicy.ForFP16();
        using var scope = new AutocastScope(PrecisionMode.Float16, policy);
        Assert.True(scope.ShouldUseHigherPrecision("softmax")); // FP32 > FP16
        Assert.False(scope.ShouldUseHigherPrecision("conv1"));  // FP16 == FP16
    }

    [Fact]
    public void ShouldUseHigherPrecision_FalseWhenNoPolicy()
    {
        using var scope = new AutocastScope(PrecisionMode.Float16);
        Assert.False(scope.ShouldUseHigherPrecision("anything"));
    }

    [Fact]
    public void CastToFP32_RoundTrip_PreservesValuesWithinHalfPrecision()
    {
        var src = new Tensor<float>([3]);
        src.AsWritableSpan()[0] = 1f;
        src.AsWritableSpan()[1] = -2f;
        src.AsWritableSpan()[2] = 3.25f;
        var fp16 = AutocastScope.CastToFP16(src);
        var fp32 = AutocastScope.CastToFP32(fp16);
        Assert.Equal(1f, fp32.AsSpan()[0], 2);
        Assert.Equal(-2f, fp32.AsSpan()[1], 2);
        Assert.Equal(3.25f, fp32.AsSpan()[2], 2);
    }

    [Fact]
    public void CastToFP32_NullInput_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => AutocastScope.CastToFP32(null!));
    }

    [Fact]
    public void CastToFP16_NullInput_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => AutocastScope.CastToFP16(null!));
    }
}
