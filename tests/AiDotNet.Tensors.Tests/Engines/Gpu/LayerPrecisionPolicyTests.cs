using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Tests for issue #197 — <see cref="LayerPrecisionPolicy"/>. Locks in
/// per-layer precision routing, exact-match-wins-over-pattern, and the
/// default factory presets' FP32 allowlist.
/// </summary>
public class LayerPrecisionPolicyTests
{
    [Fact]
    public void SetPrecision_ExactName_IsReturned()
    {
        var p = new LayerPrecisionPolicy(PrecisionMode.Float16)
            .SetPrecision("fc.weight", PrecisionMode.Float32);
        Assert.Equal(PrecisionMode.Float32, p.GetLayerPrecision("fc.weight"));
    }

    [Fact]
    public void AddPattern_MatchesCaseInsensitiveSubstring()
    {
        var p = new LayerPrecisionPolicy(PrecisionMode.Float16)
            .AddPattern("NORM", PrecisionMode.Float32);
        // Case-insensitive substring match — "layer.norm.weight" contains "NORM".
        Assert.Equal(PrecisionMode.Float32, p.GetLayerPrecision("layer.norm.weight"));
        // "bn1" does not contain "NORM" — fallthrough to default (Float16).
        Assert.Equal(PrecisionMode.Float16, p.GetLayerPrecision("bn1"));
    }

    [Fact]
    public void AddPattern_NoMatch_FallsThroughToDefault()
    {
        var p = new LayerPrecisionPolicy(PrecisionMode.BFloat16)
            .AddPattern("norm", PrecisionMode.Float32);
        Assert.Equal(PrecisionMode.BFloat16, p.GetLayerPrecision("conv1.weight"));
    }

    [Fact]
    public void ExactMatch_WinsOverPattern()
    {
        var p = new LayerPrecisionPolicy(PrecisionMode.Float16)
            .AddPattern("norm", PrecisionMode.Float32)    // all norm layers → FP32
            .SetPrecision("layernorm.weight", PrecisionMode.Float16);  // this one in FP16
        Assert.Equal(PrecisionMode.Float16, p.GetLayerPrecision("layernorm.weight"));
        Assert.Equal(PrecisionMode.Float32, p.GetLayerPrecision("batchnorm.weight"));
    }

    [Fact]
    public void KeepInFP32_ShorthandWorks()
    {
        var p = new LayerPrecisionPolicy(PrecisionMode.Float16).KeepInFP32("softmax");
        Assert.Equal(PrecisionMode.Float32, p.GetLayerPrecision("attention.softmax"));
    }

    [Fact]
    public void ShouldSkipMixedPrecision_TrueOnlyForFP32()
    {
        var p = new LayerPrecisionPolicy(PrecisionMode.Float16)
            .KeepInFP32("loss");
        Assert.True(p.ShouldSkipMixedPrecision("ce_loss"));
        Assert.False(p.ShouldSkipMixedPrecision("fc1"));
    }

    [Fact]
    public void ForFP16_KeepsBatchNormAndSoftmaxAndLossInFP32()
    {
        var p = LayerPrecisionPolicy.ForFP16();
        Assert.Equal(PrecisionMode.Float32, p.GetLayerPrecision("batchnorm1"));
        Assert.Equal(PrecisionMode.Float32, p.GetLayerPrecision("layer_norm"));
        Assert.Equal(PrecisionMode.Float32, p.GetLayerPrecision("softmax"));
        Assert.Equal(PrecisionMode.Float32, p.GetLayerPrecision("ce_loss"));
        Assert.Equal(PrecisionMode.Float32, p.GetLayerPrecision("token_embedding"));
        // Normal compute layer gets the default FP16.
        Assert.Equal(PrecisionMode.Float16, p.GetLayerPrecision("conv1"));
        Assert.Equal(PrecisionMode.Float16, p.GetLayerPrecision("fc2"));
    }

    [Fact]
    public void ForBF16_UsesBFloatForCompute()
    {
        var p = LayerPrecisionPolicy.ForBF16();
        Assert.Equal(PrecisionMode.BFloat16, p.GetLayerPrecision("conv1"));
        Assert.Equal(PrecisionMode.Float32, p.GetLayerPrecision("layernorm"));
    }

    [Fact]
    public void ForFP8_UsesFloat8E4M3ForCompute()
    {
        var p = LayerPrecisionPolicy.ForFP8();
        Assert.Equal(PrecisionMode.Float8E4M3, p.GetLayerPrecision("conv1"));
        Assert.Equal(PrecisionMode.Float32, p.GetLayerPrecision("layernorm"));
    }

    [Fact]
    public void EmptyName_ReturnsDefault()
    {
        var p = new LayerPrecisionPolicy(PrecisionMode.Float16);
        Assert.Equal(PrecisionMode.Float16, p.GetLayerPrecision(""));
    }

    [Fact]
    public void NullLayerName_ReturnsDefault()
    {
        var p = new LayerPrecisionPolicy(PrecisionMode.Float16);
        Assert.Equal(PrecisionMode.Float16, p.GetLayerPrecision(null!));
    }

    [Fact]
    public void SetPrecision_EmptyName_Throws()
    {
        var p = new LayerPrecisionPolicy(PrecisionMode.Float16);
        Assert.Throws<ArgumentException>(() => p.SetPrecision("", PrecisionMode.Float32));
    }

    [Fact]
    public void AddPattern_EmptyPattern_Throws()
    {
        var p = new LayerPrecisionPolicy(PrecisionMode.Float16);
        Assert.Throws<ArgumentException>(() => p.AddPattern("", PrecisionMode.Float32));
    }
}
