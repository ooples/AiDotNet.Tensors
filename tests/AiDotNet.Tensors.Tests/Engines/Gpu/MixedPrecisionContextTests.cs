using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #197 sub-feature 5 — <see cref="MixedPrecisionContext{T}"/>.
/// </summary>
public class MixedPrecisionContextTests
{
    [Fact]
    public void DefaultConstruction_UsesForFP16Policy()
    {
        using var ctx = new MixedPrecisionContext<float>();
        Assert.NotNull(ctx.Policy);
        Assert.Equal(PrecisionMode.Float16, ctx.DefaultPrecision);
        // Sanity: ForFP16 default keeps norm layers in FP32.
        Assert.Equal(PrecisionMode.Float32, ctx.Policy.GetLayerPrecision("layernorm.weight"));
    }

    [Fact]
    public void Constructor_AcceptsCustomPolicy()
    {
        var custom = new LayerPrecisionPolicy(PrecisionMode.BFloat16).KeepInFP32("critic");
        using var ctx = new MixedPrecisionContext<float>(custom, defaultPrecision: PrecisionMode.BFloat16);
        Assert.Same(custom, ctx.Policy);
        Assert.Equal(PrecisionMode.BFloat16, ctx.DefaultPrecision);
    }

    [Fact]
    public void BeginAutocast_WithBFloat16_ThrowsNotSupported()
    {
        // MixedPrecisionContext accepts BFloat16 at construction for
        // policy/routing purposes, but the FP16 compute path in
        // AutocastScope doesn't yet handle BFloat16 — BeginAutocast must
        // throw rather than silently degrade. This test pins that
        // contract so a future implementation change doesn't accidentally
        // promote BFloat16 to an unsupported runtime path.
        using var ctx = new MixedPrecisionContext<float>(
            defaultPrecision: PrecisionMode.BFloat16);
        Assert.Throws<NotSupportedException>(() => ctx.BeginAutocast());
    }

    [Fact]
    public void BeginAutocast_ReturnsActiveScope_WithPolicyAttached()
    {
        var policy = LayerPrecisionPolicy.ForFP16();
        using var ctx = new MixedPrecisionContext<float>(policy);
        using (var scope = ctx.BeginAutocast())
        {
            Assert.True(AutocastScope.IsEnabled);
            Assert.Same(policy, scope.Policy);
            Assert.Equal(PrecisionMode.Float16, AutocastScope.ActivePrecision);
        }
        // Scope disposed → no autocast active.
        Assert.False(AutocastScope.IsEnabled);
    }

    [Fact]
    public void Scaler_IsSharedAcrossAutocastScopes()
    {
        using var ctx = new MixedPrecisionContext<float>(initialLossScale: 42.0);
        Assert.Equal(42.0, ctx.Scaler.Scale, 3);
        using (ctx.BeginAutocast()) { /* nothing */ }
        // Scaler state is preserved across scope lifecycle.
        Assert.Equal(42.0, ctx.Scaler.Scale, 3);
    }

    [Fact]
    public void BeginAutocast_AfterDispose_Throws()
    {
        var ctx = new MixedPrecisionContext<float>();
        ctx.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ctx.BeginAutocast());
    }
}
