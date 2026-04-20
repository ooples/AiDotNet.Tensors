using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Tests for issue #197 — extensions to <see cref="AutocastScope"/>:
/// per-name tensor cache for master-FP32 / compute-FP16 pairs, and the
/// new <see cref="LayerPrecisionPolicy"/>-accepting constructor.
/// </summary>
public class AutocastScopeCacheTests
{
    [Fact]
    public void RegisterAndCastToFP16_ProducesMatchingFP16Tensor()
    {
        using var scope = new AutocastScope(PrecisionMode.Float16);
        var fp32 = new Tensor<float>([4]);
        var s = fp32.AsWritableSpan();
        s[0] = 1.5f; s[1] = -2.0f; s[2] = 0.25f; s[3] = 100f;

        var fp16 = scope.RegisterAndCastToFP16("weight1", fp32);
        Assert.Equal(fp32._shape, fp16._shape);
        var f16 = fp16.AsSpan();
        for (int i = 0; i < 4; i++)
            Assert.Equal(s[i], (float)f16[i], 2); // Half has ~3 decimal digits
    }

    [Fact]
    public void RegisterAndCastToFP16_CachesFP16CopyForReuse()
    {
        using var scope = new AutocastScope(PrecisionMode.Float16);
        var fp32 = new Tensor<float>([2]);
        var a = scope.RegisterAndCastToFP16("w", fp32);
        var b = scope.RegisterAndCastToFP16("w", fp32);
        // Second call returns the cached instance, not a re-cast.
        Assert.Same(a, b);
    }

    [Fact]
    public void GetFP32Tensor_ReturnsRegisteredMaster()
    {
        using var scope = new AutocastScope(PrecisionMode.Float16);
        var fp32 = new Tensor<float>([3]);
        scope.RegisterAndCastToFP16("x", fp32);
        Assert.Same(fp32, scope.GetFP32Tensor("x"));
    }

    [Fact]
    public void GetFP16Tensor_ReturnsNullWhenAbsent()
    {
        using var scope = new AutocastScope(PrecisionMode.Float16);
        Assert.Null(scope.GetFP16Tensor("missing"));
    }

    [Fact]
    public void HasTensor_TracksRegistrations()
    {
        using var scope = new AutocastScope(PrecisionMode.Float16);
        Assert.False(scope.HasTensor("x"));
        scope.RegisterAndCastToFP16("x", new Tensor<float>([1]));
        Assert.True(scope.HasTensor("x"));
    }

    [Fact]
    public void Dispose_ClearsCache()
    {
        AutocastScope scope;
        using (scope = new AutocastScope(PrecisionMode.Float16))
        {
            scope.RegisterAndCastToFP16("x", new Tensor<float>([1]));
        }
        Assert.False(scope.HasTensor("x"));
    }

    [Fact]
    public void ClearTensors_Works()
    {
        using var scope = new AutocastScope(PrecisionMode.Float16);
        scope.RegisterAndCastToFP16("a", new Tensor<float>([1]));
        scope.RegisterAndCastToFP16("b", new Tensor<float>([2]));
        scope.ClearTensors();
        Assert.False(scope.HasTensor("a"));
        Assert.False(scope.HasTensor("b"));
    }

    [Fact]
    public void Constructor_WithPolicy_ExposesIt()
    {
        var policy = LayerPrecisionPolicy.ForFP16();
        using var scope = new AutocastScope(PrecisionMode.Float16, policy);
        Assert.Same(policy, scope.Policy);
    }

    [Fact]
    public void Constructor_WithoutPolicy_HasNullPolicy()
    {
        using var scope = new AutocastScope(PrecisionMode.Float16);
        Assert.Null(scope.Policy);
    }

    [Fact]
    public void RegisterAndCastToFP16_NullFp32_Throws()
    {
        using var scope = new AutocastScope(PrecisionMode.Float16);
        Assert.Throws<ArgumentNullException>(() =>
            scope.RegisterAndCastToFP16("x", null!));
    }

    [Fact]
    public void RegisterAndCastToFP16_EmptyName_Throws()
    {
        using var scope = new AutocastScope(PrecisionMode.Float16);
        Assert.Throws<ArgumentException>(() =>
            scope.RegisterAndCastToFP16("", new Tensor<float>([1])));
    }
}
