using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #197 sub-feature 4 — generic <see cref="GradScaler{T}"/>.
/// </summary>
public class GradScalerGenericTests
{
    [Fact]
    public void ScaleLoss_MultipliesByCurrentScale()
    {
        var engine = new CpuEngine();
        var scaler = new GradScaler<float>(initialScale: 256.0);
        var loss = new Tensor<float>([1]);
        loss.AsWritableSpan()[0] = 3.5f;
        var scaled = scaler.ScaleLoss(loss, engine);
        Assert.Equal(3.5f * 256.0f, scaled.AsSpan()[0], 3);
    }

    [Fact]
    public void ScaleLoss_Scalar_MultipliesByCurrentScale()
    {
        var scaler = new GradScaler<float>(initialScale: 4.0);
        Assert.Equal(20f, scaler.ScaleLoss(5f), 5);
    }

    [Fact]
    public void UnscaleGradient_Scalar_DividesByScale()
    {
        var scaler = new GradScaler<float>(initialScale: 4.0);
        Assert.Equal(2.5f, scaler.UnscaleGradient(10f), 5);
    }

    [Fact]
    public void Unscale_Tensors_ClearsOverflowWhenFinite()
    {
        var engine = new CpuEngine();
        var scaler = new GradScaler<float>(initialScale: 2.0);
        var g = new Tensor<float>([3]);
        g.AsWritableSpan()[0] = 4f; g.AsWritableSpan()[1] = 8f; g.AsWritableSpan()[2] = 2f;
        var gradients = new[] { g };
        scaler.Unscale(gradients, engine);
        Assert.False(scaler.FoundInfOrNan);
        Assert.True(scaler.ShouldStep());
    }

    [Fact]
    public void Unscale_DetectsInf_SetsFoundInfOrNan()
    {
        var engine = new CpuEngine();
        var scaler = new GradScaler<float>(initialScale: 1.0);
        var g = new Tensor<float>([2]);
        g.AsWritableSpan()[0] = 1f;
        g.AsWritableSpan()[1] = float.PositiveInfinity;
        scaler.Unscale(new[] { g }, engine);
        Assert.True(scaler.FoundInfOrNan);
        Assert.False(scaler.ShouldStep());
    }

    [Fact]
    public void Unscale_DetectsNaN_SetsFoundInfOrNan()
    {
        var engine = new CpuEngine();
        var scaler = new GradScaler<float>(initialScale: 1.0);
        var g = new Tensor<float>([2]);
        g.AsWritableSpan()[0] = 1f;
        g.AsWritableSpan()[1] = float.NaN;
        scaler.Unscale(new[] { g }, engine);
        Assert.True(scaler.FoundInfOrNan);
    }

    [Fact]
    public void UnscaleGradientsAndCheck_ReturnsFalseOnOverflow()
    {
        var scaler = new GradScaler<float>(initialScale: 1.0);
        var g = new Tensor<float>([2]);
        g.AsWritableSpan()[0] = 1f;
        g.AsWritableSpan()[1] = float.PositiveInfinity;
        Assert.False(scaler.UnscaleGradientsAndCheck(g));
    }

    [Fact]
    public void UnscaleGradientsAndCheck_ReturnsTrueOnFinite()
    {
        var scaler = new GradScaler<float>(initialScale: 2.0);
        var g = new Tensor<float>([2]);
        g.AsWritableSpan()[0] = 4f;
        g.AsWritableSpan()[1] = 6f;
        Assert.True(scaler.UnscaleGradientsAndCheck(g));
    }

    [Fact]
    public void Update_BacksOffOnOverflow()
    {
        var scaler = new GradScaler<float>(initialScale: 100.0) { BackoffFactor = 0.25, MinScale = 1.0 };
        var g = new Tensor<float>([1]);
        g.AsWritableSpan()[0] = float.PositiveInfinity;
        scaler.UnscaleGradientsAndCheck(g);
        scaler.Update();
        Assert.Equal(25.0, scaler.Scale, 3);
    }

    [Fact]
    public void Update_GrowsAfterIntervalOfCleanSteps()
    {
        var scaler = new GradScaler<float>(initialScale: 4.0)
        {
            GrowthInterval = 3, GrowthFactor = 2.0, MaxScale = 1024.0
        };
        // Each Update with no overflow advances the streak. After 3 clean
        // steps Scale doubles.
        scaler.Update(); Assert.Equal(4.0, scaler.Scale, 3);
        scaler.Update(); Assert.Equal(4.0, scaler.Scale, 3);
        scaler.Update(); Assert.Equal(8.0, scaler.Scale, 3);
    }

    [Fact]
    public void Update_RespectsMaxScale()
    {
        var scaler = new GradScaler<float>(initialScale: 500.0)
        {
            GrowthInterval = 1, GrowthFactor = 10.0, MaxScale = 1000.0
        };
        scaler.Update(); // 500 * 10 = 5000, clamp to 1000
        Assert.Equal(1000.0, scaler.Scale, 3);
    }

    [Fact]
    public void Update_RespectsMinScale()
    {
        var scaler = new GradScaler<float>(initialScale: 10.0) { BackoffFactor = 0.01, MinScale = 5.0 };
        var g = new Tensor<float>([1]);
        g.AsWritableSpan()[0] = float.NaN;
        scaler.UnscaleGradientsAndCheck(g);
        scaler.Update(); // 10 * 0.01 = 0.1, clamp to 5.0
        Assert.Equal(5.0, scaler.Scale, 3);
    }

    [Fact]
    public void DynamicScaling_False_LeavesScaleUnchanged()
    {
        var scaler = new GradScaler<float>(initialScale: 42.0) { DynamicScaling = false };
        var g = new Tensor<float>([1]);
        g.AsWritableSpan()[0] = float.PositiveInfinity;
        scaler.UnscaleGradientsAndCheck(g);
        scaler.Update();
        Assert.Equal(42.0, scaler.Scale, 3);
    }

    [Fact]
    public void Reset_RestoresInitialScaleAndStreak()
    {
        var scaler = new GradScaler<float>(initialScale: 100.0) { BackoffFactor = 0.5 };
        var g = new Tensor<float>([1]);
        g.AsWritableSpan()[0] = float.PositiveInfinity;
        scaler.UnscaleGradientsAndCheck(g);
        scaler.Update();
        Assert.Equal(50.0, scaler.Scale, 3);
        scaler.Reset();
        Assert.Equal(100.0, scaler.Scale, 3);
        Assert.False(scaler.FoundInfOrNan);
    }

    [Fact]
    public void Reset_AcceptsNewInitialScale()
    {
        var scaler = new GradScaler<float>(initialScale: 100.0);
        scaler.Reset(newInitialScale: 500.0);
        Assert.Equal(500.0, scaler.Scale, 3);
    }

    [Fact]
    public void HasOverflow_DetectsInfAndNaN()
    {
        var scaler = new GradScaler<float>();
        Assert.True(scaler.HasOverflow(float.PositiveInfinity));
        Assert.True(scaler.HasOverflow(float.NegativeInfinity));
        Assert.True(scaler.HasOverflow(float.NaN));
        Assert.False(scaler.HasOverflow(42f));
    }

    [Fact]
    public void DetectOverflow_OnTensor_FindsFirstBadElement()
    {
        var scaler = new GradScaler<float>();
        var g = new Tensor<float>([4]);
        g.AsWritableSpan()[0] = 1f;
        g.AsWritableSpan()[1] = 2f;
        g.AsWritableSpan()[2] = float.PositiveInfinity;
        g.AsWritableSpan()[3] = 4f;
        Assert.True(scaler.DetectOverflow(g));
    }

    [Fact]
    public void Unscale_Vector_Works()
    {
        // Vector overload does all arithmetic through INumericOperations<T>
        // directly; no engine needed, so the signature takes only the
        // gradients array.
        var scaler = new GradScaler<float>(initialScale: 2.0);
        var gradients = new[] { new Vector<float>(new float[] { 4f, 8f, 2f }) };
        scaler.Unscale(gradients);
        Assert.Equal(2f, gradients[0][0], 3);
        Assert.Equal(4f, gradients[0][1], 3);
        Assert.Equal(1f, gradients[0][2], 3);
        Assert.False(scaler.FoundInfOrNan);
    }
}
