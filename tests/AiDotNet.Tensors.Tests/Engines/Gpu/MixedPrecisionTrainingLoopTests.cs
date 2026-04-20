using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class MixedPrecisionTrainingLoopTests
{
    [Fact]
    public void Step_NoOverflow_TakesStepAndUpdatesScale()
    {
        var engine = new CpuEngine();
        var ctx = new MixedPrecisionContext<float>(initialLossScale: 16f);
        var gradientSnapshot = new Tensor<float>(new[] { 1 });
        gradientSnapshot.AsWritableSpan()[0] = 0.5f;
        bool stepRan = false;

        using var loop = new MixedPrecisionTrainingLoop<float>(
            engine, ctx,
            forward: x => x,
            lossFunction: (logits, target) => logits,
            getGradients: () => new[] { gradientSnapshot },
            applyOptimizerStep: (_, _) => stepRan = true);

        var result = loop.Step(
            input: MakeScalarTensor(0.1f),
            target: MakeScalarTensor(0f),
            learningRate: 0.01f);

        Assert.True(result.StepTaken);
        Assert.True(stepRan);
    }

    [Fact]
    public void Step_OverflowInGradient_SkipsOptimizerStep()
    {
        var engine = new CpuEngine();
        var ctx = new MixedPrecisionContext<float>(initialLossScale: 16f);
        var grad = new Tensor<float>(new[] { 1 });
        grad.AsWritableSpan()[0] = float.PositiveInfinity;
        bool stepRan = false;

        using var loop = new MixedPrecisionTrainingLoop<float>(
            engine, ctx,
            forward: x => x,
            lossFunction: (logits, target) => logits,
            getGradients: () => new[] { grad },
            applyOptimizerStep: (_, _) => stepRan = true);

        var result = loop.Step(MakeScalarTensor(0.1f), MakeScalarTensor(0f), 0.01f);
        Assert.False(result.StepTaken);
        Assert.False(stepRan);
    }

    [Fact]
    public void Step_NullInput_Throws()
    {
        var engine = new CpuEngine();
        var ctx = new MixedPrecisionContext<float>();
        using var loop = new MixedPrecisionTrainingLoop<float>(
            engine, ctx, x => x, (l, t) => l,
            () => Array.Empty<Tensor<float>>(),
            (_, _) => { });
        Assert.Throws<ArgumentNullException>(() => loop.Step(null!, MakeScalarTensor(0f), 0.01f));
    }

    [Fact]
    public void Dispose_DisposesContext()
    {
        var engine = new CpuEngine();
        var ctx = new MixedPrecisionContext<float>();
        var loop = new MixedPrecisionTrainingLoop<float>(
            engine, ctx, x => x, (l, t) => l,
            () => Array.Empty<Tensor<float>>(), (_, _) => { });
        loop.Dispose();
        Assert.Throws<ObjectDisposedException>(() =>
            loop.Step(MakeScalarTensor(0f), MakeScalarTensor(0f), 0.01f));
    }

    private static Tensor<float> MakeScalarTensor(float v)
    {
        var t = new Tensor<float>(new[] { 1 });
        t.AsWritableSpan()[0] = v;
        return t;
    }
}
