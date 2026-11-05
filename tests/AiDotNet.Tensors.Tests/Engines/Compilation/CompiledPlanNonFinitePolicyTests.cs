// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The compiled plan's global-norm clip under each <see cref="NonFiniteGradientPolicy"/>, on the host and with the
/// parameters on the GPU: <c>loss = sum(w0 * c0) + sum(w1 * c1)</c>, so each gradient is exactly its <c>c</c> and a
/// NaN in <c>c0</c> is a NaN gradient by construction.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class CompiledPlanNonFinitePolicyTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public CompiledPlanNonFinitePolicyTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static readonly float[] C0 = { 3f, float.NaN, 4f, 0f, 0f, 0f, 0f, 0f };
    private static readonly float[] C1 = { 12f };

    private float[][] Step(bool onGpu, NonFiniteGradientPolicy policy, out Exception? thrown)
    {
        IEngine engine = onGpu ? _fixture.Engine! : new CpuEngine();
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var w0 = new Tensor<float>(new[] { C0.Length }, new Vector<float>(Enumerable.Repeat(1f, C0.Length).ToArray()));
            var w1 = new Tensor<float>(new[] { 1 }, new Vector<float>(new[] { 1f }));
            if (onGpu) { w0 = w0.Gpu(); w1 = w1.Gpu(); }
            var c0 = new Tensor<float>(new[] { C0.Length }, new Vector<float>(C0));
            var c1 = new Tensor<float>(new[] { 1 }, new Vector<float>(C1));
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                engine.TensorAdd(engine.ReduceSum(engine.TensorMultiply(w0, c0), null),
                                 engine.ReduceSum(engine.TensorMultiply(w1, c1), null));
                plan = scope.CompileTraining(new[] { w0, w1 });
            }
            using (plan)
            {
                plan.SetMaxGradNorm(1.0);
                ((INonFiniteGradientPolicyTarget)plan).SetNonFiniteGradientPolicy(policy);
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1.0f);
                thrown = Record.Exception(() => plan.Step());
                return new[] { w0.ToArray(), w1.ToArray() };
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    private void SkipUnlessAvailable(bool onGpu) => Skip.If(onGpu && !_fixture.IsAvailable, "No GPU device.");

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ZeroNonFinite_ZeroesTheNaN_ClipsByTheRest_AndTheUpdateRuns(bool onGpu)
    {
        SkipUnlessAvailable(onGpu);
        var w = Step(onGpu, NonFiniteGradientPolicy.ZeroNonFinite, out var thrown);
        Assert.Null(thrown);
        // Finite norm sqrt(9 + 16 + 144) = 13; SGD at lr 1 moves each parameter by -g / (13 + 1e-6).
        float c = (float)(1.0 / (13.0 + 1e-6));
        Assert.Equal(1f - 3f * c, w[0][0], 5);
        Assert.Equal(1f, w[0][1]);
        Assert.Equal(1f - 4f * c, w[0][2], 5);
        Assert.Equal(1f - 12f * c, w[1][0], 5);
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Error_ThrowsNamingTheFirstGradient_AndNoUpdateRuns(bool onGpu)
    {
        SkipUnlessAvailable(onGpu);
        var w = Step(onGpu, NonFiniteGradientPolicy.Error, out var thrown);
        var ex = Assert.IsType<NonFiniteGradientException>(thrown);
        Assert.Equal(0, ex.FirstNonFiniteGradient);
        Assert.Equal(1, ex.NanCount);
        Assert.Contains("gradient 0 is the first", ex.Message);
        Assert.Contains("1 NaN", ex.Message);
        Assert.All(w[0].Where((_, i) => i != 1), v => Assert.Equal(1f, v));
        Assert.Equal(1f, w[1][0]);
    }

    [SkippableTheory]
    [InlineData(false, NonFiniteGradientPolicy.LeaveUnscaled)]
    [InlineData(false, NonFiniteGradientPolicy.Propagate)]
    [InlineData(true, NonFiniteGradientPolicy.LeaveUnscaled)]
    [InlineData(true, NonFiniteGradientPolicy.Propagate)]
    public void LeaveUnscaledAndPropagate_TheStepIsSkipped(bool onGpu, NonFiniteGradientPolicy policy)
    {
        SkipUnlessAvailable(onGpu);
        var w = Step(onGpu, policy, out var thrown);
        Assert.Null(thrown);
        Assert.All(w[0], v => Assert.Equal(1f, v));
        Assert.Equal(1f, w[1][0]);
    }
}
