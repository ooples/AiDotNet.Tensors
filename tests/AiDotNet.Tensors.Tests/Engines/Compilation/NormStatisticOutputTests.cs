using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The statistic outputs of the traced norms (RMSNorm's rms, LayerNorm's mean) hold this step's values when they are
/// the only thing a plan reads, and survive the residual fusions that replace their writer.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class NormStatisticOutputTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    [Fact]
    public void TracedRmsNorm_RmsOutputAlone_RunsTheNormalization()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var input = new Tensor<float>(new float[] { 3f, 4f }, new[] { 1, 2 });
        var gamma = new Tensor<float>(new float[] { 1f, 1f }, new[] { 2 });
        var parameter = new Tensor<float>(new float[] { 0f, 0f }, new[] { 1, 2 });

        Tensor<float> rms;
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            _ = engine.RMSNorm(engine.TensorAdd(input, parameter), gamma, 0.0, out rms);
            GradientRecording.RetainOutput(rms);
            // The loss reads only the statistic: nothing else depends on the RMSNorm node.
            engine.ReduceSum(engine.TensorAdd(rms, rms), null);
            plan = scope.CompileTraining(new[] { parameter });
        }

        using (plan)
        {
            for (int step = 0; step < 3; step++)
            {
                float scale = step + 1;
                input.AsWritableSpan()[0] = 3f * scale;
                input.AsWritableSpan()[1] = 4f * scale;
                input.IncrementVersion();
                float expected = MathF.Sqrt(12.5f) * scale;
                Assert.Equal(2f * expected, plan.Step().ToArray()[0], 4);
                Assert.Equal(expected, rms.ToArray()[0], 4);
            }
        }
    }

    [Fact]
    public void RetainedLayerNormMean_AfterResidualAdd_ReturnsTheStepsMean()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        const int rows = 2, cols = 4;
        var input = new Tensor<float>(new[] { rows, cols });
        var residual = new Tensor<float>(new[] { rows, cols });
        var gamma = new Tensor<float>(new float[] { 1, 1, 1, 1 }, new[] { cols });
        var beta = new Tensor<float>(new[] { cols });

        Tensor<float> mean;
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { gamma }))
        {
            var normed = engine.LayerNorm(engine.TensorAdd(input, residual), gamma, beta, 1e-5, out mean, out _);
            GradientRecording.RetainOutput(mean);
            engine.ReduceSum(engine.TensorMultiply(normed, normed), null);
            plan = scope.CompileTraining(new[] { gamma });
        }

        using (plan)
        {
            for (int step = 0; step < 3; step++)
            {
                var x = input.AsWritableSpan();
                for (int i = 0; i < x.Length; i++) x[i] = step + i;   // row r has mean step + 4r + 1.5
                input.IncrementVersion();
                _ = plan.Step();
                var values = mean.ToArray();
                Assert.Equal(step + 1.5f, values[0], 4);
                Assert.Equal(step + 5.5f, values[1], 4);
            }
        }
    }
}
