using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A fused Conv2D/bias/ReLU chain on the CPU never touches the raw convolution or the bias output, so the plan releases
/// their storage. Gradients must stay those of the unfused plan, and turning the forward diagnostics on after compiling
/// (they write every step's output) must give the released tensors storage again rather than fail.
/// </summary>
[Collection("CompilationGlobalState")]
public class FusedConvChainStorageReleaseTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static Tensor<float>[] MakeParams() => new[]
    {
        Filled(new[] { 4, 3, 3, 3 }, 31, 0.6f), Filled(new[] { 4 }, 32, 0.5f), Filled(new[] { 4, 4, 3, 3 }, 33, 0.6f), Filled(new[] { 4 }, 34, 0.5f),
    };

    private static Tensor<float> Forward(IEngine engine, Tensor<float> input, Tensor<float>[] p)
    {
        int[] one = { 1, 1 }, pad = { 1, 1 };
        var h = engine.ReLU(engine.TensorChannelBiasAdd(engine.Conv2D(input, p[0], one, pad, one), p[1]));
        var y = engine.ReLU(engine.TensorChannelBiasAdd(engine.Conv2D(h, p[2], one, pad, one), p[3]));
        return engine.ReduceSum(engine.TensorMultiply(y, y), null);
    }

    private static float[][] Run(IEngine engine, Tensor<float> input, bool fusion, bool sanitizeAfterCompile)
    {
        string? previous = Environment.GetEnvironmentVariable("AIDOTNET_CONV_EPILOGUE_FUSION");
        Environment.SetEnvironmentVariable("AIDOTNET_CONV_EPILOGUE_FUSION", fusion ? null : "0");
        try
        {
            var ps = MakeParams();
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward(engine, input, ps);
                plan = scope.CompileTraining(ps);
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                bool sanitize = CompiledTrainingPlan<float>.SanitizeForwardBuffers;
                if (sanitizeAfterCompile) CompiledTrainingPlan<float>.SanitizeForwardBuffers = true;
                try
                {
                    var result = new float[ps.Length * 2][];
                    for (int step = 0; step < 2; step++)
                    {
                        plan.Step();
                        for (int i = 0; i < ps.Length; i++)
                            result[step * ps.Length + i] = ps[i].Grad?.ToArray() ?? Array.Empty<float>();
                    }
                    return result;
                }
                finally
                {
                    CompiledTrainingPlan<float>.SanitizeForwardBuffers = sanitize;
                }
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_CONV_EPILOGUE_FUSION", previous);
        }
    }

    [Fact]
    public void FusedChainGradients_EqualUnfused_AndSurviveDiagnosticsTurnedOnLater()
    {
        var engine = new CpuEngine();
        var input = Filled(new[] { 2, 3, 6, 6 }, 3, 2f);
        var unfused = Run(engine, input, fusion: false, sanitizeAfterCompile: false);
        var fused = Run(engine, input, fusion: true, sanitizeAfterCompile: false);
        var fusedThenDiagnosed = Run(engine, input, fusion: true, sanitizeAfterCompile: true);
        for (int k = 0; k < unfused.Length; k++)
        {
            Assert.NotEmpty(unfused[k]);
            for (int j = 0; j < unfused[k].Length; j++)
            {
                Assert.True(Math.Abs(unfused[k][j] - fused[k][j]) <= 1e-4f * (1f + Math.Abs(unfused[k][j])),
                    $"[{k}][{j}]: unfused {unfused[k][j]}, fused {fused[k][j]}");
            }
            Assert.Equal(fused[k], fusedThenDiagnosed[k]);
        }
    }
}
