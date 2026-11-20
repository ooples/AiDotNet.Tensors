using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class CompiledLeakyBiasStorageTests
{
    [Theory]
    [InlineData(false, 0.1f, false, true)]
    [InlineData(true, 0.1f, false, true)]
    [InlineData(true, 0f, false, true)]
    [InlineData(true, -0.5f, false, false)]
    [InlineData(true, 0.1f, true, false)]
    public void Reuse_preserves_every_step_and_respects_retained_or_nonmonotone_inputs(
        bool reshape, float slope, bool retainPreactivation, bool expectShared)
    {
        var baseline = Run(false, reshape, slope, retainPreactivation);
        var candidate = Run(true, reshape, slope, retainPreactivation);
        Assert.False(baseline.Shared);
        Assert.Equal(expectShared, candidate.Shared);
        Assert.Equal(baseline.Values, candidate.Values);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Recorded_output_reshape_keeps_its_values_and_respects_retention(bool retainResult)
    {
        var baseline = Run(false, true, .1f, false, true, retainResult);
        var candidate = Run(true, true, .1f, false, true, retainResult);
        Assert.False(baseline.Shared);
        Assert.Equal(!retainResult, candidate.Shared);
        Assert.Equal(baseline.Values, candidate.Values);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void An_unrecorded_live_view_prevents_storage_reuse(bool viewOfInput)
    {
        var baseline = Run(false, true, .1f, false, true, extraViewOfInput: viewOfInput);
        var candidate = Run(true, true, .1f, false, true, extraViewOfInput: viewOfInput);
        Assert.False(candidate.Shared);
        Assert.Equal(baseline.Values, candidate.Values);
    }

    private static (bool Shared, float[] Values) Run(bool enabled, bool reshape, float slope, bool retainPreactivation,
        bool reshapeResult = false, bool retainResult = false, bool? extraViewOfInput = null)
    {
        const string switchName = "AIDOTNET_CPU_COMPILED_LEAKY_STORAGE";
        string? previous = Environment.GetEnvironmentVariable(switchName);
        try
        {
            Environment.SetEnvironmentVariable(switchName, enabled ? "1" : "0");
            var engine = new CpuEngine();
            var input = new Tensor<float>(new[] { -2f, 0f, 1f, 3f, -4f, .25f }, new[] { 1, 2, 1, 3 });
            var bias = new Tensor<float>(new[] { 0f, -.25f }, new[] { 2 });
            var parameters = new[] { input, bias };
            Tensor<float> shifted, activated, result;
            Tensor<float>? externalView = null;
            ICompiledTrainingPlan<float> compiled;
            using (var graph = GraphMode.EnableTraining(parameters))
            {
                var biased = engine.TensorChannelBiasAdd(input, bias);
                shifted = reshape ? engine.Reshape(biased, new[] { 1, 2, 3 }) : biased;
                activated = engine.LeakyReLU(shifted, slope);
                result = reshapeResult ? engine.Reshape(activated, new[] { 3, 2 }) : activated;
                if (extraViewOfInput.HasValue)
                    externalView = (extraViewOfInput.Value ? shifted : activated).Reshape(6);
                if (retainPreactivation) graph.RetainOutput(shifted);
                if (retainResult) graph.RetainOutput(result);
                compiled = graph.CompileTraining(parameters, engine.ReduceSum(result));
            }
            using (externalView)
            using (compiled)
            {
                var plan = Assert.IsType<CompiledTrainingPlan<float>>(compiled);
                plan.DisableGraphStep();
                plan.ConfigureOptimizer(OptimizerType.Adam, .003, .9, .999, 1e-8);
                var values = new List<float>();
                for (int i = 0; i < 3; i++)
                {
                    values.AddRange(plan.Step().ToArray());
                    if (retainPreactivation) values.AddRange(shifted.ToArray());
                    if (retainResult) values.AddRange(result.ToArray());
                    if (externalView is not null) values.AddRange(externalView.ToArray());
                    foreach (var parameter in parameters)
                    {
                        values.AddRange(parameter.Grad!.ToArray());
                        values.AddRange(parameter.ToArray());
                    }
                }
                return (shifted.SharesStorageWith(activated), values.ToArray());
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(switchName, previous);
        }
    }
}
