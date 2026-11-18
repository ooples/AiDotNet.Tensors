using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public class CompiledCpuAliasGradientPoolingTests
{
    private sealed class CustomCpuEngine : CpuEngine { }

    private sealed class Result
    {
        internal readonly List<float[]> Values = new();
        internal long GradientBackingBytes;
        internal int PooledAliasTargets;
    }

    private static TField Field<TField>(object value, string name) =>
        (TField)(value.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(value)
            ?? throw new InvalidOperationException($"Missing field {name}."));

    private static long BackingBytes(CompiledTrainingPlan<float> plan)
    {
        var distinct = new HashSet<object>();
        long bytes = 0;
        foreach (var gradient in Field<Tensor<float>[]>(plan, "_preAllocatedGrads"))
        {
            var array = gradient.GetBackingArrayForCacheLookupUnsafe();
            Assert.NotNull(array);
            if (array is not null && distinct.Add(array)) bytes += (long)array.Length * sizeof(float);
        }
        return bytes;
    }

    private static int PooledAliasTargets(CompiledTrainingPlan<float> plan)
    {
        var gradients = Field<Tensor<float>[]>(plan, "_preAllocatedGrads");
        var pooled = Field<bool[]>(plan, "_pooledGradients");
        var aliases = Field<(Tensor<float> View, Tensor<float> Target)[]>(plan, "_aliasedGradViews");
        var targets = new HashSet<Tensor<float>>(aliases.Select(alias => alias.Target));
        int count = 0;
        for (int index = 0; index < gradients.Length; ++index)
            if (pooled[index] && targets.Contains(gradients[index])) ++count;
        return count;
    }

    private static Tensor<float> Input(int seed)
    {
        var result = new Tensor<float>(new[] { 1, 4, 1, 128 });
        for (int index = 0; index < result.Length; ++index)
            result[index] = (float)Math.Sin(index * 0.017 + seed * 0.19) * 0.2f;
        return result;
    }

    // Reshape/add gradients are identity views; the nonlinear boundaries give their
    // physical targets disjoint lifetimes. The residual operand also has two writers.
    private static Tensor<float> Forward(IEngine engine, Tensor<float> input, Tensor<float> weight)
    {
        var h = input;
        for (int block = 0; block < 12; ++block)
        {
            var flat = engine.Reshape(h, new[] { 1, 4, 128 });
            var activated = engine.Tanh(flat);
            var shaped = engine.Reshape(activated, new[] { 1, 4, 1, 128 });
            var multiplied = engine.TensorMultiply(shaped, weight);
            h = engine.TensorAdd(multiplied, shaped);
        }
        return engine.ReduceSum(engine.TensorMultiply(h, h), null);
    }

    private static Result Run(bool pooling, bool inputGradients, bool customEngine = false)
    {
        string? previous = Environment.GetEnvironmentVariable("AIDOTNET_CPU_POOL_ALIAS_GRADS");
        bool? previousInput = CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
        Environment.SetEnvironmentVariable("AIDOTNET_CPU_POOL_ALIAS_GRADS", pooling ? null : "0");
        CompiledTrainingPlan<float>.CompiledInputGradientsForTests = inputGradients;
        try
        {
            IEngine engine = customEngine ? new CustomCpuEngine() : new CpuEngine();
            var input = Input(0);
            var weight = Input(7);
            // The same learned tensor appears twice, exercising tied-parameter exclusion.
            var parameters = new[] { weight, weight };
            ICompiledTrainingPlan<float> compiled;
            using (var scope = GraphMode.Enable())
            {
                Forward(engine, input, weight);
                compiled = scope.CompileTraining(parameters);
            }
            var plan = Assert.IsType<CompiledTrainingPlan<float>>(compiled);
            var result = new Result { GradientBackingBytes = BackingBytes(plan) };
            if (pooling && !customEngine) result.PooledAliasTargets = PooledAliasTargets(plan);
            Tensor<float>? escapedGradient = null;
            float[]? finalGradient = null;
            Tensor<float>? escapedInputGradient = null;
            float[]? finalInputGradient = null;
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0001f);
                for (int step = 0; step < 4; ++step)
                {
                    using var next = Input(step + 1);
                    next.AsSpan().CopyTo(input.AsWritableSpan());
                    result.Values.Add(plan.Step().ToArray());
                    Assert.NotNull(weight.Grad);
                    result.Values.Add(weight.Grad!.ToArray());
                    result.Values.Add(weight.ToArray());
                    if (inputGradients)
                    {
                        Assert.NotNull(input.Grad);
                        result.Values.Add(input.Grad!.ToArray());
                    }
                }
                escapedGradient = weight.Grad;
                finalGradient = escapedGradient!.ToArray();
                if (inputGradients)
                {
                    escapedInputGradient = input.Grad;
                    finalInputGradient = escapedInputGradient!.ToArray();
                }
            }
            // A published parameter gradient outlives the plan and unrelated operations.
            using var unrelated = engine.TensorAdd(weight, weight);
            Assert.Equal(finalGradient, escapedGradient!.ToArray());
            if (inputGradients) Assert.Equal(finalInputGradient, escapedInputGradient!.ToArray());
            return result;
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_CPU_POOL_ALIAS_GRADS", previous);
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = previousInput;
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CpuAliasTargets_ShareRealBacking_AndPreserveTrajectoryAndPublicGradients(bool inputGradients)
    {
        Result baseline = Run(pooling: false, inputGradients);
        Result pooled = Run(pooling: true, inputGradients);
        Assert.True(pooled.PooledAliasTargets > 0, "No alias target was pooled; this does not exercise the candidate.");
        Assert.True(pooled.GradientBackingBytes * 2 < baseline.GradientBackingBytes,
            $"Expected substantial backing reduction, got {pooled.GradientBackingBytes} vs {baseline.GradientBackingBytes}.");
        Assert.Equal(baseline.Values.Count, pooled.Values.Count);
        Assert.Contains(baseline.Values[1], value => value != 0f);
        for (int index = 0; index < baseline.Values.Count; ++index)
            Assert.Equal(baseline.Values[index], pooled.Values[index]);
    }

    [Fact]
    public void CustomCpuEngine_KeepsItsExistingAliasOwnership()
    {
        Result baseline = Run(pooling: false, inputGradients: false, customEngine: true);
        Result enabled = Run(pooling: true, inputGradients: false, customEngine: true);
        Assert.Equal(baseline.GradientBackingBytes, enabled.GradientBackingBytes);
        for (int index = 0; index < baseline.Values.Count; ++index)
            Assert.Equal(baseline.Values[index], enabled.Values[index]);
    }

    [Fact]
    public void DoubleMatMul_RawCapturedGradientArraysRemainValidAcrossAliasPooling()
    {
        static double[][] RunDouble(bool pooling)
        {
            string? previous = Environment.GetEnvironmentVariable("AIDOTNET_CPU_POOL_ALIAS_GRADS");
            Environment.SetEnvironmentVariable("AIDOTNET_CPU_POOL_ALIAS_GRADS", pooling ? null : "0");
            try
            {
                var engine = new CpuEngine();
                var input = new Tensor<double>(new[] { 4, 4 });
                var weight = new Tensor<double>(new[] { 4, 4 });
                for (int index = 0; index < 16; ++index)
                {
                    input[index] = Math.Sin(index * 0.13) * 0.2;
                    weight[index] = Math.Cos(index * 0.17) * 0.1;
                }
                ICompiledTrainingPlan<double> plan;
                using (var scope = GraphMode.Enable())
                {
                    var h = input;
                    for (int block = 0; block < 5; ++block)
                    {
                        var product = engine.TensorMatMul(h, weight);
                        h = engine.Reshape(engine.Tanh(engine.Reshape(product, new[] { 2, 2, 4 })), new[] { 4, 4 });
                    }
                    engine.ReduceSum(engine.TensorMultiply(h, h), null);
                    plan = scope.CompileTraining(new[] { weight });
                }
                using (plan)
                {
                    plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.001);
                    var values = new List<double[]>();
                    for (int step = 0; step < 3; ++step)
                    {
                        input[step] += 0.02;
                        values.Add(plan.Step().ToArray());
                        Assert.NotNull(weight.Grad);
                        values.Add(weight.Grad!.ToArray());
                    }
                    return values.ToArray();
                }
            }
            finally { Environment.SetEnvironmentVariable("AIDOTNET_CPU_POOL_ALIAS_GRADS", previous); }
        }
        double[][] baseline = RunDouble(pooling: false), pooled = RunDouble(pooling: true);
        Assert.Equal(baseline.Length, pooled.Length);
        Assert.Contains(baseline[1], value => value != 0.0);
        for (int index = 0; index < baseline.Length; ++index) Assert.Equal(baseline[index], pooled[index]);
    }
}
