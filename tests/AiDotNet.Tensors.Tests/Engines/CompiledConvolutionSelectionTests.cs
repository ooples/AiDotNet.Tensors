using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class CompiledConvolutionSelectionTests
{
    [Fact]
    public void FrozenCriticKernelKeepsTheInputGradientWithoutPublishingAnUnrequestedKernelGradient()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 1, 2, 1, 17 });
        var kernel = new Tensor<float>(new[] { 3, 2, 1, 5 });
        for (int i = 0; i < input.Length; i++) input[i] = .01f * i;
        for (int i = 0; i < kernel.Length; i++) kernel[i] = (float)Math.Sin(i * .17) * .2f;
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.Conv2D(input, kernel, new[] { 1, 1 }, new[] { 0, 2 }, new[] { 1, 1 });
            var loss = engine.ReduceSum(output);
            plan = scope.CompileTraining(new[] { input }, loss);
        }
        using (plan)
        {
            for (int replay = 0; replay < 3; replay++)
            {
                kernel[0] += .01f;
                plan.Step();
                var actual = plan.Gradients[0].ToArray();
                for (int channel = 0; channel < 2; channel++)
                    for (int t = 0; t < 17; t++)
                    {
                        double expected = 0;
                        for (int outputChannel = 0; outputChannel < 3; outputChannel++)
                            for (int tap = 0; tap < 5; tap++)
                            {
                                int outputTime = t + 2 - tap;
                                if ((uint)outputTime < 17u) expected += kernel[(outputChannel * 2 + channel) * 5 + tap];
                            }
                        Assert.InRange(Math.Abs(actual[channel * 17 + t] - expected), 0, 1e-5);
                    }
                Assert.Null(kernel.Grad);
            }
        }
    }

    [Fact]
    public void SelectedSharedKernelAccumulatesBothConvolutionUsesOnEveryReplay()
    {
        var engine = new CpuEngine();
        var first = new Tensor<double>(new[] { 1, 2, 1, 19 });
        var second = new Tensor<double>(new[] { 1, 2, 1, 19 });
        var kernel = new Tensor<double>(new[] { 3, 2, 1, 5 });
        for (int i = 0; i < first.Length; i++) { first[i] = Math.Sin(i * .13); second[i] = Math.Cos(i * .17); }
        kernel.Fill(.1);
        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var a = engine.Conv2D(first, kernel, new[] { 1, 1 }, new[] { 0, 2 }, new[] { 1, 1 });
            var b = engine.Conv2D(second, kernel, new[] { 1, 1 }, new[] { 0, 2 }, new[] { 1, 1 });
            var loss = engine.ReduceSum(engine.TensorAdd(a, engine.TensorMultiplyScalar(b, .3)));
            plan = scope.CompileTraining(new[] { kernel }, loss);
        }
        using (plan)
        {
            for (int replay = 0; replay < 3; replay++)
            {
                first[0] += .1;
                plan.Step();
                var actual = plan.Gradients[0].ToArray();
                for (int output = 0; output < 3; output++)
                    for (int channel = 0; channel < 2; channel++)
                        for (int tap = 0; tap < 5; tap++)
                        {
                            double expected = 0;
                            for (int t = 0; t < 19; t++)
                            {
                                int source = t + tap - 2;
                                if ((uint)source < 19u) expected += first[channel * 19 + source] + .3 * second[channel * 19 + source];
                            }
                            Assert.InRange(Math.Abs(actual[(output * 2 + channel) * 5 + tap] - expected), 0, 1e-9);
                        }
                Assert.Null(first.Grad);
                Assert.Null(second.Grad);
            }
        }
    }
}
