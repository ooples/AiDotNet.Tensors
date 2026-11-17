// REGRESSION: two compiled training plans that share a tensor - one trains it, the other reads it as a frozen leaf -
// must each see the other's latest values. This is a GAN's adversarial step: the discriminator's plan updates its
// weights on the device, then the generator's plan scores generated samples through those same weights. On CUDA the
// reading plan fed its graph the shared tensor's stale host copy, so the generator trained against an old
// discriminator (64x64 DCGAN, generator loss at step 2: 0.694 against 0.891 for the eager reference) and, by writing
// that copy back into the shared buffer, rolled the discriminator's update back too. The same held for state the
// discriminator's plan updates in place on the device (its BatchNorm running statistics, which the generator's step reads
// with the discriminator in eval mode). The CPU plans were right.

#nullable disable
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection(MixedPrecisionTestCollection.Name)]
public class SharedParameterCompiledPlansTests
{
    private const int Steps = 12;
    private readonly ITestOutputHelper _out;
    public SharedParameterCompiledPlansTests(ITestOutputHelper o) { _out = o; }

    private static float[] RandData(int n, int seed, float scale)
    {
        var rng = new Random(seed);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return data;
    }

    // Alternates the "discriminator" plan (trains shared) and the "generator" plan (trains own through frozen
    // shared), returning each step's generator loss and the final shared weights.
    private static (float[] GeneratorLosses, float[] SharedWeights) Run(IEngine engine, bool residentParams)
    {
        var xd = new Tensor<float>(RandData(16 * 32, 1, 0.5f), new[] { 16, 32 });
        var xg = new Tensor<float>(RandData(16 * 32, 2, 0.5f), new[] { 16, 32 });
        var shared = new Tensor<float>(RandData(32 * 8, 3, 0.3f), new[] { 32, 8 });
        var own = new Tensor<float>(RandData(32 * 32, 4, 0.3f), new[] { 32, 32 });
        // A running statistic the discriminator's plan updates in place each step (a BatchNorm running mean) and the
        // generator's plan reads frozen (the discriminator in eval mode).
        var running = new Tensor<float>(new float[8], new[] { 8 });
        if (residentParams) { shared.Gpu(); own.Gpu(); }

        ICompiledTrainingPlan<float> discriminator, generator;
        using (var scope = GraphMode.Enable())
        {
            var score = engine.TensorMatMul(xd, shared);
            using (new NoGradScope<float>())
            {
                engine.TensorMultiplyScalarInPlace(running, 0.9f);
                engine.TensorAddInPlace(running, engine.TensorMultiplyScalar(engine.ReduceMean(score, new[] { 0 }, false), 0.1f));
            }
            engine.ReduceSum(engine.TensorMultiply(score, score), null);
            discriminator = scope.CompileTraining(new[] { shared });
        }
        GraphMode.SetCurrent(null);
        using (var scope = GraphMode.Enable())
        {
            var sample = engine.TensorMatMul(xg, own);
            var score = engine.TensorAdd(engine.TensorMatMul(sample, shared), engine.TensorBroadcastTo(running, new[] { 16, 8 }));
            engine.ReduceSum(engine.TensorMultiply(score, score), null);
            generator = scope.CompileTraining(new[] { own });
        }
        GraphMode.SetCurrent(null);

        var losses = new float[Steps];
        using (discriminator)
        using (generator)
        {
            discriminator.ConfigureOptimizer(OptimizerType.Adam, learningRate: 5e-2f);
            generator.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2f);
            for (int s = 0; s < Steps; s++)
            {
                discriminator.Step();
                losses[s] = generator.Step().GetFlat(0);
            }
        }
        return (losses, shared.ToArray());
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void PlanReadingASharedParameter_SeesTheOtherPlansUpdate_OnGpu(bool residentParams)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");

        var (cpuLosses, cpuShared) = Run(new CpuEngine(), residentParams: false);
        var (gpuLosses, gpuShared) = Run(gpu, residentParams);

        for (int s = 0; s < Steps; s++)
        {
            _out.WriteLine($"step {s}: cpu {cpuLosses[s]:G6} gpu {gpuLosses[s]:G6}");
            Assert.True(Math.Abs(gpuLosses[s] - cpuLosses[s]) <= 2e-3 * Math.Max(1.0, Math.Abs(cpuLosses[s])),
                $"step {s}: the generator plan's loss {gpuLosses[s]} on the GPU differs from the CPU's {cpuLosses[s]}; " +
                "it read the shared weights stale.");
        }
        for (int i = 0; i < cpuShared.Length; i++)
            Assert.True(Math.Abs(gpuShared[i] - cpuShared[i]) <= 2e-3 * Math.Max(1.0, Math.Abs(cpuShared[i])),
                $"shared weight {i}: {gpuShared[i]} on the GPU against {cpuShared[i]} on the CPU.");
    }
}
