using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A transposed convolution feeding its channel bias add (a generator's layers) compiles on the GPU to one
/// <c>FusedConvTranspose2DBias</c> node, which on CUDA adds the bias inside the single-GEMM route's overlap-add and
/// after the other routes. The plan must train like the unfused plan on the same device -- bit for bit in the forward,
/// since the overlap-add adds the bias last as the bias pass does -- and like the CPU, over layers on the GEMM route
/// (a 4x4 stem, a 3-channel head) and one off it.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledConvTransposeBiasStepTests : IDisposable
{
    private const int Steps = 5;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledConvTransposeBiasStepTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private sealed class Run
    {
        public double[] Losses = Array.Empty<double>();
        public float[][] FinalWeights = Array.Empty<float[]>();
        public string[] StepNames = Array.Empty<string>();
    }

    private static Run Train(IEngine engine, bool onDevice, int batch)
    {
        AiDotNetEngine.Current = engine;
        var z = Rand([batch, 16, 4, 4], 1, 1f);
        var target = Rand([batch, 3, 32, 32], 2, 0.5f);
        var k1 = Rand([16, 8, 4, 4], 3, 0.15f);
        var b1 = Rand([8], 4, 0.3f);
        var k2 = Rand([8, 8, 4, 4], 5, 0.2f);
        var b2 = Rand([8], 6, 0.3f);
        var k3 = Rand([8, 3, 4, 4], 7, 0.2f);
        var b3 = Rand([3], 8, 0.3f);
        var parameters = new[] { k1, b1, k2, b2, k3, b3 };
        if (onDevice) foreach (var p in parameters) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            // 4x4 -> 8x8 (input plane 16: the GEMM route), 8x8 -> 16x16 (8 x 16 taps: off it), 16x16 -> 32x32 (a
            // 3-channel head, 48 taps: the GEMM route).
            var h = engine.ReLU(engine.FusedConvTranspose2D(z, k1, b1, 2, 2, 1, 1, 0, 0, FusedActivationType.None));
            h = engine.ReLU(engine.FusedConvTranspose2D(h, k2, b2, 2, 2, 1, 1, 0, 0, FusedActivationType.None));
            h = engine.FusedConvTranspose2D(h, k3, b3, 2, 2, 1, 1, 0, 0, FusedActivationType.None);
            engine.TensorMSELoss(h, target);
            plan = scope.CompileTraining(parameters);
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.05f);
            var run = new Run { Losses = new double[Steps] };
            for (int s = 0; s < Steps; s++)
            {
                Rand(z.Shape.ToArray(), 100 + s, 1f).AsSpan().CopyTo(z.AsWritableSpan());
                run.Losses[s] = plan.Step().ToArray()[0];
            }
            run.FinalWeights = parameters.Select(p => p.ToArray()).ToArray();
            if (plan is CompiledTrainingPlan<float> concrete)
            {
                var steps = (CompiledStep<float>[]?)typeof(CompiledTrainingPlan<float>)
                    .GetField("_forwardSteps", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(concrete);
                run.StepNames = steps?.Select(s => s.OpName).ToArray() ?? Array.Empty<string>();
            }
            return run;
        }
    }

    private static Run TrainUnfused(IEngine engine, bool onDevice, int batch)
    {
        var pattern = FusionPatternRegistry.Patterns.First(p => p.Name == "ConvTranspose2DBias");
        Assert.True(FusionPatternRegistry.Unregister(pattern.Name));
        try { return Train(engine, onDevice, batch); }
        finally { FusionPatternRegistry.Register(pattern); }
    }

    private static void AssertTrainsLike(Run reference, Run fused, string label, double tolerance)
    {
        for (int s = 0; s < Steps; s++)
            Assert.True(Math.Abs(reference.Losses[s] - fused.Losses[s]) <= tolerance * Math.Max(1, Math.Abs(reference.Losses[s])),
                $"{label} step {s}: loss {fused.Losses[s]:G9} != {reference.Losses[s]:G9}");
        for (int p = 0; p < reference.FinalWeights.Length; p++)
        {
            double maxErr = 0;
            for (int i = 0; i < reference.FinalWeights[p].Length; i++)
                maxErr = Math.Max(maxErr, Math.Abs(reference.FinalWeights[p][i] - fused.FinalWeights[p][i])
                    / Math.Max(1.0, Math.Abs(reference.FinalWeights[p][i])));
            Assert.True(maxErr <= tolerance, $"{label} parameter {p}: max rel error {maxErr:E2} after {Steps} steps");
        }
    }

    private static DirectGpuTensorEngine? TryGpu()
    {
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (candidate.IsGpuAvailable) return candidate;
            candidate.Dispose();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        return null;
    }

    [SkippableTheory]
    [InlineData(2)]
    [InlineData(16)]
    public void A_gpu_plan_fuses_the_transposed_convolution_bias_and_trains_like_the_unfused_plans(int batch)
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU backend did not resolve.");
        if (gpu is null) return;
        using (gpu)
        {
            var unfused = TrainUnfused(gpu, onDevice: true, batch);
            Assert.Equal(3, unfused.StepNames.Count(n => n == "TensorChannelBiasAdd"));
            long before = System.Threading.Interlocked.Read(ref CudaBackend.ConvTransposeBiasInOverlapAdd);
            var fused = Train(gpu, onDevice: true, batch);
            long inOverlapAdd = System.Threading.Interlocked.Read(ref CudaBackend.ConvTransposeBiasInOverlapAdd) - before;
            _output.WriteLine("steps: " + string.Join(", ", fused.StepNames));
            _output.WriteLine($"biases added in the overlap-add: {inOverlapAdd}");
            Assert.Equal(3, fused.StepNames.Count(n => n == "FusedConvTranspose2DBias"));
            Assert.DoesNotContain("TensorChannelBiasAdd", fused.StepNames);
            Assert.DoesNotContain("ConvTranspose2D", fused.StepNames);
            if (gpu.GetBackend() is CudaBackend)
                Assert.True(inOverlapAdd >= 2 * Steps, $"the GEMM-route layers' biases ran in the overlap-add {inOverlapAdd} times");
            AssertTrainsLike(unfused, fused, "gpu fused vs gpu unfused", 1e-6);
            var cpu = TrainUnfused(new CpuEngine(), onDevice: false, batch);
            AssertTrainsLike(cpu, fused, "gpu fused vs cpu unfused", 1e-3);
        }
    }
}
