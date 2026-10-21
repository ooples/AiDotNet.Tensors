using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A compiled plan training with plain SGD on a GPU updates every parameter in one multi-tensor launch, guarded on the
/// device by the gradients' sum of squares. It used to launch one SGD kernel per tensor after reading that sum back to
/// the host, a blocking readback every step. A non-finite gradient must still discard the whole step.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledTrainingMultiTensorSgdTests : IDisposable
{
    private const int Batch = 16, Inputs = 8, Hidden = 16, Outputs = 4;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledTrainingMultiTensorSgdTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static DirectGpuTensorEngine? CudaEngine()
    {
        try
        {
            var gpu = new DirectGpuTensorEngine();
            if (gpu.IsGpuAvailable && gpu.GetBackend() is IMultiTensorGpuOptimizerBackend) return gpu;
            gpu.Dispose();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        return null;
    }

    [SkippableFact]
    public void Sgd_steps_without_a_blocking_readback_and_still_discard_a_non_finite_step()
    {
        var gpu = CudaEngine();
        Skip.If(gpu is null, "No GPU backend with multi-tensor optimizer support.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu!;
            var x = Rand([Batch, Inputs], 1, 1f);
            var y = Rand([Batch, Outputs], 2, 1f);
            var w1 = Rand([Inputs, Hidden], 3, 0.3f);
            var b1 = new Tensor<float>([Hidden]);
            var w2 = Rand([Hidden, Outputs], 4, 0.3f);
            var b2 = new Tensor<float>([Outputs]);
            var parameters = new[] { w1, b1, w2, b2 };
            foreach (var p in parameters) p.Gpu();

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var h = gpu!.FusedLinear(x, w1, b1, FusedActivationType.ReLU);
                var pred = gpu.FusedLinear(h, w2, b2, FusedActivationType.None);
                var diff = gpu.TensorSubtract(pred, y);
                gpu.ReduceMean(gpu.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
                plan = scope.CompileTraining(parameters);
            }

            using (plan)
            {
                ((CompiledTrainingPlan<float>)plan).DisableGraphStep();   // the optimizer path under test, every step
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.05);
                for (int s = 0; s < 3; s++) plan.Step();   // warm: buffers, the deferred-readback slot

                long readbacksBefore = GpuLaunchProbe.Readbacks;
                const int Steps = 5;
                // Capture the sites, so a failure names the call that waits instead of only counting the waits.
                bool priorCapture = GpuLaunchProbe.CaptureReadbackSites;
                GpuLaunchProbe.CaptureReadbackSites = true;
                string[] sites;
                long readbacks;
                try
                {
                    for (int s = 0; s < Steps; s++) plan.Step();
                    readbacks = GpuLaunchProbe.Readbacks - readbacksBefore;
                    sites = GpuLaunchProbe.ReadbackSites;
                }
                finally
                {
                    GpuLaunchProbe.CaptureReadbackSites = priorCapture;
                }
                _output.WriteLine($"blocking readbacks over {Steps} SGD steps: {readbacks}");
                foreach (var site in sites) _output.WriteLine("  site: " + site);
                // A backend whose guarded update reads the guard in its kernel needs no download at all. Where the
                // guard is read on the host the step still cannot be free, because that backend's reduce-to-scalar
                // returns the loss to the host: one per step. Either way the optimizer must add none of its own -
                // nothing per parameter tensor (the per-tensor guard read this replaced) and no second wait for a
                // verdict already being read.
                long allowed = gpu!.GetBackend() is IMultiTensorGpuOptimizerBackend { GuardsFiniteOnDevice: true } ? 0 : Steps;
                Assert.True(readbacks <= allowed,
                    $"{readbacks} blocking readbacks over {Steps} steps, more than the {allowed} this backend's loss needs; the SGD step waits on the device for something else"
                    + (sites.Length == 0 ? string.Empty : Environment.NewLine + string.Join(Environment.NewLine, sites)));
                Assert.Equal(0, plan.NonFiniteStepsSkipped);

                // A non-finite batch: every gradient is NaN, so the step must leave the weights exactly as they were.
                var before = parameters.Select(p => p.ToArray()).ToArray();
                var nan = x.AsWritableSpan();
                for (int i = 0; i < nan.Length; i++) nan[i] = float.NaN;
                plan.Step();
                Assert.True(plan.LastStepSkippedNonFiniteGradients, "a NaN batch was applied to the weights");
                Assert.Equal(1, plan.NonFiniteStepsSkipped);
                for (int p = 0; p < parameters.Length; p++)
                    Assert.Equal(before[p], parameters[p].ToArray());
            }
        }
    }
}
