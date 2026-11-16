using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A closure objective with a LayerNorm must be signable on every LayerNorm recording path, so its compiled plan
/// replays. The generic node (double, and float on a device engine) keeps its replay-time statistics in a holder its
/// Execute closure captures; a holder the signature could not compare left such an objective without a guard.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class LayerNormClosureObjectiveTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    [Fact]
    public void Float_host_plan_replays_each_step_on_new_data() => Check<float>(gpu: false);

    [Fact]
    public void Double_host_plan_replays_each_step_on_new_data() => Check<double>(gpu: false);

    [SkippableFact]
    public void Float_device_plan_replays_each_step_on_new_data() => Check<float>(gpu: true);

    private static void Check<T>(bool gpu)
    {
        T V(double value) => (T)Convert.ChangeType(value, typeof(T));
        DirectGpuTensorEngine? device = null;
        if (gpu)
        {
            try { device = new DirectGpuTensorEngine(); }
            catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
            Skip.If(device is null || !device.IsGpuAvailable, "No direct GPU backend available.");
        }
        using (device)
        {
            IEngine engine = device is null ? new CpuEngine() : device;
            AiDotNetEngine.Current = engine;
            const int rows = 4, features = 6;
            var gamma = new Tensor<T>(Enumerable.Repeat(V(1), features).ToArray(), new[] { features });
            var beta = new Tensor<T>(Enumerable.Repeat(V(0.5), features).ToArray(), new[] { features });
            var parameters = new[] { gamma, beta };
            int step = 0;
            Tensor<T> Data(int s)
            {
                var rng = new Random(50 + s);
                return new Tensor<T>(Enumerable.Range(0, rows * features).Select(_ => V(rng.NextDouble())).ToArray(),
                    new[] { rows, features });
            }
            Tensor<T> Loss(IEngine e, Tensor<T> x)
                => e.ReduceSum(e.TensorMultiply(e.LayerNorm(x, gamma, beta, 1e-5, out _, out _), x), null);
            Func<Tensor<T>> objective = () => Loss(engine, Data(step));   // new data, new tensors, every call

            using var recording = ClosureObjectiveGuard<T>.BeginRecording();
            CompiledTrainingPlan<T> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
                plan = scope.CompileTraining(parameters, objective());
            var guard = ClosureObjectiveGuard<T>.TakeRecorded(out var reason);
            using (plan)
            {
                Assert.True(guard is not null, $"the objective was not signable: {reason}");
                guard!.AttachTo(plan);
                var reference = new CpuEngine();
                for (step = 0; step < 4; step++)
                {
                    var rebound = guard.TryRebind(objective, parameters, out _);
                    Assert.True(rebound == ClosureGuardResult.Matched, $"step {step}: {rebound} ({guard.LastReason})");
                    double compiled = Convert.ToDouble(plan.Step().ToArray()[0]);
                    double eager = Convert.ToDouble(Loss(reference, Data(step)).ToArray()[0]);
                    Assert.Equal(eager, compiled, 3);
                }
            }
        }
    }
}
