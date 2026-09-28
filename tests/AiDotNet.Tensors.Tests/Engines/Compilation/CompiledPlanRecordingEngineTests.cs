using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A graph scope defaulted to the process-wide AiDotNetEngine.Current and only some ops rebound it to the engine they
/// were invoked on, so a model recorded on an explicit CpuEngine compiled to a plan that ran on whatever engine was
/// global at the time - measured in the full suite as "cuGraphLaunch failed: Invalid value" from a CPU benchmark whose
/// plan had been bound to another test's (since disposed) GPU engine. Every recording op now binds its own engine.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class CompiledPlanRecordingEngineTests
{
    [SkippableFact]
    public void A_plan_recorded_on_an_explicit_cpu_engine_runs_on_it_whatever_engine_is_global()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception) { }
        Skip.IfNot(gpu is not null && gpu.IsGpuAvailable, "needs a second engine to be the global one");
        var prior = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu!;
            var cpu = new CpuEngine();
            var rng = new Random(7);   // host-generated: CreateRandom(dims) would run on the global engine
            var input = Tensor<float>.CreateRandom(rng, 8, 16);
            var w1 = Tensor<float>.CreateRandom(rng, 16, 12);
            var w2 = Tensor<float>.CreateRandom(rng, 12, 4);
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var h = cpu.ReLU(cpu.TensorMatMul(input, w1));
                cpu.TensorMatMul(h, w2);
                plan = scope.CompileTraining(new[] { w1, w2 });
            }
            using (plan)
            {
                var engine = typeof(CompiledTrainingPlan<float>)
                    .GetField("_engine", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan);
                Assert.Same(cpu, engine);
                // The global engine going away must not affect a plan that never ran on it.
                AiDotNetEngine.Current = prior;
                gpu!.Dispose();
                for (int i = 0; i < 5; i++) plan.Step();
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
            gpu?.Dispose();
        }
    }
}
