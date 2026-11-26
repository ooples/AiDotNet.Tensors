using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The fused SwiGLU MoE in a compiled INFERENCE plan (a decode step): the op is one node that re-routes every
/// execution, so each replay's output and loads follow that execution's inputs, matching the eager op.
/// </summary>
[Collection("DirectGpuSerial")]
public class MoeInferencePlanTests
{
    private const int N = 3, D = 8, G = 6, K = 2, F = 4;

    private sealed class EngineScope : IDisposable
    {
        private readonly IEngine _prior = AiDotNetEngine.Current;
        public EngineScope(IEngine engine) => AiDotNetEngine.Current = engine;
        public void Dispose() => AiDotNetEngine.Current = _prior;
    }

    private static Tensor<float> T(int[] shape, Func<int, float> f)
        => new(shape, new Vector<float>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(f).ToArray()));

    [Fact]
    public void A_cpu_inference_plan_reroutes_the_moe_every_execution() => Run(new CpuEngine());

    [SkippableFact]
    public void A_gpu_inference_plan_reroutes_the_moe_every_execution()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu) Run(gpu!);
    }

    /// <summary>
    /// A forward that hands the MoE's loads to their own lifetime (TensorScope.Detach, a per-forward expert-load
    /// statistic kept past the call) compiles: Detach reads no value, so it is no host read of the traced loads. Every
    /// replay refreshes the detached loads with that execution's routing. Detach realized the loads, which an inference
    /// trace records as a host escape, so the compile threw GraphCaptureNotSupportedException.
    /// </summary>
    [Fact]
    public void A_cpu_inference_plan_compiles_a_forward_that_detaches_the_moe_loads() => RunDetached(new CpuEngine());

    [SkippableFact]
    public void A_gpu_inference_plan_compiles_a_forward_that_detaches_the_moe_loads()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu) RunDetached(gpu!);
    }

    /// <summary>
    /// A detached traced tensor keeps its own storage in a CPU plan long enough for memory planning. The plan reads it
    /// once, early; after that read memory planning counted it dead and handed its storage to the next same-shaped
    /// result, so without the plan protecting it the caller's reference showed that later result's values.
    /// </summary>
    [Fact]
    public void A_detached_traced_tensor_keeps_its_storage_in_a_memory_planned_cpu_plan()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>([2, D]);
        Tensor<float>? kept = null;
        static Tensor<float> Forward(IEngine e, Tensor<float> xi, out Tensor<float> side)
        {
            side = e.TensorMultiply(xi, xi);                  // kept past the call, and read once below
            var a = e.TensorAdd(side, xi);                    // side's last read in the plan
            var b = e.TensorAdd(a, xi);                       // same shape, produced after: memory planning's candidate
            var c = e.TensorMultiply(b, a);
            return e.TensorAdd(c, b);
        }
        using (new EngineScope(engine))
        using (var cache = new CompiledModelCache<float>())
        {
            var plan = cache.GetOrCompileInference(x, xi =>
            {
                var y = Forward(engine, xi, out var side);
                kept = TensorScope.Detach(side);
                return y;
            });
            var cpu = new CpuEngine();
            for (int t = 0; t < 3; t++)
            {
                var xt = T([2, D], i => MathF.Sin(i * 0.7f + t) + 0.1f * t);
                plan.SetInputs(new[] { xt });
                var got = plan.Execute().ToArray();
                var expect = Forward(cpu, xt, out var side).ToArray();
                for (int i = 0; i < got.Length; i++)
                    Assert.True(Math.Abs(got[i] - expect[i]) <= 1e-5f * (1 + Math.Abs(expect[i])), $"t{t} y[{i}]: {got[i]} vs {expect[i]}");
                Assert.Equal(side.ToArray(), kept!.ToArray());
            }
        }
    }

    /// <summary>
    /// Inside an open TensorScope, keeping the loads in the scope and detaching them both read no value: the plan
    /// compiles and the loads follow every replay.
    /// </summary>
    [Fact]
    public void Keeping_and_detaching_the_moe_loads_inside_a_tensor_scope_compiles()
    {
        using var scope = TensorScope.Begin();
        RunDetached(new CpuEngine(), keepInScope: scope);
    }

    /// <summary>A real host read of the loads inside the forward still makes the trace uncompilable.</summary>
    [Fact]
    public void A_host_read_of_the_moe_loads_inside_the_forward_still_refuses_the_compile()
    {
        var engine = new CpuEngine();
        var (w1, w3, w2) = Weights();
        using (new EngineScope(engine))
        using (var cache = new CompiledModelCache<float>())
            Assert.Throws<GraphCaptureNotSupportedException>(() => cache.GetOrCompileInference(new Tensor<float>([N, D]), xi =>
            {
                var y = MixtureOfExperts.SwiGluExperts(engine, xi, engine.TensorMatMul(xi, Router()), null, w1, w3, w2, K, 1, out var loads);
                _ = loads.ToArray();
                return y;
            }));
    }

    private static (Tensor<float>, Tensor<float>, Tensor<float>) Weights()
        => (T([G, D, F], i => 0.1f * MathF.Sin(i)), T([G, D, F], i => 0.1f * MathF.Cos(i * 0.7f)), T([G, F, D], i => 0.1f * MathF.Sin(i * 1.3f + 1)));

    private static Tensor<float> Router() => T([D, G], i => MathF.Sin(i * 0.37f + 2));

    private static void RunDetached(IEngine engine, TensorScope? keepInScope = null)
    {
        var (w1, w3, w2) = Weights();
        var router = Router();
        var x = new Tensor<float>([N, D]);
        Tensor<float>? kept = null;
        using (new EngineScope(engine))
        using (var cache = new CompiledModelCache<float>())
        {
            var plan = cache.GetOrCompileInference(x, xi =>
            {
                var y = MixtureOfExperts.SwiGluExperts(engine, xi, engine.TensorMatMul(xi, router), null, w1, w3, w2, K, 1, out var loads);
                if (keepInScope is not null) keepInScope.Keep(loads);
                kept = TensorScope.Detach(loads);
                return y;
            });
            Assert.NotNull(kept);
            var cpu = new CpuEngine();
            var seen = new System.Collections.Generic.HashSet<string>();
            for (int t = 0; t < 3; t++)
            {
                // Each execution's input routes to different experts.
                var xt = T([N, D], i => 2f * MathF.Sin(i * (0.9f + 0.4f * t) + 3 * t));
                plan.SetInputs(new[] { xt });
                var got = plan.Execute().ToArray();
                var expect = MixtureOfExperts.SwiGluExperts(cpu, xt, cpu.TensorMatMul(xt, router), null, w1, w3, w2, K, 1, out var eagerLoads).ToArray();
                for (int i = 0; i < got.Length; i++)
                    Assert.True(Math.Abs(got[i] - expect[i]) <= 1e-4f * (1 + Math.Abs(expect[i])), $"t{t} y[{i}]: {got[i]} vs {expect[i]}");
                Assert.Equal(eagerLoads.ToArray(), kept!.ToArray());
                seen.Add(string.Join(",", eagerLoads.ToArray()));
            }
            Assert.True(seen.Count > 1, "every execution routed alike, so a stale copy of the loads would pass");
        }
    }

    private static void Run(IEngine engine)
    {
        var w1 = T([G, D, F], i => 0.1f * MathF.Sin(i));
        var w3 = T([G, D, F], i => 0.1f * MathF.Cos(i * 0.7f));
        var w2 = T([G, F, D], i => 0.1f * MathF.Sin(i * 1.3f + 1));
        var x = new Tensor<float>([N, D]);
        var scores = new Tensor<float>([N, G]);
        CompiledInferencePlan<float> plan;
        Tensor<float> tracedLoads;
        using (new EngineScope(engine))
        using (var scope = GraphMode.EnableInference())
        {
            var y = MixtureOfExperts.SwiGluExperts(engine, x, scores, null, w1, w3, w2, K, 1, out tracedLoads);
            plan = scope.CompileInference(engine.TensorAdd(y, engine.Reshape(engine.ReduceSum(tracedLoads, new[] { 1 }, keepDims: true), [1, 1])),
                new[] { x, scores }, Array.Empty<(Tensor<float>, Tensor<float>)>());
        }
        using (plan)
            for (int t = 0; t < 3; t++)
            {
                // Each execution routes differently: the scores favour a different expert pair every time.
                var xt = T([N, D], i => MathF.Sin(i + 5 * t));
                var st = T([N, G], i => (i % G) == (t + i / G) % G || (i % G) == (t + 3) % G ? 2f : 0.1f * (i % 3));
                plan.SetInputs(new[] { xt, st });
                var got = plan.Execute().ToArray();
                var cpu = new CpuEngine();
                var expect = MixtureOfExperts.SwiGluExperts(cpu, xt, st, null, w1, w3, w2, K, 1, out var loads).ToArray();
                float loadSum = loads.ToArray().Sum();
                Assert.Equal(N * K, loadSum);
                for (int i = 0; i < got.Length; i++)
                    Assert.True(Math.Abs(got[i] - (expect[i] + loadSum)) <= 1e-4f * (1 + Math.Abs(expect[i] + loadSum)),
                        $"t{t} y[{i}]: {got[i]} vs {expect[i] + loadSum}");
            }
    }
}
