using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Stateful compiled inference plans: a recurrent step (a linear-attention-style state S, read both as the first and as
/// the second operand of a matmul) advances its state on every execution, on the CPU engine, the GPU eagerly, and the
/// GPU through a captured CUDA graph, matching an eager reference step for step.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CompiledStatefulInferenceTests
{
    private const int D = 64;
    private const int Steps = 12;

    private sealed class EngineScope : IDisposable
    {
        private readonly IEngine _prior = AiDotNetEngine.Current;
        public EngineScope(IEngine engine) => AiDotNetEngine.Current = engine;
        public void Dispose() => AiDotNetEngine.Current = _prior;
    }

    private static float[] Input(int step)
    {
        var rng = new Random(300 + step);
        var x = new float[D];
        for (int i = 0; i < D; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }

    // One step: S' = 0.9 S + tanh(q) xᵀ with q = x as a column; out = (S' q)ᵀ + qᵀ S (S as the second operand).
    private static (Tensor<float> Output, Tensor<float> Next) Step(IEngine e, Tensor<float> s, Tensor<float> x)
        => Step(e, s, x, e.Reshape(x, [D, 1]));

    private static (Tensor<float> Output, Tensor<float> Next) Step(IEngine e, Tensor<float> s, Tensor<float> x, Tensor<float> q)
    {
        var next = e.TensorAdd(e.TensorMultiplyScalar(s, 0.9f), e.TensorMatMul(e.Tanh(q), x));
        var o1 = e.Reshape(e.TensorMatMul(next, q), [1, D]);
        var o2 = e.TensorMatMul(x, s);
        return (e.TensorAdd(o1, o2), next);
    }

    private static (float[][] Outputs, float[][] States) Reference()
    {
        var cpu = new CpuEngine();
        var s = new Tensor<float>([D, D]);
        var outputs = new float[Steps][];
        var states = new float[Steps][];
        for (int t = 0; t < Steps; t++)
        {
            var (o, next) = Step(cpu, s, new Tensor<float>([1, D], new Vector<float>(Input(t))));
            outputs[t] = o.ToArray();
            s = next;
            states[t] = s.ToArray();
        }
        return (outputs, states);
    }

    private static void RunPlan(IEngine engine, bool expectGraph)
    {
        var (refOut, refState) = Reference();
        var s = new Tensor<float>([D, D]);
        var x = new Tensor<float>([1, D], new Vector<float>(Input(0)));
        var qIn = new Tensor<float>([D, 1], new Vector<float>(Input(0)));
        CompiledInferencePlan<float> plan;
        using (new EngineScope(engine))
        using (var scope = GraphMode.EnableInference())
        {
            var (o, next) = Step(engine, s, x, qIn);
            plan = scope.CompileInference(o, new[] { x, qIn }, new[] { (s, next) });
        }
        using (plan)
        {
            for (int t = 0; t < Steps; t++)
            {
                plan.SetInputs(new[] { new Tensor<float>([1, D], new Vector<float>(Input(t))), new Tensor<float>([D, 1], new Vector<float>(Input(t))) });
                var output = plan.Execute().ToArray();
                var state = s.ToArray();
                for (int i = 0; i < D * D; i++)
                    Assert.True(Math.Abs(state[i] - refState[t][i]) <= 1e-5f * (1 + Math.Abs(refState[t][i])),
                        $"step {t} state[{i}] {state[i]} vs {refState[t][i]}");
                for (int i = 0; i < D; i++)
                    Assert.True(Math.Abs(output[i] - refOut[t][i]) <= 1e-5f * (1 + Math.Abs(refOut[t][i])),
                        $"step {t} output[{i}] {output[i]} vs {refOut[t][i]}");
            }
            if (expectGraph)
            {
                var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)
                    ?? throw new InvalidOperationException("_graphExec field not found.");
                Assert.True(exec.GetValue(plan) is IntPtr h && h != IntPtr.Zero, "the stateful plan is not replaying a captured graph");
            }
        }
    }

    [Fact]
    public void A_stateful_plan_advances_its_state_on_the_cpu_engine() => RunPlan(new CpuEngine(), expectGraph: false);

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_stateful_plan_advances_its_state_on_the_gpu(bool capture)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            if (capture)
                Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Inference graph capture is CUDA-only.");
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = capture;
            try { RunPlan(gpu!, expectGraph: capture); }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }

    [Fact]
    public void The_public_entry_point_compiles_a_stateful_plan()
    {
        var (refOut, refState) = Reference();
        var cpu = new CpuEngine();
        var s = new Tensor<float>([D, D]);
        var x = new Tensor<float>([1, D], new Vector<float>(Input(0)));
        using var cache = new CompiledModelCache<float>();
        ICompiledPlan<float> plan;
        using (new EngineScope(cpu))
            plan = cache.CompileStatefulInference(new[] { x }, () =>
            {
                var (o, next) = Step(cpu, s, x);
                return (o, new[] { (s, next) });
            });
        using (plan)
            for (int t = 0; t < Steps; t++)
            {
                plan.SetInputs(new[] { new Tensor<float>([1, D], new Vector<float>(Input(t))) });
                var output = plan.Execute().ToArray();
                var state = s.ToArray();
                Assert.True(Math.Abs(output[0] - refOut[t][0]) <= 1e-5f * (1 + Math.Abs(refOut[t][0])), $"step {t} output");
                Assert.True(Math.Abs(state[D + 1] - refState[t][D + 1]) <= 1e-5f * (1 + Math.Abs(refState[t][D + 1])), $"step {t} state");
            }
    }

    [Fact]
    public void A_state_must_be_a_leaf_the_graph_reads_and_its_next_value_must_be_computed()
    {
        var cpu = new CpuEngine();
        var s = new Tensor<float>([D, D]);
        var unread = new Tensor<float>([D, D]);
        var x = new Tensor<float>([1, D], new Vector<float>(Input(0)));
        using (new EngineScope(cpu))
        {
            using (var scope = GraphMode.EnableInference())
            {
                var (o, next) = Step(cpu, s, x);
                Assert.Throws<ArgumentException>(() => scope.CompileInference(o, new[] { x }, new[] { (unread, next) }));
            }
            using (var scope = GraphMode.EnableInference())
            {
                var (o, next) = Step(cpu, s, x);
                Assert.Throws<ArgumentException>(() => scope.CompileInference(o, new[] { x }, new[] { (next, next) }));
            }
            using (var scope = GraphMode.EnableInference())
            {
                var (o, _) = Step(cpu, s, x);
                Assert.Throws<ArgumentException>(() => scope.CompileInference(o, new[] { x }, new[] { (x, o) }));
            }
        }
    }
}
