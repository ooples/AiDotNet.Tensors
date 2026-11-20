using System.Linq;
using System.Collections.Generic;
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
    [Fact]
    public void ReturnedIntermediateAndStateConsumerRefreshIndependently()
    {
        var cpu = new CpuEngine();
        var x = new Tensor<float>(new[] { 2f }, new[] { 1, 1 });
        var w = new Tensor<float>(new[] { 3f }, new[] { 1, 1 });
        var state = new Tensor<float>(new[] { 1f }, new[] { 1, 1 });
        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.EnableInference())
        {
            var product = cpu.TensorMatMul(x, w);
            var next = cpu.Sigmoid(cpu.TensorAdd(product, state));
            plan = scope.CompileInference(product, new[] { x }, new[] { (state, next) });
        }
        using (plan)
        {
            Assert.Equal(6f, plan.Execute()[0]);
            Assert.Equal(1f / (1f + (float)Math.Exp(-7)), state[0], 6);
            plan.SetInputs(new[] { new Tensor<float>(new[] { -1f }, new[] { 1, 1 }) });
            Assert.Equal(-3f, plan.Execute()[0]);
        }
    }

    [Fact]
    public void IndexSelectionObservesTopKProducerAfterScoresChange()
    {
        var cpu = new CpuEngine();
        var x = new Tensor<float>(new[] { 1f, 4f }, new[] { 2 });
        var table = new Tensor<float>(new[] { 10f, 20f }, new[] { 2 });
        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.EnableInference())
        {
            var computedTable = cpu.TensorMultiplyScalar(table, 2f);
            var scores = cpu.TensorMultiplyScalar(x, 3f);
            cpu.TensorTopK(scores, 1, 0, out var indices);
            var selected = cpu.TensorIndexSelect(computedTable, indices, 0);
            plan = scope.CompileInference(selected, new[] { x });
        }
        using (plan)
        {
            Assert.Equal(40f, plan.Execute()[0]);
            plan.SetInputs(new[] { new Tensor<float>(new[] { 5f, 1f }, new[] { 2 }) });
            Assert.Equal(20f, plan.Execute()[0]);
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ActivationClearWaitsForTheLastLiveGraph(bool discard)
    {
        var backend = DirectGpu.MockDirectGpuBackend.Create(new DirectGpu.MockBackendState());
        using var direct = new AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var state = new Tensor<float>(new[] { 3f }, new[] { 1 });
        engine.BeginGraphLifetime();
        engine.BeginGraphLifetime();
        var buffer = Assert.IsType<DirectGpu.MockGpuBuffer>(engine.PinInputResident(state));
        if (discard) engine.DropActivationCache();
        else engine.ClearActivationCache();
        Assert.Equal(0, buffer.DisposeCount);
        engine.EndGraphLifetime();
        Assert.Equal(0, buffer.DisposeCount);
        engine.EndGraphLifetime();
        Assert.Equal(1, buffer.DisposeCount);
        Assert.Equal(3f, state.ToArray()[0]);
    }

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
                if (expectGraph && t == Steps - 1) ((DirectGpuTensorEngine)engine).ClearActivationCache();
                int previousVersion = s.Version;
                var output = plan.Execute().ToArray();
                if (engine is CpuEngine && engine is not DirectGpuTensorEngine) Assert.NotEqual(previousVersion, s.Version);
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

    // A one-hot position [1, P] shifted each step by Concat(0, Narrow(state)) and read as onehot · table: the rows of
    // the table come out in order, on every path.
    private static void RunOneHotPosition(IEngine engine, bool expectGraph)
    {
        const int P = 16, W = 8;
        var table = new Tensor<float>([P, W]);
        for (int i = 0; i < table.Length; i++) table[i / W, i % W] = i * 0.25f - 3f;
        var oneHot = new Tensor<float>([1, P]);
        oneHot[0, 0] = 1f;
        var zero = new Tensor<float>([1, 1]);
        var bias = new Tensor<float>([1, W]);
        CompiledInferencePlan<float> plan;
        using (new EngineScope(engine))
        using (var scope = GraphMode.EnableInference())
        {
            var row = engine.TensorAdd(engine.TensorMatMul(oneHot, table), bias);
            var next = engine.TensorConcatenate([zero, engine.TensorNarrow(oneHot, 1, 0, P - 1)], 1);
            plan = scope.CompileInference(row, new[] { bias }, new[] { (oneHot, next) });
        }
        using (plan)
        {
            for (int t = 0; t < P; t++)
            {
                plan.SetInputs(new[] { new Tensor<float>([1, W]) });
                var got = plan.Execute().ToArray();
                for (int j = 0; j < W; j++)
                    Assert.True(got[j] == table[t, j], $"step {t} col {j}: {got[j]} vs {table[t, j]}");
            }
            if (expectGraph)
            {
                var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                Assert.True(exec.GetValue(plan) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
            }
        }
    }

    [Fact]
    public void A_one_hot_position_state_walks_the_table_on_the_cpu_engine() => RunOneHotPosition(new CpuEngine(), expectGraph: false);

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_one_hot_position_state_walks_the_table_on_the_gpu(bool capture)
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
            try { RunOneHotPosition(gpu!, expectGraph: capture); }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }

    // Two stacked recurrent layers (batched like a multi-head scan, with a per-row decay): the second layer's
    // intermediates have the first's shapes, and its input is a unary chain the first layer's output also feeds.
    private static (Tensor<float> Output, Tensor<float> Next1, Tensor<float> Next2) TwoLayers(IEngine e,
        Tensor<float> s1, Tensor<float> s2, Tensor<float> x, Tensor<float> gate)
    {
        Tensor<float> Layer(Tensor<float> s, Tensor<float> h, out Tensor<float> next)
        {
            // h [R, D]: outer = h tanh(h)ᵀ per row as [R, D, D]; next = gate ⊙ s + outer; out = σ(next · h)
            var v = e.Reshape(h, [R, D, 1]);
            var k = e.Reshape(e.Tanh(h), [R, 1, D]);
            var outer = e.BatchMatMul(v, k);
            next = e.TensorAdd(e.TensorMultiply(s, gate), outer);
            return e.Sigmoid(e.Reshape(e.BatchMatMul(next, e.Reshape(h, [R, D, 1])), [R, D]));
        }
        var y1 = Layer(s1, x, out var n1);
        var y2 = Layer(s2, y1, out var n2);
        return (y2, n1, n2);
    }

    private const int R = 4;

    // abandonCapture: the capture throws partway (as a non-capturable op does), so the plan must carry on eagerly, every
    // later call reading that call's input and advancing the states from correct values.
    private static void RunTwoLayers(IEngine engine, bool capture, bool abandonCapture)
    {
        var gate = new Tensor<float>([R, 1, 1]);
        for (int i = 0; i < R; i++) gate[i, 0, 0] = 0.5f + 0.1f * i;
        float[] In(int t)
        {
            var rng = new Random(900 + t);
            var a = new float[R * D];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
            return a;
        }

        var cpu = new CpuEngine();
        var r1 = new Tensor<float>([R, D, D]);
        var r2 = new Tensor<float>([R, D, D]);
        var refOut = new List<float[]>();
        var refS1 = new List<float[]>();
        var refS2 = new List<float[]>();
        for (int t = 0; t < 6; t++)
        {
            var (o, n1, n2) = TwoLayers(cpu, r1, r2, new Tensor<float>([R, D], new Vector<float>(In(t))), gate);
            refOut.Add(o.ToArray());
            r1 = n1;
            r2 = n2;
            refS1.Add(r1.ToArray());
            refS2.Add(r2.ToArray());
        }

        var s1 = new Tensor<float>([R, D, D]);
        var s2 = new Tensor<float>([R, D, D]);
        var x = new Tensor<float>([R, D], new Vector<float>(In(0)));
        CompiledInferencePlan<float> plan;
        using (new EngineScope(engine))
        using (var scope = GraphMode.EnableInference())
        {
            var (o, n1, n2) = TwoLayers(engine, s1, s2, x, gate);
            // The output ends in a reshape (as a language model's logits do): its step re-binds the plan's output.
            plan = scope.CompileInference(engine.Reshape(o, [1, R * D]), new[] { x }, new[] { (s1, n1), (s2, n2) });
        }
        var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
        CompiledInferencePlan<float>.InferenceCudaGraphOverride = capture;
        if (abandonCapture)
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.TestHookInsideCapture =
                () => throw new InvalidOperationException("injected: a non-capturable step");
        try
        {
            using (plan)
            {
                for (int t = 0; t < 6; t++)
                {
                    plan.SetInputs(new[] { new Tensor<float>([R, D], new Vector<float>(In(t))) });
                    var output = plan.Execute();
                    var got = output.ToArray();
                    // A decode loop reads the output through the engine (a greedy argmax over a reshaped view of it).
                    var argmax = engine.TensorArgMax(output, 1).ToArray()[0];
                    Assert.Equal(Array.IndexOf(refOut[t], refOut[t].Max()), argmax);
                    AssertClose(refS1[t], s1.ToArray(), $"step {t} layer-1 state");
                    AssertClose(refS2[t], s2.ToArray(), $"step {t} layer-2 state");
                    AssertClose(refOut[t], got, $"step {t} output");
                }
                if (capture)
                {
                    var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                    Assert.Equal(!abandonCapture, (IntPtr)exec.GetValue(plan)! != IntPtr.Zero);
                }
            }
        }
        finally
        {
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior;
            if (abandonCapture) AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.TestHookInsideCapture = null;
        }
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= 1e-4f * (1 + Math.Abs(expected[i])),
                $"{what}[{i}]: {actual[i]} vs {expected[i]}");
    }

    [Fact]
    public void Two_stacked_recurrent_layers_advance_both_states_on_the_cpu_engine()
        => RunTwoLayers(new CpuEngine(), capture: false, abandonCapture: false);

    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void Two_stacked_recurrent_layers_advance_both_states_on_the_gpu(bool capture, bool abandonCapture)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            if (capture) Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Inference graph capture is CUDA-only.");
            RunTwoLayers(gpu!, capture, abandonCapture);
        }
    }

    // A deeper stack (six recurrent layers, each with its own state): later layers' same-shaped intermediates are
    // candidates for the storage of earlier layers' dead ones.
    private static (Tensor<float> Output, Tensor<float>[] Nexts) DeepStack(IEngine e, Tensor<float>[] states,
        Tensor<float> x, Tensor<float> gate)
    {
        var nexts = new Tensor<float>[states.Length];
        var h = x;
        for (int l = 0; l < states.Length; l++)
        {
            var v = e.Reshape(h, [R, D, 1]);
            var k = e.Reshape(e.Tanh(h), [R, 1, D]);
            nexts[l] = e.TensorAdd(e.TensorMultiply(states[l], gate), e.BatchMatMul(v, k));
            var read = e.Reshape(e.BatchMatMul(nexts[l], e.Reshape(h, [R, D, 1])), [R, D]);
            h = e.TensorAdd(h, e.Sigmoid(read));
        }
        return (h, nexts);
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_deep_recurrent_stack_advances_every_state_on_the_gpu(bool capture)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            if (capture) Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Inference graph capture is CUDA-only.");
            const int layers = 6, steps = 6;
            var gate = new Tensor<float>([R, 1, 1]);
            for (int i = 0; i < R; i++) gate[i, 0, 0] = 0.6f + 0.05f * i;
            float[] In(int t)
            {
                var rng = new Random(1700 + t);
                var a = new float[R * D];
                for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
                return a;
            }

            var cpu = new CpuEngine();
            var refStates = new Tensor<float>[layers];
            for (int l = 0; l < layers; l++) refStates[l] = new Tensor<float>([R, D, D]);
            var refOut = new List<float[]>();
            var refLast = new List<float[]>();
            for (int t = 0; t < steps; t++)
            {
                var (o, n) = DeepStack(cpu, refStates, new Tensor<float>([R, D], new Vector<float>(In(t))), gate);
                refStates = n;
                refOut.Add(o.ToArray());
                refLast.Add(n[layers - 1].ToArray());
            }

            var states = new Tensor<float>[layers];
            for (int l = 0; l < layers; l++) states[l] = new Tensor<float>([R, D, D]);
            var x = new Tensor<float>([R, D], new Vector<float>(In(0)));
            CompiledInferencePlan<float> plan;
            using (new EngineScope(gpu!))
            using (var scope = GraphMode.EnableInference())
            {
                var (o, n) = DeepStack(gpu!, states, x, gate);
                var pairs = new (Tensor<float>, Tensor<float>)[layers];
                for (int l = 0; l < layers; l++) pairs[l] = (states[l], n[l]);
                plan = scope.CompileInference(o, new[] { x }, pairs);
            }
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = capture;
            try
            {
                using (plan)
                {
                    for (int t = 0; t < steps; t++)
                    {
                        plan.SetInputs(new[] { new Tensor<float>([R, D], new Vector<float>(In(t))) });
                        var got = plan.Execute().ToArray();
                        AssertClose(refLast[t], states[layers - 1].ToArray(), $"step {t} last-layer state");
                        AssertClose(refOut[t], got, $"step {t} output");
                    }
                    if (capture)
                    {
                        var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                        Assert.NotEqual(IntPtr.Zero, (IntPtr)exec.GetValue(plan)!);
                    }
                }
            }
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
