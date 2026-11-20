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

    // A replayed graph re-reads its input buffers: an input whose host data is unchanged is not uploaded again, and
    // every change - a new tensor through SetInputs, or the caller writing the compiled slot itself - is uploaded.
    [SkippableFact]
    public void A_replay_skips_unchanged_inputs_and_uploads_every_change()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Inference graph capture is CUDA-only.");
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = true;
            try
            {
                const int W = 8;
                var x = new Tensor<float>([1, W]);
                var w = new Tensor<float>([1, W]);
                for (int j = 0; j < W; j++) w[0, j] = j + 1f;
                var counter = new Tensor<float>([1, 1]);
                CompiledInferencePlan<float> plan;
                using (new EngineScope(gpu))
                using (var scope = GraphMode.EnableInference())
                {
                    var y = gpu.TensorAdd(gpu.TensorMultiply(x, w), counter);
                    var next = gpu.TensorAddScalar(counter, 1f);
                    plan = scope.CompileInference(y, new[] { x }, new[] { (counter, next) });
                }
                using (plan)
                {
                    float[] A = Enumerable.Range(0, W).Select(j => j * 0.5f).ToArray();
                    float[] B = Enumerable.Range(0, W).Select(j => -j - 1f).ToArray();
                    float[] Negz = Enumerable.Range(0, W).Select(j => j == 0 ? -0f : j * 0.5f).ToArray();
                    // (input, fed through the compiled slot itself instead of a new tensor)
                    var feeds = new (float[] Data, bool InPlace)[]
                    {
                        (A, false), (A, false), (A, false), (B, false), (B, false), (A, true), (A, true), (B, true),
                        (Negz, false), (A, false), (B, false), (B, false),
                    };
                    static int Bits(float f) => BitConverter.ToInt32(BitConverter.GetBytes(f), 0);   // net471 has no SingleToInt32Bits
                    long skippedBefore = 0;
                    int expectedSkips = 0;
                    float[]? previous = null;
                    for (int t = 0; t < feeds.Length; t++)
                    {
                        var (data, inPlace) = feeds[t];
                        if (inPlace)
                        {
                            for (int j = 0; j < W; j++) x[0, j] = data[j];
                            plan.SetInputs(new[] { x });
                        }
                        else plan.SetInputs(new[] { new Tensor<float>([1, W], new Vector<float>((float[])data.Clone())) });
                        if (t == 2) skippedBefore = plan.InputUploadsSkipped;   // the graph is replaying from here on
                        else if (t > 2 && previous is not null
                                 && data.Select(Bits).SequenceEqual(previous.Select(Bits)))
                            expectedSkips++;
                        var got = plan.Execute().ToArray();
                        for (int j = 0; j < W; j++)
                        {
                            float expect = data[j] * (j + 1f) + t;
                            Assert.True(got[j] == expect, $"step {t} col {j}: {got[j]} vs {expect}");
                        }
                        previous = data;
                    }
                    var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                    Assert.True(exec.GetValue(plan) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
                    Assert.Equal(expectedSkips, plan.InputUploadsSkipped - skippedBefore);
                }
            }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }

    // The explicit output ends a fusable linear chain (matmul + bias add) and also feeds a state update through a
    // ReLU: the chain still fuses into the output tensor, the output is written, and the state reads it.
    [Fact]
    public void An_observable_output_that_ends_a_fusable_chain_is_fused_and_still_written()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>([2, 3]);
        var w = new Tensor<float>([3, 4]);
        var bias = new Tensor<float>([4]);
        for (int i = 0; i < w.Length; i++) w[i / 4, i % 4] = (i % 5) * 0.25f - 0.5f;
        for (int i = 0; i < 4; i++) bias[i] = i * 0.1f - 0.2f;
        var state = new Tensor<float>([2, 4]);
        CompiledInferencePlan<float> plan;
        Tensor<float> sum;
        using (new EngineScope(engine))
        using (var scope = GraphMode.EnableInference())
        {
            sum = engine.TensorBroadcastAdd(engine.TensorMatMul(x, w), bias);
            var next = engine.TensorAdd(state, engine.ReLU(sum));
            plan = scope.CompileInference(sum, new[] { x }, new[] { (state, next) });
        }
        using (plan)
        {
            var steps = (Array)typeof(CompiledInferencePlan<float>).GetField("_steps", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan)!;
            var names = steps.Cast<object>().Select(st => st.GetType().GetField("OpName", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(st) as string ?? "").ToArray();
            Assert.True(names.Any(name => name.IndexOf("FusedLinear", StringComparison.Ordinal) >= 0),
                "the chain did not fuse: " + string.Join(", ", names));
            var acc = new float[8];
            for (int t = 0; t < 3; t++)
            {
                var xs = Enumerable.Range(0, 6).Select(i => (i + t) * 0.5f - 1f).ToArray();
                plan.SetInputs(new[] { new Tensor<float>([2, 3], new Vector<float>(xs)) });
                var got = plan.Execute().ToArray();
                var observed = sum.ToArray();
                for (int r = 0; r < 2; r++)
                    for (int c = 0; c < 4; c++)
                    {
                        float e = bias[c];
                        for (int k = 0; k < 3; k++) e += xs[r * 3 + k] * w[k, c];
                        Assert.True(Math.Abs(got[r * 4 + c] - e) <= 1e-5f, $"step {t} out[{r},{c}] {got[r * 4 + c]} vs {e}");
                        Assert.True(Math.Abs(observed[r * 4 + c] - e) <= 1e-5f, $"step {t} observed sum[{r},{c}] {observed[r * 4 + c]} vs {e}");
                        acc[r * 4 + c] += Math.Max(0f, e);
                    }
                var st = state.ToArray();
                for (int i = 0; i < 8; i++) Assert.True(Math.Abs(st[i] - acc[i]) <= 1e-4f, $"step {t} state[{i}] {st[i]} vs {acc[i]}");
            }
        }
    }

    // A leaf written in place by the traced step (a static decode cache: TensorAddInPlace into a persistent tensor, no
    // declared state) advances exactly once per execution, the capture call included, and the host sees each value.
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void An_in_place_write_to_a_leaf_advances_once_per_execution(bool capture)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = capture;
            try
            {
                const int N = 37;
                var cache = new Tensor<float>([N]);
                var x = new Tensor<float>([N]);
                CompiledInferencePlan<float> plan;
                using (new EngineScope(gpu))
                using (var scope = GraphMode.EnableInference())
                {
                    gpu.TensorAddInPlace(cache, x);
                    var y = gpu.TensorMultiplyScalar(cache, 2f);
                    plan = scope.CompileInference(y, new[] { x }, Array.Empty<(Tensor<float>, Tensor<float>)>());
                }
                using (plan)
                {
                    var expect = new float[N];
                    for (int t = 0; t < 6; t++)
                    {
                        var xv = Enumerable.Range(0, N).Select(i => 0.5f * i + t).ToArray();
                        plan.SetInputs(new[] { new Tensor<float>([N], new Vector<float>(xv)) });
                        var y = plan.Execute().ToArray();
                        for (int i = 0; i < N; i++) expect[i] += xv[i];
                        var got = cache.ToArray();
                        for (int i = 0; i < N; i++)
                        {
                            Assert.True(Math.Abs(got[i] - expect[i]) <= 1e-4f, $"step {t} cache[{i}]: {got[i]} vs {expect[i]}");
                            Assert.True(Math.Abs(y[i] - 2 * expect[i]) <= 1e-4f, $"step {t} y[{i}]: {y[i]} vs {2 * expect[i]}");
                        }
                    }
                    if (capture && gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)
                    {
                        var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                        Assert.True(exec.GetValue(plan) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
                    }
                }
            }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }

    // Several states are written by one copy launch: the captured graph has no memcpy node for them, and every state
    // still advances exactly once per execution (checked against a CPU recurrence, including the capture call).
    [SkippableFact]
    public void A_captured_plan_writes_all_its_states_in_one_launch()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Inference graph capture is CUDA-only.");
            var cuda = (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)gpu.GetBackend()!;
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = true;
            try
            {
                // Lengths differ (and one exceeds a single launch's 1024 x 256 grid stride) so a slot mix-up shows.
                int[] lengths = { 7, 300_001, 64 };
                var rng = new Random(5);
                float[] Rand(int n) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
                var states = lengths.Select(n => new Tensor<float>([n], new Vector<float>(Rand(n)))).ToArray();
                var expect = states.Select(t => t.ToArray()).ToArray();
                var x = new Tensor<float>([1]);
                CompiledInferencePlan<float> plan;
                using (new EngineScope(gpu))
                using (var scope = GraphMode.EnableInference())
                {
                    // next_k = state_k * 0.5 + x (x broadcast); the output reads every state so each is live.
                    var nexts = states.Select(st => gpu.TensorAdd(gpu.TensorMultiplyScalar(st, 0.5f), gpu.TensorBroadcastAdd(new Tensor<float>(st.Shape.ToArray()), x))).ToArray();
                    var y = gpu.TensorAdd(gpu.ReduceSum(nexts[0], null, false), gpu.TensorAdd(gpu.ReduceSum(nexts[1], null, false), gpu.ReduceSum(nexts[2], null, false)));
                    plan = scope.CompileInference(y, new[] { x }, states.Zip(nexts, (st, nx) => (st, nx)).ToArray());
                }
                using (plan)
                {
                    for (int t = 0; t < 5; t++)
                    {
                        float xv = 0.25f * (t + 1);
                        plan.SetInputs(new[] { new Tensor<float>([1], new Vector<float>(new[] { xv })) });
                        plan.Execute();
                        for (int k = 0; k < states.Length; k++)
                        {
                            for (int i = 0; i < expect[k].Length; i++) expect[k][i] = expect[k][i] * 0.5f + xv;
                            var got = states[k].ToArray();
                            for (int i = 0; i < got.Length; i++)
                                Assert.True(Math.Abs(got[i] - expect[k][i]) <= 1e-5f, $"step {t} state {k}[{i}]: {got[i]} vs {expect[k][i]}");
                        }
                    }
                    var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                    Assert.True(exec.GetValue(plan) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
                    Assert.Equal(0, cuda.LastCaptureMemcpyNodeCount);
                }
            }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }

    // Ops that write every output element (RMSNorm, A*B^T with beta 0, a last-axis broadcast multiply, a gather by
    // bounds-checked host indices) take their first-pass output buffers without a zero fill: that pass is the one the
    // graph captures, so a fill would replay as a memset node every step. The replayed values match the CPU engine.
    [SkippableFact]
    public void A_captured_chain_of_fully_written_outputs_has_no_memset_nodes()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Inference graph capture is CUDA-only.");
            var cuda = (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)gpu.GetBackend()!;
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = true;
            try
            {
                const int R = 2, E = 64, N = 48, K = 20;
                var rng = new Random(11);
                Tensor<float> Rand(int[] shape) => new(shape, new Vector<float>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray()));
                var gamma = Rand([E]);
                var w = Rand([N, E]);
                var scale = Rand([N]);
                var idx = new Tensor<int>(Enumerable.Range(0, K).Select(i => (i * 7) % N).ToArray(), [K]);
                var x = new Tensor<float>([R, E]);
                var state = new Tensor<float>([R, K]);
                Tensor<float> Chain(IEngine e, Tensor<float> input)
                    => e.TensorIndexSelect(e.TensorMultiply(e.TensorMatMulTransposed(e.RMSNorm(input, gamma, 1e-6, out _), w), scale), idx, 1);
                CompiledInferencePlan<float> plan;
                using (new EngineScope(gpu))
                using (var scope = GraphMode.EnableInference())
                {
                    var y = Chain(gpu, x);
                    plan = scope.CompileInference(y, new[] { x }, new[] { (state, gpu.TensorAdd(state, y)) });
                }
                using (plan)
                {
                    var cpu = new CpuEngine();
                    var acc = new float[R * K];
                    for (int t = 0; t < 4; t++)
                    {
                        var input = Rand([R, E]);
                        plan.SetInputs(new[] { input });
                        var got = plan.Execute().ToArray();
                        var expect = Chain(cpu, input).ToArray();
                        Assert.Equal(expect.Length, got.Length);
                        for (int i = 0; i < got.Length; i++)
                        {
                            Assert.True(Math.Abs(got[i] - expect[i]) <= 1e-4f * (1f + Math.Abs(expect[i])), $"step {t} [{i}]: {got[i]} vs {expect[i]}");
                            acc[i] += expect[i];
                        }
                        var st = state.ToArray();
                        for (int i = 0; i < acc.Length; i++)
                            Assert.True(Math.Abs(st[i] - acc[i]) <= 1e-4f * (1f + Math.Abs(acc[i])), $"step {t} state[{i}]: {st[i]} vs {acc[i]}");
                    }
                    var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                    Assert.True(exec.GetValue(plan) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
                    Assert.True(cuda.LastCaptureKernelNodeCount > 0, "the capture recorded no kernels");
                    Assert.Equal(0, cuda.LastCaptureMemsetNodeCount);
                }
            }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }

    // A captured inference plan whose embedding ids are computed on the device (a decode step feeding back its own
    // choice) gathers from the float ids in one launch: ties round to even, NaN and out-of-range ids give zero rows.
    [SkippableFact]
    public void A_captured_lookup_of_device_float_ids_matches_the_host_contract()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Inference graph capture is CUDA-only.");
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = true;
            try
            {
                const int V = 6, E = 4, N = 7;
                var table = new Tensor<float>([V, E]);
                for (int i = 0; i < table.Length; i++) table[i / E, i % E] = i * 0.5f + 1f;
                var raw = new Tensor<float>([N]);
                CompiledInferencePlan<float> plan;
                using (new EngineScope(gpu))
                using (var scope = GraphMode.EnableInference())
                {
                    var ids = gpu.TensorAddScalar(raw, 0.25f);      // computed on the device
                    var rows = gpu.TensorEmbeddingLookupFromFloatIndices(table, ids);
                    plan = scope.CompileInference(rows, new[] { raw }, Array.Empty<(Tensor<float>, Tensor<float>)>());
                }
                float[][] feeds =
                [
                    [0f, 1f, 2f, 3f, 4f, 5f, 0.25f],                // 0.5 -> 0 (ties to even)
                    [2.25f, 1.25f, -1f, 6f, 9f, float.NaN, -0.75f], // 2.5 -> 2, 1.5 -> 2, -0.75, 6.25, 9.25, NaN -> zero, -0.5 -> 0
                    [4.25f, 0.75f, 5.5f, 3f, 2f, 1f, 0f],           // 4.5 -> 4, 1.0, 5.75 -> 6 invalid
                ];
                using (plan)
                {
                    long before = gpu.FusedFloatIdLookups;
                    for (int rep = 0; rep < 4; rep++)
                        foreach (var feed in feeds)
                        {
                            plan.SetInputs(new[] { new Tensor<float>([N], new Vector<float>((float[])feed.Clone())) });
                            var got = plan.Execute().ToArray();
                            for (int i = 0; i < N; i++)
                            {
                                float id = feed[i] + 0.25f;
                                double r = Math.Round((double)id, MidpointRounding.ToEven);
                                bool valid = !float.IsNaN(id) && r >= 0 && r < V;
                                for (int j = 0; j < E; j++)
                                {
                                    float expect = valid ? table[(int)r, j] : 0f;
                                    Assert.True(got[i * E + j] == expect, $"rep {rep} id {id} col {j}: {got[i * E + j]} vs {expect}");
                                }
                            }
                        }
                    var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                    Assert.True(exec.GetValue(plan) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
                    Assert.True(gpu.FusedFloatIdLookups > before, "the captured lookup did not take the fused float-id gather");
                }
            }
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
