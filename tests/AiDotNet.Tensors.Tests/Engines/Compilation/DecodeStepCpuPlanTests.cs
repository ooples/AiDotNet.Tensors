using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The decode-step ops that write their state in place (GatedDeltaNetStep, CausalConv1dStep, KvCacheAppend) on the
/// CPU engine: a compiled plan replays the state write once per execution, the eager result matches the definition, and
/// malformed arguments are rejected before anything is written.
/// </summary>
public class DecodeStepCpuPlanTests
{
    private sealed class EngineScope : IDisposable
    {
        private readonly IEngine _prior = AiDotNetEngine.Current;
        public EngineScope(IEngine engine) => AiDotNetEngine.Current = engine;
        public void Dispose() => AiDotNetEngine.Current = _prior;
    }

    private static Tensor<float> T(int[] shape, Func<int, float> f)
        => new(shape, new Vector<float>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(f).ToArray()));

    private static void Close(float[] expect, float[] got, string what)
    {
        Assert.Equal(expect.Length, got.Length);
        for (int i = 0; i < got.Length; i++)
            Assert.True(Math.Abs(got[i] - expect[i]) <= 1e-5f * (1 + Math.Abs(expect[i])), $"{what}[{i}]: {got[i]} vs {expect[i]}");
    }

    // The same step run eagerly on a second state is the reference for the plan's replay.
    [Fact]
    public void A_cpu_plan_of_the_gated_delta_step_advances_the_state_once_per_execution()
    {
        const int rows = 2, dv = 3, dk = 4;
        var cpu = new CpuEngine();
        var state = new Tensor<float>([rows, dv, dk]);
        var reference = new Tensor<float>([rows, dv, dk]);
        var q = new Tensor<float>([rows, dk]); var k = new Tensor<float>([rows, dk]); var v = new Tensor<float>([rows, dv]);
        var a = new Tensor<float>([rows]); var b = new Tensor<float>([rows]);
        CompiledInferencePlan<float> plan;
        using (new EngineScope(cpu))
        using (var scope = GraphMode.EnableInference())
        {
            var y = cpu.TensorMultiplyScalar(cpu.GatedDeltaNetStep(state, q, k, v, a, b, GatedDeltaRule.ScaledKeys), 2f);
            plan = scope.CompileInference(y, new[] { q, k, v, a, b }, Array.Empty<(Tensor<float>, Tensor<float>)>());
        }
        using (plan)
            for (int t = 0; t < 4; t++)
            {
                var x = new[]
                {
                    T([rows, dk], i => MathF.Sin(i + t)), T([rows, dk], i => 0.3f * MathF.Cos(i * 2 + t)), T([rows, dv], i => MathF.Sin(3 * i - t)),
                    T([rows], i => 0.8f + 0.05f * i), T([rows], i => 0.5f - 0.1f * i),
                };
                plan.SetInputs(x);
                var y = plan.Execute().ToArray();
                var expect = cpu.GatedDeltaNetStep(reference, x[0], x[1], x[2], x[3], x[4], GatedDeltaRule.ScaledKeys).ToArray().Select(e => 2 * e).ToArray();
                Close(expect, y, $"t{t} y");
                Close(reference.ToArray(), state.ToArray(), $"t{t} state");
            }
    }

    [Fact]
    public void A_cpu_plan_of_the_causal_convolution_step_shifts_the_history_once_per_execution()
    {
        const int K = 3, batch = 2, channels = 4;
        var cpu = new CpuEngine();
        var history = new Tensor<float>([K - 1, batch, channels]);
        var reference = new Tensor<float>([K - 1, batch, channels]);
        var w = T([channels, K], i => 0.1f * (i + 1));
        var bias = T([channels], i => 0.01f * i);
        var x = new Tensor<float>([batch, channels]);
        CompiledInferencePlan<float> plan;
        using (new EngineScope(cpu))
        using (var scope = GraphMode.EnableInference())
        {
            var y = cpu.CausalConv1dStep(history, x, w, bias, silu: true);
            plan = scope.CompileInference(y, new[] { x }, Array.Empty<(Tensor<float>, Tensor<float>)>());
        }
        using (plan)
            for (int t = 0; t < 5; t++)
            {
                var xt = T([batch, channels], i => MathF.Sin(i + 7 * t));
                plan.SetInputs(new[] { xt });
                var y = plan.Execute().ToArray();
                Close(cpu.CausalConv1dStep(reference, xt, w, bias, silu: true).ToArray(), y, $"t{t} y");
                Close(reference.ToArray(), history.ToArray(), $"t{t} history");
            }
    }

    [Fact]
    public void A_cpu_plan_of_the_cache_append_advances_the_position_once_per_execution()
    {
        const int batch = 2, capacity = 4, width = 3;
        var cpu = new CpuEngine();
        var cache = new Tensor<float>([batch, capacity, width]);
        var position = new Tensor<float>([1]);
        var x = new Tensor<float>([batch, width]);
        CompiledInferencePlan<float> plan;
        using (new EngineScope(cpu))
        using (var scope = GraphMode.EnableInference())
        {
            var weights = cpu.Softmax(cpu.KvCacheAppend(cache, position, x), -1);
            plan = scope.CompileInference(weights, new[] { x }, Array.Empty<(Tensor<float>, Tensor<float>)>());
        }
        using (plan)
            for (int t = 0; t < capacity; t++)
            {
                plan.SetInputs(new[] { T([batch, width], i => 10 * t + i + 1) });
                var w = plan.Execute().ToArray();
                for (int s = 0; s < capacity; s++) Assert.Equal(s <= t ? 1f / (t + 1) : 0f, w[s], 6);
                Assert.Equal(t + 1, position.ToArray()[0]);
                var c = cache.ToArray();
                for (int b = 0; b < batch; b++)
                    for (int j = 0; j < width; j++)
                        Assert.Equal(10 * t + b * width + j + 1, c[(b * capacity + t) * width + j]);
            }
    }

    // A stateful plan (declared state updates) that also holds an in-place decode op: compiling must not leave the op's
    // state advanced, and each execution must advance it exactly once. With realizeInTrace, an RMSNorm downstream of the
    // append computes its graph-mode value eagerly, which realizes the append while the step is still being traced.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_stateful_cpu_plan_with_a_cache_append_advances_the_position_once_per_execution(bool realizeInTrace)
    {
        const int batch = 1, capacity = 4, width = 2;
        var cpu = new CpuEngine();
        var cache = new Tensor<float>([batch, capacity, width]);
        var position = new Tensor<float>([1]);
        var counter = new Tensor<float>([1]);
        var x = new Tensor<float>([batch, width]);
        ICompiledPlan<float> plan;
        using (new EngineScope(cpu))
        {
            var models = new CompiledModelCache<float>();
            plan = models.CompileStatefulInference(new[] { x }, () =>
            {
                var weights = cpu.Softmax(cpu.KvCacheAppend(cache, position, x), -1);
                if (realizeInTrace)
                    weights = cpu.TensorAdd(weights, cpu.TensorMultiplyScalar(
                        cpu.RMSNorm(cpu.Reshape(weights, [1, capacity]), T([capacity], _ => 1f), 1e-6, out _), 0f));
                var next = cpu.TensorAddScalar(counter, 1f);
                return (weights, new[] { (counter, next) });
            });
        }
        Assert.Equal(0f, position.ToArray()[0]);
        using (plan)
            for (int t = 0; t < capacity; t++)
            {
                plan.SetInputs(new[] { T([batch, width], i => t + 1) });
                var w = plan.Execute().ToArray();
                for (int s = 0; s < capacity; s++) Assert.Equal(s <= t ? 1f / (t + 1) : 0f, w[s], 6);
                Assert.Equal(t + 1, position.ToArray()[0]);
                Assert.Equal(t + 1, counter.ToArray()[0]);
            }
    }

    // Two layers' caches in one plan: memory planning must not rebind one layer's carried state onto storage another
    // tensor of the same shape (the other layer's position) later shares, so each advances once per execution.
    [Fact]
    public void Two_caches_in_one_cpu_plan_keep_their_own_positions()
    {
        const int batch = 1, capacity = 4, width = 2;
        var cpu = new CpuEngine();
        var caches = new[] { new Tensor<float>([batch, capacity, width]), new Tensor<float>([batch, capacity, width]) };
        var positions = new[] { new Tensor<float>([1]), new Tensor<float>([1]) };
        var x = new Tensor<float>([batch, width]);
        CompiledInferencePlan<float> plan;
        using (new EngineScope(cpu))
        using (var scope = GraphMode.EnableInference())
        {
            var h = x;
            Tensor<float>? y = null;
            for (int l = 0; l < 2; l++)
            {
                var w = cpu.Softmax(cpu.KvCacheAppend(caches[l], positions[l], h), -1);                  // [capacity]
                var scale = cpu.ReduceSum(cpu.TensorMultiplyScalar(w, 1f), new[] { 0 }, keepDims: true);   // [1]
                h = cpu.TensorMultiply(cpu.TensorAddScalar(h, 1f), cpu.Reshape(scale, [1, 1]));
                y = y is null ? w : cpu.TensorAdd(y, w);
            }
            plan = scope.CompileInference(y!, new[] { x }, Array.Empty<(Tensor<float>, Tensor<float>)>());
        }
        using (plan)
            for (int t = 0; t < capacity; t++)
            {
                plan.SetInputs(new[] { T([batch, width], i => t + i) });
                var w = plan.Execute().ToArray();
                for (int s = 0; s < capacity; s++) Assert.Equal(s <= t ? 2f / (t + 1) : 0f, w[s], 5);
                Assert.Equal(t + 1, positions[0].ToArray()[0]);
                Assert.Equal(t + 1, positions[1].ToArray()[0]);
                Assert.False(ReferenceEquals(((ISignatureTensor)positions[0]).SignatureStorageIdentity,
                    ((ISignatureTensor)positions[1]).SignatureStorageIdentity));
            }
    }

    // A trace that realizes a state-writing decode op (a host read of its output) and then throws is abandoned by its
    // scope's Dispose: the trace's write must be undone there too, as compiling undoes it, so the cache, position and
    // state leave the failed trace as it found them.
    [Fact]
    public void An_abandoned_trace_restores_the_state_its_decode_ops_wrote()
    {
        var cpu = new CpuEngine();
        var cache = T([1, 3, 2], i => 0.5f * i);
        var position = T([1], _ => 1f);
        var state = T([1, 2, 2], i => 0.25f * (i + 1));
        var cacheBefore = cache.ToArray();
        var stateBefore = state.ToArray();
        var x = T([1, 2], i => 9f + i);
        var q = T([1, 2], i => 0.3f + i); var k = T([1, 2], i => 0.7f - i); var v = T([1, 2], i => 0.1f * i);
        var a = T([1], _ => 0.9f); var b = T([1], _ => 0.5f);
        using (new EngineScope(cpu))
        {
            Assert.Throws<InvalidOperationException>((Action)(() =>
            {
                using var scope = GraphMode.EnableInference();
                var w = cpu.Softmax(cpu.KvCacheAppend(cache, position, x), -1);
                var o = cpu.GatedDeltaNetStep(state, q, k, v, a, b, GatedDeltaRule.ScaledKeys);
                // Host reads realize both ops while the trace is still recording.
                Assert.Equal(3, w.ToArray().Length);
                Assert.Equal(2, o.ToArray().Length);
                Assert.Equal(2f, position.ToArray()[0]);
                throw new InvalidOperationException("the model failed after the decode step");
            }));
        }
        Assert.Equal(1f, position.ToArray()[0]);
        Assert.Equal(cacheBefore, cache.ToArray());
        Assert.Equal(stateBefore, state.ToArray());
    }

    // A trace that never realizes its decode op takes no snapshot of the op's state (a KV cache can be large).
    [Fact]
    public void A_trace_that_never_realizes_its_decode_op_copies_no_state()
    {
        var cpu = new CpuEngine();
        var cache = new Tensor<float>([1, 4, 2]);
        var position = new Tensor<float>([1]);
        var x = new Tensor<float>([1, 2]);
        using (new EngineScope(cpu))
        using (var scope = GraphMode.EnableInference())
        {
            cpu.Softmax(cpu.KvCacheAppend(cache, position, x), -1);
            var field = typeof(LazyTensorScope).GetField("_traceStateSnapshots",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
            var entries = ((System.Collections.IEnumerable)field.GetValue(scope)!).Cast<object>().ToArray();
            Assert.Equal(2, entries.Length);
            var snapshot = entries[0].GetType().GetField("Snapshot", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
            Assert.All(entries, e => Assert.Null(snapshot.GetValue(e)));
        }
    }

    [Fact]
    public void Malformed_decode_step_arguments_are_rejected()
    {
        var cpu = new CpuEngine();
        var s = new Tensor<float>([2, 3, 4]);
        Tensor<float> Z(params int[] shape) => new(shape);
        Assert.Throws<ArgumentNullException>(() => cpu.GatedDeltaNetStep(null!, Z(2, 4), Z(2, 4), Z(2, 3), Z(2), Z(2), GatedDeltaRule.ScaledKeys));
        Assert.Throws<ArgumentException>(() => cpu.GatedDeltaNetStep(Z(6, 4), Z(2, 4), Z(2, 4), Z(2, 3), Z(2), Z(2), GatedDeltaRule.ScaledKeys));
        Assert.Throws<ArgumentException>(() => cpu.GatedDeltaNetStep(s, Z(2, 5), Z(2, 4), Z(2, 3), Z(2), Z(2), GatedDeltaRule.ScaledKeys));
        Assert.Throws<ArgumentException>(() => cpu.GatedDeltaNetStep(s, Z(2, 4), Z(2, 4), Z(2, 4), Z(2), Z(2), GatedDeltaRule.ScaledKeys));
        Assert.Throws<ArgumentException>(() => cpu.GatedDeltaNetStep(s, Z(2, 4), Z(2, 4), Z(2, 3), Z(3), Z(2), GatedDeltaRule.ScaledKeys));

        var h = new Tensor<float>([2, 1, 4]);
        Assert.Throws<ArgumentException>(() => cpu.CausalConv1dStep(Z(2, 4), Z(1, 4), Z(4, 3), null, false));
        Assert.Throws<ArgumentException>(() => cpu.CausalConv1dStep(h, Z(1, 4), Z(4, 2), null, false));
        Assert.Throws<ArgumentException>(() => cpu.CausalConv1dStep(h, Z(1, 5), Z(4, 3), null, false));
        Assert.Throws<ArgumentException>(() => cpu.CausalConv1dStep(h, Z(1, 4), Z(4, 3), Z(3), false));

        var cache = new Tensor<float>([1, 2, 3]);
        Assert.Throws<ArgumentException>(() => cpu.KvCacheAppend(Z(2, 3), Z(1), Z(1, 3)));
        Assert.Throws<ArgumentException>(() => cpu.KvCacheAppend(cache, Z(2), Z(1, 3)));
        Assert.Throws<ArgumentException>(() => cpu.KvCacheAppend(cache, Z(1), Z(1, 4)));
        Assert.Throws<InvalidOperationException>(() => cpu.KvCacheAppend(cache, T([1], _ => 0.5f), Z(1, 3)));
        Assert.Throws<InvalidOperationException>(() => cpu.KvCacheAppend(cache, T([1], _ => -1f), Z(1, 3)));
        // Nothing was written by the rejected calls.
        Assert.All(cache.ToArray(), c => Assert.Equal(0f, c));
    }
}
