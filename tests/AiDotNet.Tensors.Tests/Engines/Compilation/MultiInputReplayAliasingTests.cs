using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Reproduces a multi-input compiled-inference REPLAY defect surfaced by wiring a real diffusion DiT
/// (attention + AdaLN + residuals) through <see cref="CompiledModelCache{T}.GetOrCompileInference(Tensor{T}[], Func{Tensor{T}})"/>:
/// the eager forward is deterministic, but the compiled plan's replay was NOT — consecutive Execute()
/// calls with the SAME input instances produced different outputs. The smoking gun is that the plan's
/// Execute aliases/mutates an input tensor's buffer, so a denoising loop (which reuses the same noisy
/// sample tensor across steps) gets corrupted from the second step on.
///
/// The pre-existing <c>MultiInputCompiledInferenceTests</c> does not catch this because it (a) builds a
/// fresh input each trial and (b) computes its eager oracle from the input AFTER Execute, so a mutated
/// input is compared against an equally-mutated oracle. These tests snapshot inputs BEFORE Execute and
/// reuse the same instances across calls.
/// </summary>
public class MultiInputReplayAliasingTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    public MultiInputReplayAliasingTests() { AiDotNetEngine.Current = new CpuEngine(); }
    public void Dispose() { AiDotNetEngine.Current = _priorEngine; }

    private static float[] Snapshot(Tensor<float> t)
    {
        var a = new float[t.Length];
        t.AsSpan().CopyTo(a);
        return a;
    }

    [Fact]
    public void Replay_DoesNotMutateInputs_AndIsDeterministic()
    {
        var engine = new CpuEngine();
        var x = Tensor<float>.CreateRandom(new[] { 4, 8 });
        var w1 = Tensor<float>.CreateRandom(new[] { 8, 8 });
        var w2 = Tensor<float>.CreateRandom(new[] { 8, 8 });
        var t = Tensor<float>.CreateRandom(new[] { 4, 8 });

        // A small block with a residual + second matmul — closer to a transformer block than a bare
        // matmul+add, exercising more buffer reuse in the compiled plan.
        Func<Tensor<float>> forward = () =>
        {
            var h = engine.TensorMatMul(x, w1);
            h = engine.TensorBroadcastAdd(h, t);       // inject the per-step input
            h = engine.ReLU(h);
            var o = engine.TensorMatMul(h, w2);
            return engine.TensorAdd(o, x);             // residual back to the input
        };

        var xBefore = Snapshot(x);
        var tBefore = Snapshot(t);

        var cache = new CompiledModelCache<float>();
        try
        {
            var plan = cache.GetOrCompileInference(new[] { x, t }, forward);

            // First replay.
            plan.SetInputs(new[] { x, t });
            var got1 = Snapshot(plan.Execute());

            // Inputs must be untouched by Execute.
            Assert.Equal(xBefore, x.AsSpan().ToArray());
            Assert.Equal(tBefore, t.AsSpan().ToArray());

            // Second replay with the SAME instances must reproduce the first (deterministic).
            plan.SetInputs(new[] { x, t });
            var got2 = Snapshot(plan.Execute());
            for (int i = 0; i < got1.Length; i++)
                Assert.True(Math.Abs(got1[i] - got2[i]) < 1e-5f,
                    $"replay {i}: {got1[i]} vs {got2[i]} — non-deterministic compiled replay.");

            // And the replay must match a clean eager evaluation computed from the ORIGINAL inputs.
            var hE = engine.TensorMatMul(x, w1);
            hE = engine.TensorBroadcastAdd(hE, t);
            hE = engine.ReLU(hE);
            var oE = engine.TensorMatMul(hE, w2);
            var eager = Snapshot(engine.TensorAdd(oE, x));
            for (int i = 0; i < eager.Length; i++)
                Assert.True(Math.Abs(eager[i] - got1[i]) < 1e-4f,
                    $"parity {i}: eager {eager[i]} vs compiled {got1[i]}.");
        }
        finally { cache.Dispose(); }
    }

    [Fact]
    public void Replay_DifferentRankInputs_Deterministic()
    {
        // Mirrors DiT: a 4D primary input and a low-rank secondary (timestep-embedding-like) input
        // broadcast in. Reuse the SAME instances across replays.
        var engine = new CpuEngine();
        var x = Tensor<float>.CreateRandom(new[] { 1, 4, 8, 8 });
        var t = Tensor<float>.CreateRandom(new[] { 1, 1, 1, 8 });   // broadcast over the 4D primary

        Func<Tensor<float>> forward = () =>
        {
            var h = engine.ReLU(x);
            return engine.TensorBroadcastAdd(h, t);
        };

        var xBefore = Snapshot(x);
        var cache = new CompiledModelCache<float>();
        try
        {
            var plan = cache.GetOrCompileInference(new[] { x, t }, forward);
            plan.SetInputs(new[] { x, t });
            var got1 = Snapshot(plan.Execute());
            Assert.Equal(xBefore, x.AsSpan().ToArray());
            plan.SetInputs(new[] { x, t });
            var got2 = Snapshot(plan.Execute());
            for (int i = 0; i < got1.Length; i++)
                Assert.True(Math.Abs(got1[i] - got2[i]) < 1e-5f,
                    $"diff-rank replay {i}: {got1[i]} vs {got2[i]} — non-deterministic.");
        }
        finally { cache.Dispose(); }
    }

    [Fact]
    public void Replay_ExplicitAttention_Deterministic()
    {
        // Explicit single-head attention via matmul + transpose + softmax (the path a DiT block walks
        // when not using a fused SDPA kernel). softmax is the op the simpler graphs lack.
        var engine = new CpuEngine();
        int seq = 4, dim = 8;
        var x = Tensor<float>.CreateRandom(new[] { seq, dim });
        var t = Tensor<float>.CreateRandom(new[] { seq, dim });
        var wq = Tensor<float>.CreateRandom(new[] { dim, dim });
        var wk = Tensor<float>.CreateRandom(new[] { dim, dim });
        var wv = Tensor<float>.CreateRandom(new[] { dim, dim });

        Func<Tensor<float>> forward = () =>
        {
            var q = engine.TensorMatMul(x, wq);
            var k = engine.TensorMatMul(x, wk);
            var v = engine.TensorMatMul(x, wv);
            var scores = engine.TensorMatMul(q, engine.TensorTranspose(k));   // [seq, seq]
            var probs = engine.TensorSoftmax(scores, -1);
            var ctx = engine.TensorMatMul(probs, v);                          // [seq, dim]
            return engine.TensorAdd(ctx, t);
        };

        var xBefore = Snapshot(x);
        var cache = new CompiledModelCache<float>();
        try
        {
            var plan = cache.GetOrCompileInference(new[] { x, t }, forward);
            plan.SetInputs(new[] { x, t });
            var got1 = Snapshot(plan.Execute());
            Assert.Equal(xBefore, x.AsSpan().ToArray());
            plan.SetInputs(new[] { x, t });
            var got2 = Snapshot(plan.Execute());
            for (int i = 0; i < got1.Length; i++)
                Assert.True(Math.Abs(got1[i] - got2[i]) < 1e-5f,
                    $"attn replay {i}: {got1[i]} vs {got2[i]} — non-deterministic compiled replay.");
        }
        finally { cache.Dispose(); }
    }

    [Fact]
    public void Replay_AdaLNBlock_Deterministic()
    {
        // Faithful mini-DiT block: LayerNorm modulated by a broadcast scale/shift derived from the
        // SECOND input (the timestep-embedding analogue, consumed at MULTIPLE sites), two residuals,
        // and a GELU MLP. Reuse the SAME input instances across replays.
        var engine = new CpuEngine();
        int dim = 8;
        var x = Tensor<float>.CreateRandom(new[] { 4, dim });          // [seq, dim]
        var t = Tensor<float>.CreateRandom(new[] { 1, dim });          // broadcast modulation (embed-like)
        var g = Tensor<float>.CreateRandom(new[] { dim });
        var b = Tensor<float>.CreateRandom(new[] { dim });
        var w1 = Tensor<float>.CreateRandom(new[] { dim, dim });
        var w2 = Tensor<float>.CreateRandom(new[] { dim, dim });

        Func<Tensor<float>> forward = () =>
        {
            var n1 = engine.TensorLayerNorm(x, g, b);
            var mod = engine.TensorBroadcastMultiply(n1, t);          // AdaLN scale by the 2nd input
            var attnish = engine.TensorMatMul(mod, w1);
            var res1 = engine.TensorAdd(x, attnish);                  // residual 1
            var n2 = engine.TensorLayerNorm(res1, g, b);
            var mod2 = engine.TensorBroadcastMultiply(n2, t);         // 2nd consumption of input t
            var mlp = engine.GELU(engine.TensorMatMul(mod2, w2));
            return engine.TensorAdd(res1, mlp);                       // residual 2
        };

        var xBefore = Snapshot(x);
        var tBefore = Snapshot(t);
        var cache = new CompiledModelCache<float>();
        try
        {
            var plan = cache.GetOrCompileInference(new[] { x, t }, forward);
            plan.SetInputs(new[] { x, t });
            var got1 = Snapshot(plan.Execute());
            Assert.Equal(xBefore, x.AsSpan().ToArray());
            Assert.Equal(tBefore, t.AsSpan().ToArray());
            plan.SetInputs(new[] { x, t });
            var got2 = Snapshot(plan.Execute());
            for (int i = 0; i < got1.Length; i++)
                Assert.True(Math.Abs(got1[i] - got2[i]) < 1e-5f,
                    $"adaln replay {i}: {got1[i]} vs {got2[i]} — non-deterministic compiled replay.");
        }
        finally { cache.Dispose(); }
    }
}
