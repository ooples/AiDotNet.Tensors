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
    public void Replay_PatchifyReshapeOfPermute_Deterministic()
    {
        // DiT.Patchify: [B,C,H,W] -> 6D reshape -> permute {0,2,4,1,3,5} -> reshape to [B,numPatches,
        // patchDim]. The final reshape collapses a NON-contiguous permuted view — a pattern none of the
        // other repros hit (they permute 4D then feed an op that handles strides; this RESHAPES the
        // strided permute directly). Reuse the same image input across replays.
        var engine = new CpuEngine();
        int B = 1, C = 4, H = 8, W = 8, p = 2;
        int nH = H / p, nW = W / p, numPatches = nH * nW, patchDim = C * p * p;
        var x = Tensor<float>.CreateRandom(new[] { B, C, H, W });
        var t = Tensor<float>.CreateRandom(new[] { B, numPatches, patchDim });

        Func<Tensor<float>> forward = () =>
        {
            var split = engine.Reshape(x, new[] { B, C, nH, p, nW, p });
            var permuted = engine.TensorPermute(split, new[] { 0, 2, 4, 1, 3, 5 });
            var patches = engine.Reshape(permuted, new[] { B, numPatches, patchDim });
            return engine.TensorAdd(patches, t);
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
                    $"patchify replay {i}: {got1[i]} vs {got2[i]} — non-deterministic compiled replay.");
        }
        finally { cache.Dispose(); }
    }

    [Fact]
    public void Replay_AdaLNSliceAxisViews_Deterministic()
    {
        // The exact DiT-block AdaLN pattern: a modulation tensor reshaped to [B,6,1,hidden] and split
        // into six [B,1,hidden] VIEWS via Engine.TensorSliceAxis (shared backing buffer), then used as
        // scale/shift in y = x*(1+scale)+shift plus a gated residual. None of the other repros use
        // TensorSliceAxis. The second input (the modulation source) is reused across replays.
        var engine = new CpuEngine();
        int seq = 4, hidden = 8;
        var x = Tensor<float>.CreateRandom(new[] { 1, seq, hidden });
        var mod = Tensor<float>.CreateRandom(new[] { 1, 6 * hidden });    // AdaLN modulation source

        Func<Tensor<float>> forward = () =>
        {
            var modReshaped = engine.Reshape(mod, new[] { 1, 6, 1, hidden });
            var shift1 = engine.TensorSliceAxis(modReshaped, 1, 0);
            var scale1 = engine.TensorSliceAxis(modReshaped, 1, 1);
            var gate1  = engine.TensorSliceAxis(modReshaped, 1, 2);
            // y = x*(1+scale)+shift  (ApplyAdaLN)
            var scaled = engine.TensorBroadcastMultiply(x, engine.TensorAddScalar(scale1, 1f));
            var normed = engine.TensorBroadcastAdd(scaled, shift1);
            // gated residual: x + gate*normed
            var gated = engine.TensorBroadcastMultiply(normed, gate1);
            return engine.TensorAdd(x, gated);
        };

        var xBefore = Snapshot(x);
        var modBefore = Snapshot(mod);
        var cache = new CompiledModelCache<float>();
        try
        {
            var plan = cache.GetOrCompileInference(new[] { x, mod }, forward);
            plan.SetInputs(new[] { x, mod });
            var got1 = Snapshot(plan.Execute());
            Assert.Equal(xBefore, x.AsSpan().ToArray());
            Assert.Equal(modBefore, mod.AsSpan().ToArray());
            plan.SetInputs(new[] { x, mod });
            var got2 = Snapshot(plan.Execute());
            for (int i = 0; i < got1.Length; i++)
                Assert.True(Math.Abs(got1[i] - got2[i]) < 1e-5f,
                    $"adaln-slice replay {i}: {got1[i]} vs {got2[i]} — non-deterministic compiled replay.");
        }
        finally { cache.Dispose(); }
    }

    [Fact]
    public void Replay_MultiBlockMiniDiT_Deterministic()
    {
        // A deep mini-DiT: 4 stacked blocks of (AdaLN-by-input-2 -> self-attention -> residual ->
        // AdaLN -> GELU MLP -> residual). Stresses the compiled plan's buffer pool the way the real
        // multi-block DiT does, which the single-block repros do not. Reuse the SAME input instance.
        var engine = new CpuEngine();
        int seq = 4, embed = 8, heads = 2, headDim = embed / heads;
        const int blocks = 4;
        var x0 = Tensor<float>.CreateRandom(new[] { seq, embed });
        var t = Tensor<float>.CreateRandom(new[] { 1, embed });            // modulation, consumed every block
        var g = Tensor<float>.CreateRandom(new[] { embed });
        var b = Tensor<float>.CreateRandom(new[] { embed });
        var wq = new Tensor<float>[blocks]; var wk = new Tensor<float>[blocks];
        var wv = new Tensor<float>[blocks]; var wm = new Tensor<float>[blocks];
        for (int i = 0; i < blocks; i++)
        {
            wq[i] = Tensor<float>.CreateRandom(new[] { embed, embed });
            wk[i] = Tensor<float>.CreateRandom(new[] { embed, embed });
            wv[i] = Tensor<float>.CreateRandom(new[] { embed, embed });
            wm[i] = Tensor<float>.CreateRandom(new[] { embed, embed });
        }

        Func<Tensor<float>> forward = () =>
        {
            var x = (Tensor<float>)x0;
            for (int i = 0; i < blocks; i++)
            {
                var n1 = engine.TensorBroadcastMultiply(engine.TensorLayerNorm(x, g, b), t);
                var q = engine.TensorPermute(engine.Reshape(engine.TensorMatMul(n1, wq[i]), new[] { 1, seq, heads, headDim }), new[] { 0, 2, 1, 3 });
                var k = engine.TensorPermute(engine.Reshape(engine.TensorMatMul(n1, wk[i]), new[] { 1, seq, heads, headDim }), new[] { 0, 2, 1, 3 });
                var v = engine.TensorPermute(engine.Reshape(engine.TensorMatMul(n1, wv[i]), new[] { 1, seq, heads, headDim }), new[] { 0, 2, 1, 3 });
                var attn = engine.ScaledDotProductAttention(q, k, v, null, 1.0 / Math.Sqrt(headDim), out _);
                var attnFlat = engine.Reshape(engine.TensorPermute(attn, new[] { 0, 2, 1, 3 }), new[] { seq, embed });
                x = engine.TensorAdd(x, attnFlat);
                var n2 = engine.TensorBroadcastMultiply(engine.TensorLayerNorm(x, g, b), t);
                var mlp = engine.GELU(engine.TensorMatMul(n2, wm[i]));
                x = engine.TensorAdd(x, mlp);
            }
            return x;
        };

        var xBefore = Snapshot(x0);
        var cache = new CompiledModelCache<float>();
        try
        {
            var plan = cache.GetOrCompileInference(new[] { x0, t }, forward);
            plan.SetInputs(new[] { x0, t });
            var got1 = Snapshot(plan.Execute());
            Assert.Equal(xBefore, x0.AsSpan().ToArray());
            plan.SetInputs(new[] { x0, t });
            var got2 = Snapshot(plan.Execute());
            for (int i = 0; i < got1.Length; i++)
                Assert.True(Math.Abs(got1[i] - got2[i]) < 1e-5f,
                    $"minidit replay {i}: {got1[i]} vs {got2[i]} — non-deterministic compiled replay.");
        }
        finally { cache.Dispose(); }
    }

    [Fact]
    public void Replay_SelfAttentionLayerOps_Deterministic()
    {
        // Faithful mirror of SelfAttentionLayer.Forward's op sequence: QKV matmul -> reshape to heads
        // -> permute -> 4D ScaledDotProductAttention (with the secondary `out attentionWeights`) ->
        // permute back -> reshape -> add the second input. This is the exact path a DiT block walks,
        // and the one the simpler repros lack. Reuse the SAME input instance across replays.
        var engine = new CpuEngine();
        int seq = 4, embed = 8, heads = 2, headDim = embed / heads;
        var input2D = Tensor<float>.CreateRandom(new[] { seq, embed });
        var t = Tensor<float>.CreateRandom(new[] { 1, seq, embed });
        var wq = Tensor<float>.CreateRandom(new[] { embed, embed });
        var wk = Tensor<float>.CreateRandom(new[] { embed, embed });
        var wv = Tensor<float>.CreateRandom(new[] { embed, embed });

        Func<Tensor<float>> forward = () =>
        {
            var q = engine.Reshape(engine.TensorMatMul(input2D, wq), new[] { 1, seq, heads, headDim });
            var k = engine.Reshape(engine.TensorMatMul(input2D, wk), new[] { 1, seq, heads, headDim });
            var v = engine.Reshape(engine.TensorMatMul(input2D, wv), new[] { 1, seq, heads, headDim });
            var qh = engine.TensorPermute(q, new[] { 0, 2, 1, 3 });
            var kh = engine.TensorPermute(k, new[] { 0, 2, 1, 3 });
            var vh = engine.TensorPermute(v, new[] { 0, 2, 1, 3 });
            var attn = engine.ScaledDotProductAttention(qh, kh, vh, null, 1.0 / Math.Sqrt(headDim), out _);
            var back = engine.TensorPermute(attn, new[] { 0, 2, 1, 3 });
            var flat = engine.Reshape(back, new[] { 1, seq, embed });
            return engine.TensorBroadcastAdd(flat, t);
        };

        var inputBefore = Snapshot(input2D);
        var cache = new CompiledModelCache<float>();
        try
        {
            var plan = cache.GetOrCompileInference(new[] { input2D, t }, forward);
            plan.SetInputs(new[] { input2D, t });
            var got1 = Snapshot(plan.Execute());
            Assert.Equal(inputBefore, input2D.AsSpan().ToArray());
            plan.SetInputs(new[] { input2D, t });
            var got2 = Snapshot(plan.Execute());
            for (int i = 0; i < got1.Length; i++)
                Assert.True(Math.Abs(got1[i] - got2[i]) < 1e-5f,
                    $"selfattn replay {i}: {got1[i]} vs {got2[i]} — non-deterministic compiled replay.");
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
