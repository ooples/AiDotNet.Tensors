using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Regression for issue #257 — tape-recorded gradients for parameters that
/// participate via TensorMatMul → Reshape → Permute → BatchMatMul (the
/// MultiHeadAttention Q/K/V pattern) and TensorEmbeddingLookup (the
/// embedding pattern) must appear in the dictionary returned by
/// <see cref="GradientTape{T}.ComputeGradients"/>, keyed by the original
/// parameter reference. Pre-fix the embedding gradient and 3 of 4 attention
/// weight gradients were silently dropped, causing models to converge to
/// uniform output across distinct token inputs.
/// </summary>
public class Issue257TapeRefIdentityTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    /// <summary>
    /// Mimics the Transformer Q/K/V flow: a parameter weight goes through
    /// TensorMatMul → Reshape → Permute → BatchMatMul before reaching the
    /// loss. The recorded MatMul backward must produce a gradient keyed by
    /// the original parameter reference, not a transient view.
    /// </summary>
    [Fact]
    public void MatMul_FollowedByReshapeAndPermute_RoutesGradToOriginalWeight()
    {
        // Transformer-shape: [batch=1, seq=4, dim=16], heads=2, headDim=8.
        const int B = 1, S = 4, D = 16, H = 2, Dh = D / H;

        var input = new Tensor<float>([B * S, D]);
        var weight = new Tensor<float>([D, D]);
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < weight.Length; i++) weight.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);

        using var tape = new GradientTape<float>();
        // Mirror MultiHeadAttention.ApplyLinearTransformation:
        //   Q_flat   = input @ weight                        -> [B*S, D]
        //   Q_resh   = reshape(Q_flat, [B, S, H, Dh])
        //   Q_perm   = permute(Q_resh, [0, 2, 1, 3])         -> [B, H, S, Dh]
        var qFlat = _engine.TensorMatMul(input, weight);
        var qResh = _engine.Reshape(qFlat, new[] { B, S, H, Dh });
        var qPerm = _engine.TensorPermute(qResh, new[] { 0, 2, 1, 3 });

        // Use Q_perm in a downstream batched matmul (mimics attention scoring).
        // For simplicity, take its sum-of-squares — not the attention math, but
        // it exercises the same gradient chain back through Permute → Reshape →
        // MatMul → weight that the issue reproducer hits.
        var sq = _engine.TensorMultiply(qPerm, qPerm);
        var loss = _engine.ReduceSum(sq, null);

        var grads = tape.ComputeGradients(loss, sources: new[] { weight });
        Assert.True(grads.ContainsKey(weight),
            "Pre-fix: weight gradient was silently dropped because the recorded MatMul " +
            "input ref didn't match the user-facing parameter after the Reshape→Permute chain.");
        Assert.Equal(weight._shape, grads[weight]._shape);

        // dL/dweight = 2 * input^T @ Q_flat (since loss = sum(Q_flat^2) is the same
        // function as sum(Q_perm^2) — just permutes elements). Verify non-zero.
        var g = grads[weight].AsSpan();
        bool anyNonZero = false;
        for (int i = 0; i < g.Length; i++) if (Math.Abs(g[i]) > 1e-6f) { anyNonZero = true; break; }
        Assert.True(anyNonZero, "Weight gradient must be non-zero — input + weight are non-zero.");
    }

    /// <summary>
    /// Mimics the embedding flow: TensorEmbeddingLookup → Reshape →
    /// downstream loss. Pre-fix dL/dE was silently zero even though
    /// TensorEmbeddingLookup recorded on the tape, because the recorded
    /// reference vs. user-facing reference diverged.
    /// </summary>
    [Fact]
    public void EmbeddingLookup_FollowedByReshape_RoutesGradToEmbeddingTable()
    {
        const int V = 8, D = 16, B = 1, S = 4;
        var E = new Tensor<float>([V, D]);
        for (int i = 0; i < E.Length; i++) E.AsWritableSpan()[i] = 0.01f * (i + 1);

        var indices = new Tensor<int>([B * S]);
        for (int i = 0; i < B * S; i++) indices.AsWritableSpan()[i] = i % V;

        using var tape = new GradientTape<float>();
        // Embedding lookup -> reshape into [B, S, D] (the parent Transformer's pattern).
        var embOut = _engine.TensorEmbeddingLookup<float, int>(E, indices); // [B*S, D]
        var reshaped = _engine.Reshape(embOut, new[] { B, S, D });
        var loss = _engine.ReduceSum(reshaped, null);

        var grads = tape.ComputeGradients(loss, sources: new[] { E });
        Assert.True(grads.ContainsKey(E),
            "Pre-fix: dL/dE was silently dropped — TensorEmbeddingLookup recorded " +
            "but the gradient never reached the user-facing embedding table reference.");
        Assert.Equal(E._shape, grads[E]._shape);
    }

    /// <summary>
    /// The full pattern from the issue's repro: 4 distinct trainable weights
    /// (Q/K/V/O) each at the same shape [16, 16], used in MatMul. All four
    /// must appear in grads — the issue says only 1 of 4 did pre-fix.
    /// </summary>
    [Fact]
    public void FourMatMulWeights_SameShape_AllFourGetGradients()
    {
        const int N = 16;
        var input = new Tensor<float>([4, N]);
        var Wq = new Tensor<float>([N, N]);
        var Wk = new Tensor<float>([N, N]);
        var Wv = new Tensor<float>([N, N]);
        var Wo = new Tensor<float>([N, N]);

        var rng = new Random(7);
        for (int i = 0; i < input.Length; i++) input.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);
        foreach (var w in new[] { Wq, Wk, Wv, Wo })
            for (int i = 0; i < w.Length; i++) w.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);

        using var tape = new GradientTape<float>();
        // Three of the four (Q/K/V) go through Reshape+Permute before being used.
        // The fourth (O) is used directly in MatMul. The issue says Q/K/V are
        // the ones that lose gradients — make sure all four show up.
        var q = _engine.TensorMatMul(input, Wq);
        var k = _engine.TensorMatMul(input, Wk);
        var v = _engine.TensorMatMul(input, Wv);

        var qR = _engine.Reshape(q, new[] { 4, 2, 8 });
        var kR = _engine.Reshape(k, new[] { 4, 2, 8 });
        var vR = _engine.Reshape(v, new[] { 4, 2, 8 });

        var qP = _engine.TensorPermute(qR, new[] { 1, 0, 2 });
        var kP = _engine.TensorPermute(kR, new[] { 1, 0, 2 });
        var vP = _engine.TensorPermute(vR, new[] { 1, 0, 2 });

        // Combine q/k/v via element-wise multiply + sum to feed into the output projection.
        var qk = _engine.TensorMultiply(qP, kP);
        var qkv = _engine.TensorMultiply(qk, vP);
        var qkvFlat = _engine.Reshape(qkv, new[] { 4, N });

        // Output projection — Wo participates directly (mirroring _outputWeights).
        var output = _engine.TensorMatMul(qkvFlat, Wo);
        var loss = _engine.ReduceSum(output, null);

        var grads = tape.ComputeGradients(loss, sources: new[] { Wq, Wk, Wv, Wo });

        Assert.True(grads.ContainsKey(Wq), "Wq grad was dropped — same defect as #257.");
        Assert.True(grads.ContainsKey(Wk), "Wk grad was dropped — same defect as #257.");
        Assert.True(grads.ContainsKey(Wv), "Wv grad was dropped — same defect as #257.");
        Assert.True(grads.ContainsKey(Wo), "Wo grad was dropped — same defect as #257.");
    }

    /// <summary>
    /// Closer to the actual Transformer multi-head attention path: uses
    /// TensorMatMulBatched on the [B, H, S, Dh] permuted Q·K^T form, plus
    /// TensorBroadcastAdd for bias, plus a second ComputeGradients call to
    /// catch any tape-state leakage that surfaces only on iteration 2+.
    /// </summary>
    [Fact]
    public void FullMHAFlow_TwoIterations_AllParamsGetGradientsBothTimes()
    {
        const int B = 1, S = 4, D = 16, H = 2, Dh = D / H, V = 8;

        var E = new Tensor<float>([V, D]);
        var Wq = new Tensor<float>([D, D]);
        var Wk = new Tensor<float>([D, D]);
        var Wv = new Tensor<float>([D, D]);
        var Wo = new Tensor<float>([D, D]);
        var bo = new Tensor<float>([D]);

        var rng = new Random(101);
        foreach (var t in new[] { E, Wq, Wk, Wv, Wo })
            for (int i = 0; i < t.Length; i++) t.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < bo.Length; i++) bo.AsWritableSpan()[i] = 0.01f;

        var indices = new Tensor<int>([B * S]);
        for (int i = 0; i < B * S; i++) indices.AsWritableSpan()[i] = i % V;

        var sources = new[] { E, Wq, Wk, Wv, Wo, bo };

        for (int iter = 0; iter < 2; iter++)
        {
            using var tape = new GradientTape<float>();

            // 1. Embedding -> [B*S, D]
            var emb = _engine.TensorEmbeddingLookup<float, int>(E, indices);

            // 2. Q/K/V projections
            var q = _engine.TensorMatMul(emb, Wq);
            var k = _engine.TensorMatMul(emb, Wk);
            var v = _engine.TensorMatMul(emb, Wv);

            // 3. Multi-head split: [B*S, D] -> [B, S, H, Dh] -> [B, H, S, Dh]
            var qR = _engine.Reshape(q, new[] { B, S, H, Dh });
            var kR = _engine.Reshape(k, new[] { B, S, H, Dh });
            var vR = _engine.Reshape(v, new[] { B, S, H, Dh });
            var qP = _engine.TensorPermute(qR, new[] { 0, 2, 1, 3 });
            var kP = _engine.TensorPermute(kR, new[] { 0, 2, 1, 3 });
            var vP = _engine.TensorPermute(vR, new[] { 0, 2, 1, 3 });

            // 4. Attention scores + softmax (use TensorMatMul on rank-4 = batched matmul under the hood)
            var kPt = _engine.TensorPermute(kP, new[] { 0, 1, 3, 2 });
            var scores = _engine.TensorMatMul(qP, kPt);
            var attn = _engine.TensorSoftmax(scores, -1);

            // 5. Attended values: scores · V -> [B, H, S, Dh]
            var ctx = _engine.TensorMatMul(attn, vP);

            // 6. Merge heads: [B, H, S, Dh] -> [B, S, H, Dh] -> [B, S, D] -> [B*S, D]
            var ctxP = _engine.TensorPermute(ctx, new[] { 0, 2, 1, 3 });
            var ctxFlat = _engine.Reshape(ctxP, new[] { B * S, D });

            // 7. Output projection + bias
            var outProj = _engine.TensorMatMul(ctxFlat, Wo);
            var withBias = _engine.TensorBroadcastAdd(outProj, bo);

            // 8. Loss
            var loss = _engine.ReduceSum(withBias, null);

            var grads = tape.ComputeGradients(loss, sources: sources);

            Assert.True(grads.ContainsKey(E),  $"E grad missing at iter {iter}");
            Assert.True(grads.ContainsKey(Wq), $"Wq grad missing at iter {iter}");
            Assert.True(grads.ContainsKey(Wk), $"Wk grad missing at iter {iter}");
            Assert.True(grads.ContainsKey(Wv), $"Wv grad missing at iter {iter}");
            Assert.True(grads.ContainsKey(Wo), $"Wo grad missing at iter {iter}");
            Assert.True(grads.ContainsKey(bo), $"bo grad missing at iter {iter}");

            Assert.Equal(E._shape, grads[E]._shape);
            Assert.Equal(Wq._shape, grads[Wq]._shape);
            Assert.Equal(Wk._shape, grads[Wk]._shape);
            Assert.Equal(Wv._shape, grads[Wv]._shape);
            Assert.Equal(Wo._shape, grads[Wo]._shape);
            Assert.Equal(bo._shape, grads[bo]._shape);
        }
    }

    /// <summary>
    /// Same as above but without explicit sources — the caller asks for ALL
    /// gradients via <c>ComputeGradients(loss, sources: null)</c> and then
    /// looks each parameter up by reference. This is the actual code path
    /// the parent #257 reproducer uses (NeuralNetworkBase.TrainWithTape
    /// calls allGrads = tape.ComputeGradients(lossTensor, sources: null)).
    /// </summary>
    [Fact]
    public void FullMHAFlow_NullSources_AllParamsAppearInDictByReference()
    {
        const int B = 1, S = 4, D = 16, H = 2, Dh = D / H, V = 8;

        var E = new Tensor<float>([V, D]);
        var Wq = new Tensor<float>([D, D]);
        var Wk = new Tensor<float>([D, D]);
        var Wv = new Tensor<float>([D, D]);
        var Wo = new Tensor<float>([D, D]);

        var rng = new Random(202);
        foreach (var t in new[] { E, Wq, Wk, Wv, Wo })
            for (int i = 0; i < t.Length; i++) t.AsWritableSpan()[i] = (float)(rng.NextDouble() - 0.5);

        var indices = new Tensor<int>([B * S]);
        for (int i = 0; i < B * S; i++) indices.AsWritableSpan()[i] = i % V;

        using var tape = new GradientTape<float>();
        var emb = _engine.TensorEmbeddingLookup<float, int>(E, indices);
        var q = _engine.TensorMatMul(emb, Wq);
        var k = _engine.TensorMatMul(emb, Wk);
        var v = _engine.TensorMatMul(emb, Wv);
        var qR = _engine.Reshape(q, new[] { B, S, H, Dh });
        var kR = _engine.Reshape(k, new[] { B, S, H, Dh });
        var vR = _engine.Reshape(v, new[] { B, S, H, Dh });
        var qP = _engine.TensorPermute(qR, new[] { 0, 2, 1, 3 });
        var kP = _engine.TensorPermute(kR, new[] { 0, 2, 1, 3 });
        var vP = _engine.TensorPermute(vR, new[] { 0, 2, 1, 3 });
        var kPt = _engine.TensorPermute(kP, new[] { 0, 1, 3, 2 });
        var scores = _engine.TensorMatMul(qP, kPt);
        var attn = _engine.TensorSoftmax(scores, -1);
        var ctx = _engine.TensorMatMul(attn, vP);
        var ctxP = _engine.TensorPermute(ctx, new[] { 0, 2, 1, 3 });
        var ctxFlat = _engine.Reshape(ctxP, new[] { B * S, D });
        var outProj = _engine.TensorMatMul(ctxFlat, Wo);
        var loss = _engine.ReduceSum(outProj, null);

        // sources: null — the tape returns the full gradient dictionary.
        // Caller looks up each param by reference.
        var grads = tape.ComputeGradients(loss, sources: null);

        Assert.True(grads.ContainsKey(E),  "E grad missing under null-sources lookup (parent #257 surface).");
        Assert.True(grads.ContainsKey(Wq), "Wq grad missing under null-sources lookup.");
        Assert.True(grads.ContainsKey(Wk), "Wk grad missing under null-sources lookup.");
        Assert.True(grads.ContainsKey(Wv), "Wv grad missing under null-sources lookup.");
        Assert.True(grads.ContainsKey(Wo), "Wo grad missing under null-sources lookup.");
    }
}
