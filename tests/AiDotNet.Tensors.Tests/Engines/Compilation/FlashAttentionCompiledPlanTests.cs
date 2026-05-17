using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Regression tests for AiDotNet#1346 — FlashAttention was missing its
/// GraphMode.IsActive recording branch under the fused/compiled training
/// path. Tensors PR #362 added the forward-side fix (lazy-graph
/// recording mirroring LayerNorm); these tests pin the end-to-end
/// behaviour: a compiled training plan that runs FA actually produces
/// non-zero gradients on Q/K/V inputs and the plan's "loss" reduces
/// across repeated <c>Step()</c> calls.
///
/// <para>Pre-fix path: <c>plan.Gradients[Q]</c> stayed all-zero
/// because the FA lazy node was never created, so the lazy graph went
/// straight from the upstream MatMul / Permute to the downstream
/// Permute / Reshape with FA elided into a compile-time constant. No
/// FA backward fired → no gradient flowed back to Q / K / V projections.</para>
/// </summary>
[Collection("FlashAttentionCompiledPlan")]
public class FlashAttentionCompiledPlanTests
{
    /// <summary>
    /// End-to-end: a compiled training plan whose forward includes
    /// FlashAttention actually computes a non-zero gradient on Q, K, V
    /// when <c>plan.Step()</c> runs. Pre-fix the gradient buffers for
    /// Q/K/V stayed at zero regardless of input.
    /// </summary>
    [Fact]
    public void FlashAttention_InCompiledPlan_ProducesNonZeroQKVGradients()
    {
        var engine = new CpuEngine();
        const int batch = 1, heads = 2, seqQ = 4, headDim = 8;

        // Trainable Q/K/V tensors (rank-4 [B, H, Sq, D]).
        var Q = new Tensor<float>([batch, heads, seqQ, headDim]);
        var K = new Tensor<float>([batch, heads, seqQ, headDim]);
        var V = new Tensor<float>([batch, heads, seqQ, headDim]);
        var qSpan = Q.AsWritableSpan();
        var kSpan = K.AsWritableSpan();
        var vSpan = V.AsWritableSpan();
        var rng = new System.Random(42);
        for (int i = 0; i < qSpan.Length; i++) qSpan[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < kSpan.Length; i++) kSpan[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < vSpan.Length; i++) vSpan[i] = (float)(rng.NextDouble() - 0.5);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var attn = engine.FlashAttention<float>(Q, K, V, scale: null, isCausal: false, out _);
            // Loss = sum(attn). Plan seeds dL/dOutput = ones; FA backward
            // then walks softmax statistics + Q/K/V to produce dQ/dK/dV.
            engine.ReduceSum(attn, null);
            plan = scope.CompileTraining(new[] { Q, K, V });
        }

        using (plan)
        {
            plan.Step();

            var gQ = plan.Gradients[0];
            var gK = plan.Gradients[1];
            var gV = plan.Gradients[2];

            Assert.NotNull(gQ);
            Assert.NotNull(gK);
            Assert.NotNull(gV);

            Assert.Equal(Q._shape, gQ._shape);
            Assert.Equal(K._shape, gK._shape);
            Assert.Equal(V._shape, gV._shape);

            // Sum of absolute gradient values must be > 0 — pre-fix this
            // would be exactly 0 because FA had no lazy node and so no
            // backward delegate fired in plan.Step().
            float absSumQ = AbsSum(gQ.AsSpan());
            float absSumK = AbsSum(gK.AsSpan());
            float absSumV = AbsSum(gV.AsSpan());

            Assert.True(absSumQ > 1e-6f, $"dQ should be non-zero (got abs-sum={absSumQ})");
            Assert.True(absSumK > 1e-6f, $"dK should be non-zero (got abs-sum={absSumK})");
            Assert.True(absSumV > 1e-6f, $"dV should be non-zero (got abs-sum={absSumV})");
        }
    }

    /// <summary>
    /// Training convergence: a tiny 1-layer FA + linear-readout network
    /// trained with vanilla SGD on a deterministic toy task must show
    /// loss decreasing across multiple Step() calls. Pre-fix the loss
    /// stayed constant because Q/K/V parameter updates were always
    /// applied to gradients of zero.
    /// </summary>
    [Fact]
    public void FlashAttention_InCompiledPlan_LossDecreasesAcrossSteps()
    {
        var engine = new CpuEngine();
        const int batch = 1, heads = 2, seqQ = 4, headDim = 8;

        // Trainable Q/K/V tensors. Initial values fixed for determinism.
        var Q = new Tensor<float>([batch, heads, seqQ, headDim]);
        var K = new Tensor<float>([batch, heads, seqQ, headDim]);
        var V = new Tensor<float>([batch, heads, seqQ, headDim]);
        var rng = new System.Random(1234);
        FillRandomScaled(Q, rng, 0.1f);
        FillRandomScaled(K, rng, 0.1f);
        FillRandomScaled(V, rng, 0.1f);

        // Target: a fixed tensor of the FA output shape. Loss = sum((attn-target)^2)
        // is built via Subtract → Multiply (square) → ReduceSum.
        var target = new Tensor<float>([batch, heads, seqQ, headDim]);
        var tSpan = target.AsWritableSpan();
        for (int i = 0; i < tSpan.Length; i++) tSpan[i] = 0.5f;

        // Step 1: eager-evaluate loss BEFORE training to record initial value.
        float initialLoss = ComputeLoss(engine, Q, K, V, target);

        // Step 2: compile the same forward + loss as a training plan.
        ICompiledTrainingPlan<float> plan;
        Tensor<float> lossTensor;
        using (var scope = GraphMode.Enable())
        {
            var attn = engine.FlashAttention<float>(Q, K, V, scale: null, isCausal: false, out _);
            var diff = engine.TensorSubtract(attn, target);
            var sq = engine.TensorMultiply(diff, diff);
            lossTensor = engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { Q, K, V }, lossTensor);
        }

        // Step 3: train for a few steps with vanilla SGD (lr * grad).
        using (plan)
        {
            const float lr = 0.01f;
            for (int step = 0; step < 20; step++)
            {
                plan.Step();
                ApplySgd(Q, plan.Gradients[0], lr);
                ApplySgd(K, plan.Gradients[1], lr);
                ApplySgd(V, plan.Gradients[2], lr);
            }
        }

        // Step 4: re-evaluate loss eagerly with the trained Q/K/V.
        float finalLoss = ComputeLoss(engine, Q, K, V, target);

        // Loss must strictly decrease — pre-fix loss stayed constant
        // because dQ/dK/dV were all zero.
        Assert.True(finalLoss < initialLoss,
            $"Loss should decrease through FA training. initial={initialLoss}, final={finalLoss}");
    }

    private static float ComputeLoss(IEngine engine, Tensor<float> Q, Tensor<float> K, Tensor<float> V, Tensor<float> target)
    {
        var attn = engine.FlashAttention<float>(Q, K, V, scale: null, isCausal: false, out _);
        var diff = engine.TensorSubtract(attn, target);
        var sq = engine.TensorMultiply(diff, diff);
        var loss = engine.ReduceSum(sq, null);
        return loss.AsSpan()[0];
    }

    private static void ApplySgd(Tensor<float> param, Tensor<float> grad, float lr)
    {
        var pSpan = param.AsWritableSpan();
        var gSpan = grad.AsSpan();
        for (int i = 0; i < pSpan.Length; i++)
        {
            pSpan[i] -= lr * gSpan[i];
        }
    }

    private static void FillRandomScaled(Tensor<float> t, System.Random rng, float scale)
    {
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)((rng.NextDouble() - 0.5) * 2.0 * scale);
    }

    private static float AbsSum(System.ReadOnlySpan<float> span)
    {
        float sum = 0;
        for (int i = 0; i < span.Length; i++) sum += System.Math.Abs(span[i]);
        return sum;
    }

    /// <summary>
    /// End-to-end regression mirroring the HarmonicEngine PathB consumer
    /// failure (AiDotNet#1346): a minimal transformer block built like
    /// FlashAttentionLayer&lt;T&gt;.Forward — Q/K/V projection via MatMul,
    /// Reshape to [B, H, Sq, D], Permute, then FA, then Permute back +
    /// Reshape + linear readout. Trains 30 SGD steps on a deterministic
    /// vocab-prediction toy task and asserts: (a) loss decreases, (b)
    /// every projection matrix actually changes its values across
    /// training. Pre-fix #1346 the FA op was elided into a compile-time
    /// constant, so the Q/K/V matmul gradients were zero and the
    /// projection matrices stayed at random init forever.
    /// </summary>
    [Fact]
    public void FlashAttentionLayerLikeBlock_InCompiledPlan_TrainingReducesLoss()
    {
        var engine = new CpuEngine();
        const int batch = 1;
        const int seqLen = 16;
        const int dModel = 32;
        const int headCount = 2;
        const int headDim = dModel / headCount;

        var input = new Tensor<float>([batch, seqLen, dModel]);
        var rng = new System.Random(7);
        FillRandomScaled(input, rng, 1.0f);

        // Target with shape matching projected output [batch, seqLen, dModel].
        var target = new Tensor<float>([batch, seqLen, dModel]);
        var tSpan = target.AsWritableSpan();
        for (int i = 0; i < tSpan.Length; i++)
            tSpan[i] = (float)System.Math.Sin(i * 0.1);

        // Projection weights (mirrors FlashAttentionLayer<T>'s Q/K/V/O).
        var wQ = NewWeight([dModel, dModel], rng);
        var wK = NewWeight([dModel, dModel], rng);
        var wV = NewWeight([dModel, dModel], rng);
        var wO = NewWeight([dModel, dModel], rng);

        // Snapshot initial weights for the "weights actually changed" check.
        var wQ0 = wQ.AsSpan().ToArray();
        var wK0 = wK.AsSpan().ToArray();
        var wV0 = wV.AsSpan().ToArray();
        var wO0 = wO.AsSpan().ToArray();

        // Eager initial loss.
        float initialLoss = ComputeBlockLoss(engine, input, wQ, wK, wV, wO, target, batch, seqLen, dModel, headCount, headDim);

        ICompiledTrainingPlan<float> plan;
        Tensor<float> lossTensor;
        using (var scope = GraphMode.Enable())
        {
            // Q/K/V projections
            var q = engine.TensorMatMul(input, wQ);
            var k = engine.TensorMatMul(input, wK);
            var v = engine.TensorMatMul(input, wV);

            // Reshape to [B, Sq, H, D] then Permute to [B, H, Sq, D]
            q = engine.TensorPermute(engine.Reshape(q, new[] { batch, seqLen, headCount, headDim }), new[] { 0, 2, 1, 3 });
            k = engine.TensorPermute(engine.Reshape(k, new[] { batch, seqLen, headCount, headDim }), new[] { 0, 2, 1, 3 });
            v = engine.TensorPermute(engine.Reshape(v, new[] { batch, seqLen, headCount, headDim }), new[] { 0, 2, 1, 3 });

            // Flash Attention
            var attn = engine.FlashAttention<float>(q, k, v, scale: null, isCausal: false, out _);

            // Permute back + Reshape to [B, Sq, dModel]
            var permuted = engine.TensorPermute(attn, new[] { 0, 2, 1, 3 });
            var reshaped = engine.Reshape(permuted, new[] { batch, seqLen, dModel });

            // Output projection
            var projected = engine.TensorMatMul(reshaped, wO);

            // Loss = sum((projected - target)^2)
            var diff = engine.TensorSubtract(projected, target);
            var sq = engine.TensorMultiply(diff, diff);
            lossTensor = engine.ReduceSum(sq, null);

            plan = scope.CompileTraining(new[] { wQ, wK, wV, wO }, lossTensor);
        }

        using (plan)
        {
            const float lr = 1e-4f;
            for (int step = 0; step < 30; step++)
            {
                plan.Step();
                ApplySgd(wQ, plan.Gradients[0], lr);
                ApplySgd(wK, plan.Gradients[1], lr);
                ApplySgd(wV, plan.Gradients[2], lr);
                ApplySgd(wO, plan.Gradients[3], lr);
            }
        }

        float finalLoss = ComputeBlockLoss(engine, input, wQ, wK, wV, wO, target, batch, seqLen, dModel, headCount, headDim);

        // Loss must strictly decrease.
        Assert.True(finalLoss < initialLoss,
            $"Loss should decrease through FA+projections training. initial={initialLoss}, final={finalLoss}");

        // Every projection weight matrix must have actually changed.
        Assert.True(MaxAbsDelta(wQ, wQ0) > 1e-6f, "wQ must change during training (was: gradient-free in #1346)");
        Assert.True(MaxAbsDelta(wK, wK0) > 1e-6f, "wK must change during training (was: gradient-free in #1346)");
        Assert.True(MaxAbsDelta(wV, wV0) > 1e-6f, "wV must change during training (was: gradient-free in #1346)");
        Assert.True(MaxAbsDelta(wO, wO0) > 1e-6f, "wO must change during training");
    }

    private static float ComputeBlockLoss(
        IEngine engine, Tensor<float> input,
        Tensor<float> wQ, Tensor<float> wK, Tensor<float> wV, Tensor<float> wO,
        Tensor<float> target, int batch, int seqLen, int dModel, int headCount, int headDim)
    {
        var q = engine.TensorMatMul(input, wQ);
        var k = engine.TensorMatMul(input, wK);
        var v = engine.TensorMatMul(input, wV);
        q = engine.TensorPermute(engine.Reshape(q, new[] { batch, seqLen, headCount, headDim }), new[] { 0, 2, 1, 3 });
        k = engine.TensorPermute(engine.Reshape(k, new[] { batch, seqLen, headCount, headDim }), new[] { 0, 2, 1, 3 });
        v = engine.TensorPermute(engine.Reshape(v, new[] { batch, seqLen, headCount, headDim }), new[] { 0, 2, 1, 3 });
        var attn = engine.FlashAttention<float>(q, k, v, scale: null, isCausal: false, out _);
        var permuted = engine.TensorPermute(attn, new[] { 0, 2, 1, 3 });
        var reshaped = engine.Reshape(permuted, new[] { batch, seqLen, dModel });
        var projected = engine.TensorMatMul(reshaped, wO);
        var diff = engine.TensorSubtract(projected, target);
        var sq = engine.TensorMultiply(diff, diff);
        var loss = engine.ReduceSum(sq, null);
        return loss.AsSpan()[0];
    }

    private static Tensor<float> NewWeight(int[] shape, System.Random rng)
    {
        var t = new Tensor<float>(shape);
        int fanIn = shape[0];
        int fanOut = shape[1];
        float scale = (float)System.Math.Sqrt(2.0 / (fanIn + fanOut));
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)((rng.NextDouble() - 0.5) * 2.0 * scale);
        return t;
    }

    private static float MaxAbsDelta(Tensor<float> current, float[] initial)
    {
        var c = current.AsSpan();
        float max = 0;
        for (int i = 0; i < c.Length; i++)
        {
            float d = System.Math.Abs(c[i] - initial[i]);
            if (d > max) max = d;
        }
        return max;
    }
}
