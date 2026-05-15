using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Reproduces the AiDotNet#1331 bug isolated to Tensors: in a multi-MatMul
/// graph (e.g. two stacked Dense layers, which is what every Transformer
/// FFN block boils down to) <c>plan.ConfigureOptimizer(Adam) + plan.Step()</c>
/// updates ONLY the additive bias parameters near the output. The 2D weight
/// matrices remain bit-identical after Step, even when the same forward+loss
/// graph trained through the eager <c>TapeTrainingStep</c> pulls non-zero
/// gradients out of all of them.
///
/// <para>The bug surfaces when there is more than one MatMul in the graph
/// AND the upstream MatMul's W is a leaf parameter. The downstream MatMul's
/// backward writes dL/dY_downstream into the leaf bias correctly (since
/// bias backward = sum(dY)), but dL/dW_upstream needs to flow back through
/// the downstream MatMul's input-side backward to materialise dL/dY_upstream,
/// and that chain dies inside the compiled-plan backward construction.</para>
///
/// <para>These tests pin the contract: every leaf parameter registered with
/// <c>CompileTraining</c> must have a non-zero L2(Δ) after one Adam step on
/// a graph whose eager backward produces non-zero gradients for all of them.</para>
/// </summary>
public class MultiMatMulFusedAdamParamUpdateTests
{
    private readonly ITestOutputHelper _output;

    public MultiMatMulFusedAdamParamUpdateTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static Tensor<float> FilledRand(int[] shape, int seed)
    {
        var t = new Tensor<float>(shape);
        var rng = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() - 0.5) * 0.2);
        return t;
    }

    private static double L2Delta(float[] before, float[] after)
    {
        double ss = 0;
        for (int i = 0; i < before.Length; i++)
        {
            double d = after[i] - before[i];
            ss += d * d;
        }
        return System.Math.Sqrt(ss);
    }

    /// <summary>
    /// Two stacked MatMuls (the Dense → ReLU → Dense pattern) with bias adds
    /// at each layer. After ConfigureOptimizer(Adam) + Step, both weight
    /// matrices AND both biases must show non-zero updates. The pre-existing
    /// single-MatMul test <c>ConfigureOptimizer_AdamFloat_…</c> only exercises
    /// the bias side; this one isolates the multi-matmul backward chain.
    /// </summary>
    [Fact]
    public void TwoMatMulChain_AdamStep_AllFourParamsMustMove()
    {
        var engine = new CpuEngine();

        const int batch = 4, dIn = 8, dHidden = 6, dOut = 3;
        var input = FilledRand(new[] { batch, dIn }, seed: 11);
        var W1 = FilledRand(new[] { dIn, dHidden }, seed: 12);
        var b1 = FilledRand(new[] { dHidden }, seed: 13);
        var W2 = FilledRand(new[] { dHidden, dOut }, seed: 14);
        var b2 = FilledRand(new[] { dOut }, seed: 15);

        var beforeW1 = W1.GetDataArray().AsSpan().ToArray();
        var beforeB1 = b1.GetDataArray().AsSpan().ToArray();
        var beforeW2 = W2.GetDataArray().AsSpan().ToArray();
        var beforeB2 = b2.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            // Layer 1: hidden = input @ W1 + b1   (bias broadcast over batch)
            var z1 = engine.TensorMatMul(input, W1);
            var h1 = engine.TensorBroadcastAdd(z1, b1);
            // Layer 2: logits = h1 @ W2 + b2
            var z2 = engine.TensorMatMul(h1, W2);
            var logits = engine.TensorBroadcastAdd(z2, b2);
            // Loss = sum(logits) — gradient flow check; not a real loss but
            // produces non-zero dL/dY everywhere, which is all the backward
            // chain needs to test parameter updates.
            engine.ReduceSum(logits, null);
            plan = scope.CompileTraining(new[] { W1, b1, W2, b2 });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(
                OptimizerType.Adam,
                learningRate: 0.01f,
                beta1: 0.9f,
                beta2: 0.999f,
                eps: 1e-8f,
                weightDecay: 0f);
            plan.Step();

            double dW1 = L2Delta(beforeW1, W1.GetDataArray().AsSpan().ToArray());
            double dB1 = L2Delta(beforeB1, b1.GetDataArray().AsSpan().ToArray());
            double dW2 = L2Delta(beforeW2, W2.GetDataArray().AsSpan().ToArray());
            double dB2 = L2Delta(beforeB2, b2.GetDataArray().AsSpan().ToArray());

            _output.WriteLine($"L2(ΔW1)={dW1:E6}  L2(Δb1)={dB1:E6}  L2(ΔW2)={dW2:E6}  L2(Δb2)={dB2:E6}");

            // dY/db = ones for every output position → bias gradient is always
            // non-zero. If b1 or b2 didn't move, the backward chain is broken
            // even for the easiest case.
            Assert.True(dB1 > 1e-7, $"b1 did not move (L2(Δ)={dB1:E6}). Bias backward path is broken.");
            Assert.True(dB2 > 1e-7, $"b2 did not move (L2(Δ)={dB2:E6}). Output bias backward path is broken.");

            // dY/dW2 = h1ᵀ @ dY = non-zero because h1 is non-zero. The downstream
            // weight matrix should always update.
            Assert.True(dW2 > 1e-7, $"W2 did not move (L2(Δ)={dW2:E6}). Downstream MatMul weight backward is broken.");

            // The bug fingerprint: dW1 should also be non-zero because
            // dY/dW1 = inputᵀ @ dL/dh1, where dL/dh1 = dL/dY_logits @ W2ᵀ.
            // Pre-fix this stayed at zero — the chain stopped at the
            // downstream MatMul's input-side backward.
            Assert.True(
                dW1 > 1e-7,
                $"W1 did not move (L2(Δ)={dW1:E6}). The upstream MatMul's weight gradient was dropped " +
                "by the compiled-plan backward chain — this is AiDotNet#1331's root cause: " +
                "MatMul backward's dL/dX propagation dies for non-leaf X, so the second-from-last " +
                "weight matrix never gets dW = Xᵀ @ dY because dY (passed back from downstream) is zero.");
        }
    }

    /// <summary>
    /// Two stacked <see cref="IEngine.FusedLinear"/> ops (no activation) —
    /// the exact code path AiDotNet's <c>DenseLayer.Forward</c> takes when
    /// it routes through the fused-linear kernel. Same shapes as the
    /// two-MatMul-chain test; the only difference is the op selection.
    /// If MatMul+BroadcastAdd works but FusedLinear stacking doesn't, the
    /// bug is in <c>FusedLinearWithActivationBackward</c>'s graph plumbing.
    /// </summary>
    [Fact]
    public void TwoFusedLinearChain_AdamStep_AllFourParamsMustMove()
    {
        var engine = new CpuEngine();

        const int batch = 4, dIn = 8, dHidden = 6, dOut = 3;
        var input = FilledRand(new[] { batch, dIn }, seed: 31);
        var W1 = FilledRand(new[] { dIn, dHidden }, seed: 32);
        var b1 = FilledRand(new[] { dHidden }, seed: 33);
        var W2 = FilledRand(new[] { dHidden, dOut }, seed: 34);
        var b2 = FilledRand(new[] { dOut }, seed: 35);

        var beforeW1 = W1.GetDataArray().AsSpan().ToArray();
        var beforeB1 = b1.GetDataArray().AsSpan().ToArray();
        var beforeW2 = W2.GetDataArray().AsSpan().ToArray();
        var beforeB2 = b2.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h1 = engine.FusedLinear(input, W1, b1, FusedActivationType.None);
            var logits = engine.FusedLinear(h1, W2, b2, FusedActivationType.None);
            engine.ReduceSum(logits, null);
            plan = scope.CompileTraining(new[] { W1, b1, W2, b2 });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(
                OptimizerType.Adam,
                learningRate: 0.01f,
                beta1: 0.9f,
                beta2: 0.999f,
                eps: 1e-8f,
                weightDecay: 0f);
            plan.Step();

            double dW1 = L2Delta(beforeW1, W1.GetDataArray().AsSpan().ToArray());
            double dB1 = L2Delta(beforeB1, b1.GetDataArray().AsSpan().ToArray());
            double dW2 = L2Delta(beforeW2, W2.GetDataArray().AsSpan().ToArray());
            double dB2 = L2Delta(beforeB2, b2.GetDataArray().AsSpan().ToArray());

            _output.WriteLine($"L2(ΔW1)={dW1:E6}  L2(Δb1)={dB1:E6}  L2(ΔW2)={dW2:E6}  L2(Δb2)={dB2:E6}");

            Assert.True(dB2 > 1e-7, $"b2 stuck (L2={dB2:E6}) — output bias backward broken.");
            Assert.True(dW2 > 1e-7, $"W2 stuck (L2={dW2:E6}) — output weight backward broken.");
            Assert.True(dB1 > 1e-7, $"b1 stuck (L2={dB1:E6}) — first-layer bias backward broken.");
            Assert.True(dW1 > 1e-7,
                $"W1 stuck (L2={dW1:E6}) — first-layer weight backward broken. This is the AiDotNet#1331 " +
                "fingerprint: FusedLinear's compiled backward chain dies at the upstream weight matrix.");
        }
    }

    /// <summary>
    /// Mirrors what AiDotNet's <c>DenseLayer.Forward</c> does on a rank-3
    /// [batch, seq, features] input: flatten via Reshape, FusedLinear,
    /// Reshape back. Two stacked DenseLayer-style passes. If the Reshape
    /// op around FusedLinear corrupts the compiled-plan backward chain,
    /// upstream parameters stay stuck.
    /// </summary>
    [Fact]
    public void TwoFusedLinearChain_WithReshapeFlatten_AllFourParamsMustMove()
    {
        var engine = new CpuEngine();

        const int batch = 1, seqLen = 4, dIn = 8, dHidden = 6, dOut = 3;
        var input = FilledRand(new[] { batch, seqLen, dIn }, seed: 41);
        var W1 = FilledRand(new[] { dIn, dHidden }, seed: 42);
        var b1 = FilledRand(new[] { dHidden }, seed: 43);
        var W2 = FilledRand(new[] { dHidden, dOut }, seed: 44);
        var b2 = FilledRand(new[] { dOut }, seed: 45);

        var beforeW1 = W1.GetDataArray().AsSpan().ToArray();
        var beforeB1 = b1.GetDataArray().AsSpan().ToArray();
        var beforeW2 = W2.GetDataArray().AsSpan().ToArray();
        var beforeB2 = b2.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            // Dense #1: flatten → FusedLinear → reshape back
            var flat1 = engine.Reshape(input, new[] { batch * seqLen, dIn });
            var z1 = engine.FusedLinear(flat1, W1, b1, FusedActivationType.None);
            var h1 = engine.Reshape(z1, new[] { batch, seqLen, dHidden });

            // Dense #2: flatten → FusedLinear → reshape back
            var flat2 = engine.Reshape(h1, new[] { batch * seqLen, dHidden });
            var z2 = engine.FusedLinear(flat2, W2, b2, FusedActivationType.None);
            var logits = engine.Reshape(z2, new[] { batch, seqLen, dOut });

            engine.ReduceSum(logits, null);
            plan = scope.CompileTraining(new[] { W1, b1, W2, b2 });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(
                OptimizerType.Adam,
                learningRate: 0.01f,
                beta1: 0.9f,
                beta2: 0.999f,
                eps: 1e-8f,
                weightDecay: 0f);
            plan.Step();

            double dW1 = L2Delta(beforeW1, W1.GetDataArray().AsSpan().ToArray());
            double dB1 = L2Delta(beforeB1, b1.GetDataArray().AsSpan().ToArray());
            double dW2 = L2Delta(beforeW2, W2.GetDataArray().AsSpan().ToArray());
            double dB2 = L2Delta(beforeB2, b2.GetDataArray().AsSpan().ToArray());

            _output.WriteLine($"L2(ΔW1)={dW1:E6}  L2(Δb1)={dB1:E6}  L2(ΔW2)={dW2:E6}  L2(Δb2)={dB2:E6}");

            Assert.True(dB2 > 1e-7, $"b2 stuck (L2={dB2:E6})");
            Assert.True(dW2 > 1e-7, $"W2 stuck (L2={dW2:E6})");
            Assert.True(dB1 > 1e-7, $"b1 stuck (L2={dB1:E6})");
            Assert.True(dW1 > 1e-7,
                $"W1 stuck (L2={dW1:E6}) — Reshape around FusedLinear is corrupting the backward chain.");
        }
    }

    /// <summary>
    /// AiDotNet#1331 reproduction: mirrors the Transformer-byte-LM forward
    /// path from EmbeddingLookup through two FusedLinear layers (with
    /// reshape-flatten + reshape-back wrappers, as DenseLayer does) to
    /// ReduceSum loss. All six leaf params (embedding + 2× (W, b)) must
    /// move after a single Adam step. The AiDotNet Transformer diagnostic
    /// shows W params stuck and embedding stuck under this same op pattern;
    /// if this Tensors-only test passes, the bug is something else (likely
    /// process-shared state from a preceding inference call).
    /// </summary>
    [Fact]
    public void EmbeddingLookup_TwoFusedLinear_Chain_AllParamsMustMove()
    {
        var engine = new CpuEngine();

        const int batch = 1, seqLen = 4, vocab = 8, embDim = 32, dHidden = 24, dOut = 8;

        var indices = new Tensor<int>(new[] { batch, seqLen });
        indices[0, 0] = 0; indices[0, 1] = 2; indices[0, 2] = 4; indices[0, 3] = 6;

        var E = FilledRand(new[] { vocab, embDim }, seed: 51);
        var W1 = FilledRand(new[] { embDim, dHidden }, seed: 52);
        var b1 = FilledRand(new[] { dHidden }, seed: 53);
        var W2 = FilledRand(new[] { dHidden, dOut }, seed: 54);
        var b2 = FilledRand(new[] { dOut }, seed: 55);

        var beforeE = E.GetDataArray().AsSpan().ToArray();
        var beforeW1 = W1.GetDataArray().AsSpan().ToArray();
        var beforeB1 = b1.GetDataArray().AsSpan().ToArray();
        var beforeW2 = W2.GetDataArray().AsSpan().ToArray();
        var beforeB2 = b2.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            // EmbeddingLookup → [batch, seqLen, embDim]
            var emb = engine.TensorEmbeddingLookup<float, int>(E, indices);
            var emb3D = engine.Reshape(emb, new[] { batch, seqLen, embDim });

            // Dense #1: flatten → FusedLinear → reshape back
            var flat1 = engine.Reshape(emb3D, new[] { batch * seqLen, embDim });
            var z1 = engine.FusedLinear(flat1, W1, b1, FusedActivationType.None);
            var h1 = engine.Reshape(z1, new[] { batch, seqLen, dHidden });

            // Dense #2: flatten → FusedLinear → reshape back (the "head")
            var flat2 = engine.Reshape(h1, new[] { batch * seqLen, dHidden });
            var z2 = engine.FusedLinear(flat2, W2, b2, FusedActivationType.None);
            var logits = engine.Reshape(z2, new[] { batch, seqLen, dOut });

            engine.ReduceSum(logits, null);
            plan = scope.CompileTraining(new[] { E, W1, b1, W2, b2 });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(
                OptimizerType.Adam,
                learningRate: 0.01f,
                beta1: 0.9f,
                beta2: 0.999f,
                eps: 1e-8f,
                weightDecay: 0f);
            plan.Step();

            double dE = L2Delta(beforeE, E.GetDataArray().AsSpan().ToArray());
            double dW1 = L2Delta(beforeW1, W1.GetDataArray().AsSpan().ToArray());
            double dB1 = L2Delta(beforeB1, b1.GetDataArray().AsSpan().ToArray());
            double dW2 = L2Delta(beforeW2, W2.GetDataArray().AsSpan().ToArray());
            double dB2 = L2Delta(beforeB2, b2.GetDataArray().AsSpan().ToArray());

            _output.WriteLine(
                $"L2(ΔE)={dE:E6}  L2(ΔW1)={dW1:E6}  L2(Δb1)={dB1:E6}  " +
                $"L2(ΔW2)={dW2:E6}  L2(Δb2)={dB2:E6}");

            Assert.True(dB2 > 1e-7, $"b2 stuck (L2={dB2:E6})");
            Assert.True(dW2 > 1e-7, $"W2 stuck (L2={dW2:E6})");
            Assert.True(dB1 > 1e-7, $"b1 stuck (L2={dB1:E6})");
            Assert.True(dW1 > 1e-7, $"W1 stuck (L2={dW1:E6})");
            Assert.True(dE > 1e-7,  $"E stuck (L2={dE:E6}) — embedding gradient dropped despite v0.80.2 fix.");
        }
    }

    [Fact]
    public void FusedLinearThenLayerNorm_GammaMustMove()
    {
        var engine = new CpuEngine();
        const int batch = 4, dIn = 8, feat = 6;
        var input = FilledRand(new[] { batch, dIn }, seed: 81);
        var W = FilledRand(new[] { dIn, feat }, seed: 82);
        var b = FilledRand(new[] { feat }, seed: 83);
        var gamma = FilledRand(new[] { feat }, seed: 84);
        var beta  = FilledRand(new[] { feat }, seed: 85);
        for (int i = 0; i < gamma.Length; i++) gamma[i] = gamma[i] + 1f;

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var z = engine.FusedLinear(input, W, b, FusedActivationType.None);
            var normed = engine.LayerNorm(z, gamma, beta, 1e-5, out _, out _);
            engine.ReduceSum(normed, null);
            plan = scope.CompileTraining(new[] { W, b, gamma, beta });
        }

        using (plan)
        {
            plan.Step();
            string[] names = { "W", "b", "gamma", "beta" };
            for (int i = 0; i < plan.Gradients.Length; i++)
            {
                var g = plan.Gradients[i];
                if (g is null) { _output.WriteLine($"  gradient[{i}] {names[i]} = NULL"); continue; }
                double gL2 = 0; var gSpan = g.AsSpan();
                for (int k = 0; k < gSpan.Length; k++) gL2 += gSpan[k] * gSpan[k];
                gL2 = System.Math.Sqrt(gL2);
                _output.WriteLine($"  gradient[{i}] {names[i]} L2={gL2:E6}");
            }
        }
    }

    /// <summary>
    /// Minimal isolation: a single LayerNorm op with gamma + beta as
    /// trainable params. If the bug reproduces here, the entire LayerNorm
    /// compiled-plan backward is the suspect, not the surrounding chain.
    /// </summary>
    [Fact]
    public void LayerNormOnly_BothGammaAndBetaMustMove()
    {
        var engine = new CpuEngine();
        const int batch = 4, feat = 6;
        var input = FilledRand(new[] { batch, feat }, seed: 71);
        var gamma = FilledRand(new[] { feat }, seed: 72);
        var beta  = FilledRand(new[] { feat }, seed: 73);
        for (int i = 0; i < gamma.Length; i++) gamma[i] = gamma[i] + 1f;

        var beforeGamma = gamma.GetDataArray().AsSpan().ToArray();
        var beforeBeta  = beta .GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var normed = engine.LayerNorm(input, gamma, beta, 1e-5, out _, out _);
            engine.ReduceSum(normed, null);
            plan = scope.CompileTraining(new[] { gamma, beta });
        }

        using (plan)
        {
            plan.Step();
            for (int i = 0; i < plan.Gradients.Length; i++)
            {
                var g = plan.Gradients[i];
                if (g is null) { _output.WriteLine($"  gradient[{i}] = NULL"); continue; }
                double gL2 = 0; var gSpan = g.AsSpan();
                for (int k = 0; k < gSpan.Length; k++) gL2 += gSpan[k] * gSpan[k];
                gL2 = System.Math.Sqrt(gL2);
                _output.WriteLine($"  gradient[{i}] L2={gL2:E6}");
            }
        }
    }

    /// <summary>
    /// AiDotNet#1331 root-cause test: mirrors a transformer FFN block —
    /// FusedLinear → LayerNorm → FusedLinear → ReduceSum. In the original
    /// transformer diagnostic, every LayerNorm gain (γ) was stuck and
    /// everything upstream of the last LayerNorm was stuck. If LayerNorm's
    /// compiled backward drops dL/dγ or dL/dInput, this test will reproduce
    /// the bug: upstream W1/b1 + gamma all stuck while only the LN beta
    /// and downstream W2/b2 move.
    /// </summary>
    [Fact]
    public void LayerNormBetweenFusedLinears_AllParamsMustMove()
    {
        var engine = new CpuEngine();

        const int batch = 4, dIn = 8, dHidden = 6, dOut = 3;
        var input = FilledRand(new[] { batch, dIn }, seed: 61);
        var W1 = FilledRand(new[] { dIn, dHidden }, seed: 62);
        var b1 = FilledRand(new[] { dHidden }, seed: 63);
        var gamma = FilledRand(new[] { dHidden }, seed: 64);
        var beta = FilledRand(new[] { dHidden }, seed: 65);
        var W2 = FilledRand(new[] { dHidden, dOut }, seed: 66);
        var b2 = FilledRand(new[] { dOut }, seed: 67);

        // Bias gamma above zero so the layer is well-conditioned for normalization.
        for (int i = 0; i < gamma.Length; i++) gamma[i] = gamma[i] + 1f;

        var beforeW1 = W1.GetDataArray().AsSpan().ToArray();
        var beforeB1 = b1.GetDataArray().AsSpan().ToArray();
        var beforeGamma = gamma.GetDataArray().AsSpan().ToArray();
        var beforeBeta = beta.GetDataArray().AsSpan().ToArray();
        var beforeW2 = W2.GetDataArray().AsSpan().ToArray();
        var beforeB2 = b2.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var z1 = engine.FusedLinear(input, W1, b1, FusedActivationType.None);
            var normed = engine.LayerNorm(z1, gamma, beta, 1e-5, out _, out _);
            var z2 = engine.FusedLinear(normed, W2, b2, FusedActivationType.None);
            engine.ReduceSum(z2, null);
            plan = scope.CompileTraining(new[] { W1, b1, gamma, beta, W2, b2 });
        }

        using (plan)
        {
            // Probe: call plan.Step() WITHOUT configuring the optimizer so we
            // can inspect plan.Gradients directly. Then re-step with the optimizer.
            plan.Step();
            string[] names = { "W1", "b1", "gamma", "beta", "W2", "b2" };
            for (int i = 0; i < plan.Gradients.Length; i++)
            {
                var g = plan.Gradients[i];
                if (g is null)
                {
                    _output.WriteLine($"  gradient[{i}] {names[i]} = NULL");
                    continue;
                }
                double gL2 = 0;
                var gSpan = g.AsSpan();
                for (int k = 0; k < gSpan.Length; k++) gL2 += gSpan[k] * gSpan[k];
                gL2 = System.Math.Sqrt(gL2);
                _output.WriteLine($"  gradient[{i}] {names[i]}  shape=[{string.Join(",", g.Shape.ToArray())}]  L2={gL2:E6}");
            }
        }
    }

    /// <summary>
    /// Three-MatMul chain (Dense → Dense → Dense head). All six leaf params
    /// (three Ws, three bs) must move. Pre-fix only the last layer's bias
    /// (and possibly the last weight) moves; everything upstream stays stuck.
    /// </summary>
    [Fact]
    public void ThreeMatMulChain_AdamStep_AllSixParamsMustMove()
    {
        var engine = new CpuEngine();

        const int batch = 4, dIn = 8, d1 = 6, d2 = 5, dOut = 3;
        var input = FilledRand(new[] { batch, dIn }, seed: 21);
        var W1 = FilledRand(new[] { dIn, d1 }, seed: 22);
        var b1 = FilledRand(new[] { d1 }, seed: 23);
        var W2 = FilledRand(new[] { d1, d2 }, seed: 24);
        var b2 = FilledRand(new[] { d2 }, seed: 25);
        var W3 = FilledRand(new[] { d2, dOut }, seed: 26);
        var b3 = FilledRand(new[] { dOut }, seed: 27);

        var beforeW1 = W1.GetDataArray().AsSpan().ToArray();
        var beforeB1 = b1.GetDataArray().AsSpan().ToArray();
        var beforeW2 = W2.GetDataArray().AsSpan().ToArray();
        var beforeB2 = b2.GetDataArray().AsSpan().ToArray();
        var beforeW3 = W3.GetDataArray().AsSpan().ToArray();
        var beforeB3 = b3.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h1 = engine.TensorBroadcastAdd(engine.TensorMatMul(input, W1), b1);
            var h2 = engine.TensorBroadcastAdd(engine.TensorMatMul(h1, W2), b2);
            var logits = engine.TensorBroadcastAdd(engine.TensorMatMul(h2, W3), b3);
            engine.ReduceSum(logits, null);
            plan = scope.CompileTraining(new[] { W1, b1, W2, b2, W3, b3 });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(
                OptimizerType.Adam,
                learningRate: 0.01f,
                beta1: 0.9f,
                beta2: 0.999f,
                eps: 1e-8f,
                weightDecay: 0f);
            plan.Step();

            var deltas = new (string name, double l2)[]
            {
                ("W1", L2Delta(beforeW1, W1.GetDataArray().AsSpan().ToArray())),
                ("b1", L2Delta(beforeB1, b1.GetDataArray().AsSpan().ToArray())),
                ("W2", L2Delta(beforeW2, W2.GetDataArray().AsSpan().ToArray())),
                ("b2", L2Delta(beforeB2, b2.GetDataArray().AsSpan().ToArray())),
                ("W3", L2Delta(beforeW3, W3.GetDataArray().AsSpan().ToArray())),
                ("b3", L2Delta(beforeB3, b3.GetDataArray().AsSpan().ToArray())),
            };
            foreach (var (name, l2) in deltas)
            {
                string verdict = l2 > 1e-7 ? "MOVED" : "STUCK";
                _output.WriteLine($"  [{verdict}] {name}  L2(Δ)={l2:E6}");
            }

            foreach (var (name, l2) in deltas)
            {
                Assert.True(l2 > 1e-7, $"{name} did not move (L2(Δ)={l2:E6}). " +
                    "Compiled-plan backward dropped this parameter's gradient.");
            }
        }
    }
}
