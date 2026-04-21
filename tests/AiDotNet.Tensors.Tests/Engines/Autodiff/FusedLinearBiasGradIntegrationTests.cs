using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Integration coverage for issue #234 — <c>FusedLinearBackward</c> bias
/// gradient must round-trip through realistic training patterns (multi-op
/// chains, per-activation kernels, gradient accumulation, optimizer
/// update, double/float precision, bias-shape variants). The unit tests
/// in <see cref="FusedLinearRank3BiasGradTests"/> prove the individual
/// backward call produces the right shape; these tests prove the fix
/// holds up in the end-to-end training contexts that originally
/// surfaced the bug (LagLlama / MOIRAI / UniTS / TimeGPT / TimeLLM /
/// Timer FeedForwardLayer stacks).
/// </summary>
public class FusedLinearBiasGradIntegrationTests : IDisposable
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private readonly bool _previousReplayMode;

    public FusedLinearBiasGradIntegrationTests()
    {
        // AutoTrainingCompiler.ReplayMode is process-wide static state;
        // capture the prior value so Dispose can restore it and we don't
        // leak a false into tests that expect replay mode on.
        _previousReplayMode = AutoTrainingCompiler.ReplayMode;
        AutoTrainingCompiler.ReplayMode = false;
    }

    public void Dispose() => AutoTrainingCompiler.ReplayMode = _previousReplayMode;

    /// <summary>
    /// Two-layer FFL (the shape that actually trips the consuming
    /// models): <c>[B, T, F_in] → linear(act) → linear → sum</c>. Runs
    /// SGD for several steps and verifies three things about every
    /// activation kernel:
    /// <list type="number">
    ///   <item>Every gradient has a shape matching its parameter (the
    ///     #234 regression).</item>
    ///   <item>Final loss is strictly below the initial loss, with at
    ///     most one uptick allowed over the run — non-convex activations
    ///     like Sigmoid/GELU can briefly overshoot at lr = 0.01.</item>
    ///   <item>Every parameter, including both biases, actually
    ///     changed from its initial value. A non-reducing bias gradient
    ///     would skip the optimizer update (wrong-shape `TensorAdd` used
    ///     to throw) and leave the parameter pinned at init.</item>
    /// </list>
    /// </summary>
    [Theory]
    [InlineData(FusedActivationType.None)]
    [InlineData(FusedActivationType.ReLU)]
    [InlineData(FusedActivationType.Sigmoid)]
    [InlineData(FusedActivationType.GELU)]
    [InlineData(FusedActivationType.Tanh)]
    public void TwoLayerFFL_Rank3Double_TrainsCleanly(FusedActivationType act)
    {
        const int B = 2, T = 4, FIn = 3, FH = 8, FOut = 2;
        var rng = new Random(1337);

        var input = NewRandomDouble(new[] { B, T, FIn }, rng, 0.5);
        var w1 = NewRandomDouble(new[] { FIn, FH }, rng, 0.1);
        var b1 = NewRandomDouble(new[] { FH }, rng, 0.01);
        var w2 = NewRandomDouble(new[] { FH, FOut }, rng, 0.1);
        var b2 = NewRandomDouble(new[] { 1, FOut }, rng, 0.01);
        var target = NewRandomDouble(new[] { B, T, FOut }, rng, 0.5);

        // Snapshot initial parameters for the post-run mutation check.
        var w1Init = w1.AsSpan().ToArray();
        var b1Init = b1.AsSpan().ToArray();
        var w2Init = w2.AsSpan().ToArray();
        var b2Init = b2.AsSpan().ToArray();

        var lossHistory = new List<double>();
        const double lr = 0.01;
        const int steps = 5;

        for (int step = 0; step < steps; step++)
        {
            using var tape = new GradientTape<double>();
            var h = _engine.FusedLinear(input, w1, b1, act);
            var y = _engine.FusedLinear(h, w2, b2, FusedActivationType.None);
            // MSE loss.
            var diff = _engine.TensorSubtract(y, target);
            var sq = _engine.TensorMultiply(diff, diff);
            var loss = _engine.ReduceSum(sq, null);
            var grads = tape.ComputeGradients(loss, new[] { w1, b1, w2, b2 });

            // Bias-grad shape check — the bug that blocked training.
            Assert.Equal(b1._shape, grads[b1]._shape);
            Assert.Equal(b2._shape, grads[b2]._shape);
            // And for the weights, for completeness.
            Assert.Equal(w1._shape, grads[w1]._shape);
            Assert.Equal(w2._shape, grads[w2]._shape);

            lossHistory.Add(loss.AsSpan()[0]);

            // Plain SGD update — exercises the downstream TensorAdd that
            // originally threw on shape mismatch.
            ApplyGradDouble(w1, grads[w1], lr);
            ApplyGradDouble(b1, grads[b1], lr);
            ApplyGradDouble(w2, grads[w2], lr);
            ApplyGradDouble(b2, grads[b2], lr);
        }

        // Final loss below initial.
        Assert.True(lossHistory[^1] < lossHistory[0],
            $"Final loss not below initial for activation {act}: " +
            string.Join(" → ", lossHistory.Select(l => l.ToString("G4"))));

        // At most one uptick over the run. Non-convex activations can
        // overshoot once at lr=0.01, but repeated upticks mean the
        // update direction itself is wrong (which is what a broken bias
        // gradient would cause).
        int upticks = 0;
        for (int i = 1; i < lossHistory.Count; i++)
            if (lossHistory[i] > lossHistory[i - 1]) upticks++;
        Assert.True(upticks <= 1,
            $"Loss oscillated for activation {act} ({upticks} upticks): " +
            string.Join(" → ", lossHistory.Select(l => l.ToString("G4"))));

        // Every parameter — including both biases — moved. If a bias
        // gradient came back wrong-shape, the downstream TensorAdd
        // would have thrown; if it came back right-shape-but-zero,
        // that slot would have stayed at its init value.
        AssertMutated(w1Init, w1.AsSpan(), $"w1 (act={act})");
        AssertMutated(b1Init, b1.AsSpan(), $"b1 (act={act})");
        AssertMutated(w2Init, w2.AsSpan(), $"w2 (act={act})");
        AssertMutated(b2Init, b2.AsSpan(), $"b2 (act={act})");
    }

    /// <summary>
    /// Same model shape in <c>float</c>. Even though float rank-2 hits
    /// the BLAS fast path, rank-3 goes through the same fallback —
    /// verify that fallback is right for float too, not just double.
    /// Same three guarantees as the double variant: shape-correct
    /// gradients, final loss below initial, every parameter moved.
    /// </summary>
    [Fact]
    public void TwoLayerFFL_Rank3Float_TrainsCleanly()
    {
        const int B = 2, T = 4, FIn = 3, FH = 8, FOut = 2;
        var rng = new Random(99);

        var input = NewRandomFloat(new[] { B, T, FIn }, rng, 0.5f);
        var w1 = NewRandomFloat(new[] { FIn, FH }, rng, 0.1f);
        var b1 = NewRandomFloat(new[] { FH }, rng, 0.01f);
        var w2 = NewRandomFloat(new[] { FH, FOut }, rng, 0.1f);
        var b2 = NewRandomFloat(new[] { 1, FOut }, rng, 0.01f);
        var target = NewRandomFloat(new[] { B, T, FOut }, rng, 0.5f);

        var w1Init = w1.AsSpan().ToArray();
        var b1Init = b1.AsSpan().ToArray();
        var w2Init = w2.AsSpan().ToArray();
        var b2Init = b2.AsSpan().ToArray();

        float initialLoss = 0, finalLoss = 0;
        for (int step = 0; step < 5; step++)
        {
            using var tape = new GradientTape<float>();
            var h = _engine.FusedLinear(input, w1, b1, FusedActivationType.ReLU);
            var y = _engine.FusedLinear(h, w2, b2, FusedActivationType.None);
            var diff = _engine.TensorSubtract(y, target);
            var sq = _engine.TensorMultiply(diff, diff);
            var loss = _engine.ReduceSum(sq, null);
            var grads = tape.ComputeGradients(loss, new[] { w1, b1, w2, b2 });

            Assert.Equal(b1._shape, grads[b1]._shape);
            Assert.Equal(b2._shape, grads[b2]._shape);

            if (step == 0) initialLoss = loss.AsSpan()[0];
            finalLoss = loss.AsSpan()[0];

            ApplyGradFloat(w1, grads[w1], 0.01f);
            ApplyGradFloat(b1, grads[b1], 0.01f);
            ApplyGradFloat(w2, grads[w2], 0.01f);
            ApplyGradFloat(b2, grads[b2], 0.01f);
        }

        Assert.True(finalLoss < initialLoss,
            $"Final loss not below initial: {initialLoss:G4} → {finalLoss:G4}");
        AssertMutated(w1Init, w1.AsSpan(), "w1 (float)");
        AssertMutated(b1Init, b1.AsSpan(), "b1 (float)");
        AssertMutated(w2Init, w2.AsSpan(), "w2 (float)");
        AssertMutated(b2Init, b2.AsSpan(), "b2 (float)");
    }

    /// <summary>
    /// Bias-shape combinatorics: every legal bias shape that callers
    /// actually use must produce a shape-identical gradient.
    /// </summary>
    [Theory]
    [InlineData(new int[] { 3, 4 }, new int[] { 4, 6 }, new int[] { 6 })]            // [M,K] × [K,N] + [N]
    [InlineData(new int[] { 3, 4 }, new int[] { 4, 6 }, new int[] { 1, 6 })]         // [M,K] × [K,N] + [1,N]
    [InlineData(new int[] { 2, 5, 4 }, new int[] { 4, 6 }, new int[] { 6 })]         // [B,T,K] × [K,N] + [N]
    [InlineData(new int[] { 2, 5, 4 }, new int[] { 4, 6 }, new int[] { 1, 6 })]      // [B,T,K] × [K,N] + [1,N]
    [InlineData(new int[] { 2, 5, 4 }, new int[] { 4, 6 }, new int[] { 1, 1, 6 })]   // [B,T,K] × [K,N] + [1,1,N] — both SumToShape branches
    [InlineData(new int[] { 1, 8, 1 }, new int[] { 1, 24 }, new int[] { 1, 24 })]    // exact #234 repro
    public void BiasShapeVariants_AllReturnCorrectGradShape(int[] inShape, int[] wShape, int[] bShape)
    {
        var rng = new Random(2024);
        var input = NewRandomDouble(inShape, rng, 0.5);
        var weights = NewRandomDouble(wShape, rng, 0.1);
        var bias = NewRandomDouble(bShape, rng, 0.01);

        using var tape = new GradientTape<double>();
        var y = _engine.FusedLinear(input, weights, bias, FusedActivationType.None);
        var loss = _engine.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { input, weights, bias });

        Assert.Equal(bias._shape, grads[bias]._shape);

        // With loss = sum(y) the bias gradient at every feature slot is
        // (product of leading dims of output), so we can cross-check
        // values against an exact integer.
        int leading = 1;
        for (int i = 0; i < y.Rank - 1; i++) leading *= y._shape[i];
        var bSpan = grads[bias].AsSpan();
        for (int i = 0; i < bSpan.Length; i++)
            Assert.True(Math.Abs(bSpan[i] - leading) < 1e-9,
                $"bias-grad[{i}] = {bSpan[i]}, expected {leading}");
    }

    /// <summary>
    /// Cross-iteration gradient accumulation: call backward twice on
    /// separate tapes and verify the second call doesn't inherit pooled
    /// garbage from the first. (Related buffer-pool regression lived in
    /// the same function — see BackwardBufferPoolingTests.)
    /// </summary>
    [Fact]
    public void RepeatBackward_Rank3Double_GradsStayConstantForFixedInputs()
    {
        var input = NewRandomDouble(new[] { 1, 8, 1 }, new Random(5), 0.5);
        var weights = NewRandomDouble(new[] { 1, 24 }, new Random(6), 0.1);
        var bias = NewRandomDouble(new[] { 1, 24 }, new Random(7), 0.01);

        double[] firstBiasGrad = Array.Empty<double>();
        for (int i = 0; i < 3; i++)
        {
            using var tape = new GradientTape<double>();
            var y = _engine.FusedLinear(input, weights, bias, FusedActivationType.None);
            var loss = _engine.ReduceSum(y, null);
            var grads = tape.ComputeGradients(loss, new[] { bias });
            var span = grads[bias].AsSpan();
            if (i == 0)
            {
                firstBiasGrad = span.ToArray();
            }
            else
            {
                for (int j = 0; j < span.Length; j++)
                    Assert.True(Math.Abs(span[j] - firstBiasGrad[j]) < 1e-12,
                        $"iteration {i} bias-grad[{j}] drifted: {firstBiasGrad[j]} → {span[j]}");
            }
        }
    }

    // ---- helpers ---------------------------------------------------------

    private static Tensor<double> NewRandomDouble(int[] shape, Random rng, double scale)
    {
        var t = new Tensor<double>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (rng.NextDouble() - 0.5) * 2 * scale;
        return t;
    }

    private static Tensor<float> NewRandomFloat(int[] shape, Random rng, float scale)
    {
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)((rng.NextDouble() - 0.5) * 2 * scale);
        return t;
    }

    private static void ApplyGradDouble(Tensor<double> param, Tensor<double> grad, double lr)
    {
        var p = param.AsWritableSpan();
        var g = grad.AsSpan();
        for (int i = 0; i < p.Length; i++) p[i] -= lr * g[i];
    }

    private static void ApplyGradFloat(Tensor<float> param, Tensor<float> grad, float lr)
    {
        var p = param.AsWritableSpan();
        var g = grad.AsSpan();
        for (int i = 0; i < p.Length; i++) p[i] -= lr * g[i];
    }

    /// <summary>
    /// Asserts at least one element of <paramref name="after"/> differs from
    /// its initial value in <paramref name="before"/>. Used to confirm a
    /// parameter actually received an update — a shape-mismatched gradient
    /// would have thrown before we got here; a zero gradient would leave
    /// the parameter at its init value and this assertion would fire.
    /// </summary>
    private static void AssertMutated(double[] before, ReadOnlySpan<double> after, string label)
    {
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < before.Length; i++)
            if (before[i] != after[i]) return;
        Assert.Fail($"{label}: no element changed from initial value after training.");
    }

    private static void AssertMutated(float[] before, ReadOnlySpan<float> after, string label)
    {
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < before.Length; i++)
            if (before[i] != after[i]) return;
        Assert.Fail($"{label}: no element changed from initial value after training.");
    }
}
