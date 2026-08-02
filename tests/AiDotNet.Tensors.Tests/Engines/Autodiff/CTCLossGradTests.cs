using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// <c>TensorCTCLoss</c>'s gradient must be taken with respect to LOG-PROBABILITIES, which is what the
/// forward consumes.
/// </summary>
/// <remarks>
/// <para>
/// The backward computed <c>prob[t,k] - posterior[t,k]</c> — the familiar Graves-2006 form, but that is
/// the gradient with respect to the pre-softmax LOGITS. It is correct only when you additionally chain
/// through <c>logProbs = log_softmax(logits)</c>, where
/// <c>d logProbs[t,j]/d logits[t,k] = delta_jk - prob[t,k]</c> and <c>sum_j posterior[t,j] = 1</c>
/// together contribute the extra <c>prob[t,k]</c>.
/// </para>
/// <para>
/// <c>CpuEngine.TensorCTCLoss</c> runs the forward-backward recursion directly in the log domain and
/// never applies a log_softmax, so that term does not belong. The reported gradient was therefore wrong
/// by exactly <c>prob[t,k]</c> at every element. The gradcheck sweep caught it as analytical -0.214269
/// vs numerical -0.433886; reconstructing <c>analytical - numerical</c> across the class axis gave
/// 0.220, 0.593, 0.187 — summing to exactly 1.000, i.e. a probability distribution, which identified
/// the spurious term precisely.
/// </para>
/// <para>
/// CTC is the training objective for the ASR models in this library, so a gradient off by the model's
/// own softmax output corrupts every CTC training run. These tests assert closed-form invariants rather
/// than only finite differences.
/// </para>
/// </remarks>
public class CTCLossGradTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public CTCLossGradTests(ITestOutputHelper o) => _out = o;

    /// <summary>Row-wise log-softmax over the class axis of a [T, N, C] tensor.</summary>
    private static Tensor<double> LogProbs(int timeSteps, int batch, int classes, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>([timeSteps, batch, classes]);
        for (int i = 0; i < timeSteps * batch; i++)
        {
            var logits = new double[classes];
            double max = double.NegativeInfinity;
            for (int c = 0; c < classes; c++) { logits[c] = -1.0 + rng.NextDouble() * 2.0; max = Math.Max(max, logits[c]); }
            double sumExp = 0;
            for (int c = 0; c < classes; c++) sumExp += Math.Exp(logits[c] - max);
            double lse = max + Math.Log(sumExp);
            for (int c = 0; c < classes; c++) t[i * classes + c] = logits[c] - lse;
        }
        return t;
    }

    private static Tensor<int> Targets(params int[] labels)
    {
        var t = new Tensor<int>([labels.Length]);
        for (int i = 0; i < labels.Length; i++) t[i] = labels[i];
        return t;
    }

    [Fact]
    public void Gradient_MatchesFiniteDifferences()
    {
        const int T = 4, N = 1, C = 3;
        var logProbs = LogProbs(T, N, C, seed: 5);
        var targets = Targets(1, 2);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(
            _engine.TensorCTCLoss(logProbs, targets, [T], [2], 0), null);
        var grads = tape.ComputeGradients(loss, [logProbs]);
        var g = grads[logProbs];

        const double h = 1e-6;
        double worst = 0;
        for (int i = 0; i < logProbs.Length; i++)
        {
            double orig = logProbs[i];
            logProbs[i] = orig + h;
            double lp = _engine.TensorSum(_engine.TensorCTCLoss(logProbs, targets, [T], [2], 0));
            logProbs[i] = orig - h;
            double lm = _engine.TensorSum(_engine.TensorCTCLoss(logProbs, targets, [T], [2], 0));
            logProbs[i] = orig;

            double numerical = (lp - lm) / (2 * h);
            double denom = Math.Max(1.0, Math.Max(Math.Abs(g![i]), Math.Abs(numerical)));
            double rel = Math.Abs(g[i] - numerical) / denom;
            worst = Math.Max(worst, rel);
            _out.WriteLine($"[{i}] analytical={g[i]:G10} numerical={numerical:G10} rel={rel:E3}");
        }
        Assert.True(worst < 1e-6, $"worst relative gradient error {worst:E3}");
    }

    /// <summary>
    /// CLOSED FORM: the gradient w.r.t. log-probabilities is exactly -posterior, and the CTC posteriors
    /// over the class axis sum to 1 at every timestep. So the per-timestep gradient sum must be exactly
    /// -1. Under the old prob-minus-posterior formula each timestep summed to 1 - 1 = 0 instead, so this
    /// single assertion distinguishes the two formulas without any finite differencing.
    /// </summary>
    [Fact]
    public void PerTimestepGradientSum_IsExactlyMinusOne()
    {
        const int T = 5, N = 1, C = 4;
        var logProbs = LogProbs(T, N, C, seed: 9);
        var targets = Targets(1, 3, 2);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(
            _engine.TensorCTCLoss(logProbs, targets, [T], [3], 0), null);
        var grads = tape.ComputeGradients(loss, [logProbs]);
        var g = grads[logProbs];

        for (int t = 0; t < T; t++)
        {
            double sum = 0;
            for (int c = 0; c < C; c++) sum += g![t * C + c];
            _out.WriteLine($"t={t} sum over classes = {sum:G12}");
            Assert.Equal(-1.0, sum, 10);
        }
    }

    /// <summary>
    /// The same closed form with N &gt; 1, which is what actually exercises the batch stride.
    /// </summary>
    /// <remarks>
    /// Every other test here uses N = 1, where the flat index <c>t * C + c</c> happens to equal the
    /// correct <c>(t * N + n) * C + c</c>. A backward that indexes the batch axis wrongly therefore
    /// passes all of them. With N = 2 the two disagree from t = 1 onward, so this is the assertion that
    /// can actually catch it. The two sequences are given DIFFERENT target lengths (3 and 2) so a
    /// backward that reused one sequence's alpha/beta lattice for both cannot satisfy it either.
    /// </remarks>
    [Fact]
    public void PerTimestepGradientSum_IsExactlyMinusOne_Batched()
    {
        const int T = 5, N = 2, C = 4;
        var logProbs = LogProbs(T, N, C, seed: 21);
        // Concatenated: sequence 0 is [1,3,2] (length 3), sequence 1 is [2,1] (length 2).
        var targets = Targets(1, 3, 2, 2, 1);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(
            _engine.TensorCTCLoss(logProbs, targets, [T, T], [3, 2], 0), null);
        var grads = tape.ComputeGradients(loss, [logProbs]);
        var g = grads[logProbs];

        for (int t = 0; t < T; t++)
        {
            for (int n = 0; n < N; n++)
            {
                double sum = 0;
                for (int c = 0; c < C; c++) sum += g![(t * N + n) * C + c];
                _out.WriteLine($"t={t} n={n} sum over classes = {sum:G12}");
                Assert.Equal(-1.0, sum, 10);
            }
        }
    }

    /// <summary>
    /// A class that appears neither in the target sequence nor as the blank has posterior 0 at every
    /// timestep, so its gradient must be exactly 0. The old formula gave it prob[t,k] &gt; 0 —
    /// gradient pushing on a class the loss does not depend on.
    /// </summary>
    [Fact]
    public void ClassAbsentFromTargets_HasExactlyZeroGradient()
    {
        const int T = 4, N = 1, C = 4;   // classes: 0 = blank, targets use 1 and 2, class 3 is unused
        var logProbs = LogProbs(T, N, C, seed: 13);
        var targets = Targets(1, 2);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(
            _engine.TensorCTCLoss(logProbs, targets, [T], [2], 0), null);
        var grads = tape.ComputeGradients(loss, [logProbs]);
        var g = grads[logProbs];

        for (int t = 0; t < T; t++)
        {
            double gUnused = g![t * C + 3];
            _out.WriteLine($"t={t} grad for unused class 3 = {gUnused:G12}");
            Assert.Equal(0.0, gUnused);
        }
    }

    /// <summary>
    /// The gradient must be non-positive everywhere: it is -posterior, and a posterior is a
    /// probability. The old formula produced positive values wherever prob exceeded posterior.
    /// </summary>
    [Fact]
    public void Gradient_IsNonPositiveEverywhere()
    {
        const int T = 4, N = 1, C = 3;
        var logProbs = LogProbs(T, N, C, seed: 21);
        var targets = Targets(1, 2);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(
            _engine.TensorCTCLoss(logProbs, targets, [T], [2], 0), null);
        var grads = tape.ComputeGradients(loss, [logProbs]);
        var g = grads[logProbs];

        for (int i = 0; i < logProbs.Length; i++)
            Assert.True(g![i] <= 0.0, $"grad[{i}] = {g[i]:G10} is positive; -posterior must be <= 0");
    }
}
