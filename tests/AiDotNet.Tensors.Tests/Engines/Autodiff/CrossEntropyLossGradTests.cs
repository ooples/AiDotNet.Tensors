using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// <c>TensorCrossEntropyLoss</c> must report the gradient of the loss it ACTUALLY computes.
/// </summary>
/// <remarks>
/// <para>
/// The forward is <c>loss_b = -sum_{c : t_bc &gt; 0} t_bc * (x_bc - logSumExp_b)</c>, whose derivative is
/// <c>softmax_bk * (sum_c t_bc) - t_bk</c>, all over the batch size. The backward previously computed
/// <c>(softmax - targets) / n</c>, hardcoding <c>sum_c t_bc == 1</c>. The gradcheck sweep caught this as
/// analytical -0.265905 vs numerical -0.0790221 on unnormalised soft targets.
/// </para>
/// <para>
/// Sparse (rank-1 class-index) targets are covered too: the old code called
/// <c>TensorSubtract(softmax [n,C], targets [n])</c>, a shape mismatch that threw, so class-index
/// cross-entropy had no working gradient at all.
/// </para>
/// </remarks>
public class CrossEntropyLossGradTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public CrossEntropyLossGradTests(ITestOutputHelper o) => _out = o;

    private const int Batch = 3;
    private const int Classes = 4;

    private static Tensor<double> Logits(int seed)
    {
        var rng = new Random(seed);
        var x = new Tensor<double>([Batch, Classes]);
        for (int i = 0; i < x.Length; i++) x[i] = -1.0 + rng.NextDouble() * 2.0;
        return x;
    }

    /// <summary>Central finite differences on every logit, against the recorded analytical gradient.</summary>
    private void AssertGradientMatchesFiniteDifferences(Tensor<double> logits, Tensor<double> targets, string label)
    {
        using var tape = new GradientTape<double>();
        // Pin the tape to the CpuEngine under test. TensorCrossEntropyLoss does not call
        // BindEngineIfUnset (only 7 of ~294 recording sites in CpuEngine do), so the backward
        // would otherwise dispatch to AiDotNetEngine.Current — DirectGpuTensorEngine on a
        // GPU-auto-detect host — and compute its softmax in single precision. That is a separate
        // systemic defect; binding here keeps this test measuring cross-entropy alone.
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.TensorCrossEntropyLoss(logits, targets);
        var grads = tape.ComputeGradients(loss, [logits]);
        Assert.True(grads.TryGetValue(logits, out var g) && g is not null, $"{label}: no gradient recorded");

        const double eps = 1e-6;
        double worst = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            double orig = logits[i];
            logits[i] = orig + eps;
            double lp = _engine.TensorCrossEntropyLoss(logits, targets)[0];
            logits[i] = orig - eps;
            double lm = _engine.TensorCrossEntropyLoss(logits, targets)[0];
            logits[i] = orig;

            double numerical = (lp - lm) / (2 * eps);
            double analytical = g![i];
            double denom = Math.Max(1.0, Math.Max(Math.Abs(analytical), Math.Abs(numerical)));
            double rel = Math.Abs(analytical - numerical) / denom;
            worst = Math.Max(worst, rel);
            _out.WriteLine($"{label}[{i}] analytical={analytical:G10} numerical={numerical:G10} rel={rel:E3}");
        }

        Assert.True(worst < 1e-6, $"{label}: worst relative gradient error {worst:E3}");
    }

    [Fact]
    public void NormalisedSoftTargets_GradientMatchesFiniteDifferences()
    {
        var logits = Logits(11);
        var targets = new Tensor<double>([Batch, Classes]);
        var rng = new Random(22);
        for (int b = 0; b < Batch; b++)
        {
            double sum = 0;
            var row = new double[Classes];
            for (int c = 0; c < Classes; c++) { row[c] = 0.1 + rng.NextDouble(); sum += row[c]; }
            for (int c = 0; c < Classes; c++) targets[b * Classes + c] = row[c] / sum;
        }
        AssertGradientMatchesFiniteDifferences(logits, targets, "normalised");
    }

    /// <summary>
    /// The case the sweep failed on: rows that do NOT sum to 1. The forward happily accepts these,
    /// so the reported gradient must be the true derivative rather than assuming normalisation.
    /// </summary>
    [Fact]
    public void UnnormalisedSoftTargets_GradientMatchesFiniteDifferences()
    {
        var logits = Logits(33);
        var targets = new Tensor<double>([Batch, Classes]);
        var rng = new Random(44);
        for (int i = 0; i < targets.Length; i++) targets[i] = 0.15 + rng.NextDouble() * 0.7;
        AssertGradientMatchesFiniteDifferences(logits, targets, "unnormalised");
    }

    [Fact]
    public void OneHotTargets_GradientMatchesFiniteDifferences()
    {
        var logits = Logits(55);
        var targets = new Tensor<double>([Batch, Classes]);
        for (int b = 0; b < Batch; b++) targets[b * Classes + (b % Classes)] = 1.0;
        AssertGradientMatchesFiniteDifferences(logits, targets, "one-hot");
    }

    /// <summary>Sparse class-index targets — previously threw on a [n,C] vs [n] shape mismatch.</summary>
    [Fact]
    public void SparseClassIndexTargets_GradientMatchesFiniteDifferences()
    {
        var logits = Logits(66);
        var targets = new Tensor<double>([Batch]);
        for (int b = 0; b < Batch; b++) targets[b] = b % Classes;
        AssertGradientMatchesFiniteDifferences(logits, targets, "sparse");
    }

    /// <summary>
    /// One-hot targets sum to exactly 1.0, and multiplying by exactly 1.0 is bit-exact, so the
    /// corrected gradient must equal the classic (softmax - target)/n formula bit-for-bit. This pins
    /// that the fix did not perturb the normalised case every real caller uses.
    /// </summary>
    [Fact]
    public void OneHotTargets_AreBitIdenticalTo_SoftmaxMinusTargetOverN()
    {
        var logits = Logits(77);
        var targets = new Tensor<double>([Batch, Classes]);
        for (int b = 0; b < Batch; b++) targets[b * Classes + (b % Classes)] = 1.0;

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine); // see AssertGradientMatchesFiniteDifferences
        var loss = _engine.TensorCrossEntropyLoss(logits, targets);
        var grads = tape.ComputeGradients(loss, [logits]);
        var g = grads[logits];

        // Match the implementation's association exactly: it scales by a PRECOMPUTED
        // gradOutput/n (as the previous implementation also did), and x * (1.0/n) differs
        // from x / n by 1 ULP.
        var softmax = _engine.TensorSoftmax(logits, 1);
        double scale = 1.0 / Batch;
        for (int i = 0; i < logits.Length; i++)
        {
            double expected = (softmax[i] - targets[i]) * scale;
            Assert.Equal(expected, g![i]);
        }
    }
}
