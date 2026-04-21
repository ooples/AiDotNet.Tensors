using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Integration coverage for issue #233 — <c>CpuEngine.BatchNormBackward</c>
/// on rank-3 inputs must survive realistic training patterns: multi-step
/// SGD, shape variety (including the exact shape that originally
/// crashed), cross-check vs the equivalent 4D path, several epsilon
/// regimes, repeat-backward across tapes, and both floating-point
/// precisions.
/// </summary>
public class BatchNorm3DIntegrationTests : IDisposable
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private readonly bool _previousReplayMode;

    public BatchNorm3DIntegrationTests()
    {
        // Capture AutoTrainingCompiler.ReplayMode so Dispose can restore
        // the process-wide value; leaving it pinned at false would
        // shadow replay-mode coverage in any subsequent test class.
        _previousReplayMode = AutoTrainingCompiler.ReplayMode;
        AutoTrainingCompiler.ReplayMode = false;
    }

    public void Dispose() => AutoTrainingCompiler.ReplayMode = _previousReplayMode;

    /// <summary>
    /// Training loop: `[C, H, W] → BatchNorm → ReLU → MSE loss`. Verify
    /// the full forward → backward → SGD cycle runs without throwing
    /// and that the loss trends downward.
    /// </summary>
    [Theory]
    [InlineData(1, 8, 24)]   // exact crash shape from #233 (H > C)
    [InlineData(3, 4, 5)]    // canonical small
    [InlineData(8, 8, 8)]    // square channels/spatial
    [InlineData(16, 7, 7)]   // ResNet-block-ish
    [InlineData(64, 1, 1)]   // degenerate 1×1 spatial — pure per-channel
    public void BatchNorm3DTrainingLoop_DescendsLoss(int C, int H, int W)
    {
        var rng = new Random(42);
        var input = NewRandomDouble(new[] { C, H, W }, rng, 0.8);
        var gamma = NewOnesDouble(new[] { C });
        var beta = NewRandomDouble(new[] { C }, rng, 0.05);
        var target = NewRandomDouble(new[] { C, H, W }, rng, 0.5);

        const double lr = 0.01;
        double initialLoss = 0, finalLoss = 0;
        for (int step = 0; step < 6; step++)
        {
            using var tape = new GradientTape<double>();
            var y = _engine.BatchNorm(input, gamma, beta, 1e-5, out _, out _);
            var r = _engine.TensorReLU(y);
            var diff = _engine.TensorSubtract(r, target);
            var sq = _engine.TensorMultiply(diff, diff);
            var loss = _engine.ReduceSum(sq, null);
            var grads = tape.ComputeGradients(loss, new[] { gamma, beta });

            Assert.Equal(gamma._shape, grads[gamma]._shape);
            Assert.Equal(beta._shape, grads[beta]._shape);

            if (step == 0) initialLoss = loss.AsSpan()[0];
            finalLoss = loss.AsSpan()[0];

            ApplyGradDouble(gamma, grads[gamma], lr);
            ApplyGradDouble(beta, grads[beta], lr);
        }
        Assert.True(finalLoss < initialLoss,
            $"Shape [{C},{H},{W}]: loss {initialLoss:G4} → {finalLoss:G4} (should have dropped)");
    }

    /// <summary>
    /// Epsilon stress: ensure the backward formula doesn't blow up for
    /// small or large eps. Values span three orders of magnitude, each
    /// must produce a finite, shape-correct gradient set.
    /// </summary>
    [Theory]
    [InlineData(1e-8)]
    [InlineData(1e-5)]
    [InlineData(1e-3)]
    [InlineData(1e-1)]
    public void BatchNorm3DBackward_Epsilon_ProducesFiniteGradients(double epsilon)
    {
        const int C = 3, H = 4, W = 5;
        var rng = new Random(1);
        var input = NewRandomDouble(new[] { C, H, W }, rng, 1.0);
        var gamma = NewRandomDouble(new[] { C }, rng, 1.0);
        var beta = NewRandomDouble(new[] { C }, rng, 0.5);

        using var tape = new GradientTape<double>();
        var y = _engine.BatchNorm(input, gamma, beta, epsilon, out _, out _);
        var loss = _engine.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { input, gamma, beta });

        AssertAllFinite(grads[input].AsSpan(), $"input-grad @ eps={epsilon}");
        AssertAllFinite(grads[gamma].AsSpan(), $"gamma-grad @ eps={epsilon}");
        AssertAllFinite(grads[beta].AsSpan(), $"beta-grad @ eps={epsilon}");
    }

    /// <summary>
    /// A <c>[C, H, W]</c> rank-3 tensor and the equivalent
    /// <c>[1, C, H, W]</c> rank-4 tensor (reshape-only) must produce the
    /// same per-channel γ/β gradients and element-wise-equal input
    /// gradients once the singleton batch axis is dropped. This pins
    /// the 3D backward formula to the established 4D path.
    /// </summary>
    [Fact]
    public void BatchNorm3D_And_4DSingletonBatch_ProduceEquivalentGradients()
    {
        const int C = 4, H = 3, W = 5;
        var rng = new Random(7);
        var inputData = new double[C * H * W];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = rng.NextDouble() * 2 - 1;
        var gammaData = new double[C];
        for (int i = 0; i < C; i++) gammaData[i] = 0.5 + rng.NextDouble();
        var betaData = new double[C];
        for (int i = 0; i < C; i++) betaData[i] = rng.NextDouble() - 0.5;

        // 3D path — independent scope.
        var input3 = new Tensor<double>(inputData.AsSpan().ToArray(), new[] { C, H, W });
        var gamma3 = new Tensor<double>((double[])gammaData.Clone(), new[] { C });
        var beta3 = new Tensor<double>((double[])betaData.Clone(), new[] { C });
        Tensor<double> gInput3, gGamma3, gBeta3;
        using (var tape = new GradientTape<double>())
        {
            var y = _engine.BatchNorm(input3, gamma3, beta3, 1e-5, out _, out _);
            var loss = _engine.ReduceSum(y, null);
            var grads = tape.ComputeGradients(loss, new[] { input3, gamma3, beta3 });
            gInput3 = grads[input3];
            gGamma3 = grads[gamma3];
            gBeta3 = grads[beta3];
        }

        // 4D path — independent scope, separate tape. Reshape-only copy
        // of the same data [C, H, W] → [1, C, H, W] drives the
        // established 4D backward kernel.
        var input4 = new Tensor<double>(inputData.AsSpan().ToArray(), new[] { 1, C, H, W });
        var gamma4 = new Tensor<double>((double[])gammaData.Clone(), new[] { C });
        var beta4 = new Tensor<double>((double[])betaData.Clone(), new[] { C });
        Tensor<double> gInput4, gGamma4, gBeta4;
        using (var tape4 = new GradientTape<double>())
        {
            var y4 = _engine.BatchNorm(input4, gamma4, beta4, 1e-5, out _, out _);
            var loss4 = _engine.ReduceSum(y4, null);
            var grads4 = tape4.ComputeGradients(loss4, new[] { input4, gamma4, beta4 });
            gInput4 = grads4[input4];
            gGamma4 = grads4[gamma4];
            gBeta4 = grads4[beta4];
        }

        // γ + β grads: same shape, direct compare.
        AssertClose(gGamma3.AsSpan(), gGamma4.AsSpan(), "gamma", 1e-9);
        AssertClose(gBeta3.AsSpan(), gBeta4.AsSpan(), "beta", 1e-9);

        // Input grads: 3D is [C, H, W], 4D is [1, C, H, W]. Same underlying
        // element count; element-by-element should match within FP tolerance.
        Assert.Equal(gInput3.Length, gInput4.Length);
        AssertClose(gInput3.AsSpan(), gInput4.AsSpan(), "input", 1e-9);
    }

    /// <summary>
    /// Repeated backward on fresh tapes for the same inputs must produce
    /// byte-identical gradients — guards against any per-call state
    /// leaking through the per-channel accumulators.
    /// </summary>
    [Fact]
    public void BatchNorm3DBackward_RepeatedForSameInputs_IsDeterministic()
    {
        var input = NewRandomDouble(new[] { 2, 8, 24 }, new Random(11), 0.5);
        var gamma = NewRandomDouble(new[] { 2 }, new Random(12), 1.0);
        var beta = NewRandomDouble(new[] { 2 }, new Random(13), 0.5);

        double[] firstInputGrad = Array.Empty<double>();
        double[] firstGammaGrad = Array.Empty<double>();
        double[] firstBetaGrad = Array.Empty<double>();
        for (int i = 0; i < 3; i++)
        {
            using var tape = new GradientTape<double>();
            var y = _engine.BatchNorm(input, gamma, beta, 1e-5, out _, out _);
            var loss = _engine.ReduceSum(y, null);
            var grads = tape.ComputeGradients(loss, new[] { input, gamma, beta });
            var iSpan = grads[input].AsSpan();
            var gSpan = grads[gamma].AsSpan();
            var bSpan = grads[beta].AsSpan();
            if (i == 0)
            {
                firstInputGrad = iSpan.ToArray();
                firstGammaGrad = gSpan.ToArray();
                firstBetaGrad = bSpan.ToArray();
            }
            else
            {
                for (int j = 0; j < iSpan.Length; j++)
                    Assert.Equal(firstInputGrad[j], iSpan[j]);
                for (int j = 0; j < gSpan.Length; j++)
                    Assert.Equal(firstGammaGrad[j], gSpan[j]);
                for (int j = 0; j < bSpan.Length; j++)
                    Assert.Equal(firstBetaGrad[j], bSpan[j]);
            }
        }
    }

    /// <summary>
    /// Float coverage of the exact crash shape from the issue — confirm
    /// the fix isn't double-only.
    /// </summary>
    [Fact]
    public void BatchNorm3DBackward_Float_IssueCrashShape_Succeeds()
    {
        var input = new Tensor<float>(new[] { 1, 8, 24 });
        var gamma = new Tensor<float>(new[] { 1 });
        var beta = new Tensor<float>(new[] { 1 });
        for (int i = 0; i < input.Length; i++) input.AsWritableSpan()[i] = (float)(Math.Sin(i) * 0.3);
        gamma.AsWritableSpan()[0] = 1.2f;
        beta.AsWritableSpan()[0] = -0.4f;

        using var tape = new GradientTape<float>();
        var y = _engine.BatchNorm(input, gamma, beta, 1e-5, out _, out _);
        var loss = _engine.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { input, gamma, beta });

        Assert.Equal(input._shape, grads[input]._shape);
        Assert.Equal(gamma._shape, grads[gamma]._shape);
        Assert.Equal(beta._shape, grads[beta]._shape);
        AssertAllFinite(grads[input].AsSpan(), "float input-grad");
        AssertAllFinite(grads[gamma].AsSpan(), "float gamma-grad");
        AssertAllFinite(grads[beta].AsSpan(), "float beta-grad");
    }

    // ---- helpers ---------------------------------------------------------

    private static Tensor<double> NewRandomDouble(int[] shape, Random rng, double scale)
    {
        var t = new Tensor<double>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (rng.NextDouble() - 0.5) * 2 * scale;
        return t;
    }

    private static Tensor<double> NewOnesDouble(int[] shape)
    {
        var t = new Tensor<double>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = 1.0;
        return t;
    }

    private static void ApplyGradDouble(Tensor<double> param, Tensor<double> grad, double lr)
    {
        var p = param.AsWritableSpan();
        var g = grad.AsSpan();
        for (int i = 0; i < p.Length; i++) p[i] -= lr * g[i];
    }

    private static void AssertAllFinite(ReadOnlySpan<double> span, string label)
    {
        for (int i = 0; i < span.Length; i++)
            Assert.True(double.IsFinite(span[i]),
                $"{label}: non-finite at [{i}] = {span[i]}");
    }

    private static void AssertAllFinite(ReadOnlySpan<float> span, string label)
    {
        for (int i = 0; i < span.Length; i++)
            Assert.True(float.IsFinite(span[i]),
                $"{label}: non-finite at [{i}] = {span[i]}");
    }

    private static void AssertClose(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual, string label, double tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < tol,
                $"{label}[{i}]: expected {expected[i]:G9}, actual {actual[i]:G9}");
    }
}
