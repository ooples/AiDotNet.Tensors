using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Causal attention through the decomposed primitive chain (non-float element types, and every type on the GPU engine)
/// must be capturable in an inference graph: it used to build a boolean mask for ScaledDotProductAttention, a
/// heterogeneous input an inference capture rejects, so the compiled plan was dropped and inference ran eagerly. The
/// chain now masks with a same-typed additive bias; these tests pin that it compiles, matches an independent reference
/// (bottom-right aligned causality for seqQ != seqK, zero rows that see no key) and differentiates correctly.
/// </summary>
public class CausalAttentionCaptureTests
{
    public static TheoryData<int, int, int, int, int, int> CausalShapes => new()
    {
        // batch, seqQ, seqK, heads, headDim, valueDim
        { 2, 9, 9, 2, 8, 8 },
        { 1, 5, 8, 2, 4, 6 },   // seqQ < seqK: every row sees seqK - seqQ extra keys
        { 1, 6, 4, 1, 3, 3 },   // seqQ > seqK: the first two query rows see no key and output zeros
    };

    [Theory]
    [MemberData(nameof(CausalShapes))]
    public void Double_CausalCore_CompilesInInferenceCapture_AndMatchesReference(
        int batch, int seqQ, int seqK, int heads, int hd, int vd)
    {
        var engine = new CpuEngine();
        var (q, k, v, _) = Inputs(batch, seqQ, seqK, heads, hd, vd, seed: 11);
        var eager = engine.MultiHeadAttentionCore(D(q), D(k), D(v), heads, causal: true).ToArray();
        var expected = Reference(q, k, v, null, heads, out _, out _, out _);
        AssertClose(expected, eager, 1e-10, "eager");

        var qIn = D(q); var kIn = D(k); var vIn = D(v);
        CompiledInferencePlan<double> plan;
        using (var scope = GraphMode.EnableInference())
        {
            var y = engine.MultiHeadAttentionCore(qIn, kIn, vIn, heads, causal: true);
            plan = scope.CompileInference(y, new[] { qIn, kIn, vIn });
        }
        using (plan)
        {
            AssertClose(expected, plan.Execute().ToArray(), 1e-10, "compiled");
            // A replay on new inputs recomputes rather than returning a frozen constant.
            var (q2, k2, v2, _) = Inputs(batch, seqQ, seqK, heads, hd, vd, seed: 12);
            plan.SetInputs(new[] { D(q2), D(k2), D(v2) });
            AssertClose(Reference(q2, k2, v2, null, heads, out _, out _, out _), plan.Execute().ToArray(), 1e-10,
                "compiled replay");
        }
    }

    [Theory]
    [MemberData(nameof(CausalShapes))]
    public void Half_CausalCore_CompilesInInferenceCapture_AndMatchesReference(
        int batch, int seqQ, int seqK, int heads, int hd, int vd)
    {
        var engine = new CpuEngine();
        var (q, k, v, _) = Inputs(batch, seqQ, seqK, heads, hd, vd, seed: 13);
        Tensor<Half> qh = H(q), kh = H(k), vh = H(v);
        // The reference reads the Half-rounded values, so only Half arithmetic error remains.
        var expected = Reference(F(qh), F(kh), F(vh), null, heads, out _, out _, out _);
        AssertClose(expected, ToDouble(engine.MultiHeadAttentionCore(qh, kh, vh, heads, causal: true)), 2e-2, "eager");

        CompiledInferencePlan<Half> plan;
        using (var scope = GraphMode.EnableInference())
        {
            var y = engine.MultiHeadAttentionCore(qh, kh, vh, heads, causal: true);
            plan = scope.CompileInference(y, new[] { qh, kh, vh });
        }
        using (plan)
            AssertClose(expected, ToDouble(plan.Execute()), 2e-2, "compiled");
    }

    /// <summary>ScaledDotProductAttention(isCausal: true) with no supplied mask takes the same chain off the fused
    /// float kernel (top-left aligned there), so it must capture too.</summary>
    [Fact]
    public void Double_CausalSdpa_WithoutMask_CompilesInInferenceCapture_AndMatchesEager()
    {
        var engine = new CpuEngine();
        var rng = new Random(21);
        Tensor<double> Make(int s, int d)
        {
            var a = new double[2 * 2 * s * d];
            for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
            return new Tensor<double>(a, new[] { 2, 2, s, d });
        }
        var q = Make(5, 4); var k = Make(7, 4); var v = Make(7, 3);
        var eager = engine.ScaledDotProductAttention(q, k, v, isCausal: true).ToArray();

        // Independent check of the eager result: top-left causal (key j visible when j <= i).
        var expected = new double[2 * 2 * 5 * 3];
        double scale = 1.0 / Math.Sqrt(4);
        double[] qa = q.ToArray(), ka = k.ToArray(), va = v.ToArray();
        for (int bh = 0; bh < 4; bh++)
            for (int i = 0; i < 5; i++)
            {
                var p = new double[i + 1];
                double max = double.NegativeInfinity, sum = 0;
                for (int j = 0; j <= i; j++)
                {
                    for (int d = 0; d < 4; d++) p[j] += qa[(bh * 5 + i) * 4 + d] * ka[(bh * 7 + j) * 4 + d];
                    p[j] *= scale;
                    max = Math.Max(max, p[j]);
                }
                for (int j = 0; j <= i; j++) sum += p[j] = Math.Exp(p[j] - max);
                for (int j = 0; j <= i; j++)
                    for (int e = 0; e < 3; e++) expected[(bh * 5 + i) * 3 + e] += p[j] / sum * va[(bh * 7 + j) * 3 + e];
            }
        AssertClose(expected, eager, 1e-10, "eager");

        CompiledInferencePlan<double> plan;
        using (var scope = GraphMode.EnableInference())
        {
            var y = engine.ScaledDotProductAttention(q, k, v, isCausal: true);
            plan = scope.CompileInference(y, new[] { q, k, v });
        }
        using (plan)
            AssertClose(expected, plan.Execute().ToArray(), 1e-10, "compiled");
    }

    /// <summary>
    /// Tape gradients of the causal decomposed chain against central finite differences of the forward (and the
    /// analytic reference), including seqQ != seqK and rows that see no key, which must pass zero, not NaN.
    /// </summary>
    [Theory]
    [MemberData(nameof(CausalShapes))]
    public void Double_CausalCore_TapeGradients_MatchFiniteDifferences(
        int batch, int seqQ, int seqK, int heads, int hd, int vd)
    {
        var engine = new CpuEngine();
        var (qf, kf, vf, gf) = Inputs(batch, seqQ, seqK, heads, hd, vd, seed: 31);
        Tensor<double> q = D(qf), k = D(kf), v = D(vf), g = D(gf);

        double[][] grads;
        using (var tape = new GradientTape<double>())
        {
            var output = engine.MultiHeadAttentionCore(q, k, v, heads, causal: true);
            var loss = engine.ReduceSum(engine.TensorMultiply(output, g), null);
            var all = tape.ComputeGradients(loss, new[] { q, k, v });
            grads = new[] { all[q].ToArray(), all[k].ToArray(), all[v].ToArray() };
        }

        Reference(qf, kf, vf, gf, heads, out var dq, out var dk, out var dv);
        AssertClose(dq, grads[0], 1e-10, "dQ vs reference");
        AssertClose(dk, grads[1], 1e-10, "dK vs reference");
        AssertClose(dv, grads[2], 1e-10, "dV vs reference");

        var inputs = new[] { q, k, v };
        var names = new[] { "dQ", "dK", "dV" };
        double[] ga = g.ToArray();
        double Loss()
        {
            var o = engine.MultiHeadAttentionCore(q, k, v, heads, causal: true).ToArray();
            double s = 0;
            for (int i = 0; i < o.Length; i++) s += o[i] * ga[i];
            return s;
        }
        const double h = 1e-6;
        for (int t = 0; t < 3; t++)
        {
            var fd = new double[inputs[t].Length];
            for (int i = 0; i < fd.Length; i++)
            {
                double x = inputs[t][i];
                inputs[t][i] = x + h; double up = Loss();
                inputs[t][i] = x - h; double down = Loss();
                inputs[t][i] = x;
                fd[i] = (up - down) / (2 * h);
            }
            AssertClose(fd, grads[t], 1e-6, names[t] + " vs finite differences");
        }
    }

    // ---- helpers ----

    internal static (Tensor<float> Q, Tensor<float> K, Tensor<float> V, Tensor<float> Upstream) Inputs(
        int batch, int seqQ, int seqK, int heads, int hd, int vd, int seed)
    {
        var rng = new Random(seed);
        Tensor<float> Make(int seq, int width)
        {
            var data = new float[batch * seq * width];
            for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            return new Tensor<float>(data, new[] { batch, seq, width });
        }
        return (Make(seqQ, heads * hd), Make(seqK, heads * hd), Make(seqK, heads * vd), Make(seqQ, heads * vd));
    }

    private static Tensor<double> D(Tensor<float> t) => new(Array.ConvertAll(t.ToArray(), x => (double)x), t.Shape.ToArray());
    private static Tensor<Half> H(Tensor<float> t) => new(Array.ConvertAll(t.ToArray(), x => (Half)x), t.Shape.ToArray());
    private static Tensor<float> F(Tensor<Half> t) => new(Array.ConvertAll(t.ToArray(), x => (float)x), t.Shape.ToArray());
    private static double[] ToDouble(Tensor<Half> t) => Array.ConvertAll(t.ToArray(), x => (double)(float)x);

    /// <summary>Causal (bottom-right aligned) double-precision attention and, given an upstream, its gradients.</summary>
    internal static double[] Reference(
        Tensor<float> q, Tensor<float> k, Tensor<float> v, Tensor<float>? upstream, int heads,
        out double[] dq, out double[] dk, out double[] dv)
    {
        int batch = q.Shape[0], seqQ = q.Shape[1], seqK = k.Shape[1];
        int qw = q.Shape[2], vw = v.Shape[2], hd = qw / heads, vd = vw / heads;
        double scale = 1.0 / Math.Sqrt(hd);
        float[] qa = q.ToArray(), ka = k.ToArray(), va = v.ToArray();
        float[]? ga = upstream?.ToArray();
        var output = new double[batch * seqQ * vw];
        dq = new double[qa.Length]; dk = new double[ka.Length]; dv = new double[va.Length];
        var p = new double[seqK];
        var dp = new double[seqK];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < seqQ; i++)
                {
                    int limit = Math.Max(0, Math.Min(seqK, i + 1 + seqK - seqQ));
                    if (limit == 0) continue;
                    int qRow = (b * seqQ + i) * qw + h * hd;
                    double max = double.NegativeInfinity;
                    for (int j = 0; j < limit; j++)
                    {
                        int kRow = (b * seqK + j) * qw + h * hd;
                        double s = 0;
                        for (int d = 0; d < hd; d++) s += (double)qa[qRow + d] * ka[kRow + d];
                        p[j] = s * scale;
                        max = Math.Max(max, p[j]);
                    }
                    double sum = 0;
                    for (int j = 0; j < limit; j++) { p[j] = Math.Exp(p[j] - max); sum += p[j]; }
                    for (int j = 0; j < limit; j++) p[j] /= sum;
                    int oRow = (b * seqQ + i) * vw + h * vd;
                    for (int j = 0; j < limit; j++)
                    {
                        int vRow = (b * seqK + j) * vw + h * vd;
                        for (int e = 0; e < vd; e++) output[oRow + e] += p[j] * va[vRow + e];
                    }
                    if (ga is null) continue;
                    double weighted = 0;
                    for (int j = 0; j < limit; j++)
                    {
                        int vRow = (b * seqK + j) * vw + h * vd;
                        dp[j] = 0;
                        for (int e = 0; e < vd; e++)
                        {
                            dp[j] += ga[oRow + e] * (double)va[vRow + e];
                            dv[vRow + e] += p[j] * ga[oRow + e];
                        }
                        weighted += p[j] * dp[j];
                    }
                    for (int j = 0; j < limit; j++)
                    {
                        double ds = p[j] * (dp[j] - weighted) * scale;
                        int kRow = (b * seqK + j) * qw + h * hd;
                        for (int d = 0; d < hd; d++)
                        {
                            dq[qRow + d] += ds * ka[kRow + d];
                            dk[kRow + d] += ds * qa[qRow + d];
                        }
                    }
                }
        return output;
    }

    internal static void AssertClose(double[] expected, double[] actual, double tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxErr = 0;
        int worst = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(double.IsNaN(actual[i]) || double.IsInfinity(actual[i]), $"{what}[{i}] is {actual[i]}");
            double err = Math.Abs(expected[i] - actual[i]);
            if (err > maxErr) { maxErr = err; worst = i; }
        }
        Assert.True(maxErr <= tol, $"{what}: max |error| {maxErr:G4} at {worst} (tolerance {tol:G3})");
    }
}

/// <summary>
/// The GPU engine always takes the decomposed chain; a causal MultiHeadAttentionCore must compile into an inference
/// plan there (it used to throw GraphCaptureNotSupportedException and fall back to eager) and match the reference.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class CausalAttentionCaptureGpuTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public CausalAttentionCaptureGpuTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private sealed class EngineScope : IDisposable
    {
        private readonly IEngine _prior = AiDotNetEngine.Current;
        public EngineScope(IEngine engine) => AiDotNetEngine.Current = engine;
        public void Dispose() => AiDotNetEngine.Current = _prior;
    }

    [SkippableTheory]
    [MemberData(nameof(CausalAttentionCaptureTests.CausalShapes), MemberType = typeof(CausalAttentionCaptureTests))]
    public void Gpu_CausalCore_CompilesInInferenceCapture_AndMatchesReference(
        int batch, int seqQ, int seqK, int heads, int hd, int vd)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var (q, k, v, _) = CausalAttentionCaptureTests.Inputs(batch, seqQ, seqK, heads, hd, vd, seed: 41);
        var expected = CausalAttentionCaptureTests.Reference(q, k, v, null, heads, out _, out _, out _);

        CompiledInferencePlan<float> plan;
        using (new EngineScope(gpu))
        using (var scope = GraphMode.EnableInference())
        {
            var y = gpu.MultiHeadAttentionCore(q, k, v, heads, causal: true);
            plan = scope.CompileInference(y, new[] { q, k, v });
        }
        using (plan)
        {
            CausalAttentionCaptureTests.AssertClose(expected, Array.ConvertAll(plan.Execute().ToArray(), x => (double)x),
                1e-4, "compiled");
            var (q2, k2, v2, _) = CausalAttentionCaptureTests.Inputs(batch, seqQ, seqK, heads, hd, vd, seed: 42);
            plan.SetInputs(new[] { q2, k2, v2 });
            CausalAttentionCaptureTests.AssertClose(
                CausalAttentionCaptureTests.Reference(q2, k2, v2, null, heads, out _, out _, out _),
                Array.ConvertAll(plan.Execute().ToArray(), x => (double)x), 1e-4, "compiled replay");
        }
    }
}
