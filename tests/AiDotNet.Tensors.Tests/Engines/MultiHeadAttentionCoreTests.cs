using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Covers <see cref="IEngine.MultiHeadAttentionCore{T}"/>, the fused attention over head-interleaved projections,
/// against an independent double-precision transcription of softmax(scale * Q K^T) V and its gradients.
/// </summary>
/// <remarks>
/// The shapes cross the kernel's tile edges (32 query rows, 64 key columns), use head sizes that are not a multiple of
/// the vector width, unequal query and key lengths, and causal masking, so the online softmax across key blocks, the
/// scalar tails and the masked tiles all run. Gradients are checked per input with a non-uniform upstream gradient:
/// a uniform one would hide a wrong softmax backward, because the rows of a softmax Jacobian sum to zero.
/// </remarks>
public class MultiHeadAttentionCoreTests
{
    public static TheoryData<int, int, int, int, int, int, bool> Shapes => new()
    {
        // batch, seqQ, seqK, heads, headDim, valueDim, causal
        { 2, 5, 7, 3, 5, 4, false },
        { 2, 9, 9, 2, 16, 16, true },
        { 1, 40, 70, 4, 16, 8, false },
        { 3, 33, 33, 2, 8, 8, true },
        { 1, 70, 130, 2, 12, 20, true },
        { 1, 6, 4, 1, 3, 3, true },     // causal with seqQ > seqK: the first two query rows see no key
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public async Task Forward_MatchesReference(int batch, int seqQ, int seqK, int heads, int hd, int vd, bool causal)
    {
        await Task.Yield();
        var (q, k, v, _) = Inputs(batch, seqQ, seqK, heads, hd, vd, seed: 1);
        var engine = new CpuEngine();

        var output = engine.MultiHeadAttentionCore(q, k, v, heads, causal: causal);

        var expected = Reference(q, k, v, null, heads, causal, out _, out _, out _);
        AssertClose(expected, output.ToArray(), "output");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public async Task TapeGradients_MatchReference(int batch, int seqQ, int seqK, int heads, int hd, int vd, bool causal)
    {
        await Task.Yield();
        var (q, k, v, upstream) = Inputs(batch, seqQ, seqK, heads, hd, vd, seed: 2);
        var engine = new CpuEngine();

        using var tape = new GradientTape<float>();
        var output = engine.MultiHeadAttentionCore(q, k, v, heads, causal: causal);
        var loss = engine.ReduceSum(engine.TensorMultiply(output, upstream), null);
        var grads = tape.ComputeGradients(loss, new[] { q, k, v });

        Reference(q, k, v, upstream, heads, causal, out var dq, out var dk, out var dv);
        AssertClose(dq, grads[q].ToArray(), "dQ");
        AssertClose(dk, grads[k].ToArray(), "dK");
        AssertClose(dv, grads[v].ToArray(), "dV");
    }

    /// <summary>
    /// Two steps of a compiled plan: the second must give the same gradients as the first (the plan reuses its
    /// gradient buffers, so a backward that adds onto the previous step's values instead of starting fresh shows here).
    /// </summary>
    [Theory]
    [MemberData(nameof(Shapes))]
    public async Task CompiledPlanGradients_MatchReference_OnEveryStep(
        int batch, int seqQ, int seqK, int heads, int hd, int vd, bool causal)
    {
        await Task.Yield();
        var (q, k, v, upstream) = Inputs(batch, seqQ, seqK, heads, hd, vd, seed: 3);
        var engine = new CpuEngine();
        Reference(q, k, v, upstream, heads, causal, out var dq, out var dk, out var dv);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.MultiHeadAttentionCore(q, k, v, heads, causal: causal);
            engine.ReduceSum(engine.TensorMultiply(output, upstream), null);
            plan = scope.CompileTraining(new[] { q, k, v });
        }

        using (plan)
        {
            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                AssertClose(dq, plan.Gradients[0].ToArray(), $"dQ step {step}");
                AssertClose(dk, plan.Gradients[1].ToArray(), $"dK step {step}");
                AssertClose(dv, plan.Gradients[2].ToArray(), $"dV step {step}");
            }
        }
    }

    /// <summary>Self-attention on one tensor: its gradient is the sum of the query, key and value contributions.</summary>
    [Fact]
    public async Task SameTensorAsQueryKeyAndValue_SumsAllThreeGradients()
    {
        await Task.Yield();
        const int batch = 2, seq = 11, heads = 2, hd = 6;
        var (x, _, _, upstream) = Inputs(batch, seq, seq, heads, hd, hd, seed: 4);
        var engine = new CpuEngine();

        using var tape = new GradientTape<float>();
        var output = engine.MultiHeadAttentionCore(x, x, x, heads, causal: true);
        var loss = engine.ReduceSum(engine.TensorMultiply(output, upstream), null);
        var grad = tape.ComputeGradients(loss, new[] { x })[x].ToArray();

        Reference(x, x, x, upstream, heads, true, out var dq, out var dk, out var dv);
        var expected = new double[dq.Length];
        for (int i = 0; i < expected.Length; i++) expected[i] = dq[i] + dk[i] + dv[i];
        AssertClose(expected, grad, "dX");
    }

    [Fact]
    public async Task RejectsWidthNotDivisibleByHeads()
    {
        await Task.Yield();
        var (q, k, v, _) = Inputs(1, 3, 3, 1, 5, 5, seed: 5);
        Assert.Throws<ArgumentException>(() => new CpuEngine().MultiHeadAttentionCore(q, k, v, numHeads: 2));
    }

    private static (Tensor<float> Q, Tensor<float> K, Tensor<float> V, Tensor<float> Upstream) Inputs(
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

    /// <summary>Plain double-precision attention and, when <paramref name="upstream"/> is given, its gradients.</summary>
    private static double[] Reference(
        Tensor<float> q, Tensor<float> k, Tensor<float> v, Tensor<float>? upstream, int heads, bool causal,
        out double[] dq, out double[] dk, out double[] dv)
    {
        int batch = q.Shape[0], seqQ = q.Shape[1], seqK = k.Shape[1];
        int qw = q.Shape[2], vw = v.Shape[2], hd = qw / heads, vd = vw / heads;
        double scale = 1.0 / Math.Sqrt(hd);
        float[] qa = q.ToArray(), ka = k.ToArray(), va = v.ToArray();
        float[]? ga = upstream?.ToArray();
        var output = new double[batch * seqQ * vw];
        dq = new double[qa.Length];
        dk = new double[ka.Length];
        dv = new double[va.Length];
        var p = new double[seqK];
        var dp = new double[seqK];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < seqQ; i++)
                {
                    // Causal is bottom-right aligned: key j is visible when j <= i + (seqK - seqQ).
                    int limit = causal ? Math.Max(0, Math.Min(seqK, i + 1 + seqK - seqQ)) : seqK;
                    if (limit == 0) continue;   // sees no key: zero output, zero gradients
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

    private static void AssertClose(double[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxErr = 0, maxRef = 0;
        int worst = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]) || float.IsInfinity(actual[i]), $"{what}[{i}] is {actual[i]}");
            double err = Math.Abs(expected[i] - actual[i]);
            if (err > maxErr) { maxErr = err; worst = i; }
            maxRef = Math.Max(maxRef, Math.Abs(expected[i]));
        }
        Assert.True(maxErr <= 1e-5 + 1e-4 * maxRef,
            $"{what}: max |error| {maxErr:G4} at {worst} (expected {(worst >= 0 ? expected[worst] : 0):G6}, "
            + $"got {(worst >= 0 ? actual[worst] : 0):G6}); max |reference| {maxRef:G4}");
    }
}
