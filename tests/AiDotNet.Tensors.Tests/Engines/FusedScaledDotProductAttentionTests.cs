using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #889: the weightless, tape-aware ScaledDotProductAttention overload. Its fused FlashAttention-2 CPU float32 path must
/// equal the materialising attention (forward and query / key / value gradients) for causal, supplied and broadcast
/// masks, scale, soft-cap and strided (permuted) inputs; agree with finite differences; zero a row with no allowed key;
/// be independent of the thread count; and train without allocating the [B, H, S, S] attention weights.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class FusedScaledDotProductAttentionTests
{
    private static Tensor<T> Random<T>(int salt, params int[] shape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var rng = new Random(salt);
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = ops.FromDouble((rng.NextDouble() - 0.5) * 2);
        return t;
    }

    // A random mask where every query row keeps at least one key.
    private static Tensor<bool> RandomMask(int salt, params int[] shape)
    {
        var rng = new Random(salt);
        var m = new Tensor<bool>(shape);
        int keys = shape[shape.Length - 1];
        for (int i = 0; i < m.Length; i++) m[i] = rng.NextDouble() < 0.6 || i % keys == (i / keys) % keys;
        return m;
    }

    // The fully expanded [B, H, Sq, Sk] mask the materialising reference takes.
    private static Tensor<bool>? Expand(Tensor<bool>? mask, int b, int h, int sq, int sk, bool causal)
    {
        if (mask is null && !causal) return null;
        var full = new Tensor<bool>(new[] { b, h, sq, sk });
        int rank = mask?.Rank ?? 0;
        for (int bi = 0, at = 0; bi < b; bi++)
            for (int hi = 0; hi < h; hi++)
                for (int i = 0; i < sq; i++)
                    for (int j = 0; j < sk; j++, at++)
                    {
                        bool allowed = !causal || j <= i;
                        if (mask is not null && allowed)
                        {
                            int[] idx = { bi, hi, i, j };
                            int flat = 0;
                            for (int d = 0; d < rank; d++)
                            {
                                int axis = 4 - rank + d, extent = mask._shape[d];
                                flat = flat * extent + (extent == 1 ? 0 : idx[axis]);
                            }
                            allowed = mask[flat];
                        }
                        full[at] = allowed;
                    }
        return full;
    }

    private sealed record Case(string Name, int B, int H, int Sq, int Sk, int D, int Dv, bool Causal, string Mask,
        double Softcap, double? Scale, bool Permuted);

    public static TheoryData<string> Cases() => new()
    {
        "plain", "causal", "causalRect", "fullMask", "bcastMask2d", "bcastMaskHeads", "softcap", "scaled", "permuted",
        "everything", "longKeys",
    };

    private static Case Get(string name) => name switch
    {
        "plain" => new(name, 2, 3, 37, 37, 16, 16, false, "", 0, null, false),
        "causal" => new(name, 2, 2, 70, 70, 32, 24, true, "", 0, null, false),
        "causalRect" => new(name, 1, 2, 40, 90, 16, 16, true, "", 0, null, false),
        "fullMask" => new(name, 2, 2, 33, 65, 16, 8, false, "full", 0, null, false),
        "bcastMask2d" => new(name, 2, 3, 30, 41, 8, 8, false, "2d", 0, null, false),
        "bcastMaskHeads" => new(name, 2, 3, 30, 41, 8, 8, false, "b1", 0, null, false),
        "softcap" => new(name, 1, 2, 50, 50, 16, 16, false, "", 5.0, null, false),
        "scaled" => new(name, 1, 2, 20, 30, 16, 16, false, "", 0, 0.37, false),
        "permuted" => new(name, 2, 4, 45, 45, 16, 16, false, "", 0, null, true),
        "everything" => new(name, 2, 2, 66, 66, 16, 12, true, "2d", 3.0, 0.2, true),
        _ => new(name, 1, 1, 9, 300, 32, 32, false, "full", 0, null, false),
    };

    private static (Tensor<float> Q, Tensor<float> K, Tensor<float> V) Inputs(Case c, int salt,
        out Tensor<float> qSrc, out Tensor<float> kSrc, out Tensor<float> vSrc)
    {
        if (!c.Permuted)
        {
            qSrc = Random<float>(salt, c.B, c.H, c.Sq, c.D);
            kSrc = Random<float>(salt + 1, c.B, c.H, c.Sk, c.D);
            vSrc = Random<float>(salt + 2, c.B, c.H, c.Sk, c.Dv);
            return (qSrc, kSrc, vSrc);
        }
        // [B, S, H, D] projections viewed as [B, H, S, D]: strided, not contiguous.
        qSrc = Random<float>(salt, c.B, c.Sq, c.H, c.D);
        kSrc = Random<float>(salt + 1, c.B, c.Sk, c.H, c.D);
        vSrc = Random<float>(salt + 2, c.B, c.Sk, c.H, c.Dv);
        return (qSrc, kSrc, vSrc);
    }

    private static Tensor<float> View(IEngine e, Case c, Tensor<float> src)
        => c.Permuted ? e.TensorPermute(src, new[] { 0, 2, 1, 3 }) : src;

    private static Tensor<bool>? MaskFor(Case c) => c.Mask switch
    {
        "full" => RandomMask(91, c.B, c.H, c.Sq, c.Sk),
        "2d" => RandomMask(92, c.Sq, c.Sk),
        "b1" => RandomMask(93, c.B, 1, c.Sq, c.Sk),
        _ => null,
    };

    // Runs forward + backward of sum(w * attention) under a tape; returns the output and the source gradients.
    private static (float[] Out, float[] DQ, float[] DK, float[] DV) Run(CpuEngine e, Case c, bool fused, Tensor<bool>? mask)
    {
        var (q0, k0, v0) = Inputs(c, 11, out var qs, out var ks, out var vs);
        var w = Random<float>(17, c.B, c.H, c.Sq, c.Dv);
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = e;
        try
        {
            using var tape = new GradientTape<float>();
            var q = View(e, c, qs); var k = View(e, c, ks); var v = View(e, c, vs);
            Tensor<float> o = fused
                ? e.ScaledDotProductAttention(q, k, v, mask, c.Scale, c.Causal, c.Softcap)
                : e.ScaledDotProductAttention(q, k, v, Expand(mask, c.B, c.H, c.Sq, c.Sk, c.Causal), c.Scale, out _, c.Softcap);
            var loss = e.ReduceSum(e.TensorMultiply(o, w), new[] { 0, 1, 2, 3 }, keepDims: false);
            var g = tape.ComputeGradients(loss, new[] { qs, ks, vs });
            return (o.Contiguous().ToArray(), g[qs].ToArray(), g[ks].ToArray(), g[vs].ToArray());
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    private static void Close(string what, float[] expected, float[] actual, double tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]), $"{what}[{i}] is NaN");
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol * Math.Max(1.0, Math.Abs(expected[i])),
                $"{what}[{i}]: materialised {expected[i]:R} fused {actual[i]:R}");
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Fused_MatchesMaterialisedAttention_ForwardAndGradients(string name)
    {
        var c = Get(name);
        var e = new CpuEngine();
        var mask = MaskFor(c);
        var reference = Run(e, c, fused: false, mask);
        var fused = Run(e, c, fused: true, mask);
        Close($"{name} out", reference.Out, fused.Out, 2e-5);
        Close($"{name} dQ", reference.DQ, fused.DQ, 1e-4);
        Close($"{name} dK", reference.DK, fused.DK, 1e-4);
        Close($"{name} dV", reference.DV, fused.DV, 1e-4);
    }

    [Theory]
    [InlineData("causal")]
    [InlineData("everything")]
    public void Fused_Gradients_AgreeWithFiniteDifferences(string name)
    {
        var c = Get(name) with { B = 1, H = 1, Sq = 12, Sk = 12 };
        var e = new CpuEngine();
        var mask = c.Mask == "2d" ? RandomMask(92, c.Sq, c.Sk) : null;
        var (_, _, _) = Inputs(c, 11, out var qs, out var ks, out var vs);
        var w = Random<float>(17, c.B, c.H, c.Sq, c.Dv);
        double Loss()
        {
            var o = e.ScaledDotProductAttention(View(e, c, qs), View(e, c, ks), View(e, c, vs), mask, c.Scale, c.Causal, c.Softcap);
            double s = 0;
            var oc = o.Contiguous();
            for (int i = 0; i < oc.Length; i++) s += (double)oc[i] * w[i];
            return s;
        }
        var analytic = Run(e, c, fused: true, mask);
        const float h = 1e-2f;
        foreach (var (src, grad, label) in new[] { (qs, analytic.DQ, "q"), (ks, analytic.DK, "k"), (vs, analytic.DV, "v") })
        {
            for (int t = 0; t < 6; t++)
            {
                int idx = (t * 37 + 5) % src.Length;
                float x = src[idx];
                src[idx] = x + h; double up = Loss();
                src[idx] = x - h; double down = Loss();
                src[idx] = x;
                double numeric = (up - down) / (2 * h);
                Assert.True(Math.Abs(numeric - grad[idx]) <= 2e-2 * Math.Max(1.0, Math.Abs(numeric)),
                    $"{name} d{label}[{idx}]: finite difference {numeric:G6} vs tape {grad[idx]:G6}");
            }
        }
    }

    [Fact]
    public void Fused_RowWithNoAllowedKey_IsZero_AndPassesNoGradient()
    {
        var c = Get("fullMask") with { B = 1, H = 1 };
        var mask = RandomMask(94, 1, 1, c.Sq, c.Sk);
        for (int j = 0; j < c.Sk; j++) mask[(3 * c.Sk) + j] = false;   // query row 3 sees nothing
        var fused = Run(new CpuEngine(), c, fused: true, mask);
        for (int d = 0; d < c.Dv; d++) Assert.Equal(0f, fused.Out[3 * c.Dv + d]);
        for (int d = 0; d < c.D; d++) Assert.Equal(0f, fused.DQ[3 * c.D + d]);
        Assert.All(fused.DK.Concat(fused.DV), x => Assert.False(float.IsNaN(x)));
    }

    // The decomposed path (other element types, GPU engines) gives the same zeros for a row with no allowed key.
    [Fact]
    public void Decomposed_RowWithNoAllowedKey_IsZero_NotNaN()
    {
        var e = new CpuEngine();
        var q = Random<double>(1, 1, 1, 5, 4); var k = Random<double>(2, 1, 1, 6, 4); var v = Random<double>(3, 1, 1, 6, 4);
        var mask = new Tensor<bool>(new[] { 5, 6 });
        for (int i = 0; i < mask.Length; i++) mask[i] = i / 6 != 2;   // query row 2 sees nothing
        var o = e.ScaledDotProductAttention(q, k, v, mask, null, isCausal: false);
        for (int d = 0; d < 4; d++) Assert.Equal(0.0, o[0, 0, 2, d]);
        for (int i = 0; i < o.Length; i++) Assert.False(double.IsNaN(o[i]));
    }

    [Fact]
    public void Fused_IsBitIdenticalAcrossThreadCounts()
    {
        var c = Get("everything");
        var mask = MaskFor(c);
        (float[], float[], float[], float[]) At(int dop)
        {
            int previous = CpuParallelSettings.MaxDegreeOfParallelism;
            CpuParallelSettings.MaxDegreeOfParallelism = dop;
            try { return Run(new CpuEngine(), c, fused: true, mask); }
            finally { CpuParallelSettings.MaxDegreeOfParallelism = previous; }
        }
        var a = At(1);
        var b = At(16);
        Assert.Equal(a.Item1, b.Item1);
        Assert.Equal(a.Item2, b.Item2);
        Assert.Equal(a.Item3, b.Item3);
        Assert.Equal(a.Item4, b.Item4);
    }

    [Fact]
    public void Fused_DoubleFallback_MatchesMaterialisedAttention()
    {
        var e = new CpuEngine();
        var q = Random<double>(1, 1, 2, 9, 4); var k = Random<double>(2, 1, 2, 11, 4); var v = Random<double>(3, 1, 2, 11, 4);
        var mask = RandomMask(95, 9, 11);
        var fused = e.ScaledDotProductAttention(q, k, v, mask, null, isCausal: false, softcap: 2.0);
        var reference = e.ScaledDotProductAttention(q, k, v, Expand(mask, 1, 2, 9, 11, false), null, out _, 2.0);
        for (int i = 0; i < fused.Length; i++) Assert.True(Math.Abs(fused[i] - reference[i]) < 1e-12);
    }

#if NET5_0_OR_GREATER
    // Training memory: the materialising attention keeps (and allocates) [B, H, S, S] weights for its backward; the
    // fused path saves one log-sum-exp per row, so a forward + backward allocates a small fraction of that.
    [Fact]
    public void Fused_TrainingStep_DoesNotAllocateTheAttentionMatrix()
    {
        const int B = 1, H = 4, S = 2048, D = 64;
        var c = new Case("memory", B, H, S, S, D, D, true, "", 0, null, false);
        long weightsBytes = 4L * B * H * S * S;   // 64 MB of float scores (inputs, outputs and grads are ~2 MB each)
        var e = new CpuEngine();
        Run(e, c, fused: true, null);              // warm up (scratch, JIT)
        Run(e, c, fused: false, null);
        long before = GC.GetTotalAllocatedBytes(precise: true);
        Run(e, c, fused: true, null);
        long fusedBytes = GC.GetTotalAllocatedBytes(precise: true) - before;
        before = GC.GetTotalAllocatedBytes(precise: true);
        Run(e, c, fused: false, null);
        long materialisedBytes = GC.GetTotalAllocatedBytes(precise: true) - before;
        Assert.True(materialisedBytes >= weightsBytes, $"reference allocated {materialisedBytes} bytes; expected at least the weights");
        Assert.True(fusedBytes < weightsBytes / 2,
            $"fused step allocated {fusedBytes} bytes (materialised {materialisedBytes}); the [B,H,S,S] weights alone are {weightsBytes}");
    }
#endif
}
