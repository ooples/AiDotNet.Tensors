using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// FP16 activation-PAGING gate (Tensors #558) — the piece that actually realizes the resident-memory win
/// on transformers. With paging on, MixedPrecisionCompiledPlan downcasts each FP32-op activation to Half
/// and frees its float backing after its last forward use, then upcasts on demand in backward (refcounted
/// free). This test runs a chain with an FP32 op between matmuls (so there ARE float activations to page),
/// and asserts (1) paging engages (PageOutCount &gt; 0), and (2) it is TRANSPARENT — gradients match the
/// non-paged run within FP16 rounding (paging stores activations as Half, the accepted AMP trade).
/// </summary>
[Collection(MixedPrecisionTestCollection.Name)]
public class MixedPrecisionPagingTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> Rand(int r, int c, int seed)
    {
        var rng = new Random(seed);
        var d = new float[r * c];
        for (int i = 0; i < d.Length; i++) d[i] = (float)((rng.NextDouble() * 2 - 1) * 0.3);
        return new Tensor<float>(d, new[] { r, c });
    }

    [Fact]
    public void Paging_IsTransparent_AndEngages()
    {
        const int B = 4, d = 5;
        var x = Rand(B, d, 1);
        var W1 = Rand(d, d, 2);
        var c1 = Rand(B, d, 3);   // FP32 elementwise factor between the matmuls
        var W2 = Rand(d, d, 4);
        var t = Rand(B, d, 5);

        // y1 = x·W1 (FP16 act); a1 = y1 ⊙ c1 (FP32 op — its backward reads y1, a pageable float activation);
        // y2 = a1·W2 (FP16 act); loss = sum((y2 - t)^2).
        Func<Tensor<float>> forward = () =>
        {
            var y1 = _engine.TensorMatMul(x, W1);
            var a1 = _engine.TensorMultiply(y1, c1);
            var y2 = _engine.TensorMatMul(a1, W2);
            var diff = _engine.TensorSubtract(y2, t);
            var sq = _engine.TensorMultiply(diff, diff);
            return _engine.ReduceSum(sq);
        };

        // Reference: paging OFF.
        float[] gxRef, gw1Ref, gw2Ref;
        MixedPrecisionCompiledPlan.PagingTestOverride = false;
        {
            var plan = MixedPrecisionCompiledPlan.Trace(forward, _engine);
            plan.Forward();
            var g = plan.Backward();
            Assert.Equal(0, plan.PageOutCount);
            gxRef = g.Fp32[x].ToArray(); gw1Ref = g.Fp32[W1].ToArray(); gw2Ref = g.Fp32[W2].ToArray();
        }

        // Paging ON: same graph, gradients must match within FP16 rounding, and paging must engage.
        float[] gx, gw1, gw2; int pageOuts;
        MixedPrecisionCompiledPlan.PagingTestOverride = true;
        try
        {
            var plan = MixedPrecisionCompiledPlan.Trace(forward, _engine);
            plan.Forward();
            var g = plan.Backward();
            pageOuts = plan.PageOutCount;
            gx = g.Fp32[x].ToArray(); gw1 = g.Fp32[W1].ToArray(); gw2 = g.Fp32[W2].ToArray();
        }
        finally { MixedPrecisionCompiledPlan.PagingTestOverride = null; }

        Assert.True(pageOuts > 0, "paging should have paged at least one activation");
        AssertClose(gxRef, gx, "dL/dx");
        AssertClose(gw1Ref, gw1, "dL/dW1");
        AssertClose(gw2Ref, gw2, "dL/dW2");
    }

    private static void AssertClose(float[] expected, float[] got, string what)
    {
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) <= 2e-3f + 1e-2f * Math.Abs(expected[i]),
                $"{what}[{i}]: paged {got[i]} vs ref {expected[i]} (FP16 activation rounding)");
    }
}
