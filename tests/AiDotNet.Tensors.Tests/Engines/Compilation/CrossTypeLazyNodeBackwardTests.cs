using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Phase-2 keystone gate (Tensors #555, docs/fp16-activation-storage-design.md): the cross-type lazy
/// node now carries a backward, the lazy-graph analog of the verified eager-tape cast-with-backward.
/// This validates the node-level forward+backward plumbing in isolation (Realize → Backward), before it
/// is threaded through the compiled backward walk's secondary-dtype grad space.
///
/// Test 1 pins the mixed-precision use case (FP32→FP16 cast): forward realizes to FP16, backward bridges
/// the gradient back to FP32 — exact on FP16-representable values, round-trip consistent (the same gate
/// MixedPrecisionCast uses, since a cast is a non-smooth rounding staircase). Test 2 proves the
/// mechanism is NOT special-cased to an identity Jacobian: a non-identity cross-type op (y = 3·x into
/// FP16) is finite-differenced with FP16-exact steps, and the node's backward Jacobian matches the
/// forward's numerical Jacobian exactly.
/// </summary>
public class CrossTypeLazyNodeBackwardTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> Vec(params float[] v) => new(v, new[] { v.Length });

    [Fact]
    public void CastNode_Forward_Realizes_And_Backward_Bridges_FP16_To_FP32()
    {
        var x = Vec(1f, 2f, 0.5f, -1f, 8f, -0.25f);
        var outp = new Tensor<Half>(new int[] { x.Length });

        var node = new CrossTypeLazyNode<float, Half>(
            LazyNodeType.Custom, "CastToFp16",
            input: x, output: outp,
            execute: (eng, o) => MixedPrecisionCast.CastToFp16(x).AsSpan().CopyTo(o.AsWritableSpan()),
            backwardFn: (gradOut, inp, o, state, eng) => MixedPrecisionCast.CastToFp16Backward(gradOut));

        node.Realize(_engine);
        Assert.True(node.IsRealized);
        // Forward: output is the FP16 cast of x (exact for these representable values).
        var oa = outp.ToArray(); var xa = x.ToArray();
        for (int i = 0; i < xa.Length; i++) Assert.Equal(xa[i], (float)oa[i]);

        // Backward: an FP16 gradient bridges back to FP32 exactly on representable values.
        var gradOut = new Tensor<Half>(new[] { (Half)0.5f, (Half)(-2f), (Half)1f, (Half)4f, (Half)(-1f), (Half)0.25f }, new[] { x.Length });
        var gradIn = node.Backward(gradOut, _engine);
        Assert.NotNull(gradIn);
        Assert.Equal(x.Shape.ToArray(), gradIn!.Shape.ToArray());
        var gi = gradIn.ToArray(); var go = gradOut.ToArray();
        for (int i = 0; i < gi.Length; i++) Assert.Equal((float)go[i], gi[i]);
    }

    [Fact]
    public void NonIdentityCrossType_Backward_Matches_FiniteDifference_Jacobian()
    {
        const float K = 3f;
        var x = Vec(1f, 2f, 0.5f, -1f);

        CrossTypeLazyNode<float, Half> Build(Tensor<float> input) => new(
            LazyNodeType.Custom, "ScaleToFp16",
            input: input, output: new Tensor<Half>(new int[] { input.Length }),
            execute: (eng, o) =>
            {
                var xs = input.AsSpan(); var os = o.AsWritableSpan();
                for (int i = 0; i < xs.Length; i++) os[i] = (Half)(K * xs[i]);
            },
            // y = K·x  ⇒  dy/dx = K; backward maps FP16 grad -> FP32 grad scaled by K.
            backwardFn: (gradOut, inp, o, state, eng) =>
            {
                var g = gradOut.AsSpan(); var r = new float[g.Length];
                for (int i = 0; i < g.Length; i++) r[i] = K * (float)g[i];
                return new Tensor<float>(r, inp.Shape.ToArray());
            });

        // Analytic backward at gradOut = ones.
        var ones = new Tensor<Half>(new[] { (Half)1f, (Half)1f, (Half)1f, (Half)1f }, new[] { 4 });
        var analytic = Build(x).Also(n => n.Realize(_engine)).Backward(ones, _engine)!.ToArray();

        // Central finite difference of the realized forward with an FP16-exact step (h = 0.25, K·(x±h)
        // stays FP16-exact ⇒ no staircase noise). Slope must equal K, matching the analytic backward.
        const float h = 0.25f;
        var xa = x.ToArray();
        for (int i = 0; i < xa.Length; i++)
        {
            var xp = (float[])xa.Clone(); xp[i] += h;
            var xm = (float[])xa.Clone(); xm[i] -= h;
            var yp = Build(Vec(xp)).Also(n => n.Realize(_engine)).Output.ToArray();
            var ym = Build(Vec(xm)).Also(n => n.Realize(_engine)).Output.ToArray();
            float slope = ((float)yp[i] - (float)ym[i]) / (2f * h);
            Assert.Equal(K, slope, 3);          // forward numerical Jacobian == K
            Assert.Equal(slope, analytic[i], 3); // backward Jacobian == forward Jacobian
        }
    }
}

internal static class TestFluent
{
    /// <summary>Tiny side-effect helper so a node can be Realized inline in an expression.</summary>
    public static T Also<T>(this T self, Action<T> act) { act(self); return self; }
}
