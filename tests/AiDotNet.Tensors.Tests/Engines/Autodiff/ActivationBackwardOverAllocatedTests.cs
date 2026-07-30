// Copyright (c) AiDotNet. All rights reserved.
// Regression: the SIMD Tanh/Sigmoid backward kernels iterated to grad.Length (the PHYSICAL
// backing-array length) while writing into a result buffer sized to the tensor's LOGICAL length.
// Under a GradientTape the activation's forward output is POOL-allocated, and the pool hands back a
// backing array rounded UP to a bucket size — physically longer than the logical Length, yet with
// _storage.Length == Length so GetFlattenedData returns that oversized array. The unchecked AVX
// store in the backward kernel then wrote PAST the logical-sized result buffer -> AccessViolation.
//
// Discovered via the HRE standard-transformer block (attention intermediates cycle the pool, then
// the wide 4E Tanh FFN backward crashes). This test reproduces that shape/pressure through the tape
// so it faults deterministically if the bound regresses; the fix bounds the kernels by the logical
// length. Verified to CRASH with the bound reverted and PASS with it in place.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class ActivationBackwardOverAllocatedTests
{
    // One attention-then-FFN block, matching the HRE standard-transformer arm that surfaced the bug:
    // 3D permute / batched-matmul / softmax intermediates cycle the tape's buffer pool, then a wide
    // [B*S, 4E] activation backward runs against a pool-over-allocated forward output.
    private static void RunBlock(IEngine e, System.Func<Tensor<float>, Tensor<float>> act,
        Tensor<float> h, Tensor<float> wq, Tensor<float> wk, Tensor<float> wv, Tensor<float> wo,
        Tensor<float> w1, Tensor<float> w2, int B, int S, int E)
    {
        var h2 = e.Reshape<float>(h, new[] { B * S, E });
        Tensor<float> Proj(Tensor<float> w) => e.Reshape<float>(e.TensorMatMul<float>(h2, w), new[] { B, S, E });
        var q = Proj(wq); var k = Proj(wk); var v = Proj(wv);
        var kT = e.TensorPermute<float>(k, new[] { 0, 2, 1 });
        var scores = e.TensorMultiplyScalar<float>(e.BatchMatMul<float>(q, kT), 1f / (float)Math.Sqrt(E));
        var ctx = e.BatchMatMul<float>(e.Softmax<float>(scores, 2), v);
        var att = e.Reshape<float>(e.TensorMatMul<float>(e.Reshape<float>(ctx, new[] { B * S, E }), wo), new[] { B, S, E });
        var a = e.Reshape<float>(e.TensorAdd<float>(h, att), new[] { B * S, E });
        // wide FFN: [B*S,E]@[E,4E] -> activation -> @[4E,E]  (the activation whose backward crashed)
        var hid = act(e.TensorMatMul<float>(a, w1));
        var f = e.TensorMatMul<float>(hid, w2);
        var outT = e.TensorAdd<float>(a, f);
        var loss = e.ReduceSum<float>(e.TensorMultiply<float>(outT, outT), null, false);
        // backward over all params — runs the activation backward against the pooled forward output.
        // Do NOT dispose the tape here; the caller's `using` owns its lifetime.
        var tape = GradientTape<float>.Current;
        _ = tape!.ComputeGradients(loss, new[] { h, wq, wk, wv, wo, w1, w2 });
    }

    private static Tensor<float> Rand(Random r, int[] shape)
    {
        int n = 1; foreach (var d in shape) n *= d;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1);
        return new Tensor<float>(a, shape);
    }

    [Theory]
    [InlineData("tanh")]
    [InlineData("sigmoid")]
    public void ActivationBackward_WideFfnUnderPoolPressure_NoOob(string kind)
    {
        var e = new CpuEngine();
        int B = 256, S = 6, E = 64, H = 4 * E;
        var r = new Random(3);
        System.Func<Tensor<float>, Tensor<float>> act = kind == "tanh"
            ? (t => e.Tanh<float>(t))
            : (t => e.Sigmoid<float>(t));

        // Several fresh-tape steps so the buffer pool warms up and starts handing back
        // rounded-up (over-allocated) buckets — the condition the crash needed.
        for (int step = 0; step < 6; step++)
        {
            using (new GradientTape<float>())
            {
                var h = Rand(r, new[] { B, S, E });
                var wq = Rand(r, new[] { E, E }); var wk = Rand(r, new[] { E, E });
                var wv = Rand(r, new[] { E, E }); var wo = Rand(r, new[] { E, E });
                var w1 = Rand(r, new[] { E, H }); var w2 = Rand(r, new[] { H, E });
                RunBlock(e, act, h, wq, wk, wv, wo, w1, w2, B, S, E);
            }
        }
        // Reaching here without an AccessViolation is the assertion (pre-fix: hard crash in the
        // activation backward). Sanity-check the engine is still usable afterward.
        var probe = e.Tanh<float>(new Tensor<float>(new float[] { 0f }, new[] { 1, 1 })).ToArray();
        Assert.Equal(0f, probe[0], 3);
    }
}
