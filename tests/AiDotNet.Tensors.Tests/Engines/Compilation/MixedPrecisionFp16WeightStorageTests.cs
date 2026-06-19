using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Phase C completion (Tensors #558): fp16 WEIGHT STORAGE with an fp32 master. The earlier compiled
/// step kept weights in fp32 (only the activations were fp16); this exercises the other half of the
/// memory win — the weight itself lives in <see cref="Tensor{Half}"/> (2 bytes/elem) while an fp32
/// master (<see cref="MasterWeights"/>) accumulates the tiny SGD updates, then casts back to the fp16
/// storage. Verifies the new <see cref="MixedPrecisionCompiledPlan.Step(System.Collections.Generic.IReadOnlyList{Tensor{float}},System.Collections.Generic.IReadOnlyList{Tensor{Half}},MasterWeights,float,GradScaler)"/>
/// drives a real (loss-scaled) training step to convergence — the master-update path that updating fp16
/// directly cannot achieve (sub-quantum updates would round away).
/// </summary>
[Collection(MixedPrecisionTestCollection.Name)]
public class MixedPrecisionFp16WeightStorageTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> Vf(params float[] v) => new(v, new[] { v.Length });
    private static Tensor<Half> Vh(int n, Func<int, float> init)
    {
        var h = new Half[n];
        for (int i = 0; i < n; i++) h[i] = (Half)init(i);
        return new Tensor<Half>(h, new[] { n });
    }

    [Fact]
    public void Fp16WeightStorage_Fp32Master_DescendsLoss_AndConverges()
    {
        const int n = 6;
        // Target: elementwise y = x .* wTrue. Learn the fp16-stored W so x.*W approaches the target.
        var x = Vf(1f, 2f, 0.5f, -1f, 1.5f, -0.5f);
        var wTrue = new float[] { 2f, 0.5f, -1f, 1f, 0.25f, -2f };
        var ta = new float[n];
        var xa = x.ToArray();
        for (int i = 0; i < n; i++) ta[i] = xa[i] * wTrue[i];

        // FP16 STORAGE weight (leaf), initialized away from the solution.
        var W = Vh(n, _ => 0.1f);

        // Build the mixed-dtype lazy graph: d = cast16(x); h = d .* W (fp16); y = cast32(h);
        // loss = sum_i (y_i - t_i)^2.
        var d = new Tensor<Half>(new[] { n });
        var nd = new CrossTypeLazyNode<float, Half>(LazyNodeType.Custom, "cast16", x, d,
            (e, o) => MixedPrecisionCast.CastToFp16(x).AsSpan().CopyTo(o.AsWritableSpan()),
            (g, inp, o, s, e) => MixedPrecisionCast.CastToFp16Backward(g));
        d.LazySource = nd;

        var h = new Tensor<Half>(new[] { n });
        var nh = new LazyNode<Half>(LazyNodeType.Multiply, "mul16", d, W, h,
            (e, o) => e.TensorMultiply(d, W).AsSpan().CopyTo(o.AsWritableSpan()),
            (gradOut, inputs, output, state, eng, grads) =>
            {
                grads[inputs[0]] = grads.TryGetValue(inputs[0], out var e0)
                    ? eng.TensorAdd(e0, eng.TensorMultiply(gradOut, inputs[1])) : eng.TensorMultiply(gradOut, inputs[1]);
                grads[inputs[1]] = grads.TryGetValue(inputs[1], out var e1)
                    ? eng.TensorAdd(e1, eng.TensorMultiply(gradOut, inputs[0])) : eng.TensorMultiply(gradOut, inputs[0]);
            });
        h.LazySource = nh;

        var y = new Tensor<float>(new[] { n });
        var ny = new CrossTypeLazyNode<Half, float>(LazyNodeType.Custom, "cast32", h, y,
            (e, o) => MixedPrecisionCast.CastToFp32(h).AsSpan().CopyTo(o.AsWritableSpan()),
            (g, inp, o, s, e) => MixedPrecisionCast.CastToFp32Backward(g));
        y.LazySource = ny;

        var loss = new Tensor<float>(new[] { 1 });
        var nloss = new LazyNode<float>(LazyNodeType.Sum, "sqerr", y, loss,
            (e, o) => { var ya = y.ToArray(); float sm = 0; for (int i = 0; i < n; i++) { float dd = ya[i] - ta[i]; sm += dd * dd; } o.AsWritableSpan()[0] = sm; },
            (gradOut, inputs, output, state, eng, grads) =>
            {
                var yt = inputs[0]; var ya = yt.ToArray(); float go = gradOut.ToArray()[0];
                var g = new float[n];
                for (int i = 0; i < n; i++) g[i] = 2f * (ya[i] - ta[i]) * go;
                grads[yt] = grads.TryGetValue(yt, out var ex) ? eng.TensorAdd(ex, new Tensor<float>(g, yt._shape)) : new Tensor<float>(g, yt._shape);
            });
        loss.LazySource = nloss;

        var plan = MixedPrecisionCompiledPlan.Compile(loss, _engine);

        // Register the fp32 master for the fp16 storage weight.
        var masters = new MasterWeights();
        var wInit = new float[n];
        var wh = W.ToArray();
        for (int i = 0; i < n; i++) wInit[i] = (float)wh[i];
        masters.Register(W, wInit);

        var scaler = new GradScaler(new MixedPrecisionConfig { LossScale = 256f, DynamicLossScale = false });
        var noFp32 = Array.Empty<Tensor<float>>();
        var fp16Params = new[] { W };

        float first = 0, last = 0;
        int overflows = 0;
        const int steps = 200;
        for (int s = 0; s < steps; s++)
        {
            var r = plan.Step(noFp32, fp16Params, masters, learningRate: 0.05f, scaler: scaler);
            if (s == 0) first = r.Loss;
            last = r.Loss;
            if (r.FoundInfNan) overflows++;
            Assert.False(float.IsNaN(r.Loss) || float.IsInfinity(r.Loss), $"loss non-finite at step {s}");
        }

        Assert.Equal(0, overflows);
        Assert.True(first > 0, "first loss positive");
        Assert.True(last < 0.05f * first, $"fp16-weight-storage training did not converge: first {first}, last {last}");

        // The learned fp16 weights should approximate wTrue (fp16 precision tolerance).
        var wLearned = W.ToArray();
        for (int i = 0; i < n; i++)
            Assert.True(Math.Abs((float)wLearned[i] - wTrue[i]) < 0.1f,
                $"W[{i}] = {(float)wLearned[i]} did not approach {wTrue[i]}");
    }
}
