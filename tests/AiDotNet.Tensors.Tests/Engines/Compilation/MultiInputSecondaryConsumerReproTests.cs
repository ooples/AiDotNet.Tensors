using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Repro for the DiT multi-input divergence (consumer ooples/AiDotNet#1620): the secondary
/// input (timestep embedding) feeds a NON-trivial consumer — a matmul (adaLN modulation) —
/// before it is combined, whereas the existing #616 coverage adds the secondary input
/// directly. If the multi-input replay re-binds correctly the compiled output must match
/// eager for every (x, t) pair.
/// </summary>
public class MultiInputSecondaryConsumerReproTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public MultiInputSecondaryConsumerReproTests() { AiDotNetEngine.Current = new CpuEngine(); }
    public void Dispose() { AiDotNetEngine.Current = _prior; }

    [Fact]
    public void MultiInput_SecondaryInputThroughMatmul_MatchesEager()
    {
        var engine = new CpuEngine();
        var x0 = Tensor<float>.CreateRandom(new[] { 2, 4 });
        var wx = Tensor<float>.CreateRandom(new[] { 4, 5 });
        var t0 = Tensor<float>.CreateRandom(new[] { 2, 3 });
        var wt = Tensor<float>.CreateRandom(new[] { 3, 5 });

        float[] Eager(Tensor<float> x, Tensor<float> t)
        {
            var a = engine.TensorMatMul(x, wx);   // primary through matmul
            var b = engine.TensorMatMul(t, wt);   // SECONDARY through matmul (adaLN-like)
            var o = engine.TensorAdd(a, b);
            var arr = new float[o.Length];
            o.AsSpan().CopyTo(arr);
            return arr;
        }

        var cache = new CompiledModelCache<float>();
        try
        {
            var plan = cache.GetOrCompileInference(new[] { x0, t0 }, () =>
            {
                var a = engine.TensorMatMul(x0, wx);
                var b = engine.TensorMatMul(t0, wt);
                return engine.TensorAdd(a, b);
            });

            AssertReplayMatches(plan, () => Tensor<float>.CreateRandom(new[] { 2, 4 }),
                () => Tensor<float>.CreateRandom(new[] { 2, 3 }), Eager, "secondary-through-matmul");
        }
        finally { cache.Dispose(); }
    }

    // Secondary input read by MULTIPLE consumers (adaLN reads timeEmbed in every block).
    [Fact]
    public void MultiInput_SecondaryInputMultipleConsumers_MatchesEager()
    {
        var engine = new CpuEngine();
        var wx = Tensor<float>.CreateRandom(new[] { 4, 5 });
        var wt1 = Tensor<float>.CreateRandom(new[] { 3, 5 });
        var wt2 = Tensor<float>.CreateRandom(new[] { 3, 5 });
        var x0 = Tensor<float>.CreateRandom(new[] { 2, 4 });
        var t0 = Tensor<float>.CreateRandom(new[] { 2, 3 });

        float[] Eager(Tensor<float> x, Tensor<float> t)
        {
            var a = engine.TensorAdd(engine.TensorMatMul(x, wx),
                       engine.TensorAdd(engine.TensorMatMul(t, wt1), engine.TensorMatMul(t, wt2)));
            var arr = new float[a.Length]; a.AsSpan().CopyTo(arr); return arr;
        }

        var cache = new CompiledModelCache<float>();
        try
        {
            var plan = cache.GetOrCompileInference(new[] { x0, t0 }, () =>
                engine.TensorAdd(engine.TensorMatMul(x0, wx),
                    engine.TensorAdd(engine.TensorMatMul(t0, wt1), engine.TensorMatMul(t0, wt2))));
            AssertReplayMatches(plan, () => Tensor<float>.CreateRandom(new[] { 2, 4 }),
                () => Tensor<float>.CreateRandom(new[] { 2, 3 }), Eager, "multi-consumer");
        }
        finally { cache.Dispose(); }
    }

    // adaLN modulation: secondary -> matmul -> broadcast-multiply with the primary branch.
    [Fact]
    public void MultiInput_SecondaryInputBroadcastModulation_MatchesEager()
    {
        var engine = new CpuEngine();
        var wx = Tensor<float>.CreateRandom(new[] { 4, 5 });
        var wt = Tensor<float>.CreateRandom(new[] { 3, 5 });
        var x0 = Tensor<float>.CreateRandom(new[] { 2, 4 });
        var t0 = Tensor<float>.CreateRandom(new[] { 1, 3 });   // broadcast over batch

        float[] Eager(Tensor<float> x, Tensor<float> t)
        {
            var xb = engine.TensorMatMul(x, wx);                  // [2,5]
            var scale = engine.TensorMatMul(t, wt);              // [1,5]
            var o = engine.TensorBroadcastMultiply(xb, scale);   // adaLN scale
            var arr = new float[o.Length]; o.AsSpan().CopyTo(arr); return arr;
        }

        var cache = new CompiledModelCache<float>();
        try
        {
            var plan = cache.GetOrCompileInference(new[] { x0, t0 }, () =>
                engine.TensorBroadcastMultiply(engine.TensorMatMul(x0, wx), engine.TensorMatMul(t0, wt)));
            AssertReplayMatches(plan, () => Tensor<float>.CreateRandom(new[] { 2, 4 }),
                () => Tensor<float>.CreateRandom(new[] { 1, 3 }), Eager, "broadcast-modulation");
        }
        finally { cache.Dispose(); }
    }

    // Deep graph (20+ steps) so the post-specialization CPU optimization passes (CSE, fusion)
    // actually run — they are skipped below ~20 steps, which is why the shallow repros above pass
    // but the foundation-scale DiT (many blocks, all reading timeEmbed) diverges.
    [Fact]
    public void MultiInput_DeepGraph_SecondaryInEveryBlock_MatchesEager()
    {
        var engine = new CpuEngine();
        const int blocks = 12;
        var wx = new Tensor<float>[blocks];
        var wt = new Tensor<float>[blocks];
        // Deterministic small weights keep the 12-block chain's magnitude O(1) — see Det.
        for (int i = 0; i < blocks; i++)
        {
            wx[i] = Det(new[] { 5, 5 }, 100 + i);
            wt[i] = Det(new[] { 3, 5 }, 200 + i);
        }
        var x0 = Det(new[] { 2, 5 }, 1);
        var t0 = Det(new[] { 2, 3 }, 2);

        Tensor<float> Build(Tensor<float> x, Tensor<float> t)
        {
            var h = x;
            for (int i = 0; i < blocks; i++)
            {
                var hx = engine.TensorMatMul(h, wx[i]);          // block transform
                var m = engine.TensorMatMul(t, wt[i]);           // adaLN-like, reads t every block
                h = engine.TensorAdd(hx, m);                      // residual-ish combine
            }
            return h;
        }
        float[] Eager(Tensor<float> x, Tensor<float> t)
        {
            var o = Build(x, t);
            var arr = new float[o.Length]; o.AsSpan().CopyTo(arr); return arr;
        }

        var cache = new CompiledModelCache<float>();
        try
        {
            var plan = cache.GetOrCompileInference(new[] { x0, t0 }, () => Build(x0, t0));
            int trial = 0;
            AssertReplayMatches(plan, () => Det(new[] { 2, 5 }, 300 + trial),
                () => Det(new[] { 2, 3 }, 400 + trial++), Eager, "deep-graph");
        }
        finally { cache.Dispose(); }
    }

    // Deep graph where each block has CONSECUTIVE matmuls sharing the secondary input (the
    // attention q/k/v / adaLN pattern). These ARE batched by the consecutive-run rule, so this
    // verifies consecutive batching itself does not corrupt multi-input replay.
    [Fact]
    public void MultiInput_DeepGraph_ConsecutiveSecondaryMatMuls_MatchesEager()
    {
        var engine = new CpuEngine();
        const int blocks = 10;
        var wa = new Tensor<float>[blocks];
        var wb = new Tensor<float>[blocks];
        var wx = new Tensor<float>[blocks];
        // Deterministic small weights keep the 10-block chain's magnitude O(1) — see Det.
        for (int i = 0; i < blocks; i++)
        {
            wa[i] = Det(new[] { 3, 5 }, 100 + i);
            wb[i] = Det(new[] { 3, 5 }, 200 + i);
            wx[i] = Det(new[] { 5, 5 }, 300 + i);
        }
        var x0 = Det(new[] { 2, 5 }, 1);
        var t0 = Det(new[] { 2, 3 }, 2);

        Tensor<float> Build(Tensor<float> x, Tensor<float> t)
        {
            var h = x;
            for (int i = 0; i < blocks; i++)
            {
                var a = engine.TensorMatMul(t, wa[i]);   // consecutive secondary matmuls
                var b = engine.TensorMatMul(t, wb[i]);   // (a, b adjacent -> batched together)
                var hx = engine.TensorMatMul(h, wx[i]);
                h = engine.TensorAdd(hx, engine.TensorAdd(a, b));
            }
            return h;
        }
        float[] Eager(Tensor<float> x, Tensor<float> t)
        {
            var o = Build(x, t);
            var arr = new float[o.Length]; o.AsSpan().CopyTo(arr); return arr;
        }

        var cache = new CompiledModelCache<float>();
        try
        {
            var plan = cache.GetOrCompileInference(new[] { x0, t0 }, () => Build(x0, t0));
            int trial = 0;
            AssertReplayMatches(plan, () => Det(new[] { 2, 5 }, 500 + trial),
                () => Det(new[] { 2, 3 }, 600 + trial++), Eager, "consecutive-secondary");
        }
        finally { cache.Dispose(); }
    }

    // Deterministic, bounded-magnitude tensor for the DEEP tests. A 10–12 block
    // unnormalized matmul chain compounds its output magnitude exponentially with
    // random weights (|out| ~ 1e3–1e4), so the compiled path's legitimate ~1-ULP
    // *relative* GEMM rounding (which differs from eager only on net471, where the
    // SIMD summation order differs — net10 is bit-exact) becomes a large *absolute*
    // diff and the absolute 1e-3 replay tolerance flakes. Small fixed weights keep
    // |out| ~ O(1) so the same correct path stays well under 1e-3 on both TFMs —
    // this fixes the non-determinism without touching the tolerance.
    private static Tensor<float> Det(int[] shape, int seed)
    {
        int n = 1; foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(((i * 7 + seed * 13) % 17) - 8) * 0.1f;
        return new Tensor<float>(data, shape);
    }

    private static void AssertReplayMatches(
        ICompiledPlan<float> plan,
        Func<Tensor<float>> makeX, Func<Tensor<float>> makeT,
        Func<Tensor<float>, Tensor<float>, float[]> eager, string label)
    {
        for (int trial = 0; trial < 4; trial++)
        {
            var x = makeX();
            var t = makeT();
            plan.SetInputs(new[] { x, t });
            var got = plan.Execute();
            var exp = eager(x, t);
            float maxDiff = 0f;
            for (int i = 0; i < got.Length; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(exp[i] - got[i]));
            Assert.True(maxDiff < 1e-3f, $"[{label}] trial {trial}: compiled diverged from eager, maxDiff={maxDiff:E3}");
        }
    }
}
