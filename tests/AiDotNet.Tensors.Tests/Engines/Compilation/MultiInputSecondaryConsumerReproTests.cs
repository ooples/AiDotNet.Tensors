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

            for (int trial = 0; trial < 4; trial++)
            {
                var x = Tensor<float>.CreateRandom(new[] { 2, 4 });
                var t = Tensor<float>.CreateRandom(new[] { 2, 3 });
                plan.SetInputs(new[] { x, t });
                var got = plan.Execute();
                var exp = Eager(x, t);
                float maxDiff = 0f;
                for (int i = 0; i < got.Length; i++)
                    maxDiff = Math.Max(maxDiff, Math.Abs(exp[i] - got[i]));
                Assert.True(maxDiff < 1e-3f, $"trial {trial}: compiled diverged from eager, maxDiff={maxDiff:E3}");
            }
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
            AssertReplayMatches(engine, plan, () => Tensor<float>.CreateRandom(new[] { 2, 4 }),
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
            AssertReplayMatches(engine, plan, () => Tensor<float>.CreateRandom(new[] { 2, 4 }),
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
        for (int i = 0; i < blocks; i++)
        {
            wx[i] = Tensor<float>.CreateRandom(new[] { 5, 5 });
            wt[i] = Tensor<float>.CreateRandom(new[] { 3, 5 });
        }
        var x0 = Tensor<float>.CreateRandom(new[] { 2, 5 });
        var t0 = Tensor<float>.CreateRandom(new[] { 2, 3 });

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
            AssertReplayMatches(engine, plan, () => Tensor<float>.CreateRandom(new[] { 2, 5 }),
                () => Tensor<float>.CreateRandom(new[] { 2, 3 }), Eager, "deep-graph");
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
        for (int i = 0; i < blocks; i++)
        {
            wa[i] = Tensor<float>.CreateRandom(new[] { 3, 5 });
            wb[i] = Tensor<float>.CreateRandom(new[] { 3, 5 });
            wx[i] = Tensor<float>.CreateRandom(new[] { 5, 5 });
        }
        var x0 = Tensor<float>.CreateRandom(new[] { 2, 5 });
        var t0 = Tensor<float>.CreateRandom(new[] { 2, 3 });

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
            AssertReplayMatches(engine, plan, () => Tensor<float>.CreateRandom(new[] { 2, 5 }),
                () => Tensor<float>.CreateRandom(new[] { 2, 3 }), Eager, "consecutive-secondary");
        }
        finally { cache.Dispose(); }
    }

    private static void AssertReplayMatches(
        CpuEngine engine, ICompiledPlan<float> plan,
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
