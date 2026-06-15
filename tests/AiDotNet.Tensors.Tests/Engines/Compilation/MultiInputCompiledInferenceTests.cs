using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Coverage for the multi-input compiled-inference path
/// (<see cref="CompiledModelCache{T}.GetOrCompileInference(Tensor{T}[], Func{Tensor{T}})"/>).
/// Before this path existed, <c>Compile</c> marked exactly ONE traced leaf as the mutable
/// input slot and baked every other leaf as a constant. A forward with more than one
/// per-call-varying input — the canonical case being a diffusion denoiser that reads both
/// the noisy sample AND a per-step timestep embedding — therefore replayed the trace-time
/// value of all-but-the-first input, silently corrupting multi-step inference. These tests
/// assert that EVERY declared input is re-bound on replay, and that a tensor the graph never
/// reads is rejected (fail-closed) rather than silently baked.
/// </summary>
public class MultiInputCompiledInferenceTests : IDisposable
{
    // GetOrCompileInference captures AiDotNetEngine.Current as the plan's execution engine;
    // on GPU machines the module initializer flips Current to DirectGpu. Pin CPU so plan
    // replay runs on the same engine that authored the recorded ops.
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    public MultiInputCompiledInferenceTests() { AiDotNetEngine.Current = new CpuEngine(); }
    public void Dispose() { AiDotNetEngine.Current = _priorEngine; }

    // Eager reference: out = (x @ w) + t. `x` and `t` both vary per call; `w` is a constant.
    private static float[] EagerForward(CpuEngine engine, Tensor<float> x, Tensor<float> w, Tensor<float> t)
    {
        var mm = engine.TensorMatMul(x, w);
        var o = engine.TensorBroadcastAdd(mm, t);
        var arr = new float[o.Length];
        o.AsSpan().CopyTo(arr);
        return arr;
    }

    [Fact]
    public void MultiInput_RebindsEverySlot_SecondaryInputIsNotBaked()
    {
        var engine = new CpuEngine();
        var x0 = Tensor<float>.CreateRandom(new[] { 2, 3 });
        var w = Tensor<float>.CreateRandom(new[] { 3, 2 });
        var t0 = Tensor<float>.CreateRandom(new[] { 2, 2 });

        var cache = new CompiledModelCache<float>();
        try
        {
            // Compile declaring BOTH x and t as mutable inputs.
            var plan = cache.GetOrCompileInference(new[] { x0, t0 }, () =>
            {
                var mm = engine.TensorMatMul(x0, w);
                return engine.TensorBroadcastAdd(mm, t0);
            });

            // Every (x, t) pair must reproduce the eager result — both slots re-bind.
            for (int trial = 0; trial < 4; trial++)
            {
                var x = Tensor<float>.CreateRandom(new[] { 2, 3 });
                var t = Tensor<float>.CreateRandom(new[] { 2, 2 });
                plan.SetInputs(new[] { x, t });
                var got = plan.Execute();
                var expected = EagerForward(engine, x, w, t);

                Assert.Equal(expected.Length, got.Length);
                for (int i = 0; i < got.Length; i++)
                    Assert.Equal(expected[i], got[i], 3);
            }

            // The discriminating check for the bug this path fixes: hold the PRIMARY input
            // fixed and change ONLY the secondary input. If the secondary input were baked
            // (the pre-fix single-input behavior), the output would be invariant to it.
            var xFixed = Tensor<float>.CreateRandom(new[] { 2, 3 });
            var tA = Tensor<float>.CreateRandom(new[] { 2, 2 });
            var tB = Tensor<float>.CreateRandom(new[] { 2, 2 });

            plan.SetInputs(new[] { xFixed, tA });
            var outA = plan.Execute();
            var a = new float[outA.Length];
            outA.AsSpan().CopyTo(a);

            plan.SetInputs(new[] { xFixed, tB });
            var outB = plan.Execute();

            bool differs = false;
            for (int i = 0; i < outB.Length; i++)
                if (outB[i] != a[i]) { differs = true; break; }
            Assert.True(differs,
                "Output did not change when only the second input varied — that input was baked " +
                "as a constant, which is exactly the multi-step-inference corruption this path removes.");

            // ...and varying the second input must still match eager (re-bind, not just "differs").
            for (int i = 0; i < outB.Length; i++)
                Assert.Equal(EagerForward(engine, xFixed, w, tB)[i], outB[i], 3);
        }
        finally { cache.Dispose(); }
    }

    [Fact]
    public void MultiInput_NonLeafInput_FailsClosed()
    {
        var engine = new CpuEngine();
        var x = Tensor<float>.CreateRandom(new[] { 2, 3 });
        var w = Tensor<float>.CreateRandom(new[] { 3, 2 });
        // `stray` is never read by the forward, so it is not a traced leaf. Declaring it as a
        // mutable input must fail closed (so the caller falls back to eager) rather than
        // silently doing nothing and replaying stale data.
        var stray = Tensor<float>.CreateRandom(new[] { 2, 2 });

        var cache = new CompiledModelCache<float>();
        try
        {
            Assert.Throws<InvalidOperationException>(() =>
                cache.GetOrCompileInference(new[] { x, stray }, () => engine.TensorMatMul(x, w)));
        }
        finally { cache.Dispose(); }
    }

    [Fact]
    public void MultiInput_EmptyInputs_Throws()
    {
        var cache = new CompiledModelCache<float>();
        try
        {
            var engine = new CpuEngine();
            var x = Tensor<float>.CreateRandom(new[] { 2, 2 });
            Assert.Throws<ArgumentException>(() =>
                cache.GetOrCompileInference(Array.Empty<Tensor<float>>(), () => engine.ReLU(x)));
        }
        finally { cache.Dispose(); }
    }
}
