using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #662 regression guard. The compiled-inference replay (AutoTracer.TryCompile) rebuilds the lazy
/// graph by invoking each op's recorded replay delegate inside <see cref="GraphMode"/>. 52 RecordOp
/// sites used to pass <c>eng =&gt; result</c>, which returns the TRACE-TIME output tensor as a
/// constant — so the compiled plan emitted stale values regardless of the live input. The fix
/// records <c>eng =&gt; eng.OpName(capturedInputs)</c> (recompute) so the op becomes a real graph
/// node.
///
/// These tests mirror TryCompile exactly (GraphMode.Enable → run the SAME delegate body the fixed
/// sites now record → CompileInference) and then execute the plan on a NEW input. A correct
/// (recomputed) plan reflects the new input; the old stale-constant plan would return the
/// trace-time output. The <see cref="StaleConstantDelegate_IsDetected_NegativeControl"/> test pins
/// what the bug looked like so this guard can't silently pass if recompile ever regresses.
/// </summary>
public class CompiledReplayRecomputeParityTests
{
    private static void Fill(Tensor<float> t, Random r)
    {
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(r.NextDouble() * 2 - 1);
    }

    private static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    /// <summary>
    /// Build a plan from <paramref name="build"/> (the recompute delegate body) over a placeholder
    /// input, then execute on A and B (filled into the SAME placeholder, as the plan reads live
    /// data). Returns both compiled outputs.
    /// </summary>
    private static (float[] outA, float[] outB) TraceReplay(
        Tensor<float> input, Tensor<float> aData, Tensor<float> bData,
        Func<IEngine, Tensor<float>, Tensor<float>> build)
    {
        var eng = new CpuEngine();
        aData.AsSpan().CopyTo(input.AsWritableSpan());
        using var scope = GraphMode.Enable();
        var graphOut = build(eng, input);
        var plan = scope.CompileInference<float>();
        Assert.NotNull(plan);

        aData.AsSpan().CopyTo(input.AsWritableSpan());
        plan!.Execute();
        var outA = graphOut.AsSpan().ToArray();

        bData.AsSpan().CopyTo(input.AsWritableSpan());
        plan!.Execute();
        var outB = graphOut.AsSpan().ToArray();
        return (outA, outB);
    }

    private static void AssertReflectsLiveInput(
        string name, int[] inputShape, Func<CpuEngine, Tensor<float>, Tensor<float>> eager,
        Func<IEngine, Tensor<float>, Tensor<float>> build)
    {
        var eng = new CpuEngine();
        var rng = new Random(662);
        var aData = new Tensor<float>(inputShape); Fill(aData, rng);
        var bData = new Tensor<float>(inputShape); Fill(bData, rng);

        var eagerA = eager(eng, aData).AsSpan().ToArray();
        var eagerB = eager(eng, bData).AsSpan().ToArray();
        Assert.True(MaxAbsDiff(eagerA, eagerB) > 1e-3f,
            $"{name}: A and B must produce different eager outputs (test is meaningless otherwise)");

        var input = new Tensor<float>(inputShape);
        var (outA, outB) = TraceReplay(input, aData, bData, build);

        Assert.True(MaxAbsDiff(outA, eagerA) < 1e-3f, $"{name}: compiled(A) != eager(A)");
        // The crux: compiled(B) must equal eager(B), NOT the trace-time output (eagerA).
        Assert.True(MaxAbsDiff(outB, eagerB) < 1e-3f,
            $"{name}: compiled(B) != eager(B) — stale-constant replay (#662)? diff vs eagerB=" +
            $"{MaxAbsDiff(outB, eagerB):E3}, diff vs eagerA={MaxAbsDiff(outB, eagerA):E3}");
    }

    [Fact]
    public void Reshape_CompiledReplay_ReflectsLiveInput()
        => AssertReflectsLiveInput("Reshape", new[] { 2, 6 },
            (e, x) => e.Reshape(x, new[] { 3, 4 }),
            (e, x) => e.Reshape(x, new[] { 3, 4 }));

    [Fact]
    public void TensorAddScalar_CompiledReplay_ReflectsLiveInput()
        => AssertReflectsLiveInput("TensorAddScalar", new[] { 4, 5 },
            (e, x) => e.TensorAddScalar(x, 1.5f),
            (e, x) => e.TensorAddScalar(x, 1.5f));

    [Fact]
    public void AvgPool2D_CompiledReplay_ReflectsLiveInput()
        => AssertReflectsLiveInput("AvgPool2D", new[] { 1, 3, 8, 8 },
            (e, x) => e.AvgPool2D(x, 2, 2, 0),
            (e, x) => e.AvgPool2D(x, 2, 2, 0));

    [Fact]
    public void Conv2D_CompiledReplay_ReflectsLiveInput()
    {
        var kernel = new Tensor<float>(new[] { 3, 2, 3, 3 });
        Fill(kernel, new Random(99));
        AssertReflectsLiveInput("Conv2D", new[] { 1, 2, 8, 8 },
            (e, x) => e.Conv2D(x, kernel, 1, 1, 1),
            (e, x) => e.Conv2D(x, kernel, 1, 1, 1));
    }

    [Fact]
    public void LayerNorm_CompiledReplay_ReflectsLiveInput()
    {
        var gamma = new Tensor<float>(new[] { 6 });
        var beta = new Tensor<float>(new[] { 6 });
        Fill(gamma, new Random(7)); Fill(beta, new Random(8));
        AssertReflectsLiveInput("LayerNorm", new[] { 4, 6 },
            (e, x) => e.LayerNorm(x, gamma, beta, 1e-5, out _, out _),
            (e, x) => e.LayerNorm(x, gamma, beta, 1e-5, out _, out _));
    }

    /// <summary>
    /// Negative control: a plan built the OLD way (<c>eng =&gt; result</c>, i.e. feeding the
    /// trace-time output in as a constant) is stale — compiled(B) returns the trace-time A output,
    /// not eager(B). Pins the bug #662 fixed so the positive tests above can't silently regress.
    /// </summary>
    [Fact]
    public void StaleConstantDelegate_IsDetected_NegativeControl()
    {
        var eng = new CpuEngine();
        var rng = new Random(662);
        var aData = new Tensor<float>(new[] { 4, 5 }); Fill(aData, rng);
        var bData = new Tensor<float>(new[] { 4, 5 }); Fill(bData, rng);

        var eagerA = eng.TensorAddScalar(aData, 1.5f).AsSpan().ToArray();
        var eagerB = eng.TensorAddScalar(bData, 1.5f).AsSpan().ToArray();

        var input = new Tensor<float>(new[] { 4, 5 });
        aData.AsSpan().CopyTo(input.AsWritableSpan());
        // Reproduce the OLD broken delegate: precompute the result and feed it as a constant.
        var traceResult = eng.TensorAddScalar(input, 1.5f);
        var (outA, outB) = TraceReplay(input, aData, bData,
            (e, _) => e.TensorAddScalar(traceResult, 0f)); // wraps the constant; ignores live input

        // Stale: compiled(B) equals the trace-time A output, NOT eager(B).
        Assert.True(MaxAbsDiff(outB, eagerA) < 1e-3f, "negative control should be stale to A");
        Assert.True(MaxAbsDiff(outB, eagerB) > 1e-3f, "negative control should NOT reflect B");
    }
}
