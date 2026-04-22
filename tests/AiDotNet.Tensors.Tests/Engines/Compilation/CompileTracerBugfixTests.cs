using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Regression tests for issues #238 and #239.
///
/// #238: infinite recursion in the compile tracer when the forward lambda
/// exercises a cross-type engine op (FFT family, ComplexMagnitude,
/// ComplexPhase, IFFTReal). Root cause was that
/// <see cref="CrossTypeLazyNode{TIn,TOut}.Realize"/> did not suspend
/// <see cref="GraphMode"/> while running its <c>Execute</c> delegate, so
/// the delegate's re-invocation of the engine op recorded another lazy
/// node, whose subsequent <c>AsSpan</c> forced another Realize, ad
/// infinitum.
///
/// #239: <c>CompiledModelCache.GetOrCompileInference</c> silently returned
/// an unusable 0-step plan when the forward lambda allocated a fresh
/// <c>Tensor&lt;T&gt;</c> and wrote to it via the indexer (those ops
/// bypass the tracer). Downstream <c>plan.SetInputs(...)</c> then threw
/// a cryptic <c>"compiled with 0 captured input(s); got 1"</c>. The fix
/// surfaces a clear error at compile time explaining the root cause.
/// </summary>
public class CompileTracerBugfixTests
{
    // ── Issue #238 ───────────────────────────────────────────────────────

    [Fact]
    public void Issue238_CrossTypeLazyNode_Realize_SuspendsGraphMode()
    {
        // Build a CrossTypeLazyNode whose Execute calls back into an engine op
        // that checks GraphMode.IsActive. Before the fix, GraphMode was still
        // active during Realize, so the callback re-recorded a new lazy node
        // and AsSpan on the returned lazy tensor triggered infinite recursion.
        // After the fix, Realize suspends GraphMode so the callback takes the
        // eager path exactly once.

        var engine = new CpuEngine();

        using var scope = GraphMode.Enable();

        // Populate a real input tensor, then route through a cross-type op
        // (real → complex FFT) so the returned tensor has a CrossTypeLazyNode
        // as its LazySource.
        var realInput = new Tensor<double>(new double[] { 1, 0, 0, 0, 0, 0, 0, 0 },
                                           new[] { 8 });
        var lazyComplex = engine.NativeComplexFFT(realInput);

        // Force materialization via AsSpan — this is the exact trigger from
        // the issue's stack trace. Before the fix this recurses until
        // StackOverflowException; after the fix it returns a real span.
        var span = lazyComplex.AsSpan();
        Assert.Equal(8, span.Length);

        // Sanity: FFT of a unit impulse at index 0 is all-ones in both real
        // and imag components. We just need SOME non-zero deterministic
        // output to prove the eager path actually ran.
        Assert.NotEqual(0.0, span[0].Real);
    }

    [Fact]
    public void Issue238_CompileInference_DoesNotStackOverflow_OnFftMultiplyIfft()
    {
        // End-to-end version of the issue's repro: FFT → complex multiply →
        // IFFTReal inside GetOrCompileInference. Before the fix this threw
        // StackOverflowException during compile; after the fix compile
        // completes cleanly and the lazy-chain realize returns correct
        // values on first output read.
        //
        // Note on plan.StepCount: CompiledInferencePlan<T>.Compile currently
        // only captures LazyNode<T> (not CrossTypeLazyNode<TIn,TOut> or
        // LazyNode<OtherT>) so a pure cross-type chain typed at T=double
        // compiles to 0 steps. The lazy nodes still execute correctly via
        // auto-materialization on AsSpan — the reported StackOverflow was
        // about that path crashing, which this test guards against.

        var engine = new CpuEngine();
        var realInput = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 },
                                           new[] { 8 });
        using var cache = new CompiledModelCache<double>();

        // Identity filter: all-ones in complex — ensures the IFFT of
        // (FFT(x) * 1) == x (up to floating-point rounding).
        var filter = new Tensor<Complex<double>>(new[] { 8 });
        for (int i = 0; i < 8; i++) filter[i] = new Complex<double>(1.0, 0.0);

        var plan = cache.GetOrCompileInference(realInput, () =>
        {
            var spectrum = engine.NativeComplexFFT(realInput);
            var filtered = engine.NativeComplexMultiply(filter, spectrum);
            return engine.NativeComplexIFFTReal(filtered);
        });

        Assert.NotNull(plan);

        // Plan.Execute returns the lazy final output; reading it via the
        // indexer triggers the Realize cascade. Before the fix this was
        // the path that recursed; after the fix each cross-type node
        // realises exactly once.
        var output = plan.Execute();
        for (int i = 0; i < 8; i++)
            Assert.Equal(realInput[i], output[i], precision: 6);
    }

    // ── Issue #239 ───────────────────────────────────────────────────────

    [Fact]
    public void Issue239_GetOrCompileInference_ThrowsClearErrorWhenForwardRecordsNothing()
    {
        // Minimal repro from the issue: forward allocates a fresh Tensor<T>
        // and writes values through the indexer. Both ops bypass the lazy
        // graph, so the scope captures zero nodes. Before the fix this
        // silently returned an unusable 0-step plan and SetInputs threw
        // "compiled with 0 captured input(s); got 1". After the fix the
        // compile step throws a descriptive ArgumentException pointing at
        // the forward lambda.

        var input = new Tensor<double>(new[] { 2 });
        input[0] = 1.0;
        input[1] = 1.0;

        using var cache = new CompiledModelCache<double>();

        var ex = Assert.Throws<ArgumentException>(() =>
        {
            cache.GetOrCompileInference(input, () =>
            {
                var output = new Tensor<double>(new[] { 2 });
                output[0] = input[0] * 2.0;
                output[1] = input[1] * 3.0;
                return output;
            });
        });

        Assert.Equal("forward", ex.ParamName);
        // Spot-check that the error points at the bypass-the-tracer root
        // cause and references the issue — future readers can trace back.
        Assert.Contains("forward lambda did not record any tensor operations",
                        ex.Message);
        Assert.Contains("#239", ex.Message);
    }

    [Fact]
    public void Issue239_GetOrCompileInference_AllowsLegitForward()
    {
        // Positive control: a forward that uses engine ops still compiles
        // fine after the zero-op guard is added.

        var engine = new CpuEngine();
        var input = new Tensor<double>(new double[] { 1, 2 }, new[] { 2 });
        var two = new Tensor<double>(new double[] { 2, 3 }, new[] { 2 });

        using var cache = new CompiledModelCache<double>();

        var plan = cache.GetOrCompileInference(input, () =>
            engine.TensorMultiply(input, two));

        Assert.NotNull(plan);
        Assert.True(plan.StepCount >= 1);

        var output = plan.Execute();
        Assert.Equal(2.0, output[0], precision: 6);
        Assert.Equal(6.0, output[1], precision: 6);
    }

    [Fact]
    public void Issue239_IntShapeOverload_ThrowsSameError()
    {
        // The int[]-shape overload must guard the same scenario — a layer
        // library calling the shape overload shouldn't get a silently
        // broken plan either.

        using var cache = new CompiledModelCache<double>();

        var ex = Assert.Throws<ArgumentException>(() =>
        {
            cache.GetOrCompileInference(new[] { 2 }, () =>
            {
                var output = new Tensor<double>(new[] { 2 });
                output[0] = 1.0;
                output[1] = 2.0;
                return output;
            });
        });

        Assert.Equal("forward", ex.ParamName);
    }
}
