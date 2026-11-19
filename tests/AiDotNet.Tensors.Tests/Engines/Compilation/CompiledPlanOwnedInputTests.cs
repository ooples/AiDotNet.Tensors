using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The <see cref="CompiledModelCache{T}"/> overloads whose forward takes its input trace on a plan-owned copy of it, so
/// feeding a plan never writes into a caller's tensor. Traced on the caller's tensor instead, that tensor becomes the
/// plan's input slot: SetInputs(B) copies B into the caller's A, and passing A again replays B's data (the copy into
/// the slot is skipped when the caller hands back the slot itself). AiDotNet's compiled inference returned B's output
/// for A in 45 model families that way.
/// </summary>
public class CompiledPlanOwnedInputTests
{
    private static Tensor<double> Input(int seed)
    {
        var t = new Tensor<double>(new[] { 2, 3 });
        var rng = new Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static Tensor<double> Weight()
    {
        var w = new Tensor<double>(new[] { 3, 4 });
        for (int i = 0; i < w.Length; i++) w[i] = (i % 5) * 0.2 - 0.4;
        return w;
    }

    private static double[] Run(ICompiledPlan<double> plan) => plan.Execute().ToArray();

    [Fact]
    public void TheClosureOverload_WritesLaterInputsIntoTheCallersFirstTensor()
    {
        // The hazard the owned-input overloads exist to avoid, pinned so the contrast stays visible.
        var engine = new CpuEngine();
        var w = Weight();
        var a = Input(1);
        var b = Input(2);
        var aBefore = a.ToArray();
        using var cache = new CompiledModelCache<double>();
        var plan = cache.GetOrCompileInference(a, () => engine.TensorMatMul(engine.TensorTanh(a), w));
        plan.SetInputs(new[] { b });
        Assert.NotEqual(aBefore, a.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OwnedInputOverload_ReadsEachCallsInput_AndNeverWritesIt(bool symbolic)
    {
        var engine = new CpuEngine();
        var w = Weight();
        Func<Tensor<double>, Tensor<double>> forward = x => engine.TensorMatMul(engine.TensorTanh(x), w);
        var a = Input(1);
        var b = Input(2);
        var aBefore = a.ToArray();
        var bBefore = b.ToArray();
        var expectedA = forward(a).ToArray();
        var expectedB = forward(b).ToArray();
        Assert.NotEqual(expectedA, expectedB);

        using var cache = new CompiledModelCache<double>();
        ICompiledPlan<double> Plan(Tensor<double> x)
        {
            var plan = symbolic
                ? cache.GetOrCompileInference(x, forward, SymbolicShape.BatchDynamic(x._shape))
                : cache.GetOrCompileInference(x, forward);
            plan.SetInputs(new[] { x });
            return plan;
        }

        AssertClose(expectedA, Run(Plan(a)), "A");
        AssertClose(expectedB, Run(Plan(b)), "B");
        AssertClose(expectedA, Run(Plan(a)), "A again");
        Assert.Equal(aBefore, a.ToArray());
        Assert.Equal(bBefore, b.ToArray());
    }

    [Fact]
    public void MultiInputOwnedOverload_ReadsEachCallsInputs_AndNeverWritesThem()
    {
        var engine = new CpuEngine();
        var w = Weight();
        Func<Tensor<double>[], Tensor<double>> forward = x => engine.TensorMatMul(engine.TensorAdd(engine.TensorTanh(x[0]), x[1]), w);
        var a = new[] { Input(1), Input(3) };
        var b = new[] { Input(2), Input(4) };
        var a0 = a[0].ToArray();
        var a1 = a[1].ToArray();
        var expectedA = forward(a).ToArray();
        var expectedB = forward(b).ToArray();

        using var cache = new CompiledModelCache<double>();
        AssertClose(expectedA, Run(cache.GetOrCompileInference(a, forward)), "A");
        AssertClose(expectedB, Run(cache.GetOrCompileInference(b, forward)), "B");
        AssertClose(expectedA, Run(cache.GetOrCompileInference(a, forward)), "A again");
        Assert.Equal(a0, a[0].ToArray());
        Assert.Equal(a1, a[1].ToArray());
    }

    [Fact]
    public void AForwardThatIgnoresItsArgument_FailsToCompile_RatherThanReplayingStaleData()
    {
        var engine = new CpuEngine();
        var w = Weight();
        var a = Input(1);
        using var cache = new CompiledModelCache<double>();
        Assert.ThrowsAny<Exception>(() => cache.GetOrCompileInference(a, _ => engine.TensorMatMul(engine.TensorTanh(a), w)));
    }

    private static void AssertClose(double[] expected, double[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12 + 1e-9 * Math.Abs(expected[i]),
                $"{label}[{i}]: {actual[i]:R} vs eager {expected[i]:R}");
    }
}
