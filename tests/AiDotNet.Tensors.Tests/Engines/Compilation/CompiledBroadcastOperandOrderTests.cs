using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A broadcasting binary op must produce the same values through a compiled plan as it does
/// eagerly, WHICHEVER operand is the one that expands.
/// </summary>
/// <remarks>
/// <para>
/// The compiled replay seeds the output buffer from one operand and then broadcasts the other up
/// into it. It used to seed unconditionally from the FIRST operand, which is only the larger one
/// by convention: <c>big + small</c> filled the buffer and was correct, while <c>small + big</c>
/// wrote just the small operand's elements and left the rest of the buffer holding whatever was
/// there before.
/// </para>
/// <para>
/// That is not a rare shape. <see cref="IEngine.TensorBroadcastTo"/> expresses a genuine expansion
/// as <c>input + zeros(targetShape)</c>, whose first operand is by construction the SMALL one, so
/// every such broadcast replayed through a compiled plan was affected. The stale remainder read as
/// zeros from a fresh buffer -- a silent wrong answer, the sum landing at exactly 1/N of the true
/// value -- and as a previous step's bytes from a recycled one, which surfaced as NaN or ~1e35 and
/// then poisoned every parameter through the gradient.
/// </para>
/// <para>
/// These compare against the eager result rather than a hardcoded constant, so they pin the
/// invariant that actually matters: compiled and eager agree.
/// </para>
/// </remarks>
public class CompiledBroadcastOperandOrderTests
{
    private static Tensor<float> Filled(int[] shape, float scale, int salt)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = scale * (float)Math.Sin((i + salt) * 0.37);
        return t;
    }

    /// <summary>
    /// Compiles <paramref name="build"/> into a training plan and returns the loss its replay
    /// produces, alongside the eager value of the same graph.
    /// </summary>
    private static (double compiled, double eager) CompiledVsEager(
        Func<CpuEngine, (Tensor<float> output, Tensor<float>[] parameters)> build)
    {
        var engine = new CpuEngine();

        var (eagerOutput, _) = build(engine);
        double eager = engine.ReduceSum(eagerOutput, null)[0];

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var (output, parameters) = build(engine);
            engine.ReduceSum(output, null);
            plan = scope.CompileTraining(parameters);
        }

        using (plan)
        {
            // Zero learning rate: this pins the FORWARD replay, so nothing moves underneath it.
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            var loss = new Tensor<float>(new[] { 1 });
            plan.StepInto(loss);
            return (loss[0], eager);
        }
    }

    private static void AssertAgrees(string what,
        Func<CpuEngine, (Tensor<float>, Tensor<float>[])> build)
    {
        var (compiled, eager) = CompiledVsEager(build);
        Assert.False(double.IsNaN(compiled) || double.IsInfinity(compiled),
            $"{what}: compiled replay produced {compiled}, eager was {eager:G8}.");
        Assert.True(Math.Abs(compiled - eager) <= 1e-3 * Math.Max(1.0, Math.Abs(eager)),
            $"{what}: compiled {compiled:G8} != eager {eager:G8}.");
    }

    [Theory]
    [InlineData(4, 3)]
    [InlineData(8, 9)]
    public void BroadcastAdd_AgreesWithEager_WithEitherOperandFirst(int rows, int cols)
    {
        AssertAgrees("small + big", e =>
        {
            var small = Filled(new[] { rows, 1 }, 0.3f, 2);
            var big = Filled(new[] { rows, cols }, 0.1f, 9);
            return (e.TensorAdd(small, big), new[] { small, big });
        });

        AssertAgrees("big + small", e =>
        {
            var small = Filled(new[] { rows, 1 }, 0.3f, 2);
            var big = Filled(new[] { rows, cols }, 0.1f, 9);
            return (e.TensorAdd(big, small), new[] { small, big });
        });
    }

    [Theory]
    [InlineData(4, 3)]
    [InlineData(8, 9)]
    public void BroadcastTo_ExpandsEveryElement_UnderACompiledPlan(int rows, int cols)
    {
        // TensorBroadcastTo is the small-operand-first case by construction.
        AssertAgrees($"broadcastTo [{rows},1] -> [{rows},{cols}]", e =>
        {
            var source = Filled(new[] { rows, 1 }, 0.3f, 2);
            return (e.TensorBroadcastTo(source, new[] { rows, cols }), new[] { source });
        });
    }

    [Fact]
    public void BroadcastTo_HigherRanks_AgreeWithEager()
    {
        const int S = 4, C = 3;

        AssertAgrees("rank 3 [1,S,1] -> [1,S,S]", e =>
        {
            var source = Filled(new[] { 1, S, 1 }, 0.3f, 2);
            return (e.TensorBroadcastTo(source, new[] { 1, S, S }), new[] { source });
        });

        AssertAgrees("rank 4 [1,S,1,C] -> [1,S,S,C]", e =>
        {
            var source = Filled(new[] { 1, S, 1, C }, 0.3f, 2);
            return (e.TensorBroadcastTo(source, new[] { 1, S, S, C }), new[] { source });
        });
    }

    /// <summary>
    /// Subtract and multiply go through the same single builder, and subtract is NOT commutative:
    /// seeding the buffer from the wrong operand would silently compute <c>big - small</c>.
    /// </summary>
    [Theory]
    [InlineData(4, 3)]
    [InlineData(8, 9)]
    public void BroadcastSubtractAndMultiply_PreserveOperandOrder(int rows, int cols)
    {
        AssertAgrees("small - big", e =>
        {
            var small = Filled(new[] { rows, 1 }, 0.3f, 2);
            var big = Filled(new[] { rows, cols }, 0.1f, 9);
            return (e.TensorSubtract(small, big), new[] { small, big });
        });

        AssertAgrees("big - small", e =>
        {
            var small = Filled(new[] { rows, 1 }, 0.3f, 2);
            var big = Filled(new[] { rows, cols }, 0.1f, 9);
            return (e.TensorSubtract(big, small), new[] { small, big });
        });

        AssertAgrees("small * big", e =>
        {
            var small = Filled(new[] { rows, 1 }, 0.3f, 2);
            var big = Filled(new[] { rows, cols }, 0.1f, 9);
            return (e.TensorMultiply(small, big), new[] { small, big });
        });

        AssertAgrees("big * small", e =>
        {
            var small = Filled(new[] { rows, 1 }, 0.3f, 2);
            var big = Filled(new[] { rows, cols }, 0.1f, 9);
            return (e.TensorMultiply(big, small), new[] { small, big });
        });
    }

    /// <summary>
    /// A MUTUAL broadcast has no operand spanning the output; the single builder declines it and
    /// the generic path must still produce the eager answer.
    /// </summary>
    [Fact]
    public void MutualBroadcast_FallsBackAndStillAgrees()
    {
        AssertAgrees("[4,1] + [1,3]", e =>
        {
            var rowsOnly = Filled(new[] { 4, 1 }, 0.3f, 2);
            var colsOnly = Filled(new[] { 1, 3 }, 0.1f, 9);
            return (e.TensorAdd(rowsOnly, colsOnly), new[] { rowsOnly, colsOnly });
        });
    }

    [Fact]
    public void BiasAddPattern_StillAgrees()
    {
        // The big-operand-first shape that always worked — it must keep working.
        const int S = 8, C = 9;
        AssertAgrees("[S,C] + [C] bias add", e =>
        {
            var activations = Filled(new[] { S, C }, 0.3f, 2);
            var bias = Filled(new[] { C }, 0.1f, 9);
            return (e.TensorAdd(activations, bias), new[] { activations, bias });
        });
    }

    /// <summary>
    /// Every element of the expanded output must be written — the defect left most of the buffer
    /// untouched while still reporting the correct SHAPE, so a shape assertion would have passed.
    /// </summary>
    [Fact]
    public void BroadcastTo_WritesEveryElement_NotJustTheSourceLength()
    {
        const int rows = 4, cols = 3;
        var engine = new CpuEngine();
        var source = Filled(new[] { rows, 1 }, 0.3f, 2);
        var expected = engine.TensorBroadcastTo(source, new[] { rows, cols });

        Tensor<float> replayed;
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            replayed = engine.TensorBroadcastTo(source, new[] { rows, cols });
            engine.ReduceSum(replayed, null);
            plan = scope.CompileTraining(new[] { source });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.StepInto(new Tensor<float>(new[] { 1 }));
        }

        Assert.Equal(expected.Length, replayed.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - replayed[i]) < 1e-5f,
                $"element {i}: expected {expected[i]:G8}, compiled replay left {replayed[i]:G8}.");
        }
    }
}
