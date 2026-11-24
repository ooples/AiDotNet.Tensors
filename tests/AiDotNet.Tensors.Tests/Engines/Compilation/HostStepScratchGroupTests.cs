using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// CPU plans compiled inside one <see cref="HostStepScratchGroup"/> co-own the arrays their steps use only within the
/// step: two plans stepped one after the other (a GAN's discriminator and generator) hold the larger working set, not
/// the sum. Each plan must still compute exactly what it computes alone, step after step, while the other plan
/// overwrites the shared arrays in between.
/// </summary>
[Collection("CompilationGlobalState")]
public class HostStepScratchGroupTests
{
    private const int Large = 1 << 16;

    [Fact]
    public void Take_HandsAnArrayOfTheExactLengthToEachOwnerOnce()
    {
        var group = new HostStepScratchGroup();
        object first = new(), second = new();
        var a = group.Take<float>(Large, first);
        var b = group.Take<float>(Large, first);
        Assert.NotSame(a, b);                       // one owner never gets an array it already holds
        Assert.Equal(Large, a.Length);
        var shared = group.Take<float>(Large, second);
        Assert.True(ReferenceEquals(shared, a) || ReferenceEquals(shared, b));
        Assert.Equal(1, group.Shares);
        // A longer array is never handed out: a kernel given the backing array may work over all of it.
        var longer = group.Take<float>(Large - 1, second);
        Assert.Equal(Large - 1, longer.Length);
        Assert.Equal(3, group.Count);
        Assert.IsType<double[]>(group.Take<double>(Large, first));   // element types never mix
    }

    [Fact]
    public void AnArrayWhoseOwnersAreCollected_IsHandedOnAgain()
    {
        var group = new HostStepScratchGroup();
        var array = TakeForACollectedOwner(group);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        Assert.Same(array, group.Take<float>(Large, new object()));
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static float[] TakeForACollectedOwner(HostStepScratchGroup group) => group.Take<float>(Large, new object());

    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static Tensor<float>[] MakeParams(int seed) => new[]
    {
        Filled(new[] { 8, 4, 3, 3 }, seed, 0.5f), Filled(new[] { 8 }, seed + 1, 0.5f),
        Filled(new[] { 8, 8, 3, 3 }, seed + 2, 0.3f), Filled(new[] { 8 }, seed + 3, 0.5f),
    };

    // Two convolution chains whose activations ([2, 8, 64, 64] = 64K floats each) the backward reads.
    private static Tensor<float> Forward(IEngine engine, Tensor<float> x, Tensor<float>[] p)
    {
        int[] one = { 1, 1 }, pad = { 1, 1 };
        var h = engine.ReLU(engine.TensorChannelBiasAdd(engine.Conv2D(x, p[0], one, pad, one), p[1]));
        var y = engine.Tanh(engine.TensorChannelBiasAdd(engine.Conv2D(h, p[2], one, pad, one), p[3]));
        return engine.ReduceSum(engine.TensorMultiply(y, y), null);
    }

    private static (ICompiledTrainingPlan<float> Plan, Tensor<float>[] Params) Compile(IEngine engine, Tensor<float> x, int seed)
    {
        var ps = MakeParams(seed);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Forward(engine, x, ps);
            plan = scope.CompileTraining(ps);
        }
        plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.01f);
        return (plan, ps);
    }

    // Steps the two plans in turn; returns each step's losses and every parameter after it.
    private static (float[] Losses, float[][] Weights) Alternate(HostStepScratchGroup? group)
    {
        var engine = new CpuEngine();
        var xa = Filled(new[] { 2, 4, 64, 64 }, 1, 2f);
        var xb = Filled(new[] { 2, 4, 64, 64 }, 2, 2f);
        (ICompiledTrainingPlan<float> Plan, Tensor<float>[] Params) a, b;
        HostStepScratchGroup.Scope? entered = group?.Enter();
        try
        {
            a = Compile(engine, xa, 10);
            b = Compile(engine, xb, 20);
        }
        finally
        {
            entered?.Dispose();
        }
        using (a.Plan)
        using (b.Plan)
        {
            const int steps = 4;
            var losses = new float[2 * steps];
            var weights = new float[2 * steps * 4][];
            for (int s = 0; s < steps; s++)
            {
                losses[2 * s] = a.Plan.Step()[0];
                losses[2 * s + 1] = b.Plan.Step()[0];
                for (int i = 0; i < 4; i++)
                {
                    weights[(2 * s) * 4 + i] = a.Params[i].ToArray();
                    weights[(2 * s + 1) * 4 + i] = b.Params[i].ToArray();
                }
            }
            return (losses, weights);
        }
    }

    [Fact]
    public void PlansSharingAGroup_StepExactlyAsAlone_WhileHoldingLessThanTheSum()
    {
        var group = new HostStepScratchGroup();
        var shared = Alternate(group);
        var alone = Alternate(null);

        Assert.True(group.Shares > 0, "the second plan took none of the first plan's arrays, so this test proves nothing");
        Assert.Equal(alone.Losses, shared.Losses);
        for (int k = 0; k < alone.Weights.Length; k++)
            Assert.Equal(alone.Weights[k], shared.Weights[k]);
    }
}
