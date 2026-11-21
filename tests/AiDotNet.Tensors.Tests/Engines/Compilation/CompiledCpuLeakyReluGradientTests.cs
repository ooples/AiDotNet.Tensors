using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class CompiledCpuLeakyReluGradientTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    public CompiledCpuLeakyReluGradientTests() => AiDotNetEngine.Current = new CpuEngine();
    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    private static Action<IEngine> Replay<T>(Tensor<T> input, Tensor<T> upstream, Tensor<T> destination,
        double slope, bool shared = false)
    {
        var output = new Tensor<T>(input.Shape.ToArray());
        var step = new CompiledStep<T>("LeakyReLU", (_, _) => { }, output, new[] { input },
            savedState: new object[] { slope });
        var gradients = new Dictionary<Tensor<T>, Tensor<T>> { [input] = destination, [output] = upstream };
        return CompiledTrainingPlan<T>.BuildCpuLeakyReluBackward(step, gradients,
            new Dictionary<Tensor<T>, int> { [input] = shared ? 2 : 1 }, new CpuEngine(), null)
            ?? throw new InvalidOperationException("Contiguous CPU LeakyReLU must write the planned derivative.");
    }

    [Theory]
    [InlineData(.1)]
    [InlineData(.01)]
    public void FloatMatchesCanonicalOldPathIncludingZeroNaNAndInfinity(double slope)
    {
        var input = new Tensor<float>(new[] { -2f, -0f, 0f, 3f, float.NaN, float.NegativeInfinity, float.PositiveInfinity,
            -1e-39f, 1e-39f, -4f, 5f }, new[] { 11 });
        var upstream = new Tensor<float>(new[] { 1f, -2f, 3f, -4f, 5f, -6f, 7f, 8f, -9f, float.NaN, -0f }, new[] { 11 });
        var expected = new CpuEngine().LeakyReluBackward(upstream, input, slope).ToArray();
        var output = new Tensor<float>(input.Shape.ToArray());
        output.Fill(777);
        Replay(input, upstream, output, slope)(new CpuEngine());
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(output[i]));
            else Assert.Equal(MathCompat.SingleToInt32Bits(expected[i]), MathCompat.SingleToInt32Bits(output[i]));
        }
    }

    [Theory]
    [InlineData(.1)]
    [InlineData(.01)]
    public void DoubleMatchesCanonicalOldPathIncludingZeroNaNAndInfinity(double slope)
    {
        var input = new Tensor<double>(new[] { -2.0, -0.0, 0.0, 3.0, double.NaN, double.NegativeInfinity,
            double.PositiveInfinity, -1e-310, 1e-310 }, new[] { 9 });
        var upstream = new Tensor<double>(new[] { 1.0, -2, 3, -4, 5, -6, 7, 8, -9 }, new[] { 9 });
        var expected = new CpuEngine().LeakyReluBackward(upstream, input, slope).ToArray();
        var output = new Tensor<double>(input.Shape.ToArray());
        Replay(input, upstream, output, slope)(new CpuEngine());
        for (int i = 0; i < expected.Length; i++)
        {
            if (double.IsNaN(expected[i])) Assert.True(double.IsNaN(output[i]));
            else Assert.Equal(BitConverter.DoubleToInt64Bits(expected[i]), BitConverter.DoubleToInt64Bits(output[i]));
        }
    }

    [Fact]
    public void DoubleDerivativeMatchesIndependentFiniteDifferences()
    {
        double[] values = { -2, -.125, .25, 3 };
        double[] seeds = { 1.5, -2, .375, -1 };
        var input = new Tensor<double>(values, new[] { 4 });
        var upstream = new Tensor<double>(seeds, new[] { 4 });
        var output = new Tensor<double>(new[] { 4 });
        Replay(input, upstream, output, .1)(new CpuEngine());
        static double Loss(double x, double seed) => seed * (x > 0 ? x : .1 * x);
        for (int i = 0; i < values.Length; i++)
        {
            const double h = 1e-6;
            double expected = (Loss(values[i] + h, seeds[i]) - Loss(values[i] - h, seeds[i])) / (2 * h);
            Assert.InRange(Math.Abs(output[i] - expected), 0, 1e-8);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PhysicalPartialOverlapsSnapshotBothSourcesWithDistinctMemoryWrappers(bool accumulate)
    {
        double[] shared = Enumerable.Range(0, 24).Select(i => (i - 9) * .25).ToArray();
        var input = Tensor<double>.FromMemory(shared.AsMemory(1, 8), new[] { 2, 4 });
        var upstream = Tensor<double>.FromMemory(shared.AsMemory(6, 8), new[] { 2, 4 });
        var output = Tensor<double>.FromMemory(shared.AsMemory(3, 8), new[] { 2, 4 });
        var original = shared.ToArray();
        var derivative = new CpuEngine().LeakyReluBackward(upstream, input, .1).ToArray();
        var expected = original.ToArray();
        for (int i = 0; i < derivative.Length; i++) expected[3 + i] = derivative[i] + (accumulate ? original[3 + i] : 0);
        int prior = DifferentiableOps.GradWriteGeneration;
        try
        {
            DifferentiableOps.GradWriteGeneration = 0;
            Replay(input, upstream, output, .1, accumulate)(new CpuEngine());
        }
        finally { DifferentiableOps.GradWriteGeneration = prior; }
        Assert.Equal(expected, shared);
    }

    [Fact]
    public void ExactSourceAliasesAreSafeAndReboundGradientsStayLive()
    {
        var input = new Tensor<float>(new[] { -2f, 3, -4, 5 }, new[] { 4 });
        var upstream = new Tensor<float>(new[] { 1f, 2, 3, 4 }, new[] { 4 });
        var expected = new CpuEngine().LeakyReluBackward(upstream, input, .1).ToArray();
        Replay(input, upstream, upstream, .1)(new CpuEngine());
        Assert.Equal(expected, upstream.ToArray());

        var destination = new Tensor<float>(new[] { 4 });
        var replay = Replay(input, upstream, destination, .01);
        var newUpstream = new Tensor<float>(new[] { 4f, 3, 2, 1 }, new[] { 4 });
        var newDestination = new Tensor<float>(new[] { 4 });
        upstream.RebindStorageFrom(newUpstream);
        destination.RebindStorageFrom(newDestination);
        replay(new CpuEngine());
        Assert.Equal(new CpuEngine().LeakyReluBackward(upstream, input, .01).ToArray(), newDestination.ToArray());
    }

    [Fact]
    public void SharedFirstWriteOverwritesStaleStorageThenAccumulatesWithinEachGeneration()
    {
        var input = new Tensor<float>(new[] { -2f, -0f, 0f, 3f }, new[] { 4 });
        var upstream = new Tensor<float>(new[] { 1f, 2, 3, 4 }, new[] { 4 });
        var output = new Tensor<float>(new[] { 4 });
        var expected = new CpuEngine().LeakyReluBackward(upstream, input, .1).ToArray();
        var replay = Replay(input, upstream, output, .1, true);
        int prior = DifferentiableOps.GradWriteGeneration;
        try
        {
            for (int step = 0; step < 3; step++)
            {
                output.Fill(999);
                DifferentiableOps.GradWriteGeneration = DifferentiableOps.NextGradWriteGeneration();
                replay(new CpuEngine());
                Assert.Equal(expected, output.ToArray());
                replay(new CpuEngine());
                Assert.Equal(expected.Select(value => value + value), output.ToArray());
            }
        }
        finally { DifferentiableOps.GradWriteGeneration = prior; }
    }

    [Fact]
    public void CompiledReshapeFanoutSumsBothSlopesAndTheSharedParameterBranchEveryStep()
    {
        var engine = new CpuEngine();
        var input = new Tensor<double>(new[] { 8 });
        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(new[] { input }))
        {
            var view = engine.Reshape(input, new[] { 2, 4 });
            var branch = engine.TensorAdd(engine.LeakyReLU(view, .1), engine.LeakyReLU(view, .01));
            var loss = engine.TensorAdd(engine.ReduceSum(branch, null),
                engine.ReduceSum(engine.TensorMultiplyScalar(input, .25), null));
            plan = scope.CompileTraining(new[] { input }, loss);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
            for (int step = 0; step < 3; step++)
            {
                for (int i = 0; i < input.Length; i++) input[i] = (i - 3) * (step % 2 == 0 ? 1 : -1);
                plan.Step();
                for (int i = 0; i < input.Length; i++)
                    Assert.InRange(Math.Abs(plan.Gradients[0][i] - (input[i] > 0 ? 2.25 : .36)), 0, 1e-14);
            }
        }
    }

    [Fact]
    public void CustomEngineRetainsItsBackwardOverride()
    {
        var engine = new CountingBackwardEngine();
        var input = new Tensor<float>(new[] { -2f, 0, 3, 4 }, new[] { 4 });
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { input }))
        {
            var loss = engine.ReduceSum(engine.LeakyReLU(input, .1f), null);
            plan = scope.CompileTraining(new[] { input }, loss);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
            int before = engine.Calls;
            for (int step = 0; step < 3; step++) plan.Step();
            Assert.True(engine.Calls >= before + 3);
            Assert.Equal(new[] { .1f, .1f, 1f, 1f }, plan.Gradients[0].ToArray());
        }
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void CompiledStepDoesNotAllocateAnActivationSizedDerivative()
    {
        static long Allocated(int columns)
        {
            var engine = new CpuEngine();
            var input = new Tensor<float>(new[] { 32, columns });
            input.Fill(-.5f);
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { input }))
            {
                var loss = engine.ReduceSum(engine.LeakyReLU(input, .1f), null);
                plan = scope.CompileTraining(new[] { input }, loss);
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
                for (int i = 0; i < 4; i++) plan.Step();
                long before = GC.GetTotalAllocatedBytes(true);
                for (int i = 0; i < 8; i++) plan.Step();
                return GC.GetTotalAllocatedBytes(true) - before;
            }
        }
        long small = Allocated(64), large = Allocated(65536);
        Assert.True(large <= small + 128 * 1024,
            $"Eight planned LeakyReLU steps allocated {large:N0}B versus {small:N0}B; one large derivative is8MiB.");
    }
#endif

    private sealed class CountingBackwardEngine : CpuEngine
    {
        internal int Calls;
        public override Tensor<T> LeakyReluBackward<T>(Tensor<T> upstream, Tensor<T> input, double slope)
        {
            Calls++;
            return base.LeakyReluBackward(upstream, input, slope);
        }
    }
}
