using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A BatchNorm layer keeps its running statistics with an in-place EMA over the batch mean/variance the op hands out
/// (AiDotNet's BatchNormalizationLayer: runningMean *= m; runningMean += (1 - m) * batchMean). In a compiled plan the
/// BatchNorm node refreshes those tensors as a side effect of its replay, so the EMA must replay AFTER it. Measured in
/// AiDotNet (GraFPrint): every compiled step folded in the PREVIOUS step's batch statistics - after two steps the
/// running mean was exactly 0.9 * rm1 + 0.1 * mean(step 1) - while the trained weights matched eager to 1e-7.
/// </summary>
[Collection("CompilationGlobalState")]
public class BatchNormCompiledRunningStatsTests
{
    private const double Momentum = 0.9, Epsilon = 1e-5;

    [Theory]
    [InlineData(2, false)]
    [InlineData(4, false)]
    [InlineData(2, true)]
    [InlineData(4, true)]
    public void The_running_statistics_update_uses_the_current_steps_batch_statistics(int rank, bool throughCustomNode)
    {
        int[] shape = rank == 2 ? new[] { 8, 3 } : new[] { 4, 3, 2, 2 };
        const int C = 3;
        var engine = new CpuEngine();
        var input = new Tensor<double>(shape);
        var gamma = new Tensor<double>(new[] { C });
        var beta = new Tensor<double>(new[] { C });
        for (int c = 0; c < C; c++) { gamma[c] = 1.0 + 0.1 * c; beta[c] = 0.05 * c; }
        var runningMean = new Tensor<double>(new[] { C });
        var runningVar = new Tensor<double>(new[] { C });
        for (int c = 0; c < C; c++) runningVar[c] = 1.0;

        void Fill(int seed)
        {
            var rng = new System.Random(seed);
            for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * (seed + 1) - 0.3 * seed;
        }

        // Eager reference: per-channel mean/variance of an input, straight from the op.
        (double[] Mean, double[] Var) Stats()
        {
            var probe = new CpuEngine();
            var copy = new Tensor<double>(input.ToArray(), shape);
            if (throughCustomNode) for (int i = 0; i < copy.Length; i++) copy[i] *= 2;
            probe.BatchNorm(copy, gamma, beta, Epsilon, out var m, out var v);
            return (m.ToArray(), v.ToArray());
        }

        Fill(1);
        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            // GraFPrint's shape: the BatchNorm input is a custom replay node's output (recomputed from the step's
            // input at every replay), reshaped.
            Tensor<double> bnInput = input;
            if (throughCustomNode)
            {
                var graphScope = GraphMode.Current!;
                var source = input;
                var flat = new[] { input.Length };
                var custom = graphScope.RecordUnary(LazyNodeType.Custom, "DoubleIt", input, flat,
                    (eng, o) => { var src = source.AsSpan(); var dst = o.AsWritableSpan(); for (int i = 0; i < dst.Length; i++) dst[i] = 2 * src[i]; });
                { var src = input.AsSpan(); var dst = custom.AsWritableSpan(); for (int i = 0; i < dst.Length; i++) dst[i] = 2 * src[i]; }
                bnInput = engine.Reshape(custom, shape);
            }
            var output = engine.BatchNorm(bnInput, gamma, beta, Epsilon, out var batchMean, out var batchVar);
            using (new NoGradScope<double>())
            {
                engine.TensorMultiplyScalarInPlace(runningMean, Momentum);
                engine.TensorAddInPlace(runningMean, engine.TensorMultiplyScalar(batchMean, 1 - Momentum));
                engine.TensorMultiplyScalarInPlace(runningVar, Momentum);
                engine.TensorAddInPlace(runningVar, engine.TensorMultiplyScalar(batchVar, 1 - Momentum));
            }
            engine.ReduceSum(engine.TensorMultiply(output, output), null);
            plan = scope.CompileTraining(new[] { gamma, beta });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            foreach (int seed in new[] { 2, 3, 4 })
            {
                Fill(seed);
                var current = Stats();
                var meanBefore = runningMean.ToArray();
                var varBefore = runningVar.ToArray();
                plan.Step();
                for (int c = 0; c < C; c++)
                {
                    double expectedMean = Momentum * meanBefore[c] + (1 - Momentum) * current.Mean[c];
                    double expectedVar = Momentum * varBefore[c] + (1 - Momentum) * current.Var[c];
                    Assert.True(System.Math.Abs(runningMean[c] - expectedMean) < 1e-9,
                        $"rank {rank} custom {throughCustomNode}, step with seed {seed}, channel {c}: running mean {runningMean[c]:R}, expected {expectedMean:R}");
                    Assert.True(System.Math.Abs(runningVar[c] - expectedVar) < 1e-9,
                        $"rank {rank} custom {throughCustomNode}, step with seed {seed}, channel {c}: running variance {runningVar[c]:R}, expected {expectedVar:R}");
                }
            }
        }
    }
}
