using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Regression for the compiled-plan gradient-zeroing bug.
///
/// The compiled training plan skips ALL gradient-buffer zeroing when every
/// backward step is specialized (genericBackwardCount == 0), on the assumption
/// that a specialized delegate always OVERWRITES its gradient buffer (beta=0).
/// That assumption is false for a MULTI-CONSUMER tensor: the specialized
/// BroadcastAdd bias-grad delegate ACCUMULATES in place (+=) because a tensor
/// consumed by N ops receives the SUM of N gradient contributions. With the
/// buffer never zeroed, each step's accumulation lands on top of the PRIOR
/// step's gradient, so the gradient grows without bound — the N-BEATS
/// doubly-residual resident path blew its weights up to ~1e13 this way.
/// </summary>
[Collection("CompilationGlobalState")]
public class MultiConsumerGradZeroingTests
{
    /// <summary>
    /// A single bias <c>b</c> broadcast-added to TWO distinct inputs, so <c>b</c>
    /// is multi-consumer and its gradient accumulates from both BroadcastAdd
    /// backwards. The whole graph (BroadcastAdd + scalar ReduceSum + Add) compiles
    /// to an all-specialized backward, which is exactly the path that skipped
    /// zeroing. loss = sum(x1 + b) + sum(x2 + b), so d(loss)/db[f] = 2*batch on
    /// EVERY step (input- and b-independent). With the bug the second step reports
    /// ~2x that (accumulated onto step 1); the fix keeps it constant.
    /// </summary>
    [Fact]
    public void AllSpecialized_MultiConsumerBias_GradientStableAcrossSteps()
    {
        var engine = new CpuEngine();
        const int batch = 4, features = 3;

        var rng = new System.Random(7);
        var x1 = new Tensor<double>(new[] { batch, features });
        var x2 = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < x1.Length; i++) { x1[i] = rng.NextDouble(); x2[i] = rng.NextDouble(); }

        var b = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) b[i] = 0.0;

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var h1 = engine.TensorBroadcastAdd(x1, b); // b consumer #1
            var h2 = engine.TensorBroadcastAdd(x2, b); // b consumer #2 → multi-consumer
            var s1 = engine.ReduceSum(h1, null);       // scalar
            var s2 = engine.ReduceSum(h2, null);       // scalar
            engine.TensorAdd(s1, s2);                  // scalar loss
            plan = scope.CompileTraining(new[] { b });
        }

        const double expected = 2.0 * batch; // 8: each branch contributes `batch` per feature
        using (plan)
        {
            // lr = 0 so the optimizer does not move b — keeps the analytic gradient
            // clean and identical every step (the gradient is b-independent anyway).
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);

            plan.Step();
            var g1 = SnapshotGrad(b, features);

            plan.Step();
            var g2 = SnapshotGrad(b, features);

            for (int f = 0; f < features; f++)
            {
                Assert.Equal(expected, g1[f], 6);
                // Pre-fix this is ~16 (8 accumulated onto 8); post-fix it stays 8.
                Assert.Equal(expected, g2[f], 6);
                Assert.Equal(g1[f], g2[f], 9);
            }
        }
    }

    /// <summary>
    /// Same invariant on the GPU (float): the fix's GPU-resident zeroing path is a
    /// SEPARATE code path (MemsetBuffer over the precise buffer indices) from the CPU
    /// one (Array.Clear), so this exercises it directly. Soft-skips when no usable GPU
    /// is present — the CpuEngine test above still proves the fix on any runner.
    /// </summary>
    [Fact]
    public void AllSpecialized_MultiConsumerBias_GradientStableAcrossSteps_OnGpu()
    {
        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.SupportsGpu)
            return; // no usable GPU here; CPU test covers correctness

        const int batch = 4, features = 3;
        var rng = new System.Random(7);
        var x1 = new Tensor<float>(new[] { batch, features });
        var x2 = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < x1.Length; i++) { x1[i] = (float)rng.NextDouble(); x2[i] = (float)rng.NextDouble(); }
        var b = new Tensor<float>(new[] { features });

        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var h1 = gpu.TensorBroadcastAdd(x1, b);
                var h2 = gpu.TensorBroadcastAdd(x2, b);
                var s1 = gpu.ReduceSum(h1, null);
                var s2 = gpu.ReduceSum(h2, null);
                gpu.TensorAdd(s1, s2);
                plan = scope.CompileTraining(new[] { b });
            }

            const float expected = 2.0f * batch; // 8
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                plan.Step();
                var g1 = SnapshotGradF(b, features);
                plan.Step();
                var g2 = SnapshotGradF(b, features);
                for (int f = 0; f < features; f++)
                {
                    Assert.Equal(expected, g1[f], 3);
                    Assert.Equal(expected, g2[f], 3); // pre-fix would be ~16 on step 2
                    Assert.Equal(g1[f], g2[f], 4);
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    private static double[] SnapshotGrad(Tensor<double> param, int n)
    {
        var grad = param.Grad ?? throw new System.InvalidOperationException("param.Grad was null after Step");
        var copy = new double[n];
        for (int i = 0; i < n; i++) copy[i] = grad[i];
        return copy;
    }

    private static float[] SnapshotGradF(Tensor<float> param, int n)
    {
        var grad = param.Grad ?? throw new System.InvalidOperationException("param.Grad was null after Step");
        var copy = new float[n];
        for (int i = 0; i < n; i++) copy[i] = grad[i];
        return copy;
    }
}
