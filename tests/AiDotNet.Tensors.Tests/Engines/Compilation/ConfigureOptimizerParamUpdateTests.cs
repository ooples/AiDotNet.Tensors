using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Issue #350: after <c>plan.ConfigureOptimizer(Adam, …)</c> + <c>plan.Step()</c>,
/// the parameter tensors registered with the plan must have been
/// written-back-to in place. Pre-fix, AiDotNet 0.79.0 callers saw
/// <c>plan.Step()</c> return successfully (and increment internal step
/// counters) without actually updating the parameter backing memory — the
/// fused-Adam <c>_optimizerUpdate</c> closure ran, but the param tensors
/// the caller registered were unchanged. The fused-Adam compiled training
/// path then silently dead-ended every CNN training run on the consumer
/// (paper-faithful 53-layer GraFPrint, RAPIDFlow, etc.) that bumped to
/// 0.79.0.
///
/// These tests assert the post-Step contract directly: max |Δ params| > 0
/// (i.e. <c>_parameters[p].GetDataArray()</c> was actually mutated) after a
/// single Adam step on a graph with a non-zero gradient.
/// </summary>
public class ConfigureOptimizerParamUpdateTests
{
    /// <summary>
    /// T=float baseline: configure Adam, step once, parameter backing-array
    /// values differ from the pre-step snapshot.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_AdamFloat_StepUpdatesParameterInPlace()
    {
        var engine = new CpuEngine();
        // Construct CPU-resident tensors directly so the test exercises the
        // exact path that AiDotNet's NeuralNetworkBase layers hit: new
        // Tensor<T>(shape) goes through the CPU storage allocator. Going via
        // Tensor<T>.CreateRandom would route through AiDotNetEngine.Current,
        // which is OpenCL when the GPU backend is loaded — the live-backing
        // accessor correctly returns null for GPU-resident tensors and we'd
        // be testing the wrong path.
        var input = new Tensor<float>(new[] { 4, 3 });
        var weight = new Tensor<float>(new[] { 3, 2 });
        var rng = new System.Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < weight.Length; i++) weight[i] = (float)(rng.NextDouble() - 0.5);
        var preStep = weight.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var matmul = engine.TensorMatMul(input, weight);
            var sq = engine.TensorMultiply(matmul, matmul);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f);
            plan.Step();
        }

        var postStep = weight.GetDataArray();
        double maxAbs = 0;
        for (int i = 0; i < preStep.Length; i++)
            maxAbs = System.Math.Max(maxAbs, System.Math.Abs(postStep[i] - preStep[i]));
        Assert.True(maxAbs > 0,
            $"Adam(float) Step() did not update parameter in place. max |Δ| = {maxAbs}");
    }

    /// <summary>
    /// T=double regression for #350 / #341: after PR #341 gated float-only
    /// paths off T=double the dispatch went through
    /// <c>ConfigureOptimizerDouble</c>, which is structurally complete but
    /// must actually write back to the parameter tensor's backing array.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_AdamDouble_StepUpdatesParameterInPlace()
    {
        var engine = new CpuEngine();
        var input = new Tensor<double>(new[] { 4, 3 });
        var weight = new Tensor<double>(new[] { 3, 2 });
        var rng = new System.Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < weight.Length; i++) weight[i] = rng.NextDouble() - 0.5;
        var preStep = weight.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var matmul = engine.TensorMatMul(input, weight);
            var sq = engine.TensorMultiply(matmul, matmul);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f);
            plan.Step();
        }

        var postStep = weight.GetDataArray();
        double maxAbs = 0;
        for (int i = 0; i < preStep.Length; i++)
            maxAbs = System.Math.Max(maxAbs, System.Math.Abs(postStep[i] - preStep[i]));
        Assert.True(maxAbs > 0,
            $"Adam(double) Step() did not update parameter in place. max |Δ| = {maxAbs}");
    }

    /// <summary>
    /// T=double SGD also exercised: SGD has no Adam-style state, so a bug in
    /// the <c>ConfigureOptimizerDouble</c> param-array binding would
    /// manifest there too.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_SGDDouble_StepUpdatesParameterInPlace()
    {
        var engine = new CpuEngine();
        var input = new Tensor<double>(new[] { 4, 3 });
        var weight = new Tensor<double>(new[] { 3, 2 });
        var rng = new System.Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < weight.Length; i++) weight[i] = rng.NextDouble() - 0.5;
        var preStep = weight.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var matmul = engine.TensorMatMul(input, weight);
            var sq = engine.TensorMultiply(matmul, matmul);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.01f);
            plan.Step();
        }

        var postStep = weight.GetDataArray();
        double maxAbs = 0;
        for (int i = 0; i < preStep.Length; i++)
            maxAbs = System.Math.Max(maxAbs, System.Math.Abs(postStep[i] - preStep[i]));
        Assert.True(maxAbs > 0,
            $"SGD(double) Step() did not update parameter in place. max |Δ| = {maxAbs}");
    }
}
