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
    /// Issue #350 second-order bug: the AiDotNet consumer's DenseLayer routes
    /// through <c>Engine.FusedLinear(input, weights, bias, FusedActivationType.None)</c>
    /// — a single fused op. After compile + Step + Adam update, the weights
    /// parameter must have changed. Pre-fix, the fused-op's compiled backward
    /// accumulated gradient into a tensor reference that AccumulateGrad
    /// replaced into the gradMap dictionary, while CompiledTrainingPlan
    /// captured the PRE-Step gradient tensor at compile time, leaving the
    /// optimizer reading a stale zero buffer.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_AdamFloat_FusedLinearStepUpdatesParameterInPlace()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 4, 8 });
        var weight = new Tensor<float>(new[] { 8, 4 });
        var bias = new Tensor<float>(new[] { 4 });
        var rng = new System.Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < weight.Length; i++) weight[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() - 0.5);
        var preStep = weight.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            // EXACTLY the path AiDotNet's DenseLayer.Forward takes in training
            // mode: a single fused engine op, not a TensorMatMul+TensorAdd
            // primitives decomposition.
            var output = engine.FusedLinear(input, weight, bias, FusedActivationType.None);
            var sq = engine.TensorMultiply(output, output);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight, bias });
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
            $"FusedLinear path: Adam(float) Step() did not update weight in place. max |Δ| = {maxAbs}");
    }

    /// <summary>
    /// Reproduces the AiDotNet caller pattern exactly: a warmup forward call
    /// OUTSIDE GraphMode (mirrors <c>CompiledTapeTrainingStep.TryStepWithFusedOptimizer</c>
    /// line 248–249 which calls <c>forward(input)</c> before the compile to
    /// trigger lazy layer initialization), then compile with the same forward
    /// closure inside GraphMode. The bug is that AccumulateGrad replaces the
    /// pre-allocated gradient buffer for the parameter on its first write,
    /// while CompiledTrainingPlan.Compile captures the pre-allocated
    /// reference into <c>_gradients[i]</c> — so ConfigureOptimizer reads the
    /// stale all-zero buffer and the optimizer never sees a non-zero gradient.
    /// This test reproduces the AiDotNet min-repro symptom inside Tensors so
    /// it can be fixed at the source.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_FusedLinear_WithWarmupBeforeCompile_StepUpdatesWeight()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 4, 8 });
        var weight = new Tensor<float>(new[] { 8, 4 });
        var bias = new Tensor<float>(new[] { 4 });
        var target = new Tensor<float>(new[] { 4, 4 });
        var rng = new System.Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < weight.Length; i++) weight[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < target.Length; i++) target[i] = (float)(rng.NextDouble() - 0.5);
        var preStep = weight.GetDataArray().AsSpan().ToArray();

        // Warmup OUTSIDE GraphMode — mirrors AiDotNet's CompiledTapeTrainingStep
        // line 248-249 forward(input) call before plan compile. This is the
        // bit that distinguishes the bug-reproducing path from the direct
        // FusedLinear test above.
        var warmupOut = engine.FusedLinear(input, weight, bias, FusedActivationType.None);
        _ = engine.TensorSubtract(warmupOut, target);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.FusedLinear(input, weight, bias, FusedActivationType.None);
            var diff = engine.TensorSubtract(output, target);
            var sq = engine.TensorMultiply(diff, diff);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight, bias });
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
            $"FusedLinear with warmup-before-compile: weight did not change. max |Δ| = {maxAbs}");
    }

    /// <summary>
    /// Exercises the EXACT public API the AiDotNet consumer hits:
    /// <c>CompiledModelCache&lt;T&gt;.GetOrCompileTraining(compositeKey, forwardAndLoss, parameters)</c>
    /// followed by <c>plan.ConfigureOptimizer + plan.Step</c>. If the bug is
    /// in this layer (not in <c>LazyTensorScope.CompileTraining</c> which the
    /// earlier tests use directly), this is where it surfaces.
    /// </summary>
    [Fact]
    public void CompiledModelCache_GetOrCompileTraining_FusedLinear_StepUpdatesWeight()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 4, 8 });
        var weight = new Tensor<float>(new[] { 8, 4 });
        var bias = new Tensor<float>(new[] { 4 });
        var target = new Tensor<float>(new[] { 4, 4 });
        var rng = new System.Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < weight.Length; i++) weight[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < target.Length; i++) target[i] = (float)(rng.NextDouble() - 0.5);
        var preStep = weight.GetDataArray().AsSpan().ToArray();

        // Mirror AiDotNet's CompiledTapeTrainingStep flow exactly:
        var cache = new CompiledModelCache<float>();
        // Warmup forward outside any GraphMode scope, like the consumer does.
        _ = engine.FusedLinear(input, weight, bias, FusedActivationType.None);

        var parameters = new[] { weight, bias };

        var plan = cache.GetOrCompileTraining(
            input._shape,
            () =>
            {
                var output = engine.FusedLinear(input, weight, bias, FusedActivationType.None);
                var diff = engine.TensorSubtract(output, target);
                var sq = engine.TensorMultiply(diff, diff);
                return engine.ReduceSum(sq, null);
            },
            parameters);

        plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f);
        plan.Step();

        var postStep = weight.GetDataArray();
        double maxAbs = 0;
        for (int i = 0; i < preStep.Length; i++)
            maxAbs = System.Math.Max(maxAbs, System.Math.Abs(postStep[i] - preStep[i]));
        Assert.True(maxAbs > 0,
            $"CompiledModelCache + FusedLinear: weight did not change. max |Δ| = {maxAbs}");
    }

    /// <summary>
    /// T=double FusedLinear regression — same as float case but exercises the
    /// double codegen path in CompiledTrainingPlan.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_AdamDouble_FusedLinearStepUpdatesParameterInPlace()
    {
        var engine = new CpuEngine();
        var input = new Tensor<double>(new[] { 4, 8 });
        var weight = new Tensor<double>(new[] { 8, 4 });
        var bias = new Tensor<double>(new[] { 4 });
        var rng = new System.Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < weight.Length; i++) weight[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < bias.Length; i++) bias[i] = rng.NextDouble() - 0.5;
        var preStep = weight.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.FusedLinear(input, weight, bias, FusedActivationType.None);
            var sq = engine.TensorMultiply(output, output);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight, bias });
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
            $"FusedLinear path: Adam(double) Step() did not update weight in place. max |Δ| = {maxAbs}");
    }

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
