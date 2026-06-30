using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// #1745: the fused Adam/AdamW path can store its moment state (m, v) as
/// bfloat16 instead of fp32, halving the optimizer-state footprint — the
/// dominant training-step memory cost — while keeping the fp32 update math.
/// This lets large models stay on the fused fast path AND keep the reduced
/// optimizer-state memory, instead of being forced onto the eager autograd
/// tape (≈10× slower) when memory matters.
///
/// These tests build two identical plans over the same seeded weights and the
/// same deterministic graph, run N optimizer steps on each — one with fp32
/// moments, one with bf16 moments — and assert the bf16 run (a) actually
/// updates the parameter in place and (b) tracks the fp32 reference closely.
/// bfloat16 keeps the FULL float32 exponent (only the mantissa shortens 23→7
/// bits), so the trajectory deviation is bounded to a small fraction of the
/// fp32 weight movement; a broken kernel (wrong sign, no update, exponent loss)
/// blows past that bound.
/// </summary>
public class FusedBf16MomentAdamParityTests
{
    private const int Steps = 6;

    // Builds a fresh regression graph (FusedLinear → (out-target)² → sum) over a
    // single trainable weight, seeded identically every call so the fp32 and
    // bf16 runs start from the same point and see the same gradients each step.
    private static (ICompiledTrainingPlan<float> plan, Tensor<float> weight, float[] init) BuildPlan()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 4, 8 });
        var weight = new Tensor<float>(new[] { 8, 4 });
        var bias = new Tensor<float>(new[] { 4 });
        var target = new Tensor<float>(new[] { 4, 4 });
        var rng = new System.Random(12345);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < weight.Length; i++) weight[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < target.Length; i++) target[i] = (float)(rng.NextDouble() - 0.5);
        var init = weight.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.FusedLinear(input, weight, bias, FusedActivationType.None);
            var diff = engine.TensorSubtract(output, target);
            var sq = engine.TensorMultiply(diff, diff);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight });
        }
        return (plan, weight, init);
    }

    private static void AssertBf16TracksFp32(OptimizerType opt)
    {
        // fp32 reference run.
        var (planRef, wRef, init) = BuildPlan();
        using (planRef)
        {
            planRef.ConfigureOptimizer(opt, learningRate: 0.01f, weightDecay: opt == OptimizerType.AdamW ? 0.01f : 0f);
            for (int s = 0; s < Steps; s++) planRef.Step();
        }
        var fp32 = wRef.GetDataArray().AsSpan().ToArray();

        // bf16-moment run from the identical starting point.
        var (planBf16, wBf16, _) = BuildPlan();
        using (planBf16)
        {
            planBf16.RequestBf16MomentStorage(true);
            planBf16.ConfigureOptimizer(opt, learningRate: 0.01f, weightDecay: opt == OptimizerType.AdamW ? 0.01f : 0f);
            for (int s = 0; s < Steps; s++) planBf16.Step();
        }
        var bf16 = wBf16.GetDataArray().AsSpan().ToArray();

        double maxMove = 0, maxParityDiff = 0;
        for (int i = 0; i < init.Length; i++)
        {
            maxMove = System.Math.Max(maxMove, System.Math.Abs(fp32[i] - init[i]));
            maxParityDiff = System.Math.Max(maxParityDiff, System.Math.Abs(bf16[i] - fp32[i]));
        }

        // (a) the bf16 kernel actually moved the weight off its init.
        double maxBf16Move = 0;
        for (int i = 0; i < init.Length; i++)
            maxBf16Move = System.Math.Max(maxBf16Move, System.Math.Abs(bf16[i] - init[i]));
        Assert.True(maxBf16Move > 0, $"{opt}: bf16-moment Step() did not update the weight in place.");

        // (b) bf16 tracks fp32 to within a small fraction of the fp32 movement
        // (bf16's 7-bit mantissa → ~0.4%/step moment error) plus a tiny floor.
        double tolerance = 1e-4 + 0.05 * maxMove;
        Assert.True(maxParityDiff <= tolerance,
            $"{opt}: bf16-moment Adam diverged from fp32 reference. " +
            $"max|Δparity|={maxParityDiff:E4}, fp32 move={maxMove:E4}, tol={tolerance:E4}.");
    }

    [Fact]
    public void Adam_Bf16Moments_TrackFp32Reference()
        => AssertBf16TracksFp32(OptimizerType.Adam);

    [Fact]
    public void AdamW_Bf16Moments_TrackFp32Reference()
        => AssertBf16TracksFp32(OptimizerType.AdamW);

    /// <summary>
    /// bf16-moment Adam must be deterministic: two identical runs produce
    /// bit-identical weights (the rounding is round-to-nearest-even, no
    /// nondeterministic source).
    /// </summary>
    [Fact]
    public void Bf16Moments_AreDeterministic()
    {
        float[] Run()
        {
            var (plan, w, _) = BuildPlan();
            using (plan)
            {
                plan.RequestBf16MomentStorage(true);
                plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f);
                for (int s = 0; s < Steps; s++) plan.Step();
            }
            return w.GetDataArray().AsSpan().ToArray();
        }

        var a = Run();
        var b = Run();
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i]);
    }
}
