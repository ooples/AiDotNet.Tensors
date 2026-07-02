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

    private static (ICompiledTrainingPlan<float> plan, Tensor<float> a, Tensor<float> b, float[] initA, float[] initB) BuildGroupedPlan()
    {
        var engine = new CpuEngine();
        var a = new Tensor<float>(new[] { 16 });
        var b = new Tensor<float>(new[] { 16 });
        var rng = new System.Random(6789);
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);
        var initA = a.GetDataArray().AsSpan().ToArray();
        var initB = b.GetDataArray().AsSpan().ToArray();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var l1 = engine.ReduceSum(a, null);
            var l2 = engine.ReduceSum(b, null);
            engine.TensorAdd(l1, l2);
            plan = scope.CompileTraining(new[] { a, b });
        }
        return (plan, a, b, initA, initB);
    }

    private static void AssertTracksReference(
        string name,
        float[] reference,
        float[] actual,
        float[] initial,
        double moveFractionTolerance,
        double floorTolerance)
    {
        double maxMove = 0, maxParityDiff = 0, maxActualMove = 0;
        for (int i = 0; i < initial.Length; i++)
        {
            maxMove = System.Math.Max(maxMove, System.Math.Abs(reference[i] - initial[i]));
            maxParityDiff = System.Math.Max(maxParityDiff, System.Math.Abs(actual[i] - reference[i]));
            maxActualMove = System.Math.Max(maxActualMove, System.Math.Abs(actual[i] - initial[i]));
        }

        Assert.True(maxActualMove > 0, $"{name}: reduced-moment Step() did not update the weight in place.");
        double tolerance = floorTolerance + moveFractionTolerance * maxMove;
        Assert.True(maxParityDiff <= tolerance,
            $"{name}: reduced-moment Adam diverged from fp32 reference. " +
            $"max|Δparity|={maxParityDiff:E4}, fp32 move={maxMove:E4}, tol={tolerance:E4}.");
    }

    private static void AssertClose(string name, float[] expected, float[] actual, double tolerance)
    {
        for (int i = 0; i < expected.Length; i++)
            Assert.True(System.Math.Abs(actual[i] - expected[i]) <= tolerance,
                $"{name}[{i}]: expected={expected[i]:R}, actual={actual[i]:R}, tol={tolerance:E4}");
    }

    private static unsafe float[] RunDirectInt8ConstantGradient(float[] initial, float lr)
    {
        const int blockSize = 8;
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float eps = 1e-8f;
        var param = (float[])initial.Clone();
        var mQuant = new byte[param.Length];
        var vQuant = new byte[param.Length];
        int blockCount = (param.Length + blockSize - 1) / blockSize;
        var mScales = new double[blockCount];
        var vScales = new double[blockCount];
        for (int i = 0; i < mQuant.Length; i++) mQuant[i] = 128;
        for (int i = 0; i < blockCount; i++) { mScales[i] = 1e-10; vScales[i] = 1e-10; }

        for (int step = 1; step <= Steps; step++)
        {
            var grad = new float[param.Length];
            for (int i = 0; i < param.Length; i++) grad[i] = 1f;
            fixed (float* pParam = param, pGrad = grad)
            fixed (byte* pM = mQuant, pV = vQuant)
            fixed (double* pMs = mScales, pVs = vScales)
            {
                FusedOptimizer.AdamUpdateInt8BlockQuantized(
                    pParam, pGrad, pM, pV, pMs, pVs, param.Length, blockSize,
                    lr, beta1, beta2, eps, step);
            }
        }
        return param;
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

        // bf16 tracks fp32 to within a small fraction of the fp32 movement
        // (bf16's 7-bit mantissa -> ~0.4%/step moment error) plus a tiny floor.
        AssertTracksReference($"{opt} bf16", fp32, bf16, init, moveFractionTolerance: 0.05, floorTolerance: 1e-4);
    }

    [Fact]
    public void Adam_Bf16Moments_TrackFp32Reference()
        => AssertBf16TracksFp32(OptimizerType.Adam);

    [Fact]
    public void AdamW_Bf16Moments_TrackFp32Reference()
        => AssertBf16TracksFp32(OptimizerType.AdamW);

    [Fact]
    public void Adam_Bf16Moments_GroupedSchedulesTrackFp32Reference()
    {
        var (planRef, aRef, bRef, initA, initB) = BuildGroupedPlan();
        using (planRef)
        {
            planRef.ConfigureOptimizerGrouped(
                OptimizerType.Adam,
                new LrSchedule[] { LrSchedule.Constant(0.01), LrSchedule.Constant(0.003) },
                new int[] { 0, 1 });
            for (int s = 0; s < Steps; s++) planRef.Step();
        }

        var (planBf16, aBf16, bBf16, _, _) = BuildGroupedPlan();
        using (planBf16)
        {
            planBf16.RequestBf16MomentStorage(true);
            planBf16.ConfigureOptimizerGrouped(
                OptimizerType.Adam,
                new LrSchedule[] { LrSchedule.Constant(0.01), LrSchedule.Constant(0.003) },
                new int[] { 0, 1 });
            for (int s = 0; s < Steps; s++) planBf16.Step();
        }

        AssertTracksReference("group0 bf16", aRef.GetDataArray().AsSpan().ToArray(), aBf16.GetDataArray().AsSpan().ToArray(), initA, 0.05, 1e-4);
        AssertTracksReference("group1 bf16", bRef.GetDataArray().AsSpan().ToArray(), bBf16.GetDataArray().AsSpan().ToArray(), initB, 0.05, 1e-4);
    }

    [Fact]
    public void Adam_Int8BlockQuantizedMoments_TrackFp32Reference()
    {
        var (planRef, wRef, init) = BuildPlan();
        using (planRef)
        {
            planRef.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f);
            for (int s = 0; s < Steps; s++) planRef.Step();
        }

        var (planInt8, wInt8, _) = BuildPlan();
        using (planInt8)
        {
            planInt8.RequestInt8MomentStorage(true, blockSize: 8);
            planInt8.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f);
            for (int s = 0; s < Steps; s++) planInt8.Step();
        }

        AssertTracksReference(
            "Adam int8 block-quantized",
            wRef.GetDataArray().AsSpan().ToArray(),
            wInt8.GetDataArray().AsSpan().ToArray(),
            init,
            moveFractionTolerance: 0.20,
            floorTolerance: 2e-3);
    }

    [Fact]
    public void Adam_Int8BlockQuantizedMoments_GroupedSchedulesMatchDirectKernelReference()
    {
        var setup = BuildGroupedPlan();
        setup.plan.Dispose();
        var expectedA = RunDirectInt8ConstantGradient(setup.initA, lr: 0.01f);
        var expectedB = RunDirectInt8ConstantGradient(setup.initB, lr: 0.003f);

        var (planInt8, aInt8, bInt8, _, _) = BuildGroupedPlan();
        using (planInt8)
        {
            planInt8.RequestInt8MomentStorage(true, blockSize: 8);
            planInt8.ConfigureOptimizerGrouped(
                OptimizerType.Adam,
                new LrSchedule[] { LrSchedule.Constant(0.01), LrSchedule.Constant(0.003) },
                new int[] { 0, 1 });
            for (int s = 0; s < Steps; s++) planInt8.Step();
        }

        AssertClose("group0 int8", expectedA, aInt8.GetDataArray().AsSpan().ToArray(), tolerance: 1e-5);
        AssertClose("group1 int8", expectedB, bInt8.GetDataArray().AsSpan().ToArray(), tolerance: 1e-5);
    }

    [Fact]
    public void Int8BlockQuantizedMoments_RejectAdamW()
    {
        var (plan, _, _) = BuildPlan();
        using (plan)
        {
            plan.RequestInt8MomentStorage(true, blockSize: 8);
            Assert.Throws<System.NotSupportedException>(() =>
                plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0.01f));
        }
    }

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
