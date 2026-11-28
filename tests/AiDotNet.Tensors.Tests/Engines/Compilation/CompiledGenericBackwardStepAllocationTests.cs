#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

// Clamp, Log, Sqrt, Abs and Exp have no compiled specialization: their backward is the generic one, which used to build
// the gradient from allocating engine ops and leave each activation-sized temporary behind every step. A transposed
// matmul's backward is specialized into the plan's gradient buffers. Neither may allocate an activation per step.
[Collection("CompilationGlobalState")]
public sealed class CompiledGenericBackwardStepAllocationTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledGenericBackwardStepAllocationTests(ITestOutputHelper output)
    {
        _output = output;
        AiDotNetEngine.Current = new CpuEngine();
    }

    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    private const int Rows = 128, Cols = 512;
    private const long ActivationBytes = (long)Rows * Cols * sizeof(float);

    public static TheoryData<string> Ops() => new() { "clamp", "log", "exp", "sqrt", "abs", "matMulTransposed" };

    // The op runs on an intermediate (x * w) so its input's gradient is a planned buffer a parameter feeds.
    private static Tensor<float> Forward(CpuEngine e, string op, Tensor<float> x, Tensor<float> w)
    {
        if (op == "matMulTransposed") return e.TensorMatMulTransposed(x, w);
        var h = e.TensorMultiply(x, w);
        return op switch
        {
            "clamp" => e.TensorClamp(h, -0.5f, 0.5f),
            "log" => e.TensorLog(e.TensorAddScalar(e.TensorMultiply(h, h), 1f)),
            "exp" => e.TensorExp(h),
            "sqrt" => e.TensorSqrt(e.TensorAddScalar(e.TensorMultiply(h, h), 1f)),
            _ => e.TensorAbs(h),
        };
    }

    private static long BytesPerStep(string op, int steps)
    {
        var e = new CpuEngine();
        var rng = new Random(5);
        var x = new Tensor<float>(new[] { Rows, Cols });
        var w = new Tensor<float>(new[] { Rows, Cols });
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() + .5);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { x, w }))
        {
            var y = Forward(e, op, x, w);
            var loss = e.ReduceSum(e.TensorMultiply(y, y), null);
            plan = scope.CompileTraining(new[] { x, w }, loss);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
            for (int i = 0; i < 64; i++) plan.Step();
            var first = new[] { plan.Gradients[0].ToArray(), plan.Gradients[1].ToArray() };
            long before = GC.GetTotalAllocatedBytes(true);
            for (int i = 0; i < steps; i++) plan.Step();
            long perStep = (GC.GetTotalAllocatedBytes(true) - before) / steps;
            // Learning rate 0: every step sees the same inputs, so a handed-back buffer that leaked into a gradient
            // would change one.
            for (int g = 0; g < 2; g++)
                Assert.Equal(first[g], plan.Gradients[g].ToArray());
            return perStep;
        }
    }

    [Theory]
    [MemberData(nameof(Ops))]
    public void CompiledGenericBackwardStep_AllocatesNoActivationSizedTemporary(string op)
    {
        // A long steady-state window, so a collector-paced recycle in a short one can neither pass nor fail it.
        long perStep = BytesPerStep(op, 2000);
        _output.WriteLine($"{op}: {perStep} B/step");
        Assert.True(perStep < ActivationBytes * 3 / 8,
            $"{op}: a compiled training step allocated {perStep} B; one activation is {ActivationBytes} B");
    }

    // The transposed matmul's specialized backward against the analytic gradients of sum((x·wᵀ)²): dx = 2y·w and
    // dw = 2yᵀ·x, within GEMM rounding.
    [Fact]
    public void CompiledMatMulTransposedBackward_MatchesAnalyticGradients()
    {
        var e = new CpuEngine();
        var rng = new Random(9);
        var x = new Tensor<float>(new[] { 64, 96 });
        var w = new Tensor<float>(new[] { 48, 96 });
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { x, w }))
        {
            var y = e.TensorMatMulTransposed(x, w);
            plan = scope.CompileTraining(new[] { x, w }, e.ReduceSum(e.TensorMultiply(y, y), null));
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
            plan.Step();
            var y2 = e.TensorMultiplyScalar(e.TensorMatMul(x, e.TensorTranspose(w)), 2f);
            var dx = e.TensorMatMul(y2, w).ToArray();
            var dw = e.TensorMatMul(e.TensorTranspose(y2), x).ToArray();
            AssertClose(dx, plan.Gradients[0].ToArray());
            AssertClose(dw, plan.Gradients[1].ToArray());
        }
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * (1f + Math.Abs(expected[i])),
                $"element {i}: expected {expected[i]}, got {actual[i]}");
    }
}
#endif
