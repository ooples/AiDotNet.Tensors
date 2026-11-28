#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

// The specialized compiled backwards that call an allocating engine op and copy its result into the planned gradient
// must not leave a fresh activation-sized temporary behind every step: the next step's temporary of that size reuses
// the buffer, or the backward writes the gradient directly.
[Collection("CompilationGlobalState")]
public sealed class CompiledBackwardStepAllocationTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledBackwardStepAllocationTests(ITestOutputHelper output)
    {
        _output = output;
        AiDotNetEngine.Current = new CpuEngine();
    }

    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    private const int Rows = 128, Cols = 512;
    private const long ActivationBytes = (long)Rows * Cols * sizeof(float);

    public static TheoryData<string> Ops() => new()
    {
        "transpose", "sigmoid", "tanh", "softmax", "gelu", "swish", "elu", "relu", "divide", "multiplyShared",
        "maxpool", "avgpool",
    };

    // One planned op on an intermediate (x * w), so its backward is the specialized single-consumer one.
    private static Tensor<float> Forward(CpuEngine e, string op, Tensor<float> x, Tensor<float> w)
    {
        if (op == "transpose") return e.TensorMatMul(x, e.TensorTranspose(w));
        if (op == "divide") return e.TensorDivide(x, w);
        var h = e.TensorMultiply(x, w);
        return op switch
        {
            "sigmoid" => e.Sigmoid(h),
            "tanh" => e.Tanh(h),
            "softmax" => e.Softmax(h, -1),
            "gelu" => e.GELU(h),
            "swish" => e.Swish(h),
            "elu" => e.ELU(h, 1.0),
            "relu" => e.ReLU(h),
            "multiplyShared" => e.TensorMultiply(h, h),
            "maxpool" => e.MaxPool2D(e.Reshape(h, new[] { 8, 16, 32, 16 }), 2, 2),
            _ => e.AvgPool2D(e.Reshape(h, new[] { 8, 16, 32, 16 }), 2, 2),
        };
    }

    private static long BytesPerStep(string op, int steps)
    {
        var e = new CpuEngine();
        var rng = new Random(3);
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
            for (int i = 0; i < 4; i++) plan.Step();
            var first = new[] { plan.Gradients[0].ToArray(), plan.Gradients[1].ToArray() };
            long before = GC.GetTotalAllocatedBytes(true);
            for (int i = 0; i < steps; i++) plan.Step();
            long perStep = (GC.GetTotalAllocatedBytes(true) - before) / steps;
            // Learning rate 0: every step sees the same inputs, so a reused buffer that leaked into a gradient shows.
            for (int g = 0; g < 2; g++)
                Assert.Equal(first[g], plan.Gradients[g].ToArray());
            return perStep;
        }
    }

    [Theory]
    [MemberData(nameof(Ops))]
    public void CompiledBackwardStep_AllocatesNoActivationSizedTemporary(string op)
    {
        long perStep = BytesPerStep(op, 16);
        _output.WriteLine($"{op}: {perStep} B/step");
        // Below what any activation-sized temporary would add, with room for the step's small bookkeeping.
        Assert.True(perStep < ActivationBytes * 3 / 8,
            $"{op}: a compiled training step allocated {perStep} B; one activation is {ActivationBytes} B");
    }
}
#endif
