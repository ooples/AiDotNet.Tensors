using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A compiled broadcasting binary whose right operand is one scalar per row of the left ([rows, cols] op [rows, 1],
/// e.g. a column-major layer's bias) replays as one vectorised pass per row straight into the plan's output buffer.
/// The replay used to fall into the in-place broadcast kernel's general fallback, which materialized the whole
/// broadcast result every step (N-BEATS: 214 MB per CPU step over its 180 bias adds).
/// </summary>
/// <remarks>
/// The forward value is read back exactly through a gradient: with loss = sum(Y * P), dL/dP is Y itself (a multiply
/// by the seed's ones), so the plan's gradient for P must equal the eager result bit for bit. The left operand is a
/// parameter as well, so Y is recomputed every step rather than folded as a constant.
/// </remarks>
[Collection("CompilationGlobalState")]
public sealed class CompiledPerRowScalarBroadcastTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;

    public CompiledPerRowScalarBroadcastTests()
    {
        AiDotNetEngine.Current = new CpuEngine();
    }

    public void Dispose()
    {
        AiDotNetEngine.Current = _priorEngine;
    }

    public enum BinaryOp
    {
        Add,
        Subtract,
        Multiply,
    }

    private static Tensor<T> Make<T>(int[] shape, int seed, Func<double, T> convert)
    {
        var rng = new Random(seed);
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new T[n];
        for (int i = 0; i < n; i++) data[i] = convert(rng.NextDouble() * 4 - 2);
        return new Tensor<T>(data, shape);
    }

    private static Tensor<T> Apply<T>(IEngine e, BinaryOp op, Tensor<T> left, Tensor<T> right) => op switch
    {
        BinaryOp.Add => e.TensorAdd(left, right),
        BinaryOp.Subtract => e.TensorSubtract(left, right),
        _ => e.TensorMultiply(left, right),
    };

    private static void CheckBitExact<T>(BinaryOp op, int[] leftShape, int[] rightShape, Func<double, T> convert)
    {
        var engine = new CpuEngine();
        var left = Make(leftShape, 1, convert);
        var right = Make(rightShape, 2, convert);
        var probe = Make(leftShape, 3, convert);
        var expected = Apply(engine, op, left, right).ToArray();

        ICompiledTrainingPlan<T> plan;
        using (var scope = GraphMode.EnableTraining(new[] { left, probe }))
        {
            var y = Apply(engine, op, left, right);
            var loss = engine.ReduceSum(engine.TensorMultiply(y, probe), null);
            plan = scope.CompileTraining(new[] { left, probe }, loss);
        }

        using (plan)
        {
            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                var got = plan.Gradients[1].ToArray();
                Assert.Equal(expected.Length, got.Length);
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(EqualityComparer<T>.Default.Equals(expected[i], got[i]),
                        $"{op} [{string.Join(",", leftShape)}] op [{string.Join(",", rightShape)}], step {step}, element {i}: "
                        + $"compiled {got[i]}, eager {expected[i]}");
            }
        }
    }

    [Theory]
    [InlineData(BinaryOp.Add)]
    [InlineData(BinaryOp.Subtract)]
    [InlineData(BinaryOp.Multiply)]
    public void ColumnBias_Float_BitIdenticalToEager(BinaryOp op)
    {
        CheckBitExact<float>(op, [37, 129], [37, 1], v => (float)v);
        // Rank 3 with two trailing size-1 axes, and rows that exceed one 64K-element chunk.
        CheckBitExact<float>(op, [3, 5, 7, 9], [3, 5, 1, 1], v => (float)v);
        CheckBitExact<float>(op, [600, 300], [600, 1], v => (float)v);
    }

    [Theory]
    [InlineData(BinaryOp.Add)]
    [InlineData(BinaryOp.Subtract)]
    [InlineData(BinaryOp.Multiply)]
    public void ColumnBias_Double_BitIdenticalToEager(BinaryOp op)
    {
        CheckBitExact<double>(op, [37, 129], [37, 1], v => v);
        CheckBitExact<double>(op, [3, 5, 7, 9], [3, 5, 1, 1], v => v);
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// The replay of [256, 1024] + [256, 1] must not allocate the broadcast result (1 MB) each step.
    /// </summary>
    [Fact]
    public void ColumnBias_StepDoesNotAllocateTheBroadcastResult()
    {
        const int rows = 256, cols = 1024;
        var engine = new CpuEngine();
        var left = Make([rows, cols], 1, v => (float)v);
        var right = Make([rows, 1], 2, v => (float)v);
        var probe = Make([rows, cols], 3, v => (float)v);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { left, probe }))
        {
            var y = engine.TensorAdd(left, right);
            var loss = engine.ReduceSum(engine.TensorMultiply(y, probe), null);
            plan = scope.CompileTraining(new[] { left, probe }, loss);
        }

        using (plan)
        {
            for (int warm = 0; warm < 3; warm++) plan.Step();
            const int steps = 5;
            long before = GC.GetTotalAllocatedBytes(true);
            for (int step = 0; step < steps; step++) plan.Step();
            long perStep = (GC.GetTotalAllocatedBytes(true) - before) / steps;
            Assert.True(perStep < (long)rows * cols * sizeof(float) / 4,
                $"a compiled step allocated {perStep} bytes; the [{rows}, {cols}] broadcast result is {rows * cols * sizeof(float)}.");
        }
    }
#endif
}