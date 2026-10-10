using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.CpuJit;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The float JIT elementwise dispatchers must fan a large array out across threads and keep a small one serial.
/// They once returned through the serial path exactly when the array was big enough to split (an inverted
/// condition), which leaves every value correct -- only the dispatch shows it, so that is what this pins.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class ElementwiseDispatchParallelismTests
{
    // ReLUInto is the compiled plan's ReLU; it reaches the unary dispatcher at every size (eager ReLU splits large
    // arrays itself and only calls the dispatcher below its own threshold).
    public static TheoryData<string> Ops() => new() { "add", "subtract", "multiply", "divide", "reluInto" };

    private static Tensor<float> ReluInto(CpuEngine engine, Tensor<float> a)
    {
        var dst = new Tensor<float>(a._shape);
        engine.ReLUInto(dst, a);
        return dst;
    }

    [Theory]
    [MemberData(nameof(Ops))]
    public void LargeFloatElementwise_FansOut_SmallStaysSerial(string op)
    {
        if (!CpuJitSelfTest.IsVerified) return;   // the JIT dispatchers only run when the self-test verified them
        int previousDop = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = 8;
        try
        {
            var engine = new CpuEngine();
            foreach (var (n, expectParallel) in new[] { (4 * 1024 * 1024, true), (1000, false) })
            {
                var a = new Tensor<float>(new[] { n }); var b = new Tensor<float>(new[] { n });
                for (int i = 0; i < n; i++) { a[i] = (i % 97) - 48.5f; b[i] = (i % 13) + 1.5f; }
                CpuEngine.t_lastElementwiseDispatchChunks = -1;
                var r = op switch
                {
                    "add" => engine.TensorAdd(a, b),
                    "subtract" => engine.TensorSubtract(a, b),
                    "multiply" => engine.TensorMultiply(a, b),
                    "divide" => engine.TensorDivide(a, b),
                    _ => ReluInto(engine, a),
                };
                int chunks = CpuEngine.t_lastElementwiseDispatchChunks;
                if (expectParallel) Assert.True(chunks >= 2, $"{op} n={n}: dispatched to {chunks} chunk(s), expected a fan-out");
                else Assert.Equal(0, chunks);
                foreach (int i in new[] { 0, 1, n / 2, n - 1 })
                {
                    float x = a[i], y = b[i];
                    float expected = op switch { "add" => x + y, "subtract" => x - y, "multiply" => x * y, "divide" => x / y, _ => Math.Max(x, 0f) };
                    Assert.Equal(expected, r[i]);
                }
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = previousDop; }
    }
}
