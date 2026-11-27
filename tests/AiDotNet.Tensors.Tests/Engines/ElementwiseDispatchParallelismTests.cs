using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
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
    // arrays itself and only calls the dispatcher below its own threshold). AddInto is the zero-allocation add, which
    // ran single-threaded wherever the CPU JIT was unavailable (Linux, macOS, .NET Framework).
    public static TheoryData<string> Ops() => new() { "add", "addInto", "subtract", "multiply", "divide", "reluInto" };

    private static Tensor<float> AddInto(CpuEngine engine, Tensor<float> a, Tensor<float> b)
    {
        var dst = new Tensor<float>(a._shape);
        engine.TensorAddInto(dst, a, b);
        return dst;
    }

    private static Tensor<float> ReluInto(CpuEngine engine, Tensor<float> a)
    {
        var dst = new Tensor<float>(a._shape);
        engine.ReLUInto(dst, a);
        return dst;
    }

    // No JIT precondition: the dispatchers run SimdKernels, so every platform must fan out the same way.
    [Theory]
    [MemberData(nameof(Ops))]
    public void LargeFloatElementwise_FansOut_SmallStaysSerial(string op)
    {
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
                    "addInto" => AddInto(engine, a, b),
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
                    float expected = op switch { "add" or "addInto" => x + y, "subtract" => x - y, "multiply" => x * y, "divide" => x / y, _ => Math.Max(x, 0f) };
                    Assert.Equal(expected, r[i]);
                }
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = previousDop; }
    }

    // The float *Into ops that ran on one thread whatever the size (they called the SIMD kernel directly), now
    // split like TensorAddInto. Each must fan out above the grain, stay serial below it, and give exactly the bits
    // of a single-threaded run -- odd lengths, so the last chunk has a vector tail.
    public static TheoryData<string> IntoOps() => new()
    {
        "subtractInto", "multiplyInto", "divideInto", "addScalarInto", "broadcastAddInto", "broadcastMultiplyInto",
        "sinInto", "cosInto", "leakyReluInto", "eluInto",
    };

    private static Tensor<float> RunInto(CpuEngine engine, string op, Tensor<float> a, Tensor<float> b)
    {
        var dst = new Tensor<float>(a._shape);
        switch (op)
        {
            case "subtractInto": engine.TensorSubtractInto(dst, a, b); break;
            case "multiplyInto": engine.TensorMultiplyInto(dst, a, b); break;
            case "divideInto": engine.TensorDivideInto(dst, a, b); break;
            case "addScalarInto": engine.TensorAddScalarInto(dst, a, 0.375f); break;
            case "broadcastAddInto": engine.TensorBroadcastAddInto(dst, a, b); break;
            case "broadcastMultiplyInto": engine.TensorBroadcastMultiplyInto(dst, a, b); break;
            case "sinInto": engine.TensorSinInto(dst, a); break;
            case "cosInto": engine.TensorCosInto(dst, a); break;
            case "leakyReluInto": engine.LeakyReLUInto(dst, a, 0.01f); break;
            default: engine.ELUInto(dst, a, 1.0); break;
        }
        return dst;
    }

    [Theory]
    [MemberData(nameof(IntoOps))]
    public void LargeFloatIntoOps_FanOut_AndMatchASingleThreadedRunBitForBit(string op)
    {
        int previousDop = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            var engine = new CpuEngine();
            foreach (var (n, expectParallel) in new[] { (4 * 1024 * 1024 + 7, true), (1003, false) })
            {
                var a = new Tensor<float>(new[] { n }); var b = new Tensor<float>(new[] { n });
                for (int i = 0; i < n; i++) { a[i] = ((i * 37) % 997) / 50f - 9.9f; b[i] = (i % 13) + 1.5f; }
                a[1] = -0f; a[2] = float.NaN;

                CpuParallelSettings.MaxDegreeOfParallelism = 1;
                var serial = RunInto(engine, op, a, b);

                CpuParallelSettings.MaxDegreeOfParallelism = 8;
                CpuEngine.t_lastElementwiseDispatchChunks = -1;
                var split = RunInto(engine, op, a, b);
                int chunks = CpuEngine.t_lastElementwiseDispatchChunks;
                if (expectParallel) Assert.True(chunks >= 2, $"{op} n={n}: dispatched to {chunks} chunk(s), expected a fan-out");
                else Assert.Equal(0, chunks);

                for (int i = 0; i < n; i++)
                {
                    if (MathCompat.SingleToInt32Bits(split[i]) != MathCompat.SingleToInt32Bits(serial[i]))
                        Assert.Fail($"{op} n={n} element {i}: split {split[i]:R} vs single-threaded {serial[i]:R}");
                }
                // The exactly-rounded ops also match their scalar definition.
                foreach (int i in new[] { 0, 1, 2, 3, n / 2, n - 1 })
                {
                    float x = a[i], y = b[i];
                    float? expected = op switch
                    {
                        "subtractInto" => x - y,
                        "multiplyInto" or "broadcastMultiplyInto" => x * y,
                        "divideInto" => x / y,
                        "addScalarInto" => x + 0.375f,
                        "broadcastAddInto" => x + y,
                        "sinInto" => MathF.Sin(x),
                        "cosInto" => MathF.Cos(x),
                        _ => null,
                    };
                    if (expected is float e)
                        Assert.True(MathCompat.SingleToInt32Bits(e) == MathCompat.SingleToInt32Bits(split[i]),
                            $"{op} element {i}: expected {e:R}, got {split[i]:R}");
                }
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = previousDop; }
    }
}
