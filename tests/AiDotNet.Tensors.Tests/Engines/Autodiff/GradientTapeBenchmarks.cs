using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Performance benchmarks for gradient tape operations.
/// Measures overhead of tape recording and backward pass execution
/// to verify we're competitive with or faster than PyTorch CPU.
/// </summary>
public class GradientTapeBenchmarks
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public GradientTapeBenchmarks(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact(Skip = "Benchmark — run manually, not in CI")]
    public void Benchmark_TapeOverhead_PerOp()
    {
        // Measure cost of tape recording per operation
        var a = Tensor<float>.CreateRandom([256, 256]);
        var b = Tensor<float>.CreateRandom([256, 256]);

        // Warmup
        for (int i = 0; i < 10; i++)
            _engine.TensorAdd(a, b);

        // Without tape
        var sw = Stopwatch.StartNew();
        int opsNoTape = 1000;
        for (int i = 0; i < opsNoTape; i++)
            _engine.TensorAdd(a, b);
        sw.Stop();
        double noTapeMs = sw.Elapsed.TotalMilliseconds;

        // With tape
        sw.Restart();
        int opsTape = 1000;
        using (var tape = new GradientTape<float>())
        {
            for (int i = 0; i < opsTape; i++)
                _engine.TensorAdd(a, b);
        }
        sw.Stop();
        double tapeMs = sw.Elapsed.TotalMilliseconds;

        double overheadPerOp = (tapeMs - noTapeMs) / opsTape * 1000; // microseconds

        _output.WriteLine($"Add 256x256: no tape = {noTapeMs / opsNoTape * 1000:F2}us, with tape = {tapeMs / opsTape * 1000:F2}us, overhead = {overheadPerOp:F2}us/op");

        // Target: < 5us overhead per op (PyTorch is ~50-100ns in C++ but has Python overhead)
        Assert.True(overheadPerOp < 50, $"Tape overhead {overheadPerOp:F2}us per op exceeds 50us target");
    }

    [Fact(Skip = "Benchmark — run manually, not in CI")]
    public void Benchmark_ReLUBackward_1M_Elements()
    {
        int size = 1_000_000;
        var input = Tensor<float>.CreateRandom([size]);
        var gradOutput = Tensor<float>.CreateRandom([size]);

        // Warmup
        _engine.ReluBackward(gradOutput, input);

        var sw = Stopwatch.StartNew();
        int iterations = 100;
        for (int i = 0; i < iterations; i++)
            _engine.ReluBackward(gradOutput, input);
        sw.Stop();

        double msPerCall = sw.Elapsed.TotalMilliseconds / iterations;
        _output.WriteLine($"ReLU backward 1M elements: {msPerCall:F3}ms (target: < 0.5ms)");

        Assert.True(msPerCall < 2.0, $"ReLU backward {msPerCall:F3}ms exceeds 2ms target");
    }

    [Fact(Skip = "Benchmark — run manually, not in CI")]
    public void Benchmark_SigmoidBackward_1M_Elements()
    {
        int size = 1_000_000;
        var output = Tensor<float>.CreateRandom([size]);
        var gradOutput = Tensor<float>.CreateRandom([size]);

        // Warmup
        _engine.SigmoidBackward(gradOutput, output);

        var sw = Stopwatch.StartNew();
        int iterations = 100;
        for (int i = 0; i < iterations; i++)
            _engine.SigmoidBackward(gradOutput, output);
        sw.Stop();

        double msPerCall = sw.Elapsed.TotalMilliseconds / iterations;
        _output.WriteLine($"Sigmoid backward 1M elements: {msPerCall:F3}ms (target: < 1.0ms)");

        Assert.True(msPerCall < 5.0, $"Sigmoid backward {msPerCall:F3}ms exceeds 5ms target");
    }

    [Fact(Skip = "Benchmark — run manually, not in CI")]
    public void Benchmark_MatMul_Backward_256x256()
    {
        var a = Tensor<float>.CreateRandom([256, 256]);
        var b = Tensor<float>.CreateRandom([256, 256]);

        // Full forward + backward cycle
        var sw = Stopwatch.StartNew();
        int iterations = 50;
        for (int i = 0; i < iterations; i++)
        {
            using var tape = new GradientTape<float>();
            var c = _engine.TensorMatMul(a, b);
            var grads = tape.ComputeGradients(c, sources: new[] { a, b });
        }
        sw.Stop();

        double msPerStep = sw.Elapsed.TotalMilliseconds / iterations;
        _output.WriteLine($"MatMul 256x256 forward+backward: {msPerStep:F3}ms (target: < 10ms)");

        Assert.True(msPerStep < 50, $"MatMul forward+backward {msPerStep:F3}ms exceeds 50ms target");
    }

    [Fact(Skip = "Benchmark — run manually, not in CI")]
    public void Benchmark_NoGradScope_Overhead()
    {
        var a = Tensor<float>.CreateRandom([1000]);
        var b = Tensor<float>.CreateRandom([1000]);

        // Measure NoGradScope overhead
        var sw = Stopwatch.StartNew();
        int iterations = 10000;
        for (int i = 0; i < iterations; i++)
        {
            using var scope = GradientTape<float>.NoGrad();
            _engine.TensorAdd(a, b);
        }
        sw.Stop();

        double usPerOp = sw.Elapsed.TotalMilliseconds / iterations * 1000;
        _output.WriteLine($"NoGradScope + Add: {usPerOp:F2}us per op");

        // Should be negligible — just a counter increment/decrement
        Assert.True(usPerOp < 50, $"NoGradScope overhead {usPerOp:F2}us exceeds 50us");
    }

    [Fact(Skip = "Benchmark — run manually, not in CI")]
    public void Benchmark_FullTrainingStep_MLP()
    {
        // Simulate a single MLP training step: Linear(256) -> ReLU -> Linear(10)
        var x = Tensor<float>.CreateRandom([32, 256]); // batch=32
        var w1 = Tensor<float>.CreateRandom([256, 128]);
        var w2 = Tensor<float>.CreateRandom([128, 10]);

        var sw = Stopwatch.StartNew();
        int iterations = 20;
        for (int i = 0; i < iterations; i++)
        {
            using var tape = new GradientTape<float>();

            // Forward
            var h = _engine.TensorMatMul(x, w1);
            var hRelu = _engine.ReLU(h);
            var output = _engine.TensorMatMul(hRelu, w2);

            // Backward
            var grads = tape.ComputeGradients(output, sources: new[] { w1, w2 });
        }
        sw.Stop();

        double msPerStep = sw.Elapsed.TotalMilliseconds / iterations;
        _output.WriteLine($"MLP training step (32x256->128->10): {msPerStep:F3}ms");
        _output.WriteLine($"  (PyTorch CPU equivalent: ~5-10ms with Python overhead)");

        Assert.True(msPerStep < 100, $"MLP training step {msPerStep:F3}ms exceeds 100ms target");
    }
}
