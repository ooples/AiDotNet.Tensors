using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Profiles where time goes in a training step to identify the bottleneck
/// preventing us from matching PyTorch's ~1500 steps/sec for a 3-layer MLP.
/// </summary>
public class TrainingThroughputProfile
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public TrainingThroughputProfile(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Profile_MLP_StepBreakdown()
    {
        var input = Tensor<float>.CreateRandom([32, 128]);
        var w1 = Tensor<float>.CreateRandom([128, 64]);
        var w2 = Tensor<float>.CreateRandom([64, 32]);
        var w3 = Tensor<float>.CreateRandom([32, 10]);

        int warmup = 20;
        int iterations = 200;

        // Warmup
        for (int i = 0; i < warmup; i++)
        {
            using var tape = new GradientTape<float>();
            var h1 = _engine.ReLU(_engine.TensorMatMul(input, w1));
            var h2 = _engine.ReLU(_engine.TensorMatMul(h1, w2));
            var o = _engine.TensorMatMul(h2, w3);
            var l = _engine.ReduceSum(o, null);
            tape.ComputeGradients(l, new[] { w1, w2, w3 });
        }

        // === Measure FORWARD ONLY (no tape) ===
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            var h1 = _engine.ReLU(_engine.TensorMatMul(input, w1));
            var h2 = _engine.ReLU(_engine.TensorMatMul(h1, w2));
            var o = _engine.TensorMatMul(h2, w3);
            var l = _engine.ReduceSum(o, null);
        }
        sw.Stop();
        double forwardOnlyUs = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        // === Measure FORWARD (with tape recording) ===
        sw.Restart();
        Tensor<float> loss = null!;
        GradientTape<float>[] tapes = new GradientTape<float>[iterations];
        for (int i = 0; i < iterations; i++)
        {
            tapes[i] = new GradientTape<float>();
            var h1 = _engine.ReLU(_engine.TensorMatMul(input, w1));
            var h2 = _engine.ReLU(_engine.TensorMatMul(h1, w2));
            var o = _engine.TensorMatMul(h2, w3);
            loss = _engine.ReduceSum(o, null);
        }
        sw.Stop();
        double forwardTapeUs = sw.Elapsed.TotalMilliseconds * 1000 / iterations;
        // Dispose tapes
        foreach (var t in tapes) t.Dispose();

        // === Measure BACKWARD ONLY ===
        double backwardTotalUs = 0;
        for (int i = 0; i < iterations; i++)
        {
            using var tape = new GradientTape<float>();
            var h1 = _engine.ReLU(_engine.TensorMatMul(input, w1));
            var h2 = _engine.ReLU(_engine.TensorMatMul(h1, w2));
            var o = _engine.TensorMatMul(h2, w3);
            var l = _engine.ReduceSum(o, null);

            sw.Restart();
            var grads = tape.ComputeGradients(l, new[] { w1, w2, w3 });
            sw.Stop();
            backwardTotalUs += sw.Elapsed.TotalMilliseconds * 1000;
        }
        double backwardUs = backwardTotalUs / iterations;

        // === Measure FULL STEP ===
        sw.Restart();
        for (int i = 0; i < iterations; i++)
        {
            using var tape = new GradientTape<float>();
            var h1 = _engine.ReLU(_engine.TensorMatMul(input, w1));
            var h2 = _engine.ReLU(_engine.TensorMatMul(h1, w2));
            var o = _engine.TensorMatMul(h2, w3);
            var l = _engine.ReduceSum(o, null);
            tape.ComputeGradients(l, new[] { w1, w2, w3 });
        }
        sw.Stop();
        double fullStepUs = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        // === Measure individual ops ===
        sw.Restart();
        for (int i = 0; i < iterations; i++)
            _engine.TensorMatMul(input, w1); // 32x128 @ 128x64
        sw.Stop();
        double matmul1Us = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        var h1temp = Tensor<float>.CreateRandom([32, 64]);
        sw.Restart();
        for (int i = 0; i < iterations; i++)
            _engine.TensorMatMul(h1temp, w2); // 32x64 @ 64x32
        sw.Stop();
        double matmul2Us = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        var h2temp = Tensor<float>.CreateRandom([32, 32]);
        sw.Restart();
        for (int i = 0; i < iterations; i++)
            _engine.TensorMatMul(h2temp, w3); // 32x32 @ 32x10
        sw.Stop();
        double matmul3Us = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        sw.Restart();
        for (int i = 0; i < iterations; i++)
            _engine.ReLU(h1temp); // 32x64
        sw.Stop();
        double reluUs = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        double stepsPerSec = 1_000_000.0 / fullStepUs;

        _output.WriteLine("=== MLP [32x128->64->32->10] Training Step Breakdown ===");
        _output.WriteLine($"");
        _output.WriteLine($"Forward (no tape):     {forwardOnlyUs:F0}us");
        _output.WriteLine($"Forward (with tape):   {forwardTapeUs:F0}us");
        _output.WriteLine($"Tape overhead:         {forwardTapeUs - forwardOnlyUs:F0}us ({(forwardTapeUs - forwardOnlyUs) / forwardOnlyUs * 100:F1}%)");
        _output.WriteLine($"Backward:              {backwardUs:F0}us");
        _output.WriteLine($"Full step:             {fullStepUs:F0}us ({stepsPerSec:F0} steps/sec)");
        _output.WriteLine($"");
        _output.WriteLine($"--- Individual op costs (no tape) ---");
        _output.WriteLine($"MatMul 32x128 @ 128x64: {matmul1Us:F0}us");
        _output.WriteLine($"MatMul 32x64  @  64x32: {matmul2Us:F0}us");
        _output.WriteLine($"MatMul 32x32  @  32x10: {matmul3Us:F0}us");
        _output.WriteLine($"ReLU 32x64:             {reluUs:F0}us");
        _output.WriteLine($"Sum of 3 matmuls + relu: {matmul1Us + matmul2Us + matmul3Us + reluUs:F0}us");
        _output.WriteLine($"");
        _output.WriteLine($"--- Where is the gap? ---");
        _output.WriteLine($"PyTorch target:         ~667us/step (1500 steps/sec)");
        _output.WriteLine($"Our full step:          {fullStepUs:F0}us/step ({stepsPerSec:F0} steps/sec)");
        _output.WriteLine($"Gap factor:             {fullStepUs / 667:F1}x slower");
        _output.WriteLine($"");
        _output.WriteLine($"Forward accounts for:   {forwardOnlyUs / fullStepUs * 100:F0}% of step");
        _output.WriteLine($"Backward accounts for:  {backwardUs / fullStepUs * 100:F0}% of step");
        _output.WriteLine($"Overhead accounts for:  {(fullStepUs - forwardOnlyUs - backwardUs) / fullStepUs * 100:F0}% of step");
    }
}
