using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Forward results declared with <see cref="GradientRecording.RetainOutput{T}"/> stay readable after every step of a
/// mixed-precision compiled plan, as they do for the FP32 plan: a result on the loss's path (which the plan's free
/// schedule would otherwise release after its last read) and one off it (an observation the loss never reads, which
/// the plan would otherwise not compute at all) each hold the value of the latest step. Before, the plan refused any
/// retained output.
/// </summary>
[Collection(MixedPrecisionTestCollection.Name)]
public sealed class MixedPrecisionRetainedOutputTests
{
    private const int Rows = 16, In = 32, Out = 8;
    private readonly ITestOutputHelper _output;

    public MixedPrecisionRetainedOutputTests(ITestOutputHelper output) => _output = output;

    private static Tensor<float> Rand(int r, int c, int seed, double scale)
    {
        var rng = new Random(seed);
        var d = new float[r * c];
        for (int i = 0; i < d.Length; i++) d[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return new Tensor<float>(d, new[] { r, c });
    }

    private void Run(IEngine engine, string label)
    {
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var x = Rand(Rows, In, 1, 1.0);
            var w = Rand(In, Out, 2, 0.3);
            Tensor<float>? onPath = null, offPath = null;
            using var plan = MixedPrecisionCompiledPlan.Trace(() =>
            {
                var y = engine.TensorMatMul(x, w);
                var act = engine.Tanh(y);
                onPath = engine.TensorMultiply(act, act);              // read by the loss, then dead in the forward
                offPath = engine.ReduceSum(act, new[] { 1 }, false);    // an observation the loss never reads
                GradientRecording.RetainOutput(onPath);
                GradientRecording.RetainOutput(offPath);
                return engine.ReduceSum(onPath, null);
            }, engine);

            for (int step = 0; step < 4; step++)
            {
                // Host reference from the weights this step's forward reads (FP16-rounded matmul operands).
                var wNow = w.ToArray();
                var xs = x.ToArray();
                var expectedOn = new double[Rows * Out];
                var expectedOff = new double[Rows];
                for (int r = 0; r < Rows; r++)
                    for (int o = 0; o < Out; o++)
                    {
                        double acc = 0;
                        for (int i = 0; i < In; i++) acc += (double)(float)(Half)xs[r * In + i] * (float)(Half)wNow[i * Out + o];
                        double a = Math.Tanh(acc);
                        expectedOn[r * Out + o] = a * a;
                        expectedOff[r] += a;
                    }

                var result = plan.Step(new[] { w }, learningRate: 0.05f);
                var on = onPath!.ToArray();
                var off = offPath!.ToArray();
                double onDiff = on.Select((v, i) => Math.Abs(v - expectedOn[i])).Max();
                double offDiff = off.Select((v, i) => Math.Abs(v - expectedOff[i])).Max();
                _output.WriteLine($"{label} step {step}: loss {result.Loss:R}, max |retained on-path - host| {onDiff:E3}, "
                    + $"off-path {offDiff:E3}");
                Assert.True(onDiff < 5e-3, $"{label} step {step}: the on-path retained output is stale or freed ({onDiff})");
                Assert.True(offDiff < 2e-2, $"{label} step {step}: the off-path retained output is stale or missing ({offDiff})");
                double expectedLoss = expectedOn.Sum();
                Assert.True(Math.Abs(result.Loss - expectedLoss) <= 1e-2 * (1 + Math.Abs(expectedLoss)),
                    $"{label} step {step}: loss {result.Loss}, host {expectedLoss}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    [Fact]
    public void RetainedOutputs_HoldEachStepsValue_OnTheCpu() => Run(new CpuEngine(), "cpu");

    [SkippableFact]
    public void RetainedOutputs_HoldEachStepsValue_OnTheGpu()
    {
        using var probe = new CudaBackend();
        Skip.IfNot(probe.IsAvailable, "No CUDA device available.");
        using var engine = new DirectGpuTensorEngine();
        Run(engine, "gpu");
    }
}
