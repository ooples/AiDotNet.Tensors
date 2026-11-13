using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The tape's Swish, Mish, HardSwish and Softplus backwards run the engine's one-kernel derivatives on the GPU. Each
/// must agree with the derivative computed in double precision to float rounding, over the range training sees and the
/// saturated tails.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class FusedActivationBackwardGpuTests
{
    private readonly ITestOutputHelper _output;
    public FusedActivationBackwardGpuTests(ITestOutputHelper output) => _output = output;

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    private static double Softplus(double x) => x > 30 ? x : Math.Log(1.0 + Math.Exp(x));

    private static double Derivative(string op, double x) => op switch
    {
        "swish" => Sigmoid(x) * (1 + x * (1 - Sigmoid(x))),
        "mish" => Math.Tanh(Softplus(x)) + x * (1 - Math.Pow(Math.Tanh(Softplus(x)), 2)) * Sigmoid(x),
        "hardswish" => x < -3 ? 0 : x > 3 ? 1 : (2 * x + 3) / 6,
        "softplus" => Sigmoid(x),
        _ => throw new ArgumentException(op),
    };

    [SkippableTheory]
    [InlineData("swish")]
    [InlineData("mish")]
    [InlineData("hardswish")]
    [InlineData("softplus")]
    public void FusedBackward_MatchesTheDoublePrecisionDerivative(string op)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend.");
        const int n = 1 << 16;
        var x = new float[n];
        var g = new float[n];
        var rng = new Random(7);
        for (int i = 0; i < n; i++)
        {
            x[i] = i < n / 2 ? (float)(rng.NextDouble() * 12 - 6) : (float)(rng.NextDouble() * 80 - 40);
            g[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        x[0] = -3f; x[1] = 3f; x[2] = 0f;   // HardSwish's kinks and zero
        var input = gpu.UploadToGpu(new Tensor<float>(x, new[] { n }), GpuTensorRole.General);
        var gradOut = gpu.UploadToGpu(new Tensor<float>(g, new[] { n }), GpuTensorRole.General);
        IEngine engine = gpu;
        var actual = (op switch
        {
            "swish" => engine.SwishBackward(gradOut, input),
            "mish" => engine.MishBackward(gradOut, input),
            "hardswish" => engine.HardswishBackward(gradOut, input),
            _ => engine.SoftplusBackward(gradOut, input),
        }).ToArray();

        double worst = 0;
        int worstAt = -1;
        for (int i = 0; i < n; i++)
        {
            if (op == "hardswish" && (x[i] == -3f || x[i] == 3f)) continue;   // one-sided at the kinks: either side is right
            double expected = g[i] * Derivative(op, x[i]);
            // Relative to the incoming gradient: the derivatives are O(1) and cross zero (Swish near -1.28, Mish near -1.19),
            // where an error relative to the product itself measures cancellation, not the kernel.
            double error = Math.Abs(actual[i] - expected) / Math.Max(1e-3, Math.Abs(g[i]));
            if (error > worst) { worst = error; worstAt = i; }
        }
        _output.WriteLine($"{op}: worst error relative to the gradient {worst:E2} at x = {(worstAt >= 0 ? x[worstAt] : 0)}");
        Assert.True(worst < 2e-6, $"{op}: relative error {worst:E2} at x = {x[worstAt]}: gpu {actual[worstAt]}, expected {g[worstAt] * Derivative(op, x[worstAt])}");
    }
}
