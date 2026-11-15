using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The tape's Swish, Mish, HardSwish and Softplus backwards run the engine's one-kernel derivatives on the GPU. Each
/// must agree with the derivative computed in double precision to float rounding, over the range training sees and the
/// saturated tails, and the kernel must really run on the device: the engine catches NotSupportedException and answers
/// on the CPU, which would pass the numerical comparison with no GPU coverage at all.
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
        // torch's hardswish_backward: the ramp includes both kinks, so -0.5 at x = -3 and 1.5 at x = 3. The tape's
        // create-graph form uses the same convention, and the first-order kernel must agree with it.
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
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7);
        for (int i = 0; i < n; i++)
        {
            x[i] = i < n / 2 ? (float)(rng.NextDouble() * 12 - 6) : (float)(rng.NextDouble() * 80 - 40);
            g[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        x[0] = -3f; x[1] = 3f; x[2] = 0f;   // HardSwish's kinks and zero
        var input = gpu.UploadToGpu(new Tensor<float>(x, new[] { n }), GpuTensorRole.General);
        var gradOut = gpu.UploadToGpu(new Tensor<float>(g, new[] { n }), GpuTensorRole.General);
        IEngine engine = gpu;
        GpuLaunchProbe.Reset();
        var actual = (op switch
        {
            "swish" => engine.SwishBackward(gradOut, input),
            "mish" => engine.MishBackward(gradOut, input),
            "hardswish" => engine.HardswishBackward(gradOut, input),
            _ => engine.SoftplusBackward(gradOut, input),
        }).ToArray();
        string kernel = op switch { "swish" => "SwishBackward", "mish" => "MishBackward", "hardswish" => "HardswishBackward", _ => "SoftplusBackward" };
        // Every one of these overrides records its CPU fallback here before answering on the host.
        Assert.DoesNotContain(GpuLaunchProbe.Fallbacks, f => f.Contains(kernel + ":"));

        double worst = 0;
        int worstAt = 0;
        for (int i = 0; i < n; i++)
        {
            // A NaN or infinity would compare false against worst below and slip past the tolerance.
            Assert.True(!float.IsNaN(actual[i]) && !float.IsInfinity(actual[i]),
                $"{op}: non-finite gradient {actual[i]} at x = {x[i]}");
            double expected = g[i] * Derivative(op, x[i]);
            // Relative to the incoming gradient: the derivatives are O(1) and cross zero (Swish near -1.28, Mish near -1.19),
            // where an error relative to the product itself measures cancellation, not the kernel.
            double error = Math.Abs(actual[i] - expected) / Math.Max(1e-3, Math.Abs(g[i]));
            if (error > worst) { worst = error; worstAt = i; }
        }
        _output.WriteLine($"{op}: worst error relative to the gradient {worst:E2} at x = {x[worstAt]}");
        Assert.True(worst < 2e-6, $"{op}: relative error {worst:E2} at x = {x[worstAt]}: gpu {actual[worstAt]}, expected {g[worstAt] * Derivative(op, x[worstAt])}");
    }

    /// <summary>
    /// At the kinks the one-kernel derivative must equal the tape's create-graph form and torch: -0.5 at x = -3 and 1.5
    /// at x = 3. The kernels returned 0 and 1 there, so a first-order backward and a create-graph backward of the same
    /// loss disagreed. Checked on the CPU engine and on the GPU engine.
    /// </summary>
    [SkippableFact]
    public void HardswishBackward_AtTheKinks_FollowsTheRamp_OnEveryEngine()
    {
        var x = new[] { -3f, 3f, -3.0001f, 3.0001f, 0f };
        var g = new[] { 2f, 2f, 2f, 2f, 2f };
        var expected = new[] { -1f, 3f, 0f, 2f, 1f };   // g * {-0.5, 1.5, 0, 1, 0.5}
        var cpu = new CpuEngine().HardswishBackward(new Tensor<float>(g, new[] { 5 }), new Tensor<float>(x, new[] { 5 })).ToArray();
        Assert.Equal(expected, cpu);

        var recorded = AiDotNet.Tensors.Engines.Autodiff.BackwardFunctions<float>.HardSwishDerivativeRecorded(
            new CpuEngine(), new Tensor<float>(x, new[] { 5 })).ToArray();
        for (int i = 0; i < x.Length; i++) Assert.Equal(expected[i], g[i] * recorded[i]);

        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend.");
        var input = gpu.UploadToGpu(new Tensor<float>(x, new[] { 5 }), GpuTensorRole.General);
        var gradOut = gpu.UploadToGpu(new Tensor<float>(g, new[] { 5 }), GpuTensorRole.General);
        GpuLaunchProbe.Reset();
        var device = ((IEngine)gpu).HardswishBackward(gradOut, input).ToArray();
        Assert.DoesNotContain(GpuLaunchProbe.Fallbacks, f => f.Contains("HardswishBackward:"));
        Assert.Equal(expected, device);
    }
}
