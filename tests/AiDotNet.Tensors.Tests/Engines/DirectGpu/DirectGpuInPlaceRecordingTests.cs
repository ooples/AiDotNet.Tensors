using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The in-place ops called through the CONCRETE <see cref="DirectGpuTensorEngine"/> type, which reaches its public
/// overrides rather than the explicit <c>IEngine</c> implementations: under a tape each must record its derivative,
/// d/dx sum(op_(2x)), exactly as the interface path does. Skipped where no GPU device is present.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class DirectGpuInPlaceRecordingTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private static readonly float[] Xs = { -1.5f, -0.25f, 0.5f, 2f };
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public DirectGpuInPlaceRecordingTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private DirectGpuTensorEngine Gpu()
    {
        Skip.IfNot(_fixture.IsAvailable && _fixture.Engine is not null, "No GPU device.");
        return _fixture.Engine ?? throw new InvalidOperationException("The fixture reported a device but no engine.");
    }

    private static float[] Gradient(DirectGpuTensorEngine gpu, Action<Tensor<float>> inPlace)
    {
        var x = new Tensor<float>((float[])Xs.Clone(), new[] { Xs.Length });
        using var tape = new GradientTape<float>();
        var y = gpu.TensorMultiplyScalar(x, 2f);
        inPlace(y);
        return tape.ComputeGradients(gpu.ReduceSum(y, null), new[] { x })[x].ToArray();
    }

    private static void AssertGradient(Func<float, float> expected, float[] actual)
    {
        for (int i = 0; i < Xs.Length; i++)
            Assert.True(Math.Abs(expected(Xs[i]) - actual[i]) <= 1e-4f, $"x = {Xs[i]}: expected {expected(Xs[i])}, got {actual[i]}");
    }

    [SkippableFact]
    public void AddInPlace_ThroughTheConcreteEngine_Records()
    {
        var gpu = Gpu();
        AssertGradient(_ => 4f, Gradient(gpu, y => gpu.TensorAddInPlace(y, y)));   // y = 4x
    }

    [SkippableFact]
    public void MultiplyInPlace_ThroughTheConcreteEngine_Records()
    {
        var gpu = Gpu();
        AssertGradient(x => 8f * x, Gradient(gpu, y => gpu.TensorMultiplyInPlace(y, y)));   // y = 4x^2
    }

    [SkippableFact]
    public void SubtractInPlace_ThroughTheConcreteEngine_Records()
    {
        var gpu = Gpu();
        var ones = new Tensor<float>(new[] { 1f, 1f, 1f, 1f }, new[] { 4 });
        AssertGradient(_ => 2f, Gradient(gpu, y => gpu.TensorSubtractInPlace(y, ones)));   // y = 2x - 1
    }

    [SkippableFact]
    public void SigmoidInPlace_ThroughTheConcreteEngine_Records()
    {
        var gpu = Gpu();
        static float S(float v) => 1f / (1f + (float)Math.Exp(-v));
        AssertGradient(x => 2f * S(2f * x) * (1f - S(2f * x)), Gradient(gpu, y => gpu.SigmoidInPlace(y)));
    }

    [SkippableFact]
    public void ReLUInPlace_ThroughTheConcreteEngine_Records()
    {
        var gpu = Gpu();
        AssertGradient(x => x > 0 ? 2f : 0f, Gradient(gpu, y => gpu.ReLUInPlace(y)));
    }
}
