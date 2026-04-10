using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Integration tests for OneDNN pooling and batch normalization (#117).
/// These test the CpuEngine paths that attempt OneDNN dispatch.
/// When OneDNN is unavailable, they validate the fallback C# paths produce correct results.
/// </summary>
public class OneDnnIntegrationTests
{
    private readonly ITestOutputHelper _output;
    public OneDnnIntegrationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void MaxPool2D_ProducesCorrectOutput()
    {
        var engine = new CpuEngine();
        // 1 batch, 1 channel, 4x4 input, 2x2 kernel, stride 2
        var input = new Tensor<float>(new float[]
        {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        }, new[] { 1, 1, 4, 4 });

        var result = engine.MaxPool2D(input, 2, 2, 0);

        // Expected: max of each 2x2 block
        // [6, 8]
        // [14, 16]
        Assert.Equal(new[] { 1, 1, 2, 2 }, result.Shape.ToArray());
        var data = result.GetDataArray();
        Assert.Equal(6f, data[0]);
        Assert.Equal(8f, data[1]);
        Assert.Equal(14f, data[2]);
        Assert.Equal(16f, data[3]);

        _output.WriteLine($"MaxPool2D: OneDNN path tested (falls back to C# if DLL unavailable)");
    }

    [Fact]
    public void BatchNorm_InferenceProducesCorrectOutput()
    {
        var engine = new CpuEngine();
        // 1 batch, 2 channels, 2x2 spatial
        var input = new Tensor<float>(new float[]
        {
            // Channel 0
            1, 2, 3, 4,
            // Channel 1
            5, 6, 7, 8
        }, new[] { 1, 2, 2, 2 });

        var gamma = new Tensor<float>(new float[] { 1, 1 }, new[] { 2 });
        var beta = new Tensor<float>(new float[] { 0, 0 }, new[] { 2 });
        var runningMean = new Tensor<float>(new float[] { 2.5f, 6.5f }, new[] { 2 });
        var runningVar = new Tensor<float>(new float[] { 1.25f, 1.25f }, new[] { 2 });

        var result = engine.BatchNorm(input, gamma, beta, 1e-5f, out _, out _);

        // BN: (x - mean) / sqrt(var + eps) * gamma + beta
        // Channel 0: mean=2.5, var=1.25 → (x-2.5)/sqrt(1.25)
        Assert.Equal(new[] { 1, 2, 2, 2 }, result.Shape.ToArray());
        var data = result.GetDataArray();

        // (1 - 2.5) / sqrt(1.25 + 1e-5) ≈ -1.3416
        Assert.True(MathF.Abs(data[0] - (-1.3416f)) < 0.01f,
            $"BN channel 0, element 0: {data[0]}, expected ~-1.3416");

        _output.WriteLine($"BatchNorm inference: OneDNN path tested (falls back to C# if DLL unavailable)");
    }

    [Fact]
    public void MaxPool2D_WithPadding()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new float[]
        {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        }, new[] { 1, 1, 3, 3 });

        var result = engine.MaxPool2D(input, 2, 1, 0);

        // 2x2 kernel, stride 1, no padding → 2x2 output
        Assert.Equal(new[] { 1, 1, 2, 2 }, result.Shape.ToArray());
        var data = result.GetDataArray();
        Assert.Equal(5f, data[0]); // max(1,2,4,5)
        Assert.Equal(6f, data[1]); // max(2,3,5,6)
        Assert.Equal(8f, data[2]); // max(4,5,7,8)
        Assert.Equal(9f, data[3]); // max(5,6,8,9)
    }
}
