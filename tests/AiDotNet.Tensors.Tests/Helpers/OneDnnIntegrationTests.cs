using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Correctness tests for MaxPool2D and BatchNorm CpuEngine paths.
/// These validate the C# implementations produce correct numerical results.
/// When OneDNN is available, CpuEngine may dispatch to native kernels internally;
/// these tests verify correctness regardless of which path runs.
/// </summary>
public class OneDnnIntegrationTests
{
    private readonly ITestOutputHelper _output;
    public OneDnnIntegrationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void MaxPool2D_Stride2_ProducesCorrectOutput()
    {
        var engine = new CpuEngine();
        // 1 batch, 1 channel, 4x4 input, 2x2 kernel, stride 2, no padding
        var input = new Tensor<float>(new float[]
        {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        }, new[] { 1, 1, 4, 4 });

        var result = engine.MaxPool2D(input, 2, 2, 0);

        // Expected: max of each 2x2 block with stride 2
        // [6, 8; 14, 16]
        Assert.Equal(new[] { 1, 1, 2, 2 }, result.Shape.ToArray());
        var data = result.GetDataArray();
        Assert.Equal(6f, data[0]);
        Assert.Equal(8f, data[1]);
        Assert.Equal(14f, data[2]);
        Assert.Equal(16f, data[3]);
    }

    [Fact]
    public void BatchNorm_ProducesNormalizedOutput()
    {
        var engine = new CpuEngine();
        // 1 batch, 2 channels, 2x2 spatial
        var input = new Tensor<float>(new float[]
        {
            // Channel 0: values 1,2,3,4
            1, 2, 3, 4,
            // Channel 1: values 5,6,7,8
            5, 6, 7, 8
        }, new[] { 1, 2, 2, 2 });

        var gamma = new Tensor<float>(new float[] { 1, 1 }, new[] { 2 });
        var beta = new Tensor<float>(new float[] { 0, 0 }, new[] { 2 });

        // BatchNorm computes per-channel mean/variance from the input batch
        var result = engine.BatchNorm(input, gamma, beta, 1e-5f, out var mean, out var variance);

        Assert.Equal(new[] { 1, 2, 2, 2 }, result.Shape.ToArray());

        // After BatchNorm, each channel should have approximately zero mean
        var resultData = result.GetDataArray();
        float ch0Sum = resultData[0] + resultData[1] + resultData[2] + resultData[3];
        float ch1Sum = resultData[4] + resultData[5] + resultData[6] + resultData[7];
        Assert.True(MathF.Abs(ch0Sum) < 0.01f,
            $"Channel 0 sum after BN should be ~0, got {ch0Sum}");
        Assert.True(MathF.Abs(ch1Sum) < 0.01f,
            $"Channel 1 sum after BN should be ~0, got {ch1Sum}");
    }

    [Fact]
    public void MaxPool2D_Stride1_SlidingWindow()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new float[]
        {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        }, new[] { 1, 1, 3, 3 });

        // 2x2 kernel, stride 1, no padding → 2x2 output
        var result = engine.MaxPool2D(input, 2, 1, 0);

        Assert.Equal(new[] { 1, 1, 2, 2 }, result.Shape.ToArray());
        var data = result.GetDataArray();
        Assert.Equal(5f, data[0]); // max(1,2,4,5)
        Assert.Equal(6f, data[1]); // max(2,3,5,6)
        Assert.Equal(8f, data[2]); // max(4,5,7,8)
        Assert.Equal(9f, data[3]); // max(5,6,8,9)
    }
}
