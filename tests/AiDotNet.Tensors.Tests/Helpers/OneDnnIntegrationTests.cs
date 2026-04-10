using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Correctness tests for MaxPool2D and BatchNorm CpuEngine paths.
/// These validate numerical results for whichever internal path runs
/// (OneDNN native or C# fallback, depending on environment).
/// </summary>
public class OneDnnIntegrationTests
{
    [Fact]
    public void MaxPool2D_Stride2_ProducesCorrectOutput()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new float[]
        {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        }, new[] { 1, 1, 4, 4 });

        var result = engine.MaxPool2D(input, 2, 2, 0);

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
            1, 2, 3, 4,   // Channel 0: mean=2.5, var=1.25
            5, 6, 7, 8    // Channel 1: mean=6.5, var=1.25
        }, new[] { 1, 2, 2, 2 });

        var gamma = new Tensor<float>(new float[] { 1, 1 }, new[] { 2 });
        var beta = new Tensor<float>(new float[] { 0, 0 }, new[] { 2 });

        var result = engine.BatchNorm(input, gamma, beta, 1e-5f, out var mean, out var variance);

        Assert.Equal(new[] { 1, 2, 2, 2 }, result.Shape.ToArray());
        var resultData = result.GetDataArray();
        var numOps = MathHelper.GetNumericOperations<float>();

        // Verify computed mean
        var meanData = mean.GetDataArray();
        Assert.True(MathF.Abs(numOps.ToDouble(meanData[0]) - 2.5) < 0.01,
            $"Channel 0 mean should be 2.5, got {meanData[0]}");
        Assert.True(MathF.Abs(numOps.ToDouble(meanData[1]) - 6.5) < 0.01,
            $"Channel 1 mean should be 6.5, got {meanData[1]}");

        // Verify computed variance
        var varData = variance.GetDataArray();
        Assert.True(MathF.Abs(numOps.ToDouble(varData[0]) - 1.25) < 0.1,
            $"Channel 0 variance should be ~1.25, got {varData[0]}");
        Assert.True(MathF.Abs(numOps.ToDouble(varData[1]) - 1.25) < 0.1,
            $"Channel 1 variance should be ~1.25, got {varData[1]}");

        // Verify element-level normalization: (x - mean) / sqrt(var + eps) * gamma + beta
        float expectedFirst = (1f - 2.5f) / MathF.Sqrt(1.25f + 1e-5f);
        Assert.True(MathF.Abs(resultData[0] - expectedFirst) < 0.05f,
            $"First element should be ~{expectedFirst:F4}, got {resultData[0]}");

        float expectedLast = (4f - 2.5f) / MathF.Sqrt(1.25f + 1e-5f);
        Assert.True(MathF.Abs(resultData[3] - expectedLast) < 0.05f,
            $"Last ch0 element should be ~{expectedLast:F4}, got {resultData[3]}");

        // Verify zero-mean per channel
        float ch0Sum = resultData[0] + resultData[1] + resultData[2] + resultData[3];
        Assert.True(MathF.Abs(ch0Sum) < 0.01f,
            $"Channel 0 sum after BN should be ~0, got {ch0Sum}");
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

        var result = engine.MaxPool2D(input, 2, 1, 0);

        Assert.Equal(new[] { 1, 1, 2, 2 }, result.Shape.ToArray());
        var data = result.GetDataArray();
        Assert.Equal(5f, data[0]);
        Assert.Equal(6f, data[1]);
        Assert.Equal(8f, data[2]);
        Assert.Equal(9f, data[3]);
    }
}
