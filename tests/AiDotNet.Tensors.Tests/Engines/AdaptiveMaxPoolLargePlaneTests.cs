using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Adaptive max pool's bin bounds are floor(i * In / Out) and ceil((i + 1) * In / Out). Computed in int, i * In passed
/// int.MaxValue for a 45,000-row plane pooled to 50,000 rows (from output row ~47,720), so the bins there were wrong.
/// </summary>
public class AdaptiveMaxPoolLargePlaneTests
{
    [Fact]
    public void AdaptiveMaxPool2D_BinBoundsHoldPastIntRange()
    {
        const int h = 45_000, outH = 50_000;
        var data = new float[h];
        for (int i = 0; i < h; i++) data[i] = i;   // increasing, so a bin's max is its last row
        var input = new Tensor<float>(data, new[] { 1, 1, h, 1 });

        var output = new CpuEngine().TensorAdaptiveMaxPool2D(input, new[] { outH, 1 }).ToArray();

        Assert.Equal(outH, output.Length);
        for (int oh = 0; oh < outH; oh++)
        {
            long end = ((long)(oh + 1) * h + outH - 1) / outH;
            Assert.Equal((float)(end - 1), output[oh]);
        }
    }
}
