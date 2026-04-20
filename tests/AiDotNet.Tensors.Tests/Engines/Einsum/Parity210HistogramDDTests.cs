using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210HistogramDDTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void HistogramDD_2D_PlacesSamplesInCorrectBins()
    {
        // 4 samples in a 2×2 grid, each in a different quadrant of [0,10]×[0,10].
        var samples = T(new[] { 1f, 1f, 9f, 1f, 1f, 9f, 9f, 9f }, 4, 2);
        var h = E.TensorHistogramDD(samples, new[] { 2, 2 }, new[] { 0f, 0f }, new[] { 10f, 10f });
        Assert.Equal(new[] { 2, 2 }, h.Shape.ToArray());
        // Bins [0,5) and [5,10] on each axis → all 4 samples land in distinct quadrants.
        Assert.Equal(1, h[0, 0]);
        Assert.Equal(1, h[1, 0]);
        Assert.Equal(1, h[0, 1]);
        Assert.Equal(1, h[1, 1]);
    }

    [Fact]
    public void HistogramDD_3D_CountsPerCell()
    {
        var samples = T(new[] { 1f, 1f, 1f, 1f, 1f, 1f, 2f, 2f, 2f }, 3, 3);
        var h = E.TensorHistogramDD(samples, new[] { 2, 2, 2 }, new[] { 0f, 0f, 0f }, new[] { 3f, 3f, 3f });
        // Sample (1,1,1) → cell (0,0,0) two times; (2,2,2) → cell (1,1,1) once.
        Assert.Equal(2, h[0, 0, 0]);
        Assert.Equal(1, h[1, 1, 1]);
    }

    [Fact]
    public void HistogramDD_DropsOutOfRange()
    {
        var samples = T(new[] { -1f, -1f, 5f, 5f, 99f, 99f }, 3, 2);
        var h = E.TensorHistogramDD(samples, new[] { 2, 2 }, new[] { 0f, 0f }, new[] { 10f, 10f });
        // Only the (5, 5) sample falls in range; lands in bin (1, 1) (since 5 is equal to mid/upper edge of bin 0, matches upper).
        Assert.Equal(1, h[0, 0] + h[0, 1] + h[1, 0] + h[1, 1]);
    }

    [Fact]
    public void HistogramDD_WrongLengthArray_Throws()
    {
        var samples = T(new[] { 1f, 1f }, 1, 2);
        Assert.Throws<System.ArgumentException>(
            () => E.TensorHistogramDD(samples, new[] { 2 }, new[] { 0f }, new[] { 10f }));
    }
}
