using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>Advanced indexing against torch's aten::index on a [2, 3, 4, 5] arange (parity/torch_index_ref.py).</summary>
public class AdvancedIndexTests
{
    private readonly IEngine _engine = new CpuEngine();

    private static Tensor<float> X() => new Tensor<float>(Enumerable.Range(0, 120).Select(i => (float)i).ToArray(), new[] { 2, 3, 4, 5 });

    private static void AssertSame(float[] expected, int[] shape, Tensor<float> actual)
    {
        Assert.Equal(shape, actual.Shape.ToArray());
        Assert.Equal(expected, actual.ToArray());
    }

    [Fact]
    public void Index_single_dim0_MatchesTorch()
        => AssertSame(new float[] { 60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f, 81.0f, 82.0f, 83.0f, 84.0f, 85.0f, 86.0f, 87.0f, 88.0f, 89.0f, 90.0f, 91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f, 97.0f, 98.0f, 99.0f, 100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f, 110.0f, 111.0f, 112.0f, 113.0f, 114.0f, 115.0f, 116.0f, 117.0f, 118.0f, 119.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f, 81.0f, 82.0f, 83.0f, 84.0f, 85.0f, 86.0f, 87.0f, 88.0f, 89.0f, 90.0f, 91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f, 97.0f, 98.0f, 99.0f, 100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f, 110.0f, 111.0f, 112.0f, 113.0f, 114.0f, 115.0f, 116.0f, 117.0f, 118.0f, 119.0f }, new int[] { 3, 3, 4, 5 }, _engine.TensorIndex(X(), new Tensor<int>?[] { new Tensor<int>(new int[] { 1, 0, 1 }, new int[] { 3 }) }));

    [Fact]
    public void Index_adjacent_12_MatchesTorch()
        => AssertSame(new float[] { 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f }, new int[] { 2, 2, 5 }, _engine.TensorIndex(X(), new Tensor<int>?[] { null, new Tensor<int>(new int[] { 0, 2 }, new int[] { 2 }), new Tensor<int>(new int[] { 3, 1 }, new int[] { 2 }) }));

    [Fact]
    public void Index_broadcast_12_MatchesTorch()
        => AssertSame(new float[] { 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 115.0f, 116.0f, 117.0f, 118.0f, 119.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f, 100.0f, 101.0f, 102.0f, 103.0f, 104.0f }, new int[] { 2, 2, 3, 5 }, _engine.TensorIndex(X(), new Tensor<int>?[] { null, new Tensor<int>(new int[] { 0, 2 }, new int[] { 2, 1 }), new Tensor<int>(new int[] { 3, 1, 0 }, new int[] { 3 }) }));

    [Fact]
    public void Index_separated_0_2_MatchesTorch()
        => AssertSame(new float[] { 70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 90.0f, 91.0f, 92.0f, 93.0f, 94.0f, 110.0f, 111.0f, 112.0f, 113.0f, 114.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f }, new int[] { 2, 3, 5 }, _engine.TensorIndex(X(), new Tensor<int>?[] { new Tensor<int>(new int[] { 1, 0 }, new int[] { 2 }), null, new Tensor<int>(new int[] { 2, 3 }, new int[] { 2 }) }));

    [Fact]
    public void Index_negative_MatchesTorch()
        => AssertSame(new float[] { 70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 90.0f, 91.0f, 92.0f, 93.0f, 94.0f, 110.0f, 111.0f, 112.0f, 113.0f, 114.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 80.0f, 81.0f, 82.0f, 83.0f, 84.0f, 100.0f, 101.0f, 102.0f, 103.0f, 104.0f }, new int[] { 2, 3, 5 }, _engine.TensorIndex(X(), new Tensor<int>?[] { new Tensor<int>(new int[] { -1 }, new int[] { 1 }), null, new Tensor<int>(new int[] { -2, 0 }, new int[] { 2 }) }));

    [Fact]
    public void Index_trailing_none_MatchesTorch()
        => AssertSame(new float[] { 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 85.0f, 86.0f, 87.0f, 88.0f, 89.0f, 95.0f, 96.0f, 97.0f, 98.0f, 99.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f, 115.0f, 116.0f, 117.0f, 118.0f, 119.0f }, new int[] { 2, 3, 2, 5 }, _engine.TensorIndex(X(), new Tensor<int>?[] { null, null, new Tensor<int>(new int[] { 1, 3 }, new int[] { 2 }) }));

    [Fact]
    public void Index_all_four_MatchesTorch()
        => AssertSame(new float[] { 59.0f, 65.0f }, new int[] { 2 }, _engine.TensorIndex(X(), new Tensor<int>?[] { new Tensor<int>(new int[] { 0, 1 }, new int[] { 2 }), new Tensor<int>(new int[] { 2, 0 }, new int[] { 2 }), new Tensor<int>(new int[] { 3, 1 }, new int[] { 2 }), new Tensor<int>(new int[] { 4, 0 }, new int[] { 2 }) }));

    [Fact]
    public void Index_BoolMask_MatchesTorch()
        => AssertSame(new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 80.0f, 81.0f, 82.0f, 83.0f, 84.0f, 85.0f, 86.0f, 87.0f, 88.0f, 89.0f, 90.0f, 91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f, 97.0f, 98.0f, 99.0f }, new int[] { 3, 4, 5 }, _engine.TensorIndex(X(), new Tensor<bool>(new bool[] { true, false, true, false, true, false }, new int[] { 2, 3 })));

    [Fact]
    public void Index_Gradient_SumsRepeatedPositions()
    {
        var x = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        Tensor<float> g;
        using (var tape = new GradientTape<float>())
        {
            var picked = _engine.TensorIndex(x, new Tensor<int>?[] { new Tensor<int>(new[] { 2, 0, 2, 2 }, new[] { 4 }) });
            g = tape.ComputeGradients(_engine.ReduceSum(picked, null), new[] { x })[x];
        }
        Assert.Equal(new float[] { 1, 0, 3, 0 }, g.ToArray());
    }

    [Fact]
    public void Index_OutOfRange_Throws()
        => Assert.Throws<IndexOutOfRangeException>(() => _engine.TensorIndex(X(), new Tensor<int>?[] { new Tensor<int>(new[] { 2 }, new[] { 1 }) }));
}
