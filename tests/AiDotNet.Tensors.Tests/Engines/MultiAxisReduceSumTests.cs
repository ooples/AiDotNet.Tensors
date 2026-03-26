using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tests for multi-axis ReduceSum on 3D tensors.
/// Verifies the fix for the bug where ReduceSum with axes=[0,1] on a 3D tensor
/// produced identical values for all remaining features.
/// </summary>
public class MultiAxisReduceSumTests
{
    private readonly CpuEngine _engine = new();

    [Fact]
    public void ReduceSum_MultiAxis_01_On3D_ProducesDistinctValues()
    {
        // Shape [2, 3, 4] — sum over axes [0, 1] should give [4]
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });
        var rng = new Random(42);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble() * 2.0 - 1.0;

        var result = _engine.ReduceSum(tensor, new[] { 0, 1 });

        Assert.Equal(4, result.Length);

        // Verify result matches manual computation
        for (int d = 0; d < 4; d++)
        {
            double expected = 0;
            for (int b = 0; b < 2; b++)
                for (int s = 0; s < 3; s++)
                    expected += tensor[new[] { b, s, d }];

            Assert.Equal(expected, result[d], 1e-10);
        }

        // Verify values are NOT all identical (the bug produced identical values)
        bool allSame = true;
        for (int d = 1; d < 4; d++)
        {
            if (Math.Abs(result[d] - result[0]) > 1e-10)
            {
                allSame = false;
                break;
            }
        }
        Assert.False(allSame, "ReduceSum with axes=[0,1] produced identical values for all features — multi-axis reduction bug");
    }

    [Fact]
    public void ReduceSum_SingleAxis_0_On3D_IsCorrect()
    {
        // Shape [2, 3, 4] — sum over axis [0] should give [3, 4]
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });
        var rng = new Random(42);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble() * 2.0 - 1.0;

        var result = _engine.ReduceSum(tensor, new[] { 0 });

        Assert.Equal(12, result.Length); // 3 * 4

        for (int s = 0; s < 3; s++)
        {
            for (int d = 0; d < 4; d++)
            {
                double expected = 0;
                for (int b = 0; b < 2; b++)
                    expected += tensor[new[] { b, s, d }];
                Assert.Equal(expected, result[new[] { s, d }], 1e-10);
            }
        }
    }

    [Fact]
    public void ReduceSum_SingleAxis_2_On3D_IsCorrect()
    {
        // Shape [2, 3, 4] — sum over axis [2] should give [2, 3]
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });
        var rng = new Random(42);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble() * 2.0 - 1.0;

        var result = _engine.ReduceSum(tensor, new[] { 2 });

        Assert.Equal(6, result.Length); // 2 * 3

        for (int b = 0; b < 2; b++)
        {
            for (int s = 0; s < 3; s++)
            {
                double expected = 0;
                for (int d = 0; d < 4; d++)
                    expected += tensor[new[] { b, s, d }];
                Assert.Equal(expected, result[new[] { b, s }], 1e-10);
            }
        }
    }

    [Fact]
    public void TensorSum_MultiAxis_MatchesManual()
    {
        // Test the Tensor.Sum method directly (used by ReduceSum internally)
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });
        var rng = new Random(42);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble() * 2.0 - 1.0;

        var result = tensor.Sum(new[] { 0, 1 });

        Assert.Equal(4, result.Length);
        for (int d = 0; d < 4; d++)
        {
            double expected = 0;
            for (int b = 0; b < 2; b++)
                for (int s = 0; s < 3; s++)
                    expected += tensor[new[] { b, s, d }];
            Assert.Equal(expected, result[d], 1e-10);
        }
    }

    [Fact]
    public void ReduceSum_Float_MultiAxis_01_On3D_ProducesDistinctValues()
    {
        // Same test with float to verify SIMD path
        var tensor = new Tensor<float>(new[] { 2, 3, 4 });
        var rng = new Random(42);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var result = _engine.ReduceSum(tensor, new[] { 0, 1 });

        Assert.Equal(4, result.Length);

        for (int d = 0; d < 4; d++)
        {
            float expected = 0;
            for (int b = 0; b < 2; b++)
                for (int s = 0; s < 3; s++)
                    expected += tensor[new[] { b, s, d }];

            Assert.Equal(expected, result[d], 1e-5f);
        }
    }
}
