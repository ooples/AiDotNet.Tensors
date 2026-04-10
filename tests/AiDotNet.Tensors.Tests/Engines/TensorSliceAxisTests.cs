using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Integration tests for generalized TensorSliceAxis supporting any rank (1D through N-D).
/// Verifies correctness, mathematical invariants, and backward gradient flow.
/// </summary>
public class TensorSliceAxisTests
{
    private readonly CpuEngine _engine = new();

    // ═══════════════════════════════════════════════════════════════════
    // SHAPE INVARIANTS: output rank = input rank - 1, correct dimensions
    // ═══════════════════════════════════════════════════════════════════

    [Theory]
    [InlineData(new[] { 5 }, 0, 2, new int[0])]           // 1D -> scalar
    [InlineData(new[] { 3, 4 }, 0, 1, new[] { 4 })]       // 2D axis=0 -> 1D
    [InlineData(new[] { 3, 4 }, 1, 2, new[] { 3 })]       // 2D axis=1 -> 1D
    [InlineData(new[] { 2, 3, 4 }, 0, 1, new[] { 3, 4 })] // 3D axis=0 -> 2D
    [InlineData(new[] { 2, 3, 4 }, 1, 2, new[] { 2, 4 })] // 3D axis=1 -> 2D
    [InlineData(new[] { 2, 3, 4 }, 2, 3, new[] { 2, 3 })] // 3D axis=2 -> 2D
    [InlineData(new[] { 2, 3, 4, 5 }, 0, 0, new[] { 3, 4, 5 })] // 4D axis=0
    [InlineData(new[] { 2, 3, 4, 5 }, 1, 1, new[] { 2, 4, 5 })] // 4D axis=1
    [InlineData(new[] { 2, 3, 4, 5 }, 2, 2, new[] { 2, 3, 5 })] // 4D axis=2
    [InlineData(new[] { 2, 3, 4, 5 }, 3, 3, new[] { 2, 3, 4 })] // 4D axis=3
    [InlineData(new[] { 2, 3, 4, 5, 6 }, 0, 1, new[] { 3, 4, 5, 6 })] // 5D axis=0
    [InlineData(new[] { 2, 3, 4, 5, 6 }, 4, 5, new[] { 2, 3, 4, 5 })] // 5D axis=4
    public void SliceAxis_OutputShape_IsCorrect(int[] inputShape, int axis, int index, int[] expectedShape)
    {
        var tensor = CreateSequentialTensor(inputShape);
        var result = _engine.TensorSliceAxis(tensor, axis, index);

        Assert.Equal(expectedShape.Length, result.Rank);
        for (int i = 0; i < expectedShape.Length; i++)
            Assert.Equal(expectedShape[i], result.Shape[i]);
    }

    // ═══════════════════════════════════════════════════════════════════
    // VALUE INVARIANTS: sliced values match element-by-element indexing
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    public void SliceAxis_2D_Axis0_ValuesMatchManualIndexing()
    {
        // [3, 4] tensor with sequential values
        var tensor = CreateSequentialTensor(new[] { 3, 4 });
        var result = _engine.TensorSliceAxis(tensor, 0, 1); // Row 1

        // result should be [4] = [4, 5, 6, 7]
        Assert.Equal(4, result.Length);
        Assert.Equal(4f, result[0]);
        Assert.Equal(5f, result[1]);
        Assert.Equal(6f, result[2]);
        Assert.Equal(7f, result[3]);
    }

    [Fact]
    public void SliceAxis_2D_Axis1_ValuesMatchManualIndexing()
    {
        var tensor = CreateSequentialTensor(new[] { 3, 4 });
        var result = _engine.TensorSliceAxis(tensor, 1, 2); // Column 2

        // result should be [3] = [2, 6, 10]
        Assert.Equal(3, result.Length);
        Assert.Equal(2f, result[0]);
        Assert.Equal(6f, result[1]);
        Assert.Equal(10f, result[2]);
    }

    [Fact]
    public void SliceAxis_3D_MatchesOriginal3DImplementation()
    {
        // Verify the generalized implementation produces identical results
        // to what the old hardcoded 3D implementation would produce
        var tensor = CreateSequentialTensor(new[] { 2, 3, 4 });

        // axis=0, index=1: result[j,k] = tensor[1, j, k]
        var r0 = _engine.TensorSliceAxis(tensor, 0, 1);
        Assert.Equal(new[] { 3, 4 }, r0.Shape.ToArray());
        Assert.Equal(12f, r0[0, 0]); // tensor[1,0,0] = 1*3*4 + 0 + 0 = 12

        // axis=1, index=2: result[i,k] = tensor[i, 2, k]
        var r1 = _engine.TensorSliceAxis(tensor, 1, 2);
        Assert.Equal(new[] { 2, 4 }, r1.Shape.ToArray());
        Assert.Equal(8f, r1[0, 0]); // tensor[0,2,0] = 0 + 2*4 + 0 = 8

        // axis=2, index=3: result[i,j] = tensor[i, j, 3]
        var r2 = _engine.TensorSliceAxis(tensor, 2, 3);
        Assert.Equal(new[] { 2, 3 }, r2.Shape.ToArray());
        Assert.Equal(3f, r2[0, 0]); // tensor[0,0,3] = 3
    }

    [Fact]
    public void SliceAxis_4D_ValuesMatchManualIndexing()
    {
        // [2, 3, 4, 5] — VideoCLIP use case
        var tensor = CreateSequentialTensor(new[] { 2, 3, 4, 5 });

        // Slice axis=0, index=1: get frame 1 -> [3, 4, 5]
        var frame1 = _engine.TensorSliceAxis(tensor, 0, 1);
        Assert.Equal(new[] { 3, 4, 5 }, frame1.Shape.ToArray());
        // tensor[1,0,0,0] = 1 * (3*4*5) = 60
        Assert.Equal(60f, frame1[0, 0, 0]);
        // tensor[1,2,3,4] = 60 + 2*20 + 3*5 + 4 = 119
        Assert.Equal(119f, frame1[2, 3, 4]);
    }

    // ═══════════════════════════════════════════════════════════════════
    // MATHEMATICAL INVARIANT: slice + reconstruct = original
    // Slicing all indices along an axis and stacking should recover input
    // ═══════════════════════════════════════════════════════════════════

    [Theory]
    [InlineData(new[] { 3, 4 }, 0)]
    [InlineData(new[] { 3, 4 }, 1)]
    [InlineData(new[] { 2, 3, 4 }, 0)]
    [InlineData(new[] { 2, 3, 4 }, 1)]
    [InlineData(new[] { 2, 3, 4 }, 2)]
    [InlineData(new[] { 2, 3, 4, 5 }, 0)]
    [InlineData(new[] { 2, 3, 4, 5 }, 2)]
    public void SliceAndReconstruct_RecoverOriginal(int[] shape, int axis)
    {
        var tensor = CreateSequentialTensor(shape);
        int axisSize = shape[axis];

        // Slice all indices, then verify each slice has correct values
        var slices = new Tensor<float>[axisSize];
        for (int i = 0; i < axisSize; i++)
            slices[i] = _engine.TensorSliceAxis(tensor, axis, i);

        // Reconstruct: create zero tensor, set each slice back
        var reconstructed = new Tensor<float>(shape);
        for (int i = 0; i < axisSize; i++)
            _engine.TensorSetSliceAxis(reconstructed, slices[i], axis, i);

        // Verify reconstruction matches original
        var origData = tensor.GetDataArray();
        var reconData = reconstructed.GetDataArray();
        for (int i = 0; i < origData.Length; i++)
            Assert.Equal(origData[i], reconData[i]);
    }

    // ═══════════════════════════════════════════════════════════════════
    // BACKWARD: gradient flows through slice correctly
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    public void SliceAxis_Backward_ProducesGradientAtCorrectPosition()
    {
        var tensor = Tensor<float>.CreateRandom([2, 3, 4, 5]);

        using var tape = new GradientTape<float>();
        var sliced = _engine.TensorSliceAxis(tensor, 1, 2); // axis=1, index=2
        var loss = _engine.ReduceSum(sliced, null);
        var grads = tape.ComputeGradients(loss, new[] { tensor });

        Assert.True(grads.ContainsKey(tensor), "Gradient should exist for sliced tensor");
        var grad = grads[tensor];
        Assert.Equal(tensor.Shape.ToArray(), grad.Shape.ToArray());

        // Gradient should be 1.0 at [*, 2, *, *] and 0.0 elsewhere
        var gradData = grad.GetDataArray();
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                    for (int l = 0; l < 5; l++)
                    {
                        float expected = j == 2 ? 1.0f : 0.0f;
                        int flatIdx = i * 60 + j * 20 + k * 5 + l;
                        Assert.True(Math.Abs(expected - gradData[flatIdx]) < 1e-6f,
                            $"Gradient at [{i},{j},{k},{l}] should be {expected} but was {gradData[flatIdx]}");
                    }
    }

    // ═══════════════════════════════════════════════════════════════════
    // BOUNDARY: validation and edge cases
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    public void SliceAxis_InvalidAxis_Throws()
    {
        var tensor = Tensor<float>.CreateRandom([2, 3, 4]);
        Assert.Throws<ArgumentOutOfRangeException>(() => _engine.TensorSliceAxis(tensor, 3, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => _engine.TensorSliceAxis(tensor, -1, 0));
    }

    [Fact]
    public void SliceAxis_InvalidIndex_Throws()
    {
        var tensor = Tensor<float>.CreateRandom([2, 3, 4]);
        Assert.Throws<ArgumentOutOfRangeException>(() => _engine.TensorSliceAxis(tensor, 0, 2)); // size=2, max index=1
        Assert.Throws<ArgumentOutOfRangeException>(() => _engine.TensorSliceAxis(tensor, 1, -1));
    }

    [Fact]
    public void SliceAxis_1D_ProducesScalar()
    {
        var tensor = new Tensor<float>(new float[] { 10, 20, 30, 40, 50 }, new[] { 5 });
        var result = _engine.TensorSliceAxis(tensor, 0, 3);
        Assert.Equal(0, result.Rank); // scalar
        Assert.Equal(40f, result[0]);
    }

    // ═══════════════════════════════════════════════════════════════════
    // SET SLICE: TensorSetSliceAxis for all ranks
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    public void SetSliceAxis_4D_WritesCorrectValues()
    {
        var dest = new Tensor<float>(new[] { 2, 3, 4, 5 }); // all zeros
        var source = CreateSequentialTensor(new[] { 3, 4, 5 }); // sequential 0..59

        _engine.TensorSetSliceAxis(dest, source, 0, 1); // set dest[1,:,:,:] = source

        // dest[1,0,0,0] should be 0 (first element of source)
        Assert.Equal(0f, dest.GetDataArray()[1 * 60]); // offset = 1 * 3*4*5 = 60
        // dest[1,2,3,4] should be 59 (last element of source)
        Assert.Equal(59f, dest.GetDataArray()[1 * 60 + 59]);
        // dest[0,0,0,0] should still be 0 (untouched)
        Assert.Equal(0f, dest.GetDataArray()[0]);
    }

    // ═══════════════════════════════════════════════════════════════════
    // HELPERS
    // ═══════════════════════════════════════════════════════════════════

    private static Tensor<float> CreateSequentialTensor(int[] shape)
    {
        int length = 1;
        foreach (var d in shape) length *= d;
        var data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = i;
        return new Tensor<float>(data, shape);
    }
}
