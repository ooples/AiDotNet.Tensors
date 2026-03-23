using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Tests for Tensor.GetVectorAlongAxis and SetVectorAlongAxis.
/// </summary>
public class TensorSliceTests
{
    [Fact]
    public void GetVectorAlongAxis_LastDimension_ReturnsContiguousSlice()
    {
        // 2D tensor [3, 4] — extracting along axis 1 (columns) for row 1
        var data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        var tensor = new Tensor<float>(data, [3, 4]);

        var row1 = tensor.GetVectorAlongAxis(1, 1); // axis=1, fixedIndices=[row=1]

        Assert.Equal(4, row1.Length);
        Assert.Equal(5f, row1[0]);
        Assert.Equal(6f, row1[1]);
        Assert.Equal(7f, row1[2]);
        Assert.Equal(8f, row1[3]);
    }

    [Fact]
    public void GetVectorAlongAxis_FirstDimension_ReturnsStridedSlice()
    {
        // 2D tensor [3, 4] — extracting along axis 0 (rows) for column 2
        var data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        var tensor = new Tensor<float>(data, [3, 4]);

        var col2 = tensor.GetVectorAlongAxis(0, 2); // axis=0, fixedIndices=[col=2]

        Assert.Equal(3, col2.Length);
        Assert.Equal(3f, col2[0]);  // tensor[0, 2]
        Assert.Equal(7f, col2[1]);  // tensor[1, 2]
        Assert.Equal(11f, col2[2]); // tensor[2, 2]
    }

    [Fact]
    public void GetVectorAlongAxis_3D_ExtractAlongMiddleDimension()
    {
        // 3D tensor [2, 3, 4] — extract along axis 1 (features) with batch=0, dim=1
        var tensor = new Tensor<float>([2, 3, 4]);
        for (int b = 0; b < 2; b++)
            for (int f = 0; f < 3; f++)
                for (int d = 0; d < 4; d++)
                    tensor[new[] { b, f, d }] = b * 100 + f * 10 + d;

        // Extract features for batch=0, dim=1 → should get [1, 11, 21]
        var slice = tensor.GetVectorAlongAxis(1, 0, 1); // axis=1, fixed=[batch=0, dim=1]

        Assert.Equal(3, slice.Length);
        Assert.Equal(1f, slice[0]);   // tensor[0, 0, 1]
        Assert.Equal(11f, slice[1]);  // tensor[0, 1, 1]
        Assert.Equal(21f, slice[2]);  // tensor[0, 2, 1]
    }

    [Fact]
    public void GetVectorAlongAxis_3D_ExtractAlongLastDimension()
    {
        // 3D tensor [2, 3, 4] — extract along axis 2 (last dim) with batch=1, feature=2
        var tensor = new Tensor<float>([2, 3, 4]);
        for (int b = 0; b < 2; b++)
            for (int f = 0; f < 3; f++)
                for (int d = 0; d < 4; d++)
                    tensor[new[] { b, f, d }] = b * 100 + f * 10 + d;

        // Extract dims for batch=1, feature=2 → should get [120, 121, 122, 123]
        var slice = tensor.GetVectorAlongAxis(2, 1, 2); // axis=2, fixed=[batch=1, feature=2]

        Assert.Equal(4, slice.Length);
        Assert.Equal(120f, slice[0]);
        Assert.Equal(121f, slice[1]);
        Assert.Equal(122f, slice[2]);
        Assert.Equal(123f, slice[3]);
    }

    [Fact]
    public void SetVectorAlongAxis_WritesCorrectly()
    {
        var tensor = new Tensor<float>([3, 4]);
        var values = new Vector<float>(new float[] { 10, 20, 30, 40 });

        tensor.SetVectorAlongAxis(values, 1, 1); // Set row 1

        Assert.Equal(10f, tensor[new[] { 1, 0 }]);
        Assert.Equal(20f, tensor[new[] { 1, 1 }]);
        Assert.Equal(30f, tensor[new[] { 1, 2 }]);
        Assert.Equal(40f, tensor[new[] { 1, 3 }]);
    }

    [Fact]
    public void SetVectorAlongAxis_StridedWrite()
    {
        var tensor = new Tensor<float>([3, 4]);
        var values = new Vector<float>(new float[] { 100, 200, 300 });

        tensor.SetVectorAlongAxis(values, 0, 2); // Set column 2

        Assert.Equal(100f, tensor[new[] { 0, 2 }]);
        Assert.Equal(200f, tensor[new[] { 1, 2 }]);
        Assert.Equal(300f, tensor[new[] { 2, 2 }]);
    }

    [Fact]
    public void GetVectorAlongAxis_DotProduct_MatchesManual()
    {
        // Verify that extracting a slice and doing DotProduct gives same result as manual loop
        var tensor = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, [3, 3]);
        var vec = new Vector<float>(new float[] { 1, 1, 1 });

        // Manual: sum of column 1 = 2 + 5 + 8 = 15
        var col1 = tensor.GetVectorAlongAxis(0, 1); // axis=0 (rows), fixed col=1
        var engine = AiDotNetEngine.Current;
        var dot = engine.DotProduct(col1, vec);

        Assert.Equal(15f, dot, 1e-6f);
    }

    [Fact]
    public void StridedDotProduct_ReverseAccess()
    {
        var a = new Vector<float>(new float[] { 1, 2, 3 });
        var b = new Vector<float>(new float[] { 10, 20, 30, 40, 50 });

        var engine = AiDotNetEngine.Current;

        // a[0]*b[4] + a[1]*b[3] + a[2]*b[2] = 1*50 + 2*40 + 3*30 = 50 + 80 + 90 = 220
        var result = engine.DotProduct(a, b, bOffset: 4, bStride: -1);

        Assert.Equal(220f, result, 1e-6f);
    }

    [Fact]
    public void StridedDotProduct_ForwardStride2()
    {
        var a = new Vector<float>(new float[] { 1, 2, 3 });
        var b = new Vector<float>(new float[] { 10, 20, 30, 40, 50, 60 });

        var engine = AiDotNetEngine.Current;

        // a[0]*b[0] + a[1]*b[2] + a[2]*b[4] = 1*10 + 2*30 + 3*50 = 10 + 60 + 150 = 220
        var result = engine.DotProduct(a, b, bOffset: 0, bStride: 2);

        Assert.Equal(220f, result, 1e-6f);
    }
}
