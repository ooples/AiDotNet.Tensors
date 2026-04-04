using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Integration tests proving SparseTensor works with the autodiff pipeline.
/// Validates the fix for issue #98: SparseTensor incompatible with ParameterBuffer/GradientTape.
/// </summary>
public class SparseTensorAutodiffTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    [Fact]
    public void SparseTensor_Indexer_ReturnsCorrectValues()
    {
        // COO format: 3x3 identity matrix
        var sparse = new SparseTensor<float>(3, 3,
            new[] { 0, 1, 2 }, new[] { 0, 1, 2 }, new[] { 1f, 2f, 3f });

        Assert.Equal(1f, sparse[0, 0]);
        Assert.Equal(2f, sparse[1, 1]);
        Assert.Equal(3f, sparse[2, 2]);
        Assert.Equal(0f, sparse[0, 1]); // Missing entry returns zero
        Assert.Equal(0f, sparse[1, 0]);
    }

    [Fact]
    public void SparseTensor_CsrIndexer_ReturnsCorrectValues()
    {
        // Build CSR from COO
        var coo = new SparseTensor<float>(3, 3,
            new[] { 0, 1, 2 }, new[] { 0, 1, 2 }, new[] { 5f, 6f, 7f });
        var csr = coo.ToCsr();

        Assert.Equal(5f, csr[0, 0]);
        Assert.Equal(6f, csr[1, 1]);
        Assert.Equal(7f, csr[2, 2]);
        Assert.Equal(0f, csr[0, 2]);
    }

    [Fact]
    public void SparseTensor_CloneSparse_PreservesData()
    {
        var original = new SparseTensor<float>(3, 3,
            new[] { 0, 1, 2 }, new[] { 0, 1, 2 }, new[] { 1f, 2f, 3f });
        var clone = original.CloneSparse();

        Assert.Equal(original.NonZeroCount, clone.NonZeroCount);
        Assert.Equal(original.Rows, clone.Rows);
        Assert.Equal(original.Columns, clone.Columns);
        Assert.Equal(1f, clone[0, 0]);
        Assert.Equal(2f, clone[1, 1]);
        Assert.Equal(3f, clone[2, 2]);
    }

    [Fact]
    public void SparseTensor_ToDense_RoundTrips()
    {
        var sparse = new SparseTensor<float>(3, 3,
            new[] { 0, 1, 2 }, new[] { 0, 1, 2 }, new[] { 4f, 5f, 6f });
        var dense = sparse.ToDense();

        Assert.Equal(4f, dense[0, 0]);
        Assert.Equal(5f, dense[1, 1]);
        Assert.Equal(6f, dense[2, 2]);
        Assert.Equal(0f, dense[0, 1]);
    }

    [Fact]
    public void ParameterBuffer_CopyFrom_HandlesSparseParameters()
    {
        // This is the exact scenario from issue #98
        var sparse = new SparseTensor<float>(4, 4,
            new[] { 0, 1, 2, 3 }, new[] { 0, 1, 2, 3 }, new[] { 1f, 2f, 3f, 4f });

        var shapes = new[] { sparse._shape };
        var buffer = new ParameterBuffer<float>(shapes);

        // This used to throw: InvalidOperationException: Contiguous is not supported on sparse tensors
        buffer.CopyFrom(new[] { sparse });

        // Verify the buffer contains the densified sparse data
        var data = buffer.AsVector();
        Assert.Equal(4 * 4, data.Length); // Full dense 4x4 = 16 elements
        Assert.Equal(1f, data[0]);   // [0,0]
        Assert.Equal(2f, data[5]);   // [1,1]
        Assert.Equal(3f, data[10]);  // [2,2]
        Assert.Equal(4f, data[15]);  // [3,3]
        Assert.Equal(0f, data[1]);   // [0,1] = 0
    }

    [Fact]
    public void ParameterBuffer_FlattenGradients_HandlesSparseGradients()
    {
        var param = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var sparseGrad = new SparseTensor<float>(2, 2,
            new[] { 0, 1 }, new[] { 0, 1 }, new[] { 0.1f, 0.2f });

        var shapes = new[] { param._shape };
        var buffer = new ParameterBuffer<float>(shapes);
        var grads = new System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>>
        {
            { param, sparseGrad }
        };

        // This used to throw for sparse gradients
        var flatGrad = buffer.FlattenGradients(new[] { param }, grads);

        Assert.Equal(4, flatGrad.Length);
        Assert.Equal(0.1f, flatGrad[0]); // [0,0]
        Assert.Equal(0.2f, flatGrad[3]); // [1,1]
        Assert.Equal(0f, flatGrad[1]);   // [0,1]
    }

    [Fact]
    public void SparseTensor_ToDense_ThenGradientTape_ProducesGradients()
    {
        // Simulate sparse parameter in a training step
        var sparseWeight = new SparseTensor<float>(3, 3,
            new[] { 0, 1, 2 }, new[] { 0, 1, 2 }, new[] { 1f, 1f, 1f });
        var denseInput = new Tensor<float>(new float[] { 1, 2, 3 }, new[] { 3, 1 });

        // Convert sparse to dense for matmul (SpMM would be better but this tests the pipeline)
        var denseWeight = sparseWeight.ToDense();

        using var tape = new GradientTape<float>();
        var output = _engine.TensorMatMul(denseWeight, denseInput);
        var loss = _engine.TensorMeanDiff(output);
        var grads = tape.ComputeGradients(loss, sources: new[] { denseWeight });

        Assert.True(grads.ContainsKey(denseWeight));
        Assert.Equal(denseWeight.Length, grads[denseWeight].Length);
    }

    [Fact]
    public void SparseTensor_ParameterBuffer_FullTrainingCycle()
    {
        // Full integration: sparse param → buffer → tape → gradients → flatten
        var sparse = new SparseTensor<float>(2, 2,
            new[] { 0, 1 }, new[] { 0, 1 }, new[] { 1f, 1f });
        var input = new Tensor<float>(new float[] { 1, 2 }, new[] { 2, 1 });

        // Step 1: Create buffer from sparse params
        var shapes = new[] { sparse._shape };
        var buffer = new ParameterBuffer<float>(shapes);
        buffer.CopyFrom(new[] { sparse });

        // Step 2: Get dense view for computation
        var denseParam = sparse.ToDense();

        // Step 3: Forward + backward
        using var tape = new GradientTape<float>();
        var output = _engine.TensorMatMul(denseParam, input);
        var loss = _engine.TensorMeanDiff(output);
        var grads = tape.ComputeGradients(loss, sources: new[] { denseParam });

        // Step 4: Flatten gradients (the other failure point from issue #98)
        var flatGrad = buffer.FlattenGradients(new[] { denseParam }, grads);

        Assert.Equal(4, flatGrad.Length);
        // Gradient should be non-zero for at least some elements
        bool anyNonZero = false;
        for (int i = 0; i < flatGrad.Length; i++)
            if (flatGrad[i] != 0f) anyNonZero = true;
        Assert.True(anyNonZero, "Gradients should be non-zero after backward pass");
    }
}
