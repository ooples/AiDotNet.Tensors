using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Locks the contract of <see cref="SparseEmbeddingGradient{T}"/> against the existing
/// dense <c>TensorEmbeddingLookupBackward</c> path. The sparse representation is
/// only useful if its <c>ToDense</c> materialization is bit-exact with what the
/// dense backward already produces — otherwise routing the optimizer through the
/// sparse path would silently change training results.
/// </summary>
public class SparseEmbeddingGradientTests
{
    private static readonly CpuEngine Engine = new();

    /// <summary>
    /// Reference shape: a 3-token sequence into a vocab-of-8 embedding-table; per-position
    /// gradients are linearly indexed so the expected scatter-sum is easy to hand-check.
    /// </summary>
    private static (Tensor<float> Values, Tensor<long> Indices) BuildContributions(long[] flatIndices, int embeddingDim)
    {
        int numIndices = flatIndices.Length;
        var values = new Tensor<float>(new[] { numIndices, embeddingDim });
        for (int i = 0; i < numIndices; i++)
        {
            for (int d = 0; d < embeddingDim; d++)
                values[i, d] = (i + 1) * 0.5f + d * 0.1f;
        }
        var indices = new Tensor<long>(flatIndices, new[] { numIndices });
        return (values, indices);
    }

    [Fact]
    public void ToDense_MatchesEngineBackward_UniqueIndices()
    {
        const int vocabSize = 8;
        const int embeddingDim = 4;
        var (values, indices) = BuildContributions(new long[] { 2, 5, 0 }, embeddingDim);

        var sparse = new SparseEmbeddingGradient<float>(values, indices, vocabSize, embeddingDim);
        var fromSparse = sparse.ToDense(Engine);
        var fromEngineDirect = Engine.TensorEmbeddingLookupBackward<float, long>(values, indices, vocabSize, embeddingDim);

        Assert.Equal(new[] { vocabSize, embeddingDim }, fromSparse.Shape.ToArray());
        Assert.Equal(new[] { vocabSize, embeddingDim }, fromEngineDirect.Shape.ToArray());
        for (int i = 0; i < vocabSize; i++)
            for (int d = 0; d < embeddingDim; d++)
                Assert.Equal(fromEngineDirect[i, d], fromSparse[i, d]);
    }

    [Fact]
    public void ToDense_AccumulatesDuplicateIndices()
    {
        // Duplicate index 3 — both contributions must accumulate.
        const int vocabSize = 6;
        const int embeddingDim = 3;
        var (values, indices) = BuildContributions(new long[] { 3, 3, 1 }, embeddingDim);

        var sparse = new SparseEmbeddingGradient<float>(values, indices, vocabSize, embeddingDim);
        var dense = sparse.ToDense(Engine);

        // Row 3: contributions from position 0 (value (0+1)*0.5 + d*0.1) and position 1 ((1+1)*0.5 + d*0.1)
        for (int d = 0; d < embeddingDim; d++)
        {
            float expectedRow3 = (1 * 0.5f + d * 0.1f) + (2 * 0.5f + d * 0.1f);
            Assert.Equal(expectedRow3, dense[3, d], precision: 5);
        }
        // Row 1: single contribution from position 2 ((2+1)*0.5 + d*0.1)
        for (int d = 0; d < embeddingDim; d++)
        {
            float expectedRow1 = 3 * 0.5f + d * 0.1f;
            Assert.Equal(expectedRow1, dense[1, d], precision: 5);
        }
        // Untouched rows must be zero.
        foreach (int row in new[] { 0, 2, 4, 5 })
            for (int d = 0; d < embeddingDim; d++)
                Assert.Equal(0f, dense[row, d]);
    }

    [Fact]
    public void Ctor_RejectsRankMismatch()
    {
        var values3D = new Tensor<float>(new[] { 1, 2, 3 });
        var indices = new Tensor<long>(new long[] { 0 }, new[] { 1 });
        Assert.Throws<System.ArgumentException>(() =>
            new SparseEmbeddingGradient<float>(values3D, indices, vocabSize: 4, embeddingDim: 3));
    }

    [Fact]
    public void Ctor_RejectsValuesIndicesLengthMismatch()
    {
        var values = new Tensor<float>(new[] { 3, 4 });
        var indices = new Tensor<long>(new long[] { 0, 1 }, new[] { 2 }); // 2 != 3
        Assert.Throws<System.ArgumentException>(() =>
            new SparseEmbeddingGradient<float>(values, indices, vocabSize: 5, embeddingDim: 4));
    }

    [Fact]
    public void Ctor_RejectsValuesEmbeddingDimMismatch()
    {
        var values = new Tensor<float>(new[] { 2, 4 });
        var indices = new Tensor<long>(new long[] { 0, 1 }, new[] { 2 });
        Assert.Throws<System.ArgumentException>(() =>
            new SparseEmbeddingGradient<float>(values, indices, vocabSize: 5, embeddingDim: 8)); // 4 != 8
    }

    /// <summary>
    /// End-to-end parity: SparseEmbeddingGradient.Build → ToDense should reproduce the exact
    /// dense gradient the existing engine.TensorEmbeddingLookupBackward path emits — i.e. the
    /// sparse representation is a drop-in for the dense one. This is the contract a future
    /// sparse-aware optimizer relies on: every consumer that still wants dense semantics gets
    /// them bit-exact, and the alloc savings come only from skipping the materialization when
    /// the consumer doesn't need it.
    /// </summary>
    [Fact]
    public void Build_ThenToDense_MatchesEngineBackward()
    {
        const int vocabSize = 16;
        const int embeddingDim = 6;
        // Rank-3 [batch=2, seq=4, dim=6] gradOutput shape — the common case from a [batch, seq] token input.
        var gradOutput = new Tensor<float>(new[] { 2, 4, embeddingDim });
        for (int n = 0; n < 2; n++)
            for (int t = 0; t < 4; t++)
                for (int d = 0; d < embeddingDim; d++)
                    gradOutput[n, t, d] = n * 0.13f + t * 0.21f + d * 0.07f;

        // 8 indices = 2 * 4 positions. Includes duplicates within and across rows.
        var intIndices = new Tensor<int>(new[] { 3, 12, 3, 7, 12, 0, 7, 3 }, new[] { 8 });

        var sparse = SparseEmbeddingGradient<float>.Build(gradOutput, intIndices, vocabSize, embeddingDim);
        var fromSparse = sparse.ToDense(Engine);

        // Reference: feed the same flattened grad + widened-long indices to the existing dense engine call.
        var flatGrad = new Tensor<float>(gradOutput.GetDataArray(), new[] { 8, embeddingDim });
        var longIndices = new Tensor<long>(new long[] { 3, 12, 3, 7, 12, 0, 7, 3 }, new[] { 8 });
        var reference = Engine.TensorEmbeddingLookupBackward<float, long>(flatGrad, longIndices, vocabSize, embeddingDim);

        Assert.Equal(new[] { vocabSize, embeddingDim }, fromSparse.Shape.ToArray());
        for (int i = 0; i < vocabSize; i++)
            for (int d = 0; d < embeddingDim; d++)
                Assert.Equal(reference[i, d], fromSparse[i, d], precision: 5);
    }
}
