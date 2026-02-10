using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class FlashAttentionBiasTests
{
    private readonly CpuEngine _engine = new();

    // Test dimensions
    private const int Batch = 1;
    private const int Heads = 2;
    private const int SeqQ = 4;
    private const int SeqK = 4;
    private const int HeadDim = 8;

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<float>(shape);
        int totalSize = 1;
        foreach (int dim in shape) totalSize *= dim;
        for (int i = 0; i < totalSize; i++)
            tensor.SetFlat(i, (float)(rng.NextDouble() * 2 - 1));
        return tensor;
    }

    [Fact]
    public void FlashAttention_WithoutBias_Succeeds()
    {
        var query = CreateRandomTensor([Batch, Heads, SeqQ, HeadDim], 42);
        var key = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 43);
        var value = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 44);

        var result = _engine.FlashAttention(query, key, value, null, false, out var stats);

        Assert.Equal(new[] { Batch, Heads, SeqQ, HeadDim }, result.Shape);
        Assert.Equal(new[] { Batch, Heads, SeqQ }, stats.Shape);
    }

    [Fact]
    public void FlashAttention_With4DBias_ChangesOutput()
    {
        var query = CreateRandomTensor([Batch, Heads, SeqQ, HeadDim], 42);
        var key = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 43);
        var value = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 44);

        // Run without bias
        var resultNoBias = _engine.FlashAttention(query, key, value, null, false, out _);

        // Run with 4D bias [batch, heads, seqQ, seqK]
        var bias = CreateRandomTensor([Batch, Heads, SeqQ, SeqK], 99);
        var resultWithBias = _engine.FlashAttention(query, key, value, null, false, out _, bias);

        // Output should differ when bias is applied
        Assert.Equal(new[] { Batch, Heads, SeqQ, HeadDim }, resultWithBias.Shape);
        bool anyDifferent = false;
        int totalSize = Batch * Heads * SeqQ * HeadDim;
        for (int i = 0; i < totalSize; i++)
        {
            if (Math.Abs(resultNoBias.GetFlat(i) - resultWithBias.GetFlat(i)) > 1e-6f)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Attention output should change when bias is applied");
    }

    [Fact]
    public void FlashAttention_With3DBias_BroadcastsAcrossBatch()
    {
        int batch = 2;
        var query = CreateRandomTensor([batch, Heads, SeqQ, HeadDim], 42);
        var key = CreateRandomTensor([batch, Heads, SeqK, HeadDim], 43);
        var value = CreateRandomTensor([batch, Heads, SeqK, HeadDim], 44);

        // 3D bias [heads, seqQ, seqK] — broadcast across batch
        var bias = CreateRandomTensor([Heads, SeqQ, SeqK], 99);
        var result = _engine.FlashAttention(query, key, value, null, false, out var stats, bias);

        Assert.Equal(new[] { batch, Heads, SeqQ, HeadDim }, result.Shape);
        Assert.Equal(new[] { batch, Heads, SeqQ }, stats.Shape);
    }

    [Fact]
    public void FlashAttention_BiasAppliedAfterScaleBeforeSoftmax()
    {
        // Create a scenario where bias should shift attention
        var query = new Tensor<float>(new[] { 1, 1, 2, 2 });
        var key = new Tensor<float>(new[] { 1, 1, 2, 2 });
        var value = new Tensor<float>(new[] { 1, 1, 2, 2 });

        // Q = [[1,0],[0,1]], K = [[1,0],[0,1]], V = [[1,0],[0,1]]
        query.SetFlat(0, 1f); query.SetFlat(1, 0f); query.SetFlat(2, 0f); query.SetFlat(3, 1f);
        key.SetFlat(0, 1f); key.SetFlat(1, 0f); key.SetFlat(2, 0f); key.SetFlat(3, 1f);
        value.SetFlat(0, 1f); value.SetFlat(1, 0f); value.SetFlat(2, 0f); value.SetFlat(3, 1f);

        // Large negative bias on diagonal — should shift attention to off-diagonal
        var bias = new Tensor<float>(new[] { 1, 1, 2, 2 });
        bias.SetFlat(0, -100f); // suppress (0,0)
        bias.SetFlat(1, 0f);
        bias.SetFlat(2, 0f);
        bias.SetFlat(3, -100f); // suppress (1,1)

        var result = _engine.FlashAttention(query, key, value, null, false, out _, bias);

        // With diagonal suppressed, Q[0] should attend mostly to K[1] -> V[1] = [0,1]
        Assert.True(result.GetFlat(0) < 0.1f, "Q[0] output dim 0 should be near 0 (attending to V[1])");
        Assert.True(result.GetFlat(1) > 0.9f, "Q[0] output dim 1 should be near 1 (attending to V[1])");
    }

    [Fact]
    public void FlashAttention_InvalidBiasRank_Throws()
    {
        var query = CreateRandomTensor([Batch, Heads, SeqQ, HeadDim], 42);
        var key = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 43);
        var value = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 44);

        // 2D bias — invalid rank
        var badBias = CreateRandomTensor([SeqQ, SeqK], 99);

        Assert.Throws<ArgumentException>(() =>
            _engine.FlashAttention(query, key, value, null, false, out _, badBias));
    }

    [Fact]
    public void FlashAttention_MismatchedBiasDimensions_Throws()
    {
        var query = CreateRandomTensor([Batch, Heads, SeqQ, HeadDim], 42);
        var key = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 43);
        var value = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 44);

        // 4D bias with wrong seqK dimension
        var badBias = CreateRandomTensor([Batch, Heads, SeqQ, SeqK + 1], 99);

        Assert.Throws<ArgumentException>(() =>
            _engine.FlashAttention(query, key, value, null, false, out _, badBias));
    }

    [Fact]
    public void FlashAttention_Mismatched3DBiasDimensions_Throws()
    {
        var query = CreateRandomTensor([Batch, Heads, SeqQ, HeadDim], 42);
        var key = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 43);
        var value = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 44);

        // 3D bias with wrong heads dimension
        var badBias = CreateRandomTensor([Heads + 1, SeqQ, SeqK], 99);

        Assert.Throws<ArgumentException>(() =>
            _engine.FlashAttention(query, key, value, null, false, out _, badBias));
    }

    [Fact]
    public void FlashAttentionBackward_WithBias_MatchesRecomputation()
    {
        var query = CreateRandomTensor([Batch, Heads, SeqQ, HeadDim], 42);
        var key = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 43);
        var value = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 44);
        var bias = CreateRandomTensor([Batch, Heads, SeqQ, SeqK], 99);

        // Forward with bias
        var output = _engine.FlashAttention(query, key, value, null, false, out var stats, bias);
        double scale = 1.0 / Math.Sqrt(HeadDim);

        // Backward with bias — should use bias for recomputation
        var gradOutput = CreateRandomTensor([Batch, Heads, SeqQ, HeadDim], 55);
        var result = _engine.FlashAttentionBackward(
            gradOutput, query, key, value, output, stats,
            scale, false, out var gradQ, out var gradK, out var gradV, bias);

        // Gradients should have correct shapes
        Assert.Equal(new[] { Batch, Heads, SeqQ, HeadDim }, gradQ.Shape);
        Assert.Equal(new[] { Batch, Heads, SeqK, HeadDim }, gradK.Shape);
        Assert.Equal(new[] { Batch, Heads, SeqK, HeadDim }, gradV.Shape);

        // Gradients should not be all zeros
        bool gradQNonZero = false;
        bool gradKNonZero = false;
        bool gradVNonZero = false;
        int qSize = Batch * Heads * SeqQ * HeadDim;
        int kSize = Batch * Heads * SeqK * HeadDim;
        for (int i = 0; i < qSize; i++)
            if (Math.Abs(gradQ.GetFlat(i)) > 1e-10f) { gradQNonZero = true; break; }
        for (int i = 0; i < kSize; i++)
            if (Math.Abs(gradK.GetFlat(i)) > 1e-10f) { gradKNonZero = true; break; }
        for (int i = 0; i < kSize; i++)
            if (Math.Abs(gradV.GetFlat(i)) > 1e-10f) { gradVNonZero = true; break; }

        Assert.True(gradQNonZero, "gradQ should not be all zeros");
        Assert.True(gradKNonZero, "gradK should not be all zeros");
        Assert.True(gradVNonZero, "gradV should not be all zeros");
    }

    [Fact]
    public void FlashAttentionBackward_BiasChangesGradients()
    {
        var query = CreateRandomTensor([Batch, Heads, SeqQ, HeadDim], 42);
        var key = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 43);
        var value = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 44);
        var bias = CreateRandomTensor([Batch, Heads, SeqQ, SeqK], 99);
        var gradOutput = CreateRandomTensor([Batch, Heads, SeqQ, HeadDim], 55);
        double scale = 1.0 / Math.Sqrt(HeadDim);

        // Forward and backward without bias
        var outputNoBias = _engine.FlashAttention(query, key, value, null, false, out var statsNoBias);
        _engine.FlashAttentionBackward(
            gradOutput, query, key, value, outputNoBias, statsNoBias,
            scale, false, out var gradQNoBias, out _, out _);

        // Forward and backward with bias
        var outputBias = _engine.FlashAttention(query, key, value, null, false, out var statsBias, bias);
        _engine.FlashAttentionBackward(
            gradOutput, query, key, value, outputBias, statsBias,
            scale, false, out var gradQBias, out _, out _, bias);

        // Gradients should differ
        bool anyDifferent = false;
        int totalSize = Batch * Heads * SeqQ * HeadDim;
        for (int i = 0; i < totalSize; i++)
        {
            if (Math.Abs(gradQNoBias.GetFlat(i) - gradQBias.GetFlat(i)) > 1e-6f)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Backward gradients should differ when bias is used");
    }

    [Fact]
    public void FlashAttentionBackward_InvalidBiasRank_Throws()
    {
        var query = CreateRandomTensor([Batch, Heads, SeqQ, HeadDim], 42);
        var key = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 43);
        var value = CreateRandomTensor([Batch, Heads, SeqK, HeadDim], 44);
        var output = _engine.FlashAttention(query, key, value, null, false, out var stats);
        double scale = 1.0 / Math.Sqrt(HeadDim);
        var gradOutput = CreateRandomTensor([Batch, Heads, SeqQ, HeadDim], 55);

        // 5D bias — invalid rank
        var badBias = CreateRandomTensor([1, Batch, Heads, SeqQ, SeqK], 99);

        Assert.Throws<ArgumentException>(() =>
            _engine.FlashAttentionBackward(
                gradOutput, query, key, value, output, stats,
                scale, false, out _, out _, out _, badBias));
    }

    [Fact]
    public void FlashAttention_With3DBias_And4DBias_ProduceSameResultForSingleBatch()
    {
        // For batch=1, 3D bias [heads, seqQ, seqK] and 4D bias [1, heads, seqQ, seqK]
        // should produce identical results
        var query = CreateRandomTensor([1, Heads, SeqQ, HeadDim], 42);
        var key = CreateRandomTensor([1, Heads, SeqK, HeadDim], 43);
        var value = CreateRandomTensor([1, Heads, SeqK, HeadDim], 44);

        var bias3D = CreateRandomTensor([Heads, SeqQ, SeqK], 99);
        var bias4D = CreateRandomTensor([1, Heads, SeqQ, SeqK], 99); // Same seed = same values

        var result3D = _engine.FlashAttention(query, key, value, null, false, out _, bias3D);
        var result4D = _engine.FlashAttention(query, key, value, null, false, out _, bias4D);

        int totalSize = Heads * SeqQ * HeadDim;
        for (int i = 0; i < totalSize; i++)
        {
            Assert.Equal(result3D.GetFlat(i), result4D.GetFlat(i), 5);
        }
    }
}
