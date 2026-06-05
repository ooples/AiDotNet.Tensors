using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Compact sparse representation of an embedding-table gradient.
/// </summary>
/// <remarks>
/// <para>
/// Mirrors PyTorch's <c>torch.sparse_coo_tensor</c> for the embedding-lookup
/// backward path. An embedding-lookup forward pass gathers <c>numIndices</c> rows
/// out of a <c>[vocabSize, embeddingDim]</c> table. Its backward is a scatter-add
/// into the same <c>[vocabSize, embeddingDim]</c> shape; for the typical case
/// where <c>numIndices << vocabSize</c> (e.g. a 16-token sequence into a 250 002-
/// vocab BERT/XLM-R embedding), the dense gradient is overwhelmingly zero — every
/// row that wasn't accessed gets zero, then Adam reads + writes that zero for
/// every step.
/// </para>
/// <para>
/// This struct carries only the values that aren't zero, paired with the row
/// indices they belong to. The dense [vocabSize, embeddingDim] tensor is
/// recoverable on demand via <see cref="ToDense(IEngine)"/>, but optimizers and
/// downstream consumers that understand the sparse representation can skip that
/// materialization and instead apply scatter-style updates directly to the
/// accessed rows, cutting both the per-step allocation and the per-step memory
/// traffic from <c>O(vocabSize * embeddingDim)</c> to <c>O(numIndices * embeddingDim)</c>.
/// </para>
/// <para>
/// Duplicate indices are NOT pre-aggregated here. Both the dense
/// materialization path and a sparse-aware optimizer step MUST accumulate
/// (sum) gradients for repeated indices — the embedding lookup forward gathered
/// the same row multiple times, so the backward must scatter-add multiple
/// contributions back into that row.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric element type of the embedding table.</typeparam>
public readonly struct SparseEmbeddingGradient<T>
{
    /// <summary>
    /// Per-position gradient values, shape <c>[numIndices, embeddingDim]</c>.
    /// Row <c>k</c> corresponds to the gradient contribution for the embedding
    /// row whose id is <c>Indices[k]</c>.
    /// </summary>
    public Tensor<T> Values { get; }

    /// <summary>
    /// Row indices into the embedding table, shape <c>[numIndices]</c>.
    /// </summary>
    public Tensor<long> Indices { get; }

    /// <summary>The vocabulary size of the embedding table (axis 0 of the dense gradient).</summary>
    public int VocabSize { get; }

    /// <summary>The embedding dimension (axis 1 of the dense gradient).</summary>
    public int EmbeddingDim { get; }

    /// <summary>Number of accessed positions — i.e., <c>Indices.Length</c>.</summary>
    public int NumIndices => Indices.Length;

    /// <summary>True when no accessed positions are recorded; the dense gradient would be all zeros.</summary>
    public bool IsEmpty => NumIndices == 0;

    /// <summary>
    /// Constructs a sparse gradient from per-position values + their row indices.
    /// </summary>
    /// <param name="values">Shape <c>[numIndices, embeddingDim]</c>.</param>
    /// <param name="indices">Shape <c>[numIndices]</c>, each element in <c>[0, vocabSize)</c>.</param>
    /// <param name="vocabSize">Axis-0 size of the equivalent dense gradient.</param>
    /// <param name="embeddingDim">Axis-1 size of the equivalent dense gradient.</param>
    public SparseEmbeddingGradient(Tensor<T> values, Tensor<long> indices, int vocabSize, int embeddingDim)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (vocabSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabSize), "Vocabulary size must be positive.");
        if (embeddingDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(embeddingDim), "Embedding dimension must be positive.");
        if (values.Rank != 2)
            throw new ArgumentException($"Values must be rank-2 [numIndices, embeddingDim]; got rank {values.Rank}.", nameof(values));
        if (values.Shape[1] != embeddingDim)
            throw new ArgumentException(
                $"Values axis-1 ({values.Shape[1]}) must equal embeddingDim ({embeddingDim}).",
                nameof(values));
        if (indices.Rank != 1)
            throw new ArgumentException($"Indices must be rank-1 [numIndices]; got rank {indices.Rank}.", nameof(indices));
        if (values.Shape[0] != indices.Length)
            throw new ArgumentException(
                $"Values axis-0 ({values.Shape[0]}) must equal indices length ({indices.Length}).",
                nameof(values));

        Values = values;
        Indices = indices;
        VocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
    }

    /// <summary>
    /// Materializes this sparse gradient as the equivalent dense
    /// <c>[VocabSize, EmbeddingDim]</c> tensor. Use only when the consumer cannot
    /// handle the sparse representation directly (the perf win comes from
    /// avoiding this allocation in the first place).
    /// </summary>
    public Tensor<T> ToDense(IEngine engine)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        return engine.TensorEmbeddingLookupBackward<T, long>(Values, Indices, VocabSize, EmbeddingDim);
    }
}
