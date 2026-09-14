// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>The storage representation produced for one completed source gradient.</summary>
public enum StreamingSourceGradientKind
{
    /// <summary>A conventional dense tensor.</summary>
    Dense = 0,

    /// <summary>One or more indexed embedding-table contributions.</summary>
    SparseEmbedding = 1,

    /// <summary>Both dense and indexed contributions target the same source.</summary>
    DenseAndSparseEmbedding = 2,
}

/// <summary>
/// Typed, ephemeral gradient payload emitted by streaming backward.
/// </summary>
/// <remarks>
/// The payload and every tensor it references belong to the active backward pass. Consumers must
/// synchronously copy, journal, or apply them before returning from the callback.
/// </remarks>
public readonly struct StreamingSourceGradient<T>
{
    private readonly Tensor<T>? _dense;
    private readonly IReadOnlyList<SparseEmbeddingGradient<T>>? _sparseEmbedding;

    private StreamingSourceGradient(
        StreamingSourceGradientKind kind,
        Tensor<T>? dense,
        IReadOnlyList<SparseEmbeddingGradient<T>>? sparseEmbedding)
    {
        Kind = kind;
        _dense = dense;
        _sparseEmbedding = sparseEmbedding;
    }

    /// <summary>The payload representation.</summary>
    public StreamingSourceGradientKind Kind { get; }

    /// <summary>Whether this payload contains a dense contribution.</summary>
    public bool HasDense => Kind is StreamingSourceGradientKind.Dense
        or StreamingSourceGradientKind.DenseAndSparseEmbedding;

    /// <summary>Whether this payload contains indexed embedding contributions.</summary>
    public bool HasSparseEmbedding => Kind is StreamingSourceGradientKind.SparseEmbedding
        or StreamingSourceGradientKind.DenseAndSparseEmbedding;

    /// <summary>The dense contribution.</summary>
    public Tensor<T> Dense => HasDense && _dense is Tensor<T> dense
        ? dense
        : throw new InvalidOperationException("This streaming gradient has no dense contribution.");

    /// <summary>The indexed embedding contributions.</summary>
    public IReadOnlyList<SparseEmbeddingGradient<T>> SparseEmbedding =>
        HasSparseEmbedding && _sparseEmbedding is IReadOnlyList<SparseEmbeddingGradient<T>> sparse
            ? sparse
            : throw new InvalidOperationException(
                "This streaming gradient has no sparse embedding contribution.");

    /// <summary>Creates a validated payload from one or both supported representations.</summary>
    public static StreamingSourceGradient<T> Create(
        Tensor<T>? dense,
        IReadOnlyList<SparseEmbeddingGradient<T>>? sparseEmbedding)
    {
        bool hasDense = dense is not null;
        bool hasSparse = sparseEmbedding is { Count: > 0 };
        if (!hasDense && !hasSparse)
            throw new ArgumentException("A streaming gradient must contain at least one contribution.");

        StreamingSourceGradientKind kind = hasDense
            ? hasSparse
                ? StreamingSourceGradientKind.DenseAndSparseEmbedding
                : StreamingSourceGradientKind.Dense
            : StreamingSourceGradientKind.SparseEmbedding;
        return new StreamingSourceGradient<T>(kind, dense, sparseEmbedding);
    }
}
