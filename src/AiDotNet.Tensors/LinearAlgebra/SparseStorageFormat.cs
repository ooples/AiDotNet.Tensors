namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Supported sparse storage formats for 2D tensors. Mirrors PyTorch's
/// <c>torch.sparse</c> layout taxonomy.
/// </summary>
public enum SparseStorageFormat
{
    /// <summary>Coordinate format: parallel <c>(row, col, value)</c> arrays.
    /// Best for incremental construction; convert to CSR/CSC for ops.</summary>
    Coo,

    /// <summary>Compressed sparse row: <c>row_ptr</c> + <c>col_idx</c> + <c>values</c>.
    /// Optimal for row-traversal ops (SpMV, sparse × dense).</summary>
    Csr,

    /// <summary>Compressed sparse column: <c>col_ptr</c> + <c>row_idx</c> + <c>values</c>.
    /// Optimal for column-traversal ops (transpose-multiply, sparse-LHS solves).</summary>
    Csc,

    /// <summary>Block compressed sparse row: like CSR but each non-zero is a
    /// <c>BlockRowSize × BlockColSize</c> dense block. Common for LLM weights
    /// after magnitude pruning at block sizes 2/4/8/16.</summary>
    Bsr,

    /// <summary>Block compressed sparse column: column-block analog of BSR.</summary>
    Bsc,
}
