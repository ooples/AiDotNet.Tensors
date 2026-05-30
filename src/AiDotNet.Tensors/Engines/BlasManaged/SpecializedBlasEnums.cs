using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>Which side of the product the triangular/symmetric operand sits on.</summary>
public enum Side
{
    /// <summary>op(A) is on the left: op(A)·X = B  (TRSM) or C = A·B (SYMM).</summary>
    Left,
    /// <summary>op(A) is on the right: X·op(A) = B (TRSM) or C = B·A (SYMM).</summary>
    Right,
}

/// <summary>Which triangle of a triangular/symmetric matrix is referenced.</summary>
public enum Uplo
{
    /// <summary>Upper triangle (including diagonal) holds the data.</summary>
    Upper,
    /// <summary>Lower triangle (including diagonal) holds the data.</summary>
    Lower,
}

/// <summary>Whether a triangular matrix has an implicit unit diagonal.</summary>
public enum Diag
{
    /// <summary>Diagonal entries are read from the matrix.</summary>
    NonUnit,
    /// <summary>Diagonal entries are assumed to be 1 and not read.</summary>
    Unit,
}

/// <summary>Storage format of a <see cref="SparseLayout{T}"/> view.</summary>
public enum SparseLayoutFormat
{
    /// <summary>Compressed Sparse Row: RowPtr length = Rows+1, Indices = column indices.</summary>
    Csr,
    /// <summary>Compressed Sparse Column: RowPtr length = Cols+1, Indices = row indices.</summary>
    Csc,
}

/// <summary>
/// Allocation-free readonly view over a CSR/CSC sparse matrix, decoupled from the
/// heap <see cref="AiDotNet.Tensors.LinearAlgebra.SparseTensor{T}"/> class so
/// <see cref="BlasManaged.SpMM{T}"/> can be called from hot paths without boxing.
/// For CSR, <see cref="Pointers"/> is the row-pointer array (length Rows+1) and
/// <see cref="Indices"/> holds column indices. For CSC the roles transpose.
/// </summary>
public readonly ref struct SparseLayout<T> where T : unmanaged
{
    /// <summary>Number of rows in the logical (dense-equivalent) matrix.</summary>
    public int Rows { get; init; }
    /// <summary>Number of columns in the logical (dense-equivalent) matrix.</summary>
    public int Cols { get; init; }
    /// <summary>Row pointers (CSR) or column pointers (CSC).</summary>
    public ReadOnlySpan<int> Pointers { get; init; }
    /// <summary>Column indices (CSR) or row indices (CSC).</summary>
    public ReadOnlySpan<int> Indices { get; init; }
    /// <summary>Nonzero values, parallel to <see cref="Indices"/>.</summary>
    public ReadOnlySpan<T> Values { get; init; }
    /// <summary>Which compressed format the spans encode.</summary>
    public SparseLayoutFormat Format { get; init; }
}
