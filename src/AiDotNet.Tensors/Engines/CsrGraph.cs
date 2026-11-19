using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The sparsity pattern of a square graph adjacency in compressed-sparse-row form, with its transpose (compressed
/// sparse column) alongside: the structure <see cref="IEngine.GraphAttentionCsr{T}"/> attends over.
/// </summary>
/// <remarks>
/// <para>
/// Row <c>i</c> lists the nodes <c>j</c> that node <c>i</c> attends to: <c>ColumnIndices[RowPointers[i] ..
/// RowPointers[i + 1])</c>, in the order supplied. The column form lists, for each node <c>j</c>, the edges that read it, so a
/// backward pass can gather into <c>j</c> without atomics. Only the pattern is kept; edge values are not.
/// </para>
/// <para>
/// The constructor copies the topology. The public array properties return copies, so an engine can cache
/// device buffers for the lifetime of the graph.
/// </para>
/// </remarks>
public sealed class CsrGraph
{
    /// <summary>Creates a graph from a CSR pattern over <paramref name="nodeCount"/> nodes.</summary>
    /// <param name="nodeCount">The number of nodes (rows and columns).</param>
    /// <param name="rowPointers">The <c>nodeCount + 1</c> row offsets into <paramref name="columnIndices"/>.</param>
    /// <param name="columnIndices">The column (neighbour) of every edge, row by row.</param>
    public CsrGraph(int nodeCount, int[] rowPointers, int[] columnIndices)
    {
        if (nodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount), "A graph needs at least one node.");
        if (rowPointers is null) throw new ArgumentNullException(nameof(rowPointers));
        if (columnIndices is null) throw new ArgumentNullException(nameof(columnIndices));
        rowPointers = (int[])rowPointers.Clone();
        columnIndices = (int[])columnIndices.Clone();
        if (rowPointers.Length != nodeCount + 1)
            throw new ArgumentException($"rowPointers must have nodeCount + 1 = {nodeCount + 1} entries; got {rowPointers.Length}.", nameof(rowPointers));
        if (rowPointers[0] != 0 || rowPointers[nodeCount] != columnIndices.Length)
            throw new ArgumentException("rowPointers must start at 0 and end at the edge count.", nameof(rowPointers));
        for (int i = 0; i < nodeCount; i++)
            if (rowPointers[i + 1] < rowPointers[i])
                throw new ArgumentException("rowPointers must be non-decreasing.", nameof(rowPointers));
        for (int e = 0; e < columnIndices.Length; e++)
            if ((uint)columnIndices[e] >= (uint)nodeCount)
                throw new ArgumentException($"Column index {columnIndices[e]} at edge {e} is outside [0, {nodeCount}).", nameof(columnIndices));
        var seen = new int[nodeCount];
        for (int row = 0; row < nodeCount; row++)
        {
            for (int e = rowPointers[row]; e < rowPointers[row + 1]; e++)
            {
                if (seen[columnIndices[e]] == row + 1)
                    throw new ArgumentException($"Duplicate column {columnIndices[e]} in row {row}.", nameof(columnIndices));
                seen[columnIndices[e]] = row + 1;
            }
        }

        NodeCount = nodeCount;
        RowOffsets = rowPointers;
        Columns = columnIndices;

        // Transpose: for every column j, the (row, edge) pairs that read it, in ascending row order.
        int edges = columnIndices.Length;
        var colPtr = new int[nodeCount + 1];
        for (int e = 0; e < edges; e++) colPtr[columnIndices[e] + 1]++;
        for (int j = 0; j < nodeCount; j++) colPtr[j + 1] += colPtr[j];
        var fill = (int[])colPtr.Clone();
        var cscRows = new int[edges];
        var cscEdges = new int[edges];
        var edgeRows = new int[edges];
        for (int i = 0; i < nodeCount; i++)
        {
            for (int e = rowPointers[i]; e < rowPointers[i + 1]; e++)
            {
                int slot = fill[columnIndices[e]]++;
                cscRows[slot] = i;
                cscEdges[slot] = e;
                edgeRows[e] = i;
            }
        }

        ColumnPointers = colPtr;
        TransposeRowIndices = cscRows;
        TransposeEdgeIndices = cscEdges;
        EdgeRows = edgeRows;
        bool everyRowHasEdge = true;
        for (int i = 0; i < nodeCount && everyRowHasEdge; i++)
            everyRowHasEdge = rowPointers[i + 1] > rowPointers[i];
        EveryRowHasEdge = everyRowHasEdge;
    }

    /// <summary>The number of nodes.</summary>
    public int NodeCount { get; }

    /// <summary>The number of edges (stored entries).</summary>
    public int EdgeCount => Columns.Length;

    /// <summary>The <see cref="NodeCount"/> + 1 row offsets into <see cref="ColumnIndices"/>.</summary>
    public int[] RowPointers => (int[])RowOffsets.Clone();
    internal int[] RowOffsets { get; }

    /// <summary>The neighbour (column) of every edge, row by row.</summary>
    public int[] ColumnIndices => (int[])Columns.Clone();
    internal int[] Columns { get; }

    /// <summary>Whether every node has at least one edge (an empty row has no softmax to normalize over).</summary>
    public bool EveryRowHasEdge { get; }

    internal int[] ColumnPointers { get; }
    internal int[] TransposeRowIndices { get; }
    internal int[] TransposeEdgeIndices { get; }
    internal int[] EdgeRows { get; }

    /// <summary>
    /// The nonzero pattern of a dense rank-2 square adjacency matrix, or null when it is not rank 2 and square.
    /// </summary>
    /// <param name="adjacency">A <c>[nodes, nodes]</c> matrix; every entry not equal to zero is an edge.</param>
    public static CsrGraph? FromDenseAdjacency<T>(Tensor<T> adjacency)
    {
        if (adjacency is null) throw new ArgumentNullException(nameof(adjacency));
        if (adjacency.Rank != 2 || adjacency.Shape[0] != adjacency.Shape[1] || adjacency.Shape[0] <= 0) return null;
        int n = adjacency.Shape[0];
        var rowPtr = new int[n + 1];
        var cols = new List<int>();
        var comparer = EqualityComparer<T>.Default;
        using (var lease = adjacency.Lease())
        {
            var span = lease.Span;
            for (int i = 0; i < n; i++)
            {
                var row = span.Slice(i * n, n);
                for (int j = 0; j < n; j++)
                    if (!comparer.Equals(row[j], default!)) cols.Add(j);
                rowPtr[i + 1] = cols.Count;
            }
        }

        return new CsrGraph(n, rowPtr, cols.ToArray());
    }
}
