// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>
/// Multi-dimensional grid of ranks that <see cref="DTensor{T}"/> shards
/// over. A 1D mesh is just a flat list of WorldSize ranks; a 2D mesh
/// might be 4×2 (4 data-parallel × 2 tensor-parallel groups). Mirrors
/// <c>torch.distributed.device_mesh.DeviceMesh</c>.
/// </summary>
public sealed class DeviceMesh
{
    private readonly int[] _shape;
    private readonly int[] _ranks; // flattened row-major
    private readonly IProcessGroup _world;
    private readonly Dictionary<int, IProcessGroup> _dimGroups = new();

    /// <summary>Mesh shape — e.g. <c>[4, 2]</c> for a 4×2 grid.</summary>
    public IReadOnlyList<int> Shape => _shape;

    /// <summary>Total ranks in the mesh. Must equal <c>_world.WorldSize</c>.</summary>
    public int Size { get; }

    /// <summary>The underlying world process group.</summary>
    public IProcessGroup World => _world;

    /// <summary>Constructs a mesh with the given shape over
    /// <paramref name="world"/>. <paramref name="shape"/> must multiply
    /// to <c>world.WorldSize</c>.</summary>
    public DeviceMesh(IProcessGroup world, int[] shape)
    {
        _world = world ?? throw new ArgumentNullException(nameof(world));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        _shape = (int[])shape.Clone();
        long total = 1;
        foreach (var d in shape)
        {
            if (d < 1) throw new ArgumentException("Mesh dimensions must be positive.", nameof(shape));
            total *= d;
        }
        if (total != world.WorldSize)
            throw new ArgumentException(
                $"Mesh shape product {total} must equal world size {world.WorldSize}.", nameof(shape));
        Size = (int)total;

        _ranks = new int[Size];
        for (int i = 0; i < Size; i++) _ranks[i] = i;
    }

    /// <summary>Returns the process group containing all ranks that share
    /// every dim except <paramref name="dim"/>. Equivalent to
    /// <c>device_mesh.get_group(mesh_dim)</c>.</summary>
    public IProcessGroup GetDimGroup(int dim)
    {
        if (dim < 0 || dim >= _shape.Length)
            throw new ArgumentOutOfRangeException(nameof(dim));
        if (_dimGroups.TryGetValue(dim, out var cached)) return cached;

        // Color = composite of every coordinate except dim. Same color = same dim group.
        int color = ComputeDimColor(_world.Rank, dim);
        var group = _world.SplitGroup(color, key: GetCoordinate(_world.Rank, dim));
        _dimGroups[dim] = group;
        return group;
    }

    /// <summary>This rank's coordinates in the mesh.</summary>
    public int[] LocalCoordinates()
    {
        var coords = new int[_shape.Length];
        int r = _world.Rank;
        for (int d = _shape.Length - 1; d >= 0; d--)
        {
            coords[d] = r % _shape[d];
            r /= _shape[d];
        }
        return coords;
    }

    private int GetCoordinate(int rank, int dim)
    {
        for (int d = _shape.Length - 1; d > dim; d--) rank /= _shape[d];
        return rank % _shape[dim];
    }

    private int ComputeDimColor(int rank, int dim)
    {
        // Encode all coordinates except `dim` into a single int.
        int color = 0;
        for (int d = 0; d < _shape.Length; d++)
        {
            if (d == dim) continue;
            color = color * _shape[d] + GetCoordinate(rank, d);
        }
        return color;
    }
}

/// <summary>Placement strategy for one mesh dimension.</summary>
public enum Placement
{
    /// <summary>Tensor is sharded along this dim — each rank holds its slice.</summary>
    Shard,
    /// <summary>Tensor is replicated along this dim — every rank holds the full copy.</summary>
    Replicate,
    /// <summary>Tensor holds a partial reduction — sum / mean / etc. is pending.
    /// Lowered to <see cref="Replicate"/> via an all-reduce when the value is needed.</summary>
    Partial,
}

/// <summary>
/// Distributed tensor view over a <see cref="DeviceMesh"/>. Carries a
/// per-mesh-dim placement spec: each dim is either
/// <see cref="Placement.Shard"/>, <see cref="Placement.Replicate"/>, or
/// <see cref="Placement.Partial"/>. Mirrors
/// <c>torch.distributed.tensor.DTensor</c>.
///
/// <para>Construction: <see cref="FromLocal"/> takes the local shard
/// each rank already has. <see cref="Redistribute"/> changes the
/// placement spec, issuing the matching collectives (all-gather to go
/// from Shard→Replicate, all-reduce to go from Partial→Replicate, etc).</para>
/// </summary>
public sealed class DTensor<T>
{
    /// <summary>Mesh this tensor is sharded over.</summary>
    public DeviceMesh Mesh { get; }

    /// <summary>Per-mesh-dim placement spec. Length =
    /// <c>Mesh.Shape.Count</c>.</summary>
    public IReadOnlyList<Placement> Placements { get; }

    /// <summary>This rank's local shard.</summary>
    public Tensor<T> Local { get; }

    /// <summary>Logical full-tensor shape (the shape the user sees if
    /// they materialize via <c>FullTensor</c>).</summary>
    public int[] LogicalShape { get; }

    private DTensor(DeviceMesh mesh, Tensor<T> local, IReadOnlyList<Placement> placements, int[] logicalShape)
    {
        Mesh = mesh;
        Local = local;
        Placements = placements;
        LogicalShape = logicalShape;
    }

    /// <summary>
    /// Constructs a DTensor from each rank's <paramref name="local"/>
    /// shard with the given <paramref name="placements"/>. The logical
    /// shape is inferred: replicated dims keep <c>local</c>'s size,
    /// sharded dims multiply by the corresponding mesh-dim size.
    /// </summary>
    public static DTensor<T> FromLocal(DeviceMesh mesh, Tensor<T> local, IReadOnlyList<Placement> placements)
    {
        if (placements.Count != mesh.Shape.Count)
            throw new ArgumentException(
                $"Placements length {placements.Count} must equal mesh rank {mesh.Shape.Count}.",
                nameof(placements));
        // Logical shape = local shape, with each mesh-dim's contribution
        // applied. For 1D mesh on a 2D tensor sharded along axis 0:
        //   local: [N/W, K], mesh: [W], placements: [Shard(0)] → logical [N, K].
        // We use a simple convention: Shard on mesh dim D scales the
        // tensor's dim D by mesh.Shape[D]. Replicate / Partial leave it
        // unchanged. This matches torch.distributed's "shard placement
        // implies linear scaling along that mesh-dim" semantics.
        var logical = (int[])local._shape.Clone();
        for (int d = 0; d < placements.Count; d++)
        {
            if (placements[d] == Placement.Shard && d < logical.Length)
                logical[d] *= mesh.Shape[d];
        }
        return new DTensor<T>(mesh, local, placements, logical);
    }

    /// <summary>
    /// Issues the collectives required to change the placement spec to
    /// <paramref name="newPlacements"/>. Common transitions:
    /// <list type="bullet">
    ///   <item><c>Shard → Replicate</c>: all-gather along the mesh dim.</item>
    ///   <item><c>Partial → Replicate</c>: all-reduce along the mesh dim.</item>
    ///   <item><c>Partial → Shard</c>: reduce-scatter.</item>
    /// </list>
    /// </summary>
    public DTensor<T> Redistribute(IReadOnlyList<Placement> newPlacements)
    {
        if (newPlacements.Count != Mesh.Shape.Count)
            throw new ArgumentException("newPlacements length mismatch.", nameof(newPlacements));

        var current = Local;
        for (int d = 0; d < Placements.Count; d++)
        {
            var oldP = Placements[d];
            var newP = newPlacements[d];
            if (oldP == newP) continue;

            var dimGroup = Mesh.GetDimGroup(d);
            // The convention from FromLocal: a Shard placement on mesh
            // dim D shards the tensor's axis D. Thread `d` (the mesh
            // dim) through every redistribute helper as the tensor
            // axis to operate on, so multi-dim mesh sharding works.
            current = (oldP, newP) switch
            {
                (Placement.Partial, Placement.Replicate) => AllReduceCopy(current, dimGroup),
                (Placement.Partial, Placement.Shard)     => PartialToShard(current, dimGroup, d, LogicalShape[d]),
                (Placement.Shard, Placement.Replicate)   => ShardToReplicate(current, dimGroup, d, LogicalShape[d]),
                (Placement.Replicate, Placement.Shard)   => ReplicateToShard(current, dimGroup, d),
                _ => throw new NotSupportedException(
                    $"Redistribute {oldP}→{newP} on mesh dim {d} is not implemented."),
            };
        }
        return new DTensor<T>(Mesh, current, newPlacements, LogicalShape);
    }

    private static Tensor<T> AllReduceCopy(Tensor<T> t, IProcessGroup g)
    {
        var copy = new Tensor<T>((int[])t._shape.Clone());
        t.AsSpan().CopyTo(copy.AsWritableSpan());
        g.AllReduce(copy, ReduceOp.Sum);
        return copy;
    }

    private static Tensor<T> PartialToShard(Tensor<T> t, IProcessGroup g, int axis, int logicalAxisSize)
    {
        _ = logicalAxisSize; // logical size is preserved across the per-rank list shape.
        // Reduce-scatter along the requested tensor axis. Each rank
        // contributes the full partial; result is its slice of the
        // axis-aligned sum. We slice along `axis` to build the per-
        // rank input list, then ReduceScatter produces the rank's
        // chunk.
        if (axis < 0 || axis >= t.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis),
                $"PartialToShard axis {axis} out of range [0, {t.Rank}).");
        int axisLen = t._shape[axis];
        int chunkAlongAxis = (axisLen + g.WorldSize - 1) / g.WorldSize;

        var perRank = new List<Tensor<T>>(g.WorldSize);
        var perRankShape = (int[])t._shape.Clone();
        perRankShape[axis] = chunkAlongAxis;
        for (int r = 0; r < g.WorldSize; r++)
        {
            var c = new Tensor<T>(perRankShape);
            CopyAxisSlice(t, c, axis, srcStart: r * chunkAlongAxis, srcLen: Math.Min(chunkAlongAxis, Math.Max(0, axisLen - r * chunkAlongAxis)));
            perRank.Add(c);
        }
        var output = new Tensor<T>(perRankShape);
        g.ReduceScatter(perRank, output, ReduceOp.Sum);
        return output;
    }

    private static Tensor<T> ShardToReplicate(Tensor<T> t, IProcessGroup g, int axis, int logicalAxisSize)
    {
        if (axis < 0 || axis >= t.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis),
                $"ShardToReplicate axis {axis} out of range [0, {t.Rank}).");
        var perRank = new List<Tensor<T>>(g.WorldSize);
        for (int r = 0; r < g.WorldSize; r++) perRank.Add(new Tensor<T>((int[])t._shape.Clone()));
        g.AllGather(t, perRank);
        // Concatenate along the sharding axis. Output uses the LOGICAL
        // axis size (pre-padding) — every shard contributes ceil-chunk
        // padded width on the wire, but the tail rank's padding is
        // dropped here so the result matches the original unsharded
        // shape exactly.
        var fullShape = (int[])t._shape.Clone();
        fullShape[axis] = logicalAxisSize;
        var full = new Tensor<T>(fullShape);
        int chunk = t._shape[axis];
        for (int r = 0; r < g.WorldSize; r++)
        {
            int dstStart = r * chunk;
            int copyLen = Math.Min(chunk, Math.Max(0, logicalAxisSize - dstStart));
            if (copyLen <= 0) break;
            CopyAxisSlice(perRank[r], full, axis, dstStart: dstStart, srcLen: copyLen);
        }
        return full;
    }

    private static Tensor<T> ReplicateToShard(Tensor<T> t, IProcessGroup g, int axis)
    {
        if (axis < 0 || axis >= t.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis),
                $"ReplicateToShard axis {axis} out of range [0, {t.Rank}).");
        int total = t._shape[axis];
        int chunk = (total + g.WorldSize - 1) / g.WorldSize;
        int from = g.Rank * chunk;
        int actual = Math.Min(chunk, Math.Max(0, total - from));
        var shape = (int[])t._shape.Clone();
        shape[axis] = chunk;
        var shard = new Tensor<T>(shape);
        if (actual > 0)
            CopyAxisSlice(t, shard, axis, srcStart: from, srcLen: actual);
        return shard;
    }

    /// <summary>
    /// Copies a contiguous range along <paramref name="axis"/> from
    /// <paramref name="src"/> into <paramref name="dst"/>. The non-axis
    /// dimensions must match between source and destination.
    /// <paramref name="srcStart"/> / <paramref name="dstStart"/> default
    /// to 0; <paramref name="srcLen"/> defaults to dst's axis length.
    /// </summary>
    private static void CopyAxisSlice(Tensor<T> src, Tensor<T> dst, int axis,
        int srcStart = 0, int dstStart = 0, int srcLen = -1)
    {
        // Compute outer/inner strides so we can copy "lanes" of the
        // axis without having to materialise per-axis indices.
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) outer *= src._shape[i];
        for (int i = axis + 1; i < src.Rank; i++) inner *= src._shape[i];
        int srcAxis = src._shape[axis];
        int dstAxis = dst._shape[axis];
        int copyLen = srcLen < 0 ? dstAxis : srcLen;

        var s = src.AsSpan();
        var d = dst.AsWritableSpan();
        for (int o = 0; o < outer; o++)
        {
            int srcRowStart = (o * srcAxis + srcStart) * inner;
            int dstRowStart = (o * dstAxis + dstStart) * inner;
            int copyElems = copyLen * inner;
            s.Slice(srcRowStart, copyElems).CopyTo(d.Slice(dstRowStart, copyElems));
        }
    }
}
