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
        _shape = (int[])shape.Clone() ?? throw new ArgumentNullException(nameof(shape));
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
            current = (oldP, newP) switch
            {
                (Placement.Partial, Placement.Replicate) => AllReduceCopy(current, dimGroup),
                (Placement.Partial, Placement.Shard)     => PartialToShard(current, dimGroup),
                (Placement.Shard, Placement.Replicate)   => ShardToReplicate(current, dimGroup),
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

    private static Tensor<T> PartialToShard(Tensor<T> t, IProcessGroup g)
    {
        // Reduce-scatter: each rank provides its full partial; result is the rank's slice of the sum.
        int chunk = (t.Length + g.WorldSize - 1) / g.WorldSize;
        var perRank = new List<Tensor<T>>(g.WorldSize);
        var src = t.AsSpan();
        for (int r = 0; r < g.WorldSize; r++)
        {
            var c = new Tensor<T>(new[] { chunk });
            int from = r * chunk;
            int copyLen = Math.Min(chunk, Math.Max(0, src.Length - from));
            if (copyLen > 0) src.Slice(from, copyLen).CopyTo(c.AsWritableSpan().Slice(0, copyLen));
            perRank.Add(c);
        }
        var output = new Tensor<T>(new[] { chunk });
        g.ReduceScatter(perRank, output, ReduceOp.Sum);
        return output;
    }

    private static Tensor<T> ShardToReplicate(Tensor<T> t, IProcessGroup g)
    {
        // All-gather: every rank ends up with concatenation of all shards.
        var perRank = new List<Tensor<T>>(g.WorldSize);
        for (int r = 0; r < g.WorldSize; r++) perRank.Add(new Tensor<T>((int[])t._shape.Clone()));
        g.AllGather(t, perRank);
        // Concatenate along axis 0 (the canonical sharding axis).
        var fullShape = (int[])t._shape.Clone();
        fullShape[0] *= g.WorldSize;
        var full = new Tensor<T>(fullShape);
        var dst = full.AsWritableSpan();
        int offset = 0;
        for (int r = 0; r < g.WorldSize; r++)
        {
            var src = perRank[r].AsSpan();
            src.CopyTo(dst.Slice(offset, src.Length));
            offset += src.Length;
        }
        return full;
    }

    private static Tensor<T> ReplicateToShard(Tensor<T> t, IProcessGroup g, int meshDim)
    {
        _ = meshDim;
        // Take this rank's slice along axis 0.
        int total = t._shape[0];
        int chunk = (total + g.WorldSize - 1) / g.WorldSize;
        int from = g.Rank * chunk;
        int actual = Math.Min(chunk, Math.Max(0, total - from));
        var shape = (int[])t._shape.Clone();
        shape[0] = chunk;
        var shard = new Tensor<T>(shape);
        if (actual > 0)
        {
            int rowSize = t.Length / total;
            t.AsSpan().Slice(from * rowSize, actual * rowSize)
                .CopyTo(shard.AsWritableSpan().Slice(0, actual * rowSize));
        }
        return shard;
    }
}
