// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>
/// Process-group backend that runs every "rank" as a managed thread inside
/// the current process. Used for unit tests + single-machine multi-rank
/// debugging — every collective op resolves through shared in-process
/// barriers and copy-loops, no networking, no serialization. Behaviour
/// matches the wire-level NCCL / Gloo backends bit-for-bit so a test that
/// passes here passes there.
///
/// <para><b>Construction:</b> <see cref="Create"/> spawns a fixed number of
/// rank instances that share a single
/// <see cref="InProcessGroupCoordinator"/>. The coordinator owns the
/// shared barriers, per-collective rendezvous slots, and the message
/// queues that point-to-point sends use.</para>
/// </summary>
public sealed class InProcessGroup : IProcessGroup
{
    private readonly InProcessGroupCoordinator _coord;
    private bool _disposed;

    /// <inheritdoc/>
    public int WorldSize => _coord.WorldSize;

    /// <inheritdoc/>
    public int Rank { get; }

    /// <inheritdoc/>
    public string Backend => "in-process";

    private InProcessGroup(int rank, InProcessGroupCoordinator coord)
    {
        Rank = rank;
        _coord = coord;
    }

    /// <summary>Internal: the coordinator backing this rank handle. Used
    /// by sub-systems (RPC, custom collectives) that need to share state
    /// across all ranks of the same group.</summary>
    internal InProcessGroupCoordinator Coordinator => _coord;

    /// <summary>
    /// Creates an in-process group with <paramref name="worldSize"/> ranks.
    /// Returns one <see cref="IProcessGroup"/> handle per rank — the caller
    /// is expected to dispatch each to a worker thread / task.
    /// </summary>
    public static IProcessGroup[] Create(int worldSize)
    {
        if (worldSize < 1) throw new ArgumentOutOfRangeException(nameof(worldSize));
        var coord = new InProcessGroupCoordinator(worldSize);
        var ranks = new IProcessGroup[worldSize];
        for (int r = 0; r < worldSize; r++) ranks[r] = new InProcessGroup(r, coord);
        return ranks;
    }

    /// <inheritdoc/>
    public void AllReduce<T>(Tensor<T> tensor, ReduceOp op = ReduceOp.Sum)
    {
        ThrowIfDisposed();
        // Phase 1: every rank publishes its input.
        _coord.Publish(Rank, tensor);
        _coord.SyncBarrier();
        // Phase 2: rank 0 computes the reduction across published copies and writes back to a shared slot.
        if (Rank == 0)
        {
            var reduction = ReduceAcross<T>(_coord.PublishedAsArray<T>(), op);
            _coord.SetReduction(reduction);
        }
        _coord.SyncBarrier();
        // Phase 3: every rank copies the reduction into its tensor.
        var result = _coord.GetReduction<T>();
        result.AsSpan().CopyTo(tensor.AsWritableSpan());
        _coord.SyncBarrier();
    }

    /// <inheritdoc/>
    public void Reduce<T>(Tensor<T> tensor, int root, ReduceOp op = ReduceOp.Sum)
    {
        ThrowIfDisposed();
        ValidateRoot(root);
        _coord.Publish(Rank, tensor);
        _coord.SyncBarrier();
        if (Rank == root)
        {
            var reduction = ReduceAcross<T>(_coord.PublishedAsArray<T>(), op);
            reduction.AsSpan().CopyTo(tensor.AsWritableSpan());
        }
        _coord.SyncBarrier();
    }

    /// <inheritdoc/>
    public void Broadcast<T>(Tensor<T> tensor, int root)
    {
        ThrowIfDisposed();
        ValidateRoot(root);
        if (Rank == root) _coord.Publish(Rank, tensor);
        _coord.SyncBarrier();
        if (Rank != root)
        {
            var src = _coord.GetPublished<T>(root);
            src.AsSpan().CopyTo(tensor.AsWritableSpan());
        }
        _coord.SyncBarrier();
    }

    /// <inheritdoc/>
    public void AllGather<T>(Tensor<T> input, IList<Tensor<T>> output)
    {
        ThrowIfDisposed();
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (output.Count != WorldSize)
            throw new ArgumentException($"output must have {WorldSize} entries, got {output.Count}.", nameof(output));
        _coord.Publish(Rank, input);
        _coord.SyncBarrier();
        for (int r = 0; r < WorldSize; r++)
        {
            var src = _coord.GetPublished<T>(r);
            src.AsSpan().CopyTo(output[r].AsWritableSpan());
        }
        _coord.SyncBarrier();
    }

    /// <inheritdoc/>
    public void Gather<T>(Tensor<T> input, IList<Tensor<T>>? output, int root)
    {
        ThrowIfDisposed();
        ValidateRoot(root);
        _coord.Publish(Rank, input);
        _coord.SyncBarrier();
        if (Rank == root)
        {
            if (output is null || output.Count != WorldSize)
                throw new ArgumentException($"Root must pass an output list of length {WorldSize}.", nameof(output));
            for (int r = 0; r < WorldSize; r++)
            {
                var src = _coord.GetPublished<T>(r);
                src.AsSpan().CopyTo(output[r].AsWritableSpan());
            }
        }
        _coord.SyncBarrier();
    }

    /// <inheritdoc/>
    public void Scatter<T>(IList<Tensor<T>>? input, Tensor<T> output, int root)
    {
        ThrowIfDisposed();
        ValidateRoot(root);
        if (Rank == root)
        {
            if (input is null || input.Count != WorldSize)
                throw new ArgumentException($"Root must pass an input list of length {WorldSize}.", nameof(input));
            for (int r = 0; r < WorldSize; r++) _coord.PublishToRank(r, input[r]);
        }
        _coord.SyncBarrier();
        var slice = _coord.GetSentToRank<T>(Rank);
        slice.AsSpan().CopyTo(output.AsWritableSpan());
        _coord.SyncBarrier();
    }

    /// <inheritdoc/>
    public void ReduceScatter<T>(IList<Tensor<T>> input, Tensor<T> output, ReduceOp op = ReduceOp.Sum)
    {
        ThrowIfDisposed();
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Count != WorldSize)
            throw new ArgumentException($"input must have {WorldSize} entries.", nameof(input));
        // Each rank publishes its full input list. Then rank R reduces input[R] across all ranks.
        _coord.PublishList(Rank, input);
        _coord.SyncBarrier();
        var perRank = new Tensor<T>[WorldSize];
        for (int r = 0; r < WorldSize; r++)
        {
            perRank[r] = _coord.GetPublishedList<T>(r, Rank);
        }
        var reduced = ReduceAcross(perRank, op);
        reduced.AsSpan().CopyTo(output.AsWritableSpan());
        _coord.SyncBarrier();
    }

    /// <inheritdoc/>
    public void Send<T>(Tensor<T> tensor, int dst, int tag = 0)
    {
        ThrowIfDisposed();
        _coord.SendP2P(Rank, dst, tag, tensor);
    }

    /// <inheritdoc/>
    public void Recv<T>(Tensor<T> tensor, int src, int tag = 0)
    {
        ThrowIfDisposed();
        var msg = _coord.RecvP2P<T>(src, Rank, tag);
        msg.AsSpan().CopyTo(tensor.AsWritableSpan());
    }

    /// <summary>
    /// Pipeline-helper recv that returns the queued tensor whole — used
    /// by <see cref="GPipeSchedule{T}"/> and
    /// <see cref="OneForwardOneBackwardSchedule{T}"/> where the
    /// downstream stage doesn't know the upstream activation's shape
    /// ahead of time. Network-backed backends would prepend a shape
    /// header on Send; the in-process queue carries the typed tensor
    /// reference directly.
    /// </summary>
    internal Tensor<T> RecvSized<T>(int src, int tag)
    {
        ThrowIfDisposed();
        return _coord.RecvP2P<T>(src, Rank, tag);
    }

    /// <inheritdoc/>
    public Tensor<T> RecvDiscoverShape<T>(int src, int tag = 0)
    {
        ThrowIfDisposed();
        return _coord.RecvP2P<T>(src, Rank, tag);
    }

    /// <inheritdoc/>
    public void Barrier() { ThrowIfDisposed(); _coord.SyncBarrier(); }

    /// <inheritdoc/>
    public IProcessGroup? NewGroup(IReadOnlyList<int> ranks)
    {
        ThrowIfDisposed();
        if (ranks is null) throw new ArgumentNullException(nameof(ranks));
        // Validate that all ranks are in range and unique.
        var set = new HashSet<int>();
        foreach (var r in ranks)
        {
            if (r < 0 || r >= WorldSize)
                throw new ArgumentException($"Rank {r} out of range [0, {WorldSize}).", nameof(ranks));
            if (!set.Add(r)) throw new ArgumentException($"Duplicate rank {r}.", nameof(ranks));
        }
        // We barrier so every rank has computed the same membership before
        // any rank attempts to use the new group — matches PyTorch's
        // "new_group is collective and all ranks must call it" contract.
        Barrier();

        if (!set.Contains(Rank)) return null;

        // Local rank within the new group is the index in the sorted order.
        var sorted = new int[ranks.Count];
        int j = 0;
        foreach (var r in ranks) sorted[j++] = r;
        Array.Sort(sorted);
        int localRank = Array.IndexOf(sorted, Rank);
        var subCoord = _coord.MakeSubgroupCoordinator(sorted);
        return new InProcessGroup(localRank, subCoord);
    }

    /// <inheritdoc/>
    public IProcessGroup SplitGroup(int color, int? key = null)
    {
        ThrowIfDisposed();
        // Reset stale split state from any prior SplitGroup call —
        // _splits was append-only, so a second call would see members
        // from the previous call mixed in. Rank 0 clears, barrier
        // ensures all ranks see the cleared dict before publishing.
        if (Rank == 0) _coord.ResetSplitState();
        Barrier();
        // Every rank publishes its (color, key) pair; ranks with the same
        // color form a sub-group, ordered by key (ties broken by global rank).
        _coord.PublishSplit(Rank, color, key ?? Rank);
        Barrier();
        var sameColor = _coord.GetSplitMembers(color);
        // Sort by key, then by original rank.
        sameColor.Sort((a, b) => a.Key != b.Key ? a.Key.CompareTo(b.Key) : a.Rank.CompareTo(b.Rank));
        var ranks = new int[sameColor.Count];
        int localRank = -1;
        for (int i = 0; i < sameColor.Count; i++)
        {
            ranks[i] = sameColor[i].Rank;
            if (sameColor[i].Rank == Rank) localRank = i;
        }
        Barrier();
        var subCoord = _coord.MakeSubgroupCoordinator(ranks);
        return new InProcessGroup(localRank, subCoord);
    }

    /// <inheritdoc/>
    public void Dispose() { _disposed = true; }

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(nameof(InProcessGroup)); }
    private void ValidateRoot(int root)
    {
        if (root < 0 || root >= WorldSize)
            throw new ArgumentOutOfRangeException(nameof(root), $"Root {root} out of range [0, {WorldSize}).");
    }

    /// <summary>Reduces an array of tensors element-wise according to <paramref name="op"/>.</summary>
    internal static Tensor<T> ReduceAcross<T>(IReadOnlyList<Tensor<T>> tensors, ReduceOp op)
    {
        if (tensors.Count == 0) throw new ArgumentException("Cannot reduce zero tensors.");
        var ops = MathHelper.GetNumericOperations<T>();
        var first = tensors[0];
        var result = new Tensor<T>((int[])first._shape.Clone());
        first.AsSpan().CopyTo(result.AsWritableSpan());
        for (int i = 1; i < tensors.Count; i++)
        {
            var src = tensors[i].AsSpan();
            var dst = result.AsWritableSpan();
            for (int k = 0; k < dst.Length; k++)
            {
                dst[k] = op switch
                {
                    ReduceOp.Sum or ReduceOp.Avg => ops.Add(dst[k], src[k]),
                    ReduceOp.Min => ops.LessThan(src[k], dst[k]) ? src[k] : dst[k],
                    ReduceOp.Max => ops.GreaterThan(src[k], dst[k]) ? src[k] : dst[k],
                    ReduceOp.Product => ops.Multiply(dst[k], src[k]),
                    _ => throw new NotSupportedException(
                        $"Reduce op {op} requires a bitwise-int type; in-process backend doesn't yet implement it for generic T."),
                };
            }
        }
        if (op == ReduceOp.Avg)
        {
            var divisor = ops.FromDouble(tensors.Count);
            var dst = result.AsWritableSpan();
            for (int k = 0; k < dst.Length; k++) dst[k] = ops.Divide(dst[k], divisor);
        }
        return result;
    }
}

/// <summary>
/// Shared state across the threads of one in-process group. Owns the
/// barriers, per-rank tensor publishing slots, scatter/recv buffers, and
/// p2p message queues. Internal — accessed only by
/// <see cref="InProcessGroup"/>.
/// </summary>
internal sealed class InProcessGroupCoordinator
{
    private readonly Barrier _barrier;
    private readonly object[] _published;
    private readonly object[] _scattered;
    private object? _reduction;
    private readonly Dictionary<long, ConcurrentQueue<object>> _p2p = new();
    private readonly object _p2pGate = new();
    private readonly Dictionary<long, ManualResetEventSlim> _p2pReady = new();
    private readonly Dictionary<int, List<(int Rank, int Key)>> _splits = new();
    private readonly object _splitGate = new();
    private readonly object[]? _publishedLists;

    public int WorldSize { get; }

    public InProcessGroupCoordinator(int worldSize)
    {
        WorldSize = worldSize;
        _barrier = new Barrier(worldSize);
        _published = new object[worldSize];
        _scattered = new object[worldSize];
        _publishedLists = new object[worldSize];
    }

    public void SyncBarrier() => _barrier.SignalAndWait();

    public void Publish<T>(int rank, Tensor<T> tensor) => _published[rank] = tensor;

    public Tensor<T> GetPublished<T>(int rank) => (Tensor<T>)_published[rank];

    public IReadOnlyList<Tensor<T>> PublishedAsArray<T>()
    {
        var arr = new Tensor<T>[WorldSize];
        for (int r = 0; r < WorldSize; r++) arr[r] = (Tensor<T>)_published[r];
        return arr;
    }

    public void SetReduction<T>(Tensor<T> reduction) => _reduction = reduction;

    public Tensor<T> GetReduction<T>() => (Tensor<T>)_reduction!;

    public void PublishToRank<T>(int dstRank, Tensor<T> tensor) => _scattered[dstRank] = tensor;

    public Tensor<T> GetSentToRank<T>(int rank) => (Tensor<T>)_scattered[rank];

    public void PublishList<T>(int rank, IList<Tensor<T>> list)
    {
        var copy = new Tensor<T>[list.Count];
        for (int i = 0; i < list.Count; i++) copy[i] = list[i];
        _publishedLists![rank] = copy;
    }

    public Tensor<T> GetPublishedList<T>(int srcRank, int idx)
    {
        var arr = (Tensor<T>[])_publishedLists![srcRank];
        return arr[idx];
    }

    public void SendP2P<T>(int src, int dst, int tag, Tensor<T> tensor)
    {
        long key = MakeP2PKey(src, dst, tag);
        ConcurrentQueue<object> q;
        ManualResetEventSlim mre;
        lock (_p2pGate)
        {
            if (!_p2p.TryGetValue(key, out q!)) { q = new ConcurrentQueue<object>(); _p2p[key] = q; }
            if (!_p2pReady.TryGetValue(key, out mre!)) { mre = new ManualResetEventSlim(false); _p2pReady[key] = mre; }
        }
        // Snapshot the tensor on enqueue so the sender can mutate or
        // reuse the source buffer immediately. Without this the
        // receiver would observe post-send writes — divergent from
        // every real network transport's value semantics.
        var snapshot = new Tensor<T>((int[])tensor._shape.Clone());
        tensor.AsSpan().CopyTo(snapshot.AsWritableSpan());
        q.Enqueue(snapshot);
        mre.Set();
    }

    public Tensor<T> RecvP2P<T>(int src, int dst, int tag)
    {
        long key = MakeP2PKey(src, dst, tag);
        ConcurrentQueue<object> q;
        ManualResetEventSlim mre;
        lock (_p2pGate)
        {
            if (!_p2p.TryGetValue(key, out q!)) { q = new ConcurrentQueue<object>(); _p2p[key] = q; }
            if (!_p2pReady.TryGetValue(key, out mre!)) { mre = new ManualResetEventSlim(false); _p2pReady[key] = mre; }
        }
        while (true)
        {
            if (q.TryDequeue(out var msg)) return (Tensor<T>)msg;
            mre.Wait();
            mre.Reset();
        }
    }

    private static long MakeP2PKey(int src, int dst, int tag)
        => ((long)src << 40) | ((long)dst << 20) | (uint)tag;

    public void PublishSplit(int rank, int color, int key)
    {
        lock (_splitGate)
        {
            if (!_splits.TryGetValue(color, out var list))
            {
                list = new List<(int, int)>();
                _splits[color] = list;
            }
            list.Add((rank, key));
        }
    }

    /// <summary>Clears the split membership dict — called at the start
    /// of each <see cref="InProcessGroup.SplitGroup"/> by rank 0
    /// (under a barrier) so a second SplitGroup call doesn't see
    /// stale entries from the first.</summary>
    public void ResetSplitState()
    {
        lock (_splitGate) _splits.Clear();
    }

    public List<(int Rank, int Key)> GetSplitMembers(int color)
    {
        lock (_splitGate)
        {
            return _splits.TryGetValue(color, out var list) ? new List<(int, int)>(list) : new List<(int, int)>();
        }
    }

    /// <summary>
    /// Returns the shared sub-group coordinator for the given set of
    /// global ranks. Ranks with the same membership set get the same
    /// coordinator instance — without this, every call would mint a
    /// fresh per-rank coord and the sub-group's barrier would deadlock
    /// (each rank would be the only signaller in its own private
    /// 2-rank barrier). The membership set is canonicalized to a
    /// sorted comma-joined string for the dictionary key.
    /// </summary>
    public InProcessGroupCoordinator MakeSubgroupCoordinator(int[] ranksInGlobal)
    {
        // Sort + key once per call.
        var sorted = (int[])ranksInGlobal.Clone();
        Array.Sort(sorted);
        var key = string.Join(",", sorted);
        lock (_subgroupGate)
        {
            if (!_subgroupCoords.TryGetValue(key, out var coord))
            {
                coord = new InProcessGroupCoordinator(ranksInGlobal.Length);
                _subgroupCoords[key] = coord;
            }
            return coord;
        }
    }

    private readonly Dictionary<string, InProcessGroupCoordinator> _subgroupCoords = new();
    private readonly object _subgroupGate = new();
}
