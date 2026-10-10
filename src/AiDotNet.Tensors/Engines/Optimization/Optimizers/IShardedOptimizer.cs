using AiDotNet.Tensors.Engines.Distributed;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// ZeRO / FSDP integration hook. An optimizer that implements this interface can
/// participate in optimizer-state sharding: each rank holds the full state for
/// only the param IDs in <see cref="LocalParamIds"/>, and during checkpoint save
/// only that rank's slice is materialised.
///
/// Reference: Rajbhandari et al., 2019, "ZeRO: Memory Optimizations Toward
/// Training Trillion Parameter Models" §3.1 — ZeRO-1 (optimizer-state sharding).
/// FSDP follows the same shard-along-rank-axis pattern.
/// </summary>
public interface IShardedOptimizer : IOptimizer
{
    /// <summary>This rank's index in the world (0 ≤ <c>Rank</c> &lt; <c>WorldSize</c>).</summary>
    int Rank { get; }

    /// <summary>Total number of ranks participating in the shard.</summary>
    int WorldSize { get; }

    /// <summary>Param IDs (global, contiguous from 0) owned by this rank.</summary>
    IReadOnlyList<int> LocalParamIds { get; }

    /// <summary>
    /// Build a sharded <see cref="OptimizerStateDict"/> containing only the slots
    /// for params in <see cref="LocalParamIds"/>. Suitable for writing to a per-rank
    /// shard file via <c>DistributedCheckpoint.Save</c>.
    /// </summary>
    OptimizerStateDict LocalStateDict();

    /// <summary>
    /// Re-assemble a global state dict by concatenating per-rank shards. The caller
    /// is responsible for passing every rank's <see cref="LocalStateDict"/> output
    /// (typically read back via <c>DistributedCheckpoint.Reshard</c>).
    /// </summary>
    void LoadShardedStateDict(IReadOnlyList<OptimizerStateDict> perRankShards);
}

/// <summary>
/// ZeRO-1 wrapper: takes any base <see cref="OptimizerBase"/> and partitions its
/// per-parameter state across <see cref="WorldSize"/> ranks. Each rank's
/// <see cref="IOptimizer.Step"/> applies the inner optimizer only on the local
/// param-id slice. With a process group it then sums group-wide statistics across ranks and broadcasts each
/// rank's updated shard (ZeRO-1); without one the caller brings the parameters back into sync.
/// </summary>
public sealed class ZeroShardedOptimizer : IShardedOptimizer
{
    private readonly OptimizerBase _inner;
    private readonly IProcessGroup? _processGroup;

    /// <inheritdoc />
    public int Rank { get; }
    /// <inheritdoc />
    public int WorldSize { get; }

    /// <inheritdoc />
    /// <remarks>
    /// Recomputed every read so additions to <see cref="ParamGroups"/> after the
    /// shard is constructed are reflected in the live partition. Returning a frozen
    /// snapshot would silently desync the view from <see cref="LocalStateDict"/>
    /// after any <see cref="ParamGroup.AddParameter"/> call.
    /// </remarks>
    public IReadOnlyList<int> LocalParamIds => ComputeLocalIds(_inner, Rank, WorldSize);

    /// <summary>
    /// Build a sharded optimizer view of <paramref name="inner"/> with no communication: <see cref="Step()"/> updates
    /// only this rank's shard and the caller brings the other parameters back into sync. An inner optimizer with
    /// group-wide statistics (D-Adaptation, Prodigy) needs the process-group constructor.
    /// </summary>
    public ZeroShardedOptimizer(OptimizerBase inner, int rank, int worldSize)
    {
        if (inner == null) throw new System.ArgumentNullException(nameof(inner));
        if (worldSize <= 0) throw new System.ArgumentOutOfRangeException(nameof(worldSize));
        if (rank < 0 || rank >= worldSize) throw new System.ArgumentOutOfRangeException(nameof(rank));
        _inner = inner;
        Rank = rank; WorldSize = worldSize;
    }

    /// <summary>
    /// Build a ZeRO-1 optimizer over <paramref name="processGroup"/>: each <see cref="Step()"/> updates this rank's
    /// shard with its shard of the optimizer state, sums group-wide statistics across ranks, then broadcasts every
    /// rank's updated shard so all parameters are current on every rank. Every rank must step together.
    /// </summary>
    public ZeroShardedOptimizer(OptimizerBase inner, IProcessGroup processGroup)
        : this(inner, (processGroup ?? throw new System.ArgumentNullException(nameof(processGroup))).Rank, processGroup.WorldSize)
    {
        _processGroup = processGroup;
    }

    private static IReadOnlyList<int> ComputeLocalIds(OptimizerBase inner, int rank, int worldSize)
    {
        // Round-robin partition of global param IDs (assigned in the order params were added).
        var ids = new List<int>();
        int total = 0;
        foreach (var grp in inner.ParamGroups) total += grp.Parameters.Count;
        for (int id = 0; id < total; id++) if (id % worldSize == rank) ids.Add(id);
        return ids;
    }

    /// <inheritdoc />
    public IReadOnlyList<ParamGroup> ParamGroups => _inner.ParamGroups;

    /// <inheritdoc />
    public ParamGroup AddParamGroup(IDictionary<string, double>? overrides = null)
        => _inner.AddParamGroup(overrides);

    /// <inheritdoc />
    /// <remarks>
    /// ZeRO-1 contract: each rank computes the update of its own shard only. The inner step
    /// runs under <see cref="OptimizerBase.StepFilter"/>, so other ranks' parameters, gradients
    /// and optimizer state are never touched (and their state is never allocated here).
    /// Group-wide statistics (D-Adapt, Prodigy) are all-reduced through the process group.
    /// With a process group, each rank then broadcasts its updated shard (one packed
    /// broadcast per owning rank), so every rank ends the step with every parameter current.
    /// Without one, an optimizer with group-wide statistics is refused on more than one rank.
    /// </remarks>
    public void Step() => RunLocalStep(_inner.Step);

    /// <summary>
    /// <see cref="OptimizerBase.Step(IReadOnlyDictionary{Tensor{float}, Tensor{float}})"/> for this rank's shard:
    /// tensor parameters read their gradients from <paramref name="gradients"/> with no copy.
    /// </summary>
    public void Step(IReadOnlyDictionary<Tensor<float>, Tensor<float>> gradients)
    {
        if (gradients == null) throw new System.ArgumentNullException(nameof(gradients));
        RunLocalStep(() => _inner.Step(gradients));
    }

    private void RunLocalStep(System.Action innerStep)
    {
        if (_processGroup is null && WorldSize > 1 && _inner.HasGroupStatistics)
            throw new System.InvalidOperationException(
                $"{_inner.GetType().Name} adapts its step size from statistics over every parameter of a group, which a " +
                "rank holding only its shard cannot compute alone. Construct the sharded optimizer with a process group.");

        // ZeRO-1: this rank computes the update of its own shard only, with that shard's optimizer state; the others'
        // parameters, gradients and state are not touched.
        var owner = OwnerRanks();
        _inner.StepFilter = (gi, pi) => owner[gi][pi] == Rank;
        if (_processGroup is not null)
        {
            var group = _processGroup;
            _inner.GroupStatisticsReducer = statistics =>
            {
                var reduced = new Tensor<double>((double[])statistics.Clone(), new[] { statistics.Length });
                group.AllReduce(reduced, ReduceOp.Sum);
                for (int i = 0; i < statistics.Length; i++) statistics[i] = reduced.GetFlat(i);
            };
        }
        try
        {
            innerStep();
        }
        finally
        {
            _inner.StepFilter = null;
            _inner.GroupStatisticsReducer = null;
        }

        if (_processGroup is not null) BroadcastUpdatedShards(owner);
    }

    // owner[gi][pi]: the rank that owns group gi's parameter pi (round-robin over global ids, as LocalParamIds).
    private int[][] OwnerRanks()
    {
        var owner = new int[_inner.ParamGroups.Count][];
        int globalId = 0;
        for (int gi = 0; gi < owner.Length; gi++)
        {
            owner[gi] = new int[_inner.ParamGroups[gi].Parameters.Count];
            for (int pi = 0; pi < owner[gi].Length; pi++, globalId++) owner[gi][pi] = globalId % WorldSize;
        }
        return owner;
    }

    // Each rank broadcasts its freshly updated shard, packed into one buffer, so every rank ends the step with every
    // parameter current: the all-gather half of ZeRO-1.
    private void BroadcastUpdatedShards(int[][] owner)
    {
        var group = _processGroup ?? throw new System.InvalidOperationException("No process group.");
        for (int root = 0; root < WorldSize; root++)
        {
            int length = 0;
            for (int gi = 0; gi < owner.Length; gi++)
                for (int pi = 0; pi < owner[gi].Length; pi++)
                    if (owner[gi][pi] == root) length += _inner.ParamGroups[gi].Parameters[pi].Length;
            if (length == 0) continue;

            var packed = new float[length];
            if (root == Rank)
                ForEachOwned(owner, root, (gi, pi, offset, n) => System.Array.Copy(ReadParameter(gi, pi), 0, packed, offset, n));
            var wire = new Tensor<float>(packed, new[] { length });
            group.Broadcast(wire, root);
            if (root == Rank) continue;
            var received = wire.ToArray();
            ForEachOwned(owner, root, (gi, pi, offset, n) => WriteParameter(gi, pi, received, offset, n));
        }
    }

    private void ForEachOwned(int[][] owner, int root, System.Action<int, int, int, int> visit)
    {
        int offset = 0;
        for (int gi = 0; gi < owner.Length; gi++)
            for (int pi = 0; pi < owner[gi].Length; pi++)
            {
                if (owner[gi][pi] != root) continue;
                int n = _inner.ParamGroups[gi].Parameters[pi].Length;
                visit(gi, pi, offset, n);
                offset += n;
            }
    }

    // A parameter's current values: a GPU-resident tensor parameter is read from the device, the rest from its array.
    private float[] ReadParameter(int gi, int pi)
    {
        var group = _inner.ParamGroups[gi];
        var tensor = group.ParameterTensor(pi);
        if (tensor is not null && group.IsDeviceParameter(pi))
            return GpuOptimizer.TryDownload(tensor) ?? throw new System.InvalidOperationException("A GPU parameter has no device buffer.");
        return group.Parameters[pi];
    }

    private void WriteParameter(int gi, int pi, float[] source, int offset, int count)
    {
        var group = _inner.ParamGroups[gi];
        var tensor = group.ParameterTensor(pi);
        if (tensor is not null && group.IsDeviceParameter(pi))
        {
            var values = new float[count];
            System.Array.Copy(source, offset, values, 0, count);
            if (!GpuOptimizer.TryUpload(tensor, values))
                throw new System.InvalidOperationException("A GPU parameter could not be written.");
            return;
        }
        System.Array.Copy(source, offset, group.Parameters[pi], 0, count);
        tensor?.MarkModified();
    }

    /// <inheritdoc />
    public void ZeroGrad() => _inner.ZeroGrad();

    /// <inheritdoc />
    public OptimizerStateDict StateDict() => _inner.StateDict();

    /// <inheritdoc />
    public void LoadStateDict(OptimizerStateDict state) => _inner.LoadStateDict(state);

    /// <inheritdoc />
    public OptimizerStateDict LocalStateDict()
    {
        var full = _inner.StateDict();
        var local = new OptimizerStateDict();
        var localSet = new HashSet<int>(LocalParamIds);
        // Preserve every group; filter ParamIds + state to the local slice.
        foreach (var grp in full.ParamGroups)
        {
            var localGroup = new OptimizerGroupState();
            foreach (var kv in grp.Options) localGroup.Options[kv.Key] = kv.Value;
            foreach (var pid in grp.ParamIds)
                if (localSet.Contains(pid)) localGroup.ParamIds.Add(pid);
            local.ParamGroups.Add(localGroup);
        }
        foreach (var kv in full.State)
            if (localSet.Contains(kv.Key)) local.State[kv.Key] = kv.Value;
        return local;
    }

    /// <inheritdoc />
    public void LoadShardedStateDict(IReadOnlyList<OptimizerStateDict> perRankShards)
    {
        if (perRankShards == null) throw new System.ArgumentNullException(nameof(perRankShards));
        if (perRankShards.Count != WorldSize)
            throw new System.ArgumentException(
                $"expected {WorldSize} shards, got {perRankShards.Count}.", nameof(perRankShards));

        // Merge all per-rank shards into a global state dict.
        var merged = new OptimizerStateDict();
        // Use the first shard's group list as the template (all shards share the same groups).
        for (int gi = 0; gi < perRankShards[0].ParamGroups.Count; gi++)
        {
            var template = perRankShards[0].ParamGroups[gi];
            var grp = new OptimizerGroupState();
            foreach (var kv in template.Options) grp.Options[kv.Key] = kv.Value;
            // Concatenate every rank's contribution to ParamIds in rank order, preserving uniqueness.
            var seen = new HashSet<int>();
            foreach (var shard in perRankShards)
                foreach (var pid in shard.ParamGroups[gi].ParamIds)
                    if (seen.Add(pid)) grp.ParamIds.Add(pid);
            merged.ParamGroups.Add(grp);
        }
        foreach (var shard in perRankShards)
            foreach (var kv in shard.State)
                merged.State[kv.Key] = kv.Value;

        _inner.LoadStateDict(merged);
    }
}
