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
/// param-id slice, then assumes an external all-gather/broadcast brings the
/// updated parameters back into sync (this matches ZeRO-1's invariants — the
/// tensor-layer is unaware of the communication primitive itself).
/// </summary>
public sealed class ZeroShardedOptimizer : IShardedOptimizer
{
    private readonly OptimizerBase _inner;

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

    /// <summary>Build a sharded optimizer view of <paramref name="inner"/>.</summary>
    public ZeroShardedOptimizer(OptimizerBase inner, int rank, int worldSize)
    {
        if (inner == null) throw new System.ArgumentNullException(nameof(inner));
        if (worldSize <= 0) throw new System.ArgumentOutOfRangeException(nameof(worldSize));
        if (rank < 0 || rank >= worldSize) throw new System.ArgumentOutOfRangeException(nameof(rank));
        _inner = inner;
        Rank = rank; WorldSize = worldSize;
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
    /// ZeRO-1 contract: each rank steps only its local parameters; non-local parameters
    /// (and their state) must not change on this rank — they are owned and updated by
    /// other ranks, then communicated back via all-gather.
    ///
    /// We achieve that without touching every concrete optimizer by snapshotting the
    /// non-local parameters and their state before delegating to <c>_inner.Step()</c>,
    /// then restoring them afterwards. Local params + state are updated normally.
    /// </remarks>
    public void Step()
    {
        var localSet = new HashSet<int>(LocalParamIds);

        // Snapshot non-local params + their gradient (so the inner Step's writes are reversible).
        var paramSnapshots = new List<(float[] target, float[] saved)>();
        var gradSnapshots = new List<(float[] target, float[] saved)>();
        // Snapshot non-local optimizer state (deep clone of the OptimizerStateValue dictionary).
        var stateSnapshots = new List<(int gi, int pi, Dictionary<string, OptimizerStateValue> saved)>();

        int globalId = 0;
        for (int gi = 0; gi < _inner.ParamGroups.Count; gi++)
        {
            var grp = _inner.ParamGroups[gi];
            for (int pi = 0; pi < grp.Parameters.Count; pi++, globalId++)
            {
                if (localSet.Contains(globalId)) continue;
                var p = grp.Parameters[pi];
                var g = grp.Gradients[pi];
                paramSnapshots.Add((p, (float[])p.Clone()));
                gradSnapshots.Add((g, (float[])g.Clone()));
                if (_inner.StateInternal.TryGetValue((gi, pi), out var slots))
                    stateSnapshots.Add((gi, pi, CloneSlots(slots)));
            }
        }

        _inner.Step();

        // Restore non-local params + state.
        foreach (var (target, saved) in paramSnapshots) System.Array.Copy(saved, target, target.Length);
        foreach (var (target, saved) in gradSnapshots)  System.Array.Copy(saved, target, target.Length);
        foreach (var (gi, pi, saved) in stateSnapshots)
            _inner.StateInternal[(gi, pi)] = saved;
    }

    private static Dictionary<string, OptimizerStateValue> CloneSlots(Dictionary<string, OptimizerStateValue> src)
    {
        var dst = new Dictionary<string, OptimizerStateValue>(src.Count);
        foreach (var kv in src)
            dst[kv.Key] = new OptimizerStateValue
            {
                IntValue = kv.Value.IntValue,
                FloatValue = kv.Value.FloatValue,
                Tensor = kv.Value.Tensor == null ? null : (float[])kv.Value.Tensor.Clone(),
            };
        return dst;
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
