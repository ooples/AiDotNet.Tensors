using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// Shared scaffolding for the concrete optimizers: param groups, default hyper-parameters,
/// per-parameter state slots, and state-dict serialisation. Subclasses only have to:
///   1) declare the hyper-parameter <see cref="Defaults"/>,
///   2) declare which state slots each parameter needs in <see cref="StateNames"/>,
///   3) implement <see cref="Step"/> by walking <see cref="ParamGroups"/>.
/// </summary>
public abstract class OptimizerBase : IOptimizer
{
    private readonly List<ParamGroup> _groups = new List<ParamGroup>();
    /// <summary>Per-parameter state buffers, keyed by (group index, param index) → state name → value.</summary>
    protected readonly Dictionary<(int g, int p), Dictionary<string, OptimizerStateValue>> _state =
        new Dictionary<(int, int), Dictionary<string, OptimizerStateValue>>();

    /// <summary>Access to the state map for sharded-optimizer wrappers that need to snapshot
    /// and restore non-local parameter state across <see cref="Step"/> calls.</summary>
    internal Dictionary<(int g, int p), Dictionary<string, OptimizerStateValue>> StateInternal => _state;

    /// <inheritdoc />
    public IReadOnlyList<ParamGroup> ParamGroups => _groups;

    /// <summary>Default hyper-parameters injected into every newly added group.</summary>
    protected abstract IReadOnlyDictionary<string, double> Defaults { get; }

    /// <summary>Names of the per-parameter state slots required by this optimizer.</summary>
    protected abstract IReadOnlyList<string> StateNames { get; }

    /// <summary>
    /// Names of state slots that hold a single scalar per parameter (e.g. <c>"step"</c>,
    /// <c>"eta"</c>, <c>"mu"</c>) rather than a length-N tensor. <see cref="GetOrCreateState"/>
    /// allocates these as zero-initialised <see cref="OptimizerStateValue"/> placeholders
    /// instead of full tensor buffers, eliminating wasted memory on every parameter.
    /// Default: only <c>"step"</c> is treated as a scalar.
    /// </summary>
    protected virtual IReadOnlyList<string> ScalarStateNames { get; } = new[] { "step" };

    /// <inheritdoc />
    public abstract void Step();

    /// <summary>Add a parameter group; <paramref name="overrides"/> override <see cref="Defaults"/>.</summary>
    public ParamGroup AddParamGroup(IDictionary<string, double>? overrides = null)
    {
        var g = new ParamGroup();
        foreach (var kv in Defaults) g.Options[kv.Key] = kv.Value;
        if (overrides != null)
            foreach (var kv in overrides) g.Options[kv.Key] = kv.Value;
        _groups.Add(g);
        return g;
    }

    /// <summary>Convenience: single-group setup that adds all params under default hyper-params.</summary>
    public ParamGroup AddParameters(IEnumerable<(float[] param, float[] grad)> pairs)
    {
        var group = AddParamGroup();
        foreach (var (p, g) in pairs) group.AddParameter(p, g);
        return group;
    }

    /// <summary>Get or lazily create the state record for <c>group[gi].param[pi]</c>.</summary>
    protected Dictionary<string, OptimizerStateValue> GetOrCreateState(int gi, int pi, int paramLen)
    {
        var key = (gi, pi);
        if (_state.TryGetValue(key, out var dict)) return dict;
        dict = new Dictionary<string, OptimizerStateValue>();
        var scalarSet = new HashSet<string>(ScalarStateNames, StringComparer.Ordinal);
        foreach (var name in StateNames)
        {
            if (name == "step")
                dict[name] = OptimizerStateValue.FromInt(0);
            else if (scalarSet.Contains(name))
                dict[name] = OptimizerStateValue.FromFloat(0f);
            else
                dict[name] = OptimizerStateValue.FromTensor(new float[paramLen]);
        }
        _state[key] = dict;
        return dict;
    }

    /// <inheritdoc />
    public void ZeroGrad()
    {
        foreach (var g in _groups)
            for (int i = 0; i < g.Gradients.Count; i++)
                Array.Clear(g.Gradients[i], 0, g.Gradients[i].Length);
    }

    // ------------------------------------------------------------------
    // Sparse-gradient plumbing (shared by every concrete optimizer).
    //
    // The autodiff side (BackwardFunctions / DifferentiableOps / SparseEmbeddingGradient)
    // records embedding-style gradients as (row-indices, row-values) instead of dense
    // [vocab, dim] tensors. Consumers (PyTorch-style training loops) bridge that sparse
    // representation onto the optimizer by calling SetSparseGradient(gi, pi, idx, vals)
    // before Step(). Each Step() then sees the sparse view via TryGetSparseGradient and
    // scatter-updates only the touched indices, skipping the full-parameter scan that
    // is wasteful when the dense gradient is mostly zero.
    //
    // This mirrors what SparseAdamOptimizer (the only sparse-aware optimizer prior to
    // PR #567) already did, but lifted into the base class so EVERY optimizer
    // automatically gains the same fast path with zero per-subclass boilerplate.
    // Optimizers whose update math doesn't decompose elementwise (Rprop sign tracking,
    // LBFGS / Shampoo matrix state, Asgd's averaged-weight buffer) opt out by simply
    // not calling TryGetSparseGradient.
    // ------------------------------------------------------------------

    private readonly Dictionary<(int gi, int pi), (int[] idx, float[] val, bool autoClear)> _sparseGrads =
        new Dictionary<(int gi, int pi), (int[], float[], bool)>();

    /// <summary>Publish a sparse gradient (row-indices + row-values) for the next <see cref="Step"/> call.
    /// Indices are flat positions into the parameter buffer (not row indices into a 2D view) so this works
    /// for any rank/layout; embedding callers feed in <c>row*embDim + col</c> flat indices.
    /// When <paramref name="autoClear"/> is true (the default) the entry is removed at the end of <see cref="Step"/>
    /// so each step re-publishes — matching the AccumulateGrad / SparseEmbeddingGradient lifecycle on the
    /// autodiff side. Pass false for static masks that persist across steps.</summary>
    public void SetSparseGradient(int paramGroupIndex, int paramIndex, int[] indices, float[] values, bool autoClear = true)
    {
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (indices.Length != values.Length)
            throw new ArgumentException("indices and values must be the same length.", nameof(values));
        _sparseGrads[(paramGroupIndex, paramIndex)] = (indices, values, autoClear);
    }

    /// <summary>Drop a sparse gradient previously published via <see cref="SetSparseGradient"/>.</summary>
    public void ClearSparseGradient(int paramGroupIndex, int paramIndex)
        => _sparseGrads.Remove((paramGroupIndex, paramIndex));

    /// <summary>Drop every sparse-gradient entry (both auto-clear and sticky).</summary>
    public void ClearAllSparseGradients() => _sparseGrads.Clear();

    /// <summary>Subclass probe: returns true and yields the published (indices, values) when
    /// a sparse override has been wired for this parameter. Subclasses that can decompose
    /// their update elementwise should consume the sparse view; others should fall through
    /// to their existing dense kernel.</summary>
    protected bool TryGetSparseGradient(int paramGroupIndex, int paramIndex, out int[] idx, out float[] val, out int nnz)
    {
        if (_sparseGrads.TryGetValue((paramGroupIndex, paramIndex), out var pair))
        {
            idx = pair.idx;
            val = pair.val;
            nnz = pair.idx.Length;
            return true;
        }
        idx = null!;
        val = null!;
        nnz = 0;
        return false;
    }

    /// <summary>Returns true iff a sparse gradient is wired for the given param. Cheap probe
    /// for subclasses that want to short-circuit before touching the dense buffer.</summary>
    protected bool HasSparseGradient(int paramGroupIndex, int paramIndex)
        => _sparseGrads.ContainsKey((paramGroupIndex, paramIndex));

    /// <summary>Materialize the published sparse gradient (if any) into the supplied dense
    /// buffer: zeros <paramref name="dense"/> first, then scatter-adds <c>(idx, val)</c> pairs.
    /// Used by optimizers whose update math does NOT decompose elementwise (LAMB / LARS /
    /// ASGD / Rprop — trust-ratio, averaged-weight, sign-tracking) so they can still consume
    /// a sparse-published gradient via the regular dense kernel. No-op if no sparse grad is
    /// published for this (gi, pi).</summary>
    protected void MaterializeSparseIntoDense(int paramGroupIndex, int paramIndex, float[] dense)
    {
        if (dense == null) throw new ArgumentNullException(nameof(dense));
        if (!_sparseGrads.TryGetValue((paramGroupIndex, paramIndex), out var pair)) return;
        Array.Clear(dense, 0, dense.Length);
        var idx = pair.idx;
        var val = pair.val;
        for (int k = 0; k < idx.Length; k++) dense[idx[k]] += val[k];
    }

    /// <summary>Remove all auto-clear sparse-grad entries. Subclasses MUST call this from
    /// the <c>finally</c> block of <see cref="Step"/> so each step starts with a clean slate.</summary>
    protected void ClearAutoClearSparseGrads()
    {
        if (_sparseGrads.Count == 0) return;
        List<(int, int)>? toRemove = null;
        foreach (var kv in _sparseGrads)
        {
            if (kv.Value.autoClear)
            {
                toRemove ??= new List<(int, int)>();
                toRemove.Add(kv.Key);
            }
        }
        if (toRemove != null)
            foreach (var key in toRemove) _sparseGrads.Remove(key);
    }

    /// <summary>If a group's <c>"maximize"</c> option is true, flip the sign of each gradient
    /// in-place so the downstream descent kernel performs an ascent step. Gradients are written
    /// back to their original sign at the end of <see cref="Step"/> via <see cref="UnflipMaximize"/>.</summary>
    /// <returns>True if any group had maximize active (caller must call <see cref="UnflipMaximize"/>).</returns>
    protected bool ApplyMaximize()
    {
        bool any = false;
        for (int gi = 0; gi < _groups.Count; gi++)
        {
            var g = _groups[gi];
            if (g.GetOption("maximize", 0.0) == 0.0) continue;
            any = true;
            for (int pi = 0; pi < g.Gradients.Count; pi++)
            {
                var grad = g.Gradients[pi];
                for (int i = 0; i < grad.Length; i++) grad[i] = -grad[i];
            }
        }
        return any;
    }

    /// <summary>Restore the original sign of gradients flipped by <see cref="ApplyMaximize"/>.</summary>
    protected void UnflipMaximize()
    {
        for (int gi = 0; gi < _groups.Count; gi++)
        {
            var g = _groups[gi];
            if (g.GetOption("maximize", 0.0) == 0.0) continue;
            for (int pi = 0; pi < g.Gradients.Count; pi++)
            {
                var grad = g.Gradients[pi];
                for (int i = 0; i < grad.Length; i++) grad[i] = -grad[i];
            }
        }
    }

    /// <summary>
    /// Hook for subclasses to publish optimizer-level per-group state into the saved dict
    /// (e.g. D-Adaptation's <c>CurrentD</c>, Prodigy's <c>DNumerator</c>). Default: empty.
    /// Mirror this on the load side via <see cref="SetGroupExtraState"/>.
    /// </summary>
    protected virtual Dictionary<string, OptimizerStateValue> GetGroupExtraState(int groupIndex) =>
        new Dictionary<string, OptimizerStateValue>();

    /// <summary>Restore the per-group state captured by <see cref="GetGroupExtraState"/>.</summary>
    protected virtual void SetGroupExtraState(int groupIndex, Dictionary<string, OptimizerStateValue> extraState) { }

    /// <inheritdoc />
    public OptimizerStateDict StateDict()
    {
        var sd = new OptimizerStateDict();
        int paramCounter = 0;
        for (int gi = 0; gi < _groups.Count; gi++)
        {
            var group = _groups[gi];
            var groupState = new OptimizerGroupState();
            foreach (var kv in group.Options) groupState.Options[kv.Key] = kv.Value;
            // Subclass-level state lives in ExtraState so save/load round-trips it.
            foreach (var kv in GetGroupExtraState(gi))
            {
                var v = kv.Value;
                groupState.ExtraState[kv.Key] = new OptimizerStateValue
                {
                    IntValue = v.IntValue,
                    FloatValue = v.FloatValue,
                    Tensor = v.Tensor == null ? null : (float[])v.Tensor.Clone(),
                };
            }
            for (int pi = 0; pi < group.Parameters.Count; pi++)
            {
                int id = paramCounter++;
                groupState.ParamIds.Add(id);
                if (_state.TryGetValue((gi, pi), out var slots))
                {
                    var copy = new Dictionary<string, OptimizerStateValue>();
                    foreach (var kv in slots)
                    {
                        var v = kv.Value;
                        copy[kv.Key] = new OptimizerStateValue
                        {
                            IntValue = v.IntValue,
                            FloatValue = v.FloatValue,
                            Tensor = v.Tensor == null ? null : (float[])v.Tensor.Clone()
                        };
                    }
                    sd.State[id] = copy;
                }
            }
            sd.ParamGroups.Add(groupState);
        }
        return sd;
    }

    /// <inheritdoc />
    public void LoadStateDict(OptimizerStateDict state)
    {
        if (state == null) throw new ArgumentNullException(nameof(state));
        if (state.ParamGroups.Count != _groups.Count)
            throw new InvalidOperationException(
                $"state-dict has {state.ParamGroups.Count} groups but optimizer has {_groups.Count}.");

        for (int gi = 0; gi < _groups.Count; gi++)
        {
            var group = _groups[gi];
            var gs = state.ParamGroups[gi];
            foreach (var kv in gs.Options) group.Options[kv.Key] = kv.Value;
            // Restore subclass-level per-group state before per-parameter slots so a
            // freshly-loaded optimizer's first Step() observes the saved values rather than
            // recomputing from defaults.
            if (gs.ExtraState.Count > 0)
            {
                var clone = new Dictionary<string, OptimizerStateValue>();
                foreach (var kv in gs.ExtraState)
                {
                    var v = kv.Value;
                    clone[kv.Key] = new OptimizerStateValue
                    {
                        IntValue = v.IntValue,
                        FloatValue = v.FloatValue,
                        Tensor = v.Tensor == null ? null : (float[])v.Tensor.Clone(),
                    };
                }
                SetGroupExtraState(gi, clone);
            }
            if (gs.ParamIds.Count != group.Parameters.Count)
                throw new InvalidOperationException(
                    $"group {gi} has {group.Parameters.Count} params; state-dict has {gs.ParamIds.Count}.");
            for (int pi = 0; pi < group.Parameters.Count; pi++)
            {
                // Use the serialized param id (from the state-dict's ParamIds list) rather than
                // a fresh counter. This makes load symmetric with save (both sides honor the
                // explicit id mapping) and works with non-contiguous IDs that arise from
                // sharded / partial state-dict loads.
                int id = gs.ParamIds[pi];
                if (!state.State.TryGetValue(id, out var slots)) continue;
                var dst = GetOrCreateState(gi, pi, group.Parameters[pi].Length);
                foreach (var kv in slots)
                {
                    if (!dst.TryGetValue(kv.Key, out var existing))
                    {
                        dst[kv.Key] = new OptimizerStateValue
                        {
                            IntValue = kv.Value.IntValue,
                            FloatValue = kv.Value.FloatValue,
                            Tensor = kv.Value.Tensor == null ? null : (float[])kv.Value.Tensor.Clone()
                        };
                        continue;
                    }
                    existing.IntValue = kv.Value.IntValue;
                    existing.FloatValue = kv.Value.FloatValue;
                    if (kv.Value.Tensor != null && existing.Tensor != null)
                    {
                        if (kv.Value.Tensor.Length != existing.Tensor.Length)
                            throw new InvalidOperationException(
                                $"state '{kv.Key}' length {kv.Value.Tensor.Length} != {existing.Tensor.Length}.");
                        Array.Copy(kv.Value.Tensor, existing.Tensor, existing.Tensor.Length);
                    }
                    else if (kv.Value.Tensor != null)
                    {
                        existing.Tensor = (float[])kv.Value.Tensor.Clone();
                    }
                }
            }
        }
    }
}
