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

    /// <inheritdoc />
    public IReadOnlyList<ParamGroup> ParamGroups => _groups;

    /// <summary>Default hyper-parameters injected into every newly added group.</summary>
    protected abstract IReadOnlyDictionary<string, double> Defaults { get; }

    /// <summary>Names of the per-parameter state slots required by this optimizer.</summary>
    protected abstract IReadOnlyList<string> StateNames { get; }

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
        foreach (var name in StateNames)
        {
            if (name == "step") dict[name] = OptimizerStateValue.FromInt(0);
            else dict[name] = OptimizerStateValue.FromTensor(new float[paramLen]);
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

        int paramCounter = 0;
        for (int gi = 0; gi < _groups.Count; gi++)
        {
            var group = _groups[gi];
            var gs = state.ParamGroups[gi];
            foreach (var kv in gs.Options) group.Options[kv.Key] = kv.Value;
            if (gs.ParamIds.Count != group.Parameters.Count)
                throw new InvalidOperationException(
                    $"group {gi} has {group.Parameters.Count} params; state-dict has {gs.ParamIds.Count}.");
            for (int pi = 0; pi < group.Parameters.Count; pi++)
            {
                int id = paramCounter++;
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
