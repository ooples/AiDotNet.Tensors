using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Tensors.Serialization.Pickle;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// Loads a PyTorch <c>torch.save(optimizer.state_dict(), 'opt.pt')</c> file into the
/// in-house <see cref="OptimizerStateDict"/>. PyTorch's optimizer state-dict layout is:
/// <code>
/// {
///   'state':        {0: {'step': T, 'exp_avg': T, 'exp_avg_sq': T, ...}, 1: {...}, ...},
///   'param_groups': [{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8,
///                     'weight_decay': 0.0, 'amsgrad': False, 'params': [0, 1, ...]}]
/// }
/// </code>
/// We map each integer param-id, the inner state tensors, and the per-group
/// hyper-params into the flat <see cref="OptimizerStateDict"/> that
/// <see cref="IOptimizer.LoadStateDict"/> consumes. This satisfies the issue #224
/// acceptance criterion: <i>"PyTorch state-dict load: read a .pt optimizer state
/// produced by PyTorch Adam, step once, compare trajectory."</i>
/// </summary>
public static class PyTorchOptimizerStateLoader
{
    /// <summary>Load <paramref name="path"/> and convert to an <see cref="OptimizerStateDict"/>.</summary>
    public static OptimizerStateDict LoadFromFile(string path)
    {
        if (path == null) throw new ArgumentNullException(nameof(path));
        var reader = PtReader.Open(path);
        return Convert(reader);
    }

    /// <summary>Convert a previously-loaded <see cref="PtReader"/> into an <see cref="OptimizerStateDict"/>.</summary>
    public static OptimizerStateDict Convert(PtReader reader)
    {
        if (reader == null) throw new ArgumentNullException(nameof(reader));
        if (reader.RawRoot is not IDictionary root)
            throw new InvalidOperationException(
                "PyTorch optimizer state-dict must deserialise to a dict; got " +
                (reader.RawRoot?.GetType().Name ?? "null"));
        return Convert(root);
    }

    /// <summary>Convert a raw pickled top-level dict into an <see cref="OptimizerStateDict"/>.
    /// Useful for tests that synthesize the structure directly without a <c>.pt</c> file.</summary>
    public static OptimizerStateDict Convert(IDictionary root)
    {
        if (root == null) throw new ArgumentNullException(nameof(root));
        if (!TryGet(root, "state", out var stateObj))
            throw new InvalidOperationException("PyTorch optimizer state-dict has no 'state' key.");
        if (!TryGet(root, "param_groups", out var paramGroupsObj))
            throw new InvalidOperationException("PyTorch optimizer state-dict has no 'param_groups' key.");

        var sd = new OptimizerStateDict();
        if (stateObj is IDictionary stateDict)
        {
            foreach (DictionaryEntry e in stateDict)
            {
                int paramId = ToInt(e.Key);
                if (e.Value is not IDictionary innerDict)
                    throw new InvalidDataException($"state[{paramId}] is not a dict.");
                var slots = new Dictionary<string, OptimizerStateValue>();
                foreach (DictionaryEntry slot in innerDict)
                {
                    if (slot.Key is not string slotName) continue;
                    slots[MapPyTorchSlotName(slotName)] = ConvertSlotValue(slot.Value);
                }
                sd.State[paramId] = slots;
            }
        }

        if (paramGroupsObj is IList groupList)
        {
            foreach (var groupObj in groupList)
            {
                if (groupObj is not IDictionary groupDict) continue;
                var grp = new OptimizerGroupState();
                foreach (DictionaryEntry kv in groupDict)
                {
                    if (kv.Key is not string optName) continue;
                    if (optName == "params")
                    {
                        if (kv.Value is IList pids)
                            foreach (var pid in pids) grp.ParamIds.Add(ToInt(pid));
                        continue;
                    }
                    // PyTorch packs (β1, β2) as a tuple; flatten to "beta1"/"beta2".
                    if (optName == "betas" && kv.Value is IList tuple && tuple.Count >= 2)
                    {
                        grp.Options["beta1"] = ToDouble(tuple[0]);
                        grp.Options["beta2"] = ToDouble(tuple[1]);
                        continue;
                    }
                    if (optName == "etas" && kv.Value is IList etas && etas.Count >= 2)
                    {
                        grp.Options["eta_minus"] = ToDouble(etas[0]);
                        grp.Options["eta_plus"]  = ToDouble(etas[1]);
                        continue;
                    }
                    if (optName == "step_sizes" && kv.Value is IList stepSizes && stepSizes.Count >= 2)
                    {
                        grp.Options["step_min"] = ToDouble(stepSizes[0]);
                        grp.Options["step_max"] = ToDouble(stepSizes[1]);
                        continue;
                    }
                    grp.Options[optName] = ToDouble(kv.Value);
                }
                sd.ParamGroups.Add(grp);
            }
        }
        return sd;
    }

    /// <summary>Translate PyTorch's slot names to ours where they differ.</summary>
    private static string MapPyTorchSlotName(string n) => n switch
    {
        "exp_avg"        => "exp_avg",
        "exp_avg_sq"     => "exp_avg_sq",
        "max_exp_avg_sq" => "max_exp_avg_sq",
        "exp_inf"        => "exp_inf",
        "step"           => "step",
        // RMSprop / AdaDelta / etc. — same names already.
        _ => n,
    };

    private static OptimizerStateValue ConvertSlotValue(object? v)
    {
        switch (v)
        {
            case PtTensorRef tr:
                {
                    var t = PtReader.ToTensor<float>(tr);
                    var flat = t.ToArray();
                    if (flat.Length == 1) return OptimizerStateValue.FromFloat(flat[0]);
                    return OptimizerStateValue.FromTensor(flat);
                }
            case float[] floats:
                {
                    if (floats.Length == 1) return OptimizerStateValue.FromFloat(floats[0]);
                    return OptimizerStateValue.FromTensor(floats);
                }
            case long l: return OptimizerStateValue.FromInt((int)l);
            case int i:  return OptimizerStateValue.FromInt(i);
            case double d: return OptimizerStateValue.FromFloat((float)d);
            case float f:  return OptimizerStateValue.FromFloat(f);
            case null:     return new OptimizerStateValue();
            default:
                // Tuples / lists from PyTorch (e.g. running_avg shape) are not used by our optimizers
                // in any state-dict slot we round-trip; fall through to a no-op state value.
                return new OptimizerStateValue();
        }
    }

    private static bool TryGet(IDictionary d, string key, out object? value)
    {
        foreach (DictionaryEntry e in d)
            if (e.Key is string s && s == key) { value = e.Value; return true; }
        value = null; return false;
    }

    private static int ToInt(object? v) => v switch
    {
        long l => (int)l,
        int i  => i,
        _ => throw new InvalidDataException($"Expected integer, got {v?.GetType().Name ?? "null"}"),
    };

    private static double ToDouble(object? v) => v switch
    {
        double d => d,
        float f  => f,
        long l   => l,
        int i    => i,
        bool b   => b ? 1.0 : 0.0,
        _ => 0.0,
    };
}

