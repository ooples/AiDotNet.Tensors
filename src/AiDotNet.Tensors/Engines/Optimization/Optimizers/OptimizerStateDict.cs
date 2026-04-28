using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// Serialisable optimizer state. Layout is intentionally close to PyTorch's
/// <c>Optimizer.state_dict()</c> so we can interop:
///   { "state":        [param_id → tensor_dict],
///     "param_groups": [ {lr, weight_decay, …, params: [id, id, …]} ] }
/// </summary>
public sealed class OptimizerStateDict
{
    /// <summary>Per-parameter state: keyed by parameter id, then by state name (e.g. "step", "exp_avg", "exp_avg_sq").</summary>
    public Dictionary<int, Dictionary<string, OptimizerStateValue>> State { get; } = new Dictionary<int, Dictionary<string, OptimizerStateValue>>();

    /// <summary>Per-group hyper-parameters and the param-id list belonging to each group.</summary>
    public List<OptimizerGroupState> ParamGroups { get; } = new List<OptimizerGroupState>();
}

/// <summary>Hyper-parameter snapshot for a single param group.</summary>
public sealed class OptimizerGroupState
{
    /// <summary>Group hyper-parameters (lr, weight_decay, betas, eps, …).</summary>
    public Dictionary<string, double> Options { get; } = new Dictionary<string, double>();

    /// <summary>Param-id list (indices into <see cref="OptimizerStateDict.State"/>).</summary>
    public List<int> ParamIds { get; } = new List<int>();
}

/// <summary>
/// A single state-tensor or scalar. Supports the three PyTorch flavors we use
/// (int scalar for <c>step</c>, float scalar for <c>rho/η</c>, float vector for moments).
/// </summary>
public sealed class OptimizerStateValue
{
    /// <summary>Scalar integer (e.g. step counter).</summary>
    public int? IntValue { get; set; }

    /// <summary>Scalar float (e.g. running scaling factor).</summary>
    public float? FloatValue { get; set; }

    /// <summary>Tensor data (flattened) — used for <c>exp_avg</c>, <c>exp_avg_sq</c>, <c>vMax</c>, …</summary>
    public float[]? Tensor { get; set; }

    /// <summary>Build a state value holding an int.</summary>
    public static OptimizerStateValue FromInt(int v) => new OptimizerStateValue { IntValue = v };

    /// <summary>Build a state value holding a float.</summary>
    public static OptimizerStateValue FromFloat(float v) => new OptimizerStateValue { FloatValue = v };

    /// <summary>Build a state value holding a tensor (the array is referenced, not copied).</summary>
    public static OptimizerStateValue FromTensor(float[] t) => new OptimizerStateValue { Tensor = t };
}
