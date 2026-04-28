using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// Common contract for all optimizers exposed at the tensor-library layer.
///
/// Mirrors PyTorch's <c>torch.optim.Optimizer</c> step + zero_grad + state_dict surface,
/// while keeping the underlying parameter / gradient buffers as plain <see cref="float"/>[]
/// vectors (1-1 with the tensor backing arrays).
/// </summary>
public interface IOptimizer
{
    /// <summary>Per-group config (lr, weight_decay, …) plus the parameters in that group.</summary>
    IReadOnlyList<ParamGroup> ParamGroups { get; }

    /// <summary>Add a parameter group; <paramref name="overrides"/> override the optimizer's defaults.</summary>
    ParamGroup AddParamGroup(IDictionary<string, double>? overrides = null);

    /// <summary>Apply one optimization step using the gradients currently in <see cref="ParamGroup.Gradients"/>.</summary>
    void Step();

    /// <summary>Zero out every gradient buffer.</summary>
    void ZeroGrad();

    /// <summary>Capture all hyper-parameters and per-parameter state into a serialisable dictionary.</summary>
    OptimizerStateDict StateDict();

    /// <summary>Restore optimizer state previously captured by <see cref="StateDict"/>.</summary>
    void LoadStateDict(OptimizerStateDict state);
}
