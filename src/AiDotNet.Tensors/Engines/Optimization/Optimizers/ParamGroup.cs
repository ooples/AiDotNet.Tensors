using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// A parameter group: the parameter buffers, their matching gradient buffers, and the
/// per-group hyper-parameters (learning rate, weight decay, betas, …) that the optimizer
/// reads on every step.
/// </summary>
/// <remarks>
/// PyTorch parity: <c>torch.optim.Optimizer.param_groups</c> is a list of dictionaries.
/// We mirror that with a strongly-typed <see cref="ParamGroup"/> whose <see cref="Options"/>
/// dictionary holds the same string-keyed knobs (<c>"lr"</c>, <c>"weight_decay"</c>, etc.).
/// LR schedulers mutate <see cref="LearningRate"/>; users may read/write any
/// other key directly through <see cref="Options"/>.
/// </remarks>
public sealed class ParamGroup
{
    private readonly List<float[]> _params = new List<float[]>();
    private readonly List<float[]> _grads = new List<float[]>();

    /// <summary>Parameters in this group (live references — do not copy).</summary>
    public IReadOnlyList<float[]> Parameters => _params;

    /// <summary>Gradient buffers, one-to-one with <see cref="Parameters"/>.</summary>
    public IReadOnlyList<float[]> Gradients => _grads;

    /// <summary>Free-form, string-keyed hyper-parameter store (parity with PyTorch dict-shape).</summary>
    public Dictionary<string, double> Options { get; } = new Dictionary<string, double>();

    /// <summary>Convenience accessor for <c>Options["lr"]</c>.</summary>
    public double LearningRate
    {
        get => Options["lr"];
        set => Options["lr"] = value;
    }

    /// <summary>Last LR observed by a scheduler call to <c>get_last_lr()</c>.</summary>
    public double LastLearningRate { get; internal set; }

    /// <summary>Add a parameter buffer with its matching gradient buffer.</summary>
    public void AddParameter(float[] parameter, float[] gradient)
    {
        if (parameter == null) throw new ArgumentNullException(nameof(parameter));
        if (gradient == null) throw new ArgumentNullException(nameof(gradient));
        if (parameter.Length != gradient.Length)
            throw new ArgumentException("parameter and gradient buffers must be the same length.");
        _params.Add(parameter);
        _grads.Add(gradient);
    }

    /// <summary>Look up an option, falling back to <paramref name="defaultValue"/> if unset.</summary>
    public double GetOption(string key, double defaultValue) =>
        Options.TryGetValue(key, out var v) ? v : defaultValue;
}
