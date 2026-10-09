using System;
using System.Collections;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

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
    private readonly List<float[]?> _grads = new List<float[]?>();
    private readonly List<Tensor<float>?> _tensors = new List<Tensor<float>?>();
    private readonly GradientList _gradientView;

    /// <summary>Creates an empty group.</summary>
    public ParamGroup() => _gradientView = new GradientList(this);

    /// <summary>Parameters in this group (live references — do not copy).</summary>
    public IReadOnlyList<float[]> Parameters => _params;

    /// <summary>
    /// Gradient buffers, one-to-one with <see cref="Parameters"/>. A parameter added as a tensor gets its buffer
    /// on first access here; one stepped through <see cref="OptimizerBase.Step(IReadOnlyDictionary{Tensor{float}, Tensor{float}})"/>
    /// never needs it.
    /// </summary>
    public IReadOnlyList<float[]> Gradients => _gradientView;

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
        _tensors.Add(null);
    }

    /// <summary>
    /// Add a parameter tensor, updated in place. Its gradient is taken from the dictionary passed to
    /// <see cref="OptimizerBase.Step(IReadOnlyDictionary{Tensor{float}, Tensor{float}})"/> — the tape's result,
    /// with no copy — or from <see cref="Gradients"/> on a plain <see cref="OptimizerBase.Step()"/>. The tensor is
    /// marked modified after every step that updates it.
    /// </summary>
    /// <param name="parameter">A contiguous CPU float tensor that owns its whole storage array.</param>
    public void AddParameter(Tensor<float> parameter)
    {
        if (parameter == null) throw new ArgumentNullException(nameof(parameter));
        _params.Add(ResolveStorage(parameter));
        _grads.Add(null);
        _tensors.Add(parameter);
    }

    /// <summary>The tensor a parameter was added as, or null for one added as an array.</summary>
    internal Tensor<float>? ParameterTensor(int index) => _tensors[index];

    /// <summary>The gradient buffer if it exists, without creating one for a tensor parameter.</summary>
    internal float[]? PeekGradient(int index) => _grads[index];

    /// <summary>
    /// Re-reads a tensor parameter's storage before a step: a copy-on-write privatization since the last step
    /// moves the tensor to a new array, and writing the old one would change the peer it was shared with.
    /// </summary>
    internal void RefreshTensorParameters()
    {
        for (int i = 0; i < _tensors.Count; i++)
        {
            var tensor = _tensors[i];
            if (tensor is not null) _params[i] = ResolveStorage(tensor);
        }
    }

    private static float[] ResolveStorage(Tensor<float> parameter)
    {
        var storage = parameter.GetCpuBackingForContiguousWrite(out int offset);
        if (storage is null || offset != 0 || storage.Length != parameter.Length)
            throw new ArgumentException(
                "An optimizer parameter must be a contiguous CPU tensor that owns its whole storage array " +
                "(not a view, a slice, or a pooled buffer).", nameof(parameter));
        return storage;
    }

    private sealed class GradientList : IReadOnlyList<float[]>
    {
        private readonly ParamGroup _group;

        public GradientList(ParamGroup group) => _group = group;

        public int Count => _group._grads.Count;

        public float[] this[int index]
        {
            get
            {
                var gradient = _group._grads[index];
                if (gradient is null)
                {
                    gradient = new float[_group._params[index].Length];
                    _group._grads[index] = gradient;
                }
                return gradient;
            }
        }

        public IEnumerator<float[]> GetEnumerator()
        {
            for (int i = 0; i < Count; i++) yield return this[i];
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    /// <summary>Look up an option, falling back to <paramref name="defaultValue"/> if unset.</summary>
    public double GetOption(string key, double defaultValue) =>
        Options.TryGetValue(key, out var v) ? v : defaultValue;
}
