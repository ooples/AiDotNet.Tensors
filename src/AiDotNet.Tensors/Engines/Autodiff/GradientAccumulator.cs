using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Accumulates gradients across multiple mini-batches for gradient accumulation training.
/// Like PyTorch's tensor.grad, but as a separate object to avoid modifying the Tensor class.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>Usage:</para>
/// <code>
/// var accumulator = new GradientAccumulator&lt;float&gt;(engine);
/// accumulator.Register(weights);
///
/// for (int i = 0; i &lt; accumSteps; i++)
/// {
///     using var tape = new GradientTape&lt;float&gt;();
///     var loss = ComputeLoss(weights, data[i]);
///     var grads = tape.ComputeGradients(loss, sources: new[] { weights });
///     accumulator.Accumulate(grads);
/// }
///
/// accumulator.Step(learningRate: 0.01f);
/// accumulator.ZeroGrad();
/// </code>
/// </remarks>
public sealed class GradientAccumulator<T>
{
    private readonly Dictionary<Tensor<T>, Tensor<T>> _gradients;
    private readonly IEngine _engine;
    private readonly INumericOperations<T> _numOps;
    private int _accumCount;

    /// <summary>
    /// Creates a new gradient accumulator.
    /// </summary>
    public GradientAccumulator(IEngine? engine = null)
    {
        _engine = engine ?? AiDotNetEngine.Current;
        _numOps = MathHelper.GetNumericOperations<T>();
        _gradients = new Dictionary<Tensor<T>, Tensor<T>>(
            ReferenceEqualityComparer<Tensor<T>>.Instance);
    }

    /// <summary>
    /// Registers a parameter tensor for gradient accumulation.
    /// </summary>
    public void Register(Tensor<T> parameter)
    {
        if (!_gradients.ContainsKey(parameter))
        {
            var grad = new Tensor<T>(new T[parameter.Length], parameter.Shape.ToArray());
            _gradients[parameter] = grad;
        }
    }

    /// <summary>
    /// Registers multiple parameter tensors.
    /// </summary>
    public void Register(params Tensor<T>[] parameters)
    {
        foreach (var p in parameters) Register(p);
    }

    /// <summary>
    /// Accumulates gradients from a backward pass into the stored gradients.
    /// </summary>
    /// <param name="grads">Gradient dictionary from GradientTape.ComputeGradients().</param>
    public void Accumulate(Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        foreach (var kvp in grads)
        {
            if (_gradients.TryGetValue(kvp.Key, out var accumulated))
            {
                _engine.TensorAddInPlace(accumulated, kvp.Value);
            }
        }
        _accumCount++;
    }

    /// <summary>
    /// Applies accumulated gradients to parameters using SGD and zeroes gradients.
    /// Automatically divides by accumulation count for correct averaging.
    /// </summary>
    /// <param name="learningRate">The learning rate for SGD update.</param>
    public void Step(T learningRate)
    {
        T scale = _numOps.Divide(learningRate, _numOps.FromDouble(Math.Max(_accumCount, 1)));

        foreach (var kvp in _gradients)
        {
            var param = kvp.Key;
            var grad = kvp.Value;

            // param -= (lr / accumCount) * grad
            for (int i = 0; i < param.Length && i < grad.Length; i++)
                param[i] = _numOps.Subtract(param[i], _numOps.Multiply(scale, grad[i]));
        }

        ZeroGrad();
    }

    /// <summary>
    /// Zeroes all accumulated gradients.
    /// </summary>
    public void ZeroGrad()
    {
        foreach (var grad in _gradients.Values)
        {
            for (int i = 0; i < grad.Length; i++)
                grad[i] = _numOps.Zero;
        }
        _accumCount = 0;
    }

    /// <summary>
    /// Gets the accumulated gradient for a parameter, or null if not registered.
    /// </summary>
    public Tensor<T>? GetGrad(Tensor<T> parameter)
    {
        return _gradients.TryGetValue(parameter, out var grad) ? grad : null;
    }

    /// <summary>
    /// Gets the number of accumulation steps since last ZeroGrad.
    /// </summary>
    public int AccumulationCount => _accumCount;
}
