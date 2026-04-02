using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Provides all information an optimizer needs to perform a parameter update step
/// during tape-based training. Supports both first-order and second-order optimizers
/// through a unified interface that satisfies the Liskov Substitution Principle.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <remarks>
/// <para>
/// This is the central data structure for the optimizer <c>Step</c> method, replacing
/// the raw <c>(parameters, gradients)</c> tuple. It provides:
/// <list type="bullet">
/// <item><b>First-order data</b>: Parameters, gradients, and loss — sufficient for SGD, Adam, etc.</item>
/// <item><b>Re-evaluation closure</b>: Allows second-order optimizers (L-BFGS, Trust Region) to
/// re-run forward+backward at current parameter values for line search, without rebuilding
/// the computation graph from scratch.</item>
/// <item><b>Hessian-vector products</b>: Enables Newton-CG and Trust Region methods to compute
/// curvature information directly from the tape using forward-over-reverse AD.</item>
/// <item><b>Parameter buffer integration</b>: When backed by a <see cref="ParameterBuffer{T}"/>,
/// second-order optimizers get zero-copy access to the flat parameter vector.</item>
/// </list>
/// </para>
/// <para><b>Performance advantages over PyTorch:</b>
/// <list type="bullet">
/// <item><b>Gradient buffer reuse</b>: Gradient accumulator tensors are pre-allocated on first
/// re-evaluation and reused (with TensorCopy) on subsequent calls, eliminating per-step allocation.
/// PyTorch's closure pattern allocates new gradient tensors each call.</item>
/// <item><b>Stale gradient protection</b>: When a parameter stops producing gradients during
/// re-evaluation, cached buffers are zeroed to prevent stale values from leaking into the optimizer.</item>
/// <item><b>Integrated HVP</b>: Hessian-vector products are a first-class capability, not
/// a user-assembled workaround as in PyTorch's <c>torch.autograd.functional.hvp</c>.</item>
/// <item><b>Contiguous parameter buffer</b>: When a <see cref="ParameterBuffer{T}"/> is attached,
/// <c>AsVector()</c> returns the flat parameter vector with zero copies.</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Think of this as a "training toolkit" given to the optimizer.
/// Simple optimizers (like Adam) only use the basic tools — the parameters and their gradients.
/// Advanced optimizers (like L-BFGS) also use the re-evaluation tool to try different step sizes,
/// and the curvature tool to understand the shape of the loss landscape.
/// </para>
/// </remarks>
public sealed class TapeStepContext<T>
{
    private readonly Func<Tensor<T>, Tensor<T>, Tensor<T>>? _forwardFn;
    private readonly Func<Tensor<T>, Tensor<T>, Tensor<T>>? _lossFn;
    private readonly Tensor<T>? _input;
    private readonly Tensor<T>? _target;
    private readonly IEngine _engine;

    // Cached gradient buffers for reuse across re-evaluations
    private Dictionary<Tensor<T>, Tensor<T>>? _cachedGradBuffers;
    private int _evaluationCount;

    /// <summary>
    /// The trainable parameter tensors — same references used in the layer's Forward pass.
    /// Optimizers modify these in-place.
    /// </summary>
    public Tensor<T>[] Parameters { get; }

    /// <summary>
    /// Gradients of the loss with respect to each parameter, keyed by tensor reference identity.
    /// Computed by the gradient tape's reverse-mode AD.
    /// </summary>
    public Dictionary<Tensor<T>, Tensor<T>> Gradients { get; private set; }

    /// <summary>
    /// The scalar loss value from the most recent forward evaluation.
    /// </summary>
    public T Loss { get; private set; }

    /// <summary>
    /// Optional contiguous parameter buffer. When set, second-order optimizers can use
    /// <see cref="ParameterBuffer"/> to access the flat parameter/gradient vectors with zero copies.
    /// </summary>
    public ParameterBuffer<T>? ParameterBuffer { get; }

    /// <summary>
    /// Number of times <see cref="Reevaluate"/> has been called. Used for profiling
    /// and to detect excessive line search iterations.
    /// </summary>
    public int EvaluationCount => _evaluationCount;

    /// <summary>
    /// Creates a context for first-order optimizers (no re-evaluation capability).
    /// </summary>
    /// <param name="parameters">Trainable parameter tensors.</param>
    /// <param name="gradients">Gradient dictionary from tape.</param>
    /// <param name="loss">Scalar loss value.</param>
    /// <param name="parameterBuffer">Optional contiguous parameter buffer for zero-copy flat access.</param>
    public TapeStepContext(
        Tensor<T>[] parameters,
        Dictionary<Tensor<T>, Tensor<T>> gradients,
        T loss,
        ParameterBuffer<T>? parameterBuffer = null)
    {
        if (parameterBuffer is not null)
            ValidateBufferAlignment(parameters, parameterBuffer);

        Parameters = parameters;
        Gradients = gradients;
        Loss = loss;
        ParameterBuffer = parameterBuffer;
        _engine = AiDotNetEngine.Current;
        _evaluationCount = 1;
    }

    /// <summary>
    /// Creates a full context with re-evaluation and HVP capability for second-order optimizers.
    /// </summary>
    /// <param name="parameters">Trainable parameter tensors.</param>
    /// <param name="gradients">Initial gradient dictionary from tape.</param>
    /// <param name="loss">Initial scalar loss value.</param>
    /// <param name="input">The training input tensor (cached for re-evaluation).</param>
    /// <param name="target">The training target tensor (cached for re-evaluation).</param>
    /// <param name="forwardFn">Forward pass function: (input, target) → prediction.</param>
    /// <param name="lossFn">Loss function: (prediction, target) → scalar loss tensor.</param>
    /// <param name="parameterBuffer">Optional contiguous parameter buffer for zero-copy flat access.</param>
    public TapeStepContext(
        Tensor<T>[] parameters,
        Dictionary<Tensor<T>, Tensor<T>> gradients,
        T loss,
        Tensor<T> input,
        Tensor<T> target,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> forwardFn,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> lossFn,
        ParameterBuffer<T>? parameterBuffer = null)
    {
        if (parameterBuffer is not null)
            ValidateBufferAlignment(parameters, parameterBuffer);

        Parameters = parameters;
        Gradients = gradients;
        Loss = loss;
        _input = input;
        _target = target;
        _forwardFn = forwardFn;
        _lossFn = lossFn;
        ParameterBuffer = parameterBuffer;
        _engine = AiDotNetEngine.Current;
        _evaluationCount = 1;
    }

    /// <summary>
    /// Whether this context supports re-evaluation (forward+backward at current parameter values).
    /// True when constructed with forward/loss functions for second-order optimizer support.
    /// </summary>
    public bool SupportsReevaluation => _forwardFn is not null && _lossFn is not null;

    /// <summary>
    /// Re-evaluates the forward pass and gradients at the current parameter values.
    /// Used by second-order optimizers (L-BFGS, Trust Region) for line search.
    /// </summary>
    /// <returns>The new loss value after re-evaluation.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when this context was created without forward/loss functions.
    /// </exception>
    public T Reevaluate()
    {
        if (_forwardFn is null || _lossFn is null || _input is null || _target is null)
            throw new InvalidOperationException(
                "This TapeStepContext does not support re-evaluation. " +
                "Create it with forward/loss functions for second-order optimizer support.");

        _evaluationCount++;

        // Run forward + backward under a fresh tape
        using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = false });
        var prediction = _forwardFn(_input, _target);
        var lossTensor = _lossFn(prediction, _target);

        var grads = tape.ComputeGradients(lossTensor, Parameters);

        // Reuse cached gradient buffers: copy new grads into existing buffers to avoid allocation
        if (_cachedGradBuffers is not null)
        {
            foreach (var param in Parameters)
            {
                var hasNewGrad = grads.TryGetValue(param, out var newGrad);
                var hasCached = _cachedGradBuffers.TryGetValue(param, out var cachedBuf);

                if (hasNewGrad && hasCached)
                {
                    _engine.TensorCopy(newGrad, cachedBuf);
                }
                else if (hasNewGrad)
                {
                    _cachedGradBuffers[param] = newGrad;
                }
                else if (hasCached)
                {
                    // Parameter stopped producing gradients — zero the cached buffer
                    // to prevent stale gradients from leaking into the optimizer step
                    _engine.TensorFill(cachedBuf, MathHelper.GetNumericOperations<T>().Zero);
                }
            }
            Gradients = _cachedGradBuffers;
        }
        else
        {
            _cachedGradBuffers = grads;
            Gradients = grads;
        }

        if (lossTensor.Length != 1)
            throw new InvalidOperationException(
                $"Loss tensor must be scalar (length 1), got length {lossTensor.Length}. " +
                "Reduce the loss to a scalar before computing gradients.");

        Loss = lossTensor[0];

        return Loss;
    }

    /// <summary>
    /// Computes the Hessian-vector product H*v for each parameter, where H is the Hessian
    /// of the loss with respect to the parameters. Uses forward-over-reverse AD (double
    /// backward) which is O(n) per product, not O(n²).
    /// </summary>
    /// <param name="vectors">Direction vectors for each parameter (same shape as parameters).</param>
    /// <returns>Hessian-vector products keyed by parameter tensor identity.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when this context does not support re-evaluation.
    /// </exception>
    public Dictionary<Tensor<T>, Tensor<T>> HessianVectorProduct(Dictionary<Tensor<T>, Tensor<T>> vectors)
    {
        if (_forwardFn is null || _lossFn is null || _input is null || _target is null)
            throw new InvalidOperationException(
                "HVP requires re-evaluation capability. Create this context with forward/loss functions.");

        // Step 1: Forward + backward with createGraph=true to record gradient ops
        using var outerTape = new GradientTape<T>(new GradientTapeOptions { Persistent = true });
        var prediction = _forwardFn(_input, _target);
        var lossTensor = _lossFn(prediction, _target);
        var grads = outerTape.ComputeGradients(lossTensor, Parameters, createGraph: true);

        // Step 2: Compute dot product grad · v (scalar)
        Tensor<T>? dotProduct = null;
        foreach (var param in Parameters)
        {
            if (grads.TryGetValue(param, out var g) && vectors.TryGetValue(param, out var v))
            {
                var product = _engine.TensorMultiply(g, v);
                var allAxes = Enumerable.Range(0, product.Shape.Length).ToArray();
                var sum = _engine.ReduceSum(product, allAxes, keepDims: false);
                dotProduct = dotProduct is null ? sum : _engine.TensorAdd(dotProduct, sum);
            }
        }

        if (dotProduct is null)
            return new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);

        // Step 3: Compute gradient of (grad · v) w.r.t. parameters = H*v
        var hvp = outerTape.ComputeGradients(dotProduct, Parameters);
        return hvp;
    }

    /// <summary>
    /// Returns the flat parameter vector. If a <see cref="ParameterBuffer"/> is attached,
    /// this is zero-copy. Otherwise, flattens parameter tensors into a new vector.
    /// Non-contiguous parameters are materialized to contiguous form before flattening.
    /// </summary>
    public Vector<T> GetFlatParameters()
    {
        if (ParameterBuffer is not null)
            return ParameterBuffer.AsVector();

        // Fallback: flatten manually with contiguity handling
        int total = 0;
        foreach (var p in Parameters) total += p.Length;
        var flat = new Vector<T>(total);
        var span = flat.AsWritableSpan();
        int offset = 0;
        foreach (var p in Parameters)
        {
            var contiguous = p.IsContiguous ? p : p.Contiguous();
            var src = contiguous.DataVector.AsSpan().Slice(contiguous._storageOffset, contiguous.Length);
            src.CopyTo(span.Slice(offset, contiguous.Length));
            offset += p.Length;
        }
        return flat;
    }

    /// <summary>
    /// Returns the flat gradient vector aligned with the parameter layout.
    /// If a <see cref="ParameterBuffer"/> is attached, uses its efficient flattening.
    /// Non-contiguous gradients are materialized to contiguous form before flattening.
    /// </summary>
    public Vector<T> GetFlatGradients()
    {
        if (ParameterBuffer is not null)
            return ParameterBuffer.FlattenGradients(Parameters, Gradients);

        // Fallback: flatten manually with contiguity handling
        int total = 0;
        foreach (var p in Parameters) total += p.Length;
        var flat = new Vector<T>(total);
        var span = flat.AsWritableSpan();
        int offset = 0;
        foreach (var p in Parameters)
        {
            if (Gradients.TryGetValue(p, out var grad))
            {
                var contiguous = grad.IsContiguous ? grad : grad.Contiguous();
                var src = contiguous.DataVector.AsSpan().Slice(contiguous._storageOffset, contiguous.Length);
                int copyLen = Math.Min(src.Length, p.Length);
                src.Slice(0, copyLen).CopyTo(span.Slice(offset, copyLen));
            }
            offset += p.Length;
        }
        return flat;
    }

    /// <summary>
    /// Writes a flat parameter vector back into the parameter tensors.
    /// If a <see cref="ParameterBuffer"/> is attached, this is a single buffer copy.
    /// Otherwise, distributes values into individual parameter tensors.
    /// Requires contiguous parameters for direct write-back.
    /// </summary>
    /// <param name="flatParams">The flat parameter vector to write back.</param>
    public void SetFlatParameters(Vector<T> flatParams)
    {
        if (ParameterBuffer is not null)
        {
            ParameterBuffer.CopyFrom(flatParams);
            return;
        }

        // Validate total length
        int total = 0;
        foreach (var p in Parameters) total += p.Length;
        if (flatParams.Length != total)
            throw new ArgumentException(
                $"Flat parameter vector length ({flatParams.Length}) must match total parameter count ({total}).",
                nameof(flatParams));

        // Distribute into individual tensors
        var src = flatParams.AsSpan();
        int offset = 0;
        foreach (var p in Parameters)
        {
            if (!p.IsContiguous)
                throw new InvalidOperationException(
                    $"SetFlatParameters requires contiguous parameter tensors. " +
                    $"Call Contiguous() on non-contiguous parameters before using flat access.");
            var dst = p.DataVector.AsWritableSpan().Slice(p._storageOffset, p.Length);
            src.Slice(offset, p.Length).CopyTo(dst);
            offset += p.Length;
        }
    }

    private static void ValidateBufferAlignment(Tensor<T>[] parameters, ParameterBuffer<T> buffer)
    {
        if (buffer.Count != parameters.Length)
            throw new ArgumentException(
                $"ParameterBuffer has {buffer.Count} slots but {parameters.Length} parameters were provided. " +
                "The buffer must have been created with the same parameter shapes in the same order.");

        int expectedTotal = 0;
        foreach (var p in parameters) expectedTotal += p.Length;
        if (buffer.TotalSize != expectedTotal)
            throw new ArgumentException(
                $"ParameterBuffer total size ({buffer.TotalSize}) does not match parameter total ({expectedTotal}).");
    }
}
