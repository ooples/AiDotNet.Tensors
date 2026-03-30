using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Records tensor operations for reverse-mode automatic differentiation.
/// When a GradientTape is active (via <c>using</c>), engine operations automatically
/// record their forward computation and backward functions, enabling gradient computation.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <remarks>
/// <para>Follows the ambient context pattern (like PyTorch's autograd). Operations check
/// <see cref="Current"/> and record only when a tape is active — zero overhead otherwise.</para>
/// <para>Supports nesting: inner tapes record independently and restore the parent on dispose.</para>
/// <para><b>Usage:</b></para>
/// <code>
/// using (var tape = new GradientTape&lt;float&gt;())
/// {
///     var z = engine.TensorMatMul(x, w);    // Recorded
///     var y = engine.ReLU(z);                // Recorded
///     var grads = tape.ComputeGradients(y, sources: new[] { w });
///     var dW = grads[w];
/// }
/// </code>
/// </remarks>
public sealed class GradientTape<T> : IDisposable
{
    /// <summary>
    /// Thread-local current gradient tape for ambient context pattern.
    /// When a tape is active, engine operations record to it automatically.
    /// </summary>
    [ThreadStatic]
    private static GradientTape<T>? _current;

    /// <summary>
    /// Gets the current gradient tape for this thread, if any.
    /// Engine operations check this to decide whether to record.
    /// </summary>
    public static GradientTape<T>? Current => _current;

    /// <summary>
    /// Sets the current tape for this thread. Used by instance methods
    /// to modify the thread-local static field.
    /// </summary>
    private static void SetCurrentTape(GradientTape<T>? tape) => _current = tape;

    private readonly GradientTape<T>? _parent;
    private readonly List<TapeEntry<T>> _entries;
    private readonly GradientTapeOptions _options;
    private bool _disposed;

    /// <summary>
    /// Gets the number of operations recorded on this tape.
    /// </summary>
    public int EntryCount => _entries.Count;

    /// <summary>
    /// Gets the options for this tape.
    /// </summary>
    public GradientTapeOptions Options => _options;

    /// <summary>
    /// Gets whether this tape has been disposed.
    /// </summary>
    public bool IsDisposed => _disposed;

    /// <summary>
    /// Creates a new gradient tape and sets it as the current tape for this thread.
    /// The previous tape (if any) is saved and restored when this tape is disposed.
    /// </summary>
    /// <param name="options">Optional configuration. Uses <see cref="GradientTapeOptions.Default"/> if null.</param>
    public GradientTape(GradientTapeOptions? options = null)
    {
        _options = options ?? GradientTapeOptions.Default;
        _entries = new List<TapeEntry<T>>();
        _parent = _current;
        SetCurrentTape(this);
    }

    /// <summary>
    /// Records a forward operation to the tape. Called by engine operations
    /// via <see cref="DifferentiableOps.RecordIfActive{T}"/>.
    /// </summary>
    /// <param name="entry">The tape entry describing the operation and its backward function.</param>
    /// <exception cref="ObjectDisposedException">Thrown if the tape has been disposed.</exception>
    public void Record(TapeEntry<T> entry)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GradientTape<T>));
        }

        if (_options.MaxEntries > 0 && _entries.Count >= _options.MaxEntries)
        {
            _entries.RemoveAt(0);
        }

        _entries.Add(entry);
    }

    /// <summary>
    /// Computes gradients of <paramref name="loss"/> with respect to <paramref name="sources"/>
    /// by walking the tape in reverse order (reverse-mode automatic differentiation).
    /// </summary>
    /// <param name="loss">The scalar loss tensor to differentiate. Should have a single element.</param>
    /// <param name="sources">Optional set of tensors to compute gradients for.
    /// If null, computes gradients for all input tensors on the tape.</param>
    /// <returns>A dictionary mapping each source tensor to its gradient tensor.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the tape has been disposed.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the tape has no entries.</exception>
    public Dictionary<Tensor<T>, Tensor<T>> ComputeGradients(
        Tensor<T> loss,
        Tensor<T>[]? sources = null)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GradientTape<T>));
        }

        if (_entries.Count == 0)
        {
            throw new InvalidOperationException("Cannot compute gradients: the tape has no recorded operations.");
        }

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Gradient accumulator: maps each tensor (by reference identity) to its accumulated gradient
        var grads = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);

        // Seed: gradient of loss w.r.t. itself is ones with the same shape
        var onesData = new T[loss.Length];
        for (int j = 0; j < onesData.Length; j++)
            onesData[j] = numOps.One;
        var seedGrad = new Tensor<T>(onesData, loss.Shape.ToArray());
        grads[loss] = seedGrad;

        // Walk tape in reverse (reverse-mode AD)
        for (int i = _entries.Count - 1; i >= 0; i--)
        {
            var entry = _entries[i];

            // Skip if we don't have a gradient for this entry's output
            if (!grads.TryGetValue(entry.Output, out var gradOutput))
            {
                continue;
            }

            // Invoke the backward function to propagate gradients to inputs
            entry.Backward(gradOutput, entry.Inputs, entry.Output, entry.SavedState, engine, grads);
        }

        // If sources specified, filter to only those
        if (sources is not null)
        {
            var filtered = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer.Instance);
            foreach (var source in sources)
            {
                if (grads.TryGetValue(source, out var grad))
                {
                    filtered[source] = grad;
                }
            }

            if (!_options.Persistent)
            {
                _entries.Clear();
            }

            return filtered;
        }

        if (!_options.Persistent)
        {
            _entries.Clear();
        }

        return grads;
    }

    /// <summary>
    /// Clears all recorded entries from the tape without disposing it.
    /// </summary>
    public void Reset()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GradientTape<T>));
        }

        _entries.Clear();
    }

    /// <summary>
    /// Disposes the tape and restores the parent tape (if any) as the current tape.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        SetCurrentTape(_parent);
    }
}

/// <summary>
/// Reference equality comparer for use with gradient dictionaries.
/// Ensures tensor identity (not value equality) is used for gradient mapping.
/// </summary>
internal sealed class ReferenceEqualityComparer : IEqualityComparer<object>
{
    public static readonly ReferenceEqualityComparer Instance = new();

    private ReferenceEqualityComparer() { }

    public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);

    public int GetHashCode(object obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
}

/// <summary>
/// Typed reference equality comparer for gradient dictionaries.
/// </summary>
internal sealed class ReferenceEqualityComparer<TItem> : IEqualityComparer<TItem> where TItem : class
{
    public static readonly ReferenceEqualityComparer<TItem> Instance = new();

    private ReferenceEqualityComparer() { }

    public bool Equals(TItem? x, TItem? y) => ReferenceEquals(x, y);

    public int GetHashCode(TItem obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
}
