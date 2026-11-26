using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Whether the current forward is being differentiated, for code that picks an in-place or pooled-buffer fast path
/// the backward cannot differentiate through.
/// </summary>
/// <remarks>
/// A compiled training plan traces the forward with no <see cref="GradientTape{T}"/> active and then differentiates
/// the graph it recorded. A layer that takes an in-place fast path whenever no tape is active would take it during
/// that trace too, and every weight upstream would become a constant with no gradient. Checking
/// <see cref="IsRecording{T}"/> instead of <c>GradientTape&lt;T&gt;.Current</c> covers both cases. An inference trace
/// is not differentiated, so it does not count.
/// </remarks>
public static class GradientRecording
{
    /// <summary>
    /// Declares a forward result observable after a training step. Call this while recording the forward,
    /// before compilation or backward; a compiled plan keeps the result's storage until its next step or disposal.
    /// </summary>
    /// <remarks>Only declared results keep their buffers; ordinary intermediates remain eligible for fusion and reuse.
    /// On an eager tape this has the same effect as <see cref="GradientTape{T}.Retain"/>. Retention does not retain
    /// a gradient or copy the value to the host. The same tensor receives each compiled step's latest value.
    /// A mixed-precision compiled plan keeps them too: it never frees or pages a retained result, and computes one the
    /// loss does not read.</remarks>
    /// <typeparam name="T">The result's element type.</typeparam>
    /// <param name="output">A result produced by the active training forward.</param>
    /// <exception cref="ArgumentNullException"><paramref name="output"/> is null.</exception>
    /// <exception cref="ArgumentException">The result does not belong to the active compiled trace.</exception>
    /// <exception cref="InvalidOperationException">No training forward is being recorded.</exception>
    public static void RetainOutput<T>(Tensor<T> output)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (IsTrainingTrace && GraphMode.Current is { } scope)
        {
            scope.RetainOutput(output);
            return;
        }
        if (GradientTape<T>.Current is { } tape && !NoGradScope<T>.IsSuppressed)
        {
            tape.Retain(output);
            return;
        }
        throw new InvalidOperationException("RetainOutput must be called while recording a training forward, before compilation or backward.");
    }

    /// <summary>Retains a named forward result for a training invocation's output handoff.</summary>
    /// <remarks>Keys are nonempty, ordinal, and unique within the forward. A compiled plan binds each key to
    /// its own result, including when a cached plan is selected again. This does not copy or retain gradients.</remarks>
    /// <typeparam name="T">The result's element type.</typeparam>
    /// <param name="key">The stable output name within this objective.</param>
    /// <param name="output">A result produced by the active training forward.</param>
    public static void RetainOutput<T>(string key, Tensor<T> output)
    {
        if (string.IsNullOrEmpty(key)) throw new ArgumentException("A retained output key must be nonempty.", nameof(key));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (IsTrainingTrace && GraphMode.Current is { } scope)
        {
            scope.RetainOutput(key, output);
            return;
        }
        if (GradientTape<T>.Current is { } tape && !NoGradScope<T>.IsSuppressed)
        {
            RetainedOutputCapture<T>.RegisterEager(tape, key, output);
            tape.RetainOutputForCaller(output);
            return;
        }
        throw new InvalidOperationException("RetainOutput must be called while recording a training forward, before compilation or backward.");
    }

    /// <summary>True while the forward is being traced for a compiled training plan.</summary>
    public static bool IsTrainingTrace => GraphMode.IsActive && !GraphMode.IsInferenceTrace;

    /// <summary>
    /// True while the current forward is being differentiated: an active gradient tape that is not suppressed by a
    /// <see cref="NoGradScope{T}"/>, or a compiled training plan's trace.
    /// </summary>
    public static bool IsRecording<T>() =>
        (GradientTape<T>.Current is not null && !NoGradScope<T>.IsSuppressed) || IsTrainingTrace;
}
