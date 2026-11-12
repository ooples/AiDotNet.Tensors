using AiDotNet.Tensors.Engines.Compilation;

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
    /// <summary>True while the forward is being traced for a compiled training plan.</summary>
    public static bool IsTrainingTrace => GraphMode.IsActive && !GraphMode.IsInferenceTrace;

    /// <summary>
    /// True while the current forward is being differentiated: an active gradient tape that is not suppressed by a
    /// <see cref="NoGradScope{T}"/>, or a compiled training plan's trace.
    /// </summary>
    public static bool IsRecording<T>() =>
        (GradientTape<T>.Current is not null && !NoGradScope<T>.IsSuppressed) || IsTrainingTrace;
}
