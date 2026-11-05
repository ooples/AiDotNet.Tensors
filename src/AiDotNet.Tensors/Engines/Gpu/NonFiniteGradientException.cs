namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Thrown by a global-norm clip under <see cref="NonFiniteGradientPolicy.Error"/> when a gradient element is NaN or
/// infinite: PyTorch's <c>clip_grad_norm_(..., error_if_nonfinite=True)</c>, with the first offending gradient and the
/// counts. The gradients are left as they were and no optimizer update runs.
/// </summary>
public sealed class NonFiniteGradientException : ArithmeticException
{
    /// <summary>Creates the exception for the clip's findings.</summary>
    /// <param name="firstNonFiniteGradient">The index, in the clipped gradient list, of the first gradient holding a
    /// NaN or infinite element (-1 when unknown).</param>
    /// <param name="nanCount">NaN elements across every gradient.</param>
    /// <param name="infinityCount">Infinite elements across every gradient.</param>
    public NonFiniteGradientException(int firstNonFiniteGradient, long nanCount, long infinityCount)
        : base($"The total norm of order 2 for the gradients is non-finite, so it cannot be clipped: gradient " +
               $"{firstNonFiniteGradient} is the first holding a non-finite element ({nanCount} NaN, {infinityCount} " +
               "infinite in all). The gradients are left unscaled. Choose another NonFiniteGradientPolicy to scale or " +
               "repair them instead.")
    {
        FirstNonFiniteGradient = firstNonFiniteGradient;
        NanCount = nanCount;
        InfinityCount = infinityCount;
    }

    /// <summary>The index, in the clipped gradient list, of the first gradient holding a NaN or infinite element.</summary>
    public int FirstNonFiniteGradient { get; }

    /// <summary>NaN elements across every gradient.</summary>
    public long NanCount { get; }

    /// <summary>Infinite elements across every gradient.</summary>
    public long InfinityCount { get; }
}
