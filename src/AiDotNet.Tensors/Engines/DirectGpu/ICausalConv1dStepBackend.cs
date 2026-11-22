namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A causal depthwise convolution's decode step behind <see cref="IEngine.CausalConv1dStep{T}"/> as one kernel: a thread
/// per (batch, channel) column reads its K - 1 carried inputs, writes the output and shifts the column in place.
/// </summary>
[BackendSpecificRoute("CUDA one-kernel causal convolution decode step (output, SiLU and in-place history shift in one pass); every backend composes it natively from elementwise ops and a concatenation",
    nameof(IDirectGpuBackend.Multiply), nameof(IDirectGpuBackend.Add), nameof(IDirectGpuBackend.Copy))]
internal interface ICausalConv1dStepBackend
{
    /// <summary>The longest carried history a backend's kernel must accept (kernel size 16); longer ones are composed.</summary>
    internal static class Limits { internal const int MaxTaps = 15; }

    /// <summary>
    /// <c>y = x w[:, 0] + sum_m h[m - 1] w[:, m] (+ bias) (then SiLU)</c>, then <c>h[m] = h[m - 1]</c>, <c>h[0] = x</c> in
    /// place; h is [taps, batch, channels], w [channels, taps + 1], x and y [batch, channels]. bias may be null.
    /// </summary>
    void CausalConv1dStep(IGpuBuffer history, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer? bias, IGpuBuffer output,
        int taps, int batch, int channels, bool silu);
}
