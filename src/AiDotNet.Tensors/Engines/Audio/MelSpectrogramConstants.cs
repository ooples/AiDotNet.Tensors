namespace AiDotNet.Tensors.Engines.Audio;

/// <summary>
/// Flooring constants shared by the MelSpectrogram forward and its backward.
/// </summary>
/// <remarks>
/// <para>
/// These were previously duplicated as local literals in both
/// <c>CpuEngine.MelSpectrogram</c> and <c>BackwardFunctions{T}.MelSpectrogramBackward</c>.
/// That is a silent-failure hazard specific to this pair: the backward has to zero the gradient
/// exactly where the forward's clamps saturate, so changing a floor on the forward side alone
/// would leave the backward zeroing at the wrong threshold — producing a wrong gradient rather
/// than an error. Keeping one definition makes the two provably agree.
/// </para>
/// </remarks>
internal static class MelSpectrogramConstants
{
    /// <summary>
    /// Lower bound applied to the linear mel power before the log, so <c>log10(0)</c> is never
    /// evaluated. Gradient is zero at or below this floor because the output is locally constant.
    /// </summary>
    internal const double PowerFloor = 1e-10;

    /// <summary>
    /// Lower bound applied to the decibel output. Gradient is zero once the output is clamped
    /// here, for the same reason.
    /// </summary>
    internal const double MinDb = -80.0;
}
