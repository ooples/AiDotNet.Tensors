// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.LinearAlgebra.Fft;

/// <summary>
/// FFT normalization convention. Matches <c>torch.fft</c>'s <c>norm</c> argument.
/// </summary>
/// <remarks>
/// For a length-<c>N</c> transform the scale factor <c>a_forward</c> is applied
/// to the forward output and <c>a_inverse</c> to the inverse output:
/// <list type="bullet">
///   <item><term>Backward</term>
///         <description><c>a_forward = 1</c>, <c>a_inverse = 1/N</c> (default, matches numpy.fft / PyTorch default).</description></item>
///   <item><term>Forward</term>
///         <description><c>a_forward = 1/N</c>, <c>a_inverse = 1</c>.</description></item>
///   <item><term>Ortho</term>
///         <description><c>a_forward = 1/√N</c>, <c>a_inverse = 1/√N</c> (unitary transform — Parseval holds without extra scaling).</description></item>
/// </list>
/// </remarks>
public enum FftNorm
{
    /// <summary>Forward unscaled, inverse scaled by <c>1/N</c>.</summary>
    Backward,
    /// <summary>Forward scaled by <c>1/N</c>, inverse unscaled.</summary>
    Forward,
    /// <summary>Both scaled by <c>1/√N</c> — unitary (Parseval-preserving).</summary>
    Ortho,
}
