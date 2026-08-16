// Copyright (c) AiDotNet. All rights reserved.
// Capability interface for backends shipping the generic-precision, arbitrary-length FFT path.
// Declared separately from IFftBackend rather than as a default interface method because the library targets
// net471, which has no default-interface-method support.

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Optional capability interface for GPU backends that ship FFT kernels generic over element type and
    /// unrestricted in transform length. Supply-chain-clean: the kernels are custom, with no cuFFT / rocFFT /
    /// clFFT dependency.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a strict superset of <see cref="IFftBackend"/>, which is float32-only and power-of-two-only. Two
    /// limitations of that contract this one removes:
    /// </para>
    /// <list type="number">
    ///   <item><b>Length.</b> <see cref="IFftBackend"/> requires callers with non-power-of-two <c>n</c> to route
    ///     through the CPU Bluestein path, which copies a device-resident tensor to the host and back. Here
    ///     Bluestein runs on the device, so an arbitrary length stays resident.</item>
    ///   <item><b>Precision.</b> <see cref="IFftBackend"/> takes float32 buffers, so a caller holding fp16 or
    ///     bf16 activations must widen before and narrow after - two extra full passes over the tensor. Here the
    ///     buffers carry the caller's element type directly.</item>
    /// </list>
    /// <para>
    /// <b>Arithmetic is always float32</b> whatever <see cref="FftElementType"/> is passed; see the remarks on
    /// that type for why accumulating a transform in a narrow format is not viable. The narrow types buy memory
    /// traffic, which is what these kernels are bound by, and cost nothing in accumulator precision.
    /// </para>
    /// <para>
    /// Contract, matching <see cref="IFftBackend"/> except where noted:
    /// </para>
    /// <list type="bullet">
    ///   <item>Real and imaginary components live in <b>separate</b> buffers, each holding
    ///     <c>batchCount * n</c> elements of the requested type. Both are modified in place.</item>
    ///   <item>Normalization follows the Backward convention (no forward scaling, <c>1/n</c> on inverse);
    ///     callers needing Forward or Ortho apply the extra scale after the call.</item>
    ///   <item>Callers must confirm support with <see cref="SupportsFftElementType"/> first. A backend on
    ///     pre-Ampere hardware cannot compile the bfloat16 specialization, and discovering that as an NVRTC
    ///     failure deep inside a launch is considerably harder to attribute than a false return here.</item>
    /// </list>
    /// </remarks>
    public interface IFftGenericBackend
    {
        /// <summary>
        /// Whether this backend and the current device can execute the requested element type. False for
        /// <see cref="FftElementType.BFloat16"/> below compute capability 8.0 and
        /// <see cref="FftElementType.Float16"/> below 5.3.
        /// </summary>
        bool SupportsFftElementType(FftElementType type);

        /// <summary>
        /// Launch a batched length-<paramref name="n"/> complex FFT across <paramref name="batchCount"/>
        /// independent signals, in place, for any <paramref name="n"/> greater than zero.
        /// </summary>
        /// <param name="real">Real components, <c>batchCount * n</c> elements of <paramref name="type"/>. Modified in place.</param>
        /// <param name="imag">Imaginary components, same length and type. Modified in place.</param>
        /// <param name="batchCount">Number of independent signals.</param>
        /// <param name="n">Transform length. Powers of two take the direct radix-2 path; all other lengths take
        ///   the device-side Bluestein path, which is transparent to the caller and differs only in cost.</param>
        /// <param name="inverse">True for the inverse transform (conjugated twiddles and a <c>1/n</c> scale).</param>
        /// <param name="type">Element type of <paramref name="real"/> and <paramref name="imag"/>.</param>
        void LaunchFftGeneric(
            IGpuBuffer real,
            IGpuBuffer imag,
            int batchCount,
            int n,
            bool inverse,
            FftElementType type);
    }
}
