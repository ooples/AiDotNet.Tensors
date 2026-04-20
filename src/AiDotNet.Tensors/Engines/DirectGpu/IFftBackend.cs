// Copyright (c) AiDotNet. All rights reserved.
// Secondary interface for backends that ship native FFT kernels. Follows the
// same pattern as IParity210Backend / ILinalgBackend: if a backend implements
// this interface, the engine dispatches FFT ops to the custom GPU kernel;
// otherwise the call transparently falls through to the CPU path.

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Optional capability interface for GPU backends that ship native radix-2
    /// Cooley-Tukey FFT kernels. Supply-chain-clean: the kernels are custom
    /// (no cuFFT / rocFFT / clFFT dependency) and live alongside each backend.
    ///
    /// Contract:
    /// <list type="bullet">
    ///   <item>Input/output buffers: interleaved <c>re/im</c> doubles packed
    ///     into <see cref="IGpuBuffer"/> slots, <c>2·batchCount·n</c> total
    ///     elements each. Input and output may be the same buffer (in-place).</item>
    ///   <item><paramref name="n"/> must be a power of two (scalar kernels
    ///     only ship radix-2). Callers that need non-pow-2 must route through
    ///     the CPU Bluestein path.</item>
    ///   <item>Normalization follows the Backward convention (no forward
    ///     scaling, <c>1/n</c> on inverse); callers that need Forward / Ortho
    ///     apply the extra scale post-launch.</item>
    /// </list>
    /// </summary>
    public interface IFftBackend
    {
        /// <summary>
        /// Launch a batched length-<paramref name="n"/> FFT across
        /// <paramref name="batchCount"/> independent signals, in place.
        /// </summary>
        /// <param name="buffer">Interleaved re/im doubles, <c>2·batchCount·n</c> elements. Modified in place.</param>
        /// <param name="batchCount">Number of independent signals.</param>
        /// <param name="n">Transform length (power of two).</param>
        /// <param name="inverse">True for IFFT (conjugated twiddles + 1/n scale), false for forward FFT.</param>
        void LaunchFft(IGpuBuffer buffer, int batchCount, int n, bool inverse);
    }
}
