// Copyright (c) AiDotNet. All rights reserved.
// Storage element type for the generic GPU FFT path.

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Storage precision for GPU FFT buffers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This selects the precision of the <b>buffers</b> only. Arithmetic inside every kernel is performed in
    /// <c>float</c> regardless of this setting, and that is deliberate rather than conservative: an
    /// <c>n</c>-point Cooley-Tukey transform is a chain of <c>log2(n)</c> accumulation stages, and accumulating
    /// in a format with an 8-bit mantissa loses roughly one significant bit per stage. A 1024-point transform
    /// accumulated in bfloat16 retains almost nothing.
    /// </para>
    /// <para>
    /// The benefit of a narrow element type is therefore <b>bandwidth</b>, not arithmetic: each butterfly reads
    /// two complex values and writes two back, so halving the element width halves the bytes moved. On the
    /// small transforms typical of spectral neural operators the kernels are memory-bound, so the traffic
    /// saving is the whole win and the fp32 accumulator costs nothing but registers.
    /// </para>
    /// </remarks>
    public enum FftElementType
    {
        /// <summary>IEEE 754 binary32. 4 bytes per component.</summary>
        Float32 = 0,

        /// <summary>IEEE 754 binary16 (<c>__half</c>). 2 bytes per component; 10-bit mantissa, limited exponent range.</summary>
        Float16 = 1,

        /// <summary>
        /// bfloat16 (<c>__nv_bfloat16</c>). 2 bytes per component; 7-bit mantissa but the <b>same exponent range
        /// as float32</b>, which is why it tolerates the unnormalised magnitudes that appear mid-transform where
        /// <see cref="Float16"/> can overflow.
        /// </summary>
        BFloat16 = 2,
    }

    /// <summary>
    /// Helpers for <see cref="FftElementType"/>.
    /// </summary>
    public static class FftElementTypeExtensions
    {
        /// <summary>Bytes occupied by one real component (a complex value uses twice this).</summary>
        public static int ByteSize(this FftElementType type) => type switch
        {
            FftElementType.Float32 => 4,
            FftElementType.Float16 => 2,
            FftElementType.BFloat16 => 2,
            _ => throw new System.ArgumentOutOfRangeException(nameof(type), type, "Unknown FFT element type."),
        };

        /// <summary>
        /// Kernel-name suffix for the specialization compiled for this type, e.g. <c>"_bf16"</c>. Kernel sources
        /// are emitted once per element type with this suffix appended to every entry point.
        /// </summary>
        public static string KernelSuffix(this FftElementType type) => type switch
        {
            FftElementType.Float32 => "_f32",
            FftElementType.Float16 => "_f16",
            FftElementType.BFloat16 => "_bf16",
            _ => throw new System.ArgumentOutOfRangeException(nameof(type), type, "Unknown FFT element type."),
        };

        /// <summary>
        /// Minimum CUDA compute capability, times ten, required for this element type.
        /// <see cref="Float16"/> needs sm_53 for native half arithmetic; <see cref="BFloat16"/> needs sm_80,
        /// because the bfloat16 conversion intrinsics are only defined from Ampere onward. Callers must check
        /// this before requesting a specialization - the alternative is a compile failure inside NVRTC that is
        /// far harder to attribute.
        /// </summary>
        public static int MinComputeCapabilityX10(this FftElementType type) => type switch
        {
            FftElementType.Float32 => 30,
            FftElementType.Float16 => 53,
            FftElementType.BFloat16 => 80,
            _ => throw new System.ArgumentOutOfRangeException(nameof(type), type, "Unknown FFT element type."),
        };
    }
}
