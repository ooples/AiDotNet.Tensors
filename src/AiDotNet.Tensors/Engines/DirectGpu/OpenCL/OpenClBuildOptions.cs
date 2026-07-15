// Copyright (c) AiDotNet. All rights reserved.
// Shared OpenCL build options for consistent kernel compilation.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    internal static class OpenClBuildOptions
    {
        // Aggressive optimization flags for maximum performance (the default).
        private const string AggressiveMathFlags =
            "-cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-no-signed-zeros";

        // NaN/Inf-preserving aggressive flags for kernels whose contract is to propagate
        // NaN (e.g. the relu activation preserves NaN as a numerical-blowup signal — see
        // ActivationKernels relu). Both -cl-fast-relaxed-math and -cl-finite-math-only let
        // the compiler assume no NaN/Inf exists and fold away NaN-detection (isnan AND
        // raw-bit tests alike), so they must be omitted here; -cl-unsafe-math-optimizations
        // is dropped for the same safety. -cl-mad-enable and -cl-no-signed-zeros are
        // NaN-neutral and kept.
        private const string SafeAggressiveMathFlags = "-cl-mad-enable -cl-no-signed-zeros";

        // Precise (IEEE-leaning) flags for opt-in bit-closer CPU/GPU agreement. Drops the
        // fast-math flags that drive the measured divergences: -cl-fast-relaxed-math and
        // -cl-unsafe-math-optimizations (low-precision native transcendentals) and
        // -cl-finite-math-only / -cl-no-signed-zeros (the near-zero flush-to-zero seen on
        // GELU/Erfc). Keeps only -cl-mad-enable — single-rounding FMA, which the CPU paths
        // also use via FusedMultiplyAdd, so it moves the GPU toward the CPU reference rather
        // than away from it. Trades throughput for closer agreement.
        public const string PreciseMathFlags = "-cl-mad-enable";

        /// <summary>
        /// Opt-in precise-math mode. When true, kernels compile WITHOUT the aggressive
        /// fast-math flags, so transcendental/reduction results agree far more closely with
        /// the CPU engine (at a performance cost). Default off. Enable via the
        /// <c>AIDOTNET_GPU_PRECISE_MATH=1</c> environment variable, or by setting this
        /// property before a DirectGpu backend is created. Program binaries are cached per
        /// flag set (the build options are part of the cache key), so switching modes
        /// recompiles rather than reusing stale binaries.
        /// </summary>
        public static bool PreciseMath { get; set; } =
            string.Equals(
                System.Environment.GetEnvironmentVariable("AIDOTNET_GPU_PRECISE_MATH"),
                "1", System.StringComparison.Ordinal);

        // Aggressive optimization flags for maximum performance, or the precise IEEE-leaning
        // flags when PreciseMath is enabled.
        public static string OptimizationFlags => PreciseMath ? PreciseMathFlags : AggressiveMathFlags;

        // NaN/Inf-preserving flags, or the precise flag set when PreciseMath is enabled
        // (-cl-mad-enable alone also preserves NaN).
        public static string SafeMathFlags => PreciseMath ? PreciseMathFlags : SafeAggressiveMathFlags;
    }
}
