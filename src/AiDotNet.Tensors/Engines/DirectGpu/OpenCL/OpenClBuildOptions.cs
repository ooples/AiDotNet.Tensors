// Copyright (c) AiDotNet. All rights reserved.
// Shared OpenCL build options for consistent kernel compilation.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    internal static class OpenClBuildOptions
    {
        // Aggressive optimization flags for maximum performance.
        public const string OptimizationFlags =
            "-cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-no-signed-zeros";

        // NaN/Inf-preserving flags for kernels whose contract is to propagate
        // NaN (e.g. the relu activation preserves NaN as a numerical-blowup
        // signal — see ActivationKernels relu). Both -cl-fast-relaxed-math and
        // -cl-finite-math-only let the compiler assume no NaN/Inf exists and
        // fold away NaN-detection (isnan AND raw-bit tests alike), so they must
        // be omitted here; -cl-unsafe-math-optimizations is dropped for the same
        // safety. -cl-mad-enable and -cl-no-signed-zeros are NaN-neutral and kept.
        public const string SafeMathFlags = "-cl-mad-enable -cl-no-signed-zeros";
    }
}
