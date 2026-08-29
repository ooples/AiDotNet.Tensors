// Copyright (c) AiDotNet. All rights reserved.
// HIP launcher shim for categorical sampling. The HIP launch API takes the same void**-array as
// CUDA, so this mirrors the OpenCL launcher one-for-one apart from the runtime call.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP
{
    /// <summary>
    /// Gives HIP a device route for <c>TensorCategoricalSample</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Ported alongside the OpenCL kernel rather than after it. Adding a compute kernel to one
    /// backend and not the others opens a parity gap that
    /// <c>CrossBackendKernelCoverageTests.MirrorBackends_CoverEveryOpenClComputeKernel</c> fails on,
    /// and rightly so: an op that runs on device for one vendor and silently falls back to the CPU
    /// for another is a portability bug, not an optimisation.
    /// </para>
    /// <para>
    /// The kernel is bit-identical to the OpenCL and CPU implementations — the same StatelessRandom
    /// PCG hash keyed on (seed, row), the same inverse-CDF walk accumulated in <c>double</c>, the
    /// same <c>target &lt; cumulative</c> ordering. That is what lets the same seed produce the same
    /// one-hot on every backend, which the exact-parity test requires.
    /// </para>
    /// <para>
    /// NOT EXECUTED IN VERIFICATION: there is no AMD ROCm device on the machine this was written on,
    /// so the kernel is verified to compile and register and to close the mirror gap, but its output
    /// has not been observed. The algorithm is a line-for-line port of the OpenCL kernel, which IS
    /// verified against the CPU oracle.
    /// </para>
    /// </remarks>
    public sealed partial class HipBackend : ICategoricalSamplingBackend
    {
        /// <inheritdoc/>
        public bool CanCategoricalSample(int rows, int classes)
            => _categoricalModule != IntPtr.Zero
               && rows > 0
               && classes > 0
               && (long)rows * classes <= int.MaxValue
               && _kernelCache.ContainsKey("categorical_sample");

        /// <inheritdoc/>
        public unsafe bool TryCategoricalSample(
            IGpuBuffer probabilities,
            IGpuBuffer oneHot,
            int rows,
            int classes,
            ulong seed)
        {
            if (!CanCategoricalSample(rows, classes)) return false;
            if (probabilities is null || oneHot is null) return false;
            if (!_kernelCache.TryGetValue("categorical_sample", out var kernel)) return false;

            // Both buffers are indexed as row * classes + c, so both must actually hold that many
            // elements. CanCategoricalSample only checks the DIMENSIONS; an undersized buffer would
            // be written past its end on the device, which corrupts whatever is next in device
            // memory and is reported, if at all, by some unrelated later operation. CUDA already
            // rejects this via ValidateExactCategoricalBuffer and OpenCL via the same pair of
            // comparisons below; HIP was the one route that launched unchecked.
            long addressed = (long)rows * classes;
            GpuKernelDiagnostics.ValidateCapacity(
                "categorical_sample", nameof(probabilities), probabilities.Size, addressed);
            GpuKernelDiagnostics.ValidateCapacity(
                "categorical_sample", nameof(oneHot), oneHot.Size, addressed);

            if (probabilities.Size < addressed || oneHot.Size < addressed)
            {
                // Even with deep checks off, refuse rather than launch an out-of-range write.
                return false;
            }

            uint grid = (uint)((rows + DefaultBlockSize - 1) / DefaultBlockSize);
            IntPtr probabilitiesHandle = probabilities.Handle;
            IntPtr oneHotHandle = oneHot.Handle;
            int rowCount = rows;
            int classCount = classes;
            ulong deviceSeed = seed;

            void** args = stackalloc void*[5];
            args[0] = &probabilitiesHandle;
            args[1] = &oneHotHandle;
            args[2] = &rowCount;
            args[3] = &classCount;
            args[4] = &deviceSeed;

            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            return true;
        }
    }
}
