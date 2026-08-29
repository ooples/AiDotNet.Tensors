// Copyright (c) AiDotNet. All rights reserved.
// OpenCL launcher for categorical sampling. Mirrors OpenClBackend.Ann.cs's pattern: pull the
// compiled DirectOpenClKernel from _kernelCache and dispatch via kernel.Execute1D.
#if !NET462
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// Gives OpenCL a device route for <c>TensorCategoricalSample</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>ICategoricalSamplingBackend</c> was implemented by CUDA only and
    /// <c>ISeededGumbelSoftmaxBackend</c> likewise, with <c>IGpuBatchExecution</c> on Vulkan — so on
    /// OpenCL the engine had no device route at all and quietly ran the op on the CPU after the
    /// caller had selected the GPU engine. The residency probe measured it as 0 kernel launches
    /// against a required 1.
    /// </para>
    /// <para>
    /// Availability is decided by whether the program COMPILED, not by parsing device extension
    /// strings. The kernel needs fp64 for exact CPU parity, and asking the device whether it
    /// compiles is both simpler and more honest than predicting it — a device that advertises
    /// <c>cl_khr_fp64</c> but rejects the program would otherwise be claimed as supported and then
    /// throw at dispatch.
    /// </para>
    /// </remarks>
    public sealed partial class OpenClBackend : ICategoricalSamplingBackend
    {
        /// <summary>Set during initialization when the categorical program compiled and registered.</summary>
        private bool _categoricalKernelReady;

        /// <inheritdoc/>
        public bool CanCategoricalSample(int rows, int classes)
            => _categoricalKernelReady
               && _context is not null
               && rows > 0
               && classes > 0
               // One work-item per row; a row wider than this is still correct but the serial walk
               // over classes stops being the right shape for a GPU and the CPU reference is faster.
               && (long)rows * classes <= int.MaxValue;

        /// <inheritdoc/>
        public bool TryCategoricalSample(
            IGpuBuffer probabilities,
            IGpuBuffer oneHot,
            int rows,
            int classes,
            ulong seed)
        {
            if (!CanCategoricalSample(rows, classes)) return false;
            if (probabilities is not DirectOpenClGpuBuffer source) return false;
            if (oneHot is not DirectOpenClGpuBuffer destination) return false;
            if (!_kernelCache.TryGetValue("categorical_sample", out var kernel)) return false;

            kernel.SetArg(0, source.Buffer.Handle);
            kernel.SetArg(1, destination.Buffer.Handle);
            kernel.SetArg(2, rows);
            kernel.SetArg(3, classes);
            kernel.SetArg(4, seed);
            kernel.Execute1D(rows, Math.Min(256, rows));
            return true;
        }
    }
}
#endif
