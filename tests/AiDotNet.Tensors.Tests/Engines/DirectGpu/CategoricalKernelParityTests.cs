// Copyright (c) AiDotNet. All rights reserved.
#if !NETFRAMEWORK
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu
{
    /// <summary>
    /// Every backend's categorical kernel must draw its uniform from the SAME hash as the managed
    /// sampler, because the parity tests compare one-hot outputs exactly.
    /// </summary>
    /// <remarks>
    /// <para>
    /// WHY A SOURCE TEST AND NOT AN EXECUTION TEST. Five backends ship this kernel and no single
    /// machine can run more than one or two of them — this host has an OpenCL device, no Metal, no
    /// WebGPU adapter and no usable Vulkan compute path, and CI runs PoCL. So the kernel that
    /// actually executes anywhere is a minority of the kernels that ship, and a constant that drifts
    /// in one of the others is invisible until somebody runs it on that vendor's hardware and gets a
    /// different category for the same seed.
    /// </para>
    /// <para>
    /// These assertions are therefore about the CONTRACT rather than the output: every kernel keys
    /// its draw on the row index with the same PCG constants and the same 24-bit mantissa scaling,
    /// and every kernel accumulates in something wider than float. They cannot prove a kernel
    /// computes the right answer, but they do catch the single most likely way these five drift
    /// apart, which is somebody editing one and not the rest.
    /// </para>
    /// </remarks>
    public class CategoricalKernelParityTests
    {
        /// <summary>The StatelessRandom PCG constants, as used by the managed Uniform01(seed, index).</summary>
        private static readonly string[] RequiredConstants =
        {
            "747796405",   // multiplier
            "2891336453",  // increment
            "277803737",   // output mix
            "16777216",    // 2^24, the mantissa scale that turns the draw into [0,1)
        };

        public static TheoryData<string, string> KernelSources()
        {
            var data = new TheoryData<string, string>();
            data.Add("OpenCL", CategoricalKernels.GetSource());
            data.Add("HIP", HipCategoricalKernels.GetSource());
            data.Add("Metal", MetalCategoricalKernels.Source);
            data.Add("Vulkan", VulkanGlslKernels.CategoricalSampleGlsl);
#if NET8_0_OR_GREATER
            data.Add("WebGpu", AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuCategoricalKernels.CategoricalSample);
#endif
            return data;
        }

        [Theory]
        [MemberData(nameof(KernelSources))]
        public void EveryBackendUsesTheSameStatelessRandomConstants(string backend, string source)
        {
            Assert.False(string.IsNullOrWhiteSpace(source), $"{backend}: kernel source is empty.");

            foreach (var constant in RequiredConstants)
            {
                Assert.True(
                    source.Contains(constant, StringComparison.Ordinal),
                    $"{backend}'s categorical kernel is missing the StatelessRandom constant {constant}. "
                        + "All backends must draw from the identical PCG hash keyed on (seed, row), or the "
                        + "same seed selects a different category on different hardware.");
            }
        }

        [Theory]
        [MemberData(nameof(KernelSources))]
        public void EveryBackendSelectsTheLastCategoryWhenDriftOvershootsTheFinalBoundary(string backend, string source)
        {
            // The managed reference defaults `selected` to classes - 1 so floating-point drift past
            // the final cumulative boundary cannot leave the row without a selection. A kernel that
            // defaults to 0 instead would return a valid-looking one-hot for the WRONG category, and
            // only on the rows where drift happens.
            Assert.True(
                source.Contains("classes - 1", StringComparison.Ordinal)
                    || source.Contains("classes - 1u", StringComparison.Ordinal)
                    || source.Contains("cs_params.classes - 1u", StringComparison.Ordinal),
                $"{backend}'s categorical kernel does not default its selection to the last category.");
        }

        /// <summary>
        /// Float accumulation is the one thing that reliably breaks exact parity, so no backend may
        /// simply add floats in a loop.
        /// </summary>
        /// <remarks>
        /// The kernels split into two groups for a reason that is a language limit, not a choice:
        /// OpenCL, HIP and Vulkan have a 64-bit float type and accumulate in it, matching the CPU
        /// reference directly. MSL and WGSL have NO 64-bit float at all, so those two carry the sum
        /// in a compensated two-float expansion instead. Both are acceptable; plain float is not.
        /// </remarks>
        [Theory]
        [MemberData(nameof(KernelSources))]
        public void NoBackendAccumulatesInPlainFloat(string backend, string source)
        {
            bool hasNativeDouble =
                source.Contains("double ", StringComparison.Ordinal)
                || source.Contains("double(", StringComparison.Ordinal);

            bool hasCompensatedFallback =
                source.Contains("two_sum", StringComparison.Ordinal)
                && source.Contains("fma(", StringComparison.Ordinal);

            Assert.True(
                hasNativeDouble || hasCompensatedFallback,
                $"{backend}'s categorical kernel neither accumulates in double nor uses compensated "
                    + "two-float arithmetic. Summing probabilities in plain float lets the running total "
                    + "drift, and a target near a bucket edge then selects the neighbouring category — "
                    + "which still looks like a valid one-hot and so passes every smoke test.");
        }

        [Fact]
        public void TheTwoLanguagesWithoutA64BitFloatAreTheOnesUsingCompensatedArithmetic()
        {
            // Pins the reason for the split. If a future edit gives Metal or WebGpu real doubles, or
            // drops compensation from one of them, this is the test that says so.
            var compensated = new List<string>();
            foreach (var row in KernelSources())
            {
                var backend = (string)row[0];
                var source = (string)row[1];
                if (source.Contains("two_sum", StringComparison.Ordinal)) compensated.Add(backend);
            }

            Assert.Contains("Metal", compensated);
#if NET8_0_OR_GREATER
            Assert.Contains("WebGpu", compensated);
            Assert.Equal(2, compensated.Count);
#else
            Assert.Single(compensated);
#endif
        }
    }
}
#endif
