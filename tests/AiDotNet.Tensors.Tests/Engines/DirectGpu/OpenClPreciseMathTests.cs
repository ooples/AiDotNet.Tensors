using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Guards the opt-in precise-math toggle: when enabled, OpenCL programs must compile without the
/// aggressive fast-math flags (which drive the CPU/GPU divergences) and fall back to a single
/// IEEE-leaning FMA flag. Pure flag-resolution logic — no GPU required.
/// </summary>
public sealed class OpenClPreciseMathTests
{
    [Fact]
    public void DefaultMode_UsesAggressiveFastMathFlags()
    {
        bool original = OpenClBuildOptions.PreciseMath;
        try
        {
            OpenClBuildOptions.PreciseMath = false;
            Assert.Contains("-cl-fast-relaxed-math", OpenClBuildOptions.OptimizationFlags);
            Assert.Contains("-cl-unsafe-math-optimizations", OpenClBuildOptions.OptimizationFlags);
            Assert.Contains("-cl-finite-math-only", OpenClBuildOptions.OptimizationFlags);
        }
        finally
        {
            OpenClBuildOptions.PreciseMath = original;
        }
    }

    [Fact]
    public void PreciseMode_DropsEveryAggressiveFastMathFlag()
    {
        bool original = OpenClBuildOptions.PreciseMath;
        try
        {
            OpenClBuildOptions.PreciseMath = true;

            foreach (var flags in new[] { OpenClBuildOptions.OptimizationFlags, OpenClBuildOptions.SafeMathFlags })
            {
                Assert.DoesNotContain("-cl-fast-relaxed-math", flags);
                Assert.DoesNotContain("-cl-unsafe-math-optimizations", flags);
                Assert.DoesNotContain("-cl-finite-math-only", flags);
                Assert.Equal(OpenClBuildOptions.PreciseMathFlags, flags);
            }

            // The precise flag set keeps single-rounding FMA (matches the CPU FusedMultiplyAdd paths).
            Assert.Contains("-cl-mad-enable", OpenClBuildOptions.PreciseMathFlags);
        }
        finally
        {
            OpenClBuildOptions.PreciseMath = original;
        }
    }
}
