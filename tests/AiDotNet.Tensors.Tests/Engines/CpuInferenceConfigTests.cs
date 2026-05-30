using System;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Verifies the public startup thread-pin knob (<see cref="CpuInferenceConfig"/>):
/// the default resolves to ProcessorCount/2, explicit values pass through, and the
/// call is safe/idempotent regardless of whether native BLAS is present.
/// Each test restores the count to ProcessorCount so it can't leave the shared
/// process pinned low for other (parallel) tests in the run.
/// </summary>
public class CpuInferenceConfigTests
{
    [Fact]
    public void PinBlasThreadsForLatency_DefaultsToHalfTheCores()
    {
        try
        {
            int expected = Math.Max(1, Environment.ProcessorCount / 2);
            Assert.Equal(expected, CpuInferenceConfig.PinBlasThreadsForLatency());
        }
        finally { CpuInferenceConfig.PinBlasThreadsForLatency(Environment.ProcessorCount); }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void PinBlasThreadsForLatency_ExplicitValuePassesThrough(int threads)
    {
        try
        {
            Assert.Equal(threads, CpuInferenceConfig.PinBlasThreadsForLatency(threads));
        }
        finally { CpuInferenceConfig.PinBlasThreadsForLatency(Environment.ProcessorCount); }
    }

    [Fact]
    public void PinBlasThreadsForLatency_IsIdempotentAndSafe()
    {
        try
        {
            // Re-applying the same count must not throw (idempotent — no pool
            // rebuild) and must be a safe no-op when native BLAS is absent.
            int a = CpuInferenceConfig.PinBlasThreadsForLatency(2);
            int b = CpuInferenceConfig.PinBlasThreadsForLatency(2);
            Assert.Equal(a, b);
        }
        finally { CpuInferenceConfig.PinBlasThreadsForLatency(Environment.ProcessorCount); }
    }
}
