using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The module initializer no longer probes for a GPU when the assembly loads; it records that detection is owed
/// and <see cref="AiDotNetEngine.Current"/> runs it on its first read. A consumer that only calls the SIMD kernels
/// or the helpers therefore never pays for a probe, while anything that asks for an engine still gets the GPU
/// without calling <see cref="AiDotNetEngine.AutoDetectAndConfigureGpu()"/> itself. These pin the three ways the
/// owed detection ends: the first read runs it, and an engine the caller chose or a reset to CPU cancels it.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class DeferredGpuDetectionTests
{
    [Fact]
    public void The_first_read_of_current_runs_the_owed_detection()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.DeferGpuAutoDetection(verbose: false);
            Assert.True(AiDotNetEngine.IsGpuAutoDetectionDeferred);

            var engine = AiDotNetEngine.Current;

            Assert.NotNull(engine);
            Assert.False(AiDotNetEngine.IsGpuAutoDetectionDeferred);
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    [Fact]
    public void An_engine_the_caller_assigns_first_cancels_the_owed_detection()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.DeferGpuAutoDetection(verbose: false);
            var cpu = new CpuEngine();

            AiDotNetEngine.Current = cpu;

            Assert.False(AiDotNetEngine.IsGpuAutoDetectionDeferred);
            Assert.Same(cpu, AiDotNetEngine.Current);
            Assert.True(AiDotNetEngine.IsCallerAssigned);
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    [Fact]
    public void A_reset_to_cpu_cancels_the_owed_detection()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            var cpu = new CpuEngine();
            AiDotNetEngine.Current = cpu;
            AiDotNetEngine.DeferGpuAutoDetection(verbose: false);

            AiDotNetEngine.ResetToCpu();

            Assert.False(AiDotNetEngine.IsGpuAutoDetectionDeferred);
            Assert.Same(cpu, AiDotNetEngine.Current);
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    [Fact]
    public void An_explicit_detection_settles_what_was_owed()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.DeferGpuAutoDetection(verbose: false);

            AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false);

            Assert.False(AiDotNetEngine.IsGpuAutoDetectionDeferred);
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }
}
