using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// <see cref="AiDotNetEngine.IsCallerAssigned"/> tells a facade whose default auto-detects hardware that the caller
/// already chose an engine. Without it AiDotNet's AiModelBuilder default path reinstalled the GPU engine over a
/// caller's <c>AiDotNetEngine.Current = new CpuEngine()</c>, and a "CPU" training run held the whole GPU.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class CallerAssignedEngineTests
{
    [Fact]
    public void An_engine_assigned_through_the_setter_is_caller_assigned()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            Assert.True(AiDotNetEngine.IsCallerAssigned);
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    [Fact]
    public void A_reset_hands_the_choice_back_to_the_library_even_when_it_keeps_the_cpu_engine()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            var cpu = new CpuEngine();
            AiDotNetEngine.Current = cpu;
            AiDotNetEngine.ResetToCpu();
            Assert.Same(cpu, AiDotNetEngine.Current);
            Assert.False(AiDotNetEngine.IsCallerAssigned);
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }
}
