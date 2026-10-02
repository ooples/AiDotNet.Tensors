using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// <see cref="AiDotNetEngine.ResetToCpu"/> must not replace a CPU engine that is already current: the engine is
/// process-wide and read on every operation, so a replacement swaps it under work running on other threads.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class ResetToCpuKeepsCpuEngineTests
{
    [Fact]
    public void A_reset_keeps_the_current_cpu_engine()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            var cpu = new CpuEngine();
            AiDotNetEngine.Current = cpu;
            AiDotNetEngine.ResetToCpu();
            Assert.Same(cpu, AiDotNetEngine.Current);
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    [Fact]
    public void A_reset_replaces_an_engine_that_is_not_a_plain_cpu_engine()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            var derived = new DerivedCpuEngine();
            AiDotNetEngine.Current = derived;
            AiDotNetEngine.ResetToCpu();
            Assert.NotSame(derived, AiDotNetEngine.Current);
            Assert.Equal(typeof(CpuEngine), AiDotNetEngine.Current.GetType());
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    [Fact]
    public void Concurrent_resets_install_one_cpu_engine()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new DerivedCpuEngine();
            var seen = new IEngine[16];
            Parallel.For(0, seen.Length, i =>
            {
                AiDotNetEngine.ResetToCpu();
                seen[i] = AiDotNetEngine.Current;
            });
            foreach (var engine in seen) Assert.Same(seen[0], engine);
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    private sealed class DerivedCpuEngine : CpuEngine
    {
    }
}
