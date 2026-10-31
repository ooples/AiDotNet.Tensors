using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
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

    /// <summary>
    /// A GPU engine the caller assigned is the caller's: a reset demotes it but must not dispose it, or work the caller
    /// still runs on it reads freed native handles (a cudnnSetStream access violation on the CUDA lane, while another
    /// test class reset the engine its fixture had made current).
    /// </summary>
    [SkippableFact]
    public void A_reset_does_not_dispose_a_gpu_engine_the_caller_assigned()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu;
            AiDotNetEngine.ResetToCpu();
            Assert.NotSame(gpu, AiDotNetEngine.Current);

            var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
            Assert.Equal(new float[] { 2, 4, 6, 8 }, gpu.TensorAdd(a, a).ToArray());
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
