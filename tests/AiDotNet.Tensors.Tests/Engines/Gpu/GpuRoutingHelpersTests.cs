using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #334: covers the Tensors-side capability surface that
/// <c>AiDotNet.NeuralNetworkBase.Train</c> calls when deciding whether to
/// dispatch a training step to the GPU. The actual <c>Train()</c> dispatch
/// fix ships in a companion AiDotNet PR; this suite validates the helpers
/// that companion PR depends on.
/// </summary>
public class GpuRoutingHelpersTests
{
    [Fact]
    public void IsGpuActive_NullEngine_ReturnsFalse()
    {
        Assert.False(GpuRoutingHelpers.IsGpuActive(null));
    }

    [Fact]
    public void IsGpuActive_CpuEngine_ReturnsFalse()
    {
        var engine = new CpuEngine();
        Assert.False(GpuRoutingHelpers.IsGpuActive(engine));
    }

    [Fact]
    public void TryGetAsyncBackend_NullEngine_ReturnsNull()
    {
        Assert.Null(GpuRoutingHelpers.TryGetAsyncBackend(null));
    }

    [Fact]
    public void TryGetAsyncBackend_CpuEngine_ReturnsNull()
    {
        var engine = new CpuEngine();
        Assert.Null(GpuRoutingHelpers.TryGetAsyncBackend(engine));
    }

    [Fact]
    public void TryExecuteOnGpu_CpuEngineActive_RunsFallback_NotGpuAction()
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            bool gpuRan = false;
            bool cpuRan = false;
            var actualReturn = GpuRoutingHelpers.TryExecuteOnGpu(
                gpuAction: _ => { gpuRan = true; },
                cpuFallback: () => { cpuRan = true; },
                options: null,
                out var reason);

            Assert.False(actualReturn, "TryExecuteOnGpu must return false on CPU engine.");
            Assert.False(gpuRan, "GPU action must NOT run on a CpuEngine.");
            Assert.True(cpuRan, "CPU fallback must run on a CpuEngine.");
            Assert.NotNull(reason);
            Assert.Contains("not a GPU engine", reason);
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    [Fact]
    public void TryExecuteOnGpu_NullGpuAction_Throws()
    {
        Assert.Throws<System.ArgumentNullException>(() =>
            GpuRoutingHelpers.TryExecuteOnGpu(
                gpuAction: null!,
                cpuFallback: () => { },
                options: null,
                out _));
    }

    [Fact]
    public void TryExecuteOnGpu_NullCpuFallback_Throws()
    {
        Assert.Throws<System.ArgumentNullException>(() =>
            GpuRoutingHelpers.TryExecuteOnGpu(
                gpuAction: _ => { },
                cpuFallback: null!,
                options: null,
                out _));
    }

    [Fact]
    public void TryExecuteOnGpu_ConvenienceOverload_DropsReason()
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            bool cpuRan = false;
            var ranOnGpu = GpuRoutingHelpers.TryExecuteOnGpu(
                _ => { },
                () => { cpuRan = true; });

            Assert.False(ranOnGpu);
            Assert.True(cpuRan);
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    [Fact]
    public void IsGpuActiveOnCurrentEngine_CpuEngineSet_ReturnsFalse()
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            Assert.False(GpuRoutingHelpers.IsGpuActiveOnCurrentEngine());
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }
}
