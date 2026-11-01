using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="AiDotNetEngine.ResetToCpu"/> with a GPU engine current. In this namespace so the CUDA lane runs it.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class ResetToCpuGpuEngineTests
{
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
            // Vulkan's engine borrows its backend, so TensorAdd below can still succeed on a disposed engine.
            Assert.False(gpu.IsDisposed);

            var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
            Assert.Equal(new float[] { 2, 4, 6, 8 }, gpu.TensorAdd(a, a).ToArray());
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    /// <summary>
    /// The engine auto-detection adopted is parked by a reset, not disposed (an op on another thread may still hold
    /// it), and the next auto-detection reinstalls that same engine instead of building a second CUDA context.
    /// </summary>
    [SkippableFact]
    public void A_reset_parks_the_adopted_engine_and_the_next_adoption_reinstalls_it()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.ResetToCpu();
            Skip.IfNot(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false), "No GPU was adopted");
            var adopted = Assert.IsType<DirectGpuTensorEngine>(AiDotNetEngine.Current);

            AiDotNetEngine.ResetToCpu();
            Assert.IsType<CpuEngine>(AiDotNetEngine.Current);
            Assert.False(adopted.IsDisposed);

            Assert.True(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false));
            Assert.Same(adopted, AiDotNetEngine.Current);
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }
}
