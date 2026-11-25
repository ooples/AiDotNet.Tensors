using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A GPU RMSNorm met while a training forward is traced reads its input, which realizes the lazy chain under it on the
/// device (here a Swish and the multiply it feeds, as in a SwiGLU block). The Swish result is bound to a buffer the
/// resident op allocated and nothing owns. The plan's steps recompute it, so the trace-time value is
/// dead; it must be freed by the compile rather than left to the buffer's finalizer once
/// the step rebinds the tensor: a 16-layer LM kept 56 MiB of them beside its second training step, its peak.
/// </summary>
public class TraceTimeDeviceValueTests
{
    private static Tensor<float> Random(int[] shape, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    [SkippableFact]
    public void An_unowned_device_value_realized_while_tracing_is_freed_by_the_compile()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.IfNot(gpu is not null && gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "CUDA backend did not resolve.");
        using (gpu)
        {
            var prior = AiDotNetEngine.Current;
            AiDotNetEngine.Current = gpu;
            try
            {
                var x = Random(new[] { 256, 1024 }, 1);
                var y = Random(new[] { 256, 1024 }, 2);
                var gamma = new Tensor<float>(new[] { 1024 });
                for (int i = 0; i < gamma.Length; i++) gamma[i] = 1f;
                Tensor<float> swish;
                CudaBackend.CudaGpuBuffer? traced;
                bool tracedOwned;
                ICompiledTrainingPlan<float> plan;
                using (var scope = GraphMode.EnableTraining(new[] { x, y, gamma }))
                {
                    swish = gpu.Swish(x);
                    var n = gpu.RMSNorm(gpu.TensorMultiply(swish, y), gamma, 1e-6, out _);
                    traced = swish._gpuBuffer as CudaBackend.CudaGpuBuffer;
                    tracedOwned = swish._ownsGpuBuffer;
                    plan = scope.CompileTraining(new[] { x, y, gamma }, gpu.ReduceSum(gpu.TensorMultiply(n, n), null, false));
                }
                Skip.If(traced is null || tracedOwned, "The trace did not leave an unowned device value.");
                Assert.True(TestRequire.NotNull(traced, "traced").IsReleased || traced.RawDevicePointer == IntPtr.Zero,
                    "after the compile the trace-time Swish value is still allocated; only its finalizer would free it");
                using (plan as IDisposable)
                {
                    plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0);
                    float first = plan.Step().ToArray()[0];
                    for (int step = 2; step <= 6; step++)
                        Assert.Equal(first, plan.Step().ToArray()[0]);
                    Assert.True(!float.IsNaN(first) && !float.IsInfinity(first) && first > 0);
                    Assert.False(ReferenceEquals(swish._gpuBuffer, traced));
                    Assert.True(TestRequire.NotNull(traced, "traced").IsReleased || traced.RawDevicePointer == IntPtr.Zero,
                        "the trace-time Swish value is still allocated; only its finalizer would free it");
                }
            }
            finally { AiDotNetEngine.Current = prior; }
        }
    }
}
