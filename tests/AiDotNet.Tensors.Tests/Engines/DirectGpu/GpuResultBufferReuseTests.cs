using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1302: an eager GPU result owns its device buffer, so disposing the result hands the buffer back to the backend's
/// pool and the next result of the same size reuses it. Before, the result never claimed ownership: Dispose() left the
/// buffer to the finalizer, which a few-hundred-byte object holding a 64 MB device buffer gives the GC no reason to run,
/// so every call allocated afresh (TensorExp over 16M floats: 3.3 ms per call on HIP against 0.79 ms kernel time).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuResultBufferReuseTests
{
    [SkippableFact]
    public void DisposingAnEagerResult_ReturnsItsDeviceBufferForTheNextResult()
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable, "No GPU backend on this host.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            const int n = 1 << 16;
            var x = new Tensor<float>(new float[n], new[] { n }).Gpu();

            var first = engine.TensorExp(x);
            var firstBuffer = first._gpuBuffer;
            Assert.NotNull(firstBuffer);
            Assert.True(first._ownsGpuBuffer, "an eager GPU result must own its device buffer");
            IntPtr handle = firstBuffer!.Handle;
            Assert.NotEqual(IntPtr.Zero, handle);
            first.Dispose();
            // A backend that submits without waiting (Vulkan) parks the buffer until the work that used it has run;
            // once the device is idle it is in the pool on every backend.
            engine.SynchronizeStream();

            var second = engine.TensorExp(x);
            Assert.Equal(handle, second._gpuBuffer?.Handle ?? IntPtr.Zero);

            // The result still computes: exp(0) = 1 everywhere.
            var values = second.ToArray();
            Assert.All(values, v => Assert.Equal(1f, v));
            second.Dispose();
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}