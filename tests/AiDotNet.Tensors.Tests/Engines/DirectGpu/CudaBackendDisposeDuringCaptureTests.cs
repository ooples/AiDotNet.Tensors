using System;
using System.Reflection;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// cublasDestroy and cudnnDestroy synchronize the whole device, and every CudaBackend shares the device's primary
/// context. A backend disposed while another thread captured a graph on that context faulted inside the driver
/// (CUDA-lane crash dump: CudaBackend.Dispose -> cublasDestroy -> cuCtxSynchronize, access violation, while an
/// autotuner captured on another backend). Dispose now holds the context's capture gate around those destroys.
/// </summary>
[Collection("DirectGpuSerial")]
public class CudaBackendDisposeDuringCaptureTests
{
    [SkippableFact]
    public void DisposingABackendWithCublasHandles_WhileAnotherCaptures_IsSafe()
    {
        using var capturer = new CudaBackend();
        Skip.IfNot(capturer.IsAvailable, "CUDA is not available.");
        const int n = 1 << 16;
        using var x = capturer.AllocateBuffer(new float[n]);
        using var y = capturer.AllocateBuffer(n);
        int captures = 0, failedCaptures = 0;
        Exception? captureError = null;
        using var stop = new ManualResetEventSlim();
        var captureThread = new Thread(() =>
        {
            try
            {
                while (!stop.IsSet)
                {
                    IntPtr graph = capturer.CaptureGraph(() =>
                    {
                        for (int i = 0; i < 200; i++) capturer.Relu(x, y, n);
                    });
                    if (graph == IntPtr.Zero) { failedCaptures++; continue; }
                    captures++;
                    capturer.DestroyCapturedGraph(graph);
                }
            }
            catch (Exception ex) { captureError = ex; }
        }) { IsBackground = true };
        captureThread.Start();
        try
        {
            // This thread's cuBLAS handle for the backend (GEMM dispatch may route around cuBLAS, so ask for it).
            var cublasHandle = typeof(CudaBackend).GetProperty("_cublasHandle", BindingFlags.Instance | BindingFlags.NonPublic)!;
            for (int i = 0; i < 60; i++)
            {
                var backend = new CudaBackend();
                Assert.NotEqual(IntPtr.Zero, (IntPtr)cublasHandle.GetValue(backend)!);
                backend.Dispose();
            }
        }
        finally
        {
            stop.Set();
            captureThread.Join();
        }
        Assert.Null(captureError);
        Assert.True(captures > 0, "the capture thread never completed a capture");
        Assert.Equal(0, failedCaptures);
    }
}
