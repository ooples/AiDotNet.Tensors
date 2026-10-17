using System.Diagnostics;
using System.IO;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Where the CUDA driver reports a device and NVRTC loads, the backend must come up. Its constructor catches every
/// initialization failure and reports the backend unavailable, and the CUDA tests skip on an unavailable backend, so a
/// kernel module that no longer compiles turned every CUDA test into a skip and every caller into the OpenCL fallback:
/// NVRTC 13.1 rejected the conv-engine mainloops' generic lambdas as host functions (GTX 1660 Ti), and nothing failed.
/// </summary>
public class CudaBackendInitializationTests
{
    [SkippableFact]
    public void Backend_IsAvailable_WhereTheDriverHasADeviceAndNvrtcLoads()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "no CUDA driver or device");
        Skip.IfNot(NvrtcNativeBindings.IsAvailable, "NVRTC is not installed");

        // The constructor writes the reason it gave up to Trace; keep it for the failure message.
        var log = new StringWriter();
        var listener = new TextWriterTraceListener(log);
        Trace.Listeners.Add(listener);
        try
        {
            using var backend = new CudaBackend();
            listener.Flush();
            Assert.True(backend.IsAvailable, "CudaBackend failed to initialize on a machine with a CUDA device: " + log);
        }
        finally
        {
            Trace.Listeners.Remove(listener);
            listener.Dispose();
        }
    }
}
