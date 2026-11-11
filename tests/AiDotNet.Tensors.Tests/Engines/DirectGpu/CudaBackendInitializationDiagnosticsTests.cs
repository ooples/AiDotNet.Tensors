using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A host with an NVIDIA driver but no NVRTC used to get an unavailable <see cref="CudaBackend"/> whose
/// <see cref="CudaBackend.InitializationError"/> was null, so callers could only report "no error was recorded".
/// </summary>
[Collection("DirectGpuSerial")]
public class CudaBackendInitializationDiagnosticsTests
{
    [SkippableFact]
    public void DriverWithoutNvrtc_ReportsWhyTheBackendIsUnavailable()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "Requires the CUDA driver.");
        Skip.If(NvrtcNativeBindings.IsAvailable, "NVRTC is present, so the backend does not take this path.");

        using var backend = new CudaBackend();

        Assert.False(backend.IsAvailable);
        Assert.NotNull(backend.InitializationError);
        Assert.Contains("NVRTC", backend.InitializationError);
    }
}
