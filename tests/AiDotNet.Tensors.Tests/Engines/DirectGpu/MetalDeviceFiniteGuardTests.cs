using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>#1520: Metal's device-side finiteness guard (the checks are in <see cref="DeviceFiniteGuardTestBase"/>).
/// Runs on the macOS runner (its filter selects "Metal").</summary>
[Collection("DirectGpuSerial")]
public sealed class MetalDeviceFiniteGuardTests : DeviceFiniteGuardTestBase
{
    protected override BackendLease Lease()
    {
        Skip.IfNot(System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX),
            "Metal runs on macOS only.");
        var backend = new MetalBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            Skip.If(true, "Metal is not available.");
        }
        return new BackendLease(backend, owned: true);
    }

    protected override void Finish(IDirectGpuBackend backend, IGpuBuffer accumulator)
        => ((MetalBackend)backend).FinishSumOfSquaresDouble(accumulator);
}