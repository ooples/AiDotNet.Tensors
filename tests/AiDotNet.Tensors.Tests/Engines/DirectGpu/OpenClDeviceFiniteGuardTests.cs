using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>#1502: OpenCL's device-side finiteness guard (the checks are in <see cref="DeviceFiniteGuardTestBase"/>).</summary>
[Collection("DirectGpuSerial")]
public sealed class OpenClDeviceFiniteGuardTests : DeviceFiniteGuardTestBase
{
    protected override BackendLease Lease()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        var backend = new OpenClBackend(deviceIndex: 0);
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            Skip.If(true, "OpenCL backend did not initialize.");
        }
        return new BackendLease(backend, owned: true);
    }

    protected override void Finish(IDirectGpuBackend backend, IGpuBuffer accumulator)
        => ((OpenClBackend)backend).FinishSumOfSquaresDouble(accumulator);
}
