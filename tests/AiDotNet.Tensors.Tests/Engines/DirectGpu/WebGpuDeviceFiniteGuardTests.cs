#if NET7_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>#1522: WebGPU's device-side finiteness guard (the checks are in <see cref="DeviceFiniteGuardTestBase"/>).</summary>
[Collection("DirectGpuSerial")]
public sealed class WebGpuDeviceFiniteGuardTests : DeviceFiniteGuardTestBase
{
    protected override BackendLease Lease() => new(DawnHostedWebGpu.BackendOrSkip(), owned: false);

    protected override void Finish(IDirectGpuBackend backend, IGpuBuffer accumulator)
        => ((WebGpuBackend)backend).FinishSumOfSquaresDouble(accumulator);
}
#endif