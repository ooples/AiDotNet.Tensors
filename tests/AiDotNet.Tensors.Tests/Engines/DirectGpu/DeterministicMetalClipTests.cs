#if NET7_0_OR_GREATER
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class DeterministicMetalClipTests
{
    [SkippableTheory]
    [MemberData(nameof(DeterministicDeviceClipTests.Cases), MemberType = typeof(DeterministicDeviceClipTests))]
    public void MetalClip_MatchesTheIntegerReferenceBitForBit(string name, float[][] host, float maxNorm)
    {
        Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.OSX), "Metal runs on macOS only.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        DeterministicWebGpuClipTests.CheckBackend(backend, name, host, maxNorm);
    }
}
#endif
