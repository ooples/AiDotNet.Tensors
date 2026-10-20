using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>#1451: the resident program on Metal, every stage in one compute encoder and one command buffer. Runs on
/// the macOS job (metal.yml); skips elsewhere.</summary>
[Collection("DirectGpuSerial")]
public sealed class MetalResidentProgramTests : ResidentProgramTestBase<MetalBackend>
{
    private protected override MetalBackend RequireBackend()
    {
        Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.OSX), "Metal is only available on macOS.");
        var backend = new MetalBackend();
        if (!backend.IsAvailable || !backend.SupportsResidentProgram)
        {
            backend.Dispose();
            Skip.If(true, "Metal (with the resident program kernel) is not available.");
        }
        return backend;
    }

    private protected override void RunResident(MetalBackend backend, ResidentStage[] stages) => backend.RunResidentProgram(stages);

    private protected override void RunSeparately(MetalBackend backend, ResidentStage[] stages) => backend.RunStagesSeparately(stages);
}
