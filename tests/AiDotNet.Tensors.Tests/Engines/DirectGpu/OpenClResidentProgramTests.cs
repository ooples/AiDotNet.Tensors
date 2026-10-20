using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>#1449: the resident program on OpenCL, every stage in one flush of the in-order queue.</summary>
[Collection("DirectGpuSerial")]
public sealed class OpenClResidentProgramTests : ResidentProgramTestBase<OpenClBackend>
{
    private protected override OpenClBackend RequireBackend()
    {
        var backend = new OpenClBackend();
        if (!backend.IsAvailable || !backend.SupportsResidentProgram)
        {
            backend.Dispose();
            Skip.If(true, "OpenCL (with the resident program kernel) is not available.");
        }
        return backend;
    }

    private protected override int RunResident(OpenClBackend backend, ResidentStage[] stages) => backend.RunResidentProgram(stages);

    private protected override void RunSeparately(OpenClBackend backend, ResidentStage[] stages) => backend.RunStagesSeparately(stages);
}
