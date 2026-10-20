#if NET7_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1452: the resident program on WebGPU, every stage in one compute pass and one submission, run on Dawn through
/// <see cref="DawnHostedWebGpu"/> (skips unless AIDOTNET_WEBGPU_NODE_DIR is set up). Here also: an output that is one
/// of the stage's inputs, or an operand repeated, is refused.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class WebGpuResidentProgramTests : ResidentProgramTestBase<WebGpuBackend>
{
    private protected override WebGpuBackend RequireBackend()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        Skip.IfNot(backend.SupportsResidentProgram, "The WebGPU resident program shader did not build: " + backend.ResidentProgramError);
        return backend;
    }

    private protected override int RunResident(WebGpuBackend backend, ResidentStage[] stages) => backend.RunResidentProgram(stages);

    private protected override void RunSeparately(WebGpuBackend backend, ResidentStage[] stages) => backend.RunStagesSeparately(stages);

    // The Dawn-hosted backend is a process-wide singleton.
    private protected override void ReleaseBackend(WebGpuBackend backend) { }

    [SkippableFact]
    public void RefusesAnOutputThatIsAnInput_AndARepeatedOperand()
    {
        var backend = RequireBackend();
        using var chain = new Chain(backend, 4, 32, 5);
        // A chain that is not row-local (the shapes differ) binds each stage's operands separately, so these are
        // refused: a writable buffer may not be bound twice, and a buffer bound twice is snapshotted when its bind
        // group is created, before the chain's earlier stage has written it.
        var head = new ResidentStage(ResidentOp.Add, 1, 32, chain.DScale, chain.DBias, null, chain.DHidden);
        var inPlace = new[] { head, new ResidentStage(ResidentOp.Add, 4, 32, chain.DX, chain.DResidual, null, chain.DX) };
        Assert.Throws<ArgumentException>(() => backend.RunResidentProgram(inPlace));
        var repeated = new[] { head, new ResidentStage(ResidentOp.Add, 4, 32, chain.DHidden, chain.DHidden, null, chain.DOut) };
        Assert.Throws<ArgumentException>(() => backend.RunResidentProgram(repeated));

        // The row program binds each distinct buffer once, so the same stages alone run: in place, and repeated.
        Assert.Equal(1, backend.RunResidentProgram(new[] { new ResidentStage(ResidentOp.Add, 4, 32, chain.DX, chain.DResidual, null, chain.DX) }));
        Assert.Equal(1, backend.RunResidentProgram(new[] { new ResidentStage(ResidentOp.Add, 4, 32, chain.DX, chain.DX, null, chain.DOut) }));
        var x = backend.DownloadBuffer(chain.DX);
        var doubled = backend.DownloadBuffer(chain.DOut);
        for (int i = 0; i < 4 * 32; i++)
        {
            Assert.Equal(chain.X[i] + chain.Residual[i], x[i]);
            Assert.Equal(x[i] + x[i], doubled[i]);
        }
    }
}
#endif
