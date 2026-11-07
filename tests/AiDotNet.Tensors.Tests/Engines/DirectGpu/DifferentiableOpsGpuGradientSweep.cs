using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Tests.Engines.Autodiff;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Every differentiable op the CPU gradient sweep can drive, run on the GPU engine: its tape gradient must match the
/// CPU tape, and its compiled training gradient must match its own tape.
/// </summary>
/// <remarks>
/// A GPU override that runs its kernel without recording a tape edge gives every input a zero eager gradient; one
/// that runs its kernel eagerly under GraphMode is frozen into a compiled plan as a constant, so compiled training
/// gives its inputs a zero gradient (#1680). The CPU sweeps cannot see either: the GPU overrides replace the CPU code
/// they check.
/// </remarks>
[Collection("VulkanGlobalState")]
public class DifferentiableOpsGpuGradientSweep : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly ITestOutputHelper _out;

    public DifferentiableOpsGpuGradientSweep(DirectGpuTensorEngineTestFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _out = output;
    }

    [SkippableFact]
    public void EveryDifferentiableOp_GpuTapeMatchesCpu_AndGpuCompiledMatchesGpuTape()
    {
        Skip.If(!_fixture.IsAvailable || _fixture.Engine is null, "No direct GPU backend on this machine.");
        var result = FloatGradientSweep.Run(_fixture.Engine!, _out);
        result.Report(_out, "GPU");
        var all = result.AllFailures;
        Assert.True(all.Count == 0, $"{all.Count} GPU gradient mismatch(es):\n" + string.Join("\n", all.Take(80)));
    }
}
