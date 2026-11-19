using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class DirectGradientLeaseOwnershipTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public DirectGradientLeaseOwnershipTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void Committed_direct_gradient_keeps_its_lease_after_the_action_ends()
    {
        Skip.IfNot(_fixture.IsAvailable, "No native GPU backend.");
        var engine = _fixture.Engine!;
        using var operation = engine.EnterGpuOp();
        var backend = engine.GetBackend()!;
        var pool = engine.CreateScratchPool();
        pool.FirstLeasedAction = 0;
        var previousPool = DirectGpuTensorEngine.EnterScratchPool(pool);
        bool previousTracking = DifferentiableOps.TrackAdoptedAccumulators;
        using var parameter = new Tensor<float>(new[] { 4 });
        using var gradient = Tensor<float>.CreateGpuResident(new[] { 4 }, backend.DeviceType);
        engine.SuspendActivationEviction();
        try
        {
            DifferentiableOps.TrackAdoptedAccumulators = true;
            using (engine.EnterCompiledCapturePath())
            {
                Assert.True(engine.ResidentStepActive);
                engine.SetCurrentScratchAction(0);
                var original = engine.RentActionScratchOrAllocate(backend, 4, fullyWritten: true);
                Assert.True(pool.HasLeases);
                backend.Fill(original, 3f, 4);
                engine.BindResidentBuffer(gradient, original, backend);
                DifferentiableOps.CommitDirectGradWrite(new Dictionary<Tensor<float>, Tensor<float>>(), parameter, gradient);
                engine.EndScratchAction();
                engine.SetCurrentScratchAction(1);
                var next = engine.RentActionScratchOrAllocate(backend, 4, fullyWritten: true);
                Assert.NotEqual(original.Handle, next.Handle);
                backend.Fill(next, 9f, 4);
                Assert.Equal(Enumerable.Repeat(3f, 4), gradient.ToArray());
                engine.EndScratchAction();
            }
        }
        finally
        {
            DifferentiableOps.TrackAdoptedAccumulators = previousTracking;
            DirectGpuTensorEngine.RestoreScratchPool(previousPool);
            pool.Release();
            engine.ResumeActivationEviction();
        }
    }
}
