using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class RectSliceScratchOwnershipTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public RectSliceScratchOwnershipTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void A_slice_uses_action_scratch_and_an_adopted_gradient_keeps_it_after_the_action()
    {
        Skip.IfNot(_fixture.IsAvailable, "Actual native GPU required.");
        var engine = _fixture.Engine!;
        using var operation = engine.EnterGpuOp();
        var backend = engine.GetBackend()!;
        var pool = engine.CreateScratchPool(); pool.FirstLeasedAction = 0;
        var previousPool = DirectGpuTensorEngine.EnterScratchPool(pool);
        bool previousTracking = DifferentiableOps.TrackAdoptedAccumulators;
        using var input = new Tensor<float>(Enumerable.Range(0, 12).Select(i => (float)i).ToArray(), new[] { 3, 4 });
        using var parameter = new Tensor<float>(new[] { 3, 2 });
        using var gradient = Tensor<float>.CreateGpuResident(new[] { 3, 2 }, backend.DeviceType);
        engine.SuspendActivationEviction();
        try
        {
            DifferentiableOps.TrackAdoptedAccumulators = true;
            using (engine.EnterCompiledCapturePath())
            {
                Assert.True(engine.ResidentStepActive);
                engine.SetCurrentScratchAction(0);
                var first = ((IEngine)engine).TensorSlice(input, new[] { 0, 1 }, new[] { 3, 2 });
                var original = first._gpuBuffer!;
                Assert.NotNull(original);
                Assert.True(pool.HasLeases);
                Assert.True(StepScratchPool.IsPooled(original));
                engine.BindResidentBuffer(gradient, original, backend);
                DifferentiableOps.CommitDirectGradWrite(new Dictionary<Tensor<float>, Tensor<float>>(), parameter, gradient);
                engine.EndScratchAction();
                engine.SetCurrentScratchAction(1);
                var next = ((IEngine)engine).TensorSlice(input, new[] { 0, 0 }, new[] { 3, 2 });
                Assert.NotEqual(original.Handle, next._gpuBuffer!.Handle);
                Assert.Equal(new[] { 1f, 2f, 5f, 6f, 9f, 10f }, gradient.ToArray());
                Assert.Equal(new[] { 0f, 1f, 4f, 5f, 8f, 9f }, next.ToArray());
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
