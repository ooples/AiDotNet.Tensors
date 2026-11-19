using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CompiledPowScratchOwnershipTests
{
    [SkippableTheory]
    [InlineData(0f)]
    [InlineData(1f)]
    [InlineData(2f)]
    [InlineData(.5f)]
    public void Pow_uses_action_scratch_and_an_adopted_gradient_survives_the_next_action(float exponent)
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable, "Actual native GPU required.");
        using var operation = engine.EnterGpuOp();
        var backend = engine.GetBackend()!;
        var pool = engine.CreateScratchPool(); pool.FirstLeasedAction = 0;
        var previousPool = DirectGpuTensorEngine.EnterScratchPool(pool);
        bool previousTracking = DifferentiableOps.TrackAdoptedAccumulators;
        using var input = new Tensor<float>(new[] { 1f, 4f, 9f, 16f }, new[] { 4 });
        using var parameter = new Tensor<float>(new[] { 4 });
        using var gradient = Tensor<float>.CreateGpuResident(new[] { 4 }, backend.DeviceType);
        float[] expectedNext;
        using (var baseline = ((IEngine)engine).TensorPow(input, 3f)) expectedNext = baseline.ToArray();
        engine.SuspendActivationEviction();
        try
        {
            DifferentiableOps.TrackAdoptedAccumulators = true;
            using (engine.EnterCompiledCapturePath())
            {
                engine.SetCurrentScratchAction(0);
                var first = ((IEngine)engine).TensorPow(input, exponent);
                var original = first._gpuBuffer!;
                Assert.True(pool.HasLeases);
                Assert.True(StepScratchPool.IsPooled(original));
                engine.BindResidentBuffer(gradient, original, backend);
                DifferentiableOps.CommitDirectGradWrite(new Dictionary<Tensor<float>, Tensor<float>>(), parameter, gradient);
                engine.EndScratchAction();
                engine.SetCurrentScratchAction(1);
                var next = ((IEngine)engine).TensorPow(input, 3f);
                Assert.NotEqual(original.Handle, next._gpuBuffer!.Handle);
                var actual = gradient.ToArray();
                var expected = new[] { 1f, 4f, 9f, 16f }.Select(x => MathF.Pow(x, exponent)).ToArray();
                for (int i = 0; i < actual.Length; i++)
                    Assert.True(Math.Abs(actual[i] - expected[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])));
                Assert.Equal(expectedNext, next.ToArray());
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
