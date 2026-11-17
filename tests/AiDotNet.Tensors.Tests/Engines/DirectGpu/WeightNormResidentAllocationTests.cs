using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class WeightNormResidentAllocationTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public WeightNormResidentAllocationTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void Warm_native_forward_and_vjp_reuse_device_storage_without_weight_sized_host_scratch()
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend.");
        var gpu = _fixture.Engine ?? throw new InvalidOperationException();
        const int rows = 8, columns = 32768, length = rows * columns;
        var direction = new Tensor<float>(new[] { rows, columns });
        direction.AsWritableSpan().Fill(.125f);
        var gain = new Tensor<float>(new[] { rows }); gain.AsWritableSpan().Fill(.75f);
        var upstream = new Tensor<float>(new[] { rows, columns }); upstream.AsWritableSpan().Fill(.25f);
        var dv = new Tensor<float>(new[] { rows, columns });
        var dg = new Tensor<float>(new[] { rows });
        gpu.EnsureResidentInput(direction); gpu.EnsureResidentInput(gain); gpu.EnsureResidentInput(upstream);
        var pool = gpu.CreateScratchPool();
        var previousPool = DirectGpuTensorEngine.EnterScratchPool(pool);
        gpu.SuspendActivationEviction();
        try
        {
            using (gpu.EnterCompiledCapturePath())
            {
                gpu.SetCurrentScratchAction(0);
                using var warm = gpu.TensorWeightNorm(direction, gain);
                gpu.SetCurrentScratchAction(1);
                gpu.TensorWeightNormBackwardInto(upstream, direction, gain, dv, dg);
                var forwardBuffer = warm._gpuBuffer;
                var directionBuffer = dv._gpuBuffer; var gainBuffer = dg._gpuBuffer;
                int buffers = pool.BufferCount; long bytes = pool.BufferBytes;
                long start = GC.GetAllocatedBytesForCurrentThread();
                Tensor<float>? last = null;
                for (int i = 0; i < 4; i++)
                {
                    last?.Dispose();
                    gpu.SetCurrentScratchAction(0);
                    last = gpu.TensorWeightNorm(direction, gain);
                    gpu.SetCurrentScratchAction(1);
                    gpu.TensorWeightNormBackwardInto(upstream, direction, gain, dv, dg);
                    Assert.True(last.HasDeferredHostStorage);
                    Assert.Same(forwardBuffer, last._gpuBuffer);
                    Assert.Same(directionBuffer, dv._gpuBuffer); Assert.Same(gainBuffer, dg._gpuBuffer);
                    Assert.Equal(buffers, pool.BufferCount); Assert.Equal(bytes, pool.BufferBytes);
                }
                long allocated = GC.GetAllocatedBytesForCurrentThread() - start;
                Assert.True(allocated < length * sizeof(float),
                    $"Four warmed forward/VJP calls allocated {allocated:N0} managed bytes for {length:N0} weights.");
                double inv = 1 / Math.Sqrt(columns * .125 * .125 + 1e-24);
                Assert.InRange(Math.Abs(last!.ToArray()[0] - .125 * .75 * inv), 0, 1e-6);
                Assert.InRange(Math.Abs(dg.ToArray()[0] - columns * .25 * .125 * inv), 0, .001);
                Assert.InRange(Math.Abs(dv.ToArray()[0]), 0, 1e-6);
                last.Dispose();
            }
        }
        finally
        {
            DirectGpuTensorEngine.RestoreScratchPool(previousPool); pool.Release(); gpu.ResumeActivationEviction();
        }
    }

    [Fact]
    public void Warm_compiled_cpu_vjp_does_not_allocate_full_direction_contributions()
    {
        var engine = new CpuEngine();
        var previous = AiDotNetEngine.Current; AiDotNetEngine.Current = engine;
        try
        {
            const int rows = 8, columns = 32768, length = rows * columns;
            var direction = new Tensor<float>(new[] { rows, columns }); direction.AsWritableSpan().Fill(.125f);
            var gain = new Tensor<float>(new[] { rows }); gain.AsWritableSpan().Fill(.75f);
            ICompiledTrainingPlan<float> plan;
            using (var graph = GraphMode.EnableTraining(new[] { direction, gain }))
                plan = graph.CompileTraining(new[] { direction, gain }, engine.ReduceSum(engine.TensorWeightNorm(direction, gain)));
            using (plan)
            {
                plan.Step(); plan.Step();
                var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
                Assert.Contains("specialized:TensorWeightNorm", concrete.BackwardStepKinds);
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int step = 0; step < 4; step++) plan.Step();
                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                Assert.True(allocated < length * sizeof(float),
                    $"Four warmed compiled steps allocated {allocated:N0} managed bytes for {length:N0} direction elements.");
                Assert.InRange(Math.Abs(plan.Gradients[1].ToArray()[0] - Math.Sqrt(columns)), 0, .001);
                Assert.InRange(Math.Abs(plan.Gradients[0].ToArray()[0]), 0, 1e-6);
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
