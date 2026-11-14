using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// CudaStepArena: a training thread's second step is recorded, packed by lifetime into one block, and later steps get
/// their allocations from planned offsets. Every step must stay exact against the CPU, a buffer that outlives its
/// recorded lifetime must never be aliased, and a step that differs from the recording must re-plan.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CudaStepArenaGpuTests
{
    private readonly ITestOutputHelper _output;
    public CudaStepArenaGpuTests(ITestOutputHelper output) => _output = output;

    private static Tensor<float> Values(int rows, int cols, int seed) =>
        new(Enumerable.Range(0, rows * cols).Select(i => (float)Math.Sin(i * 0.173 + seed)).ToArray(), new[] { rows, cols });

    // h = x·w; a = sigmoid(h); loss = sum(a ⊙ a). Returns dL/dw and a (kept only when retain is set).
    private static (float[] Grad, Tensor<float> A) Step(IEngine engine, Tensor<float> x, Tensor<float> w, bool retain)
    {
        using var tape = new GradientTape<float>();
        var a = engine.Sigmoid(engine.TensorMatMul(x, w));
        if (retain) tape.Retain(a);
        var loss = engine.ReduceSum(engine.TensorMultiply(a, a), null);
        var grad = tape.ComputeGradients(loss, new[] { w })[w].ToArray();
        return (grad, a);
    }

    private static float[] CpuGrad(Tensor<float> x, Tensor<float> w) => Step(new CpuEngine(), x, w, retain: false).Grad;

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[i])),
                $"{what}[{i}]: gpu {actual[i]} cpu {expected[i]}");
    }

    private static (DirectGpuTensorEngine Gpu, CudaBackend Cuda) Cuda()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        if (gpu is null || !gpu.IsGpuAvailable || gpu.GetBackend() is not CudaBackend cuda)
        {
            gpu?.Dispose();
            Skip.If(true, "CUDA backend did not resolve.");
            throw new InvalidOperationException("Unreachable after Skip.If.");
        }
        return (gpu, cuda);
    }

    [SkippableFact]
    public void ReplayedSteps_ServeFromThePlan_AndMatchTheCpu()
    {
        var (gpu, cuda) = Cuda();
        using (gpu)
        {
            var xh = Values(32, 64, 1);
            var wh = Values(64, 48, 2);
            var expected = CpuGrad(xh, wh);
            var x = gpu.UploadToGpu(xh, GpuTensorRole.General);
            var w = gpu.UploadToGpu(wh, GpuTensorRole.General);
            for (int step = 0; step < 6; step++)
                AssertClose(expected, Step(gpu, x, w, retain: false).Grad, $"step {step} dL/dw");

            var arena = cuda.StepArenaForTests;
            Assert.NotNull(arena);
            if (arena is null) return;
            _output.WriteLine($"hits {arena.SlotHits}, fallbacks {arena.PoolFallbacks}, replans {arena.Replans}, plan {arena.PlannedBytes} B");
            Assert.True(arena.SlotHits > 0, "no allocation was served from the plan");
            Assert.Equal(0, arena.Replans);
        }
    }

    [SkippableFact]
    public void ABufferThatOutlivesItsRecordedLifetime_IsNeverAliased()
    {
        var (gpu, cuda) = Cuda();
        using (gpu)
        {
            var xh = Values(32, 64, 3);
            var wh = Values(64, 48, 4);
            var expectedA = new CpuEngine().Sigmoid(new CpuEngine().TensorMatMul(xh, wh)).ToArray();
            var x = gpu.UploadToGpu(xh, GpuTensorRole.General);
            var w = gpu.UploadToGpu(wh, GpuTensorRole.General);
            for (int step = 0; step < 3; step++) Step(gpu, x, w, retain: false);   // warm-up, record, first replay

            // a dies inside the recorded step; here it is kept past its step while later steps replay the plan.
            // Later steps use other inputs: memory aliased to the kept buffer would be overwritten with other values.
            var (_, kept) = Step(gpu, x, w, retain: true);
            var otherH = Values(32, 64, 30);
            var expectedOther = CpuGrad(otherH, wh);
            var other = gpu.UploadToGpu(otherH, GpuTensorRole.General);
            for (int step = 0; step < 3; step++)
                AssertClose(expectedOther, Step(gpu, other, w, retain: false).Grad, $"later step {step} dL/dw");
            AssertClose(expectedA, kept.ToArray(), "kept a");

            var arena = cuda.StepArenaForTests;
            Assert.NotNull(arena);
            if (arena is not null)
                _output.WriteLine($"hits {arena.SlotHits}, fallbacks {arena.PoolFallbacks}, replans {arena.Replans}");
        }
    }

    [SkippableFact]
    public void AStepThatDiffersFromTheRecording_RePlans_AndStaysExact()
    {
        var (gpu, cuda) = Cuda();
        using (gpu)
        {
            var wh = Values(64, 48, 6);
            var w = gpu.UploadToGpu(wh, GpuTensorRole.General);
            var small = Values(32, 64, 5);
            var large = Values(40, 64, 7);
            var xs = gpu.UploadToGpu(small, GpuTensorRole.General);
            var xl = gpu.UploadToGpu(large, GpuTensorRole.General);
            var expectedSmall = CpuGrad(small, wh);
            var expectedLarge = CpuGrad(large, wh);

            for (int step = 0; step < 4; step++) AssertClose(expectedSmall, Step(gpu, xs, w, false).Grad, $"small {step}");
            var arena = cuda.StepArenaForTests;
            Assert.NotNull(arena);
            if (arena is null) return;
            long hitsBefore = arena.SlotHits;
            for (int step = 0; step < 4; step++) AssertClose(expectedLarge, Step(gpu, xl, w, false).Grad, $"large {step}");
            _output.WriteLine($"hits {hitsBefore} -> {arena.SlotHits}, fallbacks {arena.PoolFallbacks}, replans {arena.Replans}");
            Assert.True(arena.Replans >= 1, "a mismatched step did not re-plan");
            Assert.True(arena.SlotHits > hitsBefore, "the new shape was never served from its new plan");
        }
    }

    [SkippableFact]
    public void EngineDispose_FreesArenaBlock_EvenWhenSlotTensorOutlivesEngine()
    {
        // Keep one peer CUDA backend alive on the shared primary context so cuDevicePrimaryCtxRelease cannot mask a
        // leaked arena block, and verify that disposing a transient engine frees its step-arena block cleanly even if
        // a retained slot tensor is collected after engine.Dispose().
        var (peerGpu, peerCuda) = Cuda();
        using (peerGpu)
        {
            static void RunTransientEngineAndDropRetainedSlot()
            {
                var (gpu, cuda) = Cuda();
                Tensor<float>? outlived;
                using (gpu)
                {
                    var x = gpu.UploadToGpu(Values(32, 64, 11), GpuTensorRole.General);
                    var w = gpu.UploadToGpu(Values(64, 48, 12), GpuTensorRole.General);
                    for (int step = 0; step < 3; step++) Step(gpu, x, w, retain: false);
                    Assert.NotNull(cuda.StepArenaForTests);
                    (_, outlived) = Step(gpu, x, w, retain: true);
                }
                Assert.Null(cuda.StepArenaForTests);
                GC.KeepAlive(outlived);
            }

            RunTransientEngineAndDropRetainedSlot();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            using var probe = peerCuda.AllocateBuffer(16);
            Assert.Equal(16, probe.Size);
        }
    }

    [SkippableFact]
    public void Disabled_TheArenaIsNeverCreated()
    {
        var (gpu, cuda) = Cuda();
        bool was = CudaStepArena.Enabled;
        CudaStepArena.Enabled = false;
        try
        {
            using (gpu)
            {
                var x = gpu.UploadToGpu(Values(32, 64, 8), GpuTensorRole.General);
                var w = gpu.UploadToGpu(Values(64, 48, 9), GpuTensorRole.General);
                for (int step = 0; step < 4; step++) Step(gpu, x, w, false);
                Assert.Null(cuda.StepArenaForTests);
            }
        }
        finally { CudaStepArena.Enabled = was; }
    }
}
