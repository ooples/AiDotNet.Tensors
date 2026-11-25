using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Between a captured plan's steps, allocations are served from the plan's idle step memory (as PyTorch's caching
/// allocator serves validation from what a training step freed) instead of growing the device footprint. The next
/// replay overwrites that memory, so whatever still uses it then must have been moved, and nothing a graph bakes in
/// may be lent memory.
/// </summary>
[Collection("DirectGpuSerial")]
public class IdleStepMemoryLoanTests
{
    private const int N = 1 << 20;   // 4 MB

    private static CudaBackend OpenBackendOrSkip()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        return new CudaBackend();
    }

    private static bool Inside(IGpuBuffer region, IGpuBuffer buffer)
    {
        long start = region.Handle.ToInt64(), at = buffer.Handle.ToInt64();
        return at >= start && at < start + region.SizeInBytes;
    }

    // A captured "step": scratch = acc + 1 (written before it is read), acc = scratch + 1. Each replay adds 2 to acc.
    private static IntPtr CaptureStep(CudaBackend backend, IGpuBuffer scratch, IGpuBuffer acc, IGpuBuffer ones)
        => backend.CaptureGraph(() =>
        {
            backend.Add(acc, ones, scratch, N);
            backend.Add(scratch, ones, acc, N);
        });

    // A pooled buffer of the requested size is memory the process already holds: it serves an allocation between
    // steps before idle step memory does. Lent memory was served first, so every allocation between two training
    // steps (the next batch's uploads, the loss readback) became a loan that the next step's reclaim had to move
    // (fresh memory and a device copy) before its graph launch. A size the pool cannot serve still borrows.
    [SkippableFact]
    public void AllocationsBetweenSteps_TakeAPooledBufferBeforeIdleMemory_AndBorrowOnlyWhatThePoolCannotServe()
    {
        using var backend = OpenBackendOrSkip();
        IIdleDeviceMemoryLender lender = backend;
        using var scratch = backend.AllocateBuffer(N);
        // Two buffers of N / 2 floats back in the pool.
        var first = backend.AllocateBuffer(N / 2);
        var second = backend.AllocateBuffer(N / 2);
        first.Dispose();
        second.Dispose();
        backend.Synchronize();
        var owner = new object();
        lender.LendIdle(owner, new[] { scratch });
        try
        {
            var loansBefore = backend.IdleLoanCounters.Loans;
            using var pooled = backend.AllocateBuffer(N / 2);
            using var pooledUninitialized = backend.AllocateBufferUninitialized(N / 2);
            Assert.False(Inside(scratch, pooled), "an allocation the pool could serve borrowed the idle memory");
            Assert.False(Inside(scratch, pooledUninitialized), "an uninitialized allocation the pool could serve borrowed the idle memory");
            Assert.Equal(loansBefore, backend.IdleLoanCounters.Loans);
            using var borrowed = backend.AllocateBuffer(N / 4);
            Assert.True(Inside(scratch, borrowed), "an allocation the pool could not serve did not borrow the idle memory");
            Assert.Equal(loansBefore + 1, backend.IdleLoanCounters.Loans);
        }
        finally { lender.ReclaimIdle(owner); }
    }

    [SkippableFact]
    public void AllocationsBetweenSteps_BorrowTheIdleMemory_AndOneStillAliveAtReclaimIsMovedIntact()
    {
        using var backend = OpenBackendOrSkip();
        IIdleDeviceMemoryLender lender = backend;
        using var acc = backend.AllocateBuffer(new float[N]);
        using var ones = backend.AllocateBuffer(Enumerable.Repeat(1f, N).ToArray());
        using var scratch = backend.AllocateBuffer(N);
        var owner = new object();
        IntPtr graph = CaptureStep(backend, scratch, acc, ones);
        Assert.NotEqual(IntPtr.Zero, graph);
        try
        {
            backend.LaunchCapturedGraph(graph);
            lender.LendIdle(owner, new[] { scratch });

            long allocatedBefore = GpuMemoryStats.Stats()["allocated_bytes.all.current"];
            var kept = backend.AllocateBuffer(N / 2);
            var dropped = backend.AllocateBuffer(N / 4);
            Assert.True(Inside(scratch, kept) && Inside(scratch, dropped), "the allocations did not borrow the idle memory");
            // Borrowed memory is the plan's, already counted: the account does not grow.
            Assert.Equal(allocatedBefore, GpuMemoryStats.Stats()["allocated_bytes.all.current"]);
            // Zeroed like a fresh allocation, though the replay left 1s in it.
            Assert.All(backend.DownloadBuffer(kept), v => Assert.Equal(0f, v));
            backend.Add(ones, ones, kept, N / 2);
            dropped.Dispose();

            // The next step reclaims: the live loan moves out with its contents, then the replay overwrites the memory.
            IntPtr lentAddress = kept.Handle;
            lender.ReclaimIdle(owner);
            Assert.NotEqual(lentAddress, kept.Handle);
            Assert.False(Inside(scratch, kept), "the loan still addresses the reclaimed memory");
            backend.LaunchCapturedGraph(graph);
            Assert.All(backend.DownloadBuffer(kept), v => Assert.Equal(2f, v));
            // The step computed exactly what it would have without the loans.
            Assert.Equal(4f, backend.DownloadBuffer(acc)[N - 1]);
            Assert.Equal(1, backend.IdleLoanCounters.Moves);

            // Reclaimed: nothing borrows it any more.
            using var after = backend.AllocateBuffer(N / 2);
            Assert.False(Inside(scratch, after));
            kept.Dispose();
        }
        finally
        {
            lender.ReclaimIdle(owner);
            backend.DestroyCapturedGraph(graph);
        }
    }

    [SkippableFact]
    public void ACaptureNeverBakesInLentMemory()
    {
        using var backend = OpenBackendOrSkip();
        IIdleDeviceMemoryLender lender = backend;
        using var acc = backend.AllocateBuffer(new float[N]);
        using var ones = backend.AllocateBuffer(Enumerable.Repeat(1f, N).ToArray());
        using var scratch = backend.AllocateBuffer(N);
        var owner = new object();
        IntPtr step = CaptureStep(backend, scratch, acc, ones);
        IntPtr other = IntPtr.Zero;
        try
        {
            backend.LaunchCapturedGraph(step);
            lender.LendIdle(owner, new[] { scratch });
            var borrowed = backend.AllocateBuffer(N);
            Assert.True(Inside(scratch, borrowed));
            backend.Add(ones, ones, borrowed, N);   // 2

            // A second graph reads the borrowed buffer and allocates: the loan moves before the capture begins, and the
            // capture's own allocation is not lent memory.
            using var sum = backend.AllocateBuffer(new float[N]);
            IGpuBuffer? inside = null;
            other = backend.CaptureGraph(() =>
            {
                inside = backend.AllocateBuffer(N);
                backend.Add(borrowed, ones, inside, N);   // 3
                backend.Add(sum, inside, sum, N);
            });
            Assert.NotEqual(IntPtr.Zero, other);
            Assert.False(Inside(scratch, borrowed), "a capture read a buffer that still borrowed lent memory");
            Assert.False(Inside(scratch, TestRequire.NotNull(inside, "inside")), "a capture allocated lent memory");

            // The lender's step overwrites its memory; the other graph still reads 2 + 1.
            lender.ReclaimIdle(owner);
            backend.LaunchCapturedGraph(step);
            backend.LaunchCapturedGraph(other);
            Assert.Equal(3f, backend.DownloadBuffer(sum)[N - 1]);
            Assert.Equal(4f, backend.DownloadBuffer(acc)[0]);
            borrowed.Dispose();
        }
        finally
        {
            lender.ReclaimIdle(owner);
            if (other != IntPtr.Zero) backend.DestroyCapturedGraph(other);
            backend.DestroyCapturedGraph(step);
        }
    }

    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private sealed class Run
    {
        public float[] Parameters = Array.Empty<float>();
        public List<float> Losses = new();
        public List<float[]> Validation = new();
        public long Replays;
        public long LentBytes;
        public long Loans;
    }

    // An MLP trained by a captured plan; with validate set, an eager forward of a second batch runs between steps (as
    // validation does) and one of its results is kept across the next step, as a caller holding a metric tensor would.
    private static Run Train(DirectGpuTensorEngine engine, int steps, bool validate)
    {
        var x = Filled(new[] { 256, 256 }, 1, 1f);
        var held = Filled(new[] { 256, 256 }, 2, 1f);
        var w1 = Filled(new[] { 256, 512 }, 3, 0.05f);
        var w2 = Filled(new[] { 512, 256 }, 4, 0.05f);
        var parameters = new[] { w1, w2 };
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = engine.ReLU(engine.TensorMatMul(x, w1));
            var y = engine.TensorMatMul(h, w2);
            engine.ReduceSum(engine.TensorMultiply(y, y), null);
            plan = scope.CompileTraining(parameters);
        }
        var run = new Run();
        var backend = (CudaBackend)TestRequire.NotNull(engine.GetBackend(), "the engine's backend");
        long replaysBefore = CompiledTrainingPlan<float>.GraphReplayCount;
        long loansBefore = backend.IdleLoanCounters.Loans;
        Tensor<float>? kept = null;
        float[]? keptValues = null;
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-4f);
            for (int s = 0; s < steps; s++)
            {
                run.Losses.Add(plan.Step().ToArray()[0]);
                if (kept is not null)
                {
                    // Kept across a step: its memory was moved before the replay overwrote what it borrowed.
                    Assert.Equal(TestRequire.NotNull(keptValues, "keptValues"), kept.ToArray());
                    kept = null;
                }
                run.LentBytes = Math.Max(run.LentBytes, ((CompiledTrainingPlan<float>)plan).IdleLentBytesLastStep);
                if (!validate || s < 3) continue;
                var vh = engine.ReLU(engine.TensorMatMul(held, w1));
                var vy = engine.TensorMatMul(vh, w2);
                run.Validation.Add(engine.ReduceSum(engine.TensorMultiply(vy, vy), null).ToArray());
                kept = vh;
                keptValues = vh.ToArray();
                // Lent on demand: what the plan lent is known once validation has asked for it.
                run.LentBytes = Math.Max(run.LentBytes, ((CompiledTrainingPlan<float>)plan).IdleLentBytesLastStep);
            }
            foreach (var p in parameters) run.Parameters = run.Parameters.Concat(p.ToArray()).ToArray();
        }
        run.Replays = CompiledTrainingPlan<float>.GraphReplayCount - replaysBefore;
        run.Loans = backend.IdleLoanCounters.Loans - loansBefore;
        return run;
    }

    // Finding a plan's idle step memory walks every step of the plan: about 2 ms of host time after each step of a
    // 16-layer LM, spent while the device waited for the next launch, though nothing allocated between its steps. It is
    // found only when an allocation between steps needs memory the pool cannot serve.
    [SkippableFact]
    public void IdleStepMemory_IsFoundOnlyWhenAnAllocationBetweenStepsAsksForIt()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "No CUDA device.");
            var backend = (CudaBackend)TestRequire.NotNull(gpu.GetBackend(), "the engine's backend");
            var x = Filled(new[] { 256, 256 }, 1, 1f);
            var w1 = Filled(new[] { 256, 512 }, 3, 0.05f);
            var w2 = Filled(new[] { 512, 256 }, 4, 0.05f);
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var h = gpu.ReLU(gpu.TensorMatMul(x, w1));
                var y = gpu.TensorMatMul(h, w2);
                gpu.ReduceSum(gpu.TensorMultiply(y, y), null);
                plan = scope.CompileTraining(new[] { w1, w2 });
            }
            using (plan)
            {
                var compiled = (CompiledTrainingPlan<float>)plan;
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-4f);
                long replaysBefore = CompiledTrainingPlan<float>.GraphReplayCount;
                for (int s = 0; s < 6; s++)
                {
                    plan.Step();
                    Assert.Equal(0, compiled.IdleLentBytesLastStep);   // nothing asked: nothing was walked
                }
                Assert.True(CompiledTrainingPlan<float>.GraphReplayCount > replaysBefore, "the plan never replayed");
                // A size nothing in the pool serves: it asks, and borrows the plan's idle memory.
                long loansBefore = backend.IdleLoanCounters.Loans;
                using (var odd = backend.AllocateBuffer(12_345))
                {
                    Assert.True(compiled.IdleLentBytesLastStep > 0, "an allocation the pool could not serve found no idle memory");
                    Assert.Equal(loansBefore + 1, backend.IdleLoanCounters.Loans);
                }
                plan.Step();
                Assert.Equal(0, compiled.IdleLentBytesLastStep);
            }
        }
    }

    [SkippableFact]
    public void TrainValidateTrain_MatchesTrainingWithoutValidation_AndKeepsReplaying()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "No CUDA device.");
            bool previous = CompiledTrainingPlan<float>.LendIdleStepMemoryEnabled;
            bool previousSharing = SharedStepScratch.Enabled;
            try
            {
                const int steps = 10;
                // Without co-owned step scratch: the plans trained one after another on this engine would otherwise
                // co-own their scratch, which is never lent (another plan's step writes it).
                SharedStepScratch.Enabled = false;
                var plain = Train(gpu, steps, validate: false);
                var validated = Train(gpu, steps, validate: true);
                CompiledTrainingPlan<float>.LendIdleStepMemoryEnabled = false;
                var unlent = Train(gpu, steps, validate: true);

                Assert.True(plain.Replays >= steps / 2, $"the plan replayed {plain.Replays} of {steps} steps");
                // Validation changes nothing about training: same losses, same weights, and the graph keeps replaying.
                Assert.Equal(plain.Losses, validated.Losses);
                Assert.Equal(plain.Parameters, validated.Parameters);
                Assert.Equal(plain.Replays, validated.Replays);
                // It borrowed the plan's idle memory, and computed what it computes on memory of its own.
                Assert.True(validated.LentBytes > 0, "the plan lent nothing between steps");
                Assert.True(validated.Loans > 0, "validation borrowed nothing");
                Assert.Equal(0, unlent.Loans);
                Assert.Equal(unlent.Validation.Count, validated.Validation.Count);
                for (int i = 0; i < unlent.Validation.Count; i++) Assert.Equal(unlent.Validation[i], validated.Validation[i]);
                Assert.Equal(plain.Losses, unlent.Losses);
            }
            finally
            {
                CompiledTrainingPlan<float>.LendIdleStepMemoryEnabled = previous;
                SharedStepScratch.Enabled = previousSharing;
            }
        }
    }

    // An Adam-trained MLP (moments live on the device) on a captured plan, validated between steps from step 3 on.
    // The plan is returned undisposed when keep is set: it stays between steps, lending its idle memory.
    private static (float[] Losses, ICompiledTrainingPlan<float>? Kept) TrainAdam(
        DirectGpuTensorEngine engine, int steps, bool validate, bool keep)
    {
        var x = Filled(new[] { 256, 256 }, 1, 1f);
        var held = Filled(new[] { 256, 256 }, 2, 1f);
        var w1 = Filled(new[] { 256, 512 }, 3, 0.05f);
        var w2 = Filled(new[] { 512, 256 }, 4, 0.05f);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = engine.ReLU(engine.TensorMatMul(x, w1));
            var y = engine.TensorMatMul(h, w2);
            engine.ReduceSum(engine.TensorMultiply(y, y), null);
            plan = scope.CompileTraining(new[] { w1, w2 });
        }
        var losses = new float[steps];
        try
        {
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-3f);
            for (int s = 0; s < steps; s++)
            {
                losses[s] = plan.Step().ToArray()[0];
                if (!validate || s < 3) continue;
                var vh = engine.ReLU(engine.TensorMatMul(held, w1));
                var vy = engine.TensorMatMul(vh, w2);
                float metric = engine.ReduceSum(engine.TensorMultiply(vy, vy), null).ToArray()[0];
                Assert.False(float.IsNaN(metric) || float.IsInfinity(metric), $"validation at step {s} gave {metric}");
            }
        }
        catch
        {
            plan.Dispose();
            throw;
        }
        if (keep) return (losses, plan);
        plan.Dispose();
        return (losses, null);
    }

    // A plan set up while an earlier plan is between steps allocates its device state (Adam's moments, the step I/O,
    // the clip scratch) from that plan's idle memory. Its capture moves those loans to memory of its own, but only once
    // the capture begins: the step I/O's multi-tensor tables were already written with the lent addresses, so the
    // captured update read and wrote memory the earlier plan lends again. Its losses drifted from the second step after
    // the capture, and validation between its steps (borrowing that memory) made them diverge (6.79 -> 61926).
    [SkippableFact]
    public void A_plan_set_up_while_another_lends_trains_as_it_does_alone()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "No CUDA device.");
            bool previousSharing = SharedStepScratch.Enabled;
            ICompiledTrainingPlan<float>? earlier = null;
            try
            {
                // Without co-owned step scratch, which is never lent: the earlier plan lends all of its scratch.
                SharedStepScratch.Enabled = false;
                const int steps = 12;
                var alone = TrainAdam(gpu, steps, validate: true, keep: false).Losses;
                var (_, kept) = TrainAdam(gpu, steps, validate: false, keep: true);
                earlier = kept;
                var afterAnother = TrainAdam(gpu, steps, validate: true, keep: false).Losses;
                for (int s = 0; s < steps; s++)
                    Assert.True(alone[s] == afterAnother[s],
                        $"step {s}: {afterAnother[s]:G9} set up while another plan lends, {alone[s]:G9} alone");
            }
            finally
            {
                earlier?.Dispose();
                SharedStepScratch.Enabled = previousSharing;
            }
        }
    }

    // Scratch another plan co-owns is never lent: that plan's step writes it while a borrower would still be using it.
    // Two plans on one engine step in turn and co-own scratch; validation between the first plan's steps borrows only
    // what that plan holds alone, and computes what it computes without lending.
    [SkippableFact]
    public void Scratch_another_plan_co_owns_is_never_lent()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "No CUDA device.");
            bool previousSharing = SharedStepScratch.Enabled;
            try
            {
                SharedStepScratch.Enabled = true;
                ICompiledTrainingPlan<float> Compile(int seed)
                {
                    var x = Filled(new[] { 256, 256 }, seed, 1f);
                    var w1 = Filled(new[] { 256, 512 }, seed + 1, 0.05f);
                    var w2 = Filled(new[] { 512, 256 }, seed + 2, 0.05f);
                    using var scope = GraphMode.Enable();
                    var h = gpu.ReLU(gpu.TensorMatMul(x, w1));
                    var y = gpu.TensorMatMul(h, w2);
                    gpu.ReduceSum(gpu.TensorMultiply(y, y), null);
                    var compiled = scope.CompileTraining(new[] { w1, w2 });
                    compiled.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-4f);
                    return compiled;
                }
                using var a = Compile(10);
                using var b = Compile(20);
                var held = Filled(new[] { 256, 256 }, 30, 1f);
                for (int s = 0; s < 8; s++)
                {
                    a.Step();
                    b.Step();
                    a.Step();
                    // An allocation between a's steps: whatever a lends must not be scratch b co-owns.
                    var v = gpu.ReLU(gpu.TensorMatMul(held, held));
                    var lent = ((CompiledTrainingPlan<float>)a).IdleLentBuffersForTest;
                    Assert.All(lent, buffer => Assert.False(
                        gpu.SharedStepScratch.HasOtherOwner(buffer, TestRequire.NotNull(((CompiledTrainingPlan<float>)a).ScratchPoolForTest, "the plan's scratch pool")),
                        "a lent scratch another plan co-owns"));
                    GC.KeepAlive(v);
                }
                Assert.True(gpu.SharedStepScratch.Takes > 0, "the two plans did not co-own any scratch: the test cannot tell");
            }
            finally
            {
                SharedStepScratch.Enabled = previousSharing;
            }
        }
    }
}
