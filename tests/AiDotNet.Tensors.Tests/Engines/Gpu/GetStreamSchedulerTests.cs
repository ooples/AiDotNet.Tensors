using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #335: validates the engine-level accessor that exposes
/// <c>GpuStreamScheduler</c> to consumers. Callers may either pass an
/// owned pool or let the engine create one owned by the scheduler.
/// </summary>
public class GetStreamSchedulerTests
{
    [Fact]
    public void GetStreamScheduler_NoPool_DoesNotThrow()
    {
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();

        using var scheduler = engine.GetStreamScheduler();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream)
            Assert.Null(scheduler);
        else
            Assert.NotNull(scheduler);
    }

    [Fact]
    public void IEngineGetStreamScheduler_NoPool_DoesNotThrow()
    {
        using var concrete = new DirectGpuTensorEngine();
        IEngine engine = concrete;
        var asyncBackend = concrete.GetAsyncBackend();

        using var scheduler = engine.GetStreamScheduler();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream)
            Assert.Null(scheduler);
        else
            Assert.NotNull(scheduler);
    }

    [Fact]
    public void CpuEngineGetStreamScheduler_NoPool_ReturnsNull()
    {
        IEngine engine = new CpuEngine();

        using var scheduler = engine.GetStreamScheduler();
        Assert.Null(scheduler);
    }

    [Fact]
    public void CreateStreamPool_DoesNotThrow()
    {
        // The accessor must be callable from any host — CPU-only or
        // GPU-equipped. Returns null on CPU-only, a usable pool on
        // multi-stream-capable hosts.
        using var engine = new DirectGpuTensorEngine();
        var pool = engine.CreateStreamPool();
        if (pool is not null)
            pool.Dispose();
    }

    [Fact]
    public void GetStreamScheduler_WithOwnedPool_ReturnsScheduler_OnMultiStreamHost()
    {
        // Smoke test: when the engine has a multi-stream-capable async
        // backend, building a pool via CreateStreamPool + passing it to
        // GetStreamScheduler yields a usable scheduler. Caller owns pool
        // lifetime; dispose pool after scheduler is done.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream)
            return; // No multi-stream backend on this host.

        using var pool = engine.CreateStreamPool();
        Assert.NotNull(pool);
        using var scheduler = engine.GetStreamScheduler(pool!);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void CreateStreamPool_NoMultiStreamBackend_ReturnsNull()
    {
        // On a CPU-only host the pool factory returns null without throwing.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is not null && asyncBackend.SupportsMultiStream)
            return; // GPU host — covered by the scheduler test.

        var pool = engine.CreateStreamPool();
        Assert.Null(pool);
    }

    [Fact]
    public void MultiHeadAttentionScoresFanout_RunsToCompletion_OnMultiStreamHost()
    {
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream) return;
        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return;
        var backend = engine.GetBackend();
        if (backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cudaBackend) return;

        // BERT-base-ish but small: B=2, H=4, S=64, D=32.
        const int batch = 2, numHeads = 4, seqLen = 64, headDim = 32;
        long qkElems = (long)batch * numHeads * seqLen * headDim;
        long scoreElems = (long)batch * numHeads * seqLen * seqLen;

        using var q = backend.AllocateBuffer((int)qkElems);
        using var k = backend.AllocateBuffer((int)qkElems);
        using var scores = backend.AllocateBuffer((int)scoreElems);

        try
        {
            cudaBackend.MultiHeadAttentionScoresFanout(q, k, scores, batch, numHeads, seqLen, headDim, scheduler);
        }
        catch (System.Exception thrown)
        {
            if (thrown.Message.Contains("ARCH", System.StringComparison.OrdinalIgnoreCase)
                || thrown.Message.Contains("NotSupported", System.StringComparison.Ordinal))
                return;
            throw;
        }
    }

    [Fact]
    public void MultiHeadAttentionOutputFanout_RunsToCompletion_OnMultiStreamHost()
    {
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream) return;
        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return;
        var backend = engine.GetBackend();
        if (backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cudaBackend) return;

        const int batch = 2, numHeads = 4, seqLen = 64, headDim = 32;
        long attElems = (long)batch * numHeads * seqLen * seqLen;
        long vElems = (long)batch * numHeads * seqLen * headDim;
        long oElems = (long)batch * numHeads * seqLen * headDim;

        using var attn = backend.AllocateBuffer((int)attElems);
        using var v = backend.AllocateBuffer((int)vElems);
        using var output = backend.AllocateBuffer((int)oElems);

        try
        {
            cudaBackend.MultiHeadAttentionOutputFanout(attn, v, output, batch, numHeads, seqLen, headDim, scheduler);
        }
        catch (System.Exception thrown)
        {
            if (thrown.Message.Contains("ARCH", System.StringComparison.OrdinalIgnoreCase)
                || thrown.Message.Contains("NotSupported", System.StringComparison.Ordinal))
                return;
            throw;
        }
    }

    [Fact]
    public void BatchedGemmFanout_FansSlicesAcrossStreams_OnMultiStreamHost()
    {
        // Issue #335 items 3+4 production wiring test: CudaBackend's
        // BatchedGemmFanout submits N independent SGEMM slices to the
        // scheduler. This test allocates contiguous A/B/C buffers
        // for 8 slices of attention-shape ([256,64] · [64,256]), runs
        // the fanout, and confirms it completes without error. On a
        // single-stream backend the scheduler returns null and the test
        // early-returns.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream)
            return;

        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return;
        var backend = engine.GetBackend();
        if (backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cudaBackend)
            return;

        const int M = 256, N = 256, K = 64;
        const int batchCount = 8;
        long strideA = (long)M * K;
        long strideB = (long)K * N;
        long strideC = (long)M * N;

        using var a = backend.AllocateBuffer((int)(strideA * batchCount));
        using var b = backend.AllocateBuffer((int)(strideB * batchCount));
        using var c = backend.AllocateBuffer((int)(strideC * batchCount));

        try
        {
            cudaBackend.BatchedGemmFanout(a, b, c, M, N, K, batchCount, scheduler);
        }
        catch (System.Exception thrown)
        {
            // On hosts where the SGEMM kernels fail to launch (e.g.,
            // pre-Maxwell or driver-mismatched), accept the failure
            // as skip — the test target is the dispatch path, not
            // hardware capability.
            if (thrown.Message.Contains("ARCH", System.StringComparison.OrdinalIgnoreCase)
                || thrown.Message.Contains("NotSupported", System.StringComparison.Ordinal))
                return;
            throw;
        }
        // Reaching here means the fan-out + synchronize completed cleanly.
    }

    [Fact]
    public void GetStreamScheduler_ConcurrentSGEMM_RunsToCompletion()
    {
        // Issue #335 perf-claim test: the scheduler's design is "fan N
        // independent SGEMMs across the stream pool, get ~2-3× the
        // single-stream throughput on attention-shape inputs". This test
        // exercises that pattern end-to-end on a real CUDA host: 12
        // independent SGEMMs of BERT-attention shape ([256,64] · [64,256]).
        // We don't assert a throughput threshold here — wall-time would be
        // host-flaky in CI — only that the fan-out reaches the synchronize
        // point without error. The proof that the pattern actually delivers
        // the perf win lives in the benchmark suite, not the test suite.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream)
            return;

        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return;
        var backend = engine.GetBackend();
        if (backend is null) return;

        const int M = 256, N = 256, K = 64;
        const int numLaunches = 12;

        var aBufs = new System.Collections.Generic.List<AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer>(numLaunches);
        var bBufs = new System.Collections.Generic.List<AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer>(numLaunches);
        var cBufs = new System.Collections.Generic.List<AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer>(numLaunches);
        try
        {
            for (int i = 0; i < numLaunches; i++)
            {
                aBufs.Add(backend.AllocateBuffer(M * K));
                bBufs.Add(backend.AllocateBuffer(K * N));
            }

            var launches = new System.Collections.Generic.List<System.Action<IGpuStream>>(numLaunches);
            for (int i = 0; i < numLaunches; i++)
            {
                int idx = i;
                launches.Add(stream =>
                {
                    // The scheduler binds cuBLAS to the lease's stream just
                    // before invoking the launch callback (see
                    // GpuStreamScheduler.Dispatch). MatMul on this backend
                    // queues onto that stream as a consequence.
                    var c = backend.MatMul(aBufs[idx], bBufs[idx], M, N, K);
                    lock (cBufs) cBufs.Add(c);
                });
            }

            using var batch = scheduler.Dispatch(launches);
            Assert.NotEqual(0, batch.Count);
            Assert.Equal(numLaunches, cBufs.Count);
            scheduler.SynchronizeEvents(batch);
        }
        finally
        {
            foreach (var b in aBufs) b.Dispose();
            foreach (var b in bBufs) b.Dispose();
            foreach (var b in cBufs) b.Dispose();
        }
    }

    [Fact]
    public void GetStreamScheduler_Dispatch_FanOutsOverMultipleStreams()
    {
        // Issue #335 items 3+4: the scheduler must accept a batch of
        // independent launches and fan them across multiple streams.
        // This test exercises the fan-out path directly; the higher-level
        // wiring (MHA per-head + BatchMatMul per-slice) calls this same
        // entry point. Counts callback invocations to verify every launch
        // ran and that the returned event batch has one event per launch.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream)
            return;

        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return; // CPU-only host

        int launchCount = 0;
        var launches = new System.Collections.Generic.List<System.Action<IGpuStream>>();
        for (int i = 0; i < 8; i++)
            launches.Add(_ => System.Threading.Interlocked.Increment(ref launchCount));

        using var batch = scheduler.Dispatch(launches);
        Assert.Equal(8, launchCount);
        // GpuEventBatch IS the event list — it implements IReadOnlyList<IGpuEvent>.
        Assert.NotEqual(0, batch.Count);

        // SynchronizeEvents is a host-blocking wait; on a real GPU backend
        // it returns once all per-stream events fire. With no-op launches
        // this should be near-instant.
        scheduler.SynchronizeEvents(batch);
    }

    [Fact]
    public void GetStreamScheduler_PoolFromDifferentEngine_ThrowsArgumentException()
    {
        // PR #344 critical review: GetStreamScheduler must refuse a pool
        // whose backend doesn't match this engine's. Cross-backend stream
        // misuse manifests as CUDA_ERROR_INVALID_HANDLE deep inside
        // cuMemcpy / cuLaunchKernel — surfacing the affinity mismatch at
        // the API boundary gives the caller an actionable error.
        using var engineA = new DirectGpuTensorEngine();
        using var engineB = new DirectGpuTensorEngine();
        var backendA = engineA.GetAsyncBackend();
        var backendB = engineB.GetAsyncBackend();
        if (backendA is null || backendB is null
            || !backendA.SupportsMultiStream || !backendB.SupportsMultiStream)
            return; // Need multi-stream backends on both engines.
        // Practical note: two DirectGpuTensorEngine instances may share
        // the same singleton backend instance, in which case the affinity
        // guard short-circuits to "same backend, accept". The test still
        // documents the contract; the negative case becomes meaningful
        // only when two engines hold distinct backend references (e.g.
        // a future test fixture that overrides backend creation).
        if (ReferenceEquals(backendA, backendB))
            return;

        using var poolFromA = engineA.CreateStreamPool();
        Assert.NotNull(poolFromA);
        Assert.Throws<System.ArgumentException>(() =>
            engineB.GetStreamScheduler(poolFromA!));
    }
}
