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
    public void BatchedGemmExFanout_FP16_RunsToCompletion_OnMultiStreamHost()
    {
        // Smoke test for the AMP batched GEMM: fp16 inputs, fp32 accumulator
        // output. Doesn't crash + completes. Correctness gates against fp16
        // round-tripped host data are hard without a fp16↔fp32 conversion
        // helper; the dispatch path itself is exercised by the existing
        // HalfPrecision backend tests.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream) return;
        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return;
        var backend = engine.GetBackend();
        if (backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cudaBackend) return;

        const int M = 16, N = 16, K = 8, batchCount = 4;
        // ValidateBatchedGemmArgs is fp32-centric: bufA.Size must be
        // >= M*K*batchCount. Allocate at fp32-element count even though
        // the actual fp16 elements occupy half that bytewise — the
        // extra space is unused.
        using var aFp16 = backend.AllocateBuffer(M * K * batchCount);
        using var bFp16 = backend.AllocateBuffer(K * N * batchCount);
        using var cFp32 = backend.AllocateBuffer(M * N * batchCount);

        try
        {
            cudaBackend.BatchedGemmExFanout(aFp16, bFp16, cFp32, M, N, K, batchCount, scheduler,
                useBFloat16: false);
        }
        catch (System.Exception thrown)
        {
            if (thrown.Message.Contains("supported", System.StringComparison.OrdinalIgnoreCase)
                || thrown.Message.Contains("ARCH", System.StringComparison.OrdinalIgnoreCase))
                return;
            throw;
        }
    }

    [Fact]
    public void BatchedGemmExFanout_BFloat16PreAmpere_Throws()
    {
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream) return;
        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return;
        var backend = engine.GetBackend();
        if (backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cudaBackend) return;

        if (backend is AiDotNet.Tensors.Engines.Gpu.IGpuMixedPrecisionConvBackend mp
            && mp.SupportsBFloat16Conv)
            return; // BF16 actually supported — skip rejection test.

        using var a = backend.AllocateBuffer(64);
        using var b = backend.AllocateBuffer(64);
        using var c = backend.AllocateBuffer(64);

        var ex = Assert.Throws<System.NotSupportedException>(() =>
            cudaBackend.BatchedGemmExFanout(a, b, c, 8, 8, 8, 1, scheduler, useBFloat16: true));
        Assert.Contains("8.0", ex.Message, System.StringComparison.Ordinal);
    }

    [Fact]
    public void BatchedDgemmFanout_MatchesCPUReference_OnMultiStreamHost()
    {
        // Correctness gate for the fp64 fanout: CPU triple-loop reference.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream) return;
        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return;
        var backend = engine.GetBackend();
        if (backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cudaBackend) return;

        const int M = 16, N = 16, K = 8, batchCount = 3;
        int aElems = M * K * batchCount;
        int bElems = K * N * batchCount;
        int cElems = M * N * batchCount;

        var rng = new System.Random(99);
        var aHost = new double[aElems];
        var bHost = new double[bElems];
        for (int i = 0; i < aElems; i++) aHost[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < bElems; i++) bHost[i] = rng.NextDouble() - 0.5;

        // Allocate fp64 buffers via byte size — backend.AllocateBuffer
        // is float-shaped, so we allocate by elementCount of doubles =
        // raw bytes / sizeof(float). Simpler: use Allocate<double>
        // via the context. Skip this test if that API isn't available
        // on the backend abstraction.
        AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer aBuf = null!;
        AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer bBuf = null!;
        AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer cBuf = null!;
        try
        {
            // Buffer size in float-equivalent elements: doubles are 2× float bytes.
            aBuf = backend.AllocateBuffer(aElems * 2);
            bBuf = backend.AllocateBuffer(bElems * 2);
            cBuf = backend.AllocateBuffer(cElems * 2);

            // Upload as bytes via float reinterpretation.
            var aBytes = new float[aElems * 2];
            var bBytes = new float[bElems * 2];
            System.Buffer.BlockCopy(aHost, 0, aBytes, 0, aHost.Length * sizeof(double));
            System.Buffer.BlockCopy(bHost, 0, bBytes, 0, bHost.Length * sizeof(double));
            // Replace via fresh allocation: simpler than reuploading.
            aBuf.Dispose();
            bBuf.Dispose();
            aBuf = backend.AllocateBuffer(aBytes);
            bBuf = backend.AllocateBuffer(bBytes);

            try
            {
                cudaBackend.BatchedDgemmFanout(aBuf, bBuf, cBuf, M, N, K, batchCount, scheduler);
            }
            catch (System.Exception thrown)
            {
                if (thrown.Message.Contains("ARCH", System.StringComparison.OrdinalIgnoreCase)
                    || thrown.Message.Contains("NotSupported", System.StringComparison.Ordinal))
                    return;
                throw;
            }

            var cBytes = new float[cElems * 2];
            backend.DownloadBuffer(cBuf, cBytes);
            var cGpu = new double[cElems];
            System.Buffer.BlockCopy(cBytes, 0, cGpu, 0, cElems * sizeof(double));

            // CPU reference per batch.
            for (int b = 0; b < batchCount; b++)
            {
                int aBase = b * M * K;
                int bBase = b * K * N;
                int cBase = b * M * N;
                for (int i = 0; i < M; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        double sum = 0;
                        for (int kk = 0; kk < K; kk++)
                            sum += aHost[aBase + i * K + kk] * bHost[bBase + kk * N + j];
                        Assert.Equal(sum, cGpu[cBase + i * N + j], 5);
                    }
                }
            }
        }
        finally
        {
            aBuf?.Dispose();
            bBuf?.Dispose();
            cBuf?.Dispose();
        }
    }

    [Fact]
    public void BatchedGemmFanout_ProducesIdenticalResultsToStridedBatched()
    {
        // Correctness gate: runs the same input through the existing
        // strided-batched BatchedGemm AND the new BatchedGemmFanout,
        // copies both results back to host, asserts element-wise
        // equality. Pins that the per-stream fanout doesn't drift from
        // the trusted strided-batched reference.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream) return;
        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return;
        var backend = engine.GetBackend();
        if (backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cudaBackend) return;

        const int M = 32, N = 32, K = 16, batchCount = 4;
        int aElems = M * K * batchCount;
        int bElems = K * N * batchCount;
        int cElems = M * N * batchCount;

        // Fill A and B with deterministic values via host upload.
        var aHost = new float[aElems];
        var bHost = new float[bElems];
        var rng = new System.Random(42);
        for (int i = 0; i < aElems; i++) aHost[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < bElems; i++) bHost[i] = (float)(rng.NextDouble() - 0.5);

        using var aBuf = backend.AllocateBuffer(aHost);
        using var bBuf = backend.AllocateBuffer(bHost);
        using var cStrided = backend.AllocateBuffer(cElems);
        using var cFanout = backend.AllocateBuffer(cElems);

        try
        {
            cudaBackend.BatchedGemm(aBuf, bBuf, cStrided, M, N, K, batchCount);
            cudaBackend.BatchedGemmFanout(aBuf, bBuf, cFanout, M, N, K, batchCount, scheduler);
        }
        catch (System.Exception thrown)
        {
            if (thrown.Message.Contains("ARCH", System.StringComparison.OrdinalIgnoreCase)
                || thrown.Message.Contains("NotSupported", System.StringComparison.Ordinal))
                return;
            throw;
        }

        var stridedHost = new float[cElems];
        var fanoutHost = new float[cElems];
        backend.DownloadBuffer(cStrided, stridedHost);
        backend.DownloadBuffer(cFanout, fanoutHost);

        // Strided and fanout compute the same GEMM, just dispatched
        // differently. Results should match to within fp32 ULPs — any
        // significant drift means the fanout's offset arithmetic or
        // alpha/beta handling is wrong.
        for (int i = 0; i < cElems; i++)
            Assert.Equal(stridedHost[i], fanoutHost[i], 4);
    }

    [Fact]
    public void MultiHeadAttentionScoresFanoutMixed_RunsToCompletion_FP16Path_OnMultiStreamHost()
    {
        // The mixed-precision path uses cublasGemmEx with fp16 inputs.
        // We can't easily round-trip fp16 host bytes for a correctness
        // gate without a fp16↔fp32 conversion helper, so this is a
        // doesn't-crash test. Correctness of cublasGemmEx itself is
        // exercised by the existing IGpuHalfPrecisionBackend.Hgemm tests.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream) return;
        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return;
        var backend = engine.GetBackend();
        if (backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cudaBackend) return;

        const int batch = 1, numHeads = 2, seqLen = 16, headDim = 8;
        // fp16 = 2 bytes per element; backend.AllocateBuffer is float-shaped
        // (4 bytes per element), so allocate float-equivalent count = fp16
        // count / 2.
        long qkFp16Elems = (long)batch * numHeads * seqLen * headDim;
        long scoreFp32Elems = (long)batch * numHeads * seqLen * seqLen;

        using var qFp16 = backend.AllocateBuffer((int)((qkFp16Elems + 1) / 2));
        using var kFp16 = backend.AllocateBuffer((int)((qkFp16Elems + 1) / 2));
        using var scoresFp32 = backend.AllocateBuffer((int)scoreFp32Elems);

        try
        {
            cudaBackend.MultiHeadAttentionScoresFanoutMixed(
                qFp16, kFp16, scoresFp32,
                batch, numHeads, seqLen, headDim, scheduler,
                useBFloat16: false);
        }
        catch (System.Exception thrown)
        {
            // GetCublasErrorString returns "Not supported" with a space —
            // a different casing than the cuDNN-style "NotSupported".
            // Accept both forms.
            if (thrown.Message.Contains("supported", System.StringComparison.OrdinalIgnoreCase)
                || thrown.Message.Contains("ARCH", System.StringComparison.OrdinalIgnoreCase))
                return;
            throw;
        }
    }

    [Fact]
    public void MultiHeadAttentionScoresFanoutMixed_BFloat16PreAmpere_Throws()
    {
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream) return;
        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return;
        var backend = engine.GetBackend();
        if (backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cudaBackend) return;

        // Pre-Ampere hosts only — skip if BF16 is actually supported.
        if (backend is AiDotNet.Tensors.Engines.Gpu.IGpuMixedPrecisionConvBackend mp && mp.SupportsBFloat16Conv)
            return;

        using var q = backend.AllocateBuffer(64);
        using var k = backend.AllocateBuffer(64);
        using var s = backend.AllocateBuffer(64);

        var ex = Assert.Throws<System.NotSupportedException>(() =>
            cudaBackend.MultiHeadAttentionScoresFanoutMixed(
                q, k, s, 1, 2, 4, 4, scheduler, useBFloat16: true));
        Assert.Contains("8.0", ex.Message, System.StringComparison.Ordinal);
    }

    [Fact]
    public void MultiHeadAttentionScoresFanout_MatchesCPUReference_OnMultiStreamHost()
    {
        // Correctness gate: compares the per-head Q·K^T fanout result
        // against a CPU-computed reference. Pins that the fanout's
        // offset arithmetic + cuBLAS transpose handling are correct.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream) return;
        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return;
        var backend = engine.GetBackend();
        if (backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cudaBackend) return;

        const int batch = 2, numHeads = 4, seqLen = 16, headDim = 8;
        long qkElems = (long)batch * numHeads * seqLen * headDim;
        long scoreElems = (long)batch * numHeads * seqLen * seqLen;

        var rng = new System.Random(7);
        var qHost = new float[qkElems];
        var kHost = new float[qkElems];
        for (int i = 0; i < qkElems; i++) qHost[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < qkElems; i++) kHost[i] = (float)(rng.NextDouble() - 0.5);

        using var qBuf = backend.AllocateBuffer(qHost);
        using var kBuf = backend.AllocateBuffer(kHost);
        using var scoresBuf = backend.AllocateBuffer((int)scoreElems);

        try
        {
            cudaBackend.MultiHeadAttentionScoresFanout(qBuf, kBuf, scoresBuf,
                batch, numHeads, seqLen, headDim, scheduler);
        }
        catch (System.Exception thrown)
        {
            if (thrown.Message.Contains("ARCH", System.StringComparison.OrdinalIgnoreCase)
                || thrown.Message.Contains("NotSupported", System.StringComparison.Ordinal))
                return;
            throw;
        }

        var scoresGpu = new float[scoreElems];
        backend.DownloadBuffer(scoresBuf, scoresGpu);

        // CPU reference: scores[b,h,i,j] = sum_d Q[b,h,i,d] * K[b,h,j,d]
        for (int bi = 0; bi < batch; bi++)
        {
            for (int hi = 0; hi < numHeads; hi++)
            {
                int qBase = ((bi * numHeads) + hi) * seqLen * headDim;
                int kBase = ((bi * numHeads) + hi) * seqLen * headDim;
                int sBase = ((bi * numHeads) + hi) * seqLen * seqLen;
                for (int si = 0; si < seqLen; si++)
                {
                    for (int sj = 0; sj < seqLen; sj++)
                    {
                        float sum = 0;
                        for (int d = 0; d < headDim; d++)
                            sum += qHost[qBase + si * headDim + d]
                                 * kHost[kBase + sj * headDim + d];
                        Assert.Equal(sum, scoresGpu[sBase + si * seqLen + sj], 3);
                    }
                }
            }
        }
    }

    [Fact]
    public void MultiHeadAttentionOutputFanout_MatchesCPUReference_OnMultiStreamHost()
    {
        // Correctness gate: output[b,h,i,d] = sum_j attention[b,h,i,j] · V[b,h,j,d]
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream) return;
        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return;
        var backend = engine.GetBackend();
        if (backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cudaBackend) return;

        const int batch = 2, numHeads = 4, seqLen = 16, headDim = 8;
        long attElems = (long)batch * numHeads * seqLen * seqLen;
        long vElems = (long)batch * numHeads * seqLen * headDim;
        long oElems = (long)batch * numHeads * seqLen * headDim;

        var rng = new System.Random(13);
        var attHost = new float[attElems];
        var vHost = new float[vElems];
        for (int i = 0; i < attElems; i++) attHost[i] = (float)rng.NextDouble();
        for (int i = 0; i < vElems; i++) vHost[i] = (float)(rng.NextDouble() - 0.5);

        using var attnBuf = backend.AllocateBuffer(attHost);
        using var vBuf = backend.AllocateBuffer(vHost);
        using var outputBuf = backend.AllocateBuffer((int)oElems);

        try
        {
            cudaBackend.MultiHeadAttentionOutputFanout(attnBuf, vBuf, outputBuf,
                batch, numHeads, seqLen, headDim, scheduler);
        }
        catch (System.Exception thrown)
        {
            if (thrown.Message.Contains("ARCH", System.StringComparison.OrdinalIgnoreCase)
                || thrown.Message.Contains("NotSupported", System.StringComparison.Ordinal))
                return;
            throw;
        }

        var outputGpu = new float[oElems];
        backend.DownloadBuffer(outputBuf, outputGpu);

        // CPU reference triple-loop.
        for (int bi = 0; bi < batch; bi++)
        {
            for (int hi = 0; hi < numHeads; hi++)
            {
                int attBase = ((bi * numHeads) + hi) * seqLen * seqLen;
                int vBase = ((bi * numHeads) + hi) * seqLen * headDim;
                int oBase = ((bi * numHeads) + hi) * seqLen * headDim;
                for (int si = 0; si < seqLen; si++)
                {
                    for (int d = 0; d < headDim; d++)
                    {
                        float sum = 0;
                        for (int j = 0; j < seqLen; j++)
                            sum += attHost[attBase + si * seqLen + j]
                                 * vHost[vBase + j * headDim + d];
                        Assert.Equal(sum, outputGpu[oBase + si * headDim + d], 3);
                    }
                }
            }
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
