// Copyright (c) AiDotNet. All rights reserved.
// #486 drift hunt: the PaddedBuffer/pre-pack drift passes in isolation but fails
// intermittently under the parallel CI pool. This test deliberately reproduces the
// concurrency pressure locally — many threads running the matmul→sigmoid→matmul chain
// with pooled, garbage-padded operands at once — and asserts every result still
// matches the single-thread fresh reference. If a shared GEMM buffer is being raced,
// this surfaces it as a drift; if it stays green even under heavy local concurrency,
// the contamination is specific to the CI environment (4-vCPU + coverage).

using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

// Joined to the BlasManaged-Stats-Serial collection. The native-BLAS test below asserts the
// OpenBLAS GEMM output is bit-identical to a single-thread reference — which only holds while
// the OpenBLAS internal thread count stays constant (a changed count gives a different, equally
// valid reduction order, i.e. a ~1-ULP difference). Tests that mutate the PROCESS-GLOBAL thread
// count (CpuInferenceConfig.PinBlasThreadsForLatency, ManagedVsNativeGemmAudit, the FlashAttention
// #411 guard) live in this collection; xUnit runs distinct collections in PARALLEL, so leaving
// this class uncollected let one of those flip the count mid-test and drift the bit-match under
// the parallel CI pool (the #519 / #513 contamination class — passes in isolation, flakes on CI).
// Membership in the one serial collection serializes against every such mutator.
[Collection("BlasManaged-Stats-Serial")]
public class PaddedBufferConcurrencyStressTests
{
    private static Tensor<float> RentWithGarbagePadding(int[] shape, float[] logical, float pad)
    {
        int logicalLen = 1;
        for (int i = 0; i < shape.Length; i++) logicalLen *= shape[i];
        Assert.Equal(logicalLen, logical.Length);

        // Construct an EXPLICITLY OVER-SIZED backing so the matmul pad-respect
        // guard has actual garbage past t.Length to misread. The earlier
        // implementation rented through TensorAllocator.Rent and patched the
        // overflow past t.Length, but in test runs the allocator falls through
        // to `new Tensor<T>(shape)` (TensorPool disabled / ForceFreshAllocations)
        // which produces an exact-fit backing — so no pad ever got written and
        // the test was a false green. Wrap an oversized array via
        // Tensor<float>.FromMemory (the same pattern the arena tier uses) so
        // GetDataArray() returns a backing strictly larger than t.Length.
        const int padSlots = 16;  // enough for any SIMD over-read past t.Length
        var backing = new float[logicalLen + padSlots];
        Array.Copy(logical, backing, logicalLen);
        for (int i = logicalLen; i < backing.Length; i++) backing[i] = pad;
        var t = Tensor<float>.FromMemory(new Memory<float>(backing, 0, logicalLen), shape);

        // Defend against accidental future regressions of the FromMemory wrap
        // semantics: if it ever started returning a backing trimmed to t.Length,
        // every rent here would silently become a false green again.
        var actualBacking = t.GetDataArray();
        Assert.True(actualBacking.Length > t.Length,
            $"Expected padded backing for shape [{string.Join(", ", shape)}], " +
            $"got exact length {actualBacking.Length}; FromMemory contract changed?");
        return t;
    }

    [Fact]
    public void MatMulChain_UnderHeavyConcurrency_StaysBitIdenticalToFreshReference()
    {
        const int batch = 4, hidden = 256, outFeat = 128;
        var rng = new Random(1);
        var inputVals = new float[batch * hidden];
        for (int i = 0; i < inputVals.Length; i++) inputVals[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var wVals = new float[hidden * outFeat];
        for (int i = 0; i < wVals.Length; i++) wVals[i] = (float)(rng.NextDouble() * 0.1);
        var w2Vals = new float[outFeat * hidden];
        for (int i = 0; i < w2Vals.Length; i++) w2Vals[i] = (float)(rng.NextDouble() * 0.1);

        // Single-thread fresh reference (ground truth).
        var engine = new CpuEngine();
        var inF = new Tensor<float>(new[] { batch, hidden }); inputVals.CopyTo(inF.AsWritableSpan());
        var wF = new Tensor<float>(new[] { hidden, outFeat }); wVals.CopyTo(wF.AsWritableSpan());
        var w2F = new Tensor<float>(new[] { outFeat, hidden }); w2Vals.CopyTo(w2F.AsWritableSpan());
        var refOut = engine.TensorMatMul(engine.Sigmoid(engine.TensorMatMul(inF, wF)), w2F).AsSpan().ToArray();

        int threads = Math.Max(4, Environment.ProcessorCount * 2);
        const int itersPerThread = 200;
        var drifts = new ConcurrentBag<string>();

        // Explicit start gate so every worker is parked at the gate before the
        // hot loops begin — Parallel.For only guarantees "eventually started",
        // so its bodies can stagger and the shared-buffer race window we're
        // hunting may never co-occur. A Barrier with (threads + 1) parties
        // ensures all workers are AT SignalAndWait() before main thread also
        // signals, then everyone is released simultaneously.
        using var startGate = new Barrier(threads + 1);
        var workers = new Task[threads];
        for (int w = 0; w < threads; w++)
        {
            workers[w] = Task.Factory.StartNew(() =>
            {
                var eng = new CpuEngine();
                startGate.SignalAndWait();
                for (int it = 0; it < itersPerThread; it++)
                {
                    var inPad = RentWithGarbagePadding(new[] { batch, hidden }, inputVals, 999_999f);
                    var wPad = RentWithGarbagePadding(new[] { hidden, outFeat }, wVals, -999_999f);
                    var w2Pad = RentWithGarbagePadding(new[] { outFeat, hidden }, w2Vals, 777_777f);
                    var outPad = eng.TensorMatMul(eng.Sigmoid(eng.TensorMatMul(inPad, wPad)), w2Pad).AsSpan();
                    for (int i = 0; i < refOut.Length; i++)
                        if (outPad[i] != refOut[i])
                        {
                            drifts.Add($"idx {i}: {outPad[i]} vs ref {refOut[i]} (Δ={Math.Abs((double)outPad[i] - refOut[i]):G4})");
                            break;
                        }
                }
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }

        startGate.SignalAndWait();   // release all workers simultaneously
        Task.WaitAll(workers);

        Assert.True(drifts.IsEmpty,
            $"{drifts.Count} concurrent iterations drifted from the fresh reference. First: {(drifts.IsEmpty ? "" : System.Linq.Enumerable.First(drifts))}");
    }

    // Native-BLAS concurrency guard. The chain test above uses small shapes
    // (M*N*K below MatrixMultiplyHelper's BLAS work threshold), so it runs the
    // managed kernel and never enters native OpenBLAS. This test uses a shape
    // ABOVE the threshold (M=8, K=513, N=512 → ~2.1M work) so CpuEngine.TensorMatMul
    // routes to the native cblas_sgemm path. OpenBLAS corrupts its internal
    // per-thread buffers when entered from 2+ managed threads at once — a hard
    // process crash (access violation), not a drift — unless native entry is
    // serialized (BlasProvider._nativeGemmGate). Pre-fix this segfaulted the test
    // host with as few as 2 simultaneous threads; this is the regression guard for
    // that fix. (The K=513 / N=512 weight is also genuinely padded on the net5+
    // pooling tier, so the pad-respect property is exercised here too.)
    [Fact]
    public void MatMul_NativeBlasPath_UnderHeavyConcurrency_DoesNotCrashOrDrift()
    {
        const int M = 8, K = 513, N = 512;
        var rng = new Random(2);
        var xVals = new float[M * K];
        for (int i = 0; i < xVals.Length; i++) xVals[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var wVals = new float[K * N];
        for (int i = 0; i < wVals.Length; i++) wVals[i] = (float)(rng.NextDouble() * 0.1);

        var engine = new CpuEngine();
        var xRef = new Tensor<float>(new[] { M, K }); xVals.CopyTo(xRef.AsWritableSpan());
        var wRef = new Tensor<float>(new[] { K, N }); wVals.CopyTo(wRef.AsWritableSpan());
        var refOut = engine.TensorMatMul(xRef, wRef).AsSpan().ToArray();

        int threads = Math.Max(4, Environment.ProcessorCount * 2);
        const int itersPerThread = 60;
        var drifts = new ConcurrentBag<string>();

        using var startGate = new Barrier(threads + 1);
        var workers = new Task[threads];
        for (int w = 0; w < threads; w++)
        {
            workers[w] = Task.Factory.StartNew(() =>
            {
                var eng = new CpuEngine();
                var x = new Tensor<float>(new[] { M, K }); xVals.CopyTo(x.AsWritableSpan());
                startGate.SignalAndWait();
                for (int it = 0; it < itersPerThread; it++)
                {
                    var wPad = RentWithGarbagePadding(new[] { K, N }, wVals, 999_999f);
                    var outPad = eng.TensorMatMul(x, wPad).AsSpan();
                    for (int i = 0; i < refOut.Length; i++)
                        if (outPad[i] != refOut[i])
                        {
                            drifts.Add($"idx {i}: {outPad[i]} vs ref {refOut[i]} (Δ={Math.Abs((double)outPad[i] - refOut[i]):G4})");
                            break;
                        }
                }
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }

        startGate.SignalAndWait();   // release all workers simultaneously
        Task.WaitAll(workers);

        Assert.True(drifts.IsEmpty,
            $"{drifts.Count} concurrent iterations drifted from the fresh reference. First: {(drifts.IsEmpty ? "" : System.Linq.Enumerable.First(drifts))}");
    }
}
