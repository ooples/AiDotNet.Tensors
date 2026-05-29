// Copyright (c) AiDotNet. All rights reserved.
// Guards the NON-deterministic branch of the native-GEMM concurrency gate
// (BlasProvider._nativeGemmGate). In deterministic mode every concurrent GEMM
// serializes through the native kernel (covered, bit-exact, by
// PaddedBufferConcurrencyStressTests). In NON-deterministic mode the gate is
// taken with Monitor.TryEnter and a contended caller falls back to the
// concurrency-safe managed kernel instead of blocking — so concurrent inference
// stays parallel. This test pins that path: it must NOT crash (the OpenBLAS
// concurrent-entry segfault) and must stay numerically correct.
//
// It asserts a small TOLERANCE, not bit-equality, ON PURPOSE: non-deterministic
// mode mixes the native and managed kernels across threads, and those differ by
// floating-point reduction order. That divergence is the documented contract of
// the mode, not a bug — so the tolerance guards against a genuine fault (a race
// or buffer corruption would drift by O(1)+, like the pre-#492 ~27–60), while
// allowing the legitimate ~1e-4 reduction-order difference. (In deterministic
// mode the companion test still demands exact equality.)

using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

// Toggles the process-wide deterministic flag (and OpenBLAS thread count), so it
// must serialize against the other determinism-sensitive tests.
[Collection("BlasManaged-Stats-Serial")]
public class NativeBlasNonDeterministicConcurrencyTests
{
    [Fact]
    public void MatMul_NativeBlasPath_NonDeterministicMode_ConcurrentDoesNotCrash()
    {
        // M*N*K ~2.1M clears the BLAS work threshold → native cblas path (the
        // shape that segfaulted pre-fix under concurrency).
        const int M = 8, K = 513, N = 512;
        var rng = new Random(3);
        var xVals = new float[M * K];
        for (int i = 0; i < xVals.Length; i++) xVals[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var wVals = new float[K * N];
        for (int i = 0; i < wVals.Length; i++) wVals[i] = (float)(rng.NextDouble() * 0.1);

        // IsDeterministicMode is the MERGED view (thread-local override ?? process-wide).
        // Capture and restore the two layers separately, otherwise restoring via
        // SetDeterministicMode(before) would write the merged value into the process-wide
        // field and leave any pre-existing thread-local override behind — leaking state
        // into later tests (CodeRabbit, PR #491).
        bool? beforeThreadOverride = BlasProvider.GetThreadLocalDeterministicMode();
        if (beforeThreadOverride is not null)
            BlasProvider.SetThreadLocalDeterministicMode(null);
        bool before = BlasProvider.IsDeterministicMode;
        try
        {
            // Non-deterministic mode → the gate uses TryEnter + managed fallback.
            BlasProvider.SetDeterministicMode(false);

            // This regression test guards the NATIVE concurrency gate (_nativeGemmGate).
            // If native BLAS is opted out (AIDOTNET_USE_BLAS=0) or the library fails to
            // load, TensorMatMul stays on the managed path and the test would pass without
            // covering the native path at all — fail loudly instead of silently mis-covering.
            Assert.True(BlasProvider.IsAvailable,
                "This regression test requires native BLAS to exercise _nativeGemmGate; " +
                "native is unavailable (AIDOTNET_USE_BLAS=0 or libopenblas failed to load).");

            var engine = new CpuEngine();
            var xRef = new Tensor<float>(new[] { M, K }); xVals.CopyTo(xRef.AsWritableSpan());
            var wRef = new Tensor<float>(new[] { K, N }); wVals.CopyTo(wRef.AsWritableSpan());
            var refOut = engine.TensorMatMul(xRef, wRef).AsSpan().ToArray();

            int threads = Math.Max(4, Environment.ProcessorCount * 2);
            const int itersPerThread = 60;
            var failures = new ConcurrentBag<string>();
            const float tol = 1e-2f; // see header: allows reduction-order diff, catches corruption

            using var startGate = new Barrier(threads + 1);
            var workers = new Task[threads];
            for (int w = 0; w < threads; w++)
            {
                workers[w] = Task.Factory.StartNew(() =>
                {
                    var eng = new CpuEngine();
                    var x = new Tensor<float>(new[] { M, K }); xVals.CopyTo(x.AsWritableSpan());
                    var wt = new Tensor<float>(new[] { K, N }); wVals.CopyTo(wt.AsWritableSpan());
                    startGate.SignalAndWait();
                    for (int it = 0; it < itersPerThread; it++)
                    {
                        var outc = eng.TensorMatMul(x, wt).AsSpan();
                        for (int i = 0; i < refOut.Length; i++)
                        {
                            float diff = Math.Abs(outc[i] - refOut[i]);
                            if (diff > tol)
                            {
                                failures.Add($"idx {i}: {outc[i]} vs ref {refOut[i]} (|Δ|={diff:G4})");
                                break;
                            }
                        }
                    }
                }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
            }

            startGate.SignalAndWait();
            Task.WaitAll(workers); // re-throws if any worker faulted (e.g. a crash surfaced managed)

            Assert.True(failures.IsEmpty,
                $"{failures.Count} concurrent non-deterministic results exceeded tolerance {tol}. " +
                $"First: {(failures.IsEmpty ? "" : System.Linq.Enumerable.First(failures))}");
        }
        finally
        {
            BlasProvider.SetDeterministicMode(before);                       // restore process-wide
            BlasProvider.SetThreadLocalDeterministicMode(beforeThreadOverride); // restore thread-local
        }
    }
}
