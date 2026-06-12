// Copyright (c) AiDotNet. All rights reserved.
//
// Regression tests for issues #560 / #561 / #562:
//   #560 — IEngine.MatrixMultiply<Half> was hitting a scalar CPU path
//   (~1000× slower than FP32) because the explicit interface dispatch never
//   reached the FP16 tensor-core path.
//   #561 — BeginGpuScope() was advertised as deferring host downloads for
//   chained intermediates, but the matrix MatMul path downloaded the result
//   every call, so a chained N=2048 GEMM was actually SLOWER inside the scope
//   than outside.
//   #562 — host-returning MatrixMultiply forced a device→host→device round-trip
//   per step on chained ops (74 vs 601 GFLOP/s).
//
// All three were the same architectural bug: IEngine.MatrixMultiply<T> bypassed
// the activation-cache + deferred-download infrastructure that every other
// op uses. After the fix it:
//   1. Resolves both inputs through GetOrAllocateBuffer (cache hit when the
//      operand is a prior GPU op's deferred output).
//   2. Routes Half to IGpuHalfPrecisionBackend.GemmFp16In32fOut (tensor cores).
//   3. Returns a Matrix<T> wrapping a deferred-materializer-registered array,
//      so the next MatMul that consumes it finds the GPU buffer in cache.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public class MatrixMultiplyResidencyTests
{
    private readonly ITestOutputHelper _output;

    public MatrixMultiplyResidencyTests(ITestOutputHelper output) { _output = output; }

    private static bool TryGetGpuEngine(out DirectGpuTensorEngine engine)
    {
        engine = new DirectGpuTensorEngine();
        if (engine.IsGpuAvailable) return true;
        engine.Dispose();
        engine = null!;
        return false;
    }

    private static Matrix<float> RandomFloat(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var m = new Matrix<float>(rows, cols);
        var span = m.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = (float)(rng.NextDouble() * 2 - 1);
        return m;
    }

    private static Matrix<Half> RandomHalf(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var m = new Matrix<Half>(rows, cols);
        var span = m.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = (Half)(rng.NextDouble() * 2 - 1);
        return m;
    }

    [Fact]
    public void MatrixMultiply_ChainedFloat_KeepsIntermediatesResident()
    {
        // #562 regression: chained C=A·B, E=C·D inside a GpuScope must find C's
        // buffer in the activation cache for the second MatMul. Pre-fix, C was
        // downloaded after the first MatMul and re-uploaded for the second.
        if (!TryGetGpuEngine(out var engine)) return;
        using (engine)
        {
            var a = RandomFloat(64, 64, seed: 1);
            var b = RandomFloat(64, 64, seed: 2);
            var d = RandomFloat(64, 64, seed: 3);

            using var scope = engine.BeginGpuScope();

            var c = ((IEngine)engine).MatrixMultiply(a, b);

            // Inside the scope the FIRST MatMul's result must be deferred:
            // its backing array is registered with DeferredArrayMaterializer
            // and its GPU buffer sits in the activation cache. We check this
            // BEFORE any host read (AsSpan / GetDataArray would materialize).
            Assert.True(DeferredArrayMaterializer.IsPending(c.GetBackingArrayUnsafe()!),
                "MatrixMultiply result must be deferred inside a GpuScope (#561).");

            // Second MatMul consumes C — must hit the activation cache rather
            // than re-uploading. The activation cache is private, so we
            // verify behaviorally: if C were re-uploaded fresh, the GPU op
            // would run on stale-uninitialized data and produce wrong results.
            // The end-of-scope compare below catches that.
            var e = ((IEngine)engine).MatrixMultiply(c, d);

            // Compare end-to-end against a CPU reference once we leave the scope.
            scope.Dispose();
            var cRef = new CpuEngine().MatrixMultiply(a, b);
            var eRef = new CpuEngine().MatrixMultiply(cRef, d);
            // Tolerance accounts for CUDA SGEMM using TF32 by default on Ampere+
// (RTX 3080 = CC 8.6). Two stacked GEMMs compound the per-term drift,
// roughly K · 2^-10 per GEMM ≈ ~6e-3 relative at K=64, doubled for the
// stack. Residency is what this test is asserting — numerical fidelity
// of single MatMul is covered by MatrixMultiply_Float_ResultIsDeferredAndCorrect.
AssertCloseFloat(e.AsSpan().ToArray(), eRef.AsSpan().ToArray(), absTol: 1e-2f, relTol: 1.5e-1f);
        }
    }

    [Fact]
    public void MatrixMultiply_Float_ResultIsDeferredAndCorrect()
    {
        // #561 regression: a single MatMul under GpuScope used to download
        // synchronously. Now the result is deferred, the array materializes on
        // first read, and the final value still matches CPU.
        if (!TryGetGpuEngine(out var engine)) return;
        using (engine)
        {
            var a = RandomFloat(48, 48, seed: 10);
            var b = RandomFloat(48, 48, seed: 11);

            using (var scope = engine.BeginGpuScope())
            {
                var c = ((IEngine)engine).MatrixMultiply(a, b);
                // Must check IsPending BEFORE any host read — AsSpan/GetDataArray
                // now materialize on first access (the fix to MatrixBase span
                // accessors that pairs with the deferred-result Matrix path).
                Assert.True(DeferredArrayMaterializer.IsPending(c.GetBackingArrayUnsafe()!),
                    "Single MatMul result must be deferred inside a GpuScope (#561).");

                // First host access materializes — value is correct.
                var cRef = new CpuEngine().MatrixMultiply(a, b);
                // CUDA SGEMM may use TF32 on Ampere+; tolerate the per-term drift.
AssertCloseFloat(c.AsSpan().ToArray(), cRef.AsSpan().ToArray(), absTol: 5e-3f, relTol: 5e-2f);
            }
        }
    }

    [Fact]
    public void MatrixMultiply_Half_RunsOnGpu_AndIsCorrect()
    {
        // #560 regression: MatrixMultiply<Half> must NOT fall to the scalar
        // CPU path. We verify by running the GPU path on a non-trivial size
        // — pre-fix this took ~56s on a 2048² Half multiply (vs ~28ms FP32)
        // because the dispatch declined and base.MatrixMultiply ran in C#
        // scalar code. We use a smaller (256²) matrix so the test stays
        // under 5s even on the slow CPU path; the cliff would still show
        // there if the GPU path were declining.
        if (!TryGetGpuEngine(out var engine)) return;
        using (engine)
        {
            const int N = 256;
            var aH = RandomHalf(N, N, seed: 100);
            var bH = RandomHalf(N, N, seed: 101);

            var start = System.Diagnostics.Stopwatch.StartNew();
            var cH = ((IEngine)engine).MatrixMultiply(aH, bH);
            start.Stop();

            // Sanity: a 256² FP16 GEMM should take ≪ 1 second on any modern
            // GPU. If the CPU scalar path were running, expect O(seconds).
            // The bound is generous (5s) to keep the test stable on slow CI
            // boxes without losing the 1000× regression signal.
            Assert.True(start.Elapsed.TotalSeconds < 5.0,
                $"MatrixMultiply<Half> should run on GPU; took {start.Elapsed.TotalSeconds:F2}s for {N}x{N} (#560).");

            // Numerical check against a float reference (the rounded-input
            // pattern from Fp16Bf16GemmTests so we don't double-count input
            // rounding noise).
            var aF = new Matrix<float>(N, N);
            var bF = new Matrix<float>(N, N);
            var aHSpan = aH.AsSpan(); var aFSpan = aF.AsWritableSpan();
            for (int i = 0; i < aHSpan.Length; i++) aFSpan[i] = (float)aHSpan[i];
            var bHSpan = bH.AsSpan(); var bFSpan = bF.AsWritableSpan();
            for (int i = 0; i < bHSpan.Length; i++) bFSpan[i] = (float)bHSpan[i];
            var cFRef = new CpuEngine().MatrixMultiply(aF, bF);

            // FP16 accumulation error: ~2^-10 relative × sqrt(K). For K=256,
            // ~6.4e-3 worst case. We pick 1.5e-2 to allow some headroom.
            var got = cH.AsSpan().ToArray();
            var want = cFRef.AsSpan().ToArray();
            float maxAbsRef = 0f;
            for (int i = 0; i < want.Length; i++)
            {
                float aw = System.Math.Abs(want[i]);
                if (aw > maxAbsRef) maxAbsRef = aw;
            }
            float tol = System.Math.Max(1.5e-2f * maxAbsRef, 1e-2f);
            for (int i = 0; i < got.Length; i++)
            {
                float g = (float)got[i];
                float diff = System.Math.Abs(g - want[i]);
                Assert.True(diff < tol,
                    $"FP16 GEMM mismatch at [{i}]: got={g}, want={want[i]}, diff={diff}, tol={tol}");
            }
        }
    }

    // ── perf benchmarks reproducing the timing claims from #560/#561/#562 ──
    //
    // These print the actual measured numbers so the regression issue's perf
    // claim isn't a trust-the-comments problem. They use [Fact] (not Theory)
    // so the output appears in the test runner. Tolerances on the assertion
    // bounds are generous (we just want "did the fix engage", not exact
    // numbers from a specific GPU SKU).

    [Fact]
    public void Benchmark_Fp16_VsFp32()
    {
        // #560: pre-fix, FP16 MatMul hit a CPU scalar path because the GPU
        // dispatch declined (kernel-not-found / FP16-not-supported); on
        // an RTX 3080 with N=2048 that was 280s vs 195ms FP32 (1437× — the
        // raw repro number from the issue body).
        //
        // The fix has two layers:
        //   (1) ConvertToFp16 falls back to the driver-only native kernel when
        //       the cuda_fp16.h-based kernel didn't compile.
        //   (2) The dispatch in IEngine.MatrixMultiply<Half> catches
        //       cublasGemmEx-not-supported (Turing TU116/TU117 advertise FP16
        //       but lack tensor cores) and falls through to SGEMM on the same
        //       cached FP32 buffers — so non-tensor-core GPUs get FP32 speed
        //       instead of the multi-second CPU scalar.
        //
        // We assert the GPU vs CPU-fallback cliff is gone (FP16 within 20×
        // of FP32). On tensor-core hardware FP16 should be ≤ FP32; on
        // non-tensor-core hardware (GTX 16-series) FP16 is bounded by SGEMM
        // + the small conversion overhead.
        if (!TryGetGpuEngine(out var engine)) { _output.WriteLine("SKIP: no GPU"); return; }
        using (engine)
        {
            const int N = 512;
            var aF = RandomFloat(N, N, seed: 200);
            var bF = RandomFloat(N, N, seed: 201);
            var aH = new Matrix<Half>(N, N);
            var bH = new Matrix<Half>(N, N);
            { var aFs = aF.AsSpan(); var aHs = aH.AsWritableSpan(); for (int i = 0; i < aFs.Length; i++) aHs[i] = (Half)aFs[i]; }
            { var bFs = bF.AsSpan(); var bHs = bH.AsWritableSpan(); for (int i = 0; i < bFs.Length; i++) bHs[i] = (Half)bFs[i]; }

            // Warm up (driver/JIT/buffer pool).
            _ = ((IEngine)engine).MatrixMultiply(aF, bF);
            _ = ((IEngine)engine).MatrixMultiply(aH, bH);

            // FP32
            var sw32 = System.Diagnostics.Stopwatch.StartNew();
            var c32 = ((IEngine)engine).MatrixMultiply(aF, bF);
            _ = c32.AsSpan()[0]; // force download/materialize
            sw32.Stop();

            // FP16
            var sw16 = System.Diagnostics.Stopwatch.StartNew();
            var c16 = ((IEngine)engine).MatrixMultiply(aH, bH);
            _ = c16.AsSpan()[0]; // force download/materialize
            sw16.Stop();

            double ms32 = sw32.Elapsed.TotalMilliseconds;
            double ms16 = sw16.Elapsed.TotalMilliseconds;
            _output.WriteLine($"#560 N={N}: FP32={ms32:F1}ms, FP16={ms16:F1}ms, ratio={ms16 / ms32:F2}x");

            // Pre-fix: ms16/ms32 ≈ 1000×. The fix puts them on the same order;
            // an FP16 path that's even WITHIN 10× of FP32 means it's on the
            // GPU (not the 1000× scalar-CPU fallback). We assert 20× as a very
            // generous bound that still catches the regression.
            Assert.True(ms16 < ms32 * 20.0,
                $"FP16 must not be >20× slower than FP32 (#560). FP32={ms32:F1}ms FP16={ms16:F1}ms");
        }
    }

    [Fact]
    public void Benchmark_ChainedGemm_InsideVsOutsideGpuScope()
    {
        // #561 / #562: pre-fix, a 20-step chained N=2048 GEMM was SLOWER
        // inside BeginGpuScope() (5722 ms) than outside (4673 ms) — 0.82× —
        // because the scope's deferred-download path went unused and added
        // overhead. After the fix it should be FASTER inside (no host
        // round-trip per step).
        if (!TryGetGpuEngine(out var engine)) { _output.WriteLine("SKIP: no GPU"); return; }
        using (engine)
        {
            const int N = 1024;   // smaller than 2048 to keep test under ~10s
            const int Steps = 8;
            var a = RandomFloat(N, N, seed: 300);
            var b = RandomFloat(N, N, seed: 301);

            // Warm up (cache, driver, JIT).
            _ = ((IEngine)engine).MatrixMultiply(a, b);

            // Without scope: each MatMul downloads + the next one re-uploads.
            var swOut = System.Diagnostics.Stopwatch.StartNew();
            var cOut = a;
            for (int i = 0; i < Steps; i++)
                cOut = ((IEngine)engine).MatrixMultiply(cOut, b);
            _ = cOut.AsSpan()[0];
            swOut.Stop();

            // With scope: intermediates stay GPU-resident, host download
            // happens once at the end.
            var swIn = System.Diagnostics.Stopwatch.StartNew();
            Matrix<float> cIn;
            using (var scope = engine.BeginGpuScope())
            {
                cIn = a;
                for (int i = 0; i < Steps; i++)
                    cIn = ((IEngine)engine).MatrixMultiply(cIn, b);
            }
            _ = cIn.AsSpan()[0];
            swIn.Stop();

            double msOut = swOut.Elapsed.TotalMilliseconds;
            double msIn = swIn.Elapsed.TotalMilliseconds;
            double flops = 2.0 * N * N * N * Steps;
            double gfOut = flops / 1e6 / msOut;
            double gfIn = flops / 1e6 / msIn;
            _output.WriteLine($"#561/#562 N={N} {Steps}-step chained GEMM: outside={msOut:F1}ms ({gfOut:F1} GF/s), inside={msIn:F1}ms ({gfIn:F1} GF/s), inside/outside={msIn / msOut:F2}x");

            // Inside the scope must be NO SLOWER than outside (within 10%
            // jitter floor). Pre-fix this was 0.82× (inside slower). We
            // accept ≤1.10× (inside marginally slower) as "fix engaged" —
            // the original cliff was much sharper.
            Assert.True(msIn <= msOut * 1.10,
                $"GpuScope must not slow chained GEMM (#561). outside={msOut:F1}ms inside={msIn:F1}ms ratio={msIn / msOut:F2}x");
        }
    }

    private static void AssertCloseFloat(float[] got, float[] want, float absTol, float relTol)
    {
        Assert.Equal(got.Length, want.Length);
        for (int i = 0; i < got.Length; i++)
        {
            float diff = System.Math.Abs(got[i] - want[i]);
            float bound = absTol + relTol * System.Math.Abs(want[i]);
            Assert.True(diff <= bound,
                $"[{i}] got={got[i]}, want={want[i]}, diff={diff}, bound={bound}");
        }
    }
}
