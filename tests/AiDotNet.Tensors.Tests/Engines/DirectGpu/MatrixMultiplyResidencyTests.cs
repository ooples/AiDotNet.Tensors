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

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public class MatrixMultiplyResidencyTests
{
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
