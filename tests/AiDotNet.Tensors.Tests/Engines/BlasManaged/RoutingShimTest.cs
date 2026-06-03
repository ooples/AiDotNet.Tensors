using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-issue F (#374) task F.1: routing shim verification.
///
/// <para>
/// When <see cref="BlasManagedLib.PreferManaged"/> is true,
/// <see cref="BlasProvider.TryGemm"/> and <see cref="BlasProvider.TryGemmEx"/>
/// must route to <see cref="BlasManagedLib.Gemm{T}"/> and return true. This is how
/// the 144 production call sites across the codebase get Sub-A/B/C/D's wins
/// without any caller-side edit.
/// </para>
///
/// <para>
/// Lives in BlasManaged-Stats-Serial collection because PreferManaged is a
/// process-wide static; concurrent tests setting it would race.
/// </para>
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class RoutingShimTest
{
    public RoutingShimTest()
    {
        // Each test starts with default state.
        BlasManagedLib.PreferManaged = false;
    }

    [Fact]
    public void Default_PreferManaged_Is_False()
    {
        Assert.False(BlasManagedLib.PreferManaged);
    }

    [Fact]
    public void TryGemm_With_PreferManaged_True_Returns_True_And_Produces_Correct_Output_FP32()
    {
        const int M = 64, N = 64, K = 64;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var cNative = new float[M * N];
        var cManaged = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Native path baseline (if available). When libopenblas isn't loaded,
        // TryGemm returns false and we skip the comparison.
        BlasManagedLib.PreferManaged = false;
        bool nativeOk = BlasProvider.TryGemm(M, N, K, a, 0, K, b, 0, N, cNative, 0, N);

        // Routed-managed path.
        BlasManagedLib.PreferManaged = true;
        try
        {
            bool managedOk = BlasProvider.TryGemm(M, N, K, a, 0, K, b, 0, N, cManaged, 0, N);
            Assert.True(managedOk, "TryGemm must return true when PreferManaged=true");

            // Output must be non-zero (kernel actually ran).
            bool anyNonZero = false;
            for (int i = 0; i < cManaged.Length; i++) if (cManaged[i] != 0) { anyNonZero = true; break; }
            Assert.True(anyNonZero, "Managed path produced all-zero output");

            // If native ran, verify BOTH paths produce correct GEMM output by
            // comparing each against an FP64 ground-truth computed from the same
            // FP32 inputs. Each FP32 path is "correct" if it's within K·eps_fp32
            // of truth (K=64, eps_fp32≈1.2e-7 → absolute bound ≈ 8e-6 per cell,
            // doubled to 1.6e-5 for slack). Comparing two FP32 paths against each
            // other directly is the wrong assertion: at small-magnitude output
            // cells the relative drift between two correct implementations can
            // exceed the per-implementation error bound (1/cell_magnitude scaling).
            if (nativeOk)
            {
                var cTruth = new double[M * N];
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++)
                    {
                        double sum = 0;
                        for (int kk = 0; kk < K; kk++) sum += (double)a[i * K + kk] * b[kk * N + j];
                        cTruth[i * N + j] = sum;
                    }

                const double maxAbsErr = 1.6e-5;  // 2 · K · eps_fp32 abs slack
                double nativeMaxErr = 0, managedMaxErr = 0;
                for (int i = 0; i < cTruth.Length; i++)
                {
                    nativeMaxErr = Math.Max(nativeMaxErr, Math.Abs(cTruth[i] - cNative[i]));
                    managedMaxErr = Math.Max(managedMaxErr, Math.Abs(cTruth[i] - cManaged[i]));
                }
                Assert.True(nativeMaxErr < maxAbsErr, $"Native path absolute error vs FP64 truth: {nativeMaxErr:G6} (bound {maxAbsErr})");
                Assert.True(managedMaxErr < maxAbsErr, $"Routed managed path absolute error vs FP64 truth: {managedMaxErr:G6} (bound {maxAbsErr})");
            }
        }
        finally
        {
            BlasManagedLib.PreferManaged = false;
        }
    }

    [Fact]
    public void TryGemmEx_With_PreferManaged_True_Handles_Transposed_FP32()
    {
        const int M = 64, N = 32, K = 64;
        var rng = new Random(42);
        var a = new float[K * M];  // transA=true: A is [K, M]
        var b = new float[K * N];
        var cNative = new float[M * N];
        var cManaged = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        BlasManagedLib.PreferManaged = false;
        bool nativeOk = BlasProvider.TryGemmEx(M, N, K, a, 0, M, true, b, 0, N, false, cNative, 0, N);

        BlasManagedLib.PreferManaged = true;
        try
        {
            bool managedOk = BlasProvider.TryGemmEx(M, N, K, a, 0, M, true, b, 0, N, false, cManaged, 0, N);
            Assert.True(managedOk);

            bool anyNonZero = false;
            for (int i = 0; i < cManaged.Length; i++) if (cManaged[i] != 0) { anyNonZero = true; break; }
            Assert.True(anyNonZero);

            if (nativeOk)
            {
                double maxRelDelta = 0;
                for (int i = 0; i < cNative.Length; i++)
                {
                    float n_ = cNative[i], m_ = cManaged[i];
                    if (n_ == 0 && m_ == 0) continue;
                    double rel = Math.Abs(n_ - m_) / Math.Max(Math.Abs(n_), Math.Abs(m_));
                    if (rel > maxRelDelta) maxRelDelta = rel;
                }
                Assert.True(maxRelDelta < 1e-4, $"Routed managed transposed-A drift: {maxRelDelta:G6}");
            }
        }
        finally
        {
            BlasManagedLib.PreferManaged = false;
        }
    }

    [Fact]
    public void TryGemm_DefaultDeterministicMode_RoutesToManaged_AndProducesCorrectOutput()
    {
        // Best-practice default config: deterministic mode (the managed
        // parallel-AND-reproducible kernel — what PyTorch's deterministic mode can't
        // offer). BlasProvider.ShouldRouteManaged routes deterministic GEMM to the
        // managed kernel regardless of native availability, so PreferManaged=false
        // here STILL goes managed — and TryGemm returns true even on a no-native
        // runner. This replaces an obsolete pre-Phase-1 test that asserted
        // "PreferManaged=false → native" (no longer the contract). The native path
        // for the NON-deterministic mode is covered by
        // NativeBlasNonDeterministicConcurrencyTests; we do not disable best-practice
        // modes here to chase it.
        bool? beforeThreadDet = BlasProvider.GetThreadLocalDeterministicMode();
        if (beforeThreadDet is not null) BlasProvider.SetThreadLocalDeterministicMode(null);
        bool beforeDet = BlasProvider.IsDeterministicMode;
        try
        {
            BlasProvider.SetDeterministicMode(true); // ensure the default best-practice mode regardless of test order
            BlasManagedLib.PreferManaged = false;

            const int M = 32, N = 32, K = 32;
            var rng = new Random(42);
            var a = new float[M * K];
            var b = new float[K * N];
            var c = new float[M * N];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            bool ok = BlasProvider.TryGemm(M, N, K, a, 0, K, b, 0, N, c, 0, N);
            // Deterministic mode routes to the managed kernel, which always succeeds —
            // independent of whether native BLAS is loaded.
            Assert.True(ok, "Deterministic-mode TryGemm must route to managed and return true");

            // And the result is a correct GEMM (within FP32 bound of FP64 ground truth).
            const double maxAbsErr = 1.6e-5; // 2 · K · eps_fp32
            double maxErr = 0;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                {
                    double sum = 0;
                    for (int kk = 0; kk < K; kk++) sum += (double)a[i * K + kk] * b[kk * N + j];
                    maxErr = Math.Max(maxErr, Math.Abs(sum - c[i * N + j]));
                }
            Assert.True(maxErr < maxAbsErr, $"Managed deterministic GEMM error vs FP64 truth: {maxErr:G6} (bound {maxAbsErr})");
        }
        finally
        {
            BlasProvider.SetDeterministicMode(beforeDet);
            BlasProvider.SetThreadLocalDeterministicMode(beforeThreadDet);
            BlasManagedLib.PreferManaged = false;
        }
    }

    // #374 acceptance (the managed-only routed path): the per-shape tests above cover a
    // couple of hand-picked sizes; this sweeps the WHOLE production shape catalog (the
    // 50-80 instrumented + workload shapes the shim must serve). With PreferManaged=true,
    // EVERY shape must route through the managed kernel (TryGemmEx returns true) and
    // produce a numerically correct GEMM — catching a routing/packing/transpose bug on
    // any production shape, not just the canned ones. Correctness is checked against an
    // FP64 ground truth for shapes below a work cap (the cap keeps the O(M·N·K) reference
    // fast); larger shapes assert routed + non-zero only.
    [Fact]
    public void PreferManaged_FullCatalog_RoutesToManaged_AndIsNumericallyCorrect()
    {
        const long TruthCap = 4_000_000; // M·N·K above this → routed+non-zero only
        Assert.NotEmpty(ShapeCatalog.All);
        BlasManagedLib.PreferManaged = false;
        try
        {
            BlasManagedLib.PreferManaged = true;
            foreach (var s in ShapeCatalog.All)
            {
                if (s.Dtype == DType.Single) VerifyCatalogShapeF(s, TruthCap);
                else VerifyCatalogShapeD(s, TruthCap);
            }
        }
        finally { BlasManagedLib.PreferManaged = false; }
    }

    private static void VerifyCatalogShapeF(Shape s, long truthCap)
    {
        int m = s.M, n = s.N, k = s.K;
        bool tA = s.TransA, tB = s.TransB;
        int lda = tA ? m : k, ldb = tB ? k : n, ldc = n;
        var rng = new Random(unchecked(s.Name.GetHashCode()) ^ 0x5151);
        var a = new float[m * k]; for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        var b = new float[k * n]; for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var c = new float[m * n];

        bool ok = BlasProvider.TryGemmEx(m, n, k, a, 0, lda, tA, b, 0, ldb, tB, c, 0, ldc);
        string tag = $"[{s.Name} {m}x{n}x{k} tA={tA} tB={tB} FP32]";
        Assert.True(ok, $"{tag} PreferManaged TryGemmEx must route to managed and return true");
        Assert.True(AnyNonZero(c), $"{tag} managed output is all-zero");

        if ((long)m * n * k > truthCap) return;
        double maxErr = 0, maxAbs = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int kk = 0; kk < k; kk++)
                {
                    double av = tA ? a[kk * lda + i] : a[i * lda + kk];
                    double bv = tB ? b[j * ldb + kk] : b[kk * ldb + j];
                    sum += av * bv;
                }
                maxAbs = Math.Max(maxAbs, Math.Abs(sum));
                maxErr = Math.Max(maxErr, Math.Abs(sum - c[i * n + j]));
            }
        double bound = 1e-3 * Math.Max(1.0, maxAbs);  // FP32 accumulation vs FP64 truth, magnitude-relative
        Assert.True(maxErr <= bound, $"{tag} managed GEMM err vs FP64 truth {maxErr:G6} > bound {bound:G6}");
    }

    private static void VerifyCatalogShapeD(Shape s, long truthCap)
    {
        int m = s.M, n = s.N, k = s.K;
        bool tA = s.TransA, tB = s.TransB;
        int lda = tA ? m : k, ldb = tB ? k : n, ldc = n;
        var rng = new Random(unchecked(s.Name.GetHashCode()) ^ 0x7373);
        var a = new double[m * k]; for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        var b = new double[k * n]; for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        var c = new double[m * n];

        bool ok = BlasProvider.TryGemmEx(m, n, k, a, 0, lda, tA, b, 0, ldb, tB, c, 0, ldc);
        string tag = $"[{s.Name} {m}x{n}x{k} tA={tA} tB={tB} FP64]";
        Assert.True(ok, $"{tag} PreferManaged TryGemmEx must route to managed and return true");
        Assert.True(AnyNonZero(c), $"{tag} managed output is all-zero");

        if ((long)m * n * k > truthCap) return;
        double maxErr = 0, maxAbs = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int kk = 0; kk < k; kk++)
                {
                    double av = tA ? a[kk * lda + i] : a[i * lda + kk];
                    double bv = tB ? b[j * ldb + kk] : b[kk * ldb + j];
                    sum += av * bv;
                }
                maxAbs = Math.Max(maxAbs, Math.Abs(sum));
                maxErr = Math.Max(maxErr, Math.Abs(sum - c[i * n + j]));
            }
        double bound = 1e-9 * Math.Max(1.0, maxAbs);  // FP64 accumulation; reduction-order differences only
        Assert.True(maxErr <= bound, $"{tag} managed GEMM err vs FP64 truth {maxErr:G6} > bound {bound:G6}");
    }

    private static bool AnyNonZero(float[] c)
    {
        for (int i = 0; i < c.Length; i++) if (c[i] != 0) return true;
        return false;
    }

    private static bool AnyNonZero(double[] c)
    {
        for (int i = 0; i < c.Length; i++) if (c[i] != 0) return true;
        return false;
    }
}
