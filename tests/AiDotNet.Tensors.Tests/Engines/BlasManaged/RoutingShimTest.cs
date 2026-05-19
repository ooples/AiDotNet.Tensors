using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
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

            // If native ran, compare. Tolerance: relaxed because BlasManaged uses FMA
            // (in Fast mode default) which differs from non-FMA scalar by 1 ULP.
            // Actually, the default Mode is Deterministic — should be bit-exact with
            // OpenBLAS on non-FMA paths. But OpenBLAS may use FMA too, in which case
            // results may differ by ULPs. Use relative tolerance.
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
                Assert.True(maxRelDelta < 1e-4, $"Routed managed path drift vs native: {maxRelDelta:G6}");
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
    public void TryGemm_With_PreferManaged_False_Goes_Through_Native_Path()
    {
        // This is the default behavior; just confirm TryGemm still works when
        // the toggle is off (no regression to the existing native path).
        const int M = 32, N = 32, K = 32;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        BlasManagedLib.PreferManaged = false;
        bool ok = BlasProvider.TryGemm(M, N, K, a, 0, K, b, 0, N, c, 0, N);
        // If native is available, ok=true; if not, ok=false. Both are acceptable —
        // we're just confirming the default path is unchanged.
        Assert.Equal(BlasProvider.IsAvailable, ok);
    }
}
