using System;
using System.Threading.Tasks;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// AVX-512 SGEMM microkernel (FP32). Target: Intel Skylake-X, Ice Lake,
/// Sapphire Rapids, Zen 4 / Zen 5. Uses 16-wide <see cref="Vector512"/> FMA
/// for ~2× throughput vs the Vector256 AVX2 path on zmm-capable CPUs.
///
/// <para>B1 scope: feature gate + entry point. The actual blocked kernel
/// (B2 / B3) plugs in via <see cref="SgemmBlocked"/> once the microkernel
/// and panel-packing routines are in. Until then the entry point falls
/// through to the AVX2 path so the feature gate is a no-op — no behaviour
/// change, ready for the kernel to land behind it.</para>
/// </summary>
internal static class Avx512Sgemm
{
    /// <summary>
    /// True if the AVX-512F path is callable: compiled on .NET 8+, runtime
    /// reports AVX-512F support, and the feature isn't gated off. Used as
    /// the single gate by every upstream dispatcher.
    /// </summary>
    public static bool CanUse
    {
        get
        {
#if NET8_0_OR_GREATER
            return CpuFeatures.HasAVX512F;
#else
            return false;
#endif
        }
    }

    /// <summary>
    /// Entry point for the AVX-512 blocked SGEMM. B1 ships the dispatch
    /// hook and a trivial fallback to the AVX2 <see cref="SimdGemm"/> path.
    /// B2 replaces the body with the real 32×16 microkernel; B3 adds the
    /// BLIS 5-loop driver. Keeping the signature stable means the feature
    /// gate works without touching callers once the real kernel ships.
    /// </summary>
    public static void SgemmBlocked(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n,
        bool allowParallel)
    {
        // B1: route through the existing AVX2 implementation. Parity pass
        // trivially holds because the output bytes are identical.
        SimdGemm.SgemmAddInternal(a, lda, transA, b, ldb, transB, c, m, k, n,
            allowParallel: allowParallel, clearedOutput: true);
    }
}
