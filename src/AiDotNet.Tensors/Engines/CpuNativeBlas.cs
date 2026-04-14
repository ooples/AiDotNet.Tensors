namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Native CPU BLAS loader — <b>disabled in the supply-chain-independence build</b>.
///
/// <para>
/// Historically this was a second native BLAS loader (sibling to
/// <see cref="Helpers.BlasProvider"/>) that dynamically loaded OpenBLAS or
/// MKL via <c>System.Runtime.InteropServices.NativeLibrary.TryLoad</c> and
/// exposed <c>TryGemm</c> for float/double matmul with offsets + leading
/// dimensions. After <c>feat/finish-mkl-replacement</c>, every entry point
/// returns <c>false</c> immediately so callers fall through to
/// <see cref="Simd.SimdGemm"/>'s AVX2 blocked kernel.
/// </para>
/// <para>
/// This stub replaces ~640 lines of P/Invoke + env-var parsing + library-
/// search plumbing that became unreachable after the disable. See git
/// history for the original implementation.
/// </para>
/// </summary>
internal static class CpuNativeBlas
{
    /// <summary>Always false — external CPU BLAS is disabled.</summary>
    internal static bool IsAvailable => false;

    internal static bool TryGemm(float[] a, float[] b, float[] c, int m, int n, int k) => false;

    internal static bool TryGemm(double[] a, double[] b, double[] c, int m, int n, int k) => false;

    internal static bool TryGemm(
        float[] a, int aOffset,
        float[] b, int bOffset,
        float[] c, int cOffset,
        int m, int n, int k,
        int lda, int ldb, int ldc) => false;

    internal static bool TryGemm(
        double[] a, int aOffset,
        double[] b, int bOffset,
        double[] c, int cOffset,
        int m, int n, int k,
        int lda, int ldb, int ldc) => false;
}
