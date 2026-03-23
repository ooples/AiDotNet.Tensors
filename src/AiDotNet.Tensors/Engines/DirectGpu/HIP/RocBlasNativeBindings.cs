using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// P/Invoke bindings for AMD rocBLAS — assembly-tuned BLAS for AMD GPUs.
/// Achieves 60-80% of theoretical peak TFLOPS on RDNA/CDNA architectures.
/// Install AiDotNet.Native.ROCm NuGet package to provide the native binary.
/// </summary>
internal static class RocBlasNativeBindings
{
    private const string RocBlasLibrary = "rocblas";

    private static bool _checked;
    private static bool _available;
    private static readonly object _lock = new();

    public static bool IsAvailable
    {
        get
        {
            if (!_checked)
            {
                lock (_lock)
                {
                    if (!_checked)
                    {
                        try
                        {
#if NET5_0_OR_GREATER
                            if (NativeLibrary.TryLoad(RocBlasLibrary, out var handle))
                            {
                                NativeLibrary.Free(handle);
                                _available = true;
                            }
#else
                            var handle = LoadLibrary(RocBlasLibrary);
                            if (handle != IntPtr.Zero) { FreeLibrary(handle); _available = true; }
#endif
                        }
                        catch { _available = false; }
                        _checked = true;
                    }
                }
            }
            return _available;
        }
    }

#if !NET5_0_OR_GREATER
    [DllImport("kernel32", SetLastError = true)] private static extern IntPtr LoadLibrary(string lpFileName);
    [DllImport("kernel32")] private static extern bool FreeLibrary(IntPtr hModule);
#endif

    // =====================================================================
    // rocBLAS handle management
    // =====================================================================

    [DllImport(RocBlasLibrary, EntryPoint = "rocblas_create_handle")]
    public static extern int CreateHandle(out IntPtr handle);

    [DllImport(RocBlasLibrary, EntryPoint = "rocblas_destroy_handle")]
    public static extern int DestroyHandle(IntPtr handle);

    [DllImport(RocBlasLibrary, EntryPoint = "rocblas_set_stream")]
    public static extern int SetStream(IntPtr handle, IntPtr stream);

    // =====================================================================
    // SGEMM — Single-precision General Matrix Multiplication
    // C = alpha * op(A) * op(B) + beta * C
    // =====================================================================

    [DllImport(RocBlasLibrary, EntryPoint = "rocblas_sgemm")]
    public static extern int Sgemm(
        IntPtr handle,
        int transA,  // rocblas_operation: 111=NoTrans, 112=Trans, 113=ConjTrans
        int transB,
        int M, int N, int K,
        ref float alpha,
        IntPtr A, int lda,
        IntPtr B, int ldb,
        ref float beta,
        IntPtr C, int ldc);

    [DllImport(RocBlasLibrary, EntryPoint = "rocblas_dgemm")]
    public static extern int Dgemm(
        IntPtr handle,
        int transA, int transB,
        int M, int N, int K,
        ref double alpha,
        IntPtr A, int lda,
        IntPtr B, int ldb,
        ref double beta,
        IntPtr C, int ldc);

    [DllImport(RocBlasLibrary, EntryPoint = "rocblas_hgemm")]
    public static extern int Hgemm(
        IntPtr handle,
        int transA, int transB,
        int M, int N, int K,
        ref ushort alpha,  // half
        IntPtr A, int lda,
        IntPtr B, int ldb,
        ref ushort beta,
        IntPtr C, int ldc);

    // =====================================================================
    // Batched GEMM — Multiple GEMMs in single call
    // =====================================================================

    [DllImport(RocBlasLibrary, EntryPoint = "rocblas_sgemm_strided_batched")]
    public static extern int SgemmStridedBatched(
        IntPtr handle,
        int transA, int transB,
        int M, int N, int K,
        ref float alpha,
        IntPtr A, int lda, long strideA,
        IntPtr B, int ldb, long strideB,
        ref float beta,
        IntPtr C, int ldc, long strideC,
        int batchCount);

    // =====================================================================
    // BLAS Level 1 — Vector operations
    // =====================================================================

    [DllImport(RocBlasLibrary, EntryPoint = "rocblas_saxpy")]
    public static extern int Saxpy(IntPtr handle, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy);

    [DllImport(RocBlasLibrary, EntryPoint = "rocblas_sdot")]
    public static extern int Sdot(IntPtr handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr result);

    [DllImport(RocBlasLibrary, EntryPoint = "rocblas_snrm2")]
    public static extern int Snrm2(IntPtr handle, int n, IntPtr x, int incx, IntPtr result);

    [DllImport(RocBlasLibrary, EntryPoint = "rocblas_sscal")]
    public static extern int Sscal(IntPtr handle, int n, ref float alpha, IntPtr x, int incx);

    // =====================================================================
    // BLAS Level 2 — Matrix-vector operations
    // =====================================================================

    [DllImport(RocBlasLibrary, EntryPoint = "rocblas_sgemv")]
    public static extern int Sgemv(
        IntPtr handle,
        int trans,
        int M, int N,
        ref float alpha,
        IntPtr A, int lda,
        IntPtr x, int incx,
        ref float beta,
        IntPtr y, int incy);

    // =====================================================================
    // Constants
    // =====================================================================

    public const int OperationNone = 111;
    public const int OperationTranspose = 112;
    public const int OperationConjugateTranspose = 113;
    public const int StatusSuccess = 0;
}
