// Copyright (c) AiDotNet. All rights reserved.
// hipBLAS native bindings for HIP GEMM acceleration.
using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

internal static class HipBlasNative
{
    private const string HipBlasLibrary = "hipblas";
    private static volatile bool _isAvailable;
    private static volatile bool _checkedAvailability;
    private static readonly object AvailabilityLock = new object();

    internal enum HipBlasStatus
    {
        Success = 0,
        NotInitialized = 1,
        AllocFailed = 2,
        InvalidValue = 3,
        ArchMismatch = 4,
        MappingError = 5,
        ExecutionFailed = 6,
        InternalError = 7,
        NotSupported = 8
    }

    internal enum HipBlasOperation
    {
        None = 111,
        Transpose = 112,
        ConjugateTranspose = 113
    }

    /// <summary>
    /// hipBLAS element data types (legacy <c>hipblasDatatype_t</c>). Values match
    /// the hipBLAS / rocBLAS ABI (hipblas.h): real FP16 = 150, real FP32 = 151,
    /// real FP64 = 152. Used by <see cref="hipblasGemmEx"/> to describe the FP16
    /// inputs and FP32 accumulator/output of the mixed-precision GEMM.
    /// </summary>
    internal enum HipBlasDatatype
    {
        R_16F = 150,  // 16-bit real (half)
        R_32F = 151,  // 32-bit real (float)
        R_64F = 152,  // 64-bit real (double)
    }

    /// <summary>
    /// hipBLAS GEMM algorithm selector (<c>hipblasGemmAlgo_t</c>).
    /// <c>HIPBLAS_GEMM_DEFAULT = 160</c> lets the library pick — the cuBLAS
    /// <c>CUBLAS_GEMM_DEFAULT</c> equivalent.
    /// </summary>
    internal enum HipBlasGemmAlgo
    {
        Default = 160,
    }

    internal static bool IsAvailable
    {
        get
        {
            if (_checkedAvailability)
            {
                return _isAvailable;
            }

            lock (AvailabilityLock)
            {
                if (_checkedAvailability)
                {
                    return _isAvailable;
                }

                _isAvailable = TryLoadLibrary();
                _checkedAvailability = true;
                return _isAvailable;
            }
        }
    }

    [DllImport(HipBlasLibrary, EntryPoint = "hipblasCreate", CallingConvention = CallingConvention.Cdecl)]
    internal static extern HipBlasStatus hipblasCreate(ref IntPtr handle); // lgtm[cs/unmanaged-code] HIP BLAS requires native bindings.

    [DllImport(HipBlasLibrary, EntryPoint = "hipblasDestroy", CallingConvention = CallingConvention.Cdecl)]
    internal static extern HipBlasStatus hipblasDestroy(IntPtr handle); // lgtm[cs/unmanaged-code] HIP BLAS requires native bindings.

    [DllImport(HipBlasLibrary, EntryPoint = "hipblasSetStream", CallingConvention = CallingConvention.Cdecl)]
    internal static extern HipBlasStatus hipblasSetStream(IntPtr handle, IntPtr stream); // lgtm[cs/unmanaged-code] HIP BLAS requires native bindings.

    [DllImport(HipBlasLibrary, EntryPoint = "hipblasSgemm", CallingConvention = CallingConvention.Cdecl)]
    internal static extern HipBlasStatus hipblasSgemm(
        IntPtr handle,
        HipBlasOperation transA,
        HipBlasOperation transB,
        int m,
        int n,
        int k,
        ref float alpha,
        IntPtr A,
        int lda,
        IntPtr B,
        int ldb,
        ref float beta,
        IntPtr C,
        int ldc); // lgtm[cs/unmanaged-code] HIP BLAS requires native bindings.

    /// <summary>
    /// FP16 GEMM: C = alpha·A·B + beta·C, all matrices half-precision
    /// (<c>hipblasHalf</c> = uint16). alpha/beta are passed as their FP16 bit
    /// patterns (1.0 = 0x3C00, 0.0 = 0x0000). Equivalent to <c>cublasHgemm</c>.
    /// </summary>
    [DllImport(HipBlasLibrary, EntryPoint = "hipblasHgemm", CallingConvention = CallingConvention.Cdecl)]
    internal static extern HipBlasStatus hipblasHgemm(
        IntPtr handle,
        HipBlasOperation transA,
        HipBlasOperation transB,
        int m,
        int n,
        int k,
        ref ushort alpha,
        IntPtr A,
        int lda,
        IntPtr B,
        int ldb,
        ref ushort beta,
        IntPtr C,
        int ldc); // lgtm[cs/unmanaged-code] HIP BLAS requires native bindings.

    /// <summary>
    /// Mixed-precision GEMM: FP16 inputs, FP32 accumulator + output. This is the
    /// cuBLAS-compatible <c>hipblasGemmEx</c> signature (no separate D matrix,
    /// unlike raw <c>rocblas_gemm_ex</c>): alpha/beta are FP32 scalars passed by
    /// pointer (compute type = FP32). Maps to <c>cublasGemmEx(CUDA_R_16F in,
    /// CUBLAS_COMPUTE_32F)</c> — the standard AMP forward-pass matmul.
    /// </summary>
    [DllImport(HipBlasLibrary, EntryPoint = "hipblasGemmEx", CallingConvention = CallingConvention.Cdecl)]
    internal static extern HipBlasStatus hipblasGemmEx(
        IntPtr handle,
        HipBlasOperation transA,
        HipBlasOperation transB,
        int m,
        int n,
        int k,
        ref float alpha,            // const void* (FP32 scalar, compute type)
        IntPtr A,
        HipBlasDatatype aType,
        int lda,
        IntPtr B,
        HipBlasDatatype bType,
        int ldb,
        ref float beta,             // const void* (FP32 scalar)
        IntPtr C,
        HipBlasDatatype cType,
        int ldc,
        HipBlasDatatype computeType,
        HipBlasGemmAlgo algo); // lgtm[cs/unmanaged-code] HIP BLAS requires native bindings.

    private static bool TryLoadLibrary()
    {
        string[] candidates =
        {
            HipBlasLibrary,
            "hipblas.dll",
            "libhipblas.so",
            "libhipblas.dylib"
        };

        return candidates.Where(CanLoadLibrary).Any();
    }

    private static bool CanLoadLibrary(string name)
    {
#if NETFRAMEWORK
        var handle = LoadLibrary(name);
        if (handle == IntPtr.Zero)
        {
            return false;
        }

        FreeLibrary(handle);
        return true;
#else
        if (!NativeLibrary.TryLoad(name, out var handle))
        {
            return false;
        }

        NativeLibrary.Free(handle);
        return true;
#endif
    }

#if NETFRAMEWORK
    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern IntPtr LoadLibrary(string lpFileName);

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool FreeLibrary(IntPtr hModule);
#endif
}
