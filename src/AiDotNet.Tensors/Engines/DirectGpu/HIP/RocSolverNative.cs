// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// P/Invoke bindings into <c>librocsolver</c>. Mirrors the cuSOLVER
/// surface we use: Cholesky (<c>potrf</c>), QR (<c>geqrf</c> +
/// <c>orgqr</c>), SVD (<c>gesvd</c>), symmetric eigen (<c>syevd</c>),
/// LU (<c>getrf</c> + <c>getrs</c>). rocSOLVER's API uses the same
/// LAPACK-style entry-point names as cuSOLVER, so we keep the binding
/// shapes parallel for symmetry.
/// </summary>
internal static class RocSolverNative
{
    private const string Lib = "rocsolver";

#if NET5_0_OR_GREATER
    static RocSolverNative()
    {
        AiDotNet.Tensors.Engines.NativeLibraryResolverRegistry.Register(ResolveLib);
    }

    private static IntPtr ResolveLib(string name, System.Reflection.Assembly asm, DllImportSearchPath? path)
    {
        if (name != Lib) return IntPtr.Zero;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            if (NativeLibrary.TryLoad("librocsolver.so.0", asm, path, out var h)) return h;
            if (NativeLibrary.TryLoad("librocsolver.so", asm, path, out h)) return h;
        }
        return IntPtr.Zero;
    }
#endif

    public enum Status
    {
        Success = 0,
        InvalidHandle = 1,
        NotImplemented = 2,
        InvalidPointer = 3,
        InvalidSize = 4,
        MemoryError = 5,
        InternalError = 6,
        InvalidValue = 7,
        ArchMismatch = 8,
        ZeroPivot = 9,
        NotInitialized = 10,
    }

    public enum FillMode { Lower = 121, Upper = 122 }
    public enum EigMode { NoVector = 211, Vector = 212 }
    public enum SvdAction { ComputeAll = 221, ComputeNoVectors = 222, ComputeSingularVectors = 223 }

    [DllImport(Lib)] public static extern Status rocblas_create_handle(out IntPtr handle);
    [DllImport(Lib)] public static extern Status rocblas_destroy_handle(IntPtr handle);

    // Cholesky — single precision. Real rocSOLVER signature; double-precision (<c>spotrf</c> ↔ <c>dpotrf</c>) follows the same shape.
    [DllImport(Lib)]
    public static extern Status rocsolver_spotrf(IntPtr handle, FillMode uplo, int n, IntPtr A, int lda, IntPtr devInfo);
    [DllImport(Lib)]
    public static extern Status rocsolver_dpotrf(IntPtr handle, FillMode uplo, int n, IntPtr A, int lda, IntPtr devInfo);

    // QR.
    [DllImport(Lib)]
    public static extern Status rocsolver_sgeqrf(IntPtr handle, int m, int n, IntPtr A, int lda, IntPtr ipiv);
    [DllImport(Lib)]
    public static extern Status rocsolver_dgeqrf(IntPtr handle, int m, int n, IntPtr A, int lda, IntPtr ipiv);
    [DllImport(Lib)]
    public static extern Status rocsolver_sorgqr(IntPtr handle, int m, int n, int k, IntPtr A, int lda, IntPtr ipiv);
    [DllImport(Lib)]
    public static extern Status rocsolver_dorgqr(IntPtr handle, int m, int n, int k, IntPtr A, int lda, IntPtr ipiv);

    // SVD.
    [DllImport(Lib)]
    public static extern Status rocsolver_sgesvd(IntPtr handle, SvdAction left, SvdAction right, int m, int n,
        IntPtr A, int lda, IntPtr S, IntPtr U, int ldu, IntPtr V, int ldv,
        IntPtr E, int fastAlg, IntPtr devInfo);

    // Symmetric eigen.
    [DllImport(Lib)]
    public static extern Status rocsolver_ssyevd(IntPtr handle, EigMode jobz, FillMode uplo, int n,
        IntPtr A, int lda, IntPtr W, IntPtr E, IntPtr devInfo);

    // LU.
    [DllImport(Lib)]
    public static extern Status rocsolver_sgetrf(IntPtr handle, int m, int n, IntPtr A, int lda, IntPtr ipiv, IntPtr devInfo);
    [DllImport(Lib)]
    public static extern Status rocsolver_sgetrs(IntPtr handle, int trans, int n, int nrhs,
        IntPtr A, int lda, IntPtr ipiv, IntPtr B, int ldb);

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            var s = rocblas_create_handle(out var h);
            if (s == Status.Success) { rocblas_destroy_handle(h); return true; }
            return false;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
