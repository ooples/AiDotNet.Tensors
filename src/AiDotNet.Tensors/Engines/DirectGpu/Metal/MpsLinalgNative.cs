// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Bindings against the MPS dense linear-algebra kernels:
/// <c>MPSMatrixDecompositionLU</c>, <c>MPSMatrixDecompositionCholesky</c>,
/// <c>MPSMatrixSolveLU</c>, <c>MPSMatrixSolveCholesky</c>, and the
/// SVD / eigendecomposition kernels via the Accelerate framework's
/// LAPACK interop. Same MPSMatrix descriptor pattern as the existing
/// <see cref="MpsSparseBackend"/>; the high-level wrapper allocates
/// MTLBuffer-backed MPSMatrix objects, schedules the kernel, and reads
/// the result back into a host <see cref="LinearAlgebra.Tensor{T}"/>.
/// </summary>
internal static class MpsLinalgNative
{
    private const string LibAccelerate = "/System/Library/Frameworks/Accelerate.framework/Accelerate";
    private const string LibMps = "/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders";
    private const string LibObjc = "/usr/lib/libobjc.A.dylib";

    [DllImport(LibObjc, EntryPoint = "objc_getClass")]
    private static extern IntPtr objc_getClass([MarshalAs(UnmanagedType.LPStr)] string name);

    /// <summary>True when running on macOS and the MPS / Accelerate
    /// frameworks are loadable. The dense-decomposition kernel classes
    /// (<c>MPSMatrixDecompositionLU</c>, <c>MPSMatrixDecompositionCholesky</c>)
    /// must register on framework load for IsAvailable to flip true.</summary>
    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) return false;
        try
        {
#if NET5_0_OR_GREATER
            if (!NativeLibrary.TryLoad(LibMps, out _)) return false;
            if (!NativeLibrary.TryLoad(LibAccelerate, out _)) return false;
#endif
            return objc_getClass("MPSMatrixDecompositionLU") != IntPtr.Zero
                && objc_getClass("MPSMatrixDecompositionCholesky") != IntPtr.Zero;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
