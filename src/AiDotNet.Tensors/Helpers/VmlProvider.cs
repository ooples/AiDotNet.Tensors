using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides access to Intel MKL's Vector Math Library (VML) for high-performance
/// vectorized transcendental functions (Exp, Log, Sqrt, etc.).
/// VML uses SVML intrinsics internally and is 2-5x faster than our polynomial
/// approximations for these operations.
/// </summary>
internal static class VmlProvider
{
    private static bool _initialized;
    private static bool _available;
    private static readonly object InitLock = new();

    // VML function pointers
    private static VsExpDelegate? _vsExp;
    private static VdExpDelegate? _vdExp;
    private static VsLnDelegate? _vsLn;
    private static VdLnDelegate? _vdLn;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private unsafe delegate void VsExpDelegate(int n, float* a, float* y);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private unsafe delegate void VdExpDelegate(int n, double* a, double* y);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private unsafe delegate void VsLnDelegate(int n, float* a, float* y);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private unsafe delegate void VdLnDelegate(int n, double* a, double* y);

    /// <summary>Whether MKL VML functions are available.</summary>
    public static bool IsAvailable
    {
        get
        {
            EnsureInitialized();
            return _available;
        }
    }

    /// <summary>
    /// Computes element-wise exp(x) using MKL VML. Returns false if VML not available.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryExp(float* input, float* output, int length)
    {
        if (!EnsureInitialized() || _vsExp == null) return false;
        _vsExp(length, input, output);
        return true;
    }

    /// <summary>
    /// Computes element-wise exp(x) for double using MKL VML.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryExp(double* input, double* output, int length)
    {
        if (!EnsureInitialized() || _vdExp == null) return false;
        _vdExp(length, input, output);
        return true;
    }

    /// <summary>
    /// Computes element-wise ln(x) using MKL VML.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryLn(float* input, float* output, int length)
    {
        if (!EnsureInitialized() || _vsLn == null) return false;
        _vsLn(length, input, output);
        return true;
    }

    /// <summary>
    /// Computes element-wise ln(x) for double using MKL VML.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryLn(double* input, double* output, int length)
    {
        if (!EnsureInitialized() || _vdLn == null) return false;
        _vdLn(length, input, output);
        return true;
    }

    private static bool EnsureInitialized()
    {
        if (_initialized) return _available;

        lock (InitLock)
        {
            if (_initialized) return _available;
            _available = TryLoadVml();
            _initialized = true;
            return _available;
        }
    }

    private static bool TryLoadVml()
    {
#if NET5_0_OR_GREATER
        try
        {
            // MKL.NET already loads the MKL native library.
            // Try to find VML symbols in already-loaded MKL modules.
            // The symbols are in mkl_rt.dll (Windows) or libmkl_rt.so (Linux)
            var candidates = new[]
            {
                "mkl_rt", "mkl_rt.2", "libmkl_rt", "libmkl_rt.so", "libmkl_rt.so.2",
                "mkl_sequential", "libmkl_sequential"
            };

            foreach (var lib in candidates)
            {
                if (NativeLibrary.TryLoad(lib, out var handle))
                {
                    if (TryLoadSymbols(handle))
                        return true;
                }
            }

            // Try loading from the MKL.NET package's native directory
            var mklDir = Path.GetDirectoryName(typeof(MKLNET.Blas).Assembly.Location);
            if (mklDir != null)
            {
                foreach (var dllPath in Directory.GetFiles(mklDir, "mkl_rt*"))
                {
                    if (NativeLibrary.TryLoad(dllPath, out var handle))
                    {
                        if (TryLoadSymbols(handle))
                            return true;
                    }
                }
            }
        }
        catch
        {
            // VML not available
        }
#endif
        return false;
    }

#if NET5_0_OR_GREATER
    private static bool TryLoadSymbols(IntPtr handle)
    {
        try
        {
            if (NativeLibrary.TryGetExport(handle, "vsExp", out var vsExpPtr))
                _vsExp = Marshal.GetDelegateForFunctionPointer<VsExpDelegate>(vsExpPtr);

            if (NativeLibrary.TryGetExport(handle, "vdExp", out var vdExpPtr))
                _vdExp = Marshal.GetDelegateForFunctionPointer<VdExpDelegate>(vdExpPtr);

            if (NativeLibrary.TryGetExport(handle, "vsLn", out var vsLnPtr))
                _vsLn = Marshal.GetDelegateForFunctionPointer<VsLnDelegate>(vsLnPtr);

            if (NativeLibrary.TryGetExport(handle, "vdLn", out var vdLnPtr))
                _vdLn = Marshal.GetDelegateForFunctionPointer<VdLnDelegate>(vdLnPtr);

            return _vsExp != null || _vdExp != null;
        }
        catch
        {
            return false;
        }
    }
#endif
}
