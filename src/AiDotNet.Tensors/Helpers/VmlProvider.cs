using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides access to Intel MKL's Vector Math Library (VML) for high-performance
/// vectorized transcendental functions (Exp, Log, Sqrt, etc.).
/// VML uses SVML intrinsics internally and is 2-5x faster than polynomial approximations.
///
/// Uses C# 9 unmanaged function pointers (delegate* unmanaged[Cdecl]) for zero-overhead
/// P/Invoke — no delegate allocation, no marshal overhead, just a direct call instruction.
/// Previous delegate-based approach added ~50ns overhead per call which negated the SVML benefit.
/// </summary>
internal static class VmlProvider
{
    private static bool _initialized;
    private static bool _available;
    private static readonly object InitLock = new();

#if NET5_0_OR_GREATER
    // Unmanaged function pointers — zero marshal overhead (C# 9 feature).
    // These are raw native function pointers, not managed delegates.
    private static unsafe delegate* unmanaged[Cdecl]<int, float*, float*, void> _vsExp;
    private static unsafe delegate* unmanaged[Cdecl]<int, double*, double*, void> _vdExp;
    private static unsafe delegate* unmanaged[Cdecl]<int, float*, float*, void> _vsLn;
    private static unsafe delegate* unmanaged[Cdecl]<int, double*, double*, void> _vdLn;
    private static unsafe delegate* unmanaged[Cdecl]<int, float*, float*, void> _vsTanh;
    private static unsafe delegate* unmanaged[Cdecl]<int, double*, double*, void> _vdTanh;
#endif

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
    /// Zero overhead — uses unmanaged function pointer, not delegate.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryExp(float* input, float* output, int length)
    {
#if NET5_0_OR_GREATER
        if (!EnsureInitialized() || _vsExp == null) return false;
        _vsExp(length, input, output);
        return true;
#else
        return false;
#endif
    }

    /// <summary>
    /// Computes element-wise exp(x) for double using MKL VML.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryExp(double* input, double* output, int length)
    {
#if NET5_0_OR_GREATER
        if (!EnsureInitialized() || _vdExp == null) return false;
        _vdExp(length, input, output);
        return true;
#else
        return false;
#endif
    }

    /// <summary>
    /// Computes element-wise ln(x) using MKL VML.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryLn(float* input, float* output, int length)
    {
#if NET5_0_OR_GREATER
        if (!EnsureInitialized() || _vsLn == null) return false;
        _vsLn(length, input, output);
        return true;
#else
        return false;
#endif
    }

    /// <summary>
    /// Computes element-wise ln(x) for double using MKL VML.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryLn(double* input, double* output, int length)
    {
#if NET5_0_OR_GREATER
        if (!EnsureInitialized() || _vdLn == null) return false;
        _vdLn(length, input, output);
        return true;
#else
        return false;
#endif
    }

    /// <summary>
    /// Computes element-wise tanh(x) using MKL VML.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryTanh(float* input, float* output, int length)
    {
#if NET5_0_OR_GREATER
        if (!EnsureInitialized() || _vsTanh == null) return false;
        _vsTanh(length, input, output);
        return true;
#else
        return false;
#endif
    }

    /// <summary>
    /// Computes element-wise tanh(x) for double using MKL VML.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryTanh(double* input, double* output, int length)
    {
#if NET5_0_OR_GREATER
        if (!EnsureInitialized() || _vdTanh == null) return false;
        _vdTanh(length, input, output);
        return true;
#else
        return false;
#endif
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
            // Try to load MKL runtime library from multiple candidate names
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
            // MKL not available — fall back to polynomial approximations
        }
#endif
        return false;
    }

#if NET5_0_OR_GREATER
    private static unsafe bool TryLoadSymbols(IntPtr handle)
    {
        try
        {
            // Load function pointers as unmanaged[Cdecl] — zero overhead calls.
            // Cast IntPtr directly to function pointer (can't use generics with fn ptrs).
            _vsExp = (delegate* unmanaged[Cdecl]<int, float*, float*, void>)
                TryGetExport(handle, "vsExp", "MKL_vsExp", "VSEXP");
            _vdExp = (delegate* unmanaged[Cdecl]<int, double*, double*, void>)
                TryGetExport(handle, "vdExp", "MKL_vdExp", "VDEXP");
            _vsLn = (delegate* unmanaged[Cdecl]<int, float*, float*, void>)
                TryGetExport(handle, "vsLn", "MKL_vsLn", "VSLN");
            _vdLn = (delegate* unmanaged[Cdecl]<int, double*, double*, void>)
                TryGetExport(handle, "vdLn", "MKL_vdLn", "VDLN");
            _vsTanh = (delegate* unmanaged[Cdecl]<int, float*, float*, void>)
                TryGetExport(handle, "vsTanh", "MKL_vsTanh", "VSTANH");
            _vdTanh = (delegate* unmanaged[Cdecl]<int, double*, double*, void>)
                TryGetExport(handle, "vdTanh", "MKL_vdTanh", "VDTANH");

            // Verify with a tiny test call to catch broken function pointers
            if (_vsExp != null)
            {
                try
                {
                    var testIn = new float[] { 1.0f };
                    var testOut = new float[] { 0f };
                    fixed (float* pIn = testIn)
                    fixed (float* pOut = testOut)
                    {
                        _vsExp(1, pIn, pOut);
                    }
                    if (float.IsNaN(testOut[0]) || float.IsInfinity(testOut[0]) || Math.Abs(testOut[0] - 2.71828f) > 0.01f)
                    {
                        ClearAllPointers();
                    }
                }
                catch
                {
                    ClearAllPointers();
                }
            }

            return _vsExp != null || _vdExp != null || _vsLn != null || _vdLn != null
                || _vsTanh != null || _vdTanh != null;
        }
        catch
        {
            ClearAllPointers();
            return false;
        }
    }

    /// <summary>
    /// Tries multiple symbol names and returns the first found export address, or IntPtr.Zero.
    /// </summary>
    private static IntPtr TryGetExport(IntPtr handle, params string[] names)
    {
        foreach (var name in names)
        {
            if (NativeLibrary.TryGetExport(handle, name, out var ptr))
                return ptr;
        }
        return IntPtr.Zero;
    }

    private static unsafe void ClearAllPointers()
    {
        _vsExp = null;
        _vdExp = null;
        _vsLn = null;
        _vdLn = null;
        _vsTanh = null;
        _vdTanh = null;
    }
#endif
}
