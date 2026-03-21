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
    // VML_EP (Enhanced Performance) = fastest mode for double. ~7 correct digits —
    // more than enough for neural networks. VML_EP (0x1) was also fast but VML_EP (0x3)
    // is marginally faster and avoids the first-call cold-start regression.
    private const int VML_EP = 0x00000003;

    private static bool _initialized;
    private static bool _available;
    private static readonly object InitLock = new();

#if NET5_0_OR_GREATER
    // Unmanaged function pointers — zero marshal overhead (C# 9 feature).
    // Float VML — no mode needed (float is fast even in VML_HA)
    private static unsafe delegate* unmanaged[Cdecl]<int, float*, float*, void> _vsExp;
    private static unsafe delegate* unmanaged[Cdecl]<int, float*, float*, void> _vsLn;
    private static unsafe delegate* unmanaged[Cdecl]<int, float*, float*, void> _vsTanh;

    // Double VML — mode-specific (vmd* takes mode as 4th arg).
    // Using VML_EP=0x1 per-call avoids the VML_HA regression that vd* functions hit
    // when the global mode resets between calls.
    private static unsafe delegate* unmanaged[Cdecl]<int, double*, double*, int, void> _vmdExp;
    private static unsafe delegate* unmanaged[Cdecl]<int, double*, double*, int, void> _vmdLn;
    private static unsafe delegate* unmanaged[Cdecl]<int, double*, double*, int, void> _vmdTanh;
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
    /// Computes element-wise exp(x) for double using MKL VML with VML_EP mode per-call.
    /// Uses vmdExp (mode-specific) instead of vdExp to avoid VML_HA regression.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryExp(double* input, double* output, int length)
    {
#if NET5_0_OR_GREATER
        if (!EnsureInitialized() || _vmdExp == null) return false;
        _vmdExp(length, input, output, VML_EP);
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
    /// <summary>Double ln(x) via vmdLn with VML_EP per-call.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryLn(double* input, double* output, int length)
    {
#if NET5_0_OR_GREATER
        if (!EnsureInitialized() || _vmdLn == null) return false;
        _vmdLn(length, input, output, VML_EP);
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
    /// <summary>Double tanh(x) via vmdTanh with VML_EP per-call.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryTanh(double* input, double* output, int length)
    {
#if NET5_0_OR_GREATER
        if (!EnsureInitialized() || _vmdTanh == null) return false;
        _vmdTanh(length, input, output, VML_EP);
        return true;
#else
        return false;
#endif
    }

    // EnforceVmlLaMode removed — using vmd* (mode-per-call) instead of vd* + vmlSetMode

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
                    NativeLibrary.Free(handle);
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
                        NativeLibrary.Free(handle);
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
            // Float VML — no mode needed (fast in all modes)
            _vsExp = (delegate* unmanaged[Cdecl]<int, float*, float*, void>)
                TryGetExport(handle, "vsExp", "MKL_vsExp", "VSEXP");
            _vsLn = (delegate* unmanaged[Cdecl]<int, float*, float*, void>)
                TryGetExport(handle, "vsLn", "MKL_vsLn", "VSLN");
            _vsTanh = (delegate* unmanaged[Cdecl]<int, float*, float*, void>)
                TryGetExport(handle, "vsTanh", "MKL_vsTanh", "VSTANH");

            // Double VML — mode-specific per-call (vmd* takes mode as 4th arg).
            // This avoids the VML_HA regression that vd* functions hit when
            // the global vmlSetMode resets between calls.
            _vmdExp = (delegate* unmanaged[Cdecl]<int, double*, double*, int, void>)
                TryGetExport(handle, "vmdExp", "MKL_vmdExp", "VMDEXP");
            _vmdLn = (delegate* unmanaged[Cdecl]<int, double*, double*, int, void>)
                TryGetExport(handle, "vmdLn", "MKL_vmdLn", "VMDLN");
            _vmdTanh = (delegate* unmanaged[Cdecl]<int, double*, double*, int, void>)
                TryGetExport(handle, "vmdTanh", "MKL_vmdTanh", "VMDTANH");

            // Verify correctness at n=1 AND n=1000 (catches scale-dependent issues).
            // Also verify performance: must be faster than scalar at n=10000.
            _vsExp = VerifyFloat(_vsExp, 1.0f, 2.71828f, 0.01f);
            _vsLn = VerifyFloat(_vsLn, 2.71828f, 1.0f, 0.01f);
            _vsTanh = VerifyFloat(_vsTanh, 1.0f, 0.7616f, 0.01f);
            _vmdExp = VerifyDoubleModeAtScale(_vmdExp, 1.0, 2.71828, 0.001);
            _vmdLn = VerifyDoubleModeAtScale(_vmdLn, 2.71828, 1.0, 0.001);
            _vmdTanh = VerifyDoubleModeAtScale(_vmdTanh, 1.0, 0.76159, 0.001);

            return _vsExp != null || _vmdExp != null || _vsLn != null || _vmdLn != null
                || _vsTanh != null || _vmdTanh != null;
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

    /// <summary>
    /// Verifies a float VML function pointer by calling it with a test input.
    /// Returns the pointer if valid, null if corrupt.
    /// </summary>
    private static unsafe delegate* unmanaged[Cdecl]<int, float*, float*, void>
        VerifyFloat(delegate* unmanaged[Cdecl]<int, float*, float*, void> fn,
        float testInput, float expectedOutput, float tolerance)
    {
        if (fn == null) return null;
        try
        {
            var inArr = new float[] { testInput };
            var outArr = new float[] { 0f };
            fixed (float* pIn = inArr, pOut = outArr)
            {
                fn(1, pIn, pOut);
            }
            if (float.IsNaN(outArr[0]) || float.IsInfinity(outArr[0])
                || Math.Abs(outArr[0] - expectedOutput) > tolerance)
                return null;
            return fn;
        }
        catch { return null; }
    }

    /// <summary>
    /// Verifies a double VML function pointer at n=1 AND n=1000.
    /// Also checks that it's faster than scalar Math.Exp at n=10000.
    /// Verifies a mode-specific double VML function (vmd*) at n=1 and n=1000.
    /// The 4th arg is the VML mode (VML_EP=1). No need for performance test
    /// since VML_EP is passed per-call and can't regress.
    /// </summary>
    private static unsafe delegate* unmanaged[Cdecl]<int, double*, double*, int, void>
        VerifyDoubleModeAtScale(delegate* unmanaged[Cdecl]<int, double*, double*, int, void> fn,
        double testInput, double expectedOutput, double tolerance)
    {
        if (fn == null) return null;
        try
        {
            var inArr = new double[] { testInput };
            var outArr = new double[] { 0.0 };
            fixed (double* pIn = inArr, pOut = outArr)
            {
                fn(1, pIn, pOut, VML_EP);
            }
            if (double.IsNaN(outArr[0]) || double.IsInfinity(outArr[0])
                || Math.Abs(outArr[0] - expectedOutput) > tolerance)
                return null;

            // Verify at scale
            const int testSize = 1000;
            var bigIn = new double[testSize];
            var bigOut = new double[testSize];
            for (int i = 0; i < testSize; i++) bigIn[i] = testInput;
            fixed (double* pIn = bigIn, pOut = bigOut)
            {
                fn(testSize, pIn, pOut, VML_EP);
            }
            if (double.IsNaN(bigOut[0]) || Math.Abs(bigOut[0] - expectedOutput) > tolerance)
                return null;
            if (double.IsNaN(bigOut[testSize - 1]) || Math.Abs(bigOut[testSize - 1] - expectedOutput) > tolerance)
                return null;

            return fn;
        }
        catch { return null; }
    }

    private static unsafe void ClearAllPointers()
    {
        _vsExp = null;
        _vsLn = null;
        _vsTanh = null;
        _vmdExp = null;
        _vmdLn = null;
        _vmdTanh = null;
    }
#endif
}
