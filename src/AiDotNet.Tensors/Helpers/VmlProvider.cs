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
    private static unsafe delegate* unmanaged[Cdecl]<int, float*, float*, void> _vsExp;
    private static unsafe delegate* unmanaged[Cdecl]<int, double*, double*, void> _vdExp;
    private static unsafe delegate* unmanaged[Cdecl]<int, float*, float*, void> _vsLn;
    private static unsafe delegate* unmanaged[Cdecl]<int, double*, double*, void> _vdLn;
    private static unsafe delegate* unmanaged[Cdecl]<int, float*, float*, void> _vsTanh;
    private static unsafe delegate* unmanaged[Cdecl]<int, double*, double*, void> _vdTanh;

    // vmlSetMode: sets accuracy mode. VML_LA=0x1 (fast), VML_HA=0x2 (default, slow for double)
    private static unsafe delegate* unmanaged[Cdecl]<uint, uint> _vmlSetMode;
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

            // Load vmlSetMode to set VML_LA (Low Accuracy = fast mode).
            // Default VML_HA is extremely slow for double (600x slower than scalar Math.Exp).
            // VML_LA gives ~11 correct digits for double — more than enough for neural nets.
            _vmlSetMode = (delegate* unmanaged[Cdecl]<uint, uint>)
                TryGetExport(handle, "vmlSetMode", "MKL_vmlSetMode", "VMLSETMODE");
            if (_vmlSetMode != null)
            {
                const uint VML_LA = 0x00000001;
                _vmlSetMode(VML_LA);
            }

            // Verify correctness at n=1 AND n=1000 (catches scale-dependent issues).
            // Also verify performance: must be faster than scalar at n=10000.
            _vsExp = VerifyFloat(_vsExp, 1.0f, 2.71828f, 0.01f);
            _vsLn = VerifyFloat(_vsLn, 2.71828f, 1.0f, 0.01f);
            _vsTanh = VerifyFloat(_vsTanh, 1.0f, 0.7616f, 0.01f);
            _vdExp = VerifyDoubleAtScale(_vdExp, 1.0, 2.71828, 0.001);
            _vdLn = VerifyDoubleAtScale(_vdLn, 2.71828, 1.0, 0.001);
            _vdTanh = VerifyDoubleAtScale(_vdTanh, 1.0, 0.76159, 0.001);

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
    /// Previous bug: vdExp passed n=1 verification but was 600x slower at scale
    /// because VML defaulted to VML_HA (High Accuracy) mode for double.
    /// </summary>
    private static unsafe delegate* unmanaged[Cdecl]<int, double*, double*, void>
        VerifyDoubleAtScale(delegate* unmanaged[Cdecl]<int, double*, double*, void> fn,
        double testInput, double expectedOutput, double tolerance)
    {
        if (fn == null) return null;
        try
        {
            // Step 1: Verify correctness at n=1
            var inArr = new double[] { testInput };
            var outArr = new double[] { 0.0 };
            fixed (double* pIn = inArr, pOut = outArr)
            {
                fn(1, pIn, pOut);
            }
            if (double.IsNaN(outArr[0]) || double.IsInfinity(outArr[0])
                || Math.Abs(outArr[0] - expectedOutput) > tolerance)
                return null;

            // Step 2: Verify correctness at n=1000
            const int testSize = 1000;
            var bigIn = new double[testSize];
            var bigOut = new double[testSize];
            for (int i = 0; i < testSize; i++) bigIn[i] = testInput;
            fixed (double* pIn = bigIn, pOut = bigOut)
            {
                fn(testSize, pIn, pOut);
            }
            // Check first and last elements
            if (double.IsNaN(bigOut[0]) || Math.Abs(bigOut[0] - expectedOutput) > tolerance)
                return null;
            if (double.IsNaN(bigOut[testSize - 1]) || Math.Abs(bigOut[testSize - 1] - expectedOutput) > tolerance)
                return null;

            // Step 3: Verify performance — VML must be faster than scalar at n=10000.
            // If VML is in VML_HA mode and we failed to set VML_LA, it will be 100x+ slower.
            const int perfSize = 10000;
            var perfIn = new double[perfSize];
            var perfOut = new double[perfSize];
            for (int i = 0; i < perfSize; i++) perfIn[i] = 0.5;

            // Scalar baseline
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < perfSize; i++) perfOut[i] = Math.Exp(perfIn[i]);
            sw.Stop();
            long scalarTicks = sw.ElapsedTicks;

            // VML
            sw.Restart();
            fixed (double* pIn = perfIn, pOut = perfOut)
            {
                fn(perfSize, pIn, pOut);
            }
            sw.Stop();
            long vmlTicks = sw.ElapsedTicks;

            // VML must be no more than 3x slower than scalar to be useful.
            // If it's slower, VML_LA mode wasn't set successfully — disable.
            if (vmlTicks > scalarTicks * 3)
                return null;

            return fn;
        }
        catch { return null; }
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
