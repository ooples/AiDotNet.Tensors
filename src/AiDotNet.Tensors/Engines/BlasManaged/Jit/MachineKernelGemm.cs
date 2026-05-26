using System;
#if NET5_0_OR_GREATER
using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Sub-S (#409): routes qualifying GEMMs to the hand-emitted machine-code
/// microkernels (FP64 6×8, FP32 6×16; ~57 GFLOPS/core FP64, ~95% of OpenBLAS —
/// see <see cref="MachineCodeFmaKernel"/>). First-party machine code; the caller
/// carves a tile-aligned interior for the kernel and routes the m%Mr / n%Nr tails
/// to the managed strategy, so it can only add speed, never change results.
///
/// <para>
/// Qualification (else → caller's existing path): x64 Windows, AVX2+FMA, dynamic
/// code allowed, no transpose, no epilogue, no pre-pack handle, and the interior
/// is tile-aligned (caller guarantees m%Mr==0 &amp;&amp; n%Nr==0). Each emitted
/// kernel is built once and cached for the process.
/// </para>
///
/// <para>
/// Requires function pointers + x86 intrinsics, so the kernel path is net5.0+
/// only; on net471 the <c>TryGemm*</c> methods always return false and callers use
/// the managed strategy.
/// </para>
/// </summary>
internal static class MachineKernelGemm
{
    /// <summary>Master switch (default on). Set false to force the managed path everywhere.</summary>
    internal static bool Enabled { get; set; } = true;

    /// <summary>FP64 microkernel row tile (M alignment the interior must satisfy).</summary>
    internal const int Fp64Mr = 6;
    /// <summary>FP64 microkernel column tile (N alignment the interior must satisfy).</summary>
    internal const int Fp64Nr = 8;
    /// <summary>FP32 microkernel row tile (M alignment the interior must satisfy).</summary>
    internal const int Fp32Mr = 6;
    /// <summary>FP32 microkernel column tile (N alignment the interior must satisfy).</summary>
    internal const int Fp32Nr = 16;

#if NET5_0_OR_GREATER
    private const int KcBlock = 256; // K-blocking so packed panels stay cache-resident.

    /// <summary>True when the FP64 machine kernel is usable on this process. Idempotent.</summary>
    internal static bool IsFp64Available => Enabled && TryInitFp64();
    /// <summary>True when the FP32 machine kernel is usable on this process. Idempotent.</summary>
    internal static bool IsFp32Available => Enabled && TryInitFp32();

    private static readonly object _lock = new();
    private static bool _tried64, _tried32;
    private static ExecutableMemory? _mem64, _mem32; // kept alive for the process (never disposed)
    private static nint _kern64, _kern32;

    private static bool PlatformOk() =>
        RuntimeInformation.OSArchitecture == Architecture.X64
        && RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        && Avx2.IsSupported && Fma.IsSupported
        && NativeAotDetector.IsDynamicCodeSupported;

    private static bool TryInitFp64()
    {
        if (_tried64) return _kern64 != 0;
        lock (_lock)
        {
            if (_tried64) return _kern64 != 0;
            _tried64 = true;
            if (!PlatformOk()) return false;
            try
            {
                _mem64 = ExecutableMemory.TryAllocate(MachineCodeFmaKernel.EmitFp64_6x8_PackedWindowsU4());
                if (_mem64 is not null) _kern64 = _mem64.Pointer;
            }
            catch { _mem64 = null; _kern64 = 0; }
            return _kern64 != 0;
        }
    }

    private static bool TryInitFp32()
    {
        if (_tried32) return _kern32 != 0;
        lock (_lock)
        {
            if (_tried32) return _kern32 != 0;
            _tried32 = true;
            if (!PlatformOk()) return false;
            try
            {
                _mem32 = ExecutableMemory.TryAllocate(MachineCodeFmaKernel.EmitFp32_6x16_PackedWindows());
                if (_mem32 is not null) _kern32 = _mem32.Pointer;
            }
            catch { _mem32 = null; _kern32 = 0; }
            return _kern32 != 0;
        }
    }

    /// <summary>
    /// Run C = A·B (FP64) with the machine kernel; returns false if the shape doesn't
    /// qualify. C must already be zeroed (the kernel accumulates).
    /// </summary>
    internal static unsafe bool TryGemmFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc, int m, int n, int k, bool hasEpilogue, bool hasPrePack)
    {
        if (!Enabled || transA || transB || hasEpilogue || hasPrePack) return false;
        if (m % Fp64Mr != 0 || n % Fp64Nr != 0 || m <= 0 || n <= 0 || k <= 0) return false;
        if (!TryInitFp64()) return false;
        return RunPacked<double>(a, lda, b, ldb, c, ldc, m, n, k, Fp64Mr, Fp64Nr, _kern64);
    }

    /// <summary>
    /// Run C = A·B (FP32) with the machine kernel; returns false if the shape doesn't
    /// qualify. C must already be zeroed (the kernel accumulates).
    /// </summary>
    internal static unsafe bool TryGemmFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc, int m, int n, int k, bool hasEpilogue, bool hasPrePack)
    {
        if (!Enabled || transA || transB || hasEpilogue || hasPrePack) return false;
        if (m % Fp32Mr != 0 || n % Fp32Nr != 0 || m <= 0 || n <= 0 || k <= 0) return false;
        if (!TryInitFp32()) return false;
        return RunPacked<float>(a, lda, b, ldb, c, ldc, m, n, k, Fp32Mr, Fp32Nr, _kern32);
    }

    /// <summary>
    /// Shared pack + parallel macro-loop. The <c>typeof(T)</c> branch is a JIT-time
    /// constant per instantiation (double/float) and folds away, so there's no
    /// per-call type test in the hot loop. C must be pre-zeroed; mc%Mr==0 &amp;&amp;
    /// n%Nr==0 (callers guarantee). Returns true.
    /// </summary>
    private static unsafe bool RunPacked<T>(
        ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> b, int ldb, Span<T> c, int ldc,
        int m, int n, int k, int Mr, int Nr, nint kernelAddr) where T : unmanaged
    {
        int mStripes = m / Mr;
        int nStripes = n / Nr;
        var packA = ArrayPool<T>.Shared.Rent(m * KcBlock);  // [mStripes, Kc, Mr]
        var packB = ArrayPool<T>.Shared.Rent(KcBlock * n);  // [nStripes, Kc, Nr]
        try
        {
            fixed (T* cPtr = c)
            fixed (T* paBase = packA)
            fixed (T* pbBase = packB)
            {
                // Capture buffer addresses as nint — pointers can't be captured by the
                // parallel body lambda, but the surrounding `fixed` keeps them pinned
                // for the whole (serial-outer) K-loop including the parallel region.
                nint paAddr = (nint)paBase, pbAddr = (nint)pbBase, cAddr = (nint)cPtr;
                int ldcL = ldc, nStripesL = nStripes, MrL = Mr, NrL = Nr;

                for (int pc = 0; pc < k; pc += KcBlock)
                {
                    int ekc = Math.Min(KcBlock, k - pc);

                    // Pack the Kc-block: A[:, pc:pc+ekc] (all m rows), B[pc:pc+ekc, :] (all n cols).
                    Avx2Pack.PackA<T>(a.Slice(pc), lda, false,
                        new Span<T>(paBase, m * ekc), mc: m, kc: ekc, Mr);
                    Avx2Pack.PackB<T>(b.Slice(pc * ldb), ldb, false,
                        new Span<T>(pbBase, ekc * n), nc: n, kc: ekc, Nr);

                    // Parallelize over M-stripes: each handles Mr disjoint C rows, so
                    // there's no write contention. The pc loop stays serial-outer, so
                    // C accumulates correctly across K-blocks. Small shapes run inline
                    // (ParallelForOrSerial's grain-size gate) — no thread-dispatch cost.
                    int ekcL = ekc;
                    long stripeWork = (long)nStripesL * ekcL * MrL * NrL * 2;
                    CpuParallelSettings.ParallelForOrSerial(0, mStripes, (long)mStripes * stripeWork, isr =>
                    {
                        if (typeof(T) == typeof(double))
                        {
                            var kern = (delegate* unmanaged<double*, double*, double*, long, long, void>)kernelAddr;
                            double* aS = (double*)paAddr + (long)isr * ekcL * MrL;
                            double* cR = (double*)cAddr + (long)(isr * MrL) * ldcL;
                            double* pb = (double*)pbAddr;
                            for (int jsr = 0; jsr < nStripesL; jsr++)
                                kern(aS, pb + (long)jsr * ekcL * NrL, cR + jsr * NrL, ldcL, ekcL);
                        }
                        else // float
                        {
                            var kern = (delegate* unmanaged<float*, float*, float*, long, long, void>)kernelAddr;
                            float* aS = (float*)paAddr + (long)isr * ekcL * MrL;
                            float* cR = (float*)cAddr + (long)(isr * MrL) * ldcL;
                            float* pb = (float*)pbAddr;
                            for (int jsr = 0; jsr < nStripesL; jsr++)
                                kern(aS, pb + (long)jsr * ekcL * NrL, cR + jsr * NrL, ldcL, ekcL);
                        }
                    });
                }
            }
            return true;
        }
        finally
        {
            ArrayPool<T>.Shared.Return(packA);
            ArrayPool<T>.Shared.Return(packB);
        }
    }
#else
    /// <summary>net471 has no x86 intrinsics / function pointers — always defer to managed.</summary>
    internal static bool IsFp64Available => false;
    internal static bool IsFp32Available => false;

    internal static bool TryGemmFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc, int m, int n, int k, bool hasEpilogue, bool hasPrePack) => false;

    internal static bool TryGemmFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc, int m, int n, int k, bool hasEpilogue, bool hasPrePack) => false;
#endif
}
