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

    /// <summary>
    /// Opt-in AVX-512 path (default OFF). The EVEX kernels are encoding-verified
    /// (capstone disassembly) but NOT yet execution-verified on real AVX-512 hardware
    /// (none available locally or in CI), so the verified AVX2 kernels remain the
    /// default. Flip to true only after the Avx512F-gated tests are green on an
    /// AVX-512 runner. When enabled AND Avx512F is supported, the FP64 tile widens to
    /// 6×16 and FP32 to 6×32 (512-bit), otherwise the AVX2 6×8 / 6×16 tiles are used.
    /// </summary>
    internal static bool EnableAvx512 { get; set; }

#if NET5_0_OR_GREATER
    private const int KcBlock = 256; // K-blocking so packed panels stay cache-resident.

    /// <summary>True when the FP64 machine kernel is usable on this process. Idempotent.</summary>
    internal static bool IsFp64Available => Enabled && (TryInitFp64() || UseAvx512Fp64);
    /// <summary>True when the FP32 machine kernel is usable on this process. Idempotent.</summary>
    internal static bool IsFp32Available => Enabled && (TryInitFp32() || UseAvx512Fp32);

    /// <summary>Effective FP64 N-tile (16 when the AVX-512 path is active, else 8).</summary>
    internal static int ActiveFp64Nr => UseAvx512Fp64 ? 16 : Fp64Nr;
    /// <summary>Effective FP32 N-tile (32 when the AVX-512 path is active, else 16).</summary>
    internal static int ActiveFp32Nr => UseAvx512Fp32 ? 32 : Fp32Nr;

    private static bool UseAvx512Fp64 => EnableAvx512 && Avx512F.IsSupported && TryInitAvx512Fp64();
    private static bool UseAvx512Fp32 => EnableAvx512 && Avx512F.IsSupported && TryInitAvx512Fp32();

    private static readonly object _lock = new();
    private static bool _tried64, _tried32, _triedZ64, _triedZ32;
    private static ExecutableMemory? _mem64, _mem32, _memZ64, _memZ32; // kept alive for the process
    private static nint _kern64, _kern32, _kernZ64, _kernZ32;

    private static bool IsWindows => RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

    // x64 on a supported OS with AVX2+FMA and dynamic code allowed. Windows uses the
    // Windows-x64 ABI kernels; Linux/macOS use the System V AMD64 ABI kernels.
    private static bool PlatformOk() =>
        RuntimeInformation.OSArchitecture == Architecture.X64
        && (IsWindows
            || RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
            || RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
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
                byte[] code = IsWindows
                    ? MachineCodeFmaKernel.EmitFp64_6x8_PackedWindowsU4()
                    : MachineCodeFmaKernel.EmitFp64_6x8_PackedSysVU4();
                _mem64 = ExecutableMemory.TryAllocate(code);
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
                byte[] code = IsWindows
                    ? MachineCodeFmaKernel.EmitFp32_6x16_PackedWindows()
                    : MachineCodeFmaKernel.EmitFp32_6x16_PackedSysV();
                _mem32 = ExecutableMemory.TryAllocate(code);
                if (_mem32 is not null) _kern32 = _mem32.Pointer;
            }
            catch { _mem32 = null; _kern32 = 0; }
            return _kern32 != 0;
        }
    }

    private static bool TryInitAvx512Fp64()
    {
        if (_triedZ64) return _kernZ64 != 0;
        lock (_lock)
        {
            if (_triedZ64) return _kernZ64 != 0;
            _triedZ64 = true;
            if (!PlatformOk()) return false;
            try
            {
                byte[] code = IsWindows
                    ? MachineCodeFmaKernel.EmitFp64_6x16_Avx512Windows()
                    : MachineCodeFmaKernel.EmitFp64_6x16_Avx512SysV();
                _memZ64 = ExecutableMemory.TryAllocate(code);
                if (_memZ64 is not null) _kernZ64 = _memZ64.Pointer;
            }
            catch { _memZ64 = null; _kernZ64 = 0; }
            return _kernZ64 != 0;
        }
    }

    private static bool TryInitAvx512Fp32()
    {
        if (_triedZ32) return _kernZ32 != 0;
        lock (_lock)
        {
            if (_triedZ32) return _kernZ32 != 0;
            _triedZ32 = true;
            if (!PlatformOk()) return false;
            try
            {
                byte[] code = IsWindows
                    ? MachineCodeFmaKernel.EmitFp32_6x32_Avx512Windows()
                    : MachineCodeFmaKernel.EmitFp32_6x32_Avx512SysV();
                _memZ32 = ExecutableMemory.TryAllocate(code);
                if (_memZ32 is not null) _kernZ32 = _memZ32.Pointer;
            }
            catch { _memZ32 = null; _kernZ32 = 0; }
            return _kernZ32 != 0;
        }
    }

    /// <summary>
    /// Run C = A·B (FP64) with the machine kernel; returns false if the shape doesn't
    /// qualify. C must already be zeroed (the kernel accumulates). Uses the AVX-512
    /// 6×16 kernel when that path is active, otherwise the AVX2 6×8 kernel.
    /// </summary>
    internal static unsafe bool TryGemmFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc, int m, int n, int k, bool hasEpilogue, bool hasPrePack)
    {
        if (!Enabled || transA || transB || hasEpilogue || hasPrePack) return false;
        bool z = UseAvx512Fp64;
        int nr = z ? 16 : Fp64Nr;
        nint kern = z ? _kernZ64 : _kern64;
        if (!z && !TryInitFp64()) return false;
        if (kern == 0) return false;
        if (m % Fp64Mr != 0 || n % nr != 0 || m <= 0 || n <= 0 || k <= 0) return false;
        return RunPacked<double>(a, lda, b, ldb, c, ldc, m, n, k, Fp64Mr, nr, kern);
    }

    /// <summary>
    /// Run C = A·B (FP32) with the machine kernel; returns false if the shape doesn't
    /// qualify. C must already be zeroed. Uses the AVX-512 6×32 kernel when active,
    /// otherwise the AVX2 6×16 kernel.
    /// </summary>
    internal static unsafe bool TryGemmFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc, int m, int n, int k, bool hasEpilogue, bool hasPrePack)
    {
        if (!Enabled || transA || transB || hasEpilogue || hasPrePack) return false;
        bool z = UseAvx512Fp32;
        int nr = z ? 32 : Fp32Nr;
        nint kern = z ? _kernZ32 : _kern32;
        if (!z && !TryInitFp32()) return false;
        if (kern == 0) return false;
        if (m % Fp32Mr != 0 || n % nr != 0 || m <= 0 || n <= 0 || k <= 0) return false;
        return RunPacked<float>(a, lda, b, ldb, c, ldc, m, n, k, Fp32Mr, nr, kern);
    }

    /// <summary>
    /// Shared pack + parallel macro-loop with BLIS-style cache blocking. The
    /// <c>typeof(T)</c> branch is a JIT-time constant per instantiation (double/float)
    /// and folds away, so there's no per-call type test in the hot loop. C must be
    /// pre-zeroed; m%Mr==0 &amp;&amp; n%Nr==0 (callers guarantee). Returns true.
    ///
    /// <para>
    /// Loop nest: pc (Kc K-blocks) → jc (Nc N-panels) → parallel over M-stripes.
    /// The packed-B panel for one jc is Kc×Nc sized to fit ~half of L2, so it stays
    /// resident and is re-read from L2 (not memory) by every M-stripe — the previous
    /// "pack whole B, re-stream per stripe" cost (hundreds of MB per K-block at large
    /// n) was the dominant bottleneck. The B micro-column the kernel reads (Kc×Nr)
    /// fits L1. Packing stays outside the parallel body (Span can't be captured by a
    /// lambda); the M-stripe parallelism keeps full core occupancy.
    /// </para>
    /// </summary>
    private static unsafe bool RunPacked<T>(
        ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> b, int ldb, Span<T> c, int ldc,
        int m, int n, int k, int Mr, int Nr, nint kernelAddr) where T : unmanaged
    {
        int elem = sizeof(T);
        // Nc: packed-B panel size, chosen by dtype because the bottleneck differs.
        // FP32 does 2× the FMAs/byte, so it is L3-bandwidth-bound and wins big when the
        // B panel (Kc×Nc) is kept ~half-L2-resident and re-read from L2 by every
        // M-stripe (measured +97% at 1536³). FP64 is compute-bound (already ~hardware
        // peak), so the extra N-panel/loop overhead costs more than any cache gain —
        // give it a large Nc that degenerates to whole-N (no blocking) for typical n.
        int targetBytes = elem == 4 ? 256 * 1024 : 8 * 1024 * 1024;
        int Nc = targetBytes / (KcBlock * elem);
        Nc -= Nc % Nr;
        if (Nc < Nr) Nc = Nr;
        Nc = Math.Min(Nc, ((n + Nr - 1) / Nr) * Nr); // no larger than n (rounded up to Nr)

        int mStripes = m / Mr;
        var packA = ArrayPool<T>.Shared.Rent(m * KcBlock);   // [mStripes, Kc, Mr] (whole A panel per Kc)
        var packB = ArrayPool<T>.Shared.Rent(KcBlock * Nc);  // [Nc/Nr, Kc, Nr]   (one N-panel)
        try
        {
            fixed (T* cPtr = c)
            fixed (T* paBase = packA)
            fixed (T* pbBase = packB)
            {
                // Capture buffer addresses as nint — pointers can't be captured by the
                // parallel body lambda, but the surrounding `fixed` keeps them pinned.
                nint paAddr = (nint)paBase, pbAddr = (nint)pbBase, cAddr = (nint)cPtr;
                int ldcL = ldc, MrL = Mr, NrL = Nr;

                for (int pc = 0; pc < k; pc += KcBlock)
                {
                    int ekc = Math.Min(KcBlock, k - pc);
                    // Pack the whole A panel A[:, pc:pc+ekc] once per K-block.
                    Avx2Pack.PackA<T>(a.Slice(pc), lda, false,
                        new Span<T>(paBase, m * ekc), mc: m, kc: ekc, Mr);

                    for (int jc = 0; jc < n; jc += Nc)
                    {
                        int ncEff = Math.Min(Nc, n - jc);
                        int njStripes = ncEff / Nr;
                        // Pack the B panel B[pc:pc+ekc, jc:jc+ncEff] (fits L2).
                        Avx2Pack.PackB<T>(b.Slice(pc * ldb + jc), ldb, false,
                            new Span<T>(pbBase, ekc * ncEff), nc: ncEff, kc: ekc, Nr);

                        int ekcL = ekc, jcL = jc, njStripesL = njStripes;
                        long stripeWork = (long)njStripesL * ekcL * MrL * NrL * 2;
                        CpuParallelSettings.ParallelForOrSerial(0, mStripes, (long)mStripes * stripeWork, isr =>
                        {
                            if (typeof(T) == typeof(double))
                            {
                                var kern = (delegate* unmanaged<double*, double*, double*, long, long, void>)kernelAddr;
                                double* aS = (double*)paAddr + (long)isr * ekcL * MrL;
                                double* cR = (double*)cAddr + (long)(isr * MrL) * ldcL + jcL;
                                double* pb = (double*)pbAddr;
                                for (int js = 0; js < njStripesL; js++)
                                    kern(aS, pb + (long)js * ekcL * NrL, cR + js * NrL, ldcL, ekcL);
                            }
                            else // float
                            {
                                var kern = (delegate* unmanaged<float*, float*, float*, long, long, void>)kernelAddr;
                                float* aS = (float*)paAddr + (long)isr * ekcL * MrL;
                                float* cR = (float*)cAddr + (long)(isr * MrL) * ldcL + jcL;
                                float* pb = (float*)pbAddr;
                                for (int js = 0; js < njStripesL; js++)
                                    kern(aS, pb + (long)js * ekcL * NrL, cR + js * NrL, ldcL, ekcL);
                            }
                        }, deterministicSafe: true); // M-stripe split: each C tile reduced by one worker, fixed order
                    }
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
    internal static int ActiveFp64Nr => Fp64Nr;
    internal static int ActiveFp32Nr => Fp32Nr;

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
