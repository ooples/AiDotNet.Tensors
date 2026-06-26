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

    /// <summary>
    /// #653: drive each FP32 AVX2 M-stripe with one GEBP panel call (default on). Set false to
    /// fall back to the per-N-tile call loop — used by the perf harness to A/B the throttle fix.
    /// </summary>
    internal static bool EnablePanelKernel { get; set; } = true;

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
    private static bool _tried64, _tried32, _triedZ64, _triedZ32, _triedPanel32, _triedMacro32;
    private static ExecutableMemory? _mem64, _mem32, _memZ64, _memZ32, _memPanel32, _memMacro32; // kept alive for the process
    private static nint _kern64, _kern32, _kernZ64, _kernZ32, _kernPanel32, _kernMacro32;

    // #663: expose the #653 panel kernel to the N-axis path (PackBothStrategy.RunNAxisParallelUnsafe).
    // Shares _kernPanel32 + TryInitPanelFp32 with the internal RunPacked panel use (both branches
    // added the same kernel independently; unified here), and respects the same EnablePanelKernel
    // master switch so disabling the panel disables BOTH consumers.
    /// <summary>True when the FP32 6×16 PANEL kernel is usable on this process.</summary>
    internal static bool IsFp32PanelAvailable => Enabled && EnablePanelKernel && TryInitPanelFp32();

    /// <summary>
    /// Run the FP32 6×16 panel kernel: C[0..6, 0..njr·16] += packedA[Kc×6] · packedB[njr×Kc×16]
    /// over kc K-steps, looping the Nr=16 tiles internally. C is read-modify-write (pre-zero for
    /// a fresh result). Caller guarantees the panel kernel is available (IsFp32PanelAvailable).
    /// </summary>
    internal static unsafe void RunPanelFp32(
        ReadOnlySpan<float> packedA, ReadOnlySpan<float> packedB,
        Span<float> c, int ldc, int kc, int njr)
    {
        var kern = (delegate* unmanaged<float*, float*, float*, long, long, long, void>)_kernPanel32;
        fixed (float* pa = packedA)
        fixed (float* pb = packedB)
        fixed (float* pc = c)
            kern(pa, pb, pc, ldc, kc, njr);
    }

    /// <summary>True when the FP32 6×16 MACRO kernel (whole Mr-sweep in asm) is usable.</summary>
    internal static bool IsFp32MacroAvailable => Enabled && EnablePanelKernel && TryInitMacroFp32();

    /// <summary>
    /// Run the FP32 6×16 macro kernel: for a single K-panel, does numMr Mr-blocks × njr Nr=16
    /// tiles × kc K-steps entirely in machine code, reusing packedB across Mr-blocks. C is
    /// read-modify-write. aBase points at the panel's first Mr-stripe; consecutive stripes are
    /// aStrideBytes apart; C advances Mr·ldc rows per Mr-block. Caller guarantees availability.
    /// </summary>
    internal static unsafe void RunMacroPanelFp32(
        float* aBase, float* bBase, float* cBase,
        int ldc, int kc, int njr, int numMr, int aStrideBytes)
    {
        var kern = (delegate* unmanaged<float*, float*, float*, long, long, long, long, long, void>)_kernMacro32;
        kern(aBase, bBase, cBase, ldc, kc, njr, numMr, aStrideBytes);
    }

    private static bool TryInitMacroFp32()
    {
        if (_triedMacro32) return _kernMacro32 != 0;
        lock (_lock)
        {
            if (_triedMacro32) return _kernMacro32 != 0;
            _triedMacro32 = true;
            if (!PlatformOk()) return false;
            try
            {
                byte[] code = IsWindows
                    ? MachineCodeFmaKernel.EmitFp32_6x16_MacroWindows()
                    : MachineCodeFmaKernel.EmitFp32_6x16_MacroSysV();
                _memMacro32 = ExecutableMemory.TryAllocate(code);
                if (_memMacro32 is not null) _kernMacro32 = _memMacro32.Pointer;
            }
            catch { _memMacro32 = null; _kernMacro32 = 0; }
            return _kernMacro32 != 0;
        }
    }

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

    /// <summary>
    /// #653 GEBP panel driver (AVX2 FP32 6×16): one indirect call computes all N-tiles of an
    /// M-stripe, looping N inside the machine code — the per-N-tile call in <see cref="RunPacked{T}"/>
    /// throttled the kernel ~20% below its raw ~115 GF/s. Init is lazy/idempotent like the others.
    /// </summary>
    private static bool TryInitPanelFp32()
    {
        if (_triedPanel32) return _kernPanel32 != 0;
        lock (_lock)
        {
            if (_triedPanel32) return _kernPanel32 != 0;
            _triedPanel32 = true;
            if (!PlatformOk()) return false;
            try
            {
                byte[] code = IsWindows
                    ? MachineCodeFmaKernel.EmitFp32_6x16_PanelWindows()
                    : MachineCodeFmaKernel.EmitFp32_6x16_PanelSysV();
                _memPanel32 = ExecutableMemory.TryAllocate(code);
                if (_memPanel32 is not null) _kernPanel32 = _memPanel32.Pointer;
            }
            catch { _memPanel32 = null; _kernPanel32 = 0; }
            return _kernPanel32 != 0;
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
        // #653: the AVX2 6×16 path can drive a whole M-stripe (all N-tiles) per call via the
        // GEBP panel kernel; the AVX-512 6×32 path has no panel kernel yet (falls back to per-tile).
        nint panel = (!z && EnablePanelKernel && TryInitPanelFp32()) ? _kernPanel32 : 0;
        return RunPacked<float>(a, lda, b, ldb, c, ldc, m, n, k, Fp32Mr, nr, kern, panel);
    }

    /// <summary>
    /// Shared pack + parallel macro-loop. The <c>typeof(T)</c> branch is a JIT-time constant per
    /// instantiation (double/float) and folds away, so there's no per-call type test in the hot
    /// loop. C must be pre-zeroed; m%Mr==0 &amp;&amp; n%Nr==0 (callers guarantee). Returns true.
    ///
    /// <para>
    /// #653 macro-blocking: the K-loop (pc) packs whole A (shared, read-only) and whole B once per
    /// K-block, then parallelizes over <b>N-panels</b> (the jc dimension) — NOT the inner M-stripes.
    /// The previous M-stripe split had all cores hammer ONE shared packed-B panel, maximal cache
    /// contention (the BLIS many-core anti-pattern) that capped scaling at ~3.4× on 16 cores. Giving
    /// each thread its own B sub-panel — sized so tiles·Nr·Kc stays ~half-L2-resident, with block
    /// width chosen for ≥ <see cref="CpuParallelSettings.MaxDegreeOfParallelism"/> blocks of
    /// occupancy — keeps the kernel's hot B data in that thread's private L2. Each thread writes a
    /// disjoint set of C columns, so the split is race-free and deterministic. Packing stays serial
    /// (the Span pack API can't be captured by the parallel lambda); A is reused across all panels.
    /// </para>
    /// </summary>
    private static unsafe bool RunPacked<T>(
        ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> b, int ldb, Span<T> c, int ldc,
        int m, int n, int k, int Mr, int Nr, nint kernelAddr, nint panelKernelAddr = 0) where T : unmanaged
    {
        int elem = sizeof(T);
        int mStripes = m / Mr;
        int nTiles = n / Nr;

        // N-panel block width (in Nr-tiles): enough blocks for full core occupancy, but capped so a
        // thread's packed-B sub-panel (tiles·Nr·Kc) stays ~half-L2-resident (private per thread).
        int maxDeg = Math.Max(1, CpuParallelSettings.MaxDegreeOfParallelism);
        int occTiles = Math.Max(1, (nTiles + maxDeg - 1) / maxDeg);       // ceil → ≥ maxDeg blocks
        int l2Tiles = Math.Max(1, (256 * 1024) / (KcBlock * Nr * elem));  // sub-panel ≤ ~half L2
        int tilesPerBlock = Math.Min(occTiles, l2Tiles);
        int jcBlocks = (nTiles + tilesPerBlock - 1) / tilesPerBlock;

        var packA = ArrayPool<T>.Shared.Rent(m * KcBlock);   // [mStripes, Kc, Mr] (whole A panel per Kc)
        var packB = ArrayPool<T>.Shared.Rent(KcBlock * n);   // [nTiles, Kc, Nr]   (whole B panel per Kc)
        try
        {
            fixed (T* cPtr = c)
            fixed (T* paBase = packA)
            fixed (T* pbBase = packB)
            {
                // Capture buffer addresses as nint — pointers can't be captured by the
                // parallel body lambda, but the surrounding `fixed` keeps them pinned.
                nint paAddr = (nint)paBase, pbAddr = (nint)pbBase, cAddr = (nint)cPtr;
                int ldcL = ldc, MrL = Mr, NrL = Nr, mStripesL = mStripes, nTilesL = nTiles, tpbL = tilesPerBlock;
                nint panelAddr = panelKernelAddr; // FP32 AVX2 GEBP panel kernel (0 = per-tile path)

                for (int pc = 0; pc < k; pc += KcBlock)
                {
                    int ekc = Math.Min(KcBlock, k - pc);
                    // Pack whole A[:, pc:pc+ekc] and whole B[pc:pc+ekc, :] once per K-block.
                    Avx2Pack.PackA<T>(a.Slice(pc), lda, false,
                        new Span<T>(paBase, m * ekc), mc: m, kc: ekc, Mr);
                    Avx2Pack.PackB<T>(b.Slice(pc * ldb), ldb, false,
                        new Span<T>(pbBase, ekc * n), nc: n, kc: ekc, Nr);

                    int ekcL = ekc;
                    long flopsPerBlock = (long)tpbL * mStripesL * ekcL * MrL * NrL * 2;
                    CpuParallelSettings.ParallelForOrSerial(0, jcBlocks, (long)jcBlocks * flopsPerBlock, jb =>
                    {
                        int t0 = jb * tpbL;
                        int t1 = Math.Min(t0 + tpbL, nTilesL);
                        int nt = t1 - t0;
                        if (nt <= 0) return;
                        if (typeof(T) == typeof(double))
                        {
                            var kern = (delegate* unmanaged<double*, double*, double*, long, long, void>)kernelAddr;
                            double* pa = (double*)paAddr, pb = (double*)pbAddr, cb = (double*)cAddr;
                            for (int isr = 0; isr < mStripesL; isr++)
                            {
                                double* aS = pa + (long)isr * ekcL * MrL;
                                double* cR = cb + (long)(isr * MrL) * ldcL + (long)t0 * NrL;
                                for (int t = t0; t < t1; t++)
                                    kern(aS, pb + (long)t * ekcL * NrL, cR + (long)(t - t0) * NrL, ldcL, ekcL);
                            }
                        }
                        else // float
                        {
                            float* pa = (float*)paAddr, pb = (float*)pbAddr, cb = (float*)cAddr;
                            for (int isr = 0; isr < mStripesL; isr++)
                            {
                                float* aS = pa + (long)isr * ekcL * MrL;
                                float* cR = cb + (long)(isr * MrL) * ldcL + (long)t0 * NrL;
                                if (panelAddr != 0)
                                {
                                    // One GEBP call drives all N-tiles of this M-stripe (loops N in machine code).
                                    var panel = (delegate* unmanaged<float*, float*, float*, long, long, long, void>)panelAddr;
                                    panel(aS, pb + (long)t0 * ekcL * NrL, cR, ldcL, ekcL, nt);
                                }
                                else
                                {
                                    var kern = (delegate* unmanaged<float*, float*, float*, long, long, void>)kernelAddr;
                                    for (int t = t0; t < t1; t++)
                                        kern(aS, pb + (long)t * ekcL * NrL, cR + (long)(t - t0) * NrL, ldcL, ekcL);
                                }
                            }
                        }
                    }, deterministicSafe: true); // N-panel split: each thread writes disjoint C columns
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
    internal static bool IsFp32PanelAvailable => false;
    /// <summary>net471 has no JIT panel kernel; callers must gate on <see cref="IsFp32PanelAvailable"/>.
    /// Reached only by a caller that skipped that gate — fail loud rather than silently leave C unchanged.</summary>
    internal static void RunPanelFp32(
        ReadOnlySpan<float> packedA, ReadOnlySpan<float> packedB,
        Span<float> c, int ldc, int kc, int njr) =>
        throw new PlatformNotSupportedException(
            "RunPanelFp32 requires the net5.0+ machine-code JIT path; gate on IsFp32PanelAvailable before calling.");
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
