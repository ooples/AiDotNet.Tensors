using System;
using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
using System.Threading;
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

    // #475 Phase-0 A/B support: superseded panel/macro kernels are RETIRED here (never disposed)
    // so their RX code pages stay mapped for the process lifetime. ResetFp32Kernels re-emits with a
    // changed PanelKUnroll; a previously-handed-out kernel pointer may still be mid-call on another
    // thread, so the old ExecutableMemory must NOT be freed (that would munmap/VirtualFree code a
    // thread is executing → use-after-free). Retaining the handful of A/B generations is bounded.
    private static readonly System.Collections.Generic.List<ExecutableMemory> _retiredKernels = new();

    /// <summary>
    /// #475 Phase-0 A/B support: drop the cached FP32 panel + macro kernels so the next
    /// IsFp32PanelAvailable/IsFp32MacroAvailable re-emits them with the current
    /// MachineCodeFmaKernel.PanelKUnroll (and other emit-time knobs). Test/bench use only.
    /// <para>
    /// The superseded <see cref="ExecutableMemory"/> is moved to <c>_retiredKernels</c> rather than
    /// freed, so any kernel pointer still in flight on another thread keeps executing valid mapped
    /// code (dropping it to the GC could unmap pages a thread is mid-call into — a use-after-free).
    /// </para>
    /// </summary>
    internal static void ResetFp32Kernels()
    {
        lock (_lock)
        {
            if (_memPanel32 is not null) _retiredKernels.Add(_memPanel32);
            if (_memMacro32 is not null) _retiredKernels.Add(_memMacro32);
            _triedPanel32 = false; _kernPanel32 = 0; _memPanel32 = null;
            _triedMacro32 = false; _kernMacro32 = 0; _memMacro32 = null;
        }
    }

    /// <summary>
    /// Marks a lazy kernel init as tried only after its kernel field is stored. The flag used to be set
    /// first, so a thread on the unlocked fast path could see "tried" with the kernel still 0 and
    /// silently take the fallback for that call (#685 audit).
    /// </summary>
    private static bool Publish(ref bool tried, bool result)
    {
        Volatile.Write(ref tried, true);
        return result;
    }
    private static bool TryInitMacroFp32()
    {
        if (Volatile.Read(ref _triedMacro32)) return _kernMacro32 != 0;
        lock (_lock)
        {
            if (_triedMacro32) return _kernMacro32 != 0;
            if (!PlatformOk()) return Publish(ref _triedMacro32, false);
            try
            {
                byte[] code = IsWindows
                    ? MachineCodeFmaKernel.EmitFp32_6x16_MacroWindows()
                    : MachineCodeFmaKernel.EmitFp32_6x16_MacroSysV();
                _memMacro32 = ExecutableMemory.TryAllocate(code);
                if (_memMacro32 is not null) _kernMacro32 = _memMacro32.Pointer;
            }
            catch { _memMacro32 = null; _kernMacro32 = 0; }
            return Publish(ref _triedMacro32, _kernMacro32 != 0);
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
        if (Volatile.Read(ref _tried64)) return _kern64 != 0;
        lock (_lock)
        {
            if (_tried64) return _kern64 != 0;
            if (!PlatformOk()) return Publish(ref _tried64, false);
            try
            {
                byte[] code = IsWindows
                    ? MachineCodeFmaKernel.EmitFp64_6x8_PackedWindowsU4()
                    : MachineCodeFmaKernel.EmitFp64_6x8_PackedSysVU4();
                _mem64 = ExecutableMemory.TryAllocate(code);
                if (_mem64 is not null) _kern64 = _mem64.Pointer;
            }
            catch { _mem64 = null; _kern64 = 0; }
            return Publish(ref _tried64, _kern64 != 0);
        }
    }

    private static bool TryInitFp32()
    {
        if (Volatile.Read(ref _tried32)) return _kern32 != 0;
        lock (_lock)
        {
            if (_tried32) return _kern32 != 0;
            if (!PlatformOk()) return Publish(ref _tried32, false);
            try
            {
                byte[] code = IsWindows
                    ? MachineCodeFmaKernel.EmitFp32_6x16_PackedWindows()
                    : MachineCodeFmaKernel.EmitFp32_6x16_PackedSysV();
                _mem32 = ExecutableMemory.TryAllocate(code);
                if (_mem32 is not null) _kern32 = _mem32.Pointer;
            }
            catch { _mem32 = null; _kern32 = 0; }
            return Publish(ref _tried32, _kern32 != 0);
        }
    }

    /// <summary>
    /// #653 GEBP panel driver (AVX2 FP32 6×16): one indirect call computes all N-tiles of an
    /// M-stripe, looping N inside the machine code — the per-N-tile call in <see cref="RunPacked{T}"/>
    /// throttled the kernel ~20% below its raw ~115 GF/s. Init is lazy/idempotent like the others.
    /// </summary>
    private static bool TryInitPanelFp32()
    {
        if (Volatile.Read(ref _triedPanel32)) return _kernPanel32 != 0;
        lock (_lock)
        {
            if (_triedPanel32) return _kernPanel32 != 0;
            if (!PlatformOk()) return Publish(ref _triedPanel32, false);
            try
            {
                byte[] code = IsWindows
                    ? MachineCodeFmaKernel.EmitFp32_6x16_PanelWindows()
                    : MachineCodeFmaKernel.EmitFp32_6x16_PanelSysV();
                _memPanel32 = ExecutableMemory.TryAllocate(code);
                if (_memPanel32 is not null) _kernPanel32 = _memPanel32.Pointer;
            }
            catch { _memPanel32 = null; _kernPanel32 = 0; }
            return Publish(ref _triedPanel32, _kernPanel32 != 0);
        }
    }

    private static bool TryInitAvx512Fp64()
    {
        if (Volatile.Read(ref _triedZ64)) return _kernZ64 != 0;
        lock (_lock)
        {
            if (_triedZ64) return _kernZ64 != 0;
            if (!PlatformOk()) return Publish(ref _triedZ64, false);
            try
            {
                byte[] code = IsWindows
                    ? MachineCodeFmaKernel.EmitFp64_6x16_Avx512Windows()
                    : MachineCodeFmaKernel.EmitFp64_6x16_Avx512SysV();
                _memZ64 = ExecutableMemory.TryAllocate(code);
                if (_memZ64 is not null) _kernZ64 = _memZ64.Pointer;
            }
            catch { _memZ64 = null; _kernZ64 = 0; }
            return Publish(ref _triedZ64, _kernZ64 != 0);
        }
    }

    private static bool TryInitAvx512Fp32()
    {
        if (Volatile.Read(ref _triedZ32)) return _kernZ32 != 0;
        lock (_lock)
        {
            if (_triedZ32) return _kernZ32 != 0;
            if (!PlatformOk()) return Publish(ref _triedZ32, false);
            try
            {
                byte[] code = IsWindows
                    ? MachineCodeFmaKernel.EmitFp32_6x32_Avx512Windows()
                    : MachineCodeFmaKernel.EmitFp32_6x32_Avx512SysV();
                _memZ32 = ExecutableMemory.TryAllocate(code);
                if (_memZ32 is not null) _kernZ32 = _memZ32.Pointer;
            }
            catch { _memZ32 = null; _kernZ32 = 0; }
            return Publish(ref _triedZ32, _kernZ32 != 0);
        }
    }

    /// <summary>
    /// Run C = A·B (FP64) with the machine kernel; returns false if the shape doesn't
    /// qualify. C must already be zeroed (the kernel accumulates); any m, n. Uses the AVX-512
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
        if (m <= 0 || n <= 0 || k <= 0) return false; // edges run on zero-padded panels (#1218)
        return RunPacked<double>(a, lda, b, ldb, c, ldc, m, n, k, Fp64Mr, nr, kern);
    }

    /// <summary>
    /// Run C = A·B (FP32) with the machine kernel; returns false if the shape doesn't
    /// qualify. C must already be zeroed; any m, n. Uses the AVX-512 6×32 kernel when active,
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
        if (m <= 0 || n <= 0 || k <= 0) return false; // edges run on zero-padded panels (#1218)
        // #653: the AVX2 6×16 path can drive a whole M-stripe (all N-tiles) per call via the
        // GEBP panel kernel; the AVX-512 6×32 path has no panel kernel yet (falls back to per-tile).
        nint panel = (!z && EnablePanelKernel && TryInitPanelFp32()) ? _kernPanel32 : 0;
        return RunPacked<float>(a, lda, b, ldb, c, ldc, m, n, k, Fp32Mr, nr, kern, panel);
    }

    /// <summary>
    /// Shared pack + parallel macro-loop. The <c>typeof(T)</c> branch is a JIT-time constant per
    /// instantiation (double/float) and folds away, so there's no per-call type test in the hot
    /// loop. C must be pre-zeroed (the kernel accumulates). Any m, n ≥ 1: a partial last M-stripe or
    /// N-tile runs the same kernel on a zero-padded panel into a scratch tile. Returns true.
    ///
    /// <para>
    /// #653 macro-blocking: per K-block (pc) the work is split over <b>N-panels</b> (the jc dimension),
    /// not the inner M-stripes. The M-stripe split had all cores hammer ONE shared packed-B panel,
    /// maximal cache contention (the BLIS many-core anti-pattern) that capped scaling at ~3.4× on 16
    /// cores. Each thread's B sub-panel is sized so tiles·Nr·Kc stays ~half-L2-resident, with block
    /// width chosen for ≥ <see cref="CpuParallelSettings.MaxDegreeOfParallelism"/> blocks of occupancy.
    /// Each thread writes a disjoint set of C columns, so the split is race-free and deterministic.
    /// </para>
    /// <para>
    /// #1218: packing runs in parallel too. It used to be serial on the calling thread (the Span pack
    /// API can't be captured by the lambda), which with the edge tails left 43% of a 1000³ FP64 GEMM
    /// and 70% of a 500³ one serial while the pool spun. A is packed by stripe groups in one dispatch
    /// per K-block (every N-panel reads all of it); each N-panel then packs its own B sub-panel inside
    /// the compute dispatch, so B is packed by the thread whose L2 it is about to be read from. The
    /// spans are rebuilt inside the lambdas from addresses captured as nint (the buffers stay pinned).
    /// </para>
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe bool RunPacked<T>(
        ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> b, int ldb, Span<T> c, int ldc,
        int m, int n, int k, int Mr, int Nr, nint kernelAddr, nint panelKernelAddr = 0) where T : unmanaged
    {
        int elem = sizeof(T);
        int mStripes = (m + Mr - 1) / Mr;   // the last one may be partial
        int mFull = m / Mr;
        int nTiles = (n + Nr - 1) / Nr;     // the last one may be partial
        int nFull = n / Nr;

        // N-panel block width (in Nr-tiles): enough blocks for full core occupancy, but capped so a
        // thread's packed-B sub-panel (tiles·Nr·Kc) stays ~half-L2-resident (private per thread).
        int maxDeg = Math.Max(1, CpuParallelSettings.MaxDegreeOfParallelism);
        int occTiles = Math.Max(1, (nTiles + maxDeg - 1) / maxDeg);       // ceil → ≥ maxDeg blocks
        int l2Tiles = Math.Max(1, (256 * 1024) / (KcBlock * Nr * elem));  // sub-panel ≤ ~half L2
        int tilesPerBlock = Math.Min(occTiles, l2Tiles);
        int jcBlocks = (nTiles + tilesPerBlock - 1) / tilesPerBlock;
        int aGroups = Math.Min(mStripes, maxDeg);

        var packA = ArrayPool<T>.Shared.Rent(mStripes * Mr * KcBlock); // [mStripes, Kc, Mr] (whole A panel per Kc)
        var packB = ArrayPool<T>.Shared.Rent(KcBlock * nTiles * Nr);   // [nTiles, Kc, Nr]   (whole B panel per Kc)
        try
        {
            fixed (T* aPtr = a)
            fixed (T* bPtr = b)
            fixed (T* cPtr = c)
            fixed (T* paBase = packA)
            fixed (T* pbBase = packB)
            {
                // Capture buffer addresses as nint: pointers can't be captured by the parallel body
                // lambdas, but the surrounding `fixed` keeps them pinned.
                nint aAddr = (nint)aPtr, bAddr = (nint)bPtr, paAddr = (nint)paBase, pbAddr = (nint)pbBase, cAddr = (nint)cPtr;
                int aLen = a.Length, bLen = b.Length;
                int mL = m, nL = n, ldaL = lda, ldbL = ldb, ldcL = ldc, MrL = Mr, NrL = Nr;
                int mStripesL = mStripes, mFullL = mFull, nTilesL = nTiles, nFullL = nFull, tpbL = tilesPerBlock, aGroupsL = aGroups, jcBlocksL = jcBlocks;
                nint panelAddr = panelKernelAddr; // FP32 AVX2 GEBP panel kernel (0 = per-tile path)

                for (int pc = 0; pc < k; pc += KcBlock)
                {
                    int ekcL = Math.Min(KcBlock, k - pc);
                    int pcL = pc;

                    // Pack A[:, pc:pc+ekc] by stripe groups; the partial last stripe is zero-padded.
                    CpuParallelSettings.ParallelForOrSerial(0, aGroupsL, (long)mL * ekcL, g =>
                    {
                        int s0 = (int)((long)g * mStripesL / aGroupsL);
                        int s1 = (int)((long)(g + 1) * mStripesL / aGroupsL);
                        int f1 = Math.Min(s1, mFullL);
                        T* src = (T*)aAddr;
                        T* dst = (T*)paAddr;
                        if (f1 > s0)
                        {
                            long srcOff = (long)s0 * MrL * ldaL + pcL;
                            Avx2Pack.PackA<T>(new ReadOnlySpan<T>(src + srcOff, (int)(aLen - srcOff)), ldaL, false,
                                new Span<T>(dst + (long)s0 * ekcL * MrL, (f1 - s0) * ekcL * MrL), mc: (f1 - s0) * MrL, kc: ekcL, MrL);
                        }
                        if (s1 > mFullL)
                            PackPartialStripe(src, ldaL, pcL, mFullL * MrL, mL - mFullL * MrL, dst + (long)mFullL * ekcL * MrL, ekcL, MrL);
                    }, deterministicSafe: true);

                    long flopsPerBlock = (long)tpbL * mStripesL * ekcL * MrL * NrL * 2;
                    CpuParallelSettings.ParallelForOrSerial(0, jcBlocksL, (long)jcBlocksL * flopsPerBlock, [MethodImpl(Hot)] (jb) =>
                    {
                        int t0 = jb * tpbL;
                        int t1 = Math.Min(t0 + tpbL, nTilesL);
                        int nt = t1 - t0;
                        if (nt <= 0) return;

                        // This N-panel's B sub-panel: tiles [t0, t1) of B[pc:pc+ekc, :], partial tile zero-padded.
                        long bOff = (long)pcL * ldbL;
                        Avx2Pack.PackBStripeRange<T>(new ReadOnlySpan<T>((T*)bAddr + bOff, (int)(bLen - bOff)), ldbL, false,
                            new Span<T>((T*)pbAddr + (long)t0 * ekcL * NrL, nt * ekcL * NrL), t0, nt, nL, ekcL, NrL);

                        int tFull = Math.Min(t1, nFullL); // tiles [t0, tFull) are whole; [tFull, t1) is the partial one
                        T* scratch = stackalloc T[MrL * NrL];
                        if (typeof(T) == typeof(double))
                        {
                            var kern = (delegate* unmanaged<double*, double*, double*, long, long, void>)kernelAddr;
                            double* pa = (double*)paAddr, pb = (double*)pbAddr, cb = (double*)cAddr;
                            for (int isr = 0; isr < mStripesL; isr++)
                            {
                                double* aS = pa + (long)isr * ekcL * MrL;
                                if (isr < mFullL)
                                {
                                    double* cR = cb + (long)(isr * MrL) * ldcL;
                                    for (int t = t0; t < tFull; t++)
                                        kern(aS, pb + (long)t * ekcL * NrL, cR + (long)t * NrL, ldcL, ekcL);
                                    for (int t = tFull; t < t1; t++)
                                        EdgeTile(kern, aS, pb + (long)t * ekcL * NrL, cb, ldcL, isr * MrL, t * NrL, MrL, NrL,
                                            MrL, nL - t * NrL, ekcL, (double*)scratch);
                                }
                                else
                                {
                                    for (int t = t0; t < t1; t++)
                                        EdgeTile(kern, aS, pb + (long)t * ekcL * NrL, cb, ldcL, isr * MrL, t * NrL, MrL, NrL,
                                            mL - isr * MrL, Math.Min(NrL, nL - t * NrL), ekcL, (double*)scratch);
                                }
                            }
                        }
                        else // float
                        {
                            var kern = (delegate* unmanaged<float*, float*, float*, long, long, void>)kernelAddr;
                            float* pa = (float*)paAddr, pb = (float*)pbAddr, cb = (float*)cAddr;
                            for (int isr = 0; isr < mStripesL; isr++)
                            {
                                float* aS = pa + (long)isr * ekcL * MrL;
                                if (isr < mFullL)
                                {
                                    float* cR = cb + (long)(isr * MrL) * ldcL;
                                    if (panelAddr != 0 && tFull > t0)
                                    {
                                        // One GEBP call drives all whole N-tiles of this M-stripe (loops N in machine code).
                                        var panel = (delegate* unmanaged<float*, float*, float*, long, long, long, void>)panelAddr;
                                        panel(aS, pb + (long)t0 * ekcL * NrL, cR + (long)t0 * NrL, ldcL, ekcL, tFull - t0);
                                    }
                                    else
                                    {
                                        for (int t = t0; t < tFull; t++)
                                            kern(aS, pb + (long)t * ekcL * NrL, cR + (long)t * NrL, ldcL, ekcL);
                                    }
                                    for (int t = tFull; t < t1; t++)
                                        EdgeTile(kern, aS, pb + (long)t * ekcL * NrL, cb, ldcL, isr * MrL, t * NrL, MrL, NrL,
                                            MrL, nL - t * NrL, ekcL, (float*)scratch);
                                }
                                else
                                {
                                    for (int t = t0; t < t1; t++)
                                        EdgeTile(kern, aS, pb + (long)t * ekcL * NrL, cb, ldcL, isr * MrL, t * NrL, MrL, NrL,
                                            mL - isr * MrL, Math.Min(NrL, nL - t * NrL), ekcL, (float*)scratch);
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

    // Packs rows [row0, row0 + rows) (rows < mr) of A[:, pc:pc+kc] as one stripe, zero-padding rows rows..mr-1.
    private static unsafe void PackPartialStripe<T>(T* a, int lda, int pc, int row0, int rows, T* packed, int kc, int mr)
        where T : unmanaged
    {
        for (int kk = 0; kk < kc; kk++)
        {
            T* dst = packed + (long)kk * mr;
            for (int r = 0; r < mr; r++)
                dst[r] = r < rows ? a[(long)(row0 + r) * lda + pc + kk] : default;
        }
    }

    // A tile the kernel can't write in place (rows < Mr or cols < Nr): run it on a scratch Mr×Nr tile loaded with the
    // valid part of C (the kernel accumulates; the padded rows/cols of A and B are zero), then copy the valid part back.
    private static unsafe void EdgeTile(delegate* unmanaged<double*, double*, double*, long, long, void> kern,
        double* aS, double* bT, double* c, int ldc, int row0, int col0, int mr, int nr, int rows, int cols, int kc, double* scratch)
    {
        LoadEdge(c, ldc, row0, col0, mr, nr, rows, cols, scratch);
        kern(aS, bT, scratch, nr, kc);
        StoreEdge(c, ldc, row0, col0, nr, rows, cols, scratch);
    }

    private static unsafe void EdgeTile(delegate* unmanaged<float*, float*, float*, long, long, void> kern,
        float* aS, float* bT, float* c, int ldc, int row0, int col0, int mr, int nr, int rows, int cols, int kc, float* scratch)
    {
        LoadEdge(c, ldc, row0, col0, mr, nr, rows, cols, scratch);
        kern(aS, bT, scratch, nr, kc);
        StoreEdge(c, ldc, row0, col0, nr, rows, cols, scratch);
    }

    private static unsafe void LoadEdge<T>(T* c, int ldc, int row0, int col0, int mr, int nr, int rows, int cols, T* scratch)
        where T : unmanaged
    {
        for (int r = 0; r < mr; r++)
            for (int j = 0; j < nr; j++)
                scratch[r * nr + j] = r < rows && j < cols ? c[(long)(row0 + r) * ldc + col0 + j] : default;
    }

    private static unsafe void StoreEdge<T>(T* c, int ldc, int row0, int col0, int nr, int rows, int cols, T* scratch)
        where T : unmanaged
    {
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < cols; j++)
                c[(long)(row0 + r) * ldc + col0 + j] = scratch[r * nr + j];
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
    internal static bool IsFp32MacroAvailable => false;
    internal static unsafe void RunMacroPanelFp32(
        float* aBase, float* bBase, float* cBase,
        int ldc, int kc, int njr, int numMr, int aStrideBytes) =>
        throw new PlatformNotSupportedException(
            "RunMacroPanelFp32 requires the net5.0+ machine-code JIT path; gate on IsFp32MacroAvailable before calling.");
    internal static void ResetFp32Kernels() { }
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
