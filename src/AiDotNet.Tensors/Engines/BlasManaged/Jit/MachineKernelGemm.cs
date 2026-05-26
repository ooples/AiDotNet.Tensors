using System;
#if NET5_0_OR_GREATER
using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Sub-S (#409): routes qualifying FP64 GEMMs to the hand-emitted machine-code
/// 6×8 microkernel (~57 GFLOPS/core, ~95% of OpenBLAS — see
/// <see cref="MachineCodeFmaKernel"/>). First-party machine code; falls back to
/// the existing managed strategy for any shape that doesn't qualify, so it can
/// only add speed, never change results.
///
/// <para>
/// Strict qualification (everything else → caller's existing path):
/// FP64, x64 Windows, AVX2+FMA, dynamic code allowed, no transpose, no epilogue,
/// no pre-pack handle, and m % 6 == 0 && n % 8 == 0 (tile-aligned — tail handling
/// is a follow-up). The emitted kernel is built once and cached for the process.
/// </para>
///
/// <para>
/// Requires function pointers + x86 intrinsics, so the kernel path is net5.0+
/// only; on net471 <see cref="TryGemmFp64"/> always returns false and callers use
/// the managed strategy.
/// </para>
/// </summary>
internal static class MachineKernelGemm
{
    /// <summary>Master switch (default on). Set false to force the managed path everywhere.</summary>
    internal static bool Enabled { get; set; } = true;

#if NET5_0_OR_GREATER
    private const int Mr = 6, Nr = 8;
    private const int KcBlock = 256; // K-blocking so packed panels stay cache-resident.

    private static readonly object _lock = new();
    private static bool _initTried;
    private static ExecutableMemory? _mem;
    private static unsafe delegate* unmanaged<double*, double*, double*, long, long, void> _kernel;

    private static unsafe bool TryInitKernel()
    {
        if (_initTried) return _mem is not null;
        lock (_lock)
        {
            if (_initTried) return _mem is not null;
            _initTried = true;
            if (RuntimeInformation.OSArchitecture != Architecture.X64
                || !RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                || !Avx2.IsSupported || !Fma.IsSupported
                || !NativeAotDetector.IsDynamicCodeSupported)
                return false;
            try
            {
                byte[] code = MachineCodeFmaKernel.EmitFp64_6x8_PackedWindowsU4();
                _mem = ExecutableMemory.TryAllocate(code);
                if (_mem is not null)
                    _kernel = (delegate* unmanaged<double*, double*, double*, long, long, void>)_mem.Pointer;
            }
            catch { _mem = null; }
            return _mem is not null;
        }
    }

    /// <summary>
    /// Run C = A·B with the machine-code kernel when the shape qualifies; returns
    /// false (caller uses its existing path) otherwise. C must already be zeroed
    /// (the kernel accumulates), which <see cref="BlasManaged.Gemm{T}"/> guarantees.
    /// </summary>
    internal static unsafe bool TryGemmFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc,
        int m, int n, int k,
        bool hasEpilogue, bool hasPrePack)
    {
        if (!Enabled || transA || transB || hasEpilogue || hasPrePack) return false;
        if (m % Mr != 0 || n % Nr != 0 || m <= 0 || n <= 0 || k <= 0) return false;
        if (!TryInitKernel()) return false;

        int mStripes = m / Mr;
        int nStripes = n / Nr;
        var packA = ArrayPool<double>.Shared.Rent(m * KcBlock);  // [mStripes, Kc, Mr]
        var packB = ArrayPool<double>.Shared.Rent(KcBlock * n);  // [nStripes, Kc, Nr]
        try
        {
            fixed (double* cPtr = c)
            fixed (double* paBase = packA)
            fixed (double* pbBase = packB)
            {
                // Capture buffer addresses as nint — pointers can't be captured by the
                // parallel body lambda, but the surrounding `fixed` keeps them pinned
                // for the whole (serial-outer) K-loop including the parallel region.
                nint paAddr = (nint)paBase, pbAddr = (nint)pbBase, cAddr = (nint)cPtr;
                int ldcL = ldc, nStripesL = nStripes;

                for (int pc = 0; pc < k; pc += KcBlock)
                {
                    int ekc = Math.Min(KcBlock, k - pc);

                    // Pack the Kc-block: A[:, pc:pc+ekc] (all m rows), B[pc:pc+ekc, :] (all n cols).
                    Avx2Pack.PackA<double>(a.Slice(pc), lda, false,
                        new Span<double>(paBase, m * ekc), mc: m, kc: ekc, Mr);
                    Avx2Pack.PackB<double>(b.Slice(pc * ldb), ldb, false,
                        new Span<double>(pbBase, ekc * n), nc: n, kc: ekc, Nr);

                    // Parallelize over M-stripes: each handles 6 disjoint C rows, so
                    // there's no write contention. The pc loop stays serial-outer, so
                    // C accumulates correctly across K-blocks. Small shapes run inline
                    // (ParallelForOrSerial's grain-size gate) — no thread-dispatch cost.
                    int ekcL = ekc;
                    long stripeWork = (long)nStripesL * ekcL * Mr * Nr * 2;
                    CpuParallelSettings.ParallelForOrSerial(0, mStripes, (long)mStripes * stripeWork, isr =>
                    {
                        double* aStripe = (double*)paAddr + (long)isr * ekcL * Mr;
                        double* cRow = (double*)cAddr + (long)(isr * Mr) * ldcL;
                        double* pb = (double*)pbAddr;
                        for (int jsr = 0; jsr < nStripesL; jsr++)
                            _kernel(aStripe, pb + (long)jsr * ekcL * Nr, cRow + jsr * Nr, ldcL, ekcL);
                    });
                }
            }
            return true;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(packA);
            ArrayPool<double>.Shared.Return(packB);
        }
    }
#else
    /// <summary>net471 has no x86 intrinsics / function pointers — always defer to managed.</summary>
    internal static bool TryGemmFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc,
        int m, int n, int k,
        bool hasEpilogue, bool hasPrePack) => false;
#endif
}
