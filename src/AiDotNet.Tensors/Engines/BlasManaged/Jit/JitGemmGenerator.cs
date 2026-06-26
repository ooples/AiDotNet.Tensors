using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

#if NET5_0_OR_GREATER

/// <summary>
/// #475 Phase 1 — libxsmm-style JIT GEMM generator. Emits per-shape-SPECIALIZED FP32 microkernels
/// that bake the tile geometry (Mr, Nr) and the leading dimensions (lda, ldb, ldc) as immediates
/// and address A/B/C directly (no packing), then caches each by signature. The structural win over
/// AOT OpenBLAS for small/medium GEMM: zero runtime stride arithmetic, no generic edge loops, and
/// a tile schedule fixed at emit time. Operates on row-major A[M×K], B[K×N], C[M×N]; computes
/// C := A·B (overwrite — accumulators zero-init). net5+ only (function pointers + W^X memory).
/// </summary>
internal static class JitGemmGenerator
{
    /// <summary>Master switch (default on when supported). A/B + safety override.</summary>
    internal static bool Enabled { get; set; } = true;

    // Specialized-kernel cache: (Mr, nVec, lda, ldb, ldc, f64) -> native entry point.
    // ExecutableMemory is retained for process lifetime so the emitted pages stay mapped.
    private static readonly ConcurrentDictionary<(int Mr, int NVec, int Lda, int Ldb, int Ldc, bool F64), nint> s_cache = new();
    private static readonly ConcurrentBag<ExecutableMemory> s_keepAlive = new();
    private static readonly object s_emitLock = new();

    private static bool IsWindows => RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

    internal static bool IsSupported =>
        Enabled && Fma.IsSupported && Avx2.IsSupported &&
        RuntimeInformation.ProcessArchitecture == Architecture.X64 &&
        (IsWindows || RuntimeInformation.IsOSPlatform(OSPlatform.Linux));

    /// <summary>
    /// C[m×n] := A[m×k]·B[k×n] (row-major, overwrite) via specialized JIT tiles. Returns false
    /// (caller falls back) when N is not a multiple of 8, when unsupported, or on any emit failure.
    /// Mr is tiled by 6, Nr by 16 then 8.
    /// </summary>
    internal static unsafe bool TryRunFp32(
        float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int n, int k)
    {
        if (!IsSupported || m <= 0 || n <= 0 || k <= 0) return false;
        if ((n & 7) != 0) return false; // AVX2 has no mask; partial-8 columns fall back to the caller

        for (int mi = 0; mi < m; mi += 6)
        {
            int mr = Math.Min(6, m - mi);
            int nj = 0;
            for (; nj + 16 <= n; nj += 16)
            {
                nint kern = GetOrEmit(mr, 2, lda, ldb, ldc, f64: false);
                if (kern == 0) return false;
                ((delegate* unmanaged<float*, float*, float*, long, void>)kern)(
                    a + (long)mi * lda, b + nj, c + (long)mi * ldc + nj, k);
            }
            for (; nj + 8 <= n; nj += 8)
            {
                nint kern = GetOrEmit(mr, 1, lda, ldb, ldc, f64: false);
                if (kern == 0) return false;
                ((delegate* unmanaged<float*, float*, float*, long, void>)kern)(
                    a + (long)mi * lda, b + nj, c + (long)mi * ldc + nj, k);
            }
        }
        return true;
    }

    /// <summary>C[m×n] := A[m×k]·B[k×n] (row-major, overwrite) via specialized JIT tiles, FP64.
    /// Returns false when N is not a multiple of 4 (no AVX2 mask), unsupported, or on emit failure.</summary>
    internal static unsafe bool TryRunFp64(
        double* a, int lda, double* b, int ldb, double* c, int ldc, int m, int n, int k)
    {
        if (!IsSupported || m <= 0 || n <= 0 || k <= 0) return false;
        if ((n & 3) != 0) return false; // YMM holds 4 doubles; partial-4 columns fall back

        for (int mi = 0; mi < m; mi += 6)
        {
            int mr = Math.Min(6, m - mi);
            int nj = 0;
            for (; nj + 8 <= n; nj += 8)
            {
                nint kern = GetOrEmit(mr, 2, lda, ldb, ldc, f64: true);
                if (kern == 0) return false;
                ((delegate* unmanaged<double*, double*, double*, long, void>)kern)(
                    a + (long)mi * lda, b + nj, c + (long)mi * ldc + nj, k);
            }
            for (; nj + 4 <= n; nj += 4)
            {
                nint kern = GetOrEmit(mr, 1, lda, ldb, ldc, f64: true);
                if (kern == 0) return false;
                ((delegate* unmanaged<double*, double*, double*, long, void>)kern)(
                    a + (long)mi * lda, b + nj, c + (long)mi * ldc + nj, k);
            }
        }
        return true;
    }

    private static nint GetOrEmit(int mr, int nVec, int lda, int ldb, int ldc, bool f64)
    {
        var key = (mr, nVec, lda, ldb, ldc, f64);
        if (s_cache.TryGetValue(key, out var p)) return p;
        lock (s_emitLock)
        {
            if (s_cache.TryGetValue(key, out p)) return p;
            nint entry = 0;
            try
            {
                byte[] code = IsWindows
                    ? EmitDirectTileWindows(mr, nVec, lda, ldb, ldc, f64)
                    : EmitDirectTileSysV(mr, nVec, lda, ldb, ldc, f64);
                var mem = ExecutableMemory.TryAllocate(code);
                if (mem is not null) { s_keepAlive.Add(mem); entry = mem.Pointer; }
            }
            catch { entry = 0; }
            s_cache[key] = entry;
            return entry;
        }
    }

    // ── Emitters ────────────────────────────────────────────────────────────
    // Canonical body register layout: rcx=A, rdx=B, r8=C, r9=K. Accumulators ymm0..(Mr*nVec-1)
    // (≤12), B-load scratch ymm12..(12+nVec-1), A-broadcast scratch ymm14. Mr≤6, nVec≤2.

    private static byte[] EmitDirectTileWindows(int mr, int nVec, int lda, int ldb, int ldc, bool f64)
    {
        const int RSP = 4;
        var asm = new X64Assembler();
        // Windows: rcx=A, rdx=B, r8=C, r9=K already in the canonical layout. Save xmm6–15 (the
        // body uses ymm6–14, whose low 128 bits are nonvolatile).
        asm.SubRsp(0xA0);
        for (int i = 0; i < 10; i++) asm.VmovupsXmmStoreD32(RSP, i * 0x10, 6 + i);
        EmitDirectBody(asm, mr, nVec, lda, ldb, ldc, f64);
        for (int i = 0; i < 10; i++) asm.VmovupsXmmLoadD32(6 + i, RSP, i * 0x10);
        asm.AddRsp(0xA0);
        asm.Vzeroupper();
        asm.Ret();
        return asm.ToArray();
    }

    private static byte[] EmitDirectTileSysV(int mr, int nVec, int lda, int ldb, int ldc, bool f64)
    {
        const int RCX = 1, RDX = 2, R8 = 8, R9 = 9, RSI = 6, RDI = 7;
        var asm = new X64Assembler();
        // SysV: rdi=A, rsi=B, rdx=C, rcx=K → shuffle to rcx=A, rdx=B, r8=C, r9=K (no live clobber).
        asm.MovRegReg(R9, RCX);   // r9 = K
        asm.MovRegReg(R8, RDX);   // r8 = C
        asm.MovRegReg(RDX, RSI);  // rdx = B
        asm.MovRegReg(RCX, RDI);  // rcx = A
        // All ymm are volatile on SysV — no save needed.
        EmitDirectBody(asm, mr, nVec, lda, ldb, ldc, f64);
        asm.Vzeroupper();
        asm.Ret();
        return asm.ToArray();
    }

    /// <summary>Mr×(laneW·nVec) tile (laneW = 8 floats / 4 doubles), lda/ldb/ldc baked. rcx=A,
    /// rdx=B, r8=C, r9=K. Each ymm spans 32 B regardless of dtype, so the v·32 displacements and
    /// the +32 store strides are shared; only the element size, FMA, broadcast, and load/store
    /// opcodes differ between FP32 and FP64.</summary>
    private static void EmitDirectBody(X64Assembler asm, int mr, int nVec, int lda, int ldb, int ldc, bool f64)
    {
        const int RCX = 1, RDX = 2, R8 = 8, R9 = 9;
        const int Bscr = 12, Ascr = 14; // ymm12..13 = B rows, ymm14 = A broadcast
        int esz = f64 ? 8 : 4;
        int ldaB = lda * esz, ldbB = ldb * esz, ldcB = ldc * esz;

        // Zero the accumulators (C := A·B overwrite). vxorps clears all bits → 0.0 for both dtypes.
        for (int i = 0; i < mr; i++)
            for (int v = 0; v < nVec; v++)
                asm.Vxorps(i * nVec + v, i * nVec + v, i * nVec + v);

        int done = asm.NewLabel();
        asm.TestRegSelf(R9);
        asm.JzLabel32(done);
        int loop = asm.NewLabel();
        asm.MarkLabel(loop);
        // One K-step: load the B row (nVec ymm), broadcast each A column element, FMA.
        for (int v = 0; v < nVec; v++)
            if (f64) asm.VmovupdLoad(Bscr + v, RDX, (sbyte)(v * 32)); else asm.VmovupsLoad(Bscr + v, RDX, (sbyte)(v * 32));
        for (int i = 0; i < mr; i++)
        {
            if (f64) asm.VbroadcastSdD32(Ascr, RCX, i * ldaB); else asm.VbroadcastSsD32(Ascr, RCX, i * ldaB);
            for (int v = 0; v < nVec; v++)
                if (f64) asm.Vfmadd231pd(i * nVec + v, Ascr, Bscr + v); else asm.Vfmadd231ps(i * nVec + v, Ascr, Bscr + v);
        }
        asm.AddRegImm8(RCX, (sbyte)esz);    // A col ptr += 1 element (k++)
        asm.AddRegImm32(RDX, ldbB);  // B row ptr += ldb elements (k++)
        asm.DecReg(R9);
        asm.JnzLabel32(loop);
        asm.MarkLabel(done);

        for (int i = 0; i < mr; i++)
            for (int v = 0; v < nVec; v++)
                if (f64) asm.VmovupdStoreD32(R8, i * ldcB + v * 32, i * nVec + v);
                else asm.VmovupsStoreD32(R8, i * ldcB + v * 32, i * nVec + v);
    }
}
#else
/// <summary>net471 stub — no intrinsics / function pointers; callers fall back to the managed path.</summary>
internal static class JitGemmGenerator
{
    internal static bool Enabled { get; set; }
    internal static bool IsSupported => false;
    internal static unsafe bool TryRunFp32(float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int n, int k) => false;
    internal static unsafe bool TryRunFp64(double* a, int lda, double* b, int ldb, double* c, int ldc, int m, int n, int k) => false;
}
#endif
