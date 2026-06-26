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

    // Specialized-kernel cache: (Mr, nVec, lda, ldb, ldc, f64, bias, relu) -> native entry point.
    // ExecutableMemory is retained for process lifetime so the emitted pages stay mapped.
    private static readonly ConcurrentDictionary<(int Mr, int NVec, int Lda, int Ldb, int Ldc, bool F64, bool Bias, bool Relu), nint> s_cache = new();
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
        float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int n, int k,
        float* bias = null, bool relu = false)
    {
        if (!IsSupported || m <= 0 || n <= 0 || k <= 0) return false;
        if ((n & 7) != 0) return false; // AVX2 has no mask; partial-8 columns fall back to the caller
        bool hasBias = bias != null;

        for (int mi = 0; mi < m; mi += 6)
        {
            int mr = Math.Min(6, m - mi);
            int nj = 0;
            for (; nj + 16 <= n; nj += 16)
            {
                nint kern = GetOrEmit(mr, 2, lda, ldb, ldc, f64: false, hasBias, relu);
                if (kern == 0) return false;
                ((delegate* unmanaged<float*, float*, float*, long, float*, void>)kern)(
                    a + (long)mi * lda, b + nj, c + (long)mi * ldc + nj, k, hasBias ? bias + nj : null);
            }
            for (; nj + 8 <= n; nj += 8)
            {
                nint kern = GetOrEmit(mr, 1, lda, ldb, ldc, f64: false, hasBias, relu);
                if (kern == 0) return false;
                ((delegate* unmanaged<float*, float*, float*, long, float*, void>)kern)(
                    a + (long)mi * lda, b + nj, c + (long)mi * ldc + nj, k, hasBias ? bias + nj : null);
            }
        }
        return true;
    }

    /// <summary>C[m×n] := A[m×k]·B[k×n] (row-major, overwrite) via specialized JIT tiles, FP64.
    /// Returns false when N is not a multiple of 4 (no AVX2 mask), unsupported, or on emit failure.
    /// No fused epilogue (FP64 callers apply it managed).</summary>
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
                nint kern = GetOrEmit(mr, 2, lda, ldb, ldc, f64: true, hasBias: false, relu: false);
                if (kern == 0) return false;
                ((delegate* unmanaged<double*, double*, double*, long, double*, void>)kern)(
                    a + (long)mi * lda, b + nj, c + (long)mi * ldc + nj, k, null);
            }
            for (; nj + 4 <= n; nj += 4)
            {
                nint kern = GetOrEmit(mr, 1, lda, ldb, ldc, f64: true, hasBias: false, relu: false);
                if (kern == 0) return false;
                ((delegate* unmanaged<double*, double*, double*, long, double*, void>)kern)(
                    a + (long)mi * lda, b + nj, c + (long)mi * ldc + nj, k, null);
            }
        }
        return true;
    }

    private static nint GetOrEmit(int mr, int nVec, int lda, int ldb, int ldc, bool f64, bool hasBias, bool relu)
    {
        var key = (mr, nVec, lda, ldb, ldc, f64, hasBias, relu);
        if (s_cache.TryGetValue(key, out var p)) return p;
        lock (s_emitLock)
        {
            if (s_cache.TryGetValue(key, out p)) return p;
            nint entry = 0;
            try
            {
                byte[] code = IsWindows
                    ? EmitDirectTileWindows(mr, nVec, lda, ldb, ldc, f64, hasBias, relu)
                    : EmitDirectTileSysV(mr, nVec, lda, ldb, ldc, f64, hasBias, relu);
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

    private static byte[] EmitDirectTileWindows(int mr, int nVec, int lda, int ldb, int ldc, bool f64, bool hasBias, bool relu)
    {
        const int RSP = 4, R10 = 10;
        var asm = new X64Assembler();
        // Windows: rcx=A, rdx=B, r8=C, r9=K. The 5th arg (bias) is at [rsp+0x28] — read it into r10
        // (volatile, untouched by the body) BEFORE growing rsp. Save xmm6–15 (body uses ymm6–14).
        if (hasBias) asm.MovRegFromRsp(R10, 0x28);
        asm.SubRsp(0xA0);
        for (int i = 0; i < 10; i++) asm.VmovupsXmmStoreD32(RSP, i * 0x10, 6 + i);
        EmitDirectBody(asm, mr, nVec, lda, ldb, ldc, f64, hasBias, relu);
        for (int i = 0; i < 10; i++) asm.VmovupsXmmLoadD32(6 + i, RSP, i * 0x10);
        asm.AddRsp(0xA0);
        asm.Vzeroupper();
        asm.Ret();
        return asm.ToArray();
    }

    private static byte[] EmitDirectTileSysV(int mr, int nVec, int lda, int ldb, int ldc, bool f64, bool hasBias, bool relu)
    {
        const int RCX = 1, RDX = 2, R8 = 8, R9 = 9, R10 = 10, RSI = 6, RDI = 7;
        var asm = new X64Assembler();
        // SysV 5th arg (bias) is r8 — save it to r10 before r8 is reused for C in the shuffle.
        if (hasBias) asm.MovRegReg(R10, R8);
        // SysV: rdi=A, rsi=B, rdx=C, rcx=K → shuffle to rcx=A, rdx=B, r8=C, r9=K (no live clobber).
        asm.MovRegReg(R9, RCX);   // r9 = K
        asm.MovRegReg(R8, RDX);   // r8 = C
        asm.MovRegReg(RDX, RSI);  // rdx = B
        asm.MovRegReg(RCX, RDI);  // rcx = A
        // All ymm are volatile on SysV — no save needed.
        EmitDirectBody(asm, mr, nVec, lda, ldb, ldc, f64, hasBias, relu);
        asm.Vzeroupper();
        asm.Ret();
        return asm.ToArray();
    }

    /// <summary>Mr×(laneW·nVec) tile (laneW = 8 floats / 4 doubles), lda/ldb/ldc baked. rcx=A,
    /// rdx=B, r8=C, r9=K. Each ymm spans 32 B regardless of dtype, so the v·32 displacements and
    /// the +32 store strides are shared; only the element size, FMA, broadcast, and load/store
    /// opcodes differ between FP32 and FP64.</summary>
    private static void EmitDirectBody(X64Assembler asm, int mr, int nVec, int lda, int ldb, int ldc, bool f64, bool hasBias, bool relu)
    {
        const int RCX = 1, RDX = 2, R8 = 8, R9 = 9, R10 = 10;
        const int Bscr = 12, Ascr = 14; // ymm12..13 = B rows, ymm14 = A broadcast (free after K-loop)
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

        // Fused epilogue (FP32 only — FP64 callers never set these). r10 = bias ptr for this N-tile.
        // Reuses the now-free K-loop scratch (Bscr for the bias vectors, Ascr for the ReLU zero).
        if (hasBias)
        {
            for (int v = 0; v < nVec; v++) asm.VmovupsLoad(Bscr + v, R10, (sbyte)(v * 32)); // bias[nj + v·8]
            for (int i = 0; i < mr; i++)
                for (int v = 0; v < nVec; v++) asm.Vaddps(i * nVec + v, i * nVec + v, Bscr + v);
        }
        if (relu)
        {
            asm.Vxorps(Ascr, Ascr, Ascr); // 0.0
            for (int i = 0; i < mr; i++)
                for (int v = 0; v < nVec; v++) asm.Vmaxps(i * nVec + v, i * nVec + v, Ascr);
        }

        for (int i = 0; i < mr; i++)
            for (int v = 0; v < nVec; v++)
                if (f64) asm.VmovupdStoreD32(R8, i * ldcB + v * 32, i * nVec + v);
                else asm.VmovupsStoreD32(R8, i * ldcB + v * 32, i * nVec + v);
    }

    // ── #475 Phase 2: native low-precision GEMM folded into the JIT framework ───
    // These reuse the existing SDE-verified machine-code microkernels (vdpbf16ps for BF16,
    // AMX tdpb* for int8), which use instructions this AVX2 host lacks. They are perf-validatable
    // ONLY on real AVX-512-BF16 / AMX hardware (or for correctness under Intel SDE) — hence the
    // explicit opt-in flags (default OFF): auto-using them here would #UD. Ship opt-in; the
    // structural encoding is gated by MachineKernel*Tests + the `--verify-bf16gemm`/`--verify-amx-gemm`
    // SDE entrypoints, exactly like the existing AVX-512 kernels.

    /// <summary>Opt-in for the native AVX-512-BF16 GEMM path (default off; .NET exposes no
    /// Avx512Bf16 capability, and vdpbf16ps faults without it). Enable only on verified BF16 HW / SDE.</summary>
    internal static bool EnableBf16 { get; set; }

    /// <summary>Opt-in for the native AMX int8 GEMM path (default off; faults without AMX).</summary>
    internal static bool EnableAmxInt8 { get; set; }

    /// <summary>BF16 GEMM C[m,n] += Σ_k A[m,k]·B[k,n] (raw BF16 bits, FP32 accumulate) via the
    /// SDE-verified vdpbf16ps microkernel. No-op (returns false) unless <see cref="EnableBf16"/>.</summary>
    internal static unsafe bool TryRunBf16(ReadOnlySpan<ushort> a, ReadOnlySpan<ushort> b, Span<float> c, int m, int k, int n)
        => Enabled && EnableBf16 && Jit.MachineCodeBf16Kernel.TryGemm(a, b, c, m, k, n);
}
#else
/// <summary>net471 stub — no intrinsics / function pointers; callers fall back to the managed path.</summary>
internal static class JitGemmGenerator
{
    internal static bool Enabled { get; set; }
    internal static bool IsSupported => false;
    internal static bool EnableBf16 { get; set; }
    internal static bool EnableAmxInt8 { get; set; }
    internal static unsafe bool TryRunFp32(float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int n, int k, float* bias = null, bool relu = false) => false;
    internal static unsafe bool TryRunFp64(double* a, int lda, double* b, int ldb, double* c, int ldc, int m, int n, int k) => false;
    internal static bool TryRunBf16(System.ReadOnlySpan<ushort> a, System.ReadOnlySpan<ushort> b, System.Span<float> c, int m, int k, int n) => false;
}
#endif
