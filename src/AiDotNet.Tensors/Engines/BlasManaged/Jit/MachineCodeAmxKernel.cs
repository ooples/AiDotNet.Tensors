#if NET5_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.BlasManaged.Jit;

/// <summary>
/// #380 — AMX (Advanced Matrix Extensions) tile GEMM, Phase 1: emit and execute a single
/// <c>tdpbf16ps</c> tile contraction to verify the AMX instruction encodings
/// (<see cref="X64Assembler.Tdpbf16ps"/>, tile load/store/config) and the tensor-core dot-product
/// semantics.
///
/// <para>
/// .NET exposes no AMX intrinsics, and AMX additionally needs OS XSAVE state enablement on real
/// hardware. Because the body is raw machine code, <b>Intel SDE emulates the AMX tiles on any
/// host</b> (run under <c>sde -spr</c>) — including AVX2-only dev boxes — so the encodings and the
/// BF16 tile math are verifiable without Sapphire-Rapids hardware. On a host without AMX and
/// without SDE the instructions fault <c>#UD</c>; callers must only run these under SDE there.
/// </para>
///
/// <para>
/// A single AMX BF16 tile op computes <c>C[16,16] += A[16,32] · B[16,16]</c> in FP32, where B is
/// in the AMX row-pair "VNNI" layout: tile row r (r=0..15) holds, for each output column n,
/// the pair {B[2r,n], B[2r+1,n]} as two BF16 — so a B-tile row is N=16 columns × 2 BF16 = 64
/// bytes, encoding K=32 contraction depth across its 16 rows.
/// </para>
/// </summary>
internal static class MachineCodeAmxKernel
{
    // Windows x64 ABI integer arg registers used by the emitted kernels.
    private const int RAX = 0, RCX = 1, RDX = 2, R8 = 8, R9 = 9;

    // The one-tile shape (AMX maxima for BF16): M rows × K depth → N cols.
    internal const int TileM = 16, TileK = 32, TileN = 16;
    private const int RowBytes = 64; // A: K*2 ; B(VNNI): N*2*2=... N*4? see note — all three are 64 here.

    /// <summary>
    /// Build the 64-byte AMX tile-configuration block consumed by <c>ldtilecfg</c>:
    /// byte 0 = palette (1), byte 1 = start_row (0), bytes 16+2·t = colsb (bytes per row) for
    /// tile t, bytes 48+t = rows for tile t. <paramref name="tiles"/> is (rows, colsb) per tile,
    /// in tmm index order.
    /// </summary>
    internal static byte[] BuildTileConfig(params (int rows, int colsb)[] tiles)
    {
        var cfg = new byte[64];
        cfg[0] = 1; // palette_id = 1 (the only non-init palette)
        for (int t = 0; t < tiles.Length; t++)
        {
            cfg[16 + t * 2] = (byte)(tiles[t].colsb & 0xFF);
            cfg[16 + t * 2 + 1] = (byte)((tiles[t].colsb >> 8) & 0xFF);
            cfg[48 + t] = (byte)tiles[t].rows;
        }
        return cfg;
    }

    /// <summary>
    /// Emit <c>void tileGemm(ushort* aPacked, ushort* bVnni, float* c, byte* cfg)</c>:
    /// configure tiles, load A (tmm0) and the VNNI-packed B (tmm1), zero the C accumulator
    /// (tmm2), run <c>tdpbf16ps tmm2, tmm0, tmm1</c>, store C, and release the tiles. All three
    /// operands use a 64-byte row stride (passed in rax).
    /// </summary>
    internal static byte[] EmitTileGemmProbeWindows()
    {
        var asm = new X64Assembler();
        asm.Ldtilecfg(R9);                  // ldtilecfg [r9]
        asm.MovRegImm32(RAX, RowBytes);     // rax = 64 (row stride bytes)
        asm.TileloadD(0, RCX, RAX);         // tmm0 = A      [rcx + rax]
        asm.TileloadD(1, RDX, RAX);         // tmm1 = B(VNNI)[rdx + rax]
        asm.Tilezero(2);                    // tmm2 = 0      (C accumulator)
        // tdpbf16ps: the M-row operand (A) is ModRM.rm and the K-row operand (B) is VEX.vvvv —
        // so to compute C(tmm2) = A(tmm0)·B(tmm1), A goes in rm and B in vvvv.
        asm.Tdpbf16ps(2, /*vvvv=B*/1, /*rm=A*/0); // tmm2 += A · B (FP32 accumulate)
        asm.Tilestored(R8, RAX, 2);         // [r8 + rax] = tmm2
        asm.Tilerelease();
        asm.Ret();
        return asm.ToArray();
    }

    /// <summary>
    /// Diagnostic: emit <c>void ldst(ushort* src, ushort* dst, byte* cfg)</c> that configures one
    /// 16×32-BF16 tile, loads it from <c>src</c> (tmm0), and stores it straight back to <c>dst</c>
    /// — isolating tile config + load/store from the dot-product.
    /// </summary>
    internal static byte[] EmitTileLoadStoreProbeWindows()
    {
        var asm = new X64Assembler();
        asm.Ldtilecfg(R8);              // cfg ptr is 3rd arg → r8
        asm.MovRegImm32(RAX, RowBytes); // rax = 64
        asm.TileloadD(0, RCX, RAX);     // tmm0 = src
        asm.Tilestored(RDX, RAX, 0);    // dst = tmm0
        asm.Tilerelease();
        asm.Ret();
        return asm.ToArray();
    }

    /// <summary>Run the load/store roundtrip; <paramref name="src"/>/<paramref name="dst"/> are 16×32 BF16 (512 each).</summary>
    internal static unsafe bool TryRunLoadStore(ushort[] src, ushort[] dst)
    {
        byte[] code = EmitTileLoadStoreProbeWindows();
        byte[] cfg = BuildTileConfig((TileM, TileK * 2));
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null || mem.Pointer == IntPtr.Zero) return false;
        var fn = (delegate* unmanaged<ushort*, ushort*, byte*, void>)mem.Pointer;
        fixed (ushort* sp = src)
        fixed (ushort* dp = dst)
        fixed (byte* cfgp = cfg)
            fn(sp, dp, cfgp);
        return true;
    }

    /// <summary>
    /// Emit + execute the one-tile probe. <paramref name="aPacked"/> is 16×32 BF16 (row-major, 64
    /// bytes/row); <paramref name="bVnni"/> is the VNNI-packed B, 16×32 BF16 (row r = {B[2r,n],
    /// B[2r+1,n]} over n=0..15); <paramref name="c"/> receives 16×16 FP32. Returns false only if
    /// executable memory is unavailable. MUST run under Intel SDE on hosts lacking AMX.
    /// </summary>
    internal static unsafe bool TryRunTileProbe(ushort[] aPacked, ushort[] bVnni, float[] c)
    {
        if (aPacked.Length < TileM * TileK || bVnni.Length < TileM * (TileN * 2) || c.Length < TileM * TileN)
            throw new ArgumentException("AMX tile probe needs A 16×32, B(VNNI) 16×32 BF16, C 16×16 floats.");

        byte[] code = EmitTileGemmProbeWindows();
        byte[] cfg = BuildTileConfig((TileM, TileK * 2), (TileM, TileN * 2 * 2), (TileM, TileN * 4));
        // colsb: A=K*2=64 ; B(VNNI)=N*2 BF16 per row=64 ; C=N*4=64. (TileN*2*2 == TileN*4 == 64.)

        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null || mem.Pointer == IntPtr.Zero) return false;
        var fn = (delegate* unmanaged<ushort*, ushort*, float*, byte*, void>)mem.Pointer;

        fixed (ushort* ap = aPacked)
        fixed (ushort* bp = bVnni)
        fixed (float* cp = c)
        fixed (byte* cfgp = cfg)
            fn(ap, bp, cp, cfgp);
        return true;
    }

    // ── INT8 tile path (tdpbssd) + AMX CPUID detection ─────────────────────────

    internal const int TileK8 = 64; // INT8 K-depth per tile (4-deep VNNI: 4 int8 per group)

    // CPUID.(EAX=7,ECX=0):EDX feature bits for AMX.
    internal const uint AmxBf16Bit = 1u << 22, AmxTileBit = 1u << 24, AmxInt8Bit = 1u << 25;

    /// <summary>
    /// Emit <c>void int8Tile(sbyte* aPacked, sbyte* bVnni4, int* c, byte* cfg)</c>: a single
    /// <c>tdpbssd</c> (signed·signed int8 → int32) tile op. C[16,16] += A[16,64]·B[16,64], B in
    /// the AMX 4-deep VNNI layout (row g holds {B[4g+0,n]..B[4g+3,n]} for each column n).
    /// </summary>
    internal static byte[] EmitTileGemmProbeInt8Windows()
    {
        var asm = new X64Assembler();
        asm.Ldtilecfg(R9);
        asm.MovRegImm32(RAX, RowBytes); // 64-byte rows for all three tiles
        asm.TileloadD(0, RCX, RAX);     // tmm0 = A (int8)
        asm.TileloadD(1, RDX, RAX);     // tmm1 = B (VNNI-4 int8)
        asm.Tilezero(2);                // tmm2 = 0 (int32 accumulator)
        asm.Tdpbssd(2, /*vvvv=B*/1, /*rm=A*/0); // tmm2 += A · B
        asm.Tilestored(R8, RAX, 2);
        asm.Tilerelease();
        asm.Ret();
        return asm.ToArray();
    }

    /// <summary>Run the INT8 tile probe. A is 16×64 int8, bVnni4 is 16×64 int8 (VNNI-4), c is 16×16 int32.</summary>
    internal static unsafe bool TryRunTileProbeInt8(sbyte[] aPacked, sbyte[] bVnni4, int[] c)
    {
        if (aPacked.Length < TileM * TileK8 || bVnni4.Length < TileM * (TileN * 4) || c.Length < TileM * TileN)
            throw new ArgumentException("AMX INT8 tile probe needs A 16×64, B(VNNI4) 16×64 int8, C 16×16 int32.");

        byte[] code = EmitTileGemmProbeInt8Windows();
        byte[] cfg = BuildTileConfig((TileM, TileK8), (TileM, TileN * 4), (TileM, TileN * 4));
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null || mem.Pointer == IntPtr.Zero) return false;
        var fn = (delegate* unmanaged<sbyte*, sbyte*, int*, byte*, void>)mem.Pointer;
        fixed (sbyte* ap = aPacked)
        fixed (sbyte* bp = bVnni4)
        fixed (int* cp = c)
        fixed (byte* cfgp = cfg)
            fn(ap, bp, cp, cfgp);
        return true;
    }

    /// <summary>
    /// Emit <c>void cpuid7Edx(uint* outEdx)</c>: query CPUID leaf 7, sub-leaf 0 and write EDX to
    /// <paramref name="outEdx"/> — the AMX-TILE/AMX-BF16/AMX-INT8 feature bits. rbx is preserved
    /// (non-volatile on Win x64); the out pointer is stashed in r9 before <c>cpuid</c> clobbers rcx.
    /// </summary>
    internal static byte[] EmitCpuidLeaf7EdxWindows()
    {
        var asm = new X64Assembler();
        asm.PushReg(3);              // push rbx
        asm.MovRegReg(R9, RCX);      // r9 = outEdx (saved before cpuid clobbers rcx)
        asm.MovRegImm32(0, 7);       // eax = 7  (leaf)
        asm.MovRegImm32(1, 0);       // ecx = 0  (sub-leaf)
        asm.Cpuid();
        asm.MovMemFromReg32(R9, 0, 2); // [r9] = edx
        asm.PopReg(3);               // pop rbx
        asm.Ret();
        return asm.ToArray();
    }

    /// <summary>
    /// Read CPUID.(7,0):EDX via emitted machine code (no .NET intrinsic exposes these bits). Returns
    /// false only if executable memory is unavailable; otherwise <paramref name="edx"/> holds the
    /// raw feature dword — test against <see cref="AmxTileBit"/>/<see cref="AmxBf16Bit"/>/
    /// <see cref="AmxInt8Bit"/>. NOTE: a true result here means the CPU advertises AMX, but real
    /// use also needs OS XSAVE state enablement (Linux 5.16+ XFD); gate production on both.
    /// </summary>
    internal static unsafe bool TryGetAmxCpuidEdx(out uint edx)
    {
        edx = 0;
        byte[] code = EmitCpuidLeaf7EdxWindows();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null || mem.Pointer == IntPtr.Zero) return false;
        var fn = (delegate* unmanaged<uint*, void>)mem.Pointer;
        uint local;
        fn(&local);
        edx = local;
        return true;
    }

    // ── Phase 2: full tiled AMX BF16 GEMM ──────────────────────────────────────

    /// <summary>
    /// Emit <c>void tileMac(ushort* aTile, ushort* bTile, float* cTile, byte* cfg)</c>: a single
    /// accumulating tile op — load the current C tile (tmm2), the A tile (tmm0) and the VNNI B
    /// tile (tmm1), run <c>C += A·B</c>, and store C back. The driver clears <c>cTile</c> before a
    /// K-tile sweep and calls this once per K-tile, so accumulation happens in the C buffer via
    /// the load-accumulate-store roundtrip (HW FP32 accumulate inside the tile).
    /// </summary>
    internal static byte[] EmitTileMacWindows()
    {
        var asm = new X64Assembler();
        asm.Ldtilecfg(R9);              // cfg → r9 (4th arg)
        asm.MovRegImm32(RAX, RowBytes); // rax = 64
        asm.TileloadD(2, R8, RAX);      // tmm2 = current C partial   (r8 = cTile)
        asm.TileloadD(0, RCX, RAX);     // tmm0 = A
        asm.TileloadD(1, RDX, RAX);     // tmm1 = B (VNNI)
        asm.Tdpbf16ps(2, /*vvvv=B*/1, /*rm=A*/0); // tmm2 += A · B
        asm.Tilestored(R8, RAX, 2);     // cTile = tmm2
        asm.Tilerelease();
        asm.Ret();
        return asm.ToArray();
    }

    /// <summary>
    /// Tiled AMX BF16 GEMM: <c>C[m,n] = Σ_k A[m,k]·B[k,n]</c> with A (M×K) and B (K×N) as raw BF16
    /// bit patterns, row-major, and C (M×N) accumulated in FP32. Tiles M by 16, N by 16, K by 32;
    /// packs each A sub-tile row-major and each B sub-tile into the AMX row-pair VNNI layout,
    /// zero-padding ragged M/K/N edges (BF16 0x0000 = +0.0). Returns false only if executable
    /// memory is unavailable. MUST run under Intel SDE on hosts lacking AMX.
    /// </summary>
    internal static unsafe bool TryGemm(
        ReadOnlySpan<ushort> a, ReadOnlySpan<ushort> b, Span<float> c, int m, int k, int n)
    {
        if (m <= 0 || k <= 0 || n <= 0) return true;
        if (a.Length < (long)m * k || b.Length < (long)k * n || c.Length < (long)m * n)
            throw new ArgumentException("AMX GEMM: a/b/c spans smaller than M·K / K·N / M·N.");

        byte[] code = EmitTileMacWindows();
        byte[] cfg = BuildTileConfig((TileM, TileK * 2), (TileM, TileN * 4), (TileM, TileN * 4));
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null || mem.Pointer == IntPtr.Zero) return false;
        var fn = (delegate* unmanaged<ushort*, ushort*, float*, byte*, void>)mem.Pointer;

        var aTile = new ushort[TileM * TileK];        // 16×32 BF16
        var bTile = new ushort[(TileK / 2) * (TileN * 2)]; // 16×32 BF16 (VNNI)
        var cTile = new float[TileM * TileN];         // 16×16 FP32

        fixed (ushort* atp = aTile)
        fixed (ushort* btp = bTile)
        fixed (float* ctp = cTile)
        fixed (byte* cfgp = cfg)
        {
            for (int m0 = 0; m0 < m; m0 += TileM)
            {
                int mRows = Math.Min(TileM, m - m0);
                for (int n0 = 0; n0 < n; n0 += TileN)
                {
                    int nCols = Math.Min(TileN, n - n0);
                    Array.Clear(cTile, 0, cTile.Length);

                    for (int k0 = 0; k0 < k; k0 += TileK)
                    {
                        int kCols = Math.Min(TileK, k - k0);

                        // Pack A sub-tile: [mRows][kCols] row-major, zero-padded to 16×32.
                        Array.Clear(aTile, 0, aTile.Length);
                        for (int mr = 0; mr < mRows; mr++)
                            for (int kr = 0; kr < kCols; kr++)
                                aTile[mr * TileK + kr] = a[(m0 + mr) * k + k0 + kr];

                        // Pack B sub-tile (VNNI): row r holds {B[k0+2r,n], B[k0+2r+1,n]} over n.
                        Array.Clear(bTile, 0, bTile.Length);
                        for (int r = 0; r < TileK / 2; r++)
                        {
                            int kk0 = 2 * r, kk1 = kk0 + 1;
                            for (int nc = 0; nc < nCols; nc++)
                            {
                                if (kk0 < kCols) bTile[r * (TileN * 2) + 2 * nc] = b[(k0 + kk0) * n + n0 + nc];
                                if (kk1 < kCols) bTile[r * (TileN * 2) + 2 * nc + 1] = b[(k0 + kk1) * n + n0 + nc];
                            }
                        }

                        fn(atp, btp, ctp, cfgp); // cTile += aTile · bTile
                    }

                    for (int mr = 0; mr < mRows; mr++)
                        for (int nc = 0; nc < nCols; nc++)
                            c[(m0 + mr) * n + n0 + nc] = cTile[mr * TileN + nc];
                }
            }
        }
        return true;
    }
}
#endif
