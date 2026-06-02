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
}
#endif
