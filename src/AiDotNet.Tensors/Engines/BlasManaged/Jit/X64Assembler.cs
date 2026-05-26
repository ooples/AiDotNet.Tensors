using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Sub-S (#409): a tiny, focused x86-64 + AVX2/FMA machine-code encoder — just
/// the instructions the hand-emitted GEMM microkernels need. First-party code:
/// we encode the bytes ourselves and run them via <see cref="ExecutableMemory"/>,
/// with FULL register control (all 16 YMM), which is what lets us exceed RyuJIT's
/// ~8-accumulator ceiling and approach the hardware FMA peak. No third-party
/// dependency. Windows x64 + System V are both straight x86-64; only the
/// arg-register mapping differs (handled by the kernel emitter, not here).
/// </summary>
internal sealed class X64Assembler
{
    private readonly List<byte> _code = new(256);
    private readonly Dictionary<int, int> _labelPos = new();      // label id -> byte offset
    private readonly List<(int pos, int label)> _rel8Fixups = new();
    private readonly List<(int pos, int label)> _rel32Fixups = new();

    internal byte[] ToArray()
    {
        foreach (var (pos, label) in _rel8Fixups)
        {
            int rel = _labelPos[label] - (pos + 1); // rel8: relative to byte AFTER the displacement
            if (rel < sbyte.MinValue || rel > sbyte.MaxValue)
                throw new InvalidOperationException("rel8 jump out of range");
            _code[pos] = (byte)(sbyte)rel;
        }
        foreach (var (pos, label) in _rel32Fixups)
        {
            int rel = _labelPos[label] - (pos + 4); // rel32: relative to byte AFTER the 4-byte displacement
            _code[pos] = (byte)rel; _code[pos + 1] = (byte)(rel >> 8);
            _code[pos + 2] = (byte)(rel >> 16); _code[pos + 3] = (byte)(rel >> 24);
        }
        return _code.ToArray();
    }

    private void B(params byte[] bytes) => _code.AddRange(bytes);
    private void Imm32(int v) { _code.Add((byte)v); _code.Add((byte)(v >> 8)); _code.Add((byte)(v >> 16)); _code.Add((byte)(v >> 24)); }

    // ── Labels / branches ────────────────────────────────────────────────────
    private int _nextLabel;
    internal int NewLabel() => _nextLabel++; // genuinely unique, monotonic
    internal void MarkLabel(int label) => _labelPos[label] = _code.Count;

    /// <summary>dec rcx (REX.W FF /1).</summary>
    internal void DecRcx() => B(0x48, 0xFF, 0xC9);

    /// <summary>dec reg64 (REX.W[.B] FF /1).</summary>
    internal void DecReg(int reg)
    {
        byte rex = (byte)(0x48 | (reg >= 8 ? 1 : 0));
        B(rex, 0xFF, (byte)(0xC8 | (reg & 7))); // /1: reg field=001 -> 0xC8 | rm
    }

    /// <summary>jnz rel8 to a (backward) label.</summary>
    internal void JnzLabel(int label)
    {
        B(0x75);                                   // jnz rel8
        _rel8Fixups.Add((_code.Count, label));     // displacement byte position
        _code.Add(0);                              // placeholder
    }

    /// <summary>jz rel8 to a (forward) label.</summary>
    internal void JzLabel(int label)
    {
        B(0x74);                                   // jz rel8
        _rel8Fixups.Add((_code.Count, label));
        _code.Add(0);
    }

    /// <summary>test reg64, reg64 (sets ZF if reg==0). REX.W[.R.B] 85 /r.</summary>
    internal void TestRegSelf(int reg)
    {
        byte rex = (byte)(0x48 | (reg >= 8 ? 4 : 0) | (reg >= 8 ? 1 : 0));
        B(rex, 0x85, (byte)(0xC0 | ((reg & 7) << 3) | (reg & 7)));
    }

    /// <summary>cmp reg64, imm8 (sign-extended). REX.W[.B] 83 /7 ib.</summary>
    internal void CmpRegImm8(int reg, sbyte imm)
    {
        byte rex = (byte)(0x48 | (reg >= 8 ? 1 : 0));
        B(rex, 0x83, (byte)(0xF8 | (reg & 7)), (byte)imm);
    }

    /// <summary>sub reg64, imm8 (sign-extended). REX.W[.B] 83 /5 ib.</summary>
    internal void SubRegImm8(int reg, sbyte imm)
    {
        byte rex = (byte)(0x48 | (reg >= 8 ? 1 : 0));
        B(rex, 0x83, (byte)(0xE8 | (reg & 7)), (byte)imm);
    }

    /// <summary>jmp rel32 to a label (any distance). E9 cd.</summary>
    internal void JmpLabel32(int label)
    {
        B(0xE9);
        _rel32Fixups.Add((_code.Count, label));
        B(0, 0, 0, 0);
    }

    /// <summary>jl rel32 to a label (signed less-than). 0F 8C cd.</summary>
    internal void JlLabel32(int label)
    {
        B(0x0F, 0x8C);
        _rel32Fixups.Add((_code.Count, label));
        B(0, 0, 0, 0);
    }

    /// <summary>jnz rel32 (any distance). 0F 85 cd. For loop bodies too big for rel8.</summary>
    internal void JnzLabel32(int label)
    {
        B(0x0F, 0x85);
        _rel32Fixups.Add((_code.Count, label));
        B(0, 0, 0, 0);
    }

    /// <summary>jz rel32 (any distance). 0F 84 cd.</summary>
    internal void JzLabel32(int label)
    {
        B(0x0F, 0x84);
        _rel32Fixups.Add((_code.Count, label));
        B(0, 0, 0, 0);
    }

    // VEX memory form with disp32 (mod=10): [base + disp32].
    private void VexMemDisp32(int map, int pp, int w, int L, byte opcode, int reg, int baseReg, int disp32)
    {
        int rBit = (reg < 8) ? 1 : 0;
        int bBit = (baseReg < 8) ? 1 : 0;
        byte b1 = (byte)((rBit << 7) | (1 << 6) | (bBit << 5) | map);
        byte b2 = (byte)((w << 7) | (0xF << 3) | (L << 2) | pp);
        byte modrm = (byte)(0x80 | ((reg & 7) << 3) | (baseReg & 7)); // mod=10
        B(0xC4, b1, b2, opcode, modrm);
        if ((baseReg & 7) == 4) _code.Add(0x24); // SIB for rsp/r12
        Imm32(disp32);
    }

    /// <summary>vmovupd ymm_dst, [base+disp32] (load).</summary>
    internal void VmovupdLoadD32(int dst, int baseReg, int disp32) => VexMemDisp32(Map0F, Pp66, 0, 1, 0x10, dst, baseReg, disp32);

    /// <summary>vbroadcastsd ymm_dst, [base+disp32].</summary>
    internal void VbroadcastSdD32(int dst, int baseReg, int disp32) => VexMemDisp32(Map0F38, Pp66, 0, 1, 0x19, dst, baseReg, disp32);

    internal void Ret() => B(0xC3);
    internal void Vzeroupper() => B(0xC5, 0xF8, 0x77);

    /// <summary>add reg64, imm8 (sign-extended). REX.W[.B] 83 /0 ib.</summary>
    internal void AddRegImm8(int reg, sbyte imm)
    {
        byte rex = (byte)(0x48 | (reg >= 8 ? 1 : 0));
        B(rex, 0x83, (byte)(0xC0 | (reg & 7)), (byte)imm);
    }

    /// <summary>add reg64, imm32. REX.W[.B] 81 /0 id.</summary>
    internal void AddRegImm32(int reg, int imm)
    {
        byte rex = (byte)(0x48 | (reg >= 8 ? 1 : 0));
        B(rex, 0x81, (byte)(0xC0 | (reg & 7)));
        Imm32(imm);
    }

    /// <summary>add dst64, src64 (dst += src). REX.W[.R.B] 01 /r.</summary>
    internal void AddRegReg(int dst, int src)
    {
        byte rex = (byte)(0x48 | (src >= 8 ? 4 : 0) | (dst >= 8 ? 1 : 0));
        B(rex, 0x01, (byte)(0xC0 | ((src & 7) << 3) | (dst & 7)));
    }

    /// <summary>mov dst64, src64. REX.W[.R.B] 89 /r.</summary>
    internal void MovRegReg(int dst, int src)
    {
        byte rex = (byte)(0x48 | (src >= 8 ? 4 : 0) | (dst >= 8 ? 1 : 0));
        B(rex, 0x89, (byte)(0xC0 | ((src & 7) << 3) | (dst & 7)));
    }

    /// <summary>mov dst64, [rsp+disp8]. REX.W[.R] 8B /r, mod=01, rm=100 (SIB rsp), disp8.</summary>
    internal void MovRegFromRsp(int dst, sbyte disp)
    {
        byte rex = (byte)(0x48 | (dst >= 8 ? 4 : 0));
        B(rex, 0x8B, (byte)(0x40 | ((dst & 7) << 3) | 4), 0x24, (byte)disp);
    }

    /// <summary>lea rax, [idx*8]. REX.W[.X] 8D, mod=00 reg=rax rm=100, SIB(scale8,idx,base=disp32), disp32=0.</summary>
    internal void LeaRaxIndexScale8(int idx)
    {
        byte rex = (byte)(0x48 | (idx >= 8 ? 2 : 0));
        byte sib = (byte)((3 << 6) | ((idx & 7) << 3) | 5);
        B(rex, 0x8D, 0x04, sib);
        Imm32(0);
    }

    /// <summary>sub rsp, imm32 (REX.W 81 /5).</summary>
    internal void SubRsp(int imm) { B(0x48, 0x81, 0xEC); Imm32(imm); }
    /// <summary>add rsp, imm32 (REX.W 81 /0).</summary>
    internal void AddRsp(int imm) { B(0x48, 0x81, 0xC4); Imm32(imm); }

    // ── VEX register-register form ────────────────────────────────────────────
    // VEX 3-byte: C4 [R X B mmmmm] [W vvvv L pp] opcode  ModRM(mod=11,reg,rm)
    //   R = ~reg[3], X = 1, B = ~rm[3]; vvvv = ~vvvvReg (4-bit); pp/map/L/W per op.
    private void VexRR(int map, int pp, int w, int L, byte opcode, int reg, int vvvv, int rm)
    {
        int rBit = (reg < 8) ? 1 : 0;
        int bBit = (rm < 8) ? 1 : 0;
        byte b1 = (byte)((rBit << 7) | (1 << 6) | (bBit << 5) | map);
        int vv = (~vvvv) & 0xF;
        byte b2 = (byte)((w << 7) | (vv << 3) | (L << 2) | pp);
        byte modrm = (byte)(0xC0 | ((reg & 7) << 3) | (rm & 7));
        B(0xC4, b1, b2, opcode, modrm);
    }

    // ── VEX memory form (mod=01, disp8): [base + disp8] ────────────────────────
    private void VexMemDisp8(int map, int pp, int w, int L, byte opcode, int reg, int baseReg, sbyte disp8)
    {
        int rBit = (reg < 8) ? 1 : 0;
        int bBit = (baseReg < 8) ? 1 : 0;
        byte b1 = (byte)((rBit << 7) | (1 << 6) | (bBit << 5) | map);
        // vvvv unused for these (set to 1111 -> ~0).
        byte b2 = (byte)((w << 7) | (0xF << 3) | (L << 2) | pp);
        byte modrm = (byte)(0x40 | ((reg & 7) << 3) | (baseReg & 7)); // mod=01
        B(0xC4, b1, b2, opcode, modrm);
        // SIB needed if base is rsp/r12 (rm==100). rsp = 4. Emit SIB 0x24 (base=rsp,no index).
        if ((baseReg & 7) == 4) _code.Add(0x24);
        _code.Add((byte)disp8);
    }

    // pp: 0=none, 1=66, 2=F3, 3=F2.  map: 1=0F, 2=0F38, 3=0F3A.  L: 0=128,1=256.
    private const int Pp66 = 1, PpNone = 0, Map0F = 1, Map0F38 = 2;

    /// <summary>vfmadd231pd dst, s1, s2  (dst += s1*s2). VEX.DDS.256.66.0F38.W1 B8.</summary>
    internal void Vfmadd231pd(int dst, int s1, int s2) => VexRR(Map0F38, Pp66, 1, 1, 0xB8, dst, s1, s2);

    /// <summary>vxorpd dst, s1, s2. VEX.256.66.0F.WIG 57. (zero a reg: dst=s1=s2)</summary>
    internal void Vxorpd(int dst, int s1, int s2) => VexRR(Map0F, Pp66, 0, 1, 0x57, dst, s1, s2);

    /// <summary>vaddpd dst, s1, s2. VEX.256.66.0F.WIG 58.</summary>
    internal void Vaddpd(int dst, int s1, int s2) => VexRR(Map0F, Pp66, 0, 1, 0x58, dst, s1, s2);

    /// <summary>vbroadcastsd ymm_dst, [base+disp8]. VEX.256.66.0F38.W0 19.</summary>
    internal void VbroadcastSd(int dst, int baseReg, sbyte disp8) => VexMemDisp8(Map0F38, Pp66, 0, 1, 0x19, dst, baseReg, disp8);

    /// <summary>vmovupd ymm_dst, [base+disp8] (load). VEX.256.66.0F.WIG 10.</summary>
    internal void VmovupdLoad(int dst, int baseReg, sbyte disp8) => VexMemDisp8(Map0F, Pp66, 0, 1, 0x10, dst, baseReg, disp8);

    /// <summary>vmovupd [base+disp8], ymm_src (store). VEX.256.66.0F.WIG 11.</summary>
    internal void VmovupdStore(int baseReg, sbyte disp8, int src) => VexMemDisp8(Map0F, Pp66, 0, 1, 0x11, src, baseReg, disp8);

    /// <summary>vmovups xmm_dst, [base+disp8] (128-bit load, for xmm save/restore). VEX.128.0F.WIG 10.</summary>
    internal void VmovupsXmmLoad(int dst, int baseReg, sbyte disp8) => VexMemDisp8(Map0F, PpNone, 0, 0, 0x10, dst, baseReg, disp8);

    /// <summary>vmovups [base+disp8], xmm_src (128-bit store). VEX.128.0F.WIG 11.</summary>
    internal void VmovupsXmmStore(int baseReg, sbyte disp8, int src) => VexMemDisp8(Map0F, PpNone, 0, 0, 0x11, src, baseReg, disp8);

    /// <summary>vmovups xmm_dst, [base+disp32] (128-bit load, disp32 form for offsets ≥ 0x80).</summary>
    internal void VmovupsXmmLoadD32(int dst, int baseReg, int disp32) => VexMemDisp32(Map0F, PpNone, 0, 0, 0x10, dst, baseReg, disp32);

    /// <summary>vmovups [base+disp32], xmm_src (128-bit store, disp32 form for offsets ≥ 0x80).</summary>
    internal void VmovupsXmmStoreD32(int baseReg, int disp32, int src) => VexMemDisp32(Map0F, PpNone, 0, 0, 0x11, src, baseReg, disp32);

    // ── FP32 (ps) AVX2/FMA forms ───────────────────────────────────────────────
    // Differences from the pd forms: vbroadcastss is opcode 0x18 (vs 0x19),
    // vfmadd231ps uses W0 (vs W1), and vmovups has no 66-prefix (pp=None, vs Pp66).

    /// <summary>vfmadd231ps dst, s1, s2 (dst += s1*s2). VEX.DDS.256.66.0F38.W0 B8.</summary>
    internal void Vfmadd231ps(int dst, int s1, int s2) => VexRR(Map0F38, Pp66, 0, 1, 0xB8, dst, s1, s2);

    /// <summary>vbroadcastss ymm_dst, [base+disp8]. VEX.256.66.0F38.W0 18.</summary>
    internal void VbroadcastSs(int dst, int baseReg, sbyte disp8) => VexMemDisp8(Map0F38, Pp66, 0, 1, 0x18, dst, baseReg, disp8);

    /// <summary>vmovups ymm_dst, [base+disp8] (256-bit load). VEX.256.0F.WIG 10.</summary>
    internal void VmovupsLoad(int dst, int baseReg, sbyte disp8) => VexMemDisp8(Map0F, PpNone, 0, 1, 0x10, dst, baseReg, disp8);

    /// <summary>vmovups [base+disp8], ymm_src (256-bit store). VEX.256.0F.WIG 11.</summary>
    internal void VmovupsStore(int baseReg, sbyte disp8, int src) => VexMemDisp8(Map0F, PpNone, 0, 1, 0x11, src, baseReg, disp8);

    /// <summary>lea rax, [idx*4] (FP32 row stride bytes = ldc*4). SIB scale=2.</summary>
    internal void LeaRaxIndexScale4(int idx)
    {
        byte rex = (byte)(0x48 | (idx >= 8 ? 2 : 0));
        byte sib = (byte)((2 << 6) | ((idx & 7) << 3) | 5);
        B(rex, 0x8D, 0x04, sib);
        Imm32(0);
    }

    // ── AVX-512 (EVEX) forms ────────────────────────────────────────────────
    // 4-byte EVEX prefix: 62 P0 P1 P2.
    //   P0 = [R̄ X̄ B̄ R̄'] [0 0] [mm]   (RXB + R' extend reg/rm to 4th/5th bit, all inverted)
    //   P1 = [W] [v̄v̄v̄v̄] [1] [pp]     (W, NDS vvvv inverted, fixed 1, prefix pp)
    //   P2 = [z] [L'L] [b] [V̄'] [aaa]  (z=merge/zero, L'L=length, b=broadcast, V' = vvvv[4] inverted, mask)
    // Used here only for zmm0–15 / GP base regs 0–15, no mask, no broadcast, 512-bit,
    // disp32 (mod=10, raw byte displacement — avoids EVEX's scaled disp8 entirely).

    // EVEX register-register (mod=11): dst = reg, v = NDS source, rm = the other source.
    private void EvexRR(int map, int pp, int w, byte opcode, int reg, int v, int rm)
    {
        int R = ((reg >> 3) & 1) ^ 1;
        int Rp = ((reg >> 4) & 1) ^ 1;
        int X = ((rm >> 4) & 1) ^ 1;
        int Bb = ((rm >> 3) & 1) ^ 1;
        byte p0 = (byte)((R << 7) | (X << 6) | (Bb << 5) | (Rp << 4) | (map & 3));
        int vvvv = (~v) & 0xF;
        int Vp = ((v >> 4) & 1) ^ 1;
        byte p1 = (byte)((w << 7) | (vvvv << 3) | (1 << 2) | (pp & 3));
        byte p2 = (byte)((2 << 5) | (Vp << 3)); // L'L=10 (512), z=0, b=0, aaa=000
        byte modrm = (byte)(0xC0 | ((reg & 7) << 3) | (rm & 7));
        B(0x62, p0, p1, p2, opcode, modrm);
    }

    // EVEX memory (mod=10, disp32, no SIB index): reg op [base + disp32].
    private void EvexMemDisp32(int map, int pp, int w, byte opcode, int reg, int baseReg, int disp32)
    {
        int R = ((reg >> 3) & 1) ^ 1;
        int Rp = ((reg >> 4) & 1) ^ 1;
        int X = ((baseReg >> 4) & 1) ^ 1; // base 4th bit (GP regs 0–15 → 0 → X=1); no index
        int Bb = ((baseReg >> 3) & 1) ^ 1;
        byte p0 = (byte)((R << 7) | (X << 6) | (Bb << 5) | (Rp << 4) | (map & 3));
        byte p1 = (byte)((w << 7) | (0xF << 3) | (1 << 2) | (pp & 3)); // vvvv=1111 (unused)
        byte p2 = (byte)((2 << 5) | (1 << 3));                          // L'L=512, V'=1
        byte modrm = (byte)(0x80 | ((reg & 7) << 3) | (baseReg & 7));   // mod=10
        B(0x62, p0, p1, p2, opcode, modrm);
        if ((baseReg & 7) == 4) _code.Add(0x24);                        // SIB for rsp/r12
        Imm32(disp32);
    }

    /// <summary>vfmadd231pd zmm, zmm, zmm. EVEX.512.66.0F38.W1 B8.</summary>
    internal void Vfmadd231pdZ(int dst, int s1, int s2) => EvexRR(Map0F38, Pp66, 1, 0xB8, dst, s1, s2);
    /// <summary>vfmadd231ps zmm, zmm, zmm. EVEX.512.66.0F38.W0 B8.</summary>
    internal void Vfmadd231psZ(int dst, int s1, int s2) => EvexRR(Map0F38, Pp66, 0, 0xB8, dst, s1, s2);
    /// <summary>vmovupd zmm, [base+disp32] (load). EVEX.512.66.0F.W1 10.</summary>
    internal void VmovupdLoadZ(int dst, int baseReg, int disp32) => EvexMemDisp32(Map0F, Pp66, 1, 0x10, dst, baseReg, disp32);
    /// <summary>vmovupd [base+disp32], zmm (store). EVEX.512.66.0F.W1 11.</summary>
    internal void VmovupdStoreZ(int baseReg, int disp32, int src) => EvexMemDisp32(Map0F, Pp66, 1, 0x11, src, baseReg, disp32);
    /// <summary>vmovups zmm, [base+disp32] (load). EVEX.512.0F.W0 10.</summary>
    internal void VmovupsLoadZ(int dst, int baseReg, int disp32) => EvexMemDisp32(Map0F, PpNone, 0, 0x10, dst, baseReg, disp32);
    /// <summary>vmovups [base+disp32], zmm (store). EVEX.512.0F.W0 11.</summary>
    internal void VmovupsStoreZ(int baseReg, int disp32, int src) => EvexMemDisp32(Map0F, PpNone, 0, 0x11, src, baseReg, disp32);
    /// <summary>vbroadcastsd zmm, [base+disp32]. EVEX.512.66.0F38.W1 19.</summary>
    internal void VbroadcastSdZ(int dst, int baseReg, int disp32) => EvexMemDisp32(Map0F38, Pp66, 1, 0x19, dst, baseReg, disp32);
    /// <summary>vbroadcastss zmm, [base+disp32]. EVEX.512.66.0F38.W0 18.</summary>
    internal void VbroadcastSsZ(int dst, int baseReg, int disp32) => EvexMemDisp32(Map0F38, Pp66, 0, 0x18, dst, baseReg, disp32);
}
