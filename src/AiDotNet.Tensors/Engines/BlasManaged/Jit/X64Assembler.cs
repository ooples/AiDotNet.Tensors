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

    internal byte[] ToArray()
    {
        foreach (var (pos, label) in _rel8Fixups)
        {
            int target = _labelPos[label];
            int rel = target - (pos + 1); // rel8 is relative to the byte AFTER the displacement
            if (rel < sbyte.MinValue || rel > sbyte.MaxValue)
                throw new InvalidOperationException("rel8 jump out of range");
            _code[pos] = (byte)(sbyte)rel;
        }
        return _code.ToArray();
    }

    private void B(params byte[] bytes) => _code.AddRange(bytes);
    private void Imm32(int v) { _code.Add((byte)v); _code.Add((byte)(v >> 8)); _code.Add((byte)(v >> 16)); _code.Add((byte)(v >> 24)); }

    // ── Labels / branches ────────────────────────────────────────────────────
    internal int NewLabel() => _labelPos.Count + _rel8Fixups.Count + 100000; // unique id
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

    internal void Ret() => B(0xC3);
    internal void Vzeroupper() => B(0xC5, 0xF8, 0x77);

    /// <summary>add reg64, imm8 (sign-extended). REX.W[.B] 83 /0 ib.</summary>
    internal void AddRegImm8(int reg, sbyte imm)
    {
        byte rex = (byte)(0x48 | (reg >= 8 ? 1 : 0));
        B(rex, 0x83, (byte)(0xC0 | (reg & 7)), (byte)imm);
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
}
