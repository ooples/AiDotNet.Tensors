using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.CpuJit;

/// <summary>
/// x86-64 machine code emitter for AVX2/FMA JIT compilation.
/// Writes raw instruction bytes for the subset of x86-64 we need:
/// - AVX2 vector arithmetic (VADDPS, VMULPS, VSUBPS, VMAXPS, etc.)
/// - FMA (VFMADD231PS)
/// - Memory operations (VMOVAPS, VMOVUPS, VMOVNTPS, VBROADCASTSS)
/// - Scalar/integer ops (MOV, ADD, SUB, CMP, JNE, etc.)
/// - Function prologue/epilogue (PUSH, POP, RET, SUB RSP)
///
/// This is the same approach oneDNN/Xbyak uses — emit raw opcodes
/// for the exact instruction sequence needed, with constants baked in.
/// </summary>
internal sealed class X86Emitter
{
    private readonly List<byte> _code = new(4096);
    private readonly List<(int offset, int targetLabel)> _jumpFixups = new();
    private readonly Dictionary<int, int> _labels = new();
    private int _nextLabel;

    // Data section: float constants embedded after code, loaded via MOV R11 + VBROADCASTSS [R11]
    // Each constant is 4 bytes (float), appended after code in the ExecutableBuffer.
    // At Build() time, we know the absolute address and patch MOV R11, imm64 instructions.
    private readonly List<float> _dataConstants = new();
    private readonly List<(int codeOffset, int constIndex)> _dataFixups = new(); // codeOffset points to the imm64 of MOV R11

    /// <summary>Current code size in bytes.</summary>
    public int Size => _code.Count;

    /// <summary>Reinterpret float bits as int32 (portable across all TFMs including net471).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe int FloatToInt32Bits(float value)
    {
        return *(int*)&value;
    }

    /// <summary>Allocate a new label ID for forward/backward jumps.</summary>
    public int NewLabel() => _nextLabel++;

    /// <summary>Bind a label to the current code position.</summary>
    public void BindLabel(int label)
    {
        _labels[label] = _code.Count;
    }

    // ==================== YMM Register IDs ====================
    // YMM0-YMM15 map to register IDs 0-15
    public const int YMM0 = 0, YMM1 = 1, YMM2 = 2, YMM3 = 3;
    public const int YMM4 = 4, YMM5 = 5, YMM6 = 6, YMM7 = 7;
    public const int YMM8 = 8, YMM9 = 9, YMM10 = 10, YMM11 = 11;
    public const int YMM12 = 12, YMM13 = 13, YMM14 = 14, YMM15 = 15;

    // x86-64 GPR register IDs
    public const int RAX = 0, RCX = 1, RDX = 2, RBX = 3;
    public const int RSP = 4, RBP = 5, RSI = 6, RDI = 7;
    public const int R8 = 8, R9 = 9, R10 = 10, R11 = 11;
    public const int R12 = 12, R13 = 13, R14 = 14, R15 = 15;

    // ==================== Raw byte emission ====================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Emit(byte b) => _code.Add(b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Emit(byte b1, byte b2) { _code.Add(b1); _code.Add(b2); }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EmitImm32(int imm)
    {
        _code.Add((byte)(imm & 0xFF));
        _code.Add((byte)((imm >> 8) & 0xFF));
        _code.Add((byte)((imm >> 16) & 0xFF));
        _code.Add((byte)((imm >> 24) & 0xFF));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EmitImm64(long imm)
    {
        for (int i = 0; i < 8; i++)
            _code.Add((byte)((imm >> (i * 8)) & 0xFF));
    }

    // ==================== VEX prefix encoding ====================
    // VEX 2-byte: C5 [R vvvv L pp]
    // VEX 3-byte: C4 [R X B mmmmm] [W vvvv L pp]

    /// <summary>
    /// Emits a 2-byte VEX prefix (C5).
    /// Used when no REX.X, REX.B, or 3-byte map is needed.
    /// </summary>
    /// <param name="r">REX.R bit (inverted): 0 if dst >= 8, 1 otherwise</param>
    /// <param name="vvvv">Source register (inverted, 4 bits): ~src & 0xF</param>
    /// <param name="l">Vector length: 0=128-bit, 1=256-bit</param>
    /// <param name="pp">Prefix: 0=none, 1=66, 2=F3, 3=F2</param>
    private void EmitVex2(int r, int vvvv, int l, int pp)
    {
        Emit(0xC5);
        Emit((byte)((r << 7) | (vvvv << 3) | (l << 2) | pp));
    }

    /// <summary>
    /// Emits a 3-byte VEX prefix (C4).
    /// Required for REX.X/B bits or 0F38/0F3A maps.
    /// </summary>
    private void EmitVex3(int r, int x, int b, int mmmmm, int w, int vvvv, int l, int pp)
    {
        Emit(0xC4);
        Emit((byte)((r << 7) | (x << 6) | (b << 5) | mmmmm));
        Emit((byte)((w << 7) | (vvvv << 3) | (l << 2) | pp));
    }

    /// <summary>
    /// Emits ModR/M byte.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EmitModRM(int mod, int reg, int rm)
    {
        Emit((byte)((mod << 6) | ((reg & 7) << 3) | (rm & 7)));
    }

    /// <summary>
    /// Emits SIB byte.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EmitSIB(int scale, int index, int baseReg)
    {
        Emit((byte)((scale << 6) | ((index & 7) << 3) | (baseReg & 7)));
    }

    // ==================== Helper: VEX-encoded reg,reg or reg,[base+disp] ====================

    /// <summary>
    /// Emits a VEX-encoded instruction: op ymm_dst, ymm_src1, ymm_src2 (register-register).
    /// map: 1=0F, 2=0F38, 3=0F3A. pp: 0=none, 1=66.
    /// </summary>
    private void EmitVexRR(int map, int pp, int w, byte opcode, int dst, int src1, int src2)
    {
        int rBit = (dst >> 3) ^ 1;    // inverted REX.R
        int bBit = (src2 >> 3) ^ 1;   // inverted REX.B
        int vvvv = (~src1) & 0xF;     // inverted source register

        if (map == 1 && bBit == 1 && w == 0)
        {
            // Can use 2-byte VEX
            EmitVex2(rBit, vvvv, 1 /*256-bit*/, pp);
        }
        else
        {
            EmitVex3(rBit, 1 /*X=1*/, bBit, map, w, vvvv, 1 /*256-bit*/, pp);
        }
        Emit(opcode);
        EmitModRM(3, dst, src2); // mod=11 (register)
    }

    /// <summary>
    /// Emits a VEX-encoded instruction: op ymm_dst, ymm_src1, [base + disp32].
    /// </summary>
    private void EmitVexRM(int map, int pp, int w, byte opcode, int dst, int src1, int baseReg, int disp, int l = 1)
    {
        int rBit = (dst >> 3) ^ 1;
        int bBit = (baseReg >> 3) ^ 1;
        int vvvv = (~src1) & 0xF;

        if (map == 1 && bBit == 1 && w == 0)
        {
            EmitVex2(rBit, vvvv, l, pp);
        }
        else
        {
            EmitVex3(rBit, 1, bBit, map, w, vvvv, l, pp);
        }
        Emit(opcode);

        // ModR/M + optional SIB + displacement
        if (baseReg == RSP || baseReg == R12)
        {
            // RSP/R12 as base requires SIB byte
            if (disp == 0 && baseReg != RBP && baseReg != R13)
            {
                EmitModRM(0, dst, 4); // mod=00, rm=100 (SIB follows)
                EmitSIB(0, 4, baseReg); // scale=0, index=RSP (none), base
            }
            else if (disp >= -128 && disp <= 127)
            {
                EmitModRM(1, dst, 4);
                EmitSIB(0, 4, baseReg);
                Emit((byte)(disp & 0xFF));
            }
            else
            {
                EmitModRM(2, dst, 4);
                EmitSIB(0, 4, baseReg);
                EmitImm32(disp);
            }
        }
        else if (disp == 0 && baseReg != RBP && baseReg != R13)
        {
            EmitModRM(0, dst, baseReg);
        }
        else if (disp >= -128 && disp <= 127)
        {
            EmitModRM(1, dst, baseReg);
            Emit((byte)(disp & 0xFF));
        }
        else
        {
            EmitModRM(2, dst, baseReg);
            EmitImm32(disp);
        }
    }

    /// <summary>
    /// Emits store: op [base + disp], ymm_src. (No vvvv operand.)
    /// </summary>
    private void EmitVexStore(int map, int pp, int w, byte opcode, int src, int baseReg, int disp)
    {
        // For stores, the "reg" field in ModR/M is the source YMM register
        EmitVexRM(map, pp, w, opcode, src, 0 /*vvvv unused, set to 0 → ~0=0xF*/, baseReg, disp);
    }

    // ==================== AVX2 Vector Instructions ====================

    /// <summary>VADDPS ymm, ymm, ymm — Packed float add</summary>
    public void Vaddps(int dst, int src1, int src2)
        => EmitVexRR(1, 0, 0, 0x58, dst, src1, src2);

    /// <summary>VADDPS ymm, ymm, [base+disp] — Packed float add from memory</summary>
    public void Vaddps(int dst, int src1, int baseReg, int disp)
        => EmitVexRM(1, 0, 0, 0x58, dst, src1, baseReg, disp);

    /// <summary>VMULPS ymm, ymm, ymm — Packed float multiply</summary>
    public void Vmulps(int dst, int src1, int src2)
        => EmitVexRR(1, 0, 0, 0x59, dst, src1, src2);

    /// <summary>VMULPS ymm, ymm, [base+disp] — Packed float multiply from memory</summary>
    public void Vmulps(int dst, int src1, int baseReg, int disp)
        => EmitVexRM(1, 0, 0, 0x59, dst, src1, baseReg, disp);

    /// <summary>VSUBPS ymm, ymm, ymm — Packed float subtract</summary>
    public void Vsubps(int dst, int src1, int src2)
        => EmitVexRR(1, 0, 0, 0x5C, dst, src1, src2);

    /// <summary>VSUBPS ymm, ymm, [base+disp] — Packed float subtract from memory</summary>
    public void Vsubps(int dst, int src1, int baseReg, int disp)
        => EmitVexRM(1, 0, 0, 0x5C, dst, src1, baseReg, disp);

    /// <summary>VMAXPS ymm, ymm, ymm — Packed float maximum (used for ReLU)</summary>
    public void Vmaxps(int dst, int src1, int src2)
        => EmitVexRR(1, 0, 0, 0x5F, dst, src1, src2);

    /// <summary>VMAXPS ymm, ymm, [base+disp] — Packed float maximum from memory</summary>
    public void Vmaxps(int dst, int src1, int baseReg, int disp)
        => EmitVexRM(1, 0, 0, 0x5F, dst, src1, baseReg, disp);

    /// <summary>VXORPS ymm, ymm, ymm — Packed float XOR (used to zero registers)</summary>
    public void Vxorps(int dst, int src1, int src2)
        => EmitVexRR(1, 0, 0, 0x57, dst, src1, src2);

    /// <summary>
    /// Generic packed single-precision op: op ymm_dst, ymm_src1, ymm_src2 (register-register).
    /// Opcode examples: 0x58=VADDPS, 0x59=VMULPS, 0x5C=VSUBPS, 0x5E=VDIVPS, 0x5D=VMINPS, 0x5F=VMAXPS.
    /// Open for extension: new ops require only a new opcode constant, no emitter changes.
    /// </summary>
    public void VbinaryPs(byte opcode, int dst, int src1, int src2)
        => EmitVexRR(1, 0, 0, opcode, dst, src1, src2);

    /// <summary>
    /// Generic packed single-precision op: op ymm_dst, ymm_src1, [base+disp] (register-memory).
    /// </summary>
    public void VbinaryPs(byte opcode, int dst, int src1, int baseReg, int disp)
        => EmitVexRM(1, 0, 0, opcode, dst, src1, baseReg, disp);

    // ==================== AVX2 Memory Operations ====================

    /// <summary>VMOVAPS ymm, [base+disp] — Aligned load (requires 32-byte alignment)</summary>
    public void VmovapsLoad(int dst, int baseReg, int disp)
        => EmitVexRM(1, 0, 0, 0x28, dst, 0 /*no vvvv*/, baseReg, disp);

    /// <summary>VMOVAPS [base+disp], ymm — Aligned store (requires 32-byte alignment)</summary>
    public void VmovapsStore(int src, int baseReg, int disp)
        => EmitVexStore(1, 0, 0, 0x29, src, baseReg, disp);

    /// <summary>VMOVUPS ymm, [base+disp] — Unaligned load</summary>
    public void VmovupsLoad(int dst, int baseReg, int disp)
        => EmitVexRM(1, 0, 0, 0x10, dst, 0, baseReg, disp);

    /// <summary>VMOVUPS [base+disp], ymm — Unaligned store</summary>
    public void VmovupsStore(int src, int baseReg, int disp)
        => EmitVexStore(1, 0, 0, 0x11, src, baseReg, disp);

    /// <summary>VMOVSS xmm, [base+disp] — Scalar float load (4 bytes only)</summary>
    public void VmovssLoad(int dst, int baseReg, int disp)
        => EmitVexRM(1, 2 /*F3*/, 0, 0x10, dst, 0, baseReg, disp, l: 0);

    /// <summary>VMOVSS [base+disp], xmm — Scalar float store (4 bytes only)</summary>
    public void VmovssStore(int src, int baseReg, int disp)
    {
        // VEX.LIG.F3.0F.WIG 11 /r
        EmitVexRM(1, 2 /*F3*/, 0, 0x11, src, 0, baseReg, disp, l: 0);
    }

    /// <summary>VMOVNTPS [base+disp], ymm — Non-temporal store (requires 32-byte alignment, bypasses cache)</summary>
    public void VmovntpsStore(int src, int baseReg, int disp)
        => EmitVexStore(1, 0, 0, 0x2B, src, baseReg, disp);

    /// <summary>VBROADCASTSS ymm, [base+disp] — Broadcast single float to all 8 lanes</summary>
    public void Vbroadcastss(int dst, int baseReg, int disp)
    {
        // VEX.256.66.0F38.W0 18 /r
        EmitVexRM(2 /*0F38*/, 1 /*66*/, 0, 0x18, dst, 0, baseReg, disp);
    }

    // ==================== FMA Instructions ====================

    /// <summary>VFMADD231PS ymm, ymm, ymm — Fused multiply-add: dst = src1 * src2 + dst</summary>
    public void Vfmadd231ps(int dst, int src1, int src2)
    {
        // VEX.256.66.0F38.W0 B8 /r
        EmitVexRR(2 /*0F38*/, 1 /*66*/, 0, 0xB8, dst, src1, src2);
    }

    /// <summary>VFMADD231PS ymm, ymm, [base+disp] — Fused multiply-add from memory</summary>
    public void Vfmadd231ps(int dst, int src1, int baseReg, int disp)
    {
        EmitVexRM(2, 1, 0, 0xB8, dst, src1, baseReg, disp);
    }

    // ==================== Prefetch ====================

    /// <summary>PREFETCHT0 [base+disp] — Prefetch to L1 cache</summary>
    public void Prefetcht0(int baseReg, int disp)
    {
        // REX prefix needed for R8-R15 base registers
        if (baseReg >= 8)
        {
            Emit((byte)(0x41)); // REX.B
        }
        // 0F 18 /1  (reg=1 in ModR/M)
        Emit(0x0F, 0x18);
        if (disp == 0 && (baseReg & 7) != RBP)
        {
            EmitModRM(0, 1, baseReg & 7);
        }
        else if (disp >= -128 && disp <= 127)
        {
            EmitModRM(1, 1, baseReg & 7);
            Emit((byte)(disp & 0xFF));
        }
        else
        {
            EmitModRM(2, 1, baseReg & 7);
            EmitImm32(disp);
        }
    }

    /// <summary>SFENCE — Store fence (required after non-temporal stores)</summary>
    public void Sfence()
    {
        Emit(0x0F); Emit(0xAE); Emit(0xF8);
    }

    // ==================== Scalar / Integer Instructions ====================

    /// <summary>MOV reg64, reg64 — Register-to-register move</summary>
    public void MovRR(int dst, int src)
    {
        // REX.W + 89 /r  (MOV r/m64, r64)
        int rex = 0x48 | ((src >> 3) << 2) | ((dst >> 3) & 1);
        Emit((byte)rex);
        Emit(0x89);
        EmitModRM(3, src, dst);
    }

    /// <summary>MOV reg64, imm64 — Load 64-bit immediate into register</summary>
    public void MovImm64(int reg, long imm)
    {
        // REX.W + B8+rd id
        int rex = 0x48 | ((reg >> 3) & 1); // REX.W + REX.B if reg >= 8
        Emit((byte)rex);
        Emit((byte)(0xB8 + (reg & 7)));
        EmitImm64(imm);
    }

    /// <summary>MOV reg64, imm32 (sign-extended) — C7 /0 with REX.W</summary>
    public void MovImm32(int reg, int imm)
    {
        int rex = 0x48 | ((reg >> 3) & 1);
        Emit((byte)rex);
        Emit(0xC7);
        EmitModRM(3, 0, reg);
        EmitImm32(imm);
    }

    /// <summary>ADD reg64, imm32 — Add immediate to 64-bit register</summary>
    public void AddImm32(int reg, int imm)
    {
        int rex = 0x48 | ((reg >> 3) & 1);
        Emit((byte)rex);
        if (imm >= -128 && imm <= 127)
        {
            Emit(0x83);
            EmitModRM(3, 0, reg);
            Emit((byte)(imm & 0xFF));
        }
        else
        {
            Emit(0x81);
            EmitModRM(3, 0, reg);
            EmitImm32(imm);
        }
    }

    /// <summary>SUB reg64, imm32 — Subtract immediate from 64-bit register</summary>
    public void SubImm32(int reg, int imm)
    {
        int rex = 0x48 | ((reg >> 3) & 1);
        Emit((byte)rex);
        if (imm >= -128 && imm <= 127)
        {
            Emit(0x83);
            EmitModRM(3, 5, reg);
            Emit((byte)(imm & 0xFF));
        }
        else
        {
            Emit(0x81);
            EmitModRM(3, 5, reg);
            EmitImm32(imm);
        }
    }

    /// <summary>CMP reg64, imm32 — Compare register with immediate</summary>
    public void CmpImm32(int reg, int imm)
    {
        int rex = 0x48 | ((reg >> 3) & 1);
        Emit((byte)rex);
        if (imm >= -128 && imm <= 127)
        {
            Emit(0x83);
            EmitModRM(3, 7, reg);
            Emit((byte)(imm & 0xFF));
        }
        else
        {
            Emit(0x81);
            EmitModRM(3, 7, reg);
            EmitImm32(imm);
        }
    }

    /// <summary>CMP reg64, reg64</summary>
    public void CmpRR(int reg1, int reg2)
    {
        int rex = 0x48 | ((reg2 >> 3) << 2) | ((reg1 >> 3) & 1);
        Emit((byte)rex);
        Emit(0x39);
        EmitModRM(3, reg2, reg1);
    }

    /// <summary>JL label (jump if less, signed)</summary>
    public void Jl(int label)
    {
        Emit(0x0F); Emit(0x8C);
        _jumpFixups.Add((_code.Count, label));
        EmitImm32(0); // placeholder, will be patched
    }

    /// <summary>JNE label (jump if not equal / not zero)</summary>
    public void Jne(int label)
    {
        Emit(0x0F); Emit(0x85);
        _jumpFixups.Add((_code.Count, label));
        EmitImm32(0); // placeholder
    }

    /// <summary>JMP label (unconditional jump)</summary>
    public void Jmp(int label)
    {
        Emit(0xE9);
        _jumpFixups.Add((_code.Count, label));
        EmitImm32(0);
    }

    // ==================== Function Prologue/Epilogue ====================

    /// <summary>
    /// Emit Windows x64 ABI function prologue.
    /// Saves non-volatile registers and aligns stack.
    /// Windows x64 ABI: RCX=arg0, RDX=arg1, R8=arg2, R9=arg3
    /// Non-volatile: RBX, RBP, RDI, RSI, R12-R15, XMM6-XMM15
    /// </summary>
    public void Prologue()
    {
        // PUSH RBP
        Emit(0x55);
        // MOV RBP, RSP
        Emit(0x48, 0x89); EmitModRM(3, RSP, RBP);
        // PUSH RBX (non-volatile, we use it as loop counter)
        Emit(0x53);
        // SUB RSP, 32 — shadow space for potential calls + alignment
        SubImm32(RSP, 32);
    }

    /// <summary>
    /// Emit function epilogue matching the prologue.
    /// </summary>
    public void Epilogue()
    {
        // VZEROUPPER — avoid AVX/SSE transition penalty
        Emit(0xC5, 0xF8); Emit(0x77);
        // ADD RSP, 32
        AddImm32(RSP, 32);
        // POP RBX
        Emit(0x5B);
        // POP RBP
        Emit(0x5D);
        // RET
        Emit(0xC3);
    }

    // ==================== Data Section Constants ====================

    /// <summary>
    /// Registers a float constant in the data section. Returns the constant index.
    /// Reuses existing constants with the same value.
    /// </summary>
    public int EmitDataConstant(float value)
    {
        // Check for existing constant with same bit pattern
        int bits = FloatToInt32Bits(value);
        for (int i = 0; i < _dataConstants.Count; i++)
        {
            if (FloatToInt32Bits(_dataConstants[i]) == bits)
                return i;
        }
        _dataConstants.Add(value);
        return _dataConstants.Count - 1;
    }

    /// <summary>
    /// VBROADCASTSS ymm, [data_constant] — Broadcast a data section float constant to all 8 lanes.
    /// Emits: MOV R11, imm64 (patched at Build time) + VBROADCASTSS ymm, [R11]
    /// R11 is caller-saved (volatile) in Windows x64 ABI, safe to use as scratch.
    /// </summary>
    public void VbroadcastssConst(int dst, int constIndex)
    {
        // MOV R11, imm64  (REX.WB + B8+3 for R11)
        // R11 = register 11, so REX.B is needed: 0x49, opcode B8+3=BB
        Emit(0x49, 0xBB);
        _dataFixups.Add((_code.Count, constIndex)); // record where the imm64 starts
        EmitImm64(0); // placeholder — patched by Build() with actual address

        // VBROADCASTSS ymm_dst, [R11]
        // VEX.256.66.0F38.W0 18 /r with base=R11
        Vbroadcastss(dst, R11, 0);
    }

    /// <summary>
    /// VMOVAPS ymm, [data_constant] — Load 32 bytes from a data section address (8 copies of same float, pre-broadcasted).
    /// Only use this when you've stored a pre-broadcasted vector constant (8 identical floats) in the data section.
    /// For single float constants, use VbroadcastssConst instead.
    /// </summary>

    // ==================== Finalize and Build ====================

    /// <summary>
    /// Resolves all jump targets and copies code into an ExecutableBuffer.
    /// </summary>
    public ExecutableBuffer Build()
    {
        // Resolve jump fixups
        foreach (var (offset, targetLabel) in _jumpFixups)
        {
            if (!_labels.TryGetValue(targetLabel, out int targetPos))
                throw new InvalidOperationException($"Unresolved label {targetLabel}");

            // Relative offset from end of jump instruction (offset + 4 bytes for imm32)
            int relOffset = targetPos - (offset + 4);
            _code[offset] = (byte)(relOffset & 0xFF);
            _code[offset + 1] = (byte)((relOffset >> 8) & 0xFF);
            _code[offset + 2] = (byte)((relOffset >> 16) & 0xFF);
            _code[offset + 3] = (byte)((relOffset >> 24) & 0xFF);
        }

        // Calculate data section size: 4 bytes per float constant, 32-byte aligned start
        int codeSize = _code.Count;
        int dataOffset = (codeSize + 31) & ~31; // align data section to 32 bytes
        int totalSize = _dataConstants.Count > 0 ? dataOffset + (_dataConstants.Count * 4) : codeSize;

        var buffer = new ExecutableBuffer(totalSize);
        unsafe
        {
            var dst = (byte*)buffer.Pointer;

            // Copy code
            for (int i = 0; i < codeSize; i++)
            {
                dst[i] = _code[i];
            }

            // Zero padding between code and data (only if there is a data section)
            if (_dataConstants.Count > 0)
            {
                for (int i = codeSize; i < dataOffset; i++)
                {
                    dst[i] = 0xCC; // INT3 padding (trap if executed)
                }
            }

            // Write data constants
            for (int i = 0; i < _dataConstants.Count; i++)
            {
                int constAddr = dataOffset + (i * 4);
                int bits = FloatToInt32Bits(_dataConstants[i]);
                dst[constAddr] = (byte)(bits & 0xFF);
                dst[constAddr + 1] = (byte)((bits >> 8) & 0xFF);
                dst[constAddr + 2] = (byte)((bits >> 16) & 0xFF);
                dst[constAddr + 3] = (byte)((bits >> 24) & 0xFF);
            }

            // Patch data fixups: each fixup points to an imm64 in a MOV R11 instruction
            // The imm64 needs to be the absolute address of the constant in the buffer
            long bufferBase = (long)(IntPtr)buffer.Pointer;
            foreach (var (fixupOffset, constIndex) in _dataFixups)
            {
                long constAbsAddr = bufferBase + dataOffset + (constIndex * 4);
                // Write 8 bytes (imm64) at the fixup offset
                for (int b = 0; b < 8; b++)
                {
                    dst[fixupOffset + b] = (byte)((constAbsAddr >> (b * 8)) & 0xFF);
                }
            }
        }

        // Switch from writable to executable (W^X compliance)
        buffer.MakeExecutable();
        return buffer;
    }

    /// <summary>
    /// Returns the raw code bytes (for debugging/disassembly).
    /// </summary>
    public byte[] GetCodeBytes()
    {
        return _code.ToArray();
    }
}
