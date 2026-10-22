// Copyright (c) AiDotNet. All rights reserved.
// What each Volta-family SASS opcode reads and writes, for the own SASS backend's register allocation and scheduling
// (#1442). An instruction's bits 0-8 name the operation and bits 9-11 its operand form; --sass-learn measured, on the
// catalog's kernels compiled for sm_75, where each form keeps its register sources:
//
//   form 1 (0x2..)  A at bits 24, B at 32, C at 64      (all registers)
//   form 2 (0x4..)  A at 24, B at 64;  the third operand an immediate
//   form 3 (0x6..)  A at 24, B at 64;  the third operand a constant-bank word
//   form 4 (0x8..)  A at 24, C at 64;  B an immediate (bits 32-63)
//   form 5 (0xa..)  A at 24, C at 64;  B a constant-bank word
//   form 6 (0xc..)  A at 24, C at 64;  B a uniform register at 32
//   form 7 (0xe..)  A at 24, B at 64;  C a uniform register at 32
//
// The uniform datapath (sm_75+; one value per warp) has its own registers UR0-UR62 (6-bit fields, URZ = 63) and
// predicates UP0-UP6. Its ALU operations are the vector opcode plus 0x80 (UIADD3 0x090, UIMAD 0x0a4, ULEA 0x091,
// ...), with every register field uniform; ULDC loads a constant-bank word (or pair, by the size in bits 73-75) on a
// fixed latency; S2UR and R2UR (vector A at 24 -> uniform) are variable-latency, like S2R. A uniform operation's guard
// names a uniform predicate (nvdisasm prints @UP0): the uniform datapath has no per-thread state to test.
//
// sm_80+ (learned on sm_86): LDGSTS (cp.async) reads a shared address at bits 16 and a global address pair at 24 and
// holds them under a read scoreboard; LDGDEPBAR commits the outstanding copies to a write scoreboard; DEPBAR.LE
// SBn, k waits until scoreboard n (bits 44-46) holds at most k groups. CALL.REL.NOINC is a branch (ptxas leaves a
// loop through it), its target at bits 32 like BRA's.
//
// and where predicates live: a compare's outputs at bits 81 and 84, a predicate source at 87 (negation at 90) -- the
// carry-in of the .X forms of IADD3, LEA and IMAD among them (an IMAD.X issued 5 cycles after the IADD3 whose carry it
// reads computed a wrong address high word, and its load faulted). Loads
// and stores keep their access width in bits 73-75 (4 = 32 bits, 5 = 64, 6 = 128). Variable-latency operations (the
// ones ptxas gives a scoreboard: loads, S2R, MUFU, I2F/F2I) are marked so the scheduler waits on them rather than
// counting cycles. An opcode not listed is unknown: a kernel containing one is left exactly as ptxas wrote it.

using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;

/// <summary>How an instruction's result arrives.</summary>
public enum SassLatency
{
    /// <summary>A fixed pipeline latency: later readers are delayed by stall counts.</summary>
    Fixed,

    /// <summary>A variable latency (memory, special functions): readers wait on a scoreboard.</summary>
    Variable,

    /// <summary>Control flow, barriers and other instructions the backend never moves or rewrites.</summary>
    Control,
}

/// <summary>
/// The register operands of one instruction: field position, width in registers, and whether the field names a uniform
/// register (UR0-UR62, a separate register file) rather than a vector one.
/// </summary>
public readonly record struct SassRegisterOperand(int Field, int Width, bool Uniform = false);

/// <summary>What one decoded instruction reads and writes.</summary>
public sealed class SassEffects
{
    internal SassEffects(string mnemonic, SassLatency latency, IReadOnlyList<SassRegisterOperand> writes,
        IReadOnlyList<SassRegisterOperand> reads, IReadOnlyList<int> predicateWrites, IReadOnlyList<int> predicateReads,
        bool readsAllPredicates, int fixedLatency, bool uniformPredicates = false, IReadOnlyList<int>? uniformPredicateReads = null)
    {
        UniformPredicates = uniformPredicates;
        UniformPredicateReads = uniformPredicateReads ?? [];
        Mnemonic = mnemonic;
        Latency = latency;
        Writes = writes;
        Reads = reads;
        PredicateWrites = predicateWrites;
        PredicateReads = predicateReads;
        ReadsAllPredicates = readsAllPredicates;
        FixedLatency = fixedLatency;
    }

    public string Mnemonic { get; }
    public SassLatency Latency { get; }

    /// <summary>Destination register fields (RZ, 255, means none).</summary>
    public IReadOnlyList<SassRegisterOperand> Writes { get; }

    /// <summary>Source register fields (RZ means none).</summary>
    public IReadOnlyList<SassRegisterOperand> Reads { get; }

    /// <summary>Bit positions of 3-bit predicate destinations (7, PT, means none).</summary>
    public IReadOnlyList<int> PredicateWrites { get; }

    /// <summary>Bit positions of 3-bit predicate sources, besides the guard (7, PT, means none).</summary>
    public IReadOnlyList<int> PredicateReads { get; }

    /// <summary>
    /// Bit positions of 3-bit uniform predicate sources (UP0-UP6) of a vector instruction: PLOP3 reads its third
    /// source from the uniform file when bit 67 is set.
    /// </summary>
    public IReadOnlyList<int> UniformPredicateReads { get; }

    /// <summary>The instruction reads the whole predicate file (P2R).</summary>
    public bool ReadsAllPredicates { get; }

    /// <summary>Cycles before a fixed-latency result may be read (0 for the other kinds).</summary>
    public int FixedLatency { get; }

    /// <summary>
    /// The predicate fields and the guard name uniform predicates (UP0-UP6), a file separate from P0-P6: the
    /// uniform datapath's compares, carries and guards.
    /// </summary>
    public bool UniformPredicates { get; }
}

/// <summary>The opcode table of the Volta-family encoding (sm_70 through sm_89 share it).</summary>
public static class SassOpcodeTable
{
    private const int A = SassInstruction.RaField, B = SassInstruction.RbField, C = SassInstruction.RcField;
    private const int D = SassInstruction.RdField;

    // Fixed latencies. The fewest cycles ptxas leaves between a producer and its first reader over the catalog's
    // kernels on sm_75 (--sass-learn) are 4 for the ALU, but that is a minimum over particular pairs, not a latency:
    // scheduling every ALU result at 4 faulted the first kernel. 6 ran every catalog kernel bit-identically; the wide
    // multiply-add (8, its second result register), P2R (14) and CS2R (8) take ptxas's measured minimum.
    private const int AluLatency = 6;

    private enum Sources
    {
        None,
        BOnly,      // single-source operations keep it in the B slot (MOV, MUFU, I2F, F2I, IABS)
        A,
        AB,
        ABC,
    }

    private sealed record Spec(string Mnemonic, SassLatency Latency, bool WritesD, int DWidth, Sources Sources,
        int[] PredicateWrites, int[] PredicateReads, bool Memory = false, bool Store = false, int AddressWidth = 1,
        bool ReadsAllPredicates = false, bool WideC = false, int FixedLatency = AluLatency, bool Uniform = false,
        bool SizedD = false, bool VectorSources = false, bool AsyncCopy = false, int UniformThirdPredicateFlag = -1,
        bool Mma = false, bool MatrixLoad = false);

    private static readonly Dictionary<int, Spec> s_specs = new()
    {
        [0x002] = new("MOV", SassLatency.Fixed, true, 1, Sources.BOnly, [], []),
        [0x003] = new("P2R", SassLatency.Fixed, true, 1, Sources.A, [], [], ReadsAllPredicates: true, FixedLatency: 14),
        [0x005] = new("CS2R", SassLatency.Fixed, true, 2, Sources.None, [], [], FixedLatency: 8),
        [0x007] = new("SEL", SassLatency.Fixed, true, 1, Sources.AB, [], [87]),
        [0x008] = new("FSEL", SassLatency.Fixed, true, 1, Sources.AB, [], [87]),
        [0x009] = new("FMNMX", SassLatency.Fixed, true, 1, Sources.AB, [], [87]),
        [0x00b] = new("FSETP", SassLatency.Fixed, false, 0, Sources.AB, [81, 84], [87]),
        [0x00c] = new("ISETP", SassLatency.Fixed, false, 0, Sources.AB, [81, 84], [87]),
        [0x010] = new("IADD3", SassLatency.Fixed, true, 1, Sources.ABC, [81, 84], [87, 77]),
        [0x011] = new("LEA", SassLatency.Fixed, true, 1, Sources.ABC, [81], [87]),
        [0x012] = new("LOP3", SassLatency.Fixed, true, 1, Sources.ABC, [81], [87]),
        [0x013] = new("IABS", SassLatency.Fixed, true, 1, Sources.BOnly, [], []),
        [0x017] = new("IMNMX", SassLatency.Fixed, true, 1, Sources.AB, [], [87]),
        [0x019] = new("SHF", SassLatency.Fixed, true, 1, Sources.ABC, [], []),
        [0x01c] = new("PLOP3", SassLatency.Fixed, false, 0, Sources.None, [81, 84], [87, 77, 68], UniformThirdPredicateFlag: 67),
        [0x020] = new("FMUL", SassLatency.Fixed, true, 1, Sources.AB, [], []),
        [0x021] = new("FADD", SassLatency.Fixed, true, 1, Sources.AB, [], []),
        [0x023] = new("FFMA", SassLatency.Fixed, true, 1, Sources.ABC, [], []),
        [0x024] = new("IMAD", SassLatency.Fixed, true, 1, Sources.ABC, [], [87]),
        [0x025] = new("IMAD.WIDE", SassLatency.Fixed, true, 2, Sources.ABC, [], [87], WideC: true, FixedLatency: 8),
        [0x027] = new("IMAD.HI", SassLatency.Fixed, true, 1, Sources.ABC, [], [87]),
        // sm_80+: cvt.rn.f32.s32/u32 is I2FP, a fixed-latency ALU conversion (no scoreboard; ptxas reads it 5 cycles on).
        [0x045] = new("I2FP", SassLatency.Fixed, true, 1, Sources.BOnly, [], []),
        [0x082] = new("UMOV", SassLatency.Fixed, true, 1, Sources.BOnly, [], [], Uniform: true),
        [0x087] = new("USEL", SassLatency.Fixed, true, 1, Sources.AB, [], [87], Uniform: true),
        [0x08c] = new("UISETP", SassLatency.Fixed, false, 0, Sources.AB, [81, 84], [87], Uniform: true),
        [0x090] = new("UIADD3", SassLatency.Fixed, true, 1, Sources.ABC, [81, 84], [87, 77], Uniform: true),
        [0x091] = new("ULEA", SassLatency.Fixed, true, 1, Sources.ABC, [81], [87], Uniform: true),
        [0x092] = new("ULOP3", SassLatency.Fixed, true, 1, Sources.ABC, [81], [87], Uniform: true),
        [0x099] = new("USHF", SassLatency.Fixed, true, 1, Sources.ABC, [], [], Uniform: true),
        [0x0a4] = new("UIMAD", SassLatency.Fixed, true, 1, Sources.ABC, [], [87], Uniform: true),
        [0x0a5] = new("UIMAD.WIDE", SassLatency.Fixed, true, 2, Sources.ABC, [], [87], WideC: true, Uniform: true),
        [0x0b9] = new("ULDC", SassLatency.Fixed, true, 1, Sources.None, [], [], Uniform: true, SizedD: true),
        [0x1c2] = new("R2UR", SassLatency.Variable, true, 1, Sources.A, [], [], Uniform: true, VectorSources: true),
        [0x1c3] = new("S2UR", SassLatency.Variable, true, 1, Sources.None, [], [], Uniform: true),
        [0x105] = new("F2I", SassLatency.Variable, true, 1, Sources.BOnly, [], []),
        [0x106] = new("I2F", SassLatency.Variable, true, 1, Sources.BOnly, [], []),
        [0x108] = new("MUFU", SassLatency.Variable, true, 1, Sources.BOnly, [], []),
        [0x119] = new("S2R", SassLatency.Variable, true, 1, Sources.None, [], []),
        [0x181] = new("LDG", SassLatency.Variable, true, 1, Sources.None, [], [], Memory: true, AddressWidth: 2),
        [0x184] = new("LDS", SassLatency.Variable, true, 1, Sources.None, [], [], Memory: true),
        [0x186] = new("STG", SassLatency.Variable, false, 0, Sources.None, [], [], Memory: true, Store: true, AddressWidth: 2),
        [0x188] = new("STS", SassLatency.Variable, false, 0, Sources.None, [], [], Memory: true, Store: true),
        // Local memory (register spills): [R1 + offset], the access width in bits 73-75 as for the other loads/stores.
        [0x183] = new("LDL", SassLatency.Variable, true, 1, Sources.None, [], [], Memory: true),
        [0x187] = new("STL", SassLatency.Variable, false, 0, Sources.None, [], [], Memory: true, Store: true),
        // sm_75+: ldmatrix. LDSM.16.M88.{1,2,4} loads 1 << bits 72-73 registers from the shared address at A.
        [0x03b] = new("LDSM", SassLatency.Variable, true, 1, Sources.None, [], [], MatrixLoad: true),
        // sm_80+: mma.sync m16n8k8 TF32 (HMMA.1688.F32.TF32) and m16n8k16 FP16 (HMMA.16816.F32), FP32 accumulate:
        // D (4 registers) = A (4) x B (2) + C (4), on the tensor pipe's fixed latency (ptxas sets no scoreboard; it never
        // reads a result sooner than 56 cycles on, over the TF32 and FP16 GEMM seeds on sm_86).
        [0x03c] = new("HMMA", SassLatency.Fixed, true, 4, Sources.ABC, [], [], Mma: true, FixedLatency: 56),
        [0x1ae] = new("LDGSTS", SassLatency.Variable, false, 0, Sources.None, [], [87], AsyncCopy: true),
        [0x1af] = new("LDGDEPBAR", SassLatency.Variable, false, 0, Sources.None, [], []),
        [0x118] = new("NOP", SassLatency.Control, false, 0, Sources.None, [], []),
        [0x11a] = new("DEPBAR", SassLatency.Control, false, 0, Sources.None, [], []),
        [0x11d] = new("BAR", SassLatency.Control, false, 0, Sources.None, [], []),
        [0x141] = new("BSYNC", SassLatency.Control, false, 0, Sources.None, [], []),
        [0x144] = new("CALL", SassLatency.Control, false, 0, Sources.None, [], []),
        [0x145] = new("BSSY", SassLatency.Control, false, 0, Sources.None, [], []),
        [0x147] = new("BRA", SassLatency.Control, false, 0, Sources.None, [], []),
        [0x14d] = new("EXIT", SassLatency.Control, false, 0, Sources.None, [], []),
        [0x155] = new("BMOV", SassLatency.Control, false, 0, Sources.None, [], []),
    };

    /// <summary>The effects of <paramref name="instruction"/>, or null for an opcode the table does not know.</summary>
    public static SassEffects? Decode(in SassInstruction instruction)
    {
        int op = instruction.Opcode & 0x1ff, form = instruction.Opcode >> 9;
        if (!s_specs.TryGetValue(op, out var spec)) return null;

        // Uniform operations name uniform registers in every field (R2UR reads a vector one); a vector operation
        // names a uniform register only in the field its form gives one (forms 6 and 7, bits 32).
        bool uniformD = spec.Uniform, uniformSources = spec.Uniform && !spec.VectorSources;
        var writes = new List<SassRegisterOperand>();
        var reads = new List<SassRegisterOperand>();
        if (spec.Mma)
        {
            // Only the register form (0x23c) and the FP32-accumulating shapes --sass-learn has seen (bits 75 and 83:
            // 01 = 1688.F32.TF32, 10 = 16816.F32); another variant may size its operands differently.
            int shape = (int)instruction.Bits(75, 1) | (int)instruction.Bits(83, 1) << 1;
            if (form != 1 || shape is not (1 or 2)) return null;
            writes.Add(new SassRegisterOperand(D, 4));
            reads.Add(new SassRegisterOperand(A, 4));
            reads.Add(new SassRegisterOperand(B, 2));
            reads.Add(new SassRegisterOperand(C, 4));
        }
        else if (spec.MatrixLoad)
        {
            int count = (int)instruction.Bits(72, 2);
            if (form != 4 || count == 3) return null;
            reads.Add(new SassRegisterOperand(A, 1));
            writes.Add(new SassRegisterOperand(D, 1 << count));
        }
        else if (spec.AsyncCopy)
        {
            // LDGSTS [Rs+offset], [Rg.64]: the shared address sits in the destination field (read, not written), the
            // global address pair at A; the copied data never passes through registers.
            reads.Add(new SassRegisterOperand(D, 1));
            reads.Add(new SassRegisterOperand(A, 2));
        }
        else if (spec.Memory)
        {
            // Access width in registers from the size field: 4 = 32 bits, 5 = 64, 6 = 128.
            int size = (int)instruction.Bits(73, 3);
            int width = size switch { 5 => 2, 6 => 4, _ => 1 };
            reads.Add(new SassRegisterOperand(A, spec.AddressWidth));
            if (spec.Store) reads.Add(new SassRegisterOperand(B, width));
            else writes.Add(new SassRegisterOperand(D, width));
        }
        else
        {
            int dWidth = spec.SizedD ? (instruction.Bits(73, 3) == 5 ? 2 : 1) : spec.DWidth;
            if (spec.WritesD) writes.Add(new SassRegisterOperand(D, dWidth, uniformD));
            switch (spec.Sources)
            {
                case Sources.BOnly:
                    // A register-form single source sits in B; an immediate or constant form has no register source.
                    if (form == 1) reads.Add(new SassRegisterOperand(B, 1, uniformSources));
                    else if (form == 6) reads.Add(new SassRegisterOperand(B, 1, true));
                    break;
                case Sources.A:
                    reads.Add(new SassRegisterOperand(A, 1, uniformSources));
                    break;
                case Sources.AB:
                case Sources.ABC:
                    reads.Add(new SassRegisterOperand(A, 1, uniformSources));
                    int cWidth = spec.WideC ? 2 : 1;
                    switch (form)
                    {
                        case 1:
                            reads.Add(new SassRegisterOperand(B, 1, uniformSources));
                            if (spec.Sources == Sources.ABC) reads.Add(new SassRegisterOperand(C, cWidth, uniformSources));
                            break;
                        case 2:
                        case 3:
                            // The second register source moves to bits 64; the third operand is immediate/constant. A
                            // two-source operation in these forms (FADD R, R, imm: 0x421) has only A in registers.
                            if (spec.Sources == Sources.ABC) reads.Add(new SassRegisterOperand(C, 1, uniformSources));
                            break;
                        case 4:
                        case 5:
                            if (spec.Sources == Sources.ABC) reads.Add(new SassRegisterOperand(C, cWidth, uniformSources));
                            break;
                        case 6 when !spec.Uniform:
                            reads.Add(new SassRegisterOperand(B, 1, true));
                            if (spec.Sources == Sources.ABC) reads.Add(new SassRegisterOperand(C, cWidth));
                            break;
                        case 7 when !spec.Uniform:
                            // The second source (a vector register) moves to bits 64, the third (uniform) to 32.
                            reads.Add(new SassRegisterOperand(C, 1));
                            reads.Add(new SassRegisterOperand(B, spec.Sources == Sources.ABC ? cWidth : 1, true));
                            break;
                        default:
                            return null;
                    }
                    break;
            }
        }
        // PLOP3 P0, PT, PT, PT, UP0 (a uniform compare's result moved to a vector predicate): bit 67 makes the third
        // source, bits 68-70, a uniform predicate. Missing it scheduled the PLOP3 1 cycle after the UISETP writing UP0
        // (ptxas leaves 6), and two catalog convolutions computed wrong results on the RTX 3080.
        IReadOnlyList<int> predicateReads = spec.PredicateReads, uniformPredicateReads = [];
        if (spec.UniformThirdPredicateFlag >= 0 && instruction.Bits(spec.UniformThirdPredicateFlag, 1) != 0)
        {
            predicateReads = [.. spec.PredicateReads.Where(f => f != 68)];
            uniformPredicateReads = [68];
        }
        return new SassEffects(spec.Mnemonic, spec.Latency, writes, reads, spec.PredicateWrites, predicateReads,
            spec.ReadsAllPredicates, spec.Latency == SassLatency.Fixed ? spec.FixedLatency : 0, spec.Uniform, uniformPredicateReads);
    }
}
