// Copyright (c) AiDotNet. All rights reserved.
// One 128-bit Volta-family SASS instruction (sm_70 through sm_89 share the layout below), the unit the own SASS backend
// (#1442) rewrites. The field positions were measured, not assumed: --sass-learn matched 17 624 instructions of the
// catalog's kernels (ptxas 13.1, sm_75) against nvdisasm's text with no disagreement:
//
//   bits   0-11   opcode (with operand-form bits)
//   bits  12-15   guard predicate: 12-14 the index (7 = PT, unguarded), 15 negation
//   bits  16-23   destination register Rd          (255 = RZ)
//   bits  24-31   first source register Ra
//   bits  32-39   second source register Rb (register forms)
//   bits  64-71   third source register Rc
//   bits 105-108  stall cycles before the next instruction issues
//   bit  109      yield
//   bits 110-112  write scoreboard set by a variable-latency instruction (7 = none)
//   bits 113-115  read scoreboard (sources released) (7 = none)
//   bits 116-121  scoreboards waited on before issue (bit mask)
//   bits 122-125  operand reuse cache flags
//
// Which of the register fields an opcode uses, and how wide each operand is, is the opcode table's business
// (SassOpcodeTable); this type only reads and writes bits.

using System;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;

/// <summary>A 128-bit SASS instruction as two little-endian 64-bit words.</summary>
public struct SassInstruction : IEquatable<SassInstruction>
{
    /// <summary>The register number meaning RZ (reads zero, discards writes).</summary>
    public const int ZeroRegister = 255;

    /// <summary>The predicate index meaning PT (always true).</summary>
    public const int TruePredicate = 7;

    /// <summary>The scoreboard index meaning none.</summary>
    public const int NoScoreboard = 7;

    /// <summary>Bit position of the destination register field.</summary>
    public const int RdField = 16;

    /// <summary>Bit position of the first source register field.</summary>
    public const int RaField = 24;

    /// <summary>Bit position of the second source register field (register forms).</summary>
    public const int RbField = 32;

    /// <summary>Bit position of the third source register field.</summary>
    public const int RcField = 64;

    public SassInstruction(ulong low, ulong high)
    {
        Low = low;
        High = high;
    }

    /// <summary>Bits 0-63.</summary>
    public ulong Low { get; private set; }

    /// <summary>Bits 64-127.</summary>
    public ulong High { get; private set; }

    /// <summary>Bits [position, position + width) of the instruction.</summary>
    public readonly ulong Bits(int position, int width)
    {
        if (position < 0 || width <= 0 || width > 64 || position + width > 128)
            throw new ArgumentOutOfRangeException(nameof(position), "The field must lie within the 128 bits.");
        ulong value = 0;
        for (int b = 0; b < width; b++)
        {
            int bit = position + b;
            ulong word = bit < 64 ? Low : High;
            value |= ((word >> (bit & 63)) & 1UL) << b;
        }
        return value;
    }

    /// <summary>Writes <paramref name="value"/> into bits [position, position + width).</summary>
    public void SetBits(int position, int width, ulong value)
    {
        if (position < 0 || width <= 0 || width > 64 || position + width > 128)
            throw new ArgumentOutOfRangeException(nameof(position), "The field must lie within the 128 bits.");
        if (width < 64 && value >> width != 0)
            throw new ArgumentOutOfRangeException(nameof(value), "The value does not fit the field.");
        for (int b = 0; b < width; b++)
        {
            int bit = position + b;
            ulong mask = 1UL << (bit & 63), set = ((value >> b) & 1UL) != 0 ? mask : 0UL;
            if (bit < 64) Low = (Low & ~mask) | set;
            else High = (High & ~mask) | set;
        }
    }

    /// <summary>The opcode with its operand-form bits (bits 0-11).</summary>
    public readonly int Opcode => (int)Bits(0, 12);

    /// <summary>The guard predicate index (7 = PT, unguarded).</summary>
    public int GuardPredicate
    {
        readonly get => (int)Bits(12, 3);
        set => SetBits(12, 3, (ulong)value);
    }

    /// <summary>Whether the guard predicate is negated.</summary>
    public bool GuardNegated
    {
        readonly get => Bits(15, 1) != 0;
        set => SetBits(15, 1, value ? 1UL : 0UL);
    }

    /// <summary>The 8-bit register field at <paramref name="field"/> (one of the *Field constants).</summary>
    public readonly int Register(int field) => (int)Bits(field, 8);

    /// <summary>Writes the 8-bit register field at <paramref name="field"/>.</summary>
    public void SetRegister(int field, int register)
    {
        if (register is < 0 or > ZeroRegister) throw new ArgumentOutOfRangeException(nameof(register));
        SetBits(field, 8, (ulong)register);
    }

    /// <summary>The scheduling control word (bits 105-125).</summary>
    public SassControl Control
    {
        readonly get => new((int)Bits(105, 4), Bits(109, 1) != 0, (int)Bits(110, 3), (int)Bits(113, 3), (int)Bits(116, 6),
            (int)Bits(122, 4));
        set
        {
            SetBits(105, 4, (ulong)value.Stall);
            SetBits(109, 1, value.Yield ? 1UL : 0UL);
            SetBits(110, 3, (ulong)value.WriteScoreboard);
            SetBits(113, 3, (ulong)value.ReadScoreboard);
            SetBits(116, 6, (ulong)value.WaitMask);
            SetBits(122, 4, (ulong)value.Reuse);
        }
    }

    public readonly bool Equals(SassInstruction other) => Low == other.Low && High == other.High;

    public override readonly bool Equals(object? obj) => obj is SassInstruction other && Equals(other);

    public override readonly int GetHashCode() => (Low ^ (High * 31)).GetHashCode();

    public override readonly string ToString() => "0x" + High.ToString("x16") + "_" + Low.ToString("x16");
}

/// <summary>
/// The scheduling half of an instruction: how long the issue stalls after it, whether the warp may yield, the
/// scoreboards a variable-latency instruction sets for its result (write) and for releasing its sources (read), the
/// scoreboards waited on before it issues, and which source operands the reuse cache keeps.
/// </summary>
public readonly record struct SassControl(int Stall, bool Yield, int WriteScoreboard, int ReadScoreboard, int WaitMask, int Reuse)
{
    /// <summary>No scoreboards, no reuse, the given stall.</summary>
    public static SassControl Fixed(int stall, bool yield = false) =>
        new(stall, yield, SassInstruction.NoScoreboard, SassInstruction.NoScoreboard, 0, 0);
}
