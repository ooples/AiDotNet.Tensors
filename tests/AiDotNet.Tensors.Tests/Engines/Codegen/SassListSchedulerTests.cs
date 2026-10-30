// Copyright (c) CrowdTrainer LLC. All rights reserved.
// SassListScheduler (#1442): instruction reordering within straight-line blocks. Encodings are sm_86 driver cubins'.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class SassListSchedulerTests
{
    private static readonly SassInstruction Hmma1688 = new(0x000000d64020723cUL, 0x040fee0000081020UL);   // HMMA.1688.F32.TF32 R32, R64.reuse, R214, R32
    private static readonly SassInstruction Lea = new(0x0000000603067211UL, 0x001fc800078e40ffUL);        // LEA R6, R3, R6, 0x8
    private static readonly SassInstruction Lds128 = new(0x000000004c307984UL, 0x000ff00000001c00UL);     // LDS.U.128 R48, [R76]
    private static readonly SassInstruction Exit = new(0x000000000000794dUL, 0x000fea0003800000UL);       // EXIT
    private static readonly SassInstruction Ffma = new(0x0000000502097223UL, 0x004fd00000000002UL);       // FFMA R9, R2, R5, R2

    private static SassInstruction HmmaOn(int d)
    {
        var hmma = Hmma1688;
        hmma.SetRegister(SassInstruction.RdField, d);
        hmma.SetRegister(SassInstruction.RcField, d);
        return hmma;
    }

    private static SassInstruction With(SassInstruction instruction, int field, int register)
    {
        instruction.SetRegister(field, register);
        return instruction;
    }

    // LEA R{d}, R3, R{d}... rewritten to R{d}, R{a}, R{a}: an independent ALU op on its own registers.
    private static SassInstruction LeaOn(int d, int a) =>
        With(With(With(Lea, SassInstruction.RdField, d), SassInstruction.RaField, a), SassInstruction.RbField, a);

    private static SassInstruction FfmaOf(int d, int a, int b, int c) =>
        With(With(With(With(Ffma, SassInstruction.RdField, d), SassInstruction.RaField, a), SassInstruction.RbField, b), SassInstruction.RcField, c);

    /// <summary>Two independent MMAs back to back: independent ALU work moves between them (the tensor pipe takes 8).</summary>
    [Fact]
    public void IndependentWorkFillsTheTensorPipeGap()
    {
        var code = new[] { HmmaOn(32), HmmaOn(40), LeaOn(100, 101), LeaOn(102, 103), Exit };
        var reordered = SassListScheduler.Reorder(code) ?? throw new InvalidOperationException("declined");
        int first = Array.FindIndex(reordered, i => i.Equals(code[0]) || i.Low == code[0].Low);
        int second = Array.FindIndex(reordered, i => i.Low == code[1].Low);
        Assert.True(second - first > 1, "the MMAs stayed adjacent");
        Assert.Equal(code.Select(i => i.Low).OrderBy(x => x), reordered.Select(i => i.Low).OrderBy(x => x));
        Assert.Equal(code[4].Low, reordered[4].Low);   // EXIT stays last
    }

    /// <summary>
    /// A reader never moves above the instruction it depends on, and variable-latency operations keep their order,
    /// even where the critical path would pull the later one up: here the later instruction feeds an MMA (a 56-cycle
    /// path), the earlier one nothing.
    /// </summary>
    [Fact]
    public void DependenciesAndMemoryOrderHold()
    {
        var producer = LeaOn(64, 101);                                       // writes R64, the MMA's A operand
        var consumer = With(HmmaOn(32), SassInstruction.RaField, 64);        // HMMA R32, R64, ...
        var raw = SassListScheduler.Reorder(new[] { producer, consumer, LeaOn(102, 103), Exit })
            ?? throw new InvalidOperationException("declined");
        Assert.True(Array.FindIndex(raw, r => r.Low == producer.Low) < Array.FindIndex(raw, r => r.Low == consumer.Low));

        var loadA = Lds128;                                                  // LDS R48..R51: nothing reads it
        var loadB = With(Lds128, SassInstruction.RdField, 68);              // LDS R68..R71: the MMA's A operand
        var mma = With(HmmaOn(32), SassInstruction.RaField, 68);
        var ordered = SassListScheduler.Reorder(new[] { loadA, loadB, mma, LeaOn(102, 103), Exit })
            ?? throw new InvalidOperationException("declined");
        Assert.True(Array.FindIndex(ordered, r => r.Low == loadA.Low) < Array.FindIndex(ordered, r => r.Low == loadB.Low));
    }
    /// <summary>
    /// Branch targets bound the blocks: nothing moves across one, so control entering there finds the same code. A
    /// reuse flag survives only where the next instruction is still ptxas's next one.
    /// </summary>
    [Fact]
    public void BranchTargetsAndReuseAreRespected()
    {
        // BRA at 3 back to 2: instructions 0-1 and 2-2 are separate blocks; the HMMA pair at 0-1 may not absorb 2.
        var bra = new SassInstruction(0x0000000000007947UL, 0x000fea0003800000UL);
        long offset = (2 - 4) * 16L;
        bra.SetBits(32, 50, unchecked((ulong)offset) & ((1UL << 50) - 1));
        var first = HmmaOn(32);
        first.Control = first.Control with { Reuse = 1 };
        var code = new[] { first, HmmaOn(40), LeaOn(100, 101), bra, Exit };
        var reordered = SassListScheduler.Reorder(code) ?? throw new InvalidOperationException("declined");
        Assert.Equal(code.Select(i => i.Low), reordered.Select(i => i.Low));   // two-instruction blocks are left alone

        var longer = new[] { first, HmmaOn(40), LeaOn(100, 101), LeaOn(102, 103), Exit };
        var moved = SassListScheduler.Reorder(longer) ?? throw new InvalidOperationException("declined");
        int at = Array.FindIndex(moved, i => i.Low == first.Low);
        bool stillFollowedByOriginalNext = at + 1 < moved.Length && moved[at + 1].Low == longer[1].Low;
        Assert.Equal(stillFollowedByOriginalNext, moved[at].Control.Reuse != 0);
    }
}