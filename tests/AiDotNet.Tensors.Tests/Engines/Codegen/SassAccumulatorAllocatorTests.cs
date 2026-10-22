// Copyright (c) AiDotNet. All rights reserved.
// The own SASS backend's accumulator allocator (#1442). No device and no toolkit: the instructions are built with the
// measured bit model, and what is checked is what the allocator promises -- that re-pairing is the only renaming that
// can move a mod-2 bank conflict at all, that the map it emits keeps every accumulator zeroed, and that the strict
// renamer would refuse the same map. What the removed conflicts are worth in time is a device question, not claimed.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class SassAccumulatorAllocatorTests
{
    private const int Ffma = 0x223;      // FFMA, all-register form
    private const int Cs2r = 0x005;      // CS2R Rd, SRZ
    private const int Lds128 = 0x384;    // LDS.128

    private static SassInstruction Make(int opcode, int rd, int ra = SassInstruction.ZeroRegister,
        int rb = SassInstruction.ZeroRegister, int rc = SassInstruction.ZeroRegister, int size = 4)
    {
        var instruction = new SassInstruction(0, 0);
        instruction.SetBits(0, 12, (ulong)opcode);
        instruction.GuardPredicate = SassInstruction.TruePredicate;
        instruction.SetRegister(SassInstruction.RdField, rd);
        instruction.SetRegister(SassInstruction.RaField, ra);
        instruction.SetRegister(SassInstruction.RbField, rb);
        instruction.SetRegister(SassInstruction.RcField, rc);
        instruction.SetBits(73, 3, (ulong)size);
        instruction.Control = SassControl.Fixed(1);
        return instruction;
    }

    /// <summary>
    /// The shape the GEMM's inner loop has: accumulators zeroed by CS2R pairs, fragments arriving as a 128-bit shared
    /// load, and FFMAs that each read two fragments and their own accumulator. Every FFMA here starts with all three
    /// sources on one parity, which is the conflict the allocator exists to break.
    /// </summary>
    private static SassInstruction[] FfmaBody() =>
    [
        Make(Cs2r, 8),                        // R8, R9 = 0
        Make(Cs2r, 10),                       // R10, R11 = 0
        Make(Lds128, 16, ra: 4, size: 6),     // R16..R19 = LDS.128 [R4]
        Make(Ffma, 8, ra: 16, rb: 18, rc: 8),
        Make(Ffma, 9, ra: 17, rb: 19, rc: 9),
        Make(Ffma, 10, ra: 18, rb: 16, rc: 10),
        Make(Ffma, 11, ra: 19, rb: 17, rc: 11),
    ];

    private static int[] Identity()
    {
        var map = new int[SassInstruction.ZeroRegister];
        for (int r = 0; r < map.Length; r++) map[r] = r;
        return map;
    }

    [Fact]
    public void TheAccumulatorsAreTheRegistersCs2rZeroes()
    {
        Assert.Equal(new[] { 8, 9, 10, 11 }, SassAccumulatorAllocator.Accumulators(FfmaBody()).ToArray());
        // An odd CS2R destination is not a pair, so it is not offered for re-pairing.
        Assert.Empty(SassAccumulatorAllocator.Accumulators(new[] { Make(Cs2r, 9) }));
    }

    /// <summary>
    /// The premise the whole allocator rests on: a renaming that keeps each CS2R destination pair grouped moves both
    /// halves together, so it cannot change a single mod-2 conflict. If this ever stops holding, the allocator is
    /// solving a problem that no longer exists.
    /// </summary>
    [Fact]
    public void MovingAWholePairCannotChangeAnyConflict()
    {
        var code = FfmaBody();
        int before = SassFfmaReorder.BankConflicts(code);
        var map = Identity();
        // Move the pair R8,R9 to R24,R25 and R10,R11 to R26,R27: legal for the strict renamer, and pointless.
        map[8] = 24; map[9] = 25; map[10] = 26; map[11] = 27;
        var renamed = SassRegisterRenamer.Rename(code, map) ?? throw new InvalidOperationException(SassRegisterRenamer.LastDecline);
        Assert.Equal(before, SassFfmaReorder.BankConflicts(renamed));
    }

    [Fact]
    public void TheAllocatorRepairsTheAccumulatorsAndRemovesConflicts()
    {
        var code = FfmaBody();
        var allocation = SassAccumulatorAllocator.Allocate(code)
            ?? throw new InvalidOperationException(SassAccumulatorAllocator.LastDecline);
        Assert.Equal(4, allocation.Accumulators);
        // Each FFMA's two fragments share a parity whatever the map does, so one conflict per FFMA cannot be removed;
        // the accumulator's two can, and the allocator finds both swaps (8<->9 and 10<->11) that remove them.
        Assert.Equal(12, allocation.Before);
        Assert.Equal(4, allocation.After);
        Assert.Equal(9, allocation.Map[8]);
        Assert.Equal(8, allocation.Map[9]);
        Assert.Equal(11, allocation.Map[10]);
        Assert.Equal(10, allocation.Map[11]);

        var renamed = SassAccumulatorAllocator.RenameRepairingAccumulators(code, allocation.Map)
            ?? throw new InvalidOperationException(SassAccumulatorAllocator.LastDecline);
        Assert.Equal(allocation.After, SassFfmaReorder.BankConflicts(renamed));
        // The CS2Rs are untouched: they still zero the same two pairs, which is all their constant has to do.
        Assert.Equal(code[0], renamed[0]);
        Assert.Equal(code[1], renamed[1]);
        // The fragments and the load's address are untouched as well.
        Assert.Equal(code[2], renamed[2]);
        // And the strict renamer cannot express this map at all -- it would split both CS2R pairs. That refusal is
        // the reason the re-pairing form exists, so if the strict one ever accepted it, this test should fail.
        Assert.Null(SassRegisterRenamer.Rename(code, allocation.Map));
        // Still schedulable afterwards.
        Assert.NotNull(SassScheduler.RecomputeControl(renamed));
    }

    [Fact]
    public void ARepairingRenameRefusesAMapThatLeavesTheZeroedSet()
    {
        var code = FfmaBody();
        var map = Identity();
        map[8] = 30;                                   // R30 is zeroed by nothing
        Assert.Null(SassAccumulatorAllocator.RenameRepairingAccumulators(code, map));
        Assert.Contains("no CS2R zeroes", SassAccumulatorAllocator.LastDecline, StringComparison.Ordinal);
    }

    [Fact]
    public void TheAllocatorDeclinesWhenThereIsNothingToRepair()
    {
        // No CS2R: no accumulator is known to start at a constant, so nothing may be re-paired.
        var code = new[] { Make(Ffma, 8, ra: 16, rb: 18, rc: 8), Make(Ffma, 9, ra: 17, rb: 19, rc: 9) };
        Assert.Null(SassAccumulatorAllocator.Allocate(code));
        Assert.Contains("no CS2R accumulator pair", SassAccumulatorAllocator.LastDecline, StringComparison.Ordinal);

        // An unknown opcode refuses the kernel outright, as the renamer and the scheduler do.
        var unknown = new[] { Make(Cs2r, 8), Make(Cs2r, 10), new SassInstruction(0x0000000000000fffUL, 0) };
        Assert.Null(SassAccumulatorAllocator.Allocate(unknown));

        Assert.Throws<ArgumentOutOfRangeException>(() => SassAccumulatorAllocator.Allocate(FfmaBody(), banks: 1));
        Assert.Throws<ArgumentNullException>(() => SassAccumulatorAllocator.Allocate(null!));
    }

    /// <summary>
    /// A real ptxas kernel: k1 has no CS2R, so the allocator declines it rather than inventing an allocation -- and
    /// the strict renamer still shifts it, which is what the allocator's map would feed.
    /// </summary>
    [Fact]
    public void ARealPtxasKernelWithoutAccumulatorsIsDeclinedAndStillRenames()
    {
        var code = new SassCubin(SassInstructionTests.K1CubinImage()).Instructions("k1");
        Assert.Null(SassAccumulatorAllocator.Allocate(code));
        var map = Identity();
        for (int r = 0; r < map.Length - 4; r++) map[r] = r + 4;   // +4 keeps every pair even and every quad aligned
        var renamed = SassRegisterRenamer.Rename(code, map) ?? throw new InvalidOperationException(SassRegisterRenamer.LastDecline);
        Assert.Equal(code.Length, renamed.Length);
        Assert.NotNull(SassScheduler.RecomputeControl(renamed));
    }
}
