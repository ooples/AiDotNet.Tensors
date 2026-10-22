// Copyright (c) AiDotNet. All rights reserved.
// The accumulator allocator of the own SASS backend (#1442): it chooses the register map that SassRegisterRenamer
// applies, for the one allocation decision the GEMM's inner loop actually leaves open.
//
// Why a plain renaming cannot help at all. On sm_86 a register's operand-collector bank is its number mod 2
// (--sass-probe --bank moved an FFMA loop's time with that model and not with mod 4). An accumulator lives in a
// 64-bit CS2R destination pair, and the renamer -- correctly -- refuses any map that splits or misaligns such a pair,
// so both halves move together and every accumulator keeps its parity. Parity IS the bank, so no legal map can move
// a single conflict. That is the wall the PR's "a plain renaming keeps their bank parity" note names.
//
// What gets past it. CS2R writes a CONSTANT ZERO. Nothing flows from a particular CS2R to a particular accumulator;
// the only thing the kernel needs is that every accumulator is zero before the first FFMA reads it. So the
// accumulators may be RE-PAIRED: leave the CS2R destinations exactly as ptxas wrote them, rename everything else, and
// require the map to permute the zeroed register set onto itself. Then the same pairs are still zeroed, every
// accumulator still starts at zero, and a value is free to become the low or the high half of its pair -- which is the
// only way a renaming can change its bank.
//
// The allocator reports the conflicts it removed. What they are worth in time is a device question: #1442's gate is
// 1.10x over our own PTX path on a compute-bound kernel, and the bank probe priced the whole remaining
// operand-bandwidth spread on the RTX 3080 at under 2%.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;

/// <summary>Chooses a register map that lowers an FFMA loop's bank conflicts by re-pairing its accumulators.</summary>
public static class SassAccumulatorAllocator
{
    private const int Cs2r = 0x005;        // CS2R Rd, SRZ: a constant zero into a 64-bit destination pair
    private const int Nop = 0x118;

    /// <summary>Why the last <see cref="Allocate"/> returned null (diagnostics).</summary>
    [ThreadStatic] public static string? LastDecline;

    /// <summary>One allocation: the map to rename with, and the conflicts before and after it.</summary>
    /// <param name="Map">A register map indexed by register number, identity outside the accumulators.</param>
    /// <param name="Before">Same-bank FFMA file reads as ptxas left them.</param>
    /// <param name="After">Same-bank FFMA file reads under <paramref name="Map"/>.</param>
    /// <param name="Accumulators">How many accumulator registers were re-pairable.</param>
    public readonly record struct SassAllocation(int[] Map, int Before, int After, int Accumulators);

    /// <summary>
    /// The registers covered by CS2R destination pairs: the accumulator slots, interchangeable because what CS2R
    /// writes into them is a constant.
    /// </summary>
    public static SortedSet<int> Accumulators(IReadOnlyList<SassInstruction> code)
    {
        if (code is null) throw new ArgumentNullException(nameof(code));
        var accumulators = new SortedSet<int>();
        for (int i = 0; i < code.Count; i++)
        {
            if ((code[i].Opcode & 0x1ff) != Cs2r) continue;
            int destination = code[i].Register(SassInstruction.RdField);
            // An odd or RZ destination is not a pair this may re-pair; leave it out and it stays where it is.
            if (destination == SassInstruction.ZeroRegister || destination % 2 != 0 ||
                destination + 1 >= SassInstruction.ZeroRegister)
                continue;
            accumulators.Add(destination);
            accumulators.Add(destination + 1);
        }
        return accumulators;
    }

    /// <summary>
    /// <paramref name="code"/> renamed by <paramref name="map"/> with the CS2R destination pairs left exactly as they
    /// are, or null when the map does not permute the zeroed set onto itself (an accumulator would then land on a
    /// register nothing zeroed and read whatever was there) or when the rename itself is refused.
    /// </summary>
    /// <remarks>
    /// The exemption is done by renaming a copy in which every CS2R is a NOP and then putting the original CS2Rs back:
    /// CS2R's only register operand is its destination, so this renames exactly everything else and reuses
    /// <see cref="SassRegisterRenamer.Rename"/> rather than repeating its grouping and alignment rules.
    /// </remarks>
    public static SassInstruction[]? RenameRepairingAccumulators(IReadOnlyList<SassInstruction> code,
        IReadOnlyList<int> map)
    {
        if (code is null) throw new ArgumentNullException(nameof(code));
        if (map is null) throw new ArgumentNullException(nameof(map));
        LastDecline = null;
        var zeroed = Accumulators(code);
        foreach (int register in zeroed)
            if (register >= map.Count || !zeroed.Contains(map[register]))
            {
                LastDecline = "R" + register + " is zeroed by a CS2R pair, and the map sends it to R" +
                    (register < map.Count ? map[register].ToString(System.Globalization.CultureInfo.InvariantCulture) : "?") +
                    ", which no CS2R zeroes: it would be read before anything zeroed it.";
                return null;
            }

        var masked = new SassInstruction[code.Count];
        var constantZeroAt = new List<int>();
        for (int i = 0; i < code.Count; i++)
        {
            masked[i] = code[i];
            if ((code[i].Opcode & 0x1ff) != Cs2r) continue;
            int destination = code[i].Register(SassInstruction.RdField);
            if (destination == SassInstruction.ZeroRegister || destination % 2 != 0 ||
                destination + 1 >= SassInstruction.ZeroRegister)
                continue;
            var placeholder = new SassInstruction(0, 0);
            placeholder.SetBits(0, 12, Nop);
            masked[i] = placeholder;
            constantZeroAt.Add(i);
        }
        var renamed = SassRegisterRenamer.Rename(masked, map);
        if (renamed is null)
        {
            LastDecline = SassRegisterRenamer.LastDecline;
            return null;
        }
        foreach (int i in constantZeroAt) renamed[i] = code[i];
        return renamed;
    }

    /// <summary>
    /// A map that lowers <paramref name="code"/>'s same-bank FFMA file reads, found by accepting any swap of two
    /// accumulators that strictly lowers <see cref="SassFfmaReorder.BankConflicts"/> until none does. The swaps are
    /// tried in register order, so the same kernel always yields the same map. Null when the kernel has fewer than
    /// two re-pairable accumulators, or when the rename is refused (an unknown opcode).
    /// </summary>
    public static SassAllocation? Allocate(IReadOnlyList<SassInstruction> code, int banks = 2)
    {
        if (code is null) throw new ArgumentNullException(nameof(code));
        if (banks < 2) throw new ArgumentOutOfRangeException(nameof(banks));
        LastDecline = null;
        var accumulators = Accumulators(code);
        if (accumulators.Count < 2)
        {
            LastDecline = "the kernel has no CS2R accumulator pair to re-pair.";
            return null;
        }
        var slots = new List<int>(accumulators);
        var map = new int[SassInstruction.ZeroRegister];
        for (int r = 0; r < map.Length; r++) map[r] = r;

        var renamed = RenameRepairingAccumulators(code, map);
        if (renamed is null) return null;
        int before = SassFfmaReorder.BankConflicts(renamed, banks), best = before;

        bool improved = true;
        while (improved)
        {
            improved = false;
            for (int a = 0; a < slots.Count && !improved; a++)
                for (int b = a + 1; b < slots.Count && !improved; b++)
                {
                    (map[slots[a]], map[slots[b]]) = (map[slots[b]], map[slots[a]]);
                    var candidate = RenameRepairingAccumulators(code, map);
                    int score = candidate is null ? int.MaxValue : SassFfmaReorder.BankConflicts(candidate, banks);
                    if (score < best) { best = score; improved = true; }
                    else (map[slots[a]], map[slots[b]]) = (map[slots[b]], map[slots[a]]);
                }
        }
        return new SassAllocation(map, before, best, accumulators.Count);
    }
}
