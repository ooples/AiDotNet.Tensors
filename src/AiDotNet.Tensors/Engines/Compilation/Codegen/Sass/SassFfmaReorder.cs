// Copyright (c) CrowdTrainer LLC. All rights reserved.
// FFMA ordering for the own SASS backend (#1442). A register-blocked GEMM's inner loop is runs of independent FFMAs
// (each accumulates a different register from a shared fragment of A and of B), and their order decides two operand
// costs ptxas leaves on the table on sm_86:
//
//   - the operand reuse cache: an FFMA that reads the same register in the same slot as the FFMA just before it can
//     take it from the cache (the earlier one sets that slot's reuse flag) instead of the register file;
//   - register banks: the operands still read from the file conflict when two share a bank (register mod 2 on sm_86:
//     --sass-probe --bank measured the most-conflicted renaming of an FFMA loop at 0.94x ptxas, the least at 1.01x).
//
// Within a run, any order computes the same values: each FFMA writes its own accumulator, which no other FFMA of the
// run reads or writes, and reads A and B that no FFMA of the run writes. So reordering keeps every result bit-identical.
// The order is chosen greedily: next is the remaining FFMA with the fewest same-bank file reads, then a reuse hit, then
// the earliest. Reuse flags are set wherever the next FFMA reads the same register in the slot; the scheduler drops a
// flag whose instruction's stall it lengthens (the cache is valid only when the next instruction follows at once).

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;

/// <summary>Reorders runs of independent FFMAs for operand reuse and register-bank conflicts.</summary>
public static class SassFfmaReorder
{
    private const int Ffma = 0x223;   // FFMA, all-register form
    private static readonly int[] SourceFields = { SassInstruction.RaField, SassInstruction.RbField, SassInstruction.RcField };

    /// <summary>
    /// <paramref name="code"/> with each run of at least <paramref name="minimumRun"/> consecutive, unguarded,
    /// independent register-form FFMAs that no branch enters mid-run reordered and its reuse flags recomputed; other instructions are untouched. The
    /// control words' stalls inside a run are set to 1 (the scheduler recomputes them), and the union of the run's
    /// scoreboard waits is carried by whichever FFMA now comes first.
    /// </summary>
    /// <remarks>
    /// The waits have to be hoisted, not carried along: ptxas puts the wait on the loads' scoreboard on the FFMA that
    /// reads those fragments first, so an order that moves that FFMA later would leave the new first FFMA issuing
    /// without it, reading stale fragment registers. The union on the first instruction is at least as strict as what
    /// each instruction waited for before, and every instruction keeps its own mask as well, so the result is safe
    /// before <see cref="SassScheduler.RecomputeControl"/> runs and unchanged by it afterwards.
    /// </remarks>
    public static SassInstruction[] Reorder(IReadOnlyList<SassInstruction> code, int banks = 2, int minimumRun = 4)
    {
        if (code is null) throw new ArgumentNullException(nameof(code));
        var result = new SassInstruction[code.Count];
        for (int i = 0; i < code.Count; i++) result[i] = code[i];
        // A branch target splits a run: control entering mid-run must find the same instructions after it.
        var targets = SassScheduler.BranchTargets(code);
        int start = 0;
        while (start < code.Count)
        {
            int end = start;
            while (end < code.Count && IsCandidate(code[end]) && (end == start || !targets.Contains(end))) end++;
            if (end - start >= minimumRun && Independent(code, start, end))
            {
                ReorderRun(result, start, end, banks);
                // A reuse flag on the instruction before the run kept an operand for the FFMA that used to come first.
                if (start > 0) result[start - 1].Control = result[start - 1].Control with { Reuse = 0 };
            }
            start = end == start ? start + 1 : end;
        }
        return result;
    }

    /// <summary>Same-bank pairs among the register-file reads of the FFMAs in <paramref name="code"/> (diagnostics).</summary>
    public static int BankConflicts(IReadOnlyList<SassInstruction> code, int banks = 2)
    {
        int conflicts = 0;
        for (int i = 0; i < code.Count; i++)
            if (code[i].Opcode == Ffma) conflicts += Conflicts(code[i], i > 0 && code[i - 1].Opcode == Ffma ? code[i - 1] : (SassInstruction?)null, banks, true);
        return conflicts;
    }

    private static bool IsCandidate(in SassInstruction instruction) =>
        instruction.Opcode == Ffma && instruction.GuardPredicate == SassInstruction.TruePredicate;

    private static bool Independent(IReadOnlyList<SassInstruction> code, int start, int end)
    {
        var written = new HashSet<int>();
        for (int i = start; i < end; i++)
        {
            int d = code[i].Register(SassInstruction.RdField);
            if (d == SassInstruction.ZeroRegister || !written.Add(d)) return false;
        }
        for (int i = start; i < end; i++)
        {
            int d = code[i].Register(SassInstruction.RdField);
            if (written.Contains(code[i].Register(SassInstruction.RaField)) || written.Contains(code[i].Register(SassInstruction.RbField))) return false;
            int c = code[i].Register(SassInstruction.RcField);
            if (c != d && written.Contains(c)) return false;   // another FFMA's accumulator
        }
        return true;
    }

    // Same-bank pairs among the operands `instruction` reads from the register file, given the instruction before it
    // (whose reuse flags are taken as set when `assumeReuse`, or decided by matching registers when not).
    private static int Conflicts(in SassInstruction instruction, SassInstruction? previous, int banks, bool assumeReuse)
    {
        Span<int> fromFile = stackalloc int[3];
        int count = 0;
        for (int slot = 0; slot < 3; slot++)
        {
            int r = instruction.Register(SourceFields[slot]);
            if (r == SassInstruction.ZeroRegister) continue;
            bool cached = previous is { } p && p.Register(SourceFields[slot]) == r && (!assumeReuse || (p.Control.Reuse >> slot & 1) != 0);
            if (cached) continue;
            bool duplicate = false;
            for (int k = 0; k < count; k++) duplicate |= fromFile[k] == r;
            if (!duplicate) fromFile[count++] = r;
        }
        int conflicts = 0;
        for (int x = 0; x < count; x++)
            for (int y = x + 1; y < count; y++)
                if (fromFile[x] % banks == fromFile[y] % banks) conflicts++;
        return conflicts;
    }

    private static int ReuseHits(in SassInstruction instruction, in SassInstruction previous)
    {
        int hits = 0;
        for (int slot = 0; slot < 2; slot++)
            if (instruction.Register(SourceFields[slot]) == previous.Register(SourceFields[slot])) hits++;
        return hits;
    }

    private static void ReorderRun(SassInstruction[] code, int start, int end, int banks)
    {
        var remaining = new List<SassInstruction>();
        int waits = 0;
        for (int i = start; i < end; i++) waits |= code[i].Control.WaitMask;
        for (int i = start; i < end; i++) remaining.Add(code[i]);
        var order = new List<SassInstruction>(remaining.Count);
        SassInstruction? previous = null;
        while (remaining.Count != 0)
        {
            int best = 0, bestConflicts = int.MaxValue, bestHits = -1;
            for (int k = 0; k < remaining.Count; k++)
            {
                int conflicts = Conflicts(remaining[k], previous, banks, assumeReuse: false);
                int hits = previous is { } p ? ReuseHits(remaining[k], p) : 0;
                if (conflicts < bestConflicts || (conflicts == bestConflicts && hits > bestHits))
                {
                    best = k;
                    bestConflicts = conflicts;
                    bestHits = hits;
                }
            }
            order.Add(remaining[best]);
            previous = remaining[best];
            remaining.RemoveAt(best);
        }
        for (int k = 0; k < order.Count; k++)
        {
            var instruction = order[k];
            int reuse = 0;
            if (k + 1 < order.Count)
                for (int slot = 0; slot < 2; slot++)
                    if (order[k + 1].Register(SourceFields[slot]) == instruction.Register(SourceFields[slot])) reuse |= 1 << slot;
            // The first FFMA of the new order waits for everything the run as a whole waited for; the others keep
            // their own masks, which that union already covers.
            instruction.Control = instruction.Control with
            {
                Stall = 1,
                Reuse = reuse,
                WaitMask = k == 0 ? waits : instruction.Control.WaitMask,
            };
            code[start + k] = instruction;
        }
    }
}
