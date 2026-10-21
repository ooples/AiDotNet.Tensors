// Copyright (c) AiDotNet. All rights reserved.
// The own SASS backend's scheduler (#1442): it owns the control word ptxas would otherwise choose. Given a kernel's
// instructions in issue order, it recomputes every stall count, scoreboard assignment and wait mask from
// SassOpcodeTable's view of what each instruction reads and writes:
//
//   - a variable-latency instruction (load, S2R, MUFU, conversions) gets a write scoreboard for the registers it
//     writes and a read scoreboard for the registers it reads (a later writer of those must wait for the read);
//   - an instruction waits on every scoreboard covering a register it reads or writes (and, for one it writes, on
//     the read scoreboards covering it); waiting on a scoreboard clears every register it covered;
//   - a fixed-latency result is readable FixedLatency cycles after issue: the issue of its first reader is delayed
//     by raising the stall counts before it.
//
// Two delays measured on sm_75 (a run with neither faulted): a scoreboard is set a couple of cycles after the
// instruction that sets it issues, so a wait on it issues at least ScoreboardSetDelay cycles later; and a predicate
// used as a guard (by any instruction, not only a branch) needs GuardPredicateLatency cycles after the compare that
// wrote it -- ptxas leaves exactly 12, and an ALU guard read after the ordinary 6 faulted -- while a predicate read as an
// operand (a compare's combine input, SEL's selector) needs only the ALU latency.
//
// Control flow is handled conservatively: a branch target or a control instruction waits on every scoreboard and
// issues only after every fixed-latency result is ready, so no state crosses a block boundary. The yield bits stay as
// they were. An operand reuse flag survives only where the instruction's stall is no longer than ptxas's: the reuse
// cache is valid only when the next instruction follows closely enough that the scheduler cannot have issued another
// warp in between -- lengthening the stall of an instruction that keeps its reuse flags faulted on sm_75 (ptxas's own
// schedule with four extra stall cycles on its memory instructions read stale operands).
// A kernel with an opcode the table does not know is left alone (null).

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;

/// <summary>Recomputes SASS control words from the instructions' register and predicate effects.</summary>
public static class SassScheduler
{
    // ptxas never encodes a stall above 12 (12 only after a compare); a variable-latency instruction stalling 12 or
    // more faulted on sm_75 where 11 ran. So stalls stay within 12 (11 on variable-latency instructions), and a longer
    // delay is spread over the instructions issued since its producer.
    private const int MaxStall = 12, MaxVariableStall = 11;
    private const int AllScoreboards = 0x3f;

    /// <summary>Why the last RecomputeControl call returned null (diagnostics).</summary>
    [ThreadStatic] public static string? LastDecline;
    private const int InstructionBytes = 16;

    // Write scoreboards 0-2 and read scoreboards 3-5, each used round-robin. Sharing a scoreboard is always safe
    // (it is a counter: waiting on it waits for every instruction that set it), only sometimes slower.
    private const int WriteScoreboards = 3, ReadScoreboardBase = 3;
    private const int ScoreboardSetDelay = 2;

    // A scoreboard is a small counter: an instruction that would raise one past this many outstanding operations
    // first waits on it (an unrolled kernel can otherwise put hundreds of loads on one scoreboard, which corrupted a
    // tiled convolution's results intermittently).
    private const int ScoreboardDepth = 16;

    // BMOV (a convergence-barrier register move) completes on a fixed delay nothing else tracks: ptxas gives it up to
    // 11 cycles, and a tiled convolution whose BMOV issued its successor after 1 corrupted its output about one run in
    // three (10 of 10 clean with ptxas's stall kept on it alone).
    private const int BmovStall = 11;
    private const int GuardPredicateLatency = 12;

    /// <summary>The branch targets (instruction indices) of BRA and BSSY instructions in <paramref name="code"/>.</summary>
    public static HashSet<int> BranchTargets(IReadOnlyList<SassInstruction> code)
    {
        var targets = new HashSet<int>();
        for (int i = 0; i < code.Count; i++)
        {
            int op = code[i].Opcode & 0x1ff;
            if (op != 0x147 && op != 0x145) continue;   // BRA, BSSY
            // The byte offset from the next instruction: bits 32-81, signed.
            long offset = (long)code[i].Bits(32, 50);
            if ((offset & (1L << 49)) != 0) offset -= 1L << 50;
            long target = (i + 1) * (long)InstructionBytes + offset;
            if (target % InstructionBytes != 0 || target < 0 || target / InstructionBytes > code.Count)
                throw new InvalidOperationException("Branch at instruction " + i + " targets byte " + target + ".");
            targets.Add((int)(target / InstructionBytes));
        }
        return targets;
    }

    /// <summary>
    /// <paramref name="code"/> with every control word recomputed, or null when an instruction's opcode is unknown or
    /// a required delay exceeds what the stall counts can express.
    /// </summary>
    public static SassInstruction[]? RecomputeControl(IReadOnlyList<SassInstruction> code)
    {
        if (code is null) throw new ArgumentNullException(nameof(code));
        var effects = new SassEffects[code.Count];
        for (int i = 0; i < code.Count; i++)
            if (SassOpcodeTable.Decode(code[i]) is { } e) effects[i] = e;
            else
            {
                LastDecline = "instruction " + i + " has unknown opcode 0x" + code[i].Opcode.ToString("x3");
                return null;
            }
        var targets = BranchTargets(code);
        var result = new SassInstruction[code.Count];
        var pendingWrite = new Dictionary<int, int>();   // register -> write scoreboard covering it
        var pendingRead = new Dictionary<int, int>();    // register -> read scoreboard covering it
        var readyAt = new Dictionary<int, (long Ready, long Produced)>();   // register or predicate key -> readable at
        int nextWrite = 0, nextRead = 0;
        var outstanding = new int[6];
        long cycle = 0;   // issue cycle of the current instruction
        var scoreboardSetAt = new long[6];
        for (int s = 0; s < scoreboardSetAt.Length; s++) scoreboardSetAt[s] = long.MinValue / 2;
        var predicateWrittenAt = new Dictionary<int, long>();   // predicate key -> cycle of the compare
        var originalStall = new int[code.Count];
        var issueCycle = new long[code.Count];

        for (int i = 0; i < code.Count; i++)
        {
            var instruction = code[i];
            var e = effects[i];
            var reads = Registers(instruction, e.Reads);
            var writes = Registers(instruction, e.Writes);
            var predicateReads = Predicates(instruction, e.PredicateReads, guard: false);
            int guard = instruction.GuardPredicate == SassInstruction.TruePredicate ? -1
                : PredicateKey + instruction.GuardPredicate;
            var predicateWrites = Predicates(instruction, e.PredicateWrites, guard: false);
            bool boundary = targets.Contains(i) || e.Latency == SassLatency.Control;

            // Waits.
            int wait = 0;
            if (boundary)
            {
                wait = AllScoreboards;
            }
            else
            {
                foreach (int r in reads) if (pendingWrite.TryGetValue(r, out int s)) wait |= 1 << s;
                foreach (int r in writes)
                {
                    if (pendingWrite.TryGetValue(r, out int s)) wait |= 1 << s;
                    if (pendingRead.TryGetValue(r, out int t)) wait |= 1 << t;
                }
            }
            // The scoreboards this instruction will set, chosen now so a full one is waited on first.
            int writeBoard = SassInstruction.NoScoreboard, readBoard = SassInstruction.NoScoreboard;
            if (e.Latency == SassLatency.Variable)
            {
                if (writes.Count != 0)
                {
                    writeBoard = nextWrite++ % WriteScoreboards;
                    if (outstanding[writeBoard] >= ScoreboardDepth) wait |= 1 << writeBoard;
                }
                if (reads.Count != 0)
                {
                    readBoard = ReadScoreboardBase + nextRead++ % (6 - ReadScoreboardBase);
                    if (outstanding[readBoard] >= ScoreboardDepth) wait |= 1 << readBoard;
                }
            }
            if (wait != 0)
            {
                Clear(pendingWrite, wait);
                Clear(pendingRead, wait);
                for (int s = 0; s < outstanding.Length; s++) if ((wait >> s & 1) != 0) outstanding[s] = 0;
            }

            // Fixed-latency readiness: everything at a boundary; otherwise the registers and predicates it reads or
            // overwrites. The previous instruction's stall carries the delay.
            // Each constraint: the cycle this instruction may issue at, and the cycle of the instruction it waits for
            // (only stalls of instructions issued since then delay it usefully).
            long ready = cycle, produced = cycle;
            void Need(long at, long from)
            {
                if (at > ready) ready = at;
                if (at > cycle && from < produced) produced = from;
            }
            for (int s = 0; s < scoreboardSetAt.Length; s++)
                if ((wait >> s & 1) != 0) Need(scoreboardSetAt[s] + ScoreboardSetDelay, scoreboardSetAt[s]);
            if (boundary)
            {
                foreach (var at in readyAt.Values) Need(at.Ready, at.Produced);
                foreach (long at in predicateWrittenAt.Values) Need(at + GuardPredicateLatency, at);
            }
            else
            {
                foreach (int r in reads) if (readyAt.TryGetValue(r, out var at)) Need(at.Ready, at.Produced);
                foreach (int r in writes) if (readyAt.TryGetValue(r, out var at)) Need(at.Ready, at.Produced);
                foreach (int p in predicateReads) if (readyAt.TryGetValue(p, out var at)) Need(at.Ready, at.Produced);
                if (guard >= 0 && predicateWrittenAt.TryGetValue(guard, out long written))
                    Need(written + GuardPredicateLatency, written);
                foreach (int p in predicateWrites) if (readyAt.TryGetValue(p, out var at)) Need(at.Ready, at.Produced);
                if (e.ReadsAllPredicates)
                    foreach (var pair in readyAt) if (pair.Key >= PredicateKey) Need(pair.Value.Ready, pair.Value.Produced);
            }
            if (ready > cycle)
            {
                // Spread the delay over the instructions issued since the producer, latest first.
                long missing = ready - cycle;
                for (int k = i - 1; k >= 0 && missing > 0 && issueCycle[k] >= produced; k--)
                {
                    var control = result[k].Control;
                    int cap = effects[k].Latency == SassLatency.Variable ? MaxVariableStall : MaxStall;
                    int add = (int)Math.Min(missing, cap - control.Stall);
                    if (add <= 0) continue;
                    result[k].Control = control with { Stall = control.Stall + add };
                    missing -= add;
                }
                if (missing > 0)
                {
                    LastDecline = "instruction " + i + " needs " + missing + " more cycles than the stalls since cycle " + produced + " can hold";
                    return null;
                }
                cycle = ready;
            }
            issueCycle[i] = cycle;
            if (boundary)
            {
                readyAt.Clear();
                predicateWrittenAt.Clear();
            }

            // This instruction's own scoreboards.
            if (e.Latency == SassLatency.Variable)
            {
                if (writeBoard != SassInstruction.NoScoreboard)
                {
                    outstanding[writeBoard]++;
                    scoreboardSetAt[writeBoard] = cycle;
                    foreach (int r in writes) { pendingWrite[r] = writeBoard; readyAt.Remove(r); }
                }
                if (readBoard != SassInstruction.NoScoreboard)
                {
                    outstanding[readBoard]++;
                    scoreboardSetAt[readBoard] = cycle;
                    foreach (int r in reads) pendingRead[r] = readBoard;
                }
            }
            else if (e.Latency == SassLatency.Fixed)
            {
                foreach (int r in writes) readyAt[r] = (cycle + e.FixedLatency, cycle);
                foreach (int p in predicateWrites)
                {
                    readyAt[p] = (cycle + e.FixedLatency, cycle);
                    predicateWrittenAt[p] = cycle;
                }
            }

            var original = instruction.Control;
            var updated = instruction;
            int baseStall = e.Mnemonic == "BMOV" ? BmovStall : 1;
            updated.Control = new SassControl(baseStall, original.Yield, writeBoard, readBoard, wait, original.Reuse);
            result[i] = updated;
            originalStall[i] = original.Stall;
            cycle += baseStall;
        }
        // Reuse flags only where the stall did not grow (see the file header).
        for (int i = 0; i < result.Length; i++)
        {
            var control = result[i].Control;
            if (control.Reuse != 0 && control.Stall > originalStall[i]) result[i].Control = control with { Reuse = 0 };
        }
        return result;
    }

    // Predicates live in the same readiness map as registers, offset past the register file.
    private const int PredicateKey = 1000;

    private static List<int> Registers(in SassInstruction instruction, IReadOnlyList<SassRegisterOperand> operands)
    {
        var registers = new List<int>();
        foreach (var operand in operands)
        {
            int r = instruction.Register(operand.Field);
            if (r == SassInstruction.ZeroRegister) continue;
            for (int k = 0; k < operand.Width; k++) registers.Add(r + k);
        }
        return registers;
    }

    private static List<int> Predicates(in SassInstruction instruction, IReadOnlyList<int> fields, bool guard)
    {
        var predicates = new List<int>();
        if (guard && instruction.GuardPredicate != SassInstruction.TruePredicate)
            predicates.Add(PredicateKey + instruction.GuardPredicate);
        foreach (int field in fields)
        {
            int p = (int)instruction.Bits(field, 3);
            if (p != SassInstruction.TruePredicate) predicates.Add(PredicateKey + p);
        }
        return predicates;
    }

    private static void Clear(Dictionary<int, int> pending, int waitMask)
    {
        var cleared = new List<int>();
        foreach (var pair in pending) if ((waitMask >> pair.Value & 1) != 0) cleared.Add(pair.Key);
        foreach (int r in cleared) pending.Remove(r);
    }
}
