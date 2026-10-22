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
using System.Linq;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;

/// <summary>Recomputes SASS control words from the instructions' register and predicate effects.</summary>
public static class SassScheduler
{
    // ptxas never encodes a stall above 12 (12 only after a compare); a variable-latency instruction stalling 12 or
    // more faulted on sm_75 where 11 ran. So stalls stay within 12 (11 on variable-latency instructions), and a longer
    // delay is spread over the instructions issued since its producer.
    // (sm_86: ptxas encodes up to 13, after a compare whose predicate guards a branch 13 cycles later.)
    private const int MaxVariableStall = 11;
    private const int AllScoreboards = 0x3f;

    /// <summary>Why the last RecomputeControl call returned null (diagnostics).</summary>
    [ThreadStatic] public static string? LastDecline;
    private const int InstructionBytes = 16;

    // The six scoreboards (less the one cp.async reserves) form one pool for write and read scoreboards alike. Sharing
    // one is always safe (it is a counter: waiting on it waits for every instruction that set it), only sometimes
    // slower, so each operation goes where it adds the least waiting (ChooseBoard). Round-robin cost the register-
    // blocked GEMM 5% and a dense 3x3 convolution 14% on the RTX 3080: a shared load's reader waited out a global
    // load's latency, and a fragment's loads waited for the next fragment's.
    //
    // Expected completion, in cycles after issue, for that choice only (correctness never depends on it): a write
    // scoreboard releases when the result arrives, a read scoreboard when the sources have been read.
    private const int GlobalLoadCycles = 300, SharedLoadCycles = 30, OtherVariableCycles = 20, SourceReadCycles = 20;
    private const int ScoreboardSetDelay = 2;

    // A scoreboard is a small counter: an instruction that would raise one past this many outstanding operations
    // first waits on it (an unrolled kernel can otherwise put hundreds of loads on one scoreboard, which corrupted a
    // tiled convolution's results intermittently).
    private const int ScoreboardDepth = 16;

    // BMOV (a convergence-barrier register move) completes on a fixed delay nothing else tracks: ptxas gives it up to
    // 11 cycles, and a tiled convolution whose BMOV issued its successor after 1 corrupted its output about one run in
    // three (10 of 10 clean with ptxas's stall kept on it alone).
    //
    // BAR.SYNC likewise: on sm_75 ptxas gives all of them exactly 5 cycles, and a register-blocked GEMM whose BAR issued
    // its successor (an LDS of the tile the barrier publishes) after 1 read a stale tile -- its output mismatched in 3
    // of 9 configurations, and keeping ptxas's stall on BAR alone made all 9 bit-identical.
    //
    // These delays and the guard latency are per architecture. sm_75's were measured on the device. sm_86's are the
    // largest ptxas leaves over the catalog and GEMM kernels compiled for it (--sass-learn --arch 86: BAR 6, DEPBAR 4,
    // LDGDEPBAR 4, a guard read 13 cycles after its compare at the closest) and are floors, not yet run on an sm_86
    // device. Any other architecture is declined.
    //
    // Pipe issue spacing (sm_86): a warp's instruction occupies its execution pipe for several cycles (a 16-lane pipe
    // takes two for 32 threads), and the next instruction on the same pipe must issue after it. ptxas never leaves
    // less over the catalog and GEMM kernels (--sass-learn --arch 86 reports the fewest cycles per pipe): the integer
    // ALU, IMAD and the uniform ALU 2 cycles (IMAD.WIDE and IMAD.HI 4), the conversion/special-function unit 8. The
    // uniform pipe is not interlocked: a register-blocked GEMM whose UMOV issued 1 cycle after a UIMAD faulted on the
    // RTX 3080 (illegal address).
    //
    // Uniform operand reads (sm_86): a uniform-pipe instruction reads its uniform registers late, while UMOV and ULDC
    // write theirs early (2 cycles; the other uniform operations 4). So UMOV or ULDC overwriting a register that an
    // earlier uniform instruction reads must issue UniformEarlyWriteAfterRead cycles after it: ptxas never leaves fewer
    // than 3 (UIMAD, ULEA, USEL and USHF readers alike), the vector instructions that read a uniform register take 1,
    // and the GEMM above, given the pipe spacing, still faulted with a UMOV 2 cycles after the UIMAD reading its
    // register and ran with 3.
    //
    // Fixed latencies that differ from the table's (measured on sm_75): sm_86 reads a P2R result 4 cycles after it at
    // the closest (sm_75: 14), so 6 -- the ALU margin -- holds, where 14 could not be expressed before an overwrite of
    // the P2R's register one instruction later (a declined catalog kernel).
    //
    // Latency classes (sm_86): a fixed-latency result is readable sooner by a consumer in its own class. Over the 81
    // kernels ptxas compiles for sm_86, it reads FP32 results (FADD/FMUL/FFMA) 4 cycles on in FP32 consumers, integer-
    // ALU results 4 on in ALU consumers, IMAD results 4 on in IMADs, and 5 on across classes (FADD -> FSETP, IADD3 ->
    // IMAD, IMAD -> ISETP, FMUL -> FSEL) and into memory operands. The table's 6 (sm_75) cost a dense 3x3 convolution
    // -- FMUL -> FADD chains -- 5% on the RTX 3080.
    private sealed record Timing(int GuardPredicateLatency, int MaxStall, IReadOnlyDictionary<string, int> IssueStalls,
        IReadOnlyDictionary<string, (string Pipe, int Cycles)> Pipes, int UniformEarlyWriteAfterRead = 0,
        IReadOnlyDictionary<string, int>? Latencies = null, IReadOnlyDictionary<string, string>? LatencyClasses = null,
        int SameClassLatency = 0, int CrossClassLatency = 0);

    private static readonly Dictionary<string, string> s_ampereLatencyClasses = new()
    {
        ["FADD"] = "fp", ["FMUL"] = "fp", ["FFMA"] = "fp",
        ["IADD3"] = "alu", ["LEA"] = "alu", ["LOP3"] = "alu", ["MOV"] = "alu", ["SEL"] = "alu", ["SHF"] = "alu",
        ["FSEL"] = "alu", ["FMNMX"] = "alu", ["IMNMX"] = "alu", ["IABS"] = "alu", ["ISETP"] = "alu", ["FSETP"] = "alu",
        ["PLOP3"] = "alu",
        ["IMAD"] = "imad", ["IMAD.HI"] = "imad",
    };

    private static readonly Dictionary<string, (string Pipe, int Cycles)> s_noPipes = new();

    private static readonly Dictionary<string, (string Pipe, int Cycles)> s_ampere = new()
    {
        ["IADD3"] = ("alu", 2), ["LEA"] = ("alu", 2), ["LOP3"] = ("alu", 2), ["ISETP"] = ("alu", 2), ["MOV"] = ("alu", 2),
        ["SEL"] = ("alu", 2), ["SHF"] = ("alu", 2), ["IABS"] = ("alu", 2), ["IMNMX"] = ("alu", 4), ["PLOP3"] = ("alu", 2),
        ["P2R"] = ("alu", 2), ["I2FP"] = ("alu", 2), ["FSEL"] = ("alu", 2), ["FSETP"] = ("alu", 2), ["FMNMX"] = ("alu", 2),
        ["IMAD"] = ("imad", 2), ["IMAD.WIDE"] = ("imad", 4), ["IMAD.HI"] = ("imad", 4),
        ["UMOV"] = ("uniform", 2), ["USEL"] = ("uniform", 2), ["UISETP"] = ("uniform", 2), ["UIADD3"] = ("uniform", 2),
        ["ULEA"] = ("uniform", 2), ["ULOP3"] = ("uniform", 2), ["USHF"] = ("uniform", 2), ["UIMAD"] = ("uniform", 2),
        ["UIMAD.WIDE"] = ("uniform", 2), ["ULDC"] = ("uniform", 2),
        ["MUFU"] = ("xu", 8), ["I2F"] = ("xu", 8), ["F2I"] = ("xu", 8),
    };

    private static readonly Dictionary<int, Timing> s_timings = new()
    {
        [75] = new(12, 12, new Dictionary<string, int> { ["BMOV"] = 11, ["BAR"] = 5 }, s_noPipes),
        [86] = new(13, 13, new Dictionary<string, int> { ["BMOV"] = 11, ["BAR"] = 6, ["DEPBAR"] = 4, ["LDGDEPBAR"] = 4 }, s_ampere, 3,
            new Dictionary<string, int> { ["P2R"] = 6, ["IMAD.WIDE"] = 5, ["I2FP"] = 5 }, s_ampereLatencyClasses, 4, 5),
    };

    /// <summary>
    /// The pipe <paramref name="mnemonic"/> issues to on <paramref name="architecture"/> and the fewest cycles before
    /// the next instruction on that pipe may issue, or null when the scheduler does not space it.
    /// </summary>
    public static (string Pipe, int Cycles)? PipeIssue(string mnemonic, int architecture) =>
        s_timings.TryGetValue(architecture, out var t) && t.Pipes.TryGetValue(mnemonic, out var p) ? p : null;

    /// <summary>The SM versions (major * 10 + minor) whose timings the scheduler has.</summary>
    public static IReadOnlyCollection<int> SupportedArchitectures => s_timings.Keys;

    /// <summary>The branch targets (instruction indices) of BRA, BSSY and CALL instructions in <paramref name="code"/>.</summary>
    public static HashSet<int> BranchTargets(IReadOnlyList<SassInstruction> code)
    {
        var targets = new HashSet<int>();
        for (int i = 0; i < code.Count; i++)
        {
            int op = code[i].Opcode & 0x1ff;
            if (op != 0x147 && op != 0x145 && op != 0x144) continue;   // BRA, BSSY, CALL.REL.NOINC
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
    /// a required delay exceeds what the stall counts can express, or <paramref name="architecture"/> (SM version,
    /// major * 10 + minor) is not one the scheduler has timings for.
    /// </summary>
    public static SassInstruction[]? RecomputeControl(IReadOnlyList<SassInstruction> code, int architecture = 75)
    {
        if (code is null) throw new ArgumentNullException(nameof(code));
        if (!s_timings.TryGetValue(architecture, out var timing))
        {
            LastDecline = "no timings for sm_" + architecture;
            return null;
        }
        int guardLatency = timing.GuardPredicateLatency;
        var effects = new SassEffects[code.Count];
        for (int i = 0; i < code.Count; i++)
            if (SassOpcodeTable.Decode(code[i]) is { } e) effects[i] = e;
            else
            {
                LastDecline = "instruction " + i + " has unknown opcode 0x" + code[i].Opcode.ToString("x3");
                return null;
            }
        var targets = BranchTargets(code);
        if (AsyncScoreboard(code, effects) is not int asyncBoard) return null;
        var overwritten = SourcesOverwritten(code, effects);
        var (firstUse, firstOverwrite) = NextHazards(code, effects, targets);
        int asyncMask = asyncBoard >= 0 ? 1 << asyncBoard : 0;
        var result = new SassInstruction[code.Count];
        var pendingWrite = new Dictionary<int, int>();   // register -> write scoreboard covering it
        var pendingRead = new Dictionary<int, int>();    // register -> read scoreboard covering it
        var readyAt = new Dictionary<int, (long Ready, long Produced)>();   // register or predicate key -> readable at
        var earlyAt = new Dictionary<int, (long Ready, string Class)>();     // register -> readable at, by its own class
        var pendingUse = new long[6];        // per scoreboard: the earliest first use among the operations it covers
        var pendingComplete = new long[6];   // per scoreboard: the latest expected completion among them
        for (int s = 0; s < pendingUse.Length; s++) pendingUse[s] = long.MaxValue;
        var boards = Enumerable.Range(0, 6).Where(b => b != asyncBoard).ToArray();
        var outstanding = new int[6];
        long cycle = 0;   // issue cycle of the current instruction
        var scoreboardSetAt = new long[6];
        for (int s = 0; s < scoreboardSetAt.Length; s++) scoreboardSetAt[s] = long.MinValue / 2;
        var predicateWrittenAt = new Dictionary<int, long>();   // predicate key -> cycle of the compare
        var pipeFreeAt = new Dictionary<string, (long Ready, long Produced)>();   // pipe -> next issue cycle allowed
        var uniformReadAt = new Dictionary<int, long>();   // uniform register key -> latest uniform-pipe reader's cycle
        var originalStall = new int[code.Count];
        var issueCycle = new long[code.Count];

        for (int i = 0; i < code.Count; i++)
        {
            var instruction = code[i];
            var e = effects[i];
            var reads = Registers(instruction, e.Reads);
            var writes = Registers(instruction, e.Writes);
            var predicateReads = Predicates(instruction, e.PredicateReads, e.UniformPredicates);
            predicateReads.AddRange(Predicates(instruction, e.UniformPredicateReads, uniform: true));
            int guard = instruction.GuardPredicate == SassInstruction.TruePredicate ? -1
                : (e.UniformPredicates ? UniformPredicateKey : PredicateKey) + instruction.GuardPredicate;
            var predicateWrites = Predicates(instruction, e.PredicateWrites, e.UniformPredicates);
            bool boundary = targets.Contains(i) || e.Latency == SassLatency.Control;

            // Waits.
            int wait = 0;
            if (boundary)
            {
                wait = AllScoreboards & ~asyncMask;
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
                    long complete = cycle + (e.Mnemonic == "LDG" ? GlobalLoadCycles : e.Mnemonic == "LDS" ? SharedLoadCycles : OtherVariableCycles);
                    writeBoard = ChooseBoard(boards, outstanding, pendingUse, pendingComplete, firstUse[i], complete);
                    pendingUse[writeBoard] = Math.Min(pendingUse[writeBoard], firstUse[i]);
                    pendingComplete[writeBoard] = Math.Max(pendingComplete[writeBoard], complete);
                    if (outstanding[writeBoard] >= ScoreboardDepth) wait |= 1 << writeBoard;
                }
                if (reads.Count != 0 && overwritten[i])
                {
                    long released = cycle + SourceReadCycles;
                    readBoard = ChooseBoard(boards.Where(b => b != writeBoard).ToArray(), outstanding, pendingUse, pendingComplete,
                        firstOverwrite[i], released);
                    pendingUse[readBoard] = Math.Min(pendingUse[readBoard], firstOverwrite[i]);
                    pendingComplete[readBoard] = Math.Max(pendingComplete[readBoard], released);
                    if (outstanding[readBoard] >= ScoreboardDepth) wait |= 1 << readBoard;
                }
            }
            if (wait != 0)
            {
                Clear(pendingWrite, wait);
                Clear(pendingRead, wait);
                for (int s = 0; s < outstanding.Length; s++)
                    if ((wait >> s & 1) != 0)
                    {
                        outstanding[s] = 0;
                        pendingUse[s] = long.MaxValue;
                        pendingComplete[s] = 0;
                    }
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
            timing.Pipes.TryGetValue(e.Mnemonic, out var pipe);
            if (boundary)
            {
                foreach (var at in readyAt.Values) Need(at.Ready, at.Produced);
                foreach (long at in predicateWrittenAt.Values) Need(at + guardLatency, at);
                foreach (var at in pipeFreeAt.Values) Need(at.Ready, at.Produced);
                foreach (long at in uniformReadAt.Values) Need(at + timing.UniformEarlyWriteAfterRead, at);
            }
            else
            {
                string? consumerClass = timing.LatencyClasses is { } lc && lc.TryGetValue(e.Mnemonic, out var cc) ? cc : null;
                foreach (int r in reads)
                    if (readyAt.TryGetValue(r, out var at))
                        Need(consumerClass is not null && earlyAt.TryGetValue(r, out var early) && early.Class == consumerClass ? early.Ready : at.Ready, at.Produced);
                foreach (int r in writes) if (readyAt.TryGetValue(r, out var at)) Need(at.Ready, at.Produced);
                foreach (int p in predicateReads) if (readyAt.TryGetValue(p, out var at)) Need(at.Ready, at.Produced);
                if (guard >= 0 && predicateWrittenAt.TryGetValue(guard, out long written))
                    Need(written + guardLatency, written);
                foreach (int p in predicateWrites) if (readyAt.TryGetValue(p, out var at)) Need(at.Ready, at.Produced);
                if (e.ReadsAllPredicates)
                    foreach (var pair in readyAt) if (pair.Key >= PredicateKey && pair.Key < UniformPredicateKey) Need(pair.Value.Ready, pair.Value.Produced);
                if (pipe.Pipe is not null && pipeFreeAt.TryGetValue(pipe.Pipe, out var free)) Need(free.Ready, free.Produced);
                if (e.Mnemonic is "UMOV" or "ULDC")
                    foreach (int r in writes) if (uniformReadAt.TryGetValue(r, out long read)) Need(read + timing.UniformEarlyWriteAfterRead, read);
            }
            if (ready > cycle)
            {
                // Spread the delay over the instructions issued since the producer, latest first.
                long missing = ready - cycle;
                for (int k = i - 1; k >= 0 && missing > 0 && issueCycle[k] >= produced; k--)
                {
                    var control = result[k].Control;
                    int cap = effects[k].Latency == SassLatency.Variable ? MaxVariableStall : timing.MaxStall;
                    int add = (int)Math.Min(missing, cap - control.Stall);
                    if (add <= 0) continue;
                    result[k].Control = control with { Stall = control.Stall + add };
                    missing -= add;
                }
                if (missing > 0)
                {
                    int from = i - 1;
                    while (from > 0 && issueCycle[from - 1] >= produced) from--;
                    LastDecline = "instruction " + i + " (" + e.Mnemonic + ") needs " + missing + " more cycles than the stalls since cycle " +
                        produced + " (instruction " + from + ", " + effects[from].Mnemonic + ") can hold";
                    return null;
                }
                cycle = ready;
            }
            issueCycle[i] = cycle;
            if (boundary)
            {
                readyAt.Clear();
                earlyAt.Clear();
                predicateWrittenAt.Clear();
                pipeFreeAt.Clear();
                uniformReadAt.Clear();
            }
            if (pipe.Pipe is not null) pipeFreeAt[pipe.Pipe] = (cycle + pipe.Cycles, cycle);
            if (timing.UniformEarlyWriteAfterRead > 0 && pipe.Pipe == "uniform")
                foreach (int r in reads) if (r >= UniformRegisterKey) uniformReadAt[r] = cycle;

            // This instruction's own scoreboards.
            if (e.Mnemonic == "LDGDEPBAR")
            {
                // Commits the outstanding copies to the scoreboard the DEPBARs count; only they wait on it.
                writeBoard = asyncBoard;
                scoreboardSetAt[asyncBoard] = cycle;
            }
            else if (e.Latency == SassLatency.Variable)
            {
                if (writeBoard != SassInstruction.NoScoreboard)
                {
                    outstanding[writeBoard]++;
                    scoreboardSetAt[writeBoard] = cycle;
                    foreach (int r in writes) { pendingWrite[r] = writeBoard; readyAt.Remove(r); earlyAt.Remove(r); }
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
                string? producerClass = timing.LatencyClasses is { } pc && pc.TryGetValue(e.Mnemonic, out var cls) ? cls : null;
                int latency = timing.Latencies is { } l && l.TryGetValue(e.Mnemonic, out int own) ? own
                    : producerClass is not null ? timing.CrossClassLatency : e.FixedLatency;
                foreach (int r in writes)
                {
                    readyAt[r] = (cycle + latency, cycle);
                    if (producerClass is not null) earlyAt[r] = (cycle + timing.SameClassLatency, producerClass);
                    else earlyAt.Remove(r);
                }
                foreach (int p in predicateWrites)
                {
                    readyAt[p] = (cycle + latency, cycle);
                    predicateWrittenAt[p] = cycle;
                }
            }

            var original = instruction.Control;
            var updated = instruction;
            int baseStall = timing.IssueStalls.TryGetValue(e.Mnemonic, out int fixedStall) ? fixedStall : 1;
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

    // Predicates, uniform predicates and uniform registers live in the same readiness map as vector registers, each
    // offset past the one before.
    private const int PredicateKey = 1000, UniformPredicateKey = 1008, UniformRegisterKey = 2000;

    // cp.async (sm_80+): LDGDEPBAR commits copies to a scoreboard and DEPBAR.LE SBn, k waits on that scoreboard by
    // number (bits 44-46), so the number is part of the code, not the scheduler's to choose. It is reserved for them:
    // no other instruction is given it and a block boundary does not wait on it (that would drain the copy pipeline
    // the DEPBARs deliberately keep k groups deep). -1 when the kernel has neither; null (declined) when they do not
    // name exactly one scoreboard between them.
    private static int? AsyncScoreboard(IReadOnlyList<SassInstruction> code, SassEffects[] effects)
    {
        var boards = new HashSet<int>();
        bool commits = false;
        for (int i = 0; i < code.Count; i++)
        {
            if (effects[i].Mnemonic == "DEPBAR") boards.Add((int)code[i].Bits(44, 3));
            commits |= effects[i].Mnemonic == "LDGDEPBAR";
        }
        if (boards.Count == 0 && !commits) return -1;
        if (boards.Count == 1 && commits) return boards.First();
        LastDecline = "cp.async: DEPBARs name " + boards.Count + " scoreboards" + (commits ? "" : " and nothing commits to them");
        return null;
    }

    // The scoreboard for an operation first needed at instruction `use` and expected to complete at cycle `complete`.
    // The next wait on a scoreboard comes at the earliest use among the operations it covers and lasts until all of them
    // complete, so adding this operation costs that wait max(pending completion, complete) minus the completion of
    // what it actually needs (this operation if its use comes first, else the pending ones). The cheapest wins; ties
    // go to the scoreboard holding fewer operations. (Uses are instruction indices, compared only with each other.)
    private static int ChooseBoard(int[] candidates, int[] outstanding, long[] pendingUse, long[] pendingComplete,
        long use, long complete)
    {
        int best = -1;
        long bestCost = long.MaxValue;
        foreach (int b in candidates)
        {
            long cost;
            if (outstanding[b] == 0) cost = 0;
            else
            {
                long end = Math.Max(pendingComplete[b], complete);
                long needed = pendingUse[b] < use ? pendingComplete[b] : use < pendingUse[b] ? complete : Math.Max(pendingComplete[b], complete);
                cost = end - needed;
            }
            if (cost < bestCost || (cost == bestCost && outstanding[b] < outstanding[best]))
            {
                best = b;
                bestCost = cost;
            }
        }
        return best;
    }

    // Per variable-latency instruction: the first later instruction that reads or writes a register it writes, and the
    // first that writes a register it reads, before the next block boundary (which waits on every scoreboard; it
    // stands for both when nothing in the block does).
    private static (long[] FirstUse, long[] FirstOverwrite) NextHazards(IReadOnlyList<SassInstruction> code,
        SassEffects[] effects, HashSet<int> targets)
    {
        var firstUse = new long[code.Count];
        var firstOverwrite = new long[code.Count];
        for (int i = 0; i < code.Count; i++)
        {
            if (effects[i].Latency != SassLatency.Variable) continue;
            var writes = new HashSet<int>(Registers(code[i], effects[i].Writes));
            var reads = new HashSet<int>(Registers(code[i], effects[i].Reads));
            long use = -1, overwrite = -1;
            int j = i + 1;
            for (; j < code.Count && (use < 0 || overwrite < 0); j++)
            {
                if (targets.Contains(j) || effects[j].Latency == SassLatency.Control) break;
                var laterWrites = Registers(code[j], effects[j].Writes);
                if (use < 0 && (laterWrites.Exists(writes.Contains) || Registers(code[j], effects[j].Reads).Exists(writes.Contains))) use = j;
                if (overwrite < 0 && laterWrites.Exists(reads.Contains)) overwrite = j;
            }
            firstUse[i] = use >= 0 ? use : j;
            firstOverwrite[i] = overwrite >= 0 ? overwrite : j;
        }
        return (firstUse, firstOverwrite);
    }

    // Per instruction: whether any instruction that can execute after it writes a register it reads -- a later one,
    // or an earlier one inside a loop (a backward branch's range) containing it. Only then does a variable-latency
    // instruction need a read scoreboard: nothing else can overwrite its sources before it has read them. ptxas
    // likewise leaves most shared loads without one; giving every LDS of the register-blocked GEMM a read scoreboard
    // filled them to their depth and cost it 5% on the RTX 3080.
    private static bool[] SourcesOverwritten(IReadOnlyList<SassInstruction> code, SassEffects[] effects)
    {
        var writers = new Dictionary<int, List<int>>();
        for (int i = 0; i < code.Count; i++)
            foreach (int r in Registers(code[i], effects[i].Writes))
            {
                if (!writers.TryGetValue(r, out var list)) writers[r] = list = new List<int>();
                list.Add(i);
            }
        var loops = new List<(int Start, int End)>();
        for (int i = 0; i < code.Count; i++)
        {
            int op = code[i].Opcode & 0x1ff;
            if (op != 0x147 && op != 0x144) continue;   // BRA, CALL.REL.NOINC
            long offset = (long)code[i].Bits(32, 50);
            if ((offset & (1L << 49)) != 0) offset -= 1L << 50;
            long target = ((i + 1) * (long)InstructionBytes + offset) / InstructionBytes;
            if (target <= i) loops.Add(((int)target, i));
        }
        var result = new bool[code.Count];
        for (int i = 0; i < code.Count; i++)
        {
            if (effects[i].Latency != SassLatency.Variable) continue;
            int earliest = i;
            foreach (var loop in loops) if (loop.Start <= i && i <= loop.End && loop.Start < earliest) earliest = loop.Start;
            foreach (int r in Registers(code[i], effects[i].Reads))
                if (writers.TryGetValue(r, out var list) && list.Exists(w => w != i && (w > i || w >= earliest)))
                {
                    result[i] = true;
                    break;
                }
        }
        return result;
    }

    private static List<int> Registers(in SassInstruction instruction, IReadOnlyList<SassRegisterOperand> operands)
    {
        var registers = new List<int>();
        foreach (var operand in operands)
        {
            if (operand.Uniform)
            {
                int u = instruction.UniformRegister(operand.Field);
                if (u == SassInstruction.ZeroUniformRegister) continue;
                for (int k = 0; k < operand.Width; k++) registers.Add(UniformRegisterKey + u + k);
                continue;
            }
            int r = instruction.Register(operand.Field);
            if (r == SassInstruction.ZeroRegister) continue;
            for (int k = 0; k < operand.Width; k++) registers.Add(r + k);
        }
        return registers;
    }

    private static List<int> Predicates(in SassInstruction instruction, IReadOnlyList<int> fields, bool uniform)
    {
        var predicates = new List<int>();
        foreach (int field in fields)
        {
            int p = (int)instruction.Bits(field, 3);
            if (p != SassInstruction.TruePredicate) predicates.Add((uniform ? UniformPredicateKey : PredicateKey) + p);
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
