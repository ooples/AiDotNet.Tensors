// Copyright (c) CrowdTrainer LLC. All rights reserved.
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
// Control flow: a branch target or a control instruction issues only after every fixed-latency result is ready, so no
// fixed-latency state crosses a block boundary. Scoreboards do cross branches: a branch target starts from the union of
// what every edge into it leaves pending (loop heads iterated to a fixpoint), so a loop keeps its prefetched global
// loads in flight across the back edge as ptxas does -- waiting on every scoreboard there cost a 3x3 convolution's data
// gradient 12% on the RTX 3080 (CALL.REL.NOINC, a loop exit, counts as a branch). The other control instructions
// (barriers, EXIT, convergence, DEPBAR) still wait on every scoreboard. BSSY regions (sm_86), all found on the RTX 3080
// with every register dependency already waited for: BSSY issues its divergent branch no sooner than 12 cycles on
// (ptxas never less; 1 made dense 3x3 convolutions' weight staging compute wrong values, 6 was the fewest that ran);
// the scoreboards holding read releases or other short operations are waited for at the BSSY (left pending, a dilated
// 3x3 convolution's index arithmetic inside the region came out wrong for every lane past the first channel); global
// loads stay in flight across the region, as ptxas leaves them, and their scoreboards are not touched until after the
// reconvergence target. sm_75 keeps the full wait at BSSY. The yield bits stay as they were. An operand reuse flag
// survives only where the instruction's stall is no longer than ptxas's: the reuse cache is valid only when the next
// instruction follows closely enough that the scheduler cannot have issued another warp in between -- lengthening the
// stall of an instruction that keeps its reuse flags faulted on sm_75 (ptxas's own schedule with four extra stall
// cycles on its memory instructions read stale operands).
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

    // The choice is greedy, and an empty scoreboard always costs nothing, so a run of global loads would take every
    // scoreboard and leave each later short operation (an address register's read release, a shared load) to share one
    // with a 300-cycle load. Global loads therefore stop opening new scoreboards once only ShortBoards are free of them,
    // as ptxas keeps two for short operations. A depthwise 3x3 weight gradient (20 gathered loads an iteration, each
    // address overwritten soon after) ran 0.79x ptxas on the RTX 3080 without this.
    private const int ShortBoards = 2;
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
    // LDGDEPBAR 4, a guard read 13 cycles after its compare at the closest) and are floors; with the rules below they
    // ran the catalog and GEMM kernels bit-identically on an RTX 3080. Any other architecture is declined.
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
        int SameClassLatency = 0, int CrossClassLatency = 0, int R2urToVectorUniformRead = 0, bool RegionsKeepLoads = false, int MmaSourceReadCycles = 0);

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
        // The tensor pipe: ptxas never issues an HMMA (m16n8k8 TF32 or m16n8k16 FP16) sooner than 8 cycles after the
        // previous one, and FP16 GEMM results came out wrong on the RTX 3080 with them 1 cycle apart.
        ["HMMA"] = ("tensor", 8),
    };

    private static readonly Dictionary<int, Timing> s_timings = new()
    {
        [75] = new(12, 12, new Dictionary<string, int> { ["BMOV"] = 11, ["BAR"] = 5 }, s_noPipes),
        [86] = new(13, 13, new Dictionary<string, int> { ["BMOV"] = 11, ["BAR"] = 6, ["DEPBAR"] = 4, ["LDGDEPBAR"] = 4, ["BSSY"] = 12 },
            s_ampere, 3, new Dictionary<string, int> { ["P2R"] = 6, ["IMAD.WIDE"] = 5, ["I2FP"] = 5 }, s_ampereLatencyClasses, 4, 5, R2urToVectorUniformRead: 5,
            RegionsKeepLoads: true, MmaSourceReadCycles: 7),
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
        var (firstUse, firstOverwrite, useInBlock) = NextHazards(code, effects, targets);
        int asyncMask = asyncBoard >= 0 ? 1 << asyncBoard : 0;
        var boards = Enumerable.Range(0, 6).Where(b => b != asyncBoard).ToArray();
        int[] shortBoards = boards.Take(ShortBoards).ToArray(), longBoards = boards.Skip(ShortBoards).ToArray();

        // Scoreboards stay pending across branches (see ScoreState): a branch target starts from the union of what is
        // pending on every edge into it. Forward edges are known when the target is reached; a loop head's back edges
        // come from the previous pass, repeated until no back edge brings anything its head did not assume. A head that
        // does not settle within MaxPasses waits on every scoreboard instead (the conservative schedule).
        var backEntry = new Dictionary<int, ScoreState>();
        var drainTargets = new HashSet<int>();
        // Stalls a previous pass had to spread back PAST the instruction before the one that needed them. Such a stall
        // delays every instruction in between, so that pass's issue cycles -- and the readiness, predicate, pipe,
        // uniform-read and scoreboard state recorded at them -- are stale from that point on, and a later reader
        // computed from them could issue before its producer's result is ready. Rather than patch every entry, the
        // next pass starts from these stalls and computes every cycle with them already in place.
        var minimumStall = new int[code.Count];
        int spreadPasses = 0;
        for (int pass = 0; ; pass++)
        {
            bool spreadPastPrevious = false;
            var forwardEntry = new Dictionary<int, ScoreState>();
            var backOut = new Dictionary<int, ScoreState>();
            var result = new SassInstruction[code.Count];
            var pendingWrite = new Dictionary<int, int>();   // register -> mask of write scoreboards covering it
            var pendingRead = new Dictionary<int, int>();    // register -> mask of read scoreboards covering it
            var readyAt = new Dictionary<int, (long Ready, long Produced)>();   // register or predicate key -> readable at
            var earlyAt = new Dictionary<int, (long Ready, string Class)>();     // register -> readable at, by its own class
            // Registers whose pending readiness came from a Tensor-Core MMA: carried across branches (CarriesAcrossBranches)
            // instead of waited for at them.
            var carried = new HashSet<int>();
            var pendingUse = new long[6];        // per scoreboard: the earliest first use among the operations it covers
            var pendingComplete = new long[6];   // per scoreboard: the latest expected completion among them
            for (int s = 0; s < pendingUse.Length; s++) pendingUse[s] = long.MaxValue;
            var outstanding = new int[6];
            var shortOps = new bool[6];   // per scoreboard: holds something other than a global load's result
            // Per OPEN BSSY region: the scoreboards it keeps in flight and the instruction its reconvergence ends at.
            // One pair would not do: an inner region that reconverges first would replace the outer region's set, and
            // the outer BSYNC would then wait for a global load nothing needs until after the outer region - exactly
            // the overlap this path exists to keep.
            var frozenRegions = new List<(int Mask, int Until)>();
            int frozen = 0;
            long cycle = 0;   // issue cycle of the current instruction
            var scoreboardSetAt = new long[6];
            for (int s = 0; s < scoreboardSetAt.Length; s++) scoreboardSetAt[s] = long.MinValue / 2;
            var predicateWrittenAt = new Dictionary<int, long>();   // predicate key -> cycle of the compare
            var pipeFreeAt = new Dictionary<string, (long Ready, long Produced)>();   // pipe -> next issue cycle allowed
            var uniformReadAt = new Dictionary<int, long>();
            long r2urAt = long.MinValue / 2;   // issue cycle of the last R2UR
            // An HMMA reads its A, B and C registers several cycles after it issues, and nothing interlocks a later
            // write to them: register -> the cycle it may be overwritten from (MmaSourceReadCycles after the HMMA).
            var mmaReadUntil = new Dictionary<int, long>();   // uniform register key -> latest uniform-pipe reader's cycle
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
                bool target = targets.Contains(i);
                // Regions close in any order; what is frozen here is the union of those still open.
                frozenRegions.RemoveAll(region => i >= region.Until);
                frozen = 0;
                foreach (var region in frozenRegions) frozen |= region.Mask;
                // Fixed-latency state never crosses a block boundary; scoreboards cross branches (BRA, and CALL.REL.NOINC,
                // which ptxas leaves a loop through) and branch targets, but every other control instruction (barriers,
                // EXIT, convergence, DEPBAR) waits on them all.
                bool boundary = target || e.Latency == SassLatency.Control;
                bool branch = e.Mnemonic is "BRA" or "CALL";
                bool drain = (e.Latency == SassLatency.Control && !branch) || (target && drainTargets.Contains(i));
                // An HMMA's 56-cycle result outlives the few instructions between a loop's last MMA and its branch, and
                // its readers (the next trip's MMAs, the epilogue) come long after; ptxas branches right away. So at a
                // branch to a target that is not drained, and at such a target, MMA results are not waited for: the
                // branch hands them to its target as cycles still to wait, the fall-through keeps them, and readers
                // wait as usual. NOP, DEPBAR and BAR pass them on too: they read and write no register, and the
                // readiness they skip is kept past them (see the boundary reset below), so the first reader after them
                // still waits. Every other fixed-latency result, and every other control instruction (EXIT, BSSY,
                // BSYNC, ...), still waits.
                bool fallThroughOnly = e.Mnemonic is "NOP" or "DEPBAR" or "BAR";   // touch no registers, no control transfer
                bool canCarry = (branch || fallThroughOnly || (target && e.Latency != SassLatency.Control))
                    && !(branch && drainTargets.Contains(BranchTarget(code, i))) && !(target && drainTargets.Contains(i));
                if (target && !drain)
                {
                    if (forwardEntry.TryGetValue(i, out var forward)) Merge(forward, clampSetAt: false, at: i);
                    if (backEntry.TryGetValue(i, out var back)) Merge(back, clampSetAt: true, at: i);
                }

                // Waits.
                int wait = 0;
                if (drain && e.Mnemonic == "BSSY" && timing.RegionsKeepLoads)
                {
                    // A BSSY region may keep global loads in flight, but nothing else: the scoreboards holding read
                    // releases or short operations are waited for first, and the ones left in flight are neither set
                    // nor waited on (BSYNC and the reconvergence target included) until the region has closed.
                    for (int b = 0; b < 6; b++) if (shortOps[b]) wait |= 1 << b;
                    int end = BranchTarget(code, i);
                    wait &= ~asyncMask;
                    int keep = 0;
                    for (int b = 0; b < 6; b++) if (outstanding[b] != 0 && (wait >> b & 1) == 0) keep |= 1 << b;
                    frozenRegions.Add((keep, end + 1));   // through the reconvergence target
                    frozen |= keep;
                }
                else if (drain)
                {
                    // BSYNC closing a region leaves the loads from before it pending (ptxas waits for them at their
                    // first reader after the region).
                    wait = AllScoreboards & ~asyncMask & ~frozen;
                }
                else
                {
                    foreach (int r in reads) if (pendingWrite.TryGetValue(r, out int m)) wait |= m;
                    foreach (int r in writes)
                    {
                        if (pendingWrite.TryGetValue(r, out int m)) wait |= m;
                        if (pendingRead.TryGetValue(r, out int n)) wait |= n;
                    }
                }
                // The scoreboards this instruction will set, chosen now so a full one is waited on first.
                int writeBoard = SassInstruction.NoScoreboard, readBoard = SassInstruction.NoScoreboard;
                long writeComplete = 0, readReleased = 0;
                bool global = e.Mnemonic == "LDG";
                if (e.Latency == SassLatency.Variable)
                {
                    if (writes.Count != 0)
                    {
                        long complete = cycle + (global ? GlobalLoadCycles : e.Mnemonic is "LDS" or "LDSM" ? SharedLoadCycles : OtherVariableCycles);
                        writeBoard = ChooseBoard(Unfrozen(global ? longBoards : boards), outstanding, pendingUse, pendingComplete, firstUse[i], complete, conserve: global);
                        writeComplete = complete;
                        if (outstanding[writeBoard] >= ScoreboardDepth) wait |= 1 << writeBoard;
                    }
                    // No read scoreboard when nothing can overwrite a source, or when the first use of the result comes, in
                    // the same straight-line block, no later than the first overwrite: that use waits on the write
                    // scoreboard, whose release implies the sources were read. ptxas likewise leaves most loads without one.
                    if (reads.Count != 0 && overwritten[i] && !(useInBlock[i] && firstUse[i] <= firstOverwrite[i]))
                    {
                        readReleased = cycle + SourceReadCycles;
                        readBoard = ChooseBoard(Unfrozen(shortBoards), outstanding, pendingUse, pendingComplete,
                            firstOverwrite[i], readReleased);
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
                            shortOps[s] = false;
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
                    // A carried MMA result is skipped only where nothing here needs it: this instruction (a branch target
                    // may itself be the reader, such as a loop-head MMA on the previous trip's accumulator) and, at a
                    // branch, the target's first instruction, which has no instruction before it in its block to hold
                    // the delay - so it is waited for here instead (#1540).
                    HashSet<int>? touched = null;
                    if (canCarry)
                    {
                        touched = new HashSet<int>(reads);
                        touched.UnionWith(writes);
                        touched.UnionWith(predicateReads);
                        touched.UnionWith(predicateWrites);
                        if (guard >= 0) touched.Add(guard);
                        if (branch)
                        {
                            int to = BranchTarget(code, i);
                            var te = effects[to];
                            touched.UnionWith(Registers(code[to], te.Reads));
                            touched.UnionWith(Registers(code[to], te.Writes));
                        }
                    }
                    foreach (var pair in readyAt)
                        if (!(canCarry && carried.Contains(pair.Key) && touched is not null && !touched.Contains(pair.Key)))
                            Need(pair.Value.Ready, pair.Value.Produced);
                    foreach (long at in predicateWrittenAt.Values) Need(at + guardLatency, at);
                    foreach (var at in pipeFreeAt.Values) Need(at.Ready, at.Produced);
                    foreach (long at in uniformReadAt.Values) Need(at + timing.UniformEarlyWriteAfterRead, at);
                    foreach (long at in mmaReadUntil.Values) Need(at, at - timing.MmaSourceReadCycles);
                }
                else
                {
                    // sm_86: an IMAD.MOV overwriting the A operand of the HMMA issued 2 cycles earlier made an FP16 GEMM's
                    // results wrong on the RTX 3080; ptxas never overwrites an HMMA source sooner than 7 cycles on.
                    foreach (int r in writes) if (mmaReadUntil.TryGetValue(r, out long until)) Need(until, until - timing.MmaSourceReadCycles);
                    string? consumerClass = timing.LatencyClasses is { } lc && lc.TryGetValue(e.Mnemonic, out var cc) ? cc : null;
                    foreach (int r in reads)
                        if (readyAt.TryGetValue(r, out var at))
                            Need(consumerClass is not null && earlyAt.TryGetValue(r, out var early) && early.Class == consumerClass ? early.Ready : at.Ready, at.Produced);
                    foreach (int r in writes) if (readyAt.TryGetValue(r, out var at)) Need(at.Ready, at.Produced);
                    foreach (int p in predicateReads) if (readyAt.TryGetValue(p, out var at)) Need(at.Ready, at.Produced);
                    // A variable-latency instruction reads a predicate operand as early as a guard: cp.async's zero-fill predicate
                    // (LDGSTS ..., P5) issued 5 cycles after the ISETP writing it made the TF32 Tensor-Core GEMM's results wrong on
                    // the RTX 3080; ptxas leaves the guard latency.
                    if (e.Latency == SassLatency.Variable)
                        foreach (int p in predicateReads) if (predicateWrittenAt.TryGetValue(p, out long setAt)) Need(setAt + guardLatency, setAt);
                    if (guard >= 0 && predicateWrittenAt.TryGetValue(guard, out long written))
                        Need(written + guardLatency, written);
                    foreach (int p in predicateWrites) if (readyAt.TryGetValue(p, out var at)) Need(at.Ready, at.Produced);
                    if (e.ReadsAllPredicates)
                        foreach (var pair in readyAt) if (pair.Key >= PredicateKey && pair.Key < UniformPredicateKey) Need(pair.Value.Ready, pair.Value.Produced);
                    if (pipe.Pipe is not null && pipeFreeAt.TryGetValue(pipe.Pipe, out var free)) Need(free.Ready, free.Produced);
                    // sm_86: a vector instruction reading any uniform register issues 5 cycles after an R2UR (to any
                    // register): an I2F reading UR13 2-4 cycles after R2UR UR5 read a wrong value on the RTX 3080, 5 ran
                    // (ptxas leaves 6). Uniform-pipe readers are not affected (ptxas issues a UIADD3 1 cycle on).
                    if (timing.R2urToVectorUniformRead > 0 && !e.UniformPredicates && reads.Exists(r => r >= UniformRegisterKey))
                        Need(r2urAt + timing.R2urToVectorUniformRead, r2urAt);
                    // R2UR and S2UR write their uniform register early too: an R2UR UR5 one cycle after a USEL reading UR5 made
                    // an FP32 GEMM's results wrong on the RTX 3080 (ptxas leaves 3).
                    if (e.Mnemonic is "UMOV" or "ULDC" or "R2UR" or "S2UR")
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
                        if (k < i - 1) spreadPastPrevious = true;
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
                    var kept = canCarry ? readyAt.Where(pair => carried.Contains(pair.Key)).ToList() : null;
                    readyAt.Clear();
                    carried.Clear();
                    if (kept is not null)
                        foreach (var pair in kept) { readyAt[pair.Key] = pair.Value; carried.Add(pair.Key); }
                    earlyAt.Clear();
                    predicateWrittenAt.Clear();
                    pipeFreeAt.Clear();
                    uniformReadAt.Clear();
                    mmaReadUntil.Clear();
                }
                if (pipe.Pipe is not null) pipeFreeAt[pipe.Pipe] = (cycle + pipe.Cycles, cycle);
                if (e.Mnemonic == "R2UR") r2urAt = cycle;
                if (timing.MmaSourceReadCycles > 0 && e.Mnemonic == "HMMA")
                    foreach (int r in reads) mmaReadUntil[r] = cycle + timing.MmaSourceReadCycles;
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
                        shortOps[writeBoard] |= !global;
                        pendingUse[writeBoard] = Math.Min(pendingUse[writeBoard], firstUse[i]);
                        pendingComplete[writeBoard] = Math.Max(pendingComplete[writeBoard], writeComplete);
                        scoreboardSetAt[writeBoard] = cycle;
                        foreach (int r in writes) { pendingWrite[r] = 1 << writeBoard; readyAt.Remove(r); earlyAt.Remove(r); carried.Remove(r); }
                    }
                    if (readBoard != SassInstruction.NoScoreboard)
                    {
                        outstanding[readBoard]++;
                        shortOps[readBoard] = true;
                        pendingUse[readBoard] = Math.Min(pendingUse[readBoard], firstOverwrite[i]);
                        pendingComplete[readBoard] = Math.Max(pendingComplete[readBoard], readReleased);
                        scoreboardSetAt[readBoard] = cycle;
                        // Or-ed: an earlier reader of the register, on another scoreboard, may still be pending.
                        foreach (int r in reads)
                            if (r < UniformRegisterKey) pendingRead[r] = (pendingRead.TryGetValue(r, out int m) ? m : 0) | 1 << readBoard;
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
                        if (e.Mnemonic == "HMMA") carried.Add(r);
                        else carried.Remove(r);
                    }
                    foreach (int p in predicateWrites)
                    {
                        readyAt[p] = (cycle + latency, cycle);
                        predicateWrittenAt[p] = cycle;
                    }
                }

                var original = instruction.Control;
                var updated = instruction;
                int baseStall = Math.Max(minimumStall[i],
                    timing.IssueStalls.TryGetValue(e.Mnemonic, out int fixedStall) ? fixedStall : 1);
                updated.Control = new SassControl(baseStall, original.Yield, writeBoard, readBoard, wait, original.Reuse);
                result[i] = updated;
                originalStall[i] = original.Stall;
                cycle += baseStall;

                // A branch hands what is pending to its target.
                if (branch)
                {
                    int to = BranchTarget(code, i);
                    var state = new ScoreState(pendingWrite, pendingRead, outstanding, scoreboardSetAt, shortOps);
                    if (canCarry)
                        foreach (int r in carried)
                            if (readyAt.TryGetValue(r, out var at) && at.Ready > cycle) state.Carried[r] = at.Ready - cycle;
                    var edges = to > i ? forwardEntry : backOut;
                    if (to > i || !drainTargets.Contains(to))
                        edges[to] = edges.TryGetValue(to, out var known) ? known.Union(state) : state;
                }
            }
            // Reuse flags only where the stall did not grow (see the file header).
            for (int i = 0; i < result.Length; i++)
            {
                var control = result[i].Control;
                if (control.Reuse != 0 && control.Stall > originalStall[i]) result[i].Control = control with { Reuse = 0 };
                // A stall of 12 or more is encoded only with the yield bit clear: ptxas never sets both (0 of 13,000+
                // control words), and nvdisasm rejects the combination as an undefined value. Keeping ptxas's yield on an
                // instruction whose stall grew to 12 produced such words, and one of them faulted a reordered kernel.
                if (result[i].Control is { Stall: >= 12, Yield: true } raised) result[i].Control = raised with { Yield = false };
            }
            if (spreadPastPrevious)
            {
                // Re-run with this pass's stalls as the floor, so every cycle is computed with them in place. Stalls
                // only grow and are capped, so this settles; a kernel that does not is declined rather than given a
                // schedule whose later waits were computed from cycles that no longer hold.
                if (spreadPasses++ >= MaxPasses)
                {
                    LastDecline = "the stalls spread back past the preceding instruction did not settle";
                    return null;
                }
                for (int k = 0; k < result.Length; k++) minimumStall[k] = result[k].Control.Stall;
                pass--;   // a re-run is not one of the scoreboard fixpoint's passes
                continue;
            }
            if (backOut.All(edge => backEntry.TryGetValue(edge.Key, out var assumed) && assumed.Covers(edge.Value))) return result;
            if (pass >= MaxPasses)
            {
                if (pass > MaxPasses)
                {
                    LastDecline = "scoreboard state at the loop heads did not settle";
                    return null;
                }
                drainTargets.UnionWith(backOut.Keys);
                backEntry.Clear();
                continue;
            }
            foreach (var edge in backOut)
                backEntry[edge.Key] = backEntry.TryGetValue(edge.Key, out var assumed) ? assumed.Union(edge.Value) : edge.Value;

            // Inside a BSSY region, the scoreboards still holding loads from before it are neither set nor reused
            // (see the file header); the region falls back to every board only if none other is left.
            int[] Unfrozen(int[] candidates)
            {
                if (frozen == 0) return candidates;
                var free = candidates.Where(b => (frozen >> b & 1) == 0).ToArray();
                return free.Length != 0 ? free : candidates;
            }

            // Adds a predecessor's pending scoreboards to the current state. A back edge's set cycles belong to the
            // previous trip and are taken as at least ScoreboardSetDelay ago; the choice model treats its boards as
            // holding a global load needed now.
            void Merge(ScoreState state, bool clampSetAt, int at)
            {
                // MMA results an edge brings, counted from this target's first issue cycle: only instructions from the
                // target on may hold the remaining delay.
                foreach (var pair in state.Carried)
                {
                    long ready = cycle + pair.Value;
                    if (!readyAt.TryGetValue(pair.Key, out var known) || known.Ready < ready) readyAt[pair.Key] = (ready, cycle);
                    earlyAt.Remove(pair.Key);
                    carried.Add(pair.Key);
                }
                foreach (var pair in state.Write) pendingWrite[pair.Key] = (pendingWrite.TryGetValue(pair.Key, out int m) ? m : 0) | pair.Value;
                foreach (var pair in state.Read) pendingRead[pair.Key] = (pendingRead.TryGetValue(pair.Key, out int m) ? m : 0) | pair.Value;
                for (int b = 0; b < 6; b++)
                {
                    if (state.Outstanding[b] == 0) continue;
                    outstanding[b] = Math.Max(outstanding[b], state.Outstanding[b]);
                    shortOps[b] |= state.Short[b];
                    long setAt = clampSetAt ? Math.Min(state.SetAt[b], cycle - ScoreboardSetDelay) : state.SetAt[b];
                    scoreboardSetAt[b] = Math.Max(scoreboardSetAt[b], setAt);
                    // A use is an INSTRUCTION INDEX (ChooseBoard compares it with firstUse / firstOverwrite), not a
                    // cycle: storing the cycle here made a merged board look needed far in the future, so ChooseBoard
                    // would pile short operations onto the board carrying a global load.
                    pendingUse[b] = Math.Min(pendingUse[b], at);
                    pendingComplete[b] = Math.Max(pendingComplete[b], cycle + GlobalLoadCycles);
                }
            }
        }
    }

    // Passes over a kernel with loops before its loop heads' scoreboard state settles (or they fall back to waiting).
    private const int MaxPasses = 24;

    // What is pending on the scoreboards at one program point: per register, the write and read scoreboards (masks)
    // that may still cover it, and per scoreboard how many operations it may hold and when it was last set. Along
    // different paths into a branch target these differ; the union is safe (an instruction then waits on every
    // scoreboard any path left covering its registers, and a scoreboard is a counter: waiting on one that is already
    // clear costs nothing).
    private sealed class ScoreState
    {
        public ScoreState(Dictionary<int, int> write, Dictionary<int, int> read, int[] outstanding, long[] setAt, bool[] shortOps)
        {
            Short = (bool[])shortOps.Clone();
            Carried = new Dictionary<int, long>();
            Write = new Dictionary<int, int>(write);
            Read = new Dictionary<int, int>(read);
            Outstanding = (int[])outstanding.Clone();
            SetAt = (long[])setAt.Clone();
        }

        public Dictionary<int, int> Write { get; }
        public Dictionary<int, int> Read { get; }
        public int[] Outstanding { get; }
        public long[] SetAt { get; }
        public bool[] Short { get; }

        // MMA results still in flight at the branch: register key -> cycles left after the target can first issue.
        public Dictionary<int, long> Carried { get; }

        public ScoreState Union(ScoreState other)
        {
            var union = new ScoreState(Write, Read, Outstanding, SetAt, Short);
            foreach (var pair in Carried) union.Carried[pair.Key] = pair.Value;
            foreach (var pair in other.Carried)
                union.Carried[pair.Key] = Math.Max(union.Carried.TryGetValue(pair.Key, out long c) ? c : 0, pair.Value);
            foreach (var pair in other.Write) union.Write[pair.Key] = (union.Write.TryGetValue(pair.Key, out int m) ? m : 0) | pair.Value;
            foreach (var pair in other.Read) union.Read[pair.Key] = (union.Read.TryGetValue(pair.Key, out int m) ? m : 0) | pair.Value;
            for (int b = 0; b < Outstanding.Length; b++)
            {
                union.Outstanding[b] = Math.Max(union.Outstanding[b], other.Outstanding[b]);
                union.SetAt[b] = Math.Max(union.SetAt[b], other.SetAt[b]);
                union.Short[b] |= other.Short[b];
            }
            return union;
        }

        // Whether this state already accounts for everything `other` leaves pending (set cycles aside: a loop head
        // clamps them).
        public bool Covers(ScoreState other)
        {
            foreach (var pair in other.Write) if (!Write.TryGetValue(pair.Key, out int m) || (pair.Value & ~m) != 0) return false;
            foreach (var pair in other.Read) if (!Read.TryGetValue(pair.Key, out int m) || (pair.Value & ~m) != 0) return false;
            for (int b = 0; b < Outstanding.Length; b++)
                if (other.Outstanding[b] > Outstanding[b] || (other.Short[b] && !Short[b])) return false;
            foreach (var pair in other.Carried) if (!Carried.TryGetValue(pair.Key, out long c) || c < pair.Value) return false;
            return true;
        }
    }

    // The instruction index a BRA or CALL.REL.NOINC at `at` jumps to.
    private static int BranchTarget(IReadOnlyList<SassInstruction> code, int at)
    {
        long offset = (long)code[at].Bits(32, 50);
        if ((offset & (1L << 49)) != 0) offset -= 1L << 50;
        return (int)(((at + 1) * (long)InstructionBytes + offset) / InstructionBytes);
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
    // go to the scoreboard holding fewer operations -- except for a global load (`conserve`), which joins a scoreboard
    // already holding loads its consumer waits for anyway rather than open an empty one: global loads may only hold
    // all but ShortBoards scoreboards, and the first few each taking its own left the later groups (consumed later)
    // sharing one with each other. (Uses are instruction indices, compared only with each other.)
    private static int ChooseBoard(int[] candidates, int[] outstanding, long[] pendingUse, long[] pendingComplete,
        long use, long complete, bool conserve = false)
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
            if (cost < bestCost || (cost == bestCost && (conserve
                    ? outstanding[b] > 0 && outstanding[best] == 0
                    : outstanding[b] < outstanding[best])))
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
    private static (long[] FirstUse, long[] FirstOverwrite, bool[] UseInBlock) NextHazards(IReadOnlyList<SassInstruction> code,
        SassEffects[] effects, HashSet<int> targets)
    {
        var firstUse = new long[code.Count];
        var firstOverwrite = new long[code.Count];
        var useInBlock = new bool[code.Count];
        for (int i = 0; i < code.Count; i++)
        {
            if (effects[i].Latency != SassLatency.Variable) continue;
            var writes = new HashSet<int>(Registers(code[i], effects[i].Writes));
            // Vector sources only: a uniform operand is read at issue (see VectorSources).
            var reads = new HashSet<int>(VectorSources(code[i], effects[i]));
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
            useInBlock[i] = use >= 0;
        }
        return (firstUse, firstOverwrite, useInBlock);
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
            foreach (int r in VectorSources(code[i], effects[i]))
                if (writers.TryGetValue(r, out var list) && list.Exists(w => w != i && (w > i || w >= earliest)))
                {
                    result[i] = true;
                    break;
                }
        }
        return result;
    }

    // The sources a variable-latency instruction's read scoreboard covers: its vector registers. A uniform operand (an
    // LDS/STS offset) is read at issue: ptxas overwrites one with a UMOV right after the LDS reading it, without a wait.
    private static IEnumerable<int> VectorSources(in SassInstruction instruction, SassEffects effects) =>
        Registers(instruction, effects.Reads).Where(r => r < UniformRegisterKey);

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

    // Waiting on `waitMask` clears those scoreboards from every register's mask.
    private static void Clear(Dictionary<int, int> pending, int waitMask)
    {
        foreach (int r in pending.Keys.ToList())
        {
            int left = pending[r] & ~waitMask;
            if (left == 0) pending.Remove(r);
            else pending[r] = left;
        }
    }
}
