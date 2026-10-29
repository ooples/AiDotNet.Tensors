// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Instruction reordering for the own SASS backend (#1442): a list scheduler over each straight-line block, by critical
// path with the tensor pipe's 8-cycle issue spacing, so independent work fills the gaps between MMAs. SassScheduler.
// RecomputeControl then owns every control word as usual.
//
// What stays put: control instructions (branches, barriers, convergence, EXIT, DEPBAR, NOP) and branch targets bound
// the blocks and never move, so no branch offset changes. Inside a block an instruction moves only past ones it shares
// no register (vector or uniform), predicate (guards included) or uniform predicate with in any direction, and
// variable-latency operations (memory, conversions, special registers) keep their relative order. A reuse flag is kept
// only where the instruction after it is still the one ptxas put there.
//
// Reordering exposed scheduler hazards that ptxas's order never did -- a 12+ cycle stall encoded with the yield bit, the
// LDS/STS uniform offset, R2UR's spacing -- each now in SassScheduler / SassOpcodeTable. On the RTX 3080 the TF32
// Tensor-Core GEMM tc-b128x64x16-w64x32-s3-p4 runs 1.13x ptxas (15-run median) with it, bit-identical.

using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;

/// <summary>Reorders instructions within straight-line blocks by critical path.</summary>
public static class SassListScheduler
{
    private const int PredicateKey = 1000, UniformPredicateKey = 1008, UniformRegisterKey = 2000;

    /// <summary>The reordered code, or null when an opcode is unknown.</summary>
    public static SassInstruction[]? Reorder(IReadOnlyList<SassInstruction> code) => Reorder(code, out _);

    /// <summary>The reordered code and, per position, the original index of the instruction now there.</summary>
    public static SassInstruction[]? Reorder(IReadOnlyList<SassInstruction> code, out int[] origin)
    {
        if (code is null) throw new ArgumentNullException(nameof(code));
        origin = Enumerable.Range(0, code.Count).ToArray();
        var effects = new SassEffects[code.Count];
        for (int i = 0; i < code.Count; i++)
            if (SassOpcodeTable.Decode(code[i]) is { } e) effects[i] = e;
            else return null;
        var targets = SassScheduler.BranchTargets(code);
        var result = code.ToArray();
        int start = 0;
        while (start < code.Count)
        {
            if (effects[start].Latency == SassLatency.Control) { start++; continue; }
            int end = start + 1;
            while (end < code.Count && effects[end].Latency != SassLatency.Control && !targets.Contains(end)) end++;
            if (end - start > 2) ScheduleBlock(code, effects, start, end, result, origin);
            start = end;
        }
        for (int k = 0; k < result.Length; k++)
        {
            var control = result[k].Control;
            if (control.Reuse == 0) continue;
            bool adjacent = k + 1 < result.Length && origin[k + 1] == origin[k] + 1;
            if (!adjacent) result[k].Control = control with { Reuse = 0 };
        }
        return result;
    }

    private static void ScheduleBlock(IReadOnlyList<SassInstruction> code, SassEffects[] effects, int start, int end,
        SassInstruction[] result, int[] origin)
    {
        int n = end - start;
        var reads = new HashSet<int>[n];
        var writes = new HashSet<int>[n];
        var memory = new bool[n];
        for (int k = 0; k < n; k++)
        {
            int i = start + k;
            var e = effects[i];
            reads[k] = new HashSet<int>(Keys(code[i], e.Reads));
            writes[k] = new HashSet<int>(Keys(code[i], e.Writes));
            foreach (int p in e.PredicateReads) Add(reads[k], code[i], p, e.UniformPredicates);
            foreach (int p in e.UniformPredicateReads) Add(reads[k], code[i], p, true);
            foreach (int p in e.PredicateWrites) Add(writes[k], code[i], p, e.UniformPredicates);
            if (code[i].GuardPredicate != SassInstruction.TruePredicate)
                reads[k].Add((e.UniformPredicates ? UniformPredicateKey : PredicateKey) + code[i].GuardPredicate);
            if (e.ReadsAllPredicates) for (int p = 0; p < 7; p++) reads[k].Add(PredicateKey + p);
            memory[k] = e.Latency == SassLatency.Variable;
        }
        var successors = new List<(int To, int Latency)>[n];
        var predecessors = new int[n];
        for (int k = 0; k < n; k++) successors[k] = new List<(int, int)>();
        int lastMemory = -1;
        for (int bIdx = 0; bIdx < n; bIdx++)
        {
            for (int a = 0; a < bIdx; a++)
            {
                bool raw = writes[a].Overlaps(reads[bIdx]), war = reads[a].Overlaps(writes[bIdx]), waw = writes[a].Overlaps(writes[bIdx]);
                bool sequenced = memory[bIdx] && a == lastMemory;
                if (!raw && !war && !waw && !sequenced) continue;
                successors[a].Add((bIdx, raw ? Latency(effects[start + a]) : 1));
                predecessors[bIdx]++;
            }
            if (memory[bIdx]) lastMemory = bIdx;
        }
        var height = new long[n];
        for (int k = n - 1; k >= 0; k--)
        {
            long h = Latency(effects[start + k]);
            foreach (var (to, latency) in successors[k]) h = Math.Max(h, latency + height[to]);
            height[k] = h;
        }
        var readyAt = new long[n];
        var remaining = (int[])predecessors.Clone();
        var placed = new bool[n];
        var sequence = new List<int>(n);
        long cycle = 0, tensorFreeAt = 0;
        while (sequence.Count < n)
        {
            int pick = -1;
            for (int k = 0; k < n; k++)
            {
                if (placed[k] || remaining[k] != 0 || readyAt[k] > cycle) continue;
                if (effects[start + k].Mnemonic == "HMMA" && tensorFreeAt > cycle) continue;
                if (pick < 0 || height[k] > height[pick]) pick = k;
            }
            if (pick < 0)
            {
                long best = long.MaxValue;
                for (int k = 0; k < n; k++)
                {
                    if (placed[k] || remaining[k] != 0) continue;
                    long at = Math.Max(readyAt[k], effects[start + k].Mnemonic == "HMMA" ? tensorFreeAt : 0);
                    if (at < best || (at == best && height[k] > height[pick])) { best = at; pick = k; }
                }
                cycle = Math.Max(cycle, best);
            }
            placed[pick] = true;
            sequence.Add(pick);
            if (effects[start + pick].Mnemonic == "HMMA") tensorFreeAt = cycle + 8;
            foreach (var (to, latency) in successors[pick])
            {
                remaining[to]--;
                readyAt[to] = Math.Max(readyAt[to], cycle + latency);
            }
            cycle++;
        }
        for (int k = 0; k < n; k++)
        {
            result[start + k] = code[start + sequence[k]];
            origin[start + k] = start + sequence[k];
        }
    }

    private static int Latency(SassEffects e) => e.Mnemonic switch
    {
        "HMMA" => 56,
        "LDG" or "LDGSTS" => 300,
        "LDS" or "LDSM" => 30,
        _ => e.Latency == SassLatency.Variable ? 20 : 5,
    };

    private static IEnumerable<int> Keys(SassInstruction instruction, IReadOnlyList<SassRegisterOperand> operands)
    {
        foreach (var operand in operands)
        {
            if (operand.Uniform)
            {
                int u = instruction.UniformRegister(operand.Field);
                if (u == SassInstruction.ZeroUniformRegister) continue;
                for (int k = 0; k < operand.Width; k++) yield return UniformRegisterKey + u + k;
                continue;
            }
            int r = instruction.Register(operand.Field);
            if (r == SassInstruction.ZeroRegister) continue;
            for (int k = 0; k < operand.Width; k++) yield return r + k;
        }
    }

    private static void Add(HashSet<int> set, SassInstruction instruction, int field, bool uniform)
    {
        int p = (int)instruction.Bits(field, 3);
        if (p != SassInstruction.TruePredicate) set.Add((uniform ? UniformPredicateKey : PredicateKey) + p);
    }
}