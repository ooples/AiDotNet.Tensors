// Copyright (c) AiDotNet. All rights reserved.
// --sass-learn: empirical SASS field layouts for the own SASS backend (#1442). Every catalog spec is emitted as PTX,
// compiled by ptxas for the target arch and disassembled by nvdisasm -hex; each instruction's text is matched against
// its 128-bit encoding to find, per opcode and operand slot, which bit field holds a register (an 8-bit R field), a
// predicate (3 bits + negate) or the guard predicate, and the control word (stall, yield, barriers, wait mask, reuse).
// A layout is reported only when every sighting agrees; disagreements are listed. The result is what the backend's
// re-allocation and re-scheduling may rewrite.
//
// Options: --arch 75|86 (default 75), --cuda-bin <dir>, --out <dir> (default %TEMP%\aidotnet-sass-learn).

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

internal static class SassLearnTool
{
    /// <summary>One disassembled instruction: its text parts and its two 64-bit words.</summary>
    internal sealed record SassLine(string Kernel, int Offset, string Guard, string Opcode, string[] Operands, ulong Low,
        ulong High);

    private static readonly Regex InstructionLine = new(
        @"^\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P[0-7T]\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);\s*/\*\s*0x([0-9a-f]{16})\s*\*/\s*$",
        RegexOptions.Compiled);
    private static readonly Regex HighLine = new(@"^\s*/\*\s*0x([0-9a-f]{16})\s*\*/\s*$", RegexOptions.Compiled);
    private static readonly Regex Register = new(@"^-?\|?R(\d+|Z)\|?(\.reuse)?$", RegexOptions.Compiled);
    private static readonly Regex Predicate = new(@"^!?(P[0-7]|PT)$", RegexOptions.Compiled);

    internal static void Run(string[] args)
    {
        string arch = Value(args, "--arch") ?? "75";
        string bin = Value(args, "--cuda-bin") ?? DefaultCudaBin()
            ?? throw new InvalidOperationException("No CUDA toolkit bin directory: pass --cuda-bin.");
        string work = Value(args, "--out") ?? Path.Combine(Path.GetTempPath(), "aidotnet-sass-learn");
        Directory.CreateDirectory(work);
        int major = int.Parse(arch, CultureInfo.InvariantCulture) / 10, minor = int.Parse(arch, CultureInfo.InvariantCulture) % 10;

        var lines = new List<SassLine>();
        foreach (var entry in CodegenKernelCatalog.All)
        {
            foreach (var (tag, spec) in new[] { ("verify", entry.Verify), ("bench", entry.Bench) })
            {
                string ptx;
                try
                {
                    ptx = new PtxAffineEmitter().Emit(spec, major, minor);
                }
                catch (NotSupportedException)
                {
                    continue;
                }
                lines.AddRange(Disassemble(ptx, entry.Name + "_" + tag, arch, bin, work));
            }
        }
        // The register-blocked GEMM (the backend's compute-bound gate kernel): every seed configuration, both layouts
        // of A, as the probe's --gemm mode runs them.
        foreach (bool transposeA in new[] { false, true })
        {
            var layout = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxGemmLayout(transposeA, false);
            foreach (var config in AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxGemmEvolution.Seeds(layout, 2048, 2048, 2048, ccMajor: major))
            {
                string ptx = AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxSgemmGenerator.Emit(config, layout, major, minor);
                lines.AddRange(Disassemble(ptx, "sgemm_" + config.Name.Replace('-', '_') + "_" + layout.Name, arch, bin, work));
            }
        }
        // The Tensor-Core GEMM (--sass-probe --tc): every seed configuration, TF32 and FP16 (sm_80+: mma.sync, cp.async).
        if (major >= 8)
            foreach (var precision in new[] { AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCorePrecision.Tf32,
                         AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCorePrecision.Fp16 })
            {
                var layout = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxGemmLayout(false, false);
                foreach (var config in AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmEvolution.Seeds(layout, 4096, 4096, 4096, precision))
                {
                    string ptx = AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmGenerator.Emit(config, layout, major, minor, precision);
                    lines.AddRange(Disassemble(ptx, "tc_" + precision.ToString().ToLowerInvariant() + "_" + config.Name.Replace('-', '_'), arch, bin, work));
                }
            }
        // The bank probe's FFMA-bound kernel (--sass-probe --bank): on sm_80+ its int-to-float conversion is I2FP.
        lines.AddRange(Disassemble(SassProbeTool.BankProbePtx(major, minor), "bank_probe", arch, bin, work));
        Console.WriteLine("SASS LEARN sm_" + arch + ": " + lines.Count + " instructions from " +
            lines.Select(l => l.Kernel).Distinct().Count() + " kernels");
        ReportGuard(lines);
        ReportRegisterFields(lines);
        ReportUniformFields(lines);
        ReportControl(lines);
        ReportOpcodes(lines);
        ReportPredicateFields(lines);
        ReportVariantBits(lines);
        ReportScoreboards(lines);
        ReportTableConsistency(lines);
        ReportScheduling(work, major * 10 + minor);
    }

    // The scheduler over every kernel compiled above, offline (no device needed, so sm_86 is covered from an sm_75
    // machine): how many it schedules and why it declines the rest.
    private static void ReportScheduling(string work, int architecture)
    {
        int scheduled = 0;
        var declines = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (string path in Directory.GetFiles(work, "*.cubin"))
        {
            var cubin = new AiDotNet.Tensors.Engines.Compilation.Codegen.Sass.SassCubin(File.ReadAllBytes(path));
            foreach (string kernel in cubin.Kernels)
            {
                if (AiDotNet.Tensors.Engines.Compilation.Codegen.Sass.SassScheduler.RecomputeControl(cubin.Instructions(kernel), architecture) is not null)
                {
                    scheduled++;
                    continue;
                }
                string reason = Regex.Replace(AiDotNet.Tensors.Engines.Compilation.Codegen.Sass.SassScheduler.LastDecline ?? "?", @"\d+", "N");
                declines[reason] = declines.TryGetValue(reason, out int n) ? n + 1 : 1;
            }
        }
        Console.WriteLine();
        Console.WriteLine("SCHEDULING sm_" + architecture + " - " + scheduled + " kernels scheduled, " + declines.Values.Sum() + " declined");
        foreach (var pair in declines) Console.WriteLine("  declined x" + pair.Value + ": " + pair.Key);
    }

    // SassOpcodeTable against ptxas's own schedule. Walking each kernel's straight-line code: a variable-latency
    // instruction's write scoreboard covers the registers the table says it writes, its read scoreboard the registers
    // it reads; an instruction the table says touches a covered register must wait on that scoreboard -- ptxas always
    // does, so a miss means the table names a register the instruction does not touch. Conversely an instruction
    // that waits on a scoreboard none of its operands needs is reported (ptxas also waits for other reasons, so this
    // is a hint, not an error). Fixed latencies are measured: the fewest stall cycles ptxas left between a producer
    // and its first dependent reader, per producer opcode.
    private static void ReportTableConsistency(List<SassLine> lines)
    {
        int unknown = 0, missedWaits = 0, checkedReads = 0, checkedPredicates = 0, missedPredicates = 0, checkedOperands = 0, missedOperands = 0;
        var unknownOps = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var minimumLatency = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var misses = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (var kernel in lines.GroupBy(l => l.Kernel))
        {
            var pendingWrite = new Dictionary<int, int>();   // register -> scoreboard
            var pendingRead = new Dictionary<int, int>();
            var fixedReady = new Dictionary<int, (long Cycle, string Op)>();   // register -> cycle written, producer
            var predicateReady = new Dictionary<int, long>();
            var lastReader = new Dictionary<int, (long Cycle, string Op)>();   // register -> latest fixed reader
            long cycle = 0;
            foreach (var line in kernel)
            {
                var instruction = new AiDotNet.Tensors.Engines.Compilation.Codegen.Sass.SassInstruction(line.Low, line.High);
                var effects = AiDotNet.Tensors.Engines.Compilation.Codegen.Sass.SassOpcodeTable.Decode(instruction);
                var control = instruction.Control;
                foreach (var key in pendingWrite.Where(p => (control.WaitMask >> p.Value & 1) != 0).Select(p => p.Key).ToList())
                    pendingWrite.Remove(key);
                foreach (var key in pendingRead.Where(p => (control.WaitMask >> p.Value & 1) != 0).Select(p => p.Key).ToList())
                    pendingRead.Remove(key);
                if (effects is null)
                {
                    unknown++;
                    string unknownKey = line.Opcode + " 0x" + (line.Low & 0xfff).ToString("x3", CultureInfo.InvariantCulture) + " e.g. " + string.Join(", ", line.Operands);
                    if (!unknownOps.Keys.Any(k => k.StartsWith(line.Opcode + " 0x" + (line.Low & 0xfff).ToString("x3", CultureInfo.InvariantCulture), StringComparison.Ordinal)))
                        unknownOps[unknownKey] = 0;
                    string existing = unknownOps.Keys.First(k => k.StartsWith(line.Opcode + " 0x" + (line.Low & 0xfff).ToString("x3", CultureInfo.InvariantCulture), StringComparison.Ordinal));
                    unknownOps[existing]++;
                    pendingWrite.Clear(); pendingRead.Clear(); fixedReady.Clear();
                    continue;
                }
                IEnumerable<int> Registers(IEnumerable<AiDotNet.Tensors.Engines.Compilation.Codegen.Sass.SassRegisterOperand> ops) =>
                    ops.SelectMany(o =>
                    {
                        if (o.Uniform)
                        {
                            int u = instruction.UniformRegister(o.Field);
                            return u == 63 ? Enumerable.Empty<int>() : Enumerable.Range(1000 + u, o.Width);
                        }
                        int r = instruction.Register(o.Field);
                        return r == 255 ? Enumerable.Empty<int>() : Enumerable.Range(r, o.Width);
                    });
                var reads = Registers(effects.Reads).ToList();
                var writes = Registers(effects.Writes).ToList();
                foreach (int r in reads.Concat(writes))
                {
                    if (pendingWrite.ContainsKey(r))
                    {
                        missedWaits++;
                        string key = line.Opcode + " (register R" + r + ")";
                        misses[line.Opcode] = misses.TryGetValue(line.Opcode, out int m) ? m + 1 : 1;
                        pendingWrite.Remove(r);
                    }
                }
                foreach (int r in writes)
                    if (pendingRead.ContainsKey(r)) { missedWaits++; misses[line.Opcode + " (WAR)"] = misses.TryGetValue(line.Opcode + " (WAR)", out int m2) ? m2 + 1 : 1; pendingRead.Remove(r); }
                foreach (int r in writes)
                    if (lastReader.TryGetValue(r, out var reader))
                    {
                        string key = "WAR " + reader.Op;
                        int distance = (int)(cycle - reader.Cycle);
                        minimumLatency[key] = minimumLatency.TryGetValue(key, out int d1) ? Math.Min(d1, distance) : distance;
                        lastReader.Remove(r);
                    }
                if (effects.Latency == AiDotNet.Tensors.Engines.Compilation.Codegen.Sass.SassLatency.Fixed)
                    foreach (int r in reads) lastReader[r] = (cycle, line.Opcode);
                foreach (int r in reads)
                {
                    checkedReads++;
                    if (fixedReady.TryGetValue(r, out var producer))
                    {
                        int distance = (int)(cycle - producer.Cycle);
                        minimumLatency[producer.Op] = minimumLatency.TryGetValue(producer.Op, out int d) ? Math.Min(d, distance) : distance;
                        fixedReady.Remove(r);
                    }
                }
                int predicateBase = effects.UniformPredicates ? 8 : 0;
                var predicateReads = effects.PredicateReads.Select(f => (int)instruction.Bits(f, 3)).Where(p => p != 7).Select(p => predicateBase + p)
                    .Concat(effects.UniformPredicateReads.Select(f => (int)instruction.Bits(f, 3)).Where(p => p != 7).Select(p => 8 + p)).ToList();
                if (instruction.GuardPredicate != 7) predicateReads.Add(predicateBase + instruction.GuardPredicate);
                foreach (int pr in predicateReads)
                    if (predicateReady.TryGetValue(pr, out long written))
                    {
                        string key = "pred->" + (effects.Latency == AiDotNet.Tensors.Engines.Compilation.Codegen.Sass.SassLatency.Control ? "control" : "other");
                        int distance = (int)(cycle - written);
                        minimumLatency[key] = minimumLatency.TryGetValue(key, out int d0) ? Math.Min(d0, distance) : distance;
                        predicateReady.Remove(pr);
                    }
                foreach (int pw in effects.PredicateWrites.Select(f => (int)instruction.Bits(f, 3)).Where(p => p != 7).Select(p => predicateBase + p))
                    predicateReady[pw] = cycle;
                // Register renaming (SassRegisterRenamer) needs the table to name every vector register field: each
                // register nvdisasm prints must be a decoded field's base, and each decoded field's register must be
                // printed (a field the table invents would have non-register bits rewritten).
                var printed = new HashSet<int>(line.Operands.SelectMany(o =>
                    Regex.Matches(o, @"(?<![A-Z_])R(\d+)").Select(x => int.Parse(x.Groups[1].Value, CultureInfo.InvariantCulture))));
                var decoded = new HashSet<int>(effects.Reads.Concat(effects.Writes).Where(o => !o.Uniform)
                    .Select(o => instruction.Register(o.Field)).Where(r => r != 255));
                checkedOperands += printed.Count;
                foreach (int r in printed.Where(r => !decoded.Contains(r)))
                {
                    missedOperands++;
                    string key = line.Opcode + " 0x" + Field(line, 0, 12).ToString("x3") + " (R" + r + " printed, not decoded)";
                    misses[key] = misses.TryGetValue(key, out int mo) ? mo + 1 : 1;
                }
                foreach (int r in decoded.Where(r => !printed.Contains(r)))
                {
                    missedOperands++;
                    string key = line.Opcode + " 0x" + Field(line, 0, 12).ToString("x3") + " (R" + r + " decoded, not printed)";
                    misses[key] = misses.TryGetValue(key, out int mo) ? mo + 1 : 1;
                }
                // Every predicate the listing names as an operand must be one the table reads or writes: ptxas keeps
                // no scoreboard for predicates, so a missing one (IMAD.X's carry-in) is invisible to the wait check.
                // Keyed by file (UPn as 8 + n), so a uniform predicate the table decodes as a vector one (PLOP3's third
                // source) is a miss.
                var covered = new HashSet<int>(effects.PredicateReads.Concat(effects.PredicateWrites)
                    .Select(f => predicateBase + (int)instruction.Bits(f, 3))
                    .Concat(effects.UniformPredicateReads.Select(f => 8 + (int)instruction.Bits(f, 3))));
                foreach (string operand in line.Operands)
                {
                    var pm = Regex.Match(operand, @"^!?(U?)P([0-6])$");
                    if (!pm.Success) continue;
                    checkedPredicates++;
                    int printedKey = (pm.Groups[1].Value.Length != 0 ? 8 : 0) + int.Parse(pm.Groups[2].Value, CultureInfo.InvariantCulture);
                    if (covered.Contains(printedKey)) continue;
                    missedPredicates++;
                    string key = line.Opcode + " (predicate " + operand + ")";
                    misses[key] = misses.TryGetValue(key, out int mp) ? mp + 1 : 1;
                }
                if (effects.Latency == AiDotNet.Tensors.Engines.Compilation.Codegen.Sass.SassLatency.Control)
                {
                    pendingWrite.Clear(); pendingRead.Clear(); fixedReady.Clear(); predicateReady.Clear(); lastReader.Clear();
                }
                else if (effects.Latency == AiDotNet.Tensors.Engines.Compilation.Codegen.Sass.SassLatency.Variable)
                {
                    if (control.WriteScoreboard != 7) foreach (int r in writes) pendingWrite[r] = control.WriteScoreboard;
                    if (control.ReadScoreboard != 7) foreach (int r in reads) pendingRead[r] = control.ReadScoreboard;
                    foreach (int r in writes) fixedReady.Remove(r);
                }
                else
                {
                    foreach (int r in writes) fixedReady[r] = (cycle, line.Opcode);
                }
                cycle += Math.Max(1, control.Stall);
            }
        }
        Console.WriteLine();
        Console.WriteLine("TABLE CONSISTENCY - " + unknown + " instructions of unknown opcodes; " + checkedReads +
            " register reads checked; " + missedWaits + " dependencies ptxas did not wait on (table errors); " +
            checkedPredicates + " predicate operands checked, " + missedPredicates + " not in the table (table errors); " +
            checkedOperands + " register operands checked, " + missedOperands + " decoded and printed sets disagree (table errors)");
        foreach (var pair in unknownOps) Console.WriteLine("  unknown " + pair.Key + " x" + pair.Value);
        foreach (var pair in misses) Console.WriteLine("  missed wait at " + pair.Key + " x" + pair.Value);
        foreach (var g in lines.GroupBy(l => Base(l.Opcode)))
            Console.Write(g.Key + ":" + g.Max(l => (int)Field(l, 105, 4)) + " ");
        Console.WriteLine();
        Console.WriteLine("  fewest stall cycles between a fixed-latency producer and its first reader: " +
            string.Join(", ", minimumLatency.Select(p => p.Key + " " + p.Value)));
    }

    // For each opcode and operand slot holding a predicate: the 3-bit fields (bits 64-95) equal to its index, and
    // the bit after it holding its negation.
    private static void ReportPredicateFields(List<SassLine> lines)
    {
        Console.WriteLine();
        Console.WriteLine("PREDICATE FIELDS - mnemonic/operands: slot -> bit position (sightings)");
        foreach (var group in lines.GroupBy(l => (Base(l.Opcode), l.Operands.Length)).OrderBy(g => g.Key.Item1, StringComparer.Ordinal))
        {
            var parts = new List<string>();
            for (int slot = 0; slot < group.Key.Item2; slot++)
            {
                var sightings = group.Where(l => Predicate.IsMatch(l.Operands[slot])).ToList();
                if (sightings.Count == 0) continue;
                static ulong Index(string p) => p.TrimStart('!') == "PT" ? 7UL : ulong.Parse(p.TrimStart('!').Substring(1), CultureInfo.InvariantCulture);
                var fits = Enumerable.Range(64, 32).Where(pos => sightings.All(l => Field(l, pos, 3) == Index(l.Operands[slot]))).ToList();
                parts.Add(slot + "->" + (fits.Count == 0 ? "?" : string.Join("|", fits)) + " (" + sightings.Count + ")");
            }
            if (parts.Count != 0) Console.WriteLine(("  " + group.Key.Item1 + "/" + group.Key.Item2).PadRight(18) + string.Join("  ", parts));
        }
    }

    // Bits 72-104 that are constant within each full mnemonic of a base opcode but differ between them: where
    // modifiers such as a load's width live.
    private static void ReportVariantBits(List<SassLine> lines)
    {
        Console.WriteLine();
        Console.WriteLine("VARIANT BITS - base: modifier -> constant bits in 72-104 that separate the variants");
        foreach (var family in lines.GroupBy(l => Base(l.Opcode)).Where(f => f.Select(l => l.Opcode).Distinct().Count() > 1))
        {
            var variants = family.GroupBy(l => l.Opcode).ToList();
            var constant = new Dictionary<string, Dictionary<int, ulong>>();
            foreach (var variant in variants)
            {
                var bits = new Dictionary<int, ulong>();
                for (int b = 72; b < 105; b++)
                {
                    var values = variant.Select(l => Field(l, b, 1)).Distinct().ToList();
                    if (values.Count == 1) bits[b] = values[0];
                }
                constant[variant.Key] = bits;
            }
            var separating = Enumerable.Range(72, 33).Where(b => constant.Values.All(c => c.ContainsKey(b)) &&
                constant.Values.Select(c => c[b]).Distinct().Count() > 1).ToList();
            if (separating.Count == 0) continue;
            Console.WriteLine("  " + family.Key + ": bits " + string.Join(",", separating));
            foreach (var pair in constant)
                Console.WriteLine("      " + pair.Key.PadRight(24) + string.Join("", separating.Select(b => pair.Value[b].ToString(CultureInfo.InvariantCulture))));
        }
    }

    // Which mnemonics set a write or read scoreboard (variable latency) and how often.
    private static void ReportScoreboards(List<SassLine> lines)
    {
        Console.WriteLine();
        Console.WriteLine("SCOREBOARDS - mnemonic: sets write SB / sets read SB / total");
        foreach (var group in lines.GroupBy(l => Base(l.Opcode)).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            int write = group.Count(l => Field(l, 110, 3) != 7), read = group.Count(l => Field(l, 113, 3) != 7);
            if (write + read == 0) continue;
            Console.WriteLine(("  " + group.Key).PadRight(14) + write + " / " + read + " / " + group.Count());
        }
    }

    // The 12-bit opcode values each mnemonic (with modifiers) was encoded with, and an example of its operands.
    private static void ReportOpcodes(List<SassLine> lines)
    {
        Console.WriteLine();
        Console.WriteLine("OPCODES - mnemonic: 12-bit opcode(s) [count] e.g. operands");
        foreach (var group in lines.GroupBy(l => l.Opcode).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            var codes = group.Select(l => Field(l, 0, 12)).Distinct().OrderBy(c => c).Select(c => "0x" + c.ToString("x3"));
            var example = group.First();
            Console.WriteLine(("  " + group.Key).PadRight(26) + string.Join(",", codes).PadRight(22) +
                ("[" + group.Count() + "]").PadRight(8) + string.Join(", ", example.Operands));
        }
    }

    /// <summary>Compiles PTX for the arch and returns its instructions with their encodings.</summary>
    internal static List<SassLine> Disassemble(string ptx, string name, string arch, string bin, string work)
    {
        string ptxPath = Path.Combine(work, name + ".ptx"), cubin = Path.ChangeExtension(ptxPath, ".cubin");
        File.WriteAllText(ptxPath, ptx);
        Require(Run(Tool(bin, "ptxas"), $"-arch=sm_{arch} \"{ptxPath}\" -o \"{cubin}\""), "ptxas " + name);
        var dis = Run(Tool(bin, "nvdisasm"), $"-hex \"{cubin}\"");
        Require(dis, "nvdisasm " + name);
        return Parse(dis.StdOut, name);
    }

    internal static List<SassLine> Parse(string listing, string kernel)
    {
        var result = new List<SassLine>();
        string[] text = listing.Replace("\r\n", "\n").Split('\n');
        for (int i = 0; i + 1 < text.Length; i++)
        {
            var m = InstructionLine.Match(text[i]);
            if (!m.Success) continue;
            var high = HighLine.Match(text[i + 1]);
            if (!high.Success) continue;
            string operandText = m.Groups[4].Value.Trim();
            var operands = operandText.Length == 0 ? Array.Empty<string>()
                : operandText.Split(',').Select(o => o.Trim()).ToArray();
            result.Add(new SassLine(kernel, int.Parse(m.Groups[1].Value, NumberStyles.HexNumber, CultureInfo.InvariantCulture),
                m.Groups[2].Value.Trim(), m.Groups[3].Value, operands,
                ulong.Parse(m.Groups[5].Value, NumberStyles.HexNumber, CultureInfo.InvariantCulture),
                ulong.Parse(high.Groups[1].Value, NumberStyles.HexNumber, CultureInfo.InvariantCulture)));
            i++;
        }
        return result;
    }

    /// <summary>Bits [position, position + width) of the 128-bit instruction.</summary>
    internal static ulong Field(SassLine line, int position, int width)
    {
        ulong value = 0;
        for (int b = 0; b < width; b++)
        {
            int bit = position + b;
            ulong word = bit < 64 ? line.Low : line.High;
            value |= ((word >> (bit & 63)) & 1UL) << b;
        }
        return value;
    }

    private static string Base(string opcode) => opcode.Split('.')[0];

    // The guard: bits 12-14 the predicate (7 = PT, no guard), bit 15 negation.
    private static void ReportGuard(List<SassLine> lines)
    {
        int agree = 0, disagree = 0;
        foreach (var line in lines)
        {
            ulong index = Field(line, 12, 3), negate = Field(line, 15, 1);
            string expected = line.Guard.Length == 0 ? "PT" : line.Guard.TrimStart('@');
            string actual = (negate == 1 ? "!" : "") + (index == 7 ? "PT" : "P" + index);
            if (expected == actual) agree++;
            else disagree++;
        }
        Console.WriteLine("guard predicate bits 12-15: " + agree + " agree, " + disagree + " disagree");
    }

    // For each opcode and operand slot holding a plain register: the 8-bit fields (at byte boundaries) that equal the
    // register number in every sighting.
    private static void ReportRegisterFields(List<SassLine> lines)
    {
        int[] candidates = { 16, 24, 32, 40, 64, 72 };
        var groups = lines.GroupBy(l => (Base(l.Opcode) + "@" + Field(l, 0, 12).ToString("x3"), l.Operands.Length));
        Console.WriteLine();
        Console.WriteLine("REGISTER FIELDS - opcode/operand count: slot -> bit position (sightings)");
        foreach (var group in groups.OrderBy(g => g.Key.Item1, StringComparer.Ordinal))
        {
            var parts = new List<string>();
            for (int slot = 0; slot < group.Key.Item2; slot++)
            {
                var sightings = group.Where(l => Register.IsMatch(l.Operands[slot])).ToList();
                if (sightings.Count == 0) continue;
                var fits = candidates.Where(p => sightings.All(l => Field(l, p, 8) == RegisterNumber(l.Operands[slot]))).ToList();
                parts.Add(slot + "->" + (fits.Count == 0 ? "?" : string.Join("|", fits)) + " (" + sightings.Count + ")");
            }
            Console.WriteLine(("  " + group.Key.Item1 + "/" + group.Key.Item2).PadRight(18) + string.Join("  ", parts));
        }
    }

    private static readonly Regex UniformRegister = new(@"^-?\|?UR(\d+|Z)\|?$", RegexOptions.Compiled);
    private static readonly Regex UniformPredicate = new(@"^!?(UP[0-6]|UPT)$", RegexOptions.Compiled);

    // The uniform datapath (sm_75+) has its own register file (UR0-UR62, URZ = 63) and predicates (UP0-UP6, UPT = 7).
    // For each opcode and operand slot holding one: the bit positions that equal its number in every sighting.
    private static void ReportUniformFields(List<SassLine> lines)
    {
        Console.WriteLine();
        Console.WriteLine("UNIFORM FIELDS - opcode/operand count: slot -> bit position (sightings)");
        var groups = lines.GroupBy(l => (Base(l.Opcode) + "@" + Field(l, 0, 12).ToString("x3"), l.Operands.Length));
        foreach (var group in groups.OrderBy(g => g.Key.Item1, StringComparer.Ordinal))
        {
            var parts = new List<string>();
            for (int slot = 0; slot < group.Key.Item2; slot++)
            {
                var registers = group.Where(l => UniformRegister.IsMatch(l.Operands[slot])).ToList();
                if (registers.Count != 0)
                {
                    var fits = Enumerable.Range(12, 100).Where(p => registers.All(l =>
                    {
                        var g = UniformRegister.Match(l.Operands[slot]).Groups[1].Value;
                        return Field(l, p, 6) == (g == "Z" ? 63UL : ulong.Parse(g, CultureInfo.InvariantCulture));
                    })).ToList();
                    parts.Add(slot + "->UR@" + (fits.Count == 0 ? "?" : string.Join("|", fits)) + " (" + registers.Count + ")");
                }
                var predicates = group.Where(l => UniformPredicate.IsMatch(l.Operands[slot])).ToList();
                if (predicates.Count != 0)
                {
                    var fits = Enumerable.Range(12, 100).Where(p => predicates.All(l =>
                    {
                        var g = UniformPredicate.Match(l.Operands[slot]).Groups[1].Value;
                        return Field(l, p, 3) == (g == "UPT" ? 7UL : ulong.Parse(g.Substring(2), CultureInfo.InvariantCulture));
                    })).ToList();
                    parts.Add(slot + "->UP@" + (fits.Count == 0 ? "?" : string.Join("|", fits)) + " (" + predicates.Count + ")");
                }
            }
            if (parts.Count != 0) Console.WriteLine(("  " + group.Key.Item1 + "/" + group.Key.Item2).PadRight(24) + string.Join("  ", parts));
        }
    }

    private static ulong RegisterNumber(string operand)
    {
        var m = Register.Match(operand);
        return m.Groups[1].Value == "Z" ? 255UL : ulong.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
    }

    // The control word in bits 105-127: stall (4), yield (1), write barrier (3), read barrier (3), wait mask (6),
    // reuse (4). Checks the reuse flags against the operands' .reuse markers.
    private static void ReportControl(List<SassLine> lines)
    {
        int reuseAgree = 0, reuseDisagree = 0;
        var stalls = new int[16];
        foreach (var line in lines)
        {
            stalls[Field(line, 105, 4)]++;
            ulong reuse = Field(line, 122, 4);
            bool marked = line.Operands.Any(o => o.EndsWith(".reuse", StringComparison.Ordinal));
            if ((reuse != 0) == marked) reuseAgree++;
            else reuseDisagree++;
        }
        Console.WriteLine();
        Console.WriteLine("CONTROL - reuse flags (bits 122-125) vs .reuse markers: " + reuseAgree + " agree, " + reuseDisagree +
            " disagree; stall histogram (bits 105-108): " + string.Join(" ", stalls.Select((c, i) => i + ":" + c)));
    }

    private static string Tool(string bin, string name) =>
        Path.Combine(bin, RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? name + ".exe" : name);

    private static void Require((int ExitCode, string StdOut, string StdErr) run, string what)
    {
        if (run.ExitCode != 0) throw new InvalidOperationException(what + " failed: " + run.StdErr + run.StdOut);
    }

    private static (int ExitCode, string StdOut, string StdErr) Run(string exe, string arguments)
    {
        var info = new ProcessStartInfo(exe, arguments)
        {
            RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
        };
        using var process = Process.Start(info) ?? throw new InvalidOperationException("Could not start " + exe);
        var stdout = process.StandardOutput.ReadToEndAsync();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        return (process.ExitCode, stdout.Result, stderr);
    }

    internal static string? DefaultCudaBin()
    {
        string? path = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (path is not null && Directory.Exists(Path.Combine(path, "bin"))) return Path.Combine(path, "bin");
        string root = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";
        return Directory.Exists(root)
            ? Directory.GetDirectories(root).OrderByDescending(d => d, StringComparer.Ordinal).Select(d => Path.Combine(d, "bin"))
                .FirstOrDefault(Directory.Exists)
            : null;
    }

    private static string? Value(string[] args, string flag)
    {
        int i = Array.IndexOf(args, flag);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }
}
