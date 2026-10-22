// Copyright (c) AiDotNet. All rights reserved.
// A chain of declared kernels as ONE persistent cooperative launch (#890 phase 4, on the resident program of phase 3,
// #1430): the stages' own PTX, composed into one entry that walks every stage's blocks with a grid barrier between
// stages, so a launch-bound chain pays one launch instead of one per kernel.
//
// Each stage is its kernel's body unchanged except where one kernel becomes part of another:
//
//   %ctaid.x        becomes a VIRTUAL block: the composed grid loops over the stage's own block count, so every
//                   virtual block does exactly the work that block did in the stage's own launch -- the arithmetic,
//                   and so the result, is bit-identical to launching the stages separately.
//   ret             ends the virtual block, not the kernel.
//   labels          take a per-stage prefix; parameters are renamed per stage and concatenated.
//   shared          each stage's first array is overlaid on one buffer (resident_shared) sized for the largest; any
//                   further array of a stage keeps its own, prefixed. Stages are separated by the grid barrier, which
//                   opens and closes with bar.sync, so no thread touches one stage's shared data once the next has
//                   begun.
//   ld.global.nc    becomes ld.global: the read-only path is undefined for data written during the kernel, and a
//                   later stage reads what an earlier one wrote. The barrier's membar.gl orders it.
//
// The barrier is the phase-3 one (CudaResidentProgramKernels): thread 0 of each block reads the generation,
// fences, counts its arrival; the last arrival resets the count and publishes a new generation, the rest spin on it.
// It is safe only because a cooperative launch keeps every block co-resident.
//
// Declined (NotSupportedException): a stage with a 2-D block, dynamic shared memory, a block size other than the
// others', or shared memory past 48 KB (the overlaid buffer plus the stages' further arrays).

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>A composed resident kernel: one entry over every stage, and its argument names in order.</summary>
public sealed class PtxResidentComposition
{
    internal PtxResidentComposition(string ptx, string entryPoint, uint blockX, IReadOnlyList<string> arguments,
        uint maxStageBlocks, int sharedBytes, IReadOnlyList<CodegenLibraryFunction> calls)
    {
        Calls = calls;
        Ptx = ptx;
        EntryPoint = entryPoint;
        BlockX = blockX;
        Arguments = arguments;
        MaxStageBlocks = maxStageBlocks;
        SharedBytes = sharedBytes;
    }

    public string Ptx { get; }
    public string EntryPoint { get; }

    /// <summary>Threads per block: every stage's.</summary>
    public uint BlockX { get; }

    /// <summary>Every stage's arguments, stage by stage, then <see cref="PtxResidentComposer.BarrierArgument"/>.</summary>
    public IReadOnlyList<string> Arguments { get; }

    /// <summary>The most blocks any stage launched on its own: a grid larger than this would idle.</summary>
    public uint MaxStageBlocks { get; }

    /// <summary>Static shared memory: the overlaid buffer (the largest stage's first array) plus further arrays.</summary>
    public int SharedBytes { get; }

    /// <summary>Math-library functions any stage calls: the composed module is linked with their libraries.</summary>
    public IReadOnlyList<CodegenLibraryFunction> Calls { get; }
}

/// <summary>Composes lowered kernels into one resident (persistent cooperative) kernel.</summary>
public static class PtxResidentComposer
{
    /// <summary>The argument naming the two-word grid barrier (zeroed once; reusable launch after launch).</summary>
    public const string BarrierArgument = "#resident_barrier";

    private const int MaxStaticShared = 48 * 1024;

    // The buffer every stage's first shared array is overlaid on.
    private const string OverlaidShared = "resident_shared";

    /// <summary>Composes <paramref name="steps"/>, in order, as one entry named <paramref name="entryPoint"/>.</summary>
    public static PtxResidentComposition Compose(IReadOnlyList<PtxProgramStep> steps, string entryPoint,
        int computeMajor, int computeMinor)
    {
        if (steps is null) throw new ArgumentNullException(nameof(steps));
        if (steps.Count == 0) throw new ArgumentException("A resident kernel needs at least one stage.", nameof(steps));
        if (string.IsNullOrWhiteSpace(entryPoint)) throw new ArgumentException("A resident kernel needs a name.", nameof(entryPoint));

        uint blockX = steps[0].BlockX;
        var parameters = new List<string>();
        var arguments = new List<string>();
        var shared = new StringBuilder();
        var body = new StringBuilder();
        var externs = new List<string>();
        var calls = new List<CodegenLibraryFunction>();
        int maxP = 0, maxR = 0, maxRd = 0, maxF = 0, sharedBytes = 0, overlaidBytes = 0, overlaidAlign = 0;
        uint maxBlocks = 0;

        for (int k = 0; k < steps.Count; k++)
        {
            var step = steps[k];
            if (step.BlockY != 1) throw new NotSupportedException(step.EntryPoint + " has a 2-D block.");
            if (step.DynamicSharedMemoryBytes != 0) throw new NotSupportedException(step.EntryPoint + " uses dynamic shared memory.");
            if (step.BlockX != blockX)
                throw new NotSupportedException(step.EntryPoint + " runs " + step.BlockX + "-thread blocks; the chain runs " + blockX + ".");
            maxBlocks = Math.Max(maxBlocks, step.Blocks);
            string prefix = "s" + k.ToString(CultureInfo.InvariantCulture) + "_";
            var stage = Parse(step, prefix);
            foreach (string declaration in stage.Externs)
                if (!externs.Contains(declaration)) externs.Add(declaration);
            foreach (var call in step.Calls)
                if (!calls.Exists(c => c.Symbol == call.Symbol)) calls.Add(call);
            maxP = Math.Max(maxP, stage.P); maxR = Math.Max(maxR, stage.R); maxRd = Math.Max(maxRd, stage.Rd); maxF = Math.Max(maxF, stage.F);
            foreach (string parameter in stage.Parameters) parameters.Add(prefix + parameter);
            arguments.AddRange(step.Arguments);
            for (int s = 0; s < stage.Shared.Count; s++)
            {
                var (_, align, bytes) = stage.Shared[s];
                if (s == 0)
                {
                    overlaidBytes = Math.Max(overlaidBytes, bytes);
                    overlaidAlign = Math.Max(overlaidAlign, align);
                    continue;
                }
                shared.Append("    .shared .align ").Append(align.ToString(CultureInfo.InvariantCulture)).Append(" .b8 ")
                    .Append(prefix).Append(stage.Shared[s].Name).Append('[')
                    .Append(bytes.ToString(CultureInfo.InvariantCulture)).Append("];\n");
                sharedBytes += bytes;
            }

            string block = prefix + "BLOCK", next = prefix + "NEXTBLOCK", done = prefix + "DONE";
            body.Append("    mov.u32 %resvb, %ctaid.x;\n")
                .Append(block).Append(":\n")
                .Append("    setp.ge.u32 %resdone, %resvb, ").Append(step.Blocks.ToString(CultureInfo.InvariantCulture)).Append(";\n")
                .Append("    @%resdone bra ").Append(done).Append(";\n")
                .Append(stage.Body)
                .Append(next).Append(":\n")
                .Append("    add.u32 %resvb, %resvb, %resnct;\n")
                .Append("    bra ").Append(block).Append(";\n")
                .Append(done).Append(":\n");
            if (k < steps.Count - 1) AppendGridBarrier(body, prefix);
        }
        if (overlaidBytes > 0)
            shared.Insert(0, "    .shared .align " + overlaidAlign.ToString(CultureInfo.InvariantCulture) + " .b8 " + OverlaidShared +
                "[" + overlaidBytes.ToString(CultureInfo.InvariantCulture) + "];\n");
        sharedBytes += overlaidBytes;
        if (sharedBytes > MaxStaticShared)
            throw new NotSupportedException("The stages' shared memory totals " + sharedBytes + " bytes, past 48 KB.");
        parameters.Add("resident_barrier");
        arguments.Add(BarrierArgument);

        var ptx = new StringBuilder();
        ptx.Append(".version ").Append(PtxAffineEmitter.PtxIsaVersionFor(computeMajor, computeMinor)).Append('\n')
           .Append(".target sm_").Append(computeMajor.ToString(CultureInfo.InvariantCulture))
           .Append(computeMinor.ToString(CultureInfo.InvariantCulture)).Append('\n')
           .Append(".address_size 64\n\n")
           .Append("// composed by PtxResidentComposer: ").Append(steps.Count.ToString(CultureInfo.InvariantCulture))
           .Append(" stage(s), one cooperative launch\n");
        foreach (string declaration in externs) ptx.Append(declaration).Append('\n');
        ptx.Append(".visible .entry ").Append(entryPoint).Append("(\n");
        for (int i = 0; i < parameters.Count; i++)
            ptx.Append("    .param .u64 ").Append(parameters[i]).Append(i == parameters.Count - 1 ? "\n" : ",\n");
        ptx.Append(")\n.maxntid ").Append(blockX.ToString(CultureInfo.InvariantCulture)).Append(", 1, 1\n{\n")
           .Append(shared)
           .Append("    .reg .pred %p<").Append(maxP.ToString(CultureInfo.InvariantCulture)).Append(">;\n")
           .Append("    .reg .b32 %r<").Append(maxR.ToString(CultureInfo.InvariantCulture)).Append(">;\n")
           .Append("    .reg .b64 %rd<").Append(maxRd.ToString(CultureInfo.InvariantCulture)).Append(">;\n")
           .Append("    .reg .f32 %f<").Append(maxF.ToString(CultureInfo.InvariantCulture)).Append(">;\n")
           .Append("    .reg .b32 %resvb, %resnct, %resgen, %resarr, %reslast, %rescur, %resdummy;\n")
           .Append("    .reg .b64 %resbar;\n")
           .Append("    .reg .pred %resdone, %resleader, %reslastone, %resspin;\n")
           .Append("    ld.param.u64 %resbar, [resident_barrier];\n")
           .Append("    mov.u32 %resnct, %nctaid.x;\n")
           .Append(body)
           .Append("    ret;\n}\n");
        return new PtxResidentComposition(ptx.ToString(), entryPoint, blockX, arguments, maxBlocks, sharedBytes, calls);
    }

    // The phase-3 grid barrier, in PTX. Every thread reaches both block barriers.
    private static void AppendGridBarrier(StringBuilder body, string prefix)
    {
        string spin = prefix + "SPIN", released = prefix + "RELEASED", wait = prefix + "BARWAIT";
        body.Append("    bar.sync 0;\n")
            .Append("    mov.u32 %resarr, %tid.x;\n")
            .Append("    setp.ne.u32 %resleader, %resarr, 0;\n")
            .Append("    @%resleader bra ").Append(wait).Append(";\n")
            .Append("    ld.volatile.global.u32 %resgen, [%resbar+4];\n")
            .Append("    membar.gl;\n")
            .Append("    atom.global.add.u32 %resarr, [%resbar], 1;\n")
            .Append("    sub.u32 %reslast, %resnct, 1;\n")
            .Append("    setp.eq.u32 %reslastone, %resarr, %reslast;\n")
            .Append("    @!%reslastone bra ").Append(spin).Append(";\n")
            .Append("    st.global.u32 [%resbar], 0;\n")
            .Append("    membar.gl;\n")
            .Append("    atom.global.add.u32 %resdummy, [%resbar+4], 1;\n")
            .Append("    bra ").Append(released).Append(";\n")
            .Append(spin).Append(":\n")
            .Append("    ld.volatile.global.u32 %rescur, [%resbar+4];\n")
            .Append("    setp.eq.u32 %resspin, %rescur, %resgen;\n")
            .Append("    @%resspin bra ").Append(spin).Append(";\n")
            .Append(released).Append(":\n")
            .Append("    membar.gl;\n")
            .Append(wait).Append(":\n")
            .Append("    bar.sync 0;\n");
    }

    private sealed class Stage
    {
        internal List<string> Parameters { get; } = new();
        internal List<string> Externs { get; } = new();
        internal List<(string Name, int Align, int Bytes)> Shared { get; } = new();
        internal int P, R, Rd, F;
        internal string Body = "";
    }

    private static readonly Regex Label = new(@"^([A-Za-z_]\w*):\s*$", RegexOptions.Compiled);
    private static readonly Regex SharedDecl = new(@"^\s*\.shared\s+\.align\s+(\d+)\s+\.b8\s+([A-Za-z_]\w*)\[(\d+)\];\s*$", RegexOptions.Compiled);
    private static readonly Regex RegDecl = new(@"^\s*\.reg\s+\.(pred|b32|b64|f32)\s+%(p|r|rd|f)<(\d+)>;\s*$", RegexOptions.Compiled);

    private static Stage Parse(PtxProgramStep step, string prefix)
    {
        var stage = new Stage();
        string[] lines = step.Ptx.Replace("\r\n", "\n").Split('\n');
        int at = 0;
        // Module scope ahead of the entry: the library functions the stage calls are declared there.
        while (at < lines.Length && !lines[at].Contains(".entry", StringComparison.Ordinal))
        {
            if (lines[at].TrimStart().StartsWith(".extern .func", StringComparison.Ordinal)) stage.Externs.Add(lines[at].Trim());
            at++;
        }
        if (at == lines.Length) throw new NotSupportedException(step.EntryPoint + " has no entry.");
        for (at++; at < lines.Length && !lines[at].TrimStart().StartsWith(")", StringComparison.Ordinal); at++)
        {
            var m = Regex.Match(lines[at], @"\.param\s+\.u64\s+([A-Za-z_]\w*)");
            if (!m.Success) throw new NotSupportedException(step.EntryPoint + " has a parameter that is not a pointer: " + lines[at].Trim());
            stage.Parameters.Add(m.Groups[1].Value);
        }
        if (stage.Parameters.Count != step.Arguments.Count)
            throw new NotSupportedException(step.EntryPoint + " declares " + stage.Parameters.Count + " parameters for " + step.Arguments.Count + " arguments.");
        while (at < lines.Length && lines[at].Trim() != "{") at++;
        at++;

        // Declarations, then the body up to the entry's closing brace.
        var labels = new HashSet<string>(StringComparer.Ordinal);
        var bodyLines = new List<string>();
        int depth = 1;
        for (; at < lines.Length; at++)
        {
            string line = lines[at];
            string trimmed = line.Trim();
            // A library call opens its own { } scope for its .param slots: only the entry's own brace ends the body.
            if (trimmed == "{") depth++;
            if (trimmed == "}" && --depth == 0) break;
            var reg = RegDecl.Match(line);
            if (reg.Success)
            {
                int count = int.Parse(reg.Groups[3].Value, CultureInfo.InvariantCulture);
                switch (reg.Groups[2].Value)
                {
                    case "p": stage.P = Math.Max(stage.P, count); break;
                    case "r": stage.R = Math.Max(stage.R, count); break;
                    case "rd": stage.Rd = Math.Max(stage.Rd, count); break;
                    default: stage.F = Math.Max(stage.F, count); break;
                }
                continue;
            }
            var shared = SharedDecl.Match(line);
            if (shared.Success)
            {
                stage.Shared.Add((shared.Groups[2].Value, int.Parse(shared.Groups[1].Value, CultureInfo.InvariantCulture),
                    int.Parse(shared.Groups[3].Value, CultureInfo.InvariantCulture)));
                continue;
            }
            if (trimmed.StartsWith(".reg", StringComparison.Ordinal) || trimmed.StartsWith(".shared", StringComparison.Ordinal) ||
                trimmed.StartsWith(".extern", StringComparison.Ordinal) || trimmed.StartsWith(".local", StringComparison.Ordinal))
                throw new NotSupportedException(step.EntryPoint + " declares what a composed stage cannot carry: " + trimmed);
            foreach (string forbidden in new[] { "%tid.y", "%tid.z", "%ctaid.y", "%ctaid.z", "%nctaid", "%ntid" })
                if (line.Contains(forbidden, StringComparison.Ordinal))
                    throw new NotSupportedException(step.EntryPoint + " reads " + forbidden + ".");
            var label = Label.Match(line);
            if (label.Success) labels.Add(label.Groups[1].Value);
            bodyLines.Add(line);
        }

        var sb = new StringBuilder();
        foreach (string original in bodyLines)
        {
            string line = original;
            var label = Label.Match(line);
            if (label.Success)
            {
                sb.Append(prefix).Append(label.Groups[1].Value).Append(":\n");
                continue;
            }
            if (line.Trim() == "ret;")
            {
                sb.Append("    bra ").Append(prefix).Append("NEXTBLOCK;\n");
                continue;
            }
            line = line.Replace("%ctaid.x", "%resvb").Replace("ld.global.nc.", "ld.global.");
            line = Regex.Replace(line, @"\bbra(\.uni)?\s+([A-Za-z_]\w*);",
                m => labels.Contains(m.Groups[2].Value) ? "bra" + m.Groups[1].Value + " " + prefix + m.Groups[2].Value + ";" : m.Value);
            foreach (string parameter in stage.Parameters)
                line = line.Replace("[" + parameter + "]", "[" + prefix + parameter + "]");
            for (int s = 0; s < stage.Shared.Count; s++)
                line = Regex.Replace(line, @"\b" + Regex.Escape(stage.Shared[s].Name) + @"\b",
                    s == 0 ? OverlaidShared : prefix + stage.Shared[s].Name);
            sb.Append(line).Append('\n');
        }
        stage.Body = sb.ToString();
        return stage;
    }

}
