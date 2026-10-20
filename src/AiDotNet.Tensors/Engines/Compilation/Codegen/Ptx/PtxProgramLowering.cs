// Copyright (c) AiDotNet. All rights reserved.
// Lowers a whole declared program (CodegenProgram) to an ordered list of PTX launches -- what a backend runs when it
// routes a call to a declared program instead of a hand-written kernel.
//
// The choices are the ones the conveyor measured: the longest run of two or more passes that share a row shape is ONE
// row-fused kernel (EmitRowProgram); a reduction the split planner says would leave the device idle runs as the
// planner's two-kernel split through a temporary; every other pass is one affine kernel. Arguments are named --
// program tensors by name, and scratch the plan itself needs (a split's partials) by a "#temp" name -- so a caller
// binds its own buffers and allocates the rest.

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>One kernel launch of a lowered program.</summary>
public sealed record PtxProgramStep(
    string Ptx, string EntryPoint, uint Blocks, uint BlockX, uint BlockY, IReadOnlyList<string> Arguments);

/// <summary>A program lowered to launches, and the float buffers its tensors and scratch need.</summary>
public sealed class PtxProgramLowering
{
    private PtxProgramLowering(IReadOnlyList<PtxProgramStep> steps, IReadOnlyDictionary<string, long> buffers)
    {
        Steps = steps;
        Buffers = buffers;
    }

    /// <summary>The launches, in order.</summary>
    public IReadOnlyList<PtxProgramStep> Steps { get; }

    /// <summary>Elements of every buffer the steps name: program tensors and "#temp" scratch.</summary>
    public IReadOnlyDictionary<string, long> Buffers { get; }

    /// <summary>Lowers <paramref name="program"/> for the given architecture.</summary>
    public static PtxProgramLowering Create(CodegenProgram program, string name, int computeMajor, int computeMinor)
    {
        if (program is null) throw new ArgumentNullException(nameof(program));
        if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("A lowering needs a name.", nameof(name));

        // A launch binds each pass's inputs and its one output by name; an argmax, extra outputs or an atomic
        // (scatter) output would need parameters -- and zeroing -- this lowering does not provide.
        foreach (var pass in program.Passes)
            if (pass.SecondaryOutput is not null || pass.ExtraOutputs.Count != 0 || pass.Output.NeedsAtomicStore)
                throw new NotSupportedException(pass.Name + " has outputs a program launch does not bind.");

        var steps = new List<PtxProgramStep>();
        var buffers = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var source in program.Sources()) buffers[source.Name] = source.ElementCount;
        foreach (var pass in program.Passes) buffers[pass.Output.Name] = pass.Output.ElementCount;

        var passes = program.Passes;
        int at = 0, temp = 0;
        while (at < passes.Count)
        {
            CodegenRowFusionPlan? fused = null;
            int end = at;
            for (int stop = passes.Count; stop >= at + 2 && fused is null; stop--)
            {
                var run = new CodegenKernelSpec[stop - at];
                var runTemps = new long[run.Length];
                for (int k = 0; k < run.Length; k++)
                {
                    run[k] = passes[at + k];
                    runTemps[k] = run[k].Output.ElementCount;
                }
                fused = CodegenRowFusion.TryPlan(new CodegenProgram(run, runTemps, program.Description), out _);
                if (fused is not null) end = stop;
            }

            if (fused is not null)
            {
                var emitter = new PtxAffineEmitter();
                string entry = Entry(name, at);
                string ptx = emitter.EmitRowProgram(fused, entry, computeMajor, computeMinor);
                steps.Add(new PtxProgramStep(ptx, entry, emitter.LaunchBlocks,
                    (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY, fused.Parameters));
                at = end;
                continue;
            }

            var pass = passes[at++];
            if (CodegenSplitReduction.TryPlan(pass) is { } split)
            {
                string scratch = "#temp" + temp++.ToString(CultureInfo.InvariantCulture);
                buffers[scratch] = split.TempElements;

                var partialArgs = new List<string>();
                foreach (int source in CodegenSplitReduction.PartialOperandSources(pass))
                    partialArgs.Add(pass.Inputs[source].Name);
                partialArgs.Add(scratch);
                steps.Add(Single(split.Partial, partialArgs, computeMajor, computeMinor));

                var combineArgs = new List<string> { scratch };
                foreach (int source in CodegenSplitReduction.CombineOperandSources(pass))
                    combineArgs.Add(pass.Inputs[source].Name);
                combineArgs.Add(pass.Output.Name);
                steps.Add(Single(split.Combine, combineArgs, computeMajor, computeMinor));
                continue;
            }

            var args = new List<string>();
            foreach (var input in pass.Inputs) args.Add(input.Name);
            args.Add(pass.Output.Name);
            steps.Add(Single(pass, args, computeMajor, computeMinor));
        }
        return new PtxProgramLowering(steps, buffers);
    }

    private static PtxProgramStep Single(CodegenKernelSpec spec, IReadOnlyList<string> args, int major, int minor)
    {
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(spec, major, minor);
        return new PtxProgramStep(ptx, spec.Name, emitter.LaunchBlocks,
            (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY, args);
    }

    // Row-fused entry points are named after the program and the run's first pass index, so two runs of one program
    // cannot collide.
    private static string Entry(string name, int firstPass)
    {
        var sb = new System.Text.StringBuilder("rowfused_");
        foreach (char ch in name) sb.Append(char.IsLetterOrDigit(ch) ? ch : '_');
        return sb.Append('_').Append(firstPass.ToString(CultureInfo.InvariantCulture)).ToString();
    }
}
