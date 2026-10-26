// Copyright (c) AiDotNet. All rights reserved.
// Lowers a whole declared program (CodegenProgram) to an ordered list of PTX launches -- what a backend runs when it
// routes a call to a declared program instead of a hand-written kernel.
//
// The choices are the ones the conveyor measured: the longest run of two or more passes that share a row shape is ONE
// row-fused kernel (EmitRowProgram); a reduction the split planner says would leave the device idle runs as the
// planner's two-kernel split through a temporary; every other pass is one affine kernel. Arguments are named --
// program tensors by name, and scratch the plan itself needs (a split's partials) by a "#temp" name -- so a caller
// binds its own buffers and allocates the rest.
//
// The conveyor runs THIS lowering too, so what it verifies and times is what a backend launches; it only supplies
// its tuned winners for the passes that are not fused (lowerPass).

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>One kernel launch of a lowered program.</summary>
public sealed record PtxProgramStep(
    string Ptx, string EntryPoint, uint Blocks, uint BlockX, uint BlockY, IReadOnlyList<string> Arguments,
    uint DynamicSharedMemoryBytes = 0)
{
    /// <summary>Math-library functions the kernel calls: its module must be linked with their libraries.</summary>
    public IReadOnlyList<CodegenLibraryFunction> Calls { get; init; } = Array.Empty<CodegenLibraryFunction>();
}

/// <summary>One kernel of a caller-supplied pass lowering.</summary>
public sealed record PtxPassKernel(
    string Ptx, string EntryPoint, uint Blocks, uint BlockX, uint BlockY, uint DynamicSharedMemoryBytes = 0);

/// <summary>
/// A pass lowered by the caller: one kernel over the pass's own parameters, or -- with <paramref name="Split"/> -- the
/// split's partial and combine kernels, wired through a temporary as the planner's split is.
/// </summary>
public sealed record PtxPassLowering(IReadOnlyList<PtxPassKernel> Kernels, CodegenSplitPlan? Split, string Label);

/// <summary>A program lowered to launches, and the float buffers its tensors and scratch need.</summary>
public sealed class PtxProgramLowering
{
    private PtxProgramLowering(IReadOnlyList<PtxProgramStep> steps, IReadOnlyDictionary<string, long> buffers,
        IReadOnlyList<string> passLowerings, IReadOnlyDictionary<string, int> elementBytes)
    {
        Steps = steps;
        Buffers = buffers;
        PassLowerings = passLowerings;
        ElementBytes = elementBytes;
    }

    /// <summary>
    /// Bytes per element of every program tensor that is not 4 bytes wide (an fp64 tensor, a byte mask); a name absent
    /// here, "#temp" scratch included, is 4 bytes. A caller allocating scratch sizes it by this, not by element count.
    /// </summary>
    public IReadOnlyDictionary<string, int> ElementBytes { get; }

    /// <summary>The bytes buffer <paramref name="name"/> occupies.</summary>
    public long BufferBytes(string name) =>
        Buffers[name] * (ElementBytes.TryGetValue(name, out int bytes) ? bytes : sizeof(float));

    /// <summary>How each pass was lowered, parallel to the program's passes ("row-fused x3", "split x2", ...).</summary>
    public IReadOnlyList<string> PassLowerings { get; }

    /// <summary>The launches, in order.</summary>
    public IReadOnlyList<PtxProgramStep> Steps { get; }

    /// <summary>Elements of every buffer the steps name: program tensors and "#temp" scratch.</summary>
    public IReadOnlyDictionary<string, long> Buffers { get; }

    /// <summary>
    /// Lowers <paramref name="program"/> for the given architecture. <paramref name="lowerPass"/>, when it returns a
    /// lowering, replaces the default for a pass that is not row-fused; <paramref name="rowFusion"/> false lowers every
    /// pass on its own.
    /// </summary>
    public static PtxProgramLowering Create(CodegenProgram program, string name, int computeMajor, int computeMinor,
        Func<CodegenKernelSpec, PtxPassLowering?>? lowerPass = null, bool rowFusion = true)
    {
        if (program is null) throw new ArgumentNullException(nameof(program));
        if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("A lowering needs a name.", nameof(name));

        // A launch binds each pass's inputs, its output and its extra outputs (an argmax) by name, in parameter order;
        // an atomic (scatter) output would need zeroing this lowering does not provide.
        foreach (var pass in program.Passes)
            if (pass.Output.NeedsAtomicStore)
                throw new NotSupportedException(pass.Name + " has outputs a program launch does not bind.");

        var steps = new List<PtxProgramStep>();
        var labels = new List<string>();
        var buffers = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var source in program.Sources()) buffers[source.Name] = source.ElementCount;
        foreach (var pass in program.Passes)
        {
            buffers[pass.Output.Name] = pass.Output.ElementCount;
            foreach (var extra in pass.ExtraOutputs) buffers[extra.Binding.Name] = extra.Binding.ElementCount;
        }
        var elementBytes = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var pass in program.Passes)
        {
            foreach (var input in pass.Inputs)
                if (input.ElementBytes != sizeof(float)) elementBytes[input.Name] = input.ElementBytes;
            if (pass.Output.ElementBytes != sizeof(float)) elementBytes[pass.Output.Name] = pass.Output.ElementBytes;
            foreach (var extra in pass.ExtraOutputs)
                if (extra.Binding.ElementBytes != sizeof(float)) elementBytes[extra.Binding.Name] = extra.Binding.ElementBytes;
        }

        var passes = program.Passes;
        int at = 0, temp = 0;
        while (at < passes.Count)
        {
            CodegenRowFusionPlan? fused = null;
            int end = at;
            for (int stop = passes.Count; rowFusion && stop >= at + 2 && fused is null; stop--)
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
                    (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY, fused.Parameters) { Calls = emitter.CalledFunctions });
                string label = "row-fused x" + (end - at).ToString(CultureInfo.InvariantCulture);
                if (fused.ColumnPasses.Count != 0)
                {
                    // Column passes: the row kernel wrote one partial per block; one kernel sums them.
                    var partials = new List<string>();
                    var outputs = new List<string>();
                    foreach (string columnPass in fused.ColumnPasses)
                        foreach (var candidate in fused.Program.Passes)
                            if (string.Equals(candidate.Name, columnPass, StringComparison.Ordinal))
                            {
                                string partial = CodegenRowFusionPlan.PartialName(candidate.Output.Name);
                                buffers[partial] = (long)emitter.LaunchBlocks * fused.Width;
                                partials.Add(partial);
                                outputs.Add(candidate.Output.Name);
                            }
                    var combiner = new PtxAffineEmitter();
                    string combineEntry = entry + "_combine";
                    string combinePtx = combiner.EmitColumnCombine(fused, combineEntry, checked((int)emitter.LaunchBlocks),
                        computeMajor, computeMinor);
                    partials.AddRange(outputs);
                    steps.Add(new PtxProgramStep(combinePtx, combineEntry, combiner.LaunchBlocks,
                        (uint)combiner.LaunchBlockX, (uint)combiner.LaunchBlockY, partials));
                    label += " + column combine";
                }
                for (int k = at; k < end; k++) labels.Add(label);
                at = end;
                continue;
            }

            var pass = passes[at++];
            var custom = lowerPass?.Invoke(pass);
            var split = custom is null ? CodegenSplitReduction.TryPlan(pass) : custom.Split;
            if (split is not null)
            {
                string scratch = "#temp" + temp++.ToString(CultureInfo.InvariantCulture);
                buffers[scratch] = split.TempElements;

                var partialArgs = new List<string>();
                foreach (int source in CodegenSplitReduction.PartialOperandSources(pass))
                    partialArgs.Add(pass.Inputs[source].Name);
                partialArgs.Add(scratch);

                var combineArgs = new List<string> { scratch };
                foreach (int source in CodegenSplitReduction.CombineOperandSources(pass))
                    combineArgs.Add(pass.Inputs[source].Name);
                combineArgs.Add(pass.Output.Name);

                if (custom is not null)
                {
                    steps.Add(Step(custom.Kernels[0], partialArgs));
                    steps.Add(Step(custom.Kernels[1], combineArgs));
                    labels.Add(custom.Label);
                }
                else
                {
                    steps.Add(Single(split.Partial, partialArgs, computeMajor, computeMinor));
                    steps.Add(Single(split.Combine, combineArgs, computeMajor, computeMinor));
                    labels.Add("modelled split x2");
                }
                continue;
            }

            var args = new List<string>();
            foreach (var input in pass.Inputs) args.Add(input.Name);
            args.Add(pass.Output.Name);
            foreach (var extra in pass.ExtraOutputs) args.Add(extra.Binding.Name);
            if (custom is not null)
            {
                steps.Add(Step(custom.Kernels[0], args));
                labels.Add(custom.Label);
            }
            else
            {
                steps.Add(Single(pass, args, computeMajor, computeMinor));
                labels.Add("single");
            }
        }
        return new PtxProgramLowering(steps, buffers, labels, elementBytes);
    }

    private static PtxProgramStep Step(PtxPassKernel kernel, IReadOnlyList<string> args) =>
        new(kernel.Ptx, kernel.EntryPoint, kernel.Blocks, kernel.BlockX, kernel.BlockY, args,
            kernel.DynamicSharedMemoryBytes);

    private static PtxProgramStep Single(CodegenKernelSpec spec, IReadOnlyList<string> args, int major, int minor)
    {
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(spec, major, minor);
        return new PtxProgramStep(ptx, spec.Name, emitter.LaunchBlocks,
            (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY, args) { Calls = emitter.CalledFunctions };
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
