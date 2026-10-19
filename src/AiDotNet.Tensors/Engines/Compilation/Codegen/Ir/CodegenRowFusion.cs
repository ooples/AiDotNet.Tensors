// Copyright (c) AiDotNet. All rights reserved.
// Fusing a row-wise program -- softmax, LayerNorm, RMSNorm, GroupNorm -- into ONE kernel.
//
// As separate passes each statistic re-reads the whole input: LayerNorm at 8192x1024 is three
// reads and a write of 32 MB, and on a GTX 1660 Ti it ran at 0.62x PyTorch's single fused
// kernel even though every pass was near DRAM bandwidth. Bandwidth was not the limit; bytes
// were. A kernel that keeps each row in registers across every pass reads x once, writes y
// once, and turns each statistic into a register-to-register reduction.
//
// A ROW is the program's leading axes, its COLUMNS the trailing ones: LayerNorm is (i | j),
// GroupNorm is (n, g | cg, s). A row up to 1024 wide is one warp; wider rows are one block,
// whose warps combine through shared memory. Every binding is addressed as
// rowPart(row axes) + colPart(column axes), and is one of three kinds:
//
//   Matrix    reads the innermost column axis with stride 1: streamed, 128-bit where it can.
//   Feature   does not read the innermost column axis: ONE value per chunk of columns
//             (GroupNorm's gamma[g, cg] is constant across a chunk of s).
//   Row       reads no column axis: a per-row statistic.
//
// The plan is the IR's half: it decides whether a program has this shape and classifies every
// binding. Emission is PtxAffineEmitter.EmitRowProgram.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>How a binding of a row-fused program is addressed.</summary>
public enum CodegenRowOperandKind
{
    /// <summary>Reads the innermost column axis with stride 1: one value per element.</summary>
    Matrix,

    /// <summary>Reads column axes but not the innermost: one value per chunk of columns.</summary>
    Feature,

    /// <summary>Reads no column axis: a per-row statistic.</summary>
    Row,
}

/// <summary>Where one binding lives: its kind and its element strides per axis.</summary>
public sealed class CodegenRowBinding
{
    internal CodegenRowBinding(string name, CodegenRowOperandKind kind, long[] rowStrides, long[] columnStrides)
    {
        Name = name;
        Kind = kind;
        RowStrides = rowStrides;
        ColumnStrides = columnStrides;
    }

    /// <summary>Tensor name, shared by every pass that reads or writes it.</summary>
    public string Name { get; }

    /// <summary>How the kernel addresses it.</summary>
    public CodegenRowOperandKind Kind { get; }

    /// <summary>Element stride of each row axis (0 when not read).</summary>
    public IReadOnlyList<long> RowStrides { get; }

    /// <summary>Element stride of each column axis (0 when not read).</summary>
    public IReadOnlyList<long> ColumnStrides { get; }
}

/// <summary>A program accepted for single-kernel row fusion.</summary>
public sealed class CodegenRowFusionPlan
{
    internal CodegenRowFusionPlan(
        CodegenProgram program, int[] rowExtents, int[] columnExtents,
        IReadOnlyList<string> parameters, IReadOnlyDictionary<string, CodegenRowBinding> bindings)
    {
        Program = program;
        RowExtents = rowExtents;
        ColumnExtents = columnExtents;
        long rows = 1, width = 1;
        foreach (int e in rowExtents) rows *= e;
        foreach (int e in columnExtents) width *= e;
        Rows = checked((int)rows);
        Width = checked((int)width);
        Parameters = parameters;
        Bindings = bindings;
        var kinds = new Dictionary<string, CodegenRowOperandKind>(StringComparer.Ordinal);
        foreach (var pair in bindings) kinds[pair.Key] = pair.Value.Kind;
        Kinds = kinds;
    }

    /// <summary>The program being fused.</summary>
    public CodegenProgram Program { get; }

    /// <summary>Extents of the row axes, outermost first.</summary>
    public IReadOnlyList<int> RowExtents { get; }

    /// <summary>Extents of the column axes, outermost first.</summary>
    public IReadOnlyList<int> ColumnExtents { get; }

    /// <summary>Rows: the product of the row extents.</summary>
    public int Rows { get; }

    /// <summary>Elements per row: the product of the column extents.</summary>
    public int Width { get; }

    /// <summary>
    /// The fused kernel's parameters, by name and in order: every program source, then every
    /// pass output. A launcher binds buffers by these names.
    /// </summary>
    public IReadOnlyList<string> Parameters { get; }

    /// <summary>Address of every named tensor.</summary>
    public IReadOnlyDictionary<string, CodegenRowBinding> Bindings { get; }

    /// <summary>The kind of every named tensor.</summary>
    public IReadOnlyDictionary<string, CodegenRowOperandKind> Kinds { get; }

    /// <summary>Elements each thread holds, at the scalar geometry.</summary>
    public int PerLane => Geometry(1) is { } g ? g.PerThread : 0;

    /// <summary>
    /// Threads per row and elements per access for a requested vector width: the
    /// narrowest power-of-two thread count from 32 to 1024 that leaves each thread at most
    /// <see cref="CodegenRowFusion.MaxPerThread"/> elements, such that one STEP (threads x
    /// vector columns) never straddles a run of the innermost column axis. Null when no
    /// geometry fits.
    /// </summary>
    public (int Threads, int Vector, int Step, int PerThread)? Geometry(int maxVector)
    {
        for (int vector = maxVector; vector >= 1; vector /= 2)
        {
            if (vector != 1 && vector != 2 && vector != 4) continue;
            for (int threads = CodegenWarpReduction.Lanes; threads <= CodegenRowFusion.MaxRowThreads; threads *= 2)
            {
                long step = (long)threads * vector;
                if (Width % step != 0) continue;
                if (ColumnExtents[ColumnExtents.Count - 1] % step != 0) continue;
                long perThread = Width / threads;
                if (perThread > CodegenRowFusion.MaxPerThread) continue;
                return (threads, vector, (int)step, (int)perThread);
            }
        }
        return null;
    }
}

/// <summary>Decides whether a program can run as one row-per-warp-or-block kernel.</summary>
public static class CodegenRowFusion
{
    /// <summary>Most elements of one source a thread holds in registers.</summary>
    public const int MaxPerThread = 32;

    /// <summary>Most threads cooperating on one row: one full block.</summary>
    public const int MaxRowThreads = 1024;

    /// <summary>Widest row: every thread of a full block holding its maximum.</summary>
    public const int MaxWidth = MaxPerThread * MaxRowThreads;

    /// <summary>Returns the plan, or null with the reason the program does not fit.</summary>
    public static CodegenRowFusionPlan? TryPlan(CodegenProgram program, out string reason)
    {
        if (program is null) throw new ArgumentNullException(nameof(program));
        reason = "";
        if (program.Passes.Count == 0) { reason = "no passes"; return null; }

        // The row/column split comes from the first REDUCING pass: its parallel axes are the
        // row. A program with no reduction is pointwise; its last axis is the column.
        CodegenKernelSpec? reducing = null;
        foreach (var pass in program.Passes)
            if (pass.Reduce != CodegenReduceKind.None) { reducing = pass; break; }
        var shape = program.Passes[0].Space.Axes;
        int rowAxes = reducing is null ? shape.Count - 1 : reducing.Space.ParallelAxes.Length;
        if (rowAxes < 1 || rowAxes >= shape.Count) { reason = "no row/column split"; return null; }

        var extents = new int[shape.Count];
        for (int a = 0; a < shape.Count; a++) extents[a] = shape[a].Extent;
        var rowExtents = new int[rowAxes];
        var columnExtents = new int[shape.Count - rowAxes];
        Array.Copy(extents, 0, rowExtents, 0, rowAxes);
        Array.Copy(extents, rowAxes, columnExtents, 0, columnExtents.Length);

        var bindings = new Dictionary<string, CodegenRowBinding>(StringComparer.Ordinal);
        var produced = new HashSet<string>(StringComparer.Ordinal);
        foreach (var pass in program.Passes)
        {
            var axes = pass.Space.Axes;
            bool reduces = pass.Reduce != CodegenReduceKind.None;
            if (axes.Count != extents.Length)
            {
                reason = pass.Name + " has a different axis count";
                return null;
            }
            for (int a = 0; a < axes.Count; a++)
            {
                bool shouldReduce = reduces && a >= rowAxes;
                if (axes[a].Extent != extents[a] || axes[a].IsReduction != shouldReduce)
                {
                    reason = pass.Name + " axis " + a + " does not match the row/column split";
                    return null;
                }
            }
            if (pass.Algebra != CodegenAlgebra.Real || pass.SecondaryOutput is not null ||
                pass.ExtraOutputs.Count != 0)
            {
                reason = pass.Name + " has an argmax, extra outputs or a non-real algebra";
                return null;
            }

            foreach (var input in pass.Inputs)
            {
                var binding = Classify(input, rowAxes, extents);
                if (binding is null)
                {
                    reason = pass.Name + " reads '" + input.Name + "' with an index map the row kernel does not hold";
                    return null;
                }
                if (bindings.TryGetValue(input.Name, out var known) && !Same(known, binding))
                {
                    reason = "'" + input.Name + "' is read two different ways";
                    return null;
                }
                bindings[input.Name] = binding;
                // A matrix the kernel PRODUCED is not held in registers; only sources are.
                if (binding.Kind != CodegenRowOperandKind.Row && produced.Contains(input.Name))
                {
                    reason = pass.Name + " reads the non-row intermediate '" + input.Name + "'";
                    return null;
                }
            }

            // The reduction's epilogue acts on a per-row value, so it can only read
            // per-row operands; a column-dependent one would have no single value.
            if (reduces)
                foreach (int? operand in new[] { pass.BiasInput, pass.ScaleInput, pass.PostScaleInput, pass.PostBiasInput })
                    if (operand is { } at && bindings[pass.Inputs[at].Name].Kind != CodegenRowOperandKind.Row)
                    {
                        reason = pass.Name + "'s epilogue reads a non-row operand";
                        return null;
                    }

            var output = Classify(pass.Output, rowAxes, extents);
            var expected = reduces ? CodegenRowOperandKind.Row : CodegenRowOperandKind.Matrix;
            if (output is null || output.Kind != expected || pass.Output.NeedsAtomicStore)
            {
                reason = pass.Name + "'s output is not a " + expected + " binding";
                return null;
            }
            if (!reduces && !IsFlatColumns(output, columnExtents))
            {
                reason = pass.Name + "'s output is not stored in column order";
                return null;
            }
            bindings[pass.Output.Name] = output;
            produced.Add(pass.Output.Name);
        }

        var parameters = new List<string>();
        foreach (var source in program.Sources()) parameters.Add(source.Name);
        foreach (var pass in program.Passes) parameters.Add(pass.Output.Name);
        var plan = new CodegenRowFusionPlan(program, rowExtents, columnExtents, parameters, bindings);
        if (plan.Geometry(1) is null)
        {
            reason = "no thread geometry holds a " + plan.Width + "-wide row within " +
                     MaxPerThread + " elements per thread";
            return null;
        }
        return plan;
    }

    /// <summary>
    /// Plans a single POINTWISE spec as a one-pass row program, so it streams through the
    /// row kernel's 128-bit accesses.
    /// </summary>
    /// <remarks>
    /// Measured on a GTX 1660 Ti at 8192x1024: bias+GELU 316.1 us one element per thread,
    /// 260.9 us through the row kernel. A multi-axis spec is used as it is. A flat spec whose
    /// every binding is the same contiguous run is viewed as rows of the widest of 1024,
    /// 512, 256 or 128 that divides it -- the same bytes, indexed in two parts.
    /// </remarks>
    public static CodegenRowFusionPlan? TryPlanStreaming(CodegenKernelSpec spec)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        if (spec.Reduce != CodegenReduceKind.None || spec.Space.ReductionAxes.Length != 0) return null;
        if (spec.ExtraOutputs.Count != 0 || spec.SecondaryOutput is not null) return null;
        if (!Distinct(spec)) return null;

        var axes = spec.Space.Axes;
        if (axes.Count >= 2)
        {
            // Streaming is for rows that fill a warp; a narrow last axis would be all guards.
            if (axes[axes.Count - 1].Extent % CodegenWarpReduction.Lanes != 0) return null;
            return TryPlan(new CodegenProgram(new[] { spec }, new long[] { spec.Output.ElementCount }, spec.Name), out _);
        }
        if (axes.Count != 1) return null;

        int n = axes[0].Extent;
        foreach (var binding in Bindings(spec))
            if (binding.Shape.Count != 1 || binding.Shape[0] != n || !IsAxis(binding.Map[0], 0) ||
                binding.HasIndirection || binding.ElementType != CodegenElementType.Float32)
                return null;

        int width = 0;
        foreach (int candidate in new[] { 1024, 512, 256, 128 })
            if (n % candidate == 0) { width = candidate; break; }
        if (width == 0) return null;
        int rows = n / width;

        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("row", rows), CodegenAxis.Parallel("j", width));
        CodegenTensorBinding Reshape(CodegenTensorBinding b) => new(
            b.ParameterIndex, b.Name, new[] { rows, width },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, b.IsOutput, b.ElementType);
        var inputs = new CodegenTensorBinding[spec.Inputs.Count];
        for (int i = 0; i < inputs.Length; i++) inputs[i] = Reshape(spec.Inputs[i]);
        var products = new int[spec.ProductInputs.Count];
        for (int i = 0; i < products.Length; i++) products[i] = spec.ProductInputs[i];

        var reshaped = new CodegenKernelSpec(spec.Name, space, inputs, Reshape(spec.Output), products,
            CodegenReduceKind.None, biasInput: spec.BiasInput, scaleInput: spec.ScaleInput,
            activation: spec.Activation, reduceScale: spec.ReduceScale, preReduce: spec.PreReduce,
            preBiasInput: spec.PreBiasInput, preBiasScale: spec.PreBiasScale,
            reduceOffset: spec.ReduceOffset,
            postScaleInput: spec.PostScaleInput, postBiasInput: spec.PostBiasInput);
        return TryPlan(new CodegenProgram(new[] { reshaped }, new long[] { n }, spec.Name), out _);
    }

    /// <summary>
    /// A row program binds by NAME, a spec by position; two operands sharing a name would
    /// collapse into one parameter, so such a spec is not streamed this way.
    /// </summary>
    private static bool Distinct(CodegenKernelSpec spec)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var binding in Bindings(spec))
            if (!names.Add(binding.Name)) return false;
        return true;
    }

    private static IEnumerable<CodegenTensorBinding> Bindings(CodegenKernelSpec spec)
    {
        foreach (var input in spec.Inputs) yield return input;
        yield return spec.Output;
    }

    /// <summary>
    /// Element strides per axis for a row-major binding whose every dimension is one axis
    /// with coefficient one, and the kind those strides imply; null for anything else.
    /// </summary>
    private static CodegenRowBinding? Classify(CodegenTensorBinding binding, int rowAxes, int[] extents)
    {
        if (binding.HasIndirection || binding.ElementType != CodegenElementType.Float32) return null;
        var shape = binding.Shape;
        var strides = new long[extents.Length];
        long stride = 1;
        for (int d = shape.Count - 1; d >= 0; d--)
        {
            var expr = binding.Map[d];
            if (expr.Terms.Count != 1 || expr.Terms[0].Coefficient != 1 || expr.Constant != 0 || expr.Divisor != 1)
                return null;
            int axis = expr.Terms[0].Axis;
            if (axis < 0 || axis >= extents.Length || shape[d] != extents[axis] || strides[axis] != 0)
                return null;
            strides[axis] = stride;
            stride *= shape[d];
        }

        var rowStrides = new long[rowAxes];
        var columnStrides = new long[extents.Length - rowAxes];
        Array.Copy(strides, 0, rowStrides, 0, rowAxes);
        Array.Copy(strides, rowAxes, columnStrides, 0, columnStrides.Length);

        bool anyColumn = false;
        foreach (long s in columnStrides) if (s != 0) anyColumn = true;
        long innermost = columnStrides[columnStrides.Length - 1];
        CodegenRowOperandKind kind;
        if (!anyColumn) kind = CodegenRowOperandKind.Row;
        else if (innermost == 1) kind = CodegenRowOperandKind.Matrix;
        else if (innermost == 0) kind = CodegenRowOperandKind.Feature;
        else return null;   // the innermost column axis is read, but not contiguously
        return new CodegenRowBinding(binding.Name, kind, rowStrides, columnStrides);
    }

    /// <summary>True when the column strides are exactly the row-major strides of the columns.</summary>
    private static bool IsFlatColumns(CodegenRowBinding binding, int[] columnExtents)
    {
        long stride = 1;
        for (int c = columnExtents.Length - 1; c >= 0; c--)
        {
            if (binding.ColumnStrides[c] != stride) return false;
            stride *= columnExtents[c];
        }
        return true;
    }

    private static bool Same(CodegenRowBinding a, CodegenRowBinding b)
    {
        if (a.Kind != b.Kind) return false;
        for (int i = 0; i < a.RowStrides.Count; i++) if (a.RowStrides[i] != b.RowStrides[i]) return false;
        for (int i = 0; i < a.ColumnStrides.Count; i++) if (a.ColumnStrides[i] != b.ColumnStrides[i]) return false;
        return true;
    }

    private static bool IsAxis(CodegenAffineExpr expr, int axis) =>
        expr.Terms.Count == 1 && expr.Terms[0].Axis == axis && expr.Terms[0].Coefficient == 1 &&
        expr.Constant == 0 && expr.Divisor == 1;
}
