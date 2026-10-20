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
//
// COLUMN PASSES reduce DOWN the rows, one output per column -- LayerNorm backward's dgamma and
// dbeta. As their own kernels each re-read dy and x, and each was a two-kernel split: five
// launches for LayerNorm backward where the hand-written kernels take two. Fused, the row
// kernel accumulates each column pass per thread across a persistent loop over rows, writes
// one partial per block, and one combine kernel sums the partials in block order.

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
    internal CodegenRowBinding(string name, CodegenRowOperandKind kind, long[] rowStrides, long[] columnStrides,
        long constantOffset, string key)
    {
        Name = name;
        Kind = kind;
        RowStrides = rowStrides;
        ColumnStrides = columnStrides;
        ConstantOffset = constantOffset;
        Key = key;
    }

    /// <summary>
    /// Element offset of the binding's constant indices -- BatchNorm's shift reads x[0, c, 0],
    /// which is x at a fixed n and position.
    /// </summary>
    public long ConstantOffset { get; }

    /// <summary>
    /// Identity of this ADDRESSING of the tensor: its name and index map. One tensor may be read
    /// two ways in one program (streamed, and at a fixed position per row); each is its own key.
    /// </summary>
    public string Key { get; }

    /// <summary>Tensor name, shared by every pass that reads or writes it.</summary>
    public string Name { get; }

    /// <summary>How the kernel addresses it.</summary>
    public CodegenRowOperandKind Kind { get; }

    /// <summary>Element stride of each row axis (0 when not read).</summary>
    public IReadOnlyList<long> RowStrides { get; }

    /// <summary>Element stride of each column axis (0 when not read).</summary>
    public IReadOnlyList<long> ColumnStrides { get; }
}

/// <summary>
/// The thread layout of one row kernel: Threads per row, each access Vector wide, one STEP
/// (Threads x Vector columns) at a time, Chunks steps per row. HELD rows keep every step in
/// registers; streamed rows sweep. A RAGGED layout's last step overhangs the row, and its
/// out-of-row lanes are predicated off. A WRAPPED layout is ragged over several column axes:
/// a step may cross runs of the innermost axis, so each lane tracks its (outer, inner) column
/// position as it advances.
/// </summary>
public readonly struct CodegenRowGeometry
{
    public CodegenRowGeometry(int threads, int vector, int step, int chunks, bool held, bool ragged, bool wrapped = false)
    {
        Threads = threads;
        Vector = vector;
        Step = step;
        Chunks = chunks;
        Held = held;
        Ragged = ragged || wrapped;
        Wrapped = wrapped;
    }

    public int Threads { get; }
    public int Vector { get; }
    public int Step { get; }
    public int Chunks { get; }
    public bool Held { get; }
    public bool Ragged { get; }
    public bool Wrapped { get; }

    /// <summary>Elements each thread holds (0 when streamed).</summary>
    public int PerThread => Held ? Chunks * Vector : 0;
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
        // By NAME, the first addressing of each tensor: what most callers mean by "x" or "gamma".
        var kinds = new Dictionary<string, CodegenRowOperandKind>(StringComparer.Ordinal);
        foreach (var binding in bindings.Values)
            if (!kinds.ContainsKey(binding.Name)) kinds[binding.Name] = binding.Kind;
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

    /// <summary>Address of every distinct addressing of a tensor, by <see cref="CodegenRowBinding.Key"/>.</summary>
    public IReadOnlyDictionary<string, CodegenRowBinding> Bindings { get; }

    /// <summary>The addressing of <paramref name="binding"/> as it appears in a pass.</summary>
    public CodegenRowBinding BindingFor(CodegenTensorBinding binding) =>
        Bindings[CodegenRowFusion.KeyOf(binding ?? throw new ArgumentNullException(nameof(binding)))];

    /// <summary>The kind of every named tensor.</summary>
    public IReadOnlyDictionary<string, CodegenRowOperandKind> Kinds { get; }

    /// <summary>Elements each thread holds, at the scalar geometry (0 when streamed).</summary>
    public int PerLane => Layout(1)?.PerThread ?? 0;

    /// <summary>
    /// Passes over the ROW axes only -- per-row arithmetic on statistics, such as BatchNorm's
    /// rstd from its two moments. Every thread of the row computes them from registers.
    /// </summary>
    public IReadOnlyCollection<string> RowPasses { get; internal set; } = Array.Empty<string>();

    /// <summary>
    /// Passes that reduce over the ROW axes, one output per column (LayerNorm backward's
    /// dgamma). In <see cref="Program"/> they are rewritten into the program's axis order. The
    /// row kernel writes their per-block partials (<see cref="PartialName"/>); a combine kernel
    /// (PtxAffineEmitter.EmitColumnCombine) finishes them.
    /// </summary>
    public IReadOnlyList<string> ColumnPasses { get; internal set; } = Array.Empty<string>();

    /// <summary>The kernel parameter holding a column pass's per-block partials.</summary>
    public static string PartialName(string columnPassOutput) => "#" + columnPassOutput + ":partials";

    /// <summary>
    /// Most elements of one source a thread holds. A plan with column passes holds at most
    /// <see cref="CodegenRowFusion.ColumnFusedPerThread"/>: each column pass keeps one
    /// accumulator per held element, so a 1024-wide row is a 128-thread block rather than a
    /// warp of 32 elements a thread.
    /// </summary>
    public int MaxPerThread => ColumnPasses.Count == 0 ? CodegenRowFusion.MaxPerThread : CodegenRowFusion.ColumnFusedPerThread;

    /// <summary>
    /// Threads per row and elements per access for a requested vector width: the
    /// narrowest power-of-two thread count from 32 to 1024 that leaves each thread at most
    /// <see cref="CodegenRowFusion.MaxPerThread"/> elements, such that one STEP (threads x
    /// vector columns) never straddles a run of the innermost column axis. Null when no
    /// geometry fits.
    /// </summary>
    public (int Threads, int Vector, int Step, int PerThread)? Geometry(int maxVector)
    {
        // HELD: the narrowest power-of-two thread count that holds the row in registers.
        for (int vector = maxVector; vector >= 1; vector /= 2)
        {
            if (vector != 1 && vector != 2 && vector != 4) continue;
            for (int threads = CodegenWarpReduction.Lanes; threads <= CodegenRowFusion.MaxRowThreads; threads *= 2)
            {
                long step = (long)threads * vector;
                if (Width % step != 0) continue;
                if (ColumnExtents[ColumnExtents.Count - 1] % step != 0) continue;
                long perThread = Width / threads;
                if (perThread > MaxPerThread) continue;
                return (threads, vector, (int)step, (int)perThread);
            }
        }
        return null;
    }

    /// <summary>
    /// The STREAMED geometry for a row too wide to hold: the most threads (a multiple of 32,
    /// up to a block) and the widest access such that one step divides the innermost column
    /// run. A BatchNorm channel is 32 x 3136 elements and 3136 = 64 x 49, so this is
    /// 224 threads x 2 elements rather than any power of two. Null when nothing divides.
    /// </summary>
    public (int Threads, int Vector, int Step, int Chunks)? StreamGeometry(int maxVector)
    {
        int innermost = ColumnExtents[ColumnExtents.Count - 1];
        (int, int, int, int)? best = null;
        long bestThreads = 0;
        for (int vector = maxVector; vector >= 1; vector /= 2)
        {
            if (vector != 1 && vector != 2 && vector != 4) continue;
            for (int threads = CodegenRowFusion.MaxRowThreads; threads >= CodegenWarpReduction.Lanes;
                 threads -= CodegenWarpReduction.Lanes)
            {
                long step = (long)threads * vector;
                if (innermost % step != 0) continue;
                // Prefer more threads; at equal threads, the wider access found first.
                if (threads > bestThreads)
                {
                    bestThreads = threads;
                    best = (threads, vector, (int)step, (int)(Width / step));
                }
                break;
            }
        }
        return best;
    }

    /// <summary>
    /// The RAGGED geometry, for a row no step divides -- LayerNorm over 100 or 1000 features,
    /// softmax over a 50257-word vocabulary. It needs ONE column axis (so a step is a contiguous
    /// run and only the last one overhangs), a vector that divides the width (so a vector is
    /// wholly inside the row or wholly outside it), and vector-aligned rows in every streamed
    /// binding. Held when some power-of-two thread count holds the row at most
    /// <see cref="CodegenRowFusion.MaxPerThread"/> elements per thread; streamed by a full block
    /// otherwise. Null when the row has several column axes.
    /// </summary>
    public CodegenRowGeometry? RaggedGeometry(int maxVector)
    {
        if (ColumnExtents.Count != 1) return null;
        for (int vector = maxVector; vector >= 1; vector /= 2)
        {
            if (vector != 1 && vector != 2 && vector != 4) continue;
            if (Width % vector != 0 || !VectorAligned(vector)) continue;
            for (int threads = CodegenWarpReduction.Lanes; threads <= CodegenRowFusion.MaxRowThreads; threads *= 2)
            {
                int step = threads * vector;
                int chunks = (Width + step - 1) / step;
                if ((long)chunks * vector <= MaxPerThread)
                    return new CodegenRowGeometry(threads, vector, step, chunks, held: true, ragged: true);
            }
            int blockStep = CodegenRowFusion.RaggedStreamThreads * vector;
            return new CodegenRowGeometry(CodegenRowFusion.RaggedStreamThreads, vector, blockStep,
                (Width + blockStep - 1) / blockStep, held: false, ragged: true);
        }
        return null;
    }

    /// <summary>
    /// The WRAPPED geometry, for a row over several column axes whose innermost run no step
    /// divides -- BatchNorm's (n, h*w) at 7x7, 14x14 or 28x28, GroupNorm's (channels, h*w).
    /// Each binding must address the outer column axes as ONE stride (<see cref="OuterStride"/>),
    /// so a column is (outer, inner); the vector divides the innermost extent, so a vector never
    /// crosses a run. Held when a power-of-two thread count holds the row in at most
    /// <see cref="CodegenRowFusion.MaxPerThread"/> elements a thread, streamed by a
    /// <see cref="CodegenRowFusion.RaggedStreamThreads"/>-thread block otherwise.
    /// </summary>
    public CodegenRowGeometry? WrappedGeometry(int maxVector)
    {
        if (ColumnExtents.Count < 2) return null;
        foreach (var binding in Bindings.Values)
            if (binding.Kind != CodegenRowOperandKind.Row && OuterStride(binding) is null) return null;
        int inner = ColumnExtents[ColumnExtents.Count - 1];
        for (int vector = maxVector; vector >= 1; vector /= 2)
        {
            if (vector != 1 && vector != 2 && vector != 4) continue;
            if (inner % vector != 0 || !VectorAligned(vector)) continue;
            for (int threads = CodegenWarpReduction.Lanes; threads <= CodegenRowFusion.MaxRowThreads; threads *= 2)
            {
                int step = threads * vector;
                int chunks = (Width + step - 1) / step;
                if ((long)chunks * vector <= MaxPerThread)
                    return new CodegenRowGeometry(threads, vector, step, chunks, held: true, ragged: true, wrapped: true);
            }
            int blockStep = CodegenRowFusion.RaggedStreamThreads * vector;
            return new CodegenRowGeometry(CodegenRowFusion.RaggedStreamThreads, vector, blockStep,
                (Width + blockStep - 1) / blockStep, held: false, ragged: true, wrapped: true);
        }
        return null;
    }

    /// <summary>
    /// The element stride of a binding's OUTER column index -- the flattened index over every
    /// column axis but the innermost -- or null when those axes do not flatten to one stride
    /// (each outer axis's stride is the next one's stride times its extent, or the binding
    /// does not read the outer axes at all).
    /// </summary>
    public long? OuterStride(CodegenRowBinding binding)
    {
        int last = ColumnExtents.Count - 1;
        for (int k = 0; k < last - 1; k++)
            if (binding.ColumnStrides[k] != binding.ColumnStrides[k + 1] * ColumnExtents[k + 1]) return null;
        return binding.ColumnStrides[last - 1];
    }

    // Every streamed binding's row starts (and fixed offset) on a vector boundary.
    private bool VectorAligned(int vector)
    {
        foreach (var binding in Bindings.Values)
        {
            if (binding.Kind != CodegenRowOperandKind.Matrix) continue;
            if (binding.ConstantOffset % vector != 0) return false;
            foreach (long stride in binding.RowStrides)
                if (stride % vector != 0) return false;
            for (int k = 0; k < binding.ColumnStrides.Count - 1; k++)
                if (binding.ColumnStrides[k] % vector != 0) return false;
        }
        return true;
    }

    /// <summary>
    /// The layout the emitter uses: a row held exactly, else held raggedly (one column axis) or
    /// wrapped (several), else streamed exactly, else streamed raggedly or wrapped. Null when
    /// none fits. A held layout beats an exact stream: it reads the row once, and ragged-held
    /// LayerNorm ran at the DRAM ceiling (1024x3136: 245 GB/s of ~256 on a GTX 1660 Ti).
    /// </summary>
    public CodegenRowGeometry? Layout(int maxVector)
    {
        if (Geometry(maxVector) is { } h)
            return new CodegenRowGeometry(h.Threads, h.Vector, h.Step, h.PerThread / h.Vector, held: true, ragged: false);
        var ragged = RaggedGeometry(maxVector) ?? WrappedGeometry(maxVector);
        if (ragged is { Held: true } r) return r;
        if (StreamGeometry(maxVector) is { } s)
            return new CodegenRowGeometry(s.Threads, s.Vector, s.Step, s.Chunks, held: false, ragged: false);
        return ragged;
    }

    /// <summary>True when the row is held in registers; false when it is streamed in sweeps.</summary>
    public bool Holds(int maxVector) => Layout(maxVector) is { Held: true };
}

/// <summary>Decides whether a program can run as one row-per-warp-or-block kernel.</summary>
public static class CodegenRowFusion
{
    /// <summary>Most elements of one source a thread holds in registers.</summary>
    public const int MaxPerThread = 32;

    /// <summary>Most threads cooperating on one row: one full block.</summary>
    public const int MaxRowThreads = 1024;

    /// <summary>
    /// Most elements a thread holds when the plan has column passes: each column pass adds an
    /// accumulator per element, and at 32 a LayerNorm backward thread held 64 sources and 64
    /// accumulators.
    /// </summary>
    public const int ColumnFusedPerThread = 8;

    /// <summary>
    /// Threads of a ragged streamed row. A 1024-thread block caps ptxas at 64 registers, and an
    /// unrolled sweep of 4-wide loads spilled 72 bytes a thread at that cap (LayerNorm 64x50000).
    /// </summary>
    public const int RaggedStreamThreads = 256;

    /// <summary>
    /// Widest row HELD in registers: every thread of a full block holding its maximum. Wider
    /// rows are streamed in sweeps instead.
    /// </summary>
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
        // The row/column shape is the reducing pass's: a program may OPEN with a row pass.
        var shape = (reducing ?? program.Passes[0]).Space.Axes;
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
        var rowPasses = new HashSet<string>(StringComparer.Ordinal);
        var columnPasses = new List<string>();
        var rewrittenPasses = new Dictionary<string, CodegenKernelSpec>(StringComparer.Ordinal);
        foreach (var pass in program.Passes)
        {
            var axes = pass.Space.Axes;
            bool reduces = pass.Reduce != CodegenReduceKind.None;

            // A ROW PASS: pointwise over the row axes alone, reading and writing statistics.
            if (!reduces && axes.Count == rowAxes)
            {
                bool shaped = true;
                for (int a = 0; a < rowAxes; a++)
                    if (axes[a].IsReduction || axes[a].Extent != extents[a]) shaped = false;
                if (!shaped || pass.SecondaryOutput is not null || pass.ExtraOutputs.Count != 0 ||
                    pass.Algebra != CodegenAlgebra.Real)
                {
                    reason = pass.Name + " is neither (row | column) nor a row pass";
                    return null;
                }
                foreach (var input in pass.Inputs)
                {
                    var binding = Classify(input, rowAxes, extents);
                    if (binding is null || binding.Kind != CodegenRowOperandKind.Row)
                    {
                        reason = pass.Name + " is a row pass reading the non-row '" + input.Name + "'";
                        return null;
                    }
                    bindings[binding.Key] = binding;
                }
                var rowOut = Classify(pass.Output, rowAxes, extents);
                if (rowOut is null || rowOut.Kind != CodegenRowOperandKind.Row || rowOut.ConstantOffset != 0)
                {
                    reason = pass.Name + "'s output is not per-row";
                    return null;
                }
                bindings[rowOut.Key] = rowOut;
                produced.Add(pass.Output.Name);
                rowPasses.Add(pass.Name);
                continue;
            }

            // A COLUMN PASS: a sum down the rows, one output per column. Its axes alone cannot say
            // so -- (j | i) at 100x100 has the shape of a row reduction (i | j) -- but its maps
            // can: read as a row reduction it reads x and dy transposed, which does not classify.
            if (reduces && pass.Space.ParallelAxes.Length == extents.Length - rowAxes &&
                pass.Space.ReductionAxes.Length == rowAxes && axes.Count == extents.Length &&
                !ReadsAsARowReduction(pass, rowAxes, extents))
            {
                var rewritten = RewriteColumnPass(pass, rowAxes, extents, out string columnReason);
                if (rewritten is null) { reason = columnReason; return null; }
                foreach (var input in rewritten.Inputs)
                {
                    var binding = Classify(input, rowAxes, extents);
                    if (binding is null)
                    {
                        reason = pass.Name + " reads '" + input.Name + "' with an index map the row kernel does not hold";
                        return null;
                    }
                    if (binding.Kind != CodegenRowOperandKind.Row && produced.Contains(input.Name))
                    {
                        reason = pass.Name + " reads the non-row intermediate '" + input.Name + "'";
                        return null;
                    }
                    bindings[binding.Key] = binding;
                }
                var columnOut = Classify(rewritten.Output, rowAxes, extents);
                if (columnOut is null || columnOut.Kind == CodegenRowOperandKind.Row || columnOut.ConstantOffset != 0 ||
                    !IsContiguousColumnVector(columnOut, columnExtents))
                {
                    reason = pass.Name + "'s output is not one contiguous value per column";
                    return null;
                }
                bindings[columnOut.Key] = columnOut;
                produced.Add(pass.Output.Name);
                columnPasses.Add(pass.Name);
                rewrittenPasses[pass.Name] = rewritten;
                continue;
            }

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
                if (rewrittenPasses.ContainsKey(ProducerOf(program, input.Name) ?? ""))
                {
                    reason = pass.Name + " reads '" + input.Name + "', which is final only after the combine";
                    return null;
                }
                var binding = Classify(input, rowAxes, extents);
                if (binding is null)
                {
                    reason = pass.Name + " reads '" + input.Name + "' with an index map the row kernel does not hold";
                    return null;
                }
                bindings[binding.Key] = binding;
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
                    if (operand is { } at && bindings[KeyOf(pass.Inputs[at])].Kind != CodegenRowOperandKind.Row)
                    {
                        reason = pass.Name + "'s epilogue reads a non-row operand";
                        return null;
                    }

            var output = Classify(pass.Output, rowAxes, extents);
            var expected = reduces ? CodegenRowOperandKind.Row : CodegenRowOperandKind.Matrix;
            if (output is null || output.Kind != expected || output.ConstantOffset != 0 || pass.Output.NeedsAtomicStore)
            {
                reason = pass.Name + "'s output is not a " + expected + " binding";
                return null;
            }
            bindings[output.Key] = output;
            produced.Add(pass.Output.Name);
        }

        var parameters = new List<string>();
        foreach (var source in program.Sources()) parameters.Add(source.Name);
        foreach (var pass in program.Passes)
            parameters.Add(rewrittenPasses.ContainsKey(pass.Name)
                ? CodegenRowFusionPlan.PartialName(pass.Output.Name) : pass.Output.Name);
        // The emitter reads the column passes in the program's axis order.
        var fusedProgram = program;
        if (rewrittenPasses.Count != 0)
        {
            var passes = new List<CodegenKernelSpec>(program.Passes.Count);
            foreach (var pass in program.Passes)
                passes.Add(rewrittenPasses.TryGetValue(pass.Name, out var rewritten) ? rewritten : pass);
            fusedProgram = new CodegenProgram(passes, program.TempElements, program.Description);
        }
        var plan = new CodegenRowFusionPlan(fusedProgram, rowExtents, columnExtents, parameters, bindings)
        {
            RowPasses = rowPasses,
            ColumnPasses = columnPasses,
        };
        if (plan.Layout(1) is null)
        {
            reason = "no thread geometry fits a " + plan.Width + "-wide row: its outer column axes do not flatten";
            return null;
        }
        // Column passes accumulate per held element; a streamed row has nothing to hold them in.
        if (columnPasses.Count != 0 && (plan.Layout(1) is not { Held: true } || plan.Layout(4) is not { Held: true }))
        {
            reason = "column passes need the row held in registers";
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
            postScaleInput: spec.PostScaleInput, postBiasInput: spec.PostBiasInput,
            body: spec.Body);
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
        long stride = 1, constantOffset = 0;
        for (int d = shape.Count - 1; d >= 0; d--)
        {
            var expr = binding.Map[d];
            // A constant index: a fixed position along a dimension no axis walks.
            if (expr.Terms.Count == 0 && expr.Divisor == 1)
            {
                if (expr.Constant < 0 || expr.Constant >= shape[d]) return null;
                constantOffset += expr.Constant * stride;
                stride *= shape[d];
                continue;
            }
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
        return new CodegenRowBinding(binding.Name, kind, rowStrides, columnStrides, constantOffset, KeyOf(binding));
    }

    /// <summary>
    /// True when the pass reads as an ordinary row reduction: its leading parallel axes are the
    /// rows, with the program's extents, and every binding classifies with a per-row output.
    /// </summary>
    private static bool ReadsAsARowReduction(CodegenKernelSpec pass, int rowAxes, int[] extents)
    {
        var axes = pass.Space.Axes;
        for (int a = 0; a < axes.Count; a++)
            if (axes[a].IsReduction != (a >= rowAxes) || axes[a].Extent != extents[a]) return false;
        foreach (var input in pass.Inputs)
            if (Classify(input, rowAxes, extents) is null) return false;
        return Classify(pass.Output, rowAxes, extents) is { Kind: CodegenRowOperandKind.Row };
    }

    /// <summary>The pass that produces <paramref name="name"/>, or null for a source.</summary>
    private static string? ProducerOf(CodegenProgram program, string name)
    {
        foreach (var pass in program.Passes)
            if (string.Equals(pass.Output.Name, name, StringComparison.Ordinal)) return pass.Name;
        return null;
    }

    /// <summary>
    /// A column pass rewritten into the program's axis order: its k-th reduction axis is row
    /// axis k and its k-th parallel axis is column axis k, every index map renumbered to match.
    /// Null, with the reason, when it carries what the combine cannot finish -- an epilogue that
    /// reads a tensor, extra outputs, a non-real algebra, an indirect index.
    /// </summary>
    private static CodegenKernelSpec? RewriteColumnPass(CodegenKernelSpec pass, int rowAxes, int[] extents, out string reason)
    {
        reason = "";
        if (pass.Reduce != CodegenReduceKind.Sum || pass.Algebra != CodegenAlgebra.Real ||
            pass.ExtraOutputs.Count != 0 || pass.SecondaryOutput is not null ||
            pass.BiasInput is not null || pass.ScaleInput is not null ||
            pass.PostScaleInput is not null || pass.PostBiasInput is not null)
        {
            reason = pass.Name + " is a column reduction with an epilogue or outputs the combine does not finish";
            return null;
        }
        var axes = pass.Space.Axes;
        int[] parallel = pass.Space.ParallelAxes, reduction = pass.Space.ReductionAxes;
        var image = new int[axes.Count];
        var programAxes = new CodegenAxis[extents.Length];
        for (int k = 0; k < reduction.Length; k++)
        {
            if (axes[reduction[k]].Extent != extents[k])
            {
                reason = pass.Name + " reduces over axes that are not the rows";
                return null;
            }
            image[reduction[k]] = k;
            programAxes[k] = CodegenAxis.Reduce(axes[reduction[k]].Name, extents[k]);
        }
        for (int k = 0; k < parallel.Length; k++)
        {
            if (axes[parallel[k]].Extent != extents[rowAxes + k])
            {
                reason = pass.Name + "'s outputs are not the columns";
                return null;
            }
            image[parallel[k]] = rowAxes + k;
            programAxes[rowAxes + k] = CodegenAxis.Parallel(axes[parallel[k]].Name, extents[rowAxes + k]);
        }

        CodegenTensorBinding? Renumber(CodegenTensorBinding binding)
        {
            if (binding.HasIndirection) return null;
            var map = new CodegenAffineExpr[binding.Map.Count];
            for (int d = 0; d < map.Length; d++)
            {
                var expr = binding.Map[d];
                var terms = new CodegenAffineTerm[expr.Terms.Count];
                for (int t = 0; t < terms.Length; t++)
                    terms[t] = new CodegenAffineTerm(image[expr.Terms[t].Axis], expr.Terms[t].Coefficient);
                map[d] = new CodegenAffineExpr(terms, expr.Constant, expr.Divisor, expr.RequiresExactDivision);
            }
            var shape = new int[binding.Shape.Count];
            for (int d = 0; d < shape.Length; d++) shape[d] = binding.Shape[d];
            return new CodegenTensorBinding(binding.ParameterIndex, binding.Name, shape, map, binding.IsOutput, binding.ElementType);
        }

        var inputs = new CodegenTensorBinding[pass.Inputs.Count];
        for (int i = 0; i < inputs.Length; i++)
        {
            var renumbered = Renumber(pass.Inputs[i]);
            if (renumbered is null) { reason = pass.Name + " reads through an indirect index"; return null; }
            inputs[i] = renumbered;
        }
        var output = Renumber(pass.Output);
        if (output is null) { reason = pass.Name + " writes through an indirect index"; return null; }
        var products = new int[pass.ProductInputs.Count];
        for (int i = 0; i < products.Length; i++) products[i] = pass.ProductInputs[i];
        return new CodegenKernelSpec(pass.Name, new CodegenIterationSpace(programAxes), inputs, output, products,
            CodegenReduceKind.Sum, activation: pass.Activation, reduceScale: pass.ReduceScale, preReduce: pass.PreReduce,
            preBiasInput: pass.PreBiasInput, preBiasScale: pass.PreBiasScale, reduceOffset: pass.ReduceOffset,
            body: pass.Body);
    }

    // One value per column, laid out as the flattened column index: the combine writes column c at c.
    private static bool IsContiguousColumnVector(CodegenRowBinding binding, int[] columnExtents)
    {
        foreach (long rowStride in binding.RowStrides) if (rowStride != 0) return false;
        long expected = 1;
        for (int k = columnExtents.Length - 1; k >= 0; k--)
        {
            if (binding.ColumnStrides[k] != expected) return false;
            expected *= columnExtents[k];
        }
        return true;
    }

    /// <summary>The addressing key of a binding: its name and index map.</summary>
    internal static string KeyOf(CodegenTensorBinding binding)
    {
        var sb = new System.Text.StringBuilder(binding.Name).Append('|');
        foreach (var expr in binding.Map)
        {
            sb.Append(expr.Constant).Append('/').Append(expr.Divisor).Append(':');
            foreach (var term in expr.Terms) sb.Append(term.Axis).Append('*').Append(term.Coefficient).Append('+');
            sb.Append(';');
        }
        return sb.ToString();
    }

    private static bool IsAxis(CodegenAffineExpr expr, int axis) =>
        expr.Terms.Count == 1 && expr.Terms[0].Axis == axis && expr.Terms[0].Coefficient == 1 &&
        expr.Constant == 0 && expr.Divisor == 1;
}
