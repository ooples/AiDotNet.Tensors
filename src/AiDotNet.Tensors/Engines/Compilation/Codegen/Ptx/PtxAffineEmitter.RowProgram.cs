// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Single-kernel emission of a row-fused program (CodegenRowFusion).
//
// HELD rows (up to 32 elements per thread) keep the row's sources in registers across every
// pass: each row is one warp, or one block past a warp, and each statistic is a shuffle
// reduction plus a shared-memory combine across the block's warps.
//
// STREAMED rows (a BatchNorm channel: 32 x 3136 elements) cannot be held, so the passes are
// scheduled into SWEEPS over the row. Statistics that do not depend on one another share a
// sweep -- one read of the row feeds all of them -- and a pass that needs a statistic of the
// current sweep starts the next. Row passes (per-row arithmetic on statistics) run between
// sweeps from registers.
//
// RAGGED rows (a width no step divides) overhang the row in their last step: a lane past the
// row loads nothing, contributes the reduction's identity and stores nothing. WRAPPED rows are
// ragged over several column axes -- BatchNorm's (n, h*w) at 7x7 -- where a step crosses runs
// of the innermost axis: each lane carries its (outer, inner) column position, advanced by
// the step's constant increment with one carry, so no step divides.
//
// COLUMN PASSES (sums down the rows, one output per column) make the kernel PERSISTENT: a fixed
// grid of at most ColumnPartialBlockCap blocks loops over the rows, each thread accumulating
// its columns in registers. After the loop a block row writes its partials straight from
// registers; warp rows first sum their warps through shared memory, in warp order. The
// partials are summed in block order by EmitColumnCombine, so the result is deterministic.
//
// It is a partial of PtxAffineEmitter so that every per-element operation -- the body, the
// pre-reduction, the epilogue, the activations -- is the SAME instruction sequence the
// multi-pass kernels emit.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

public sealed partial class PtxAffineEmitter
{
    /// <summary>
    /// Emits <paramref name="plan"/> as one kernel named <paramref name="kernelName"/>, whose
    /// parameters are <see cref="CodegenRowFusionPlan.Parameters"/> in order. Launch with
    /// <see cref="LaunchBlocks"/> blocks of <see cref="LaunchBlockX"/> threads.
    /// </summary>
    public string EmitRowProgram(CodegenRowFusionPlan plan, string kernelName, int computeMajor, int computeMinor)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));
        if (string.IsNullOrWhiteSpace(kernelName)) throw new ArgumentException("Kernel needs a name.", nameof(kernelName));

        const int Lanes = CodegenWarpReduction.Lanes;
        int maxVector = EnableVectorLoads ? 4 : 1;
        var shape = plan.Layout(maxVector)
            ?? throw new NotSupportedException("No thread geometry fits a " + plan.Width + "-wide row.");
        bool isHeld = shape.Held;
        bool ragged = shape.Ragged;
        bool wrapped = shape.Wrapped;
        int rowThreads = shape.Threads;
        int vector = shape.Vector;
        int step = shape.Step;
        int perThread = shape.PerThread;
        int chunks = shape.Chunks;
        // A held row past a warp, and every streamed row, is one block.
        bool blockRow = rowThreads > Lanes || !isHeld;
        int blockThreads = blockRow ? rowThreads : BlockThreads;
        long threads = (long)plan.Rows * rowThreads;
        long blocks = (threads + blockThreads - 1) / blockThreads;
        bool columnFused = plan.ColumnPasses.Count != 0;
        int rowsPerBlock = blockRow ? 1 : blockThreads / Lanes;
        if (columnFused)
        {
            if (!isHeld) throw new NotSupportedException("Column passes need the row held in registers.");
            if (!blockRow && plan.Width > blockThreads)
                throw new NotSupportedException("A warp row with column passes is at most one block of columns wide.");
            blocks = Math.Min((plan.Rows + (long)rowsPerBlock - 1) / rowsPerBlock, ColumnPartialBlockCap);
        }

        _sb.Clear(); _body.Clear(); _calledFunctions.Clear(); _r = _f = _p = _rd = 0;
        EmittedLoads = 0; ElidedGuards = 0; IndirectIndexLoads = 0; AtomicStores = 0;
        ExtraOutputStores = 0; VectorisedStores = 0; VectorisedLoads = 0;
        SharedMemoryBytes = 0; StagedOperands = "none";
        LoopedAxes = isHeld ? 0 : 1;
        // Each thread covers Width / rowThreads elements of its row, which is what every
        // launch invariant divides the iteration space by.
        CoarsenedLanes = plan.Width / rowThreads;
        UsedTwoDimensionalBlock = false; WarpReduced = true; Unrolled = isHeld;
        LaunchBlockX = blockThreads; LaunchBlockY = 1; LaunchBlockThreads = blockThreads;
        LaunchBlocks = checked((uint)blocks);

        _sb.Append(".version ").Append(PtxIsaVersionFor(computeMajor, computeMinor)).Append('\n')
           .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
           .Append(".address_size 64\n\n")
           .Append("// generated by PtxAffineEmitter.EmitRowProgram: ")
           .Append(plan.Program.Description).Append(", ")
           .Append(I(rowThreads)).Append(" threads per row, ").Append(I(vector)).Append("-wide, ")
           .Append(isHeld ? "held" : "streamed").Append(ragged ? ", ragged" : "").Append('\n')
           .Append(".visible .entry ").Append(kernelName).Append("(\n");
        for (int i = 0; i < plan.Parameters.Count; i++)
            _sb.Append("    .param .u64 p").Append(I(i)).Append(i == plan.Parameters.Count - 1 ? "\n" : ",\n");
        // The launch's block size bounds the registers: without it ptxas may allocate more than a
        // 1024-thread block can be given, and the launch fails with LAUNCH_OUT_OF_RESOURCES.
        _sb.Append(")\n.maxntid ").Append(I(blockThreads)).Append(", 1, 1\n{\n");

        var basePtr = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < plan.Parameters.Count; i++)
        {
            string reg = NextRd();
            L($"ld.param.u64 {reg}, [p{I(i)}];");
            basePtr[plan.Parameters[i]] = reg;
        }

        string ctaid = NextR(), tid = NextR(), gid = NextR(), row = NextR(), lanePos = NextR();
        L($"mov.u32 {ctaid}, %ctaid.x;");
        L($"mov.u32 {tid}, %tid.x;");
        L($"mad.lo.u32 {gid}, {ctaid}, {I(blockThreads)}, {tid};");
        // The guard tests the same thread count the grid is built from. A warp row has
        // Rows x 32 threads, a multiple of 32, so it retires whole warps; a block row has
        // exactly one block per row and the guard never fires, so every thread reaches
        // every barrier.
        // A persistent (column-fused) kernel has no early exit: every thread reaches the
        // barriers that combine its warps' partials, rows or none.
        if (!columnFused)
        {
            string guard = NextP();
            L($"setp.ge.u32 {guard}, {gid}, {I(threads)};");
            L($"@{guard} bra END;");
        }
        if (blockRow)
        {
            L($"mov.u32 {row}, {ctaid};");
            L($"mov.u32 {lanePos}, {tid};");
        }
        else
        {
            L($"shr.u32 {row}, {gid}, 5;");
            L($"and.b32 {lanePos}, {gid}, 31;");
        }
        string leader = NextP();
        L($"setp.eq.u32 {leader}, {lanePos}, 0;");

        // Row-axis values, outermost first, from the flat row index. Emitted where the row is
        // known: here, or at the top of a persistent kernel's row loop.
        var rowValue = new string[plan.RowExtents.Count];
        void EmitRowValues()
        {
            string rest = row;
            for (int k = plan.RowExtents.Count - 1; k >= 0; k--)
            {
                if (k == 0) { rowValue[k] = rest; break; }
                string v = NextR(), q = NextR();
                L($"rem.u32 {v}, {rest}, {I(plan.RowExtents[k])};");
                L($"div.u32 {q}, {rest}, {I(plan.RowExtents[k])};");
                rowValue[k] = v;
                rest = q;
            }
        }
        if (!columnFused) EmitRowValues();
        string laneBytes = NextRd();
        L($"mul.wide.u32 {laneBytes}, {lanePos}, {I(4 * vector)};");
        // A ragged row's guard compares this lane's first column of a step with the width.
        string? laneColumn = null;
        if (ragged)
        {
            laneColumn = NextR();
            L($"mul.lo.u32 {laneColumn}, {lanePos}, {I(vector)};");
        }
        // The predicate of the step being emitted when it overhangs the row; null when whole.
        string? stepGuard = null;

        // The reduction's identity in place of an out-of-row lane's term.
        string Guarded(CodegenKernelSpec pass, string term)
        {
            if (stepGuard is null) return term;
            string kept = NextF();
            string identity = pass.Reduce == CodegenReduceKind.Max ? "0fFF800000" : "0f00000000";
            L($"selp.f32 {kept}, {term}, {identity}, {stepGuard};");
            return kept;
        }
        string Store() => stepGuard is null ? "st.global" : $"@{stepGuard} st.global";

        // Per-binding address of this row's first element: base + 4 * rowPart.
        var rowStart = new Dictionary<string, string>(StringComparer.Ordinal);
        string RowStart(CodegenRowBinding b)
        {
            if (rowStart.TryGetValue(b.Key, out string? p)) return p;
            string address = basePtr[b.Name];
            if (b.ConstantOffset != 0)
            {
                string fixedAt = NextRd();
                L($"add.s64 {fixedAt}, {address}, {I64(b.ConstantOffset * 4)};");
                address = fixedAt;
            }
            for (int k = 0; k < b.RowStrides.Count; k++)
            {
                if (b.RowStrides[k] == 0) continue;
                string next = NextRd();
                L($"mad.wide.u32 {next}, {rowValue[k]}, {I64(b.RowStrides[k] * 4)}, {address};");
                address = next;
            }
            return rowStart[b.Key] = address;
        }

        // The row start plus, for a Matrix binding, this thread's position in the step.
        var rowBase = new Dictionary<string, string>(StringComparer.Ordinal);
        string RowBase(CodegenRowBinding b)
        {
            if (rowBase.TryGetValue(b.Key, out string? p)) return p;
            string address = RowStart(b);
            if (b.Kind == CodegenRowOperandKind.Matrix)
            {
                string withLane = NextRd();
                L($"add.s64 {withLane}, {address}, {laneBytes};");
                address = withLane;
            }
            return rowBase[b.Key] = address;
        }

        // ---- WRAPPED positions: (outer, inner) of this lane's column in a step.
        int innerExtent = plan.ColumnExtents[plan.ColumnExtents.Count - 1];
        int outerCount = plan.Width / innerExtent;
        int stepOuter = step / innerExtent, stepInner = step % innerExtent;
        (string Outer, string Inner) firstPosition = ("", "");
        if (wrapped && laneColumn is not null)
        {
            string o = NextR(), i = NextR();
            L($"div.u32 {o}, {laneColumn}, {I(innerExtent)};");
            L($"rem.u32 {i}, {laneColumn}, {I(innerExtent)};");
            firstPosition = (o, i);
        }

        // Advances a position by one step: (o, i) + (stepOuter, stepInner), one carry.
        void AdvanceInto(string fromOuter, string fromInner, string toOuter, string toInner)
        {
            string carry = NextP();
            L($"add.u32 {toInner}, {fromInner}, {I(stepInner)};");
            L($"add.u32 {toOuter}, {fromOuter}, {I(stepOuter)};");
            L($"setp.ge.u32 {carry}, {toInner}, {I(innerExtent)};");
            L($"@{carry} sub.u32 {toInner}, {toInner}, {I(innerExtent)};");
            L($"@{carry} add.u32 {toOuter}, {toOuter}, 1;");
        }

        // A binding's address at a position: row start + outer * outerStride + inner * innerStride.
        string WrappedAddress(CodegenRowBinding b, (string Outer, string Inner) at)
        {
            long outerStride = plan.OuterStride(b)
                ?? throw new InvalidOperationException(b.Name + "'s outer column axes do not flatten.");
            long innerStride = b.ColumnStrides[b.ColumnStrides.Count - 1];
            string address = RowStart(b);
            if (outerStride != 0)
            {
                string next = NextRd();
                L($"mad.wide.u32 {next}, {at.Outer}, {I64(outerStride * 4)}, {address};");
                address = next;
            }
            if (innerStride != 0)
            {
                string next = NextRd();
                L($"mad.wide.u32 {next}, {at.Inner}, {I64(innerStride * 4)}, {address};");
                address = next;
            }
            return address;
        }

        // Compile-time column offset (elements) of step `chunk`: a step never straddles a run
        // of the innermost column axis.
        long ChunkOffset(CodegenRowBinding b, int chunk)
        {
            long column = (long)chunk * step;
            long offset = 0;
            for (int c = plan.ColumnExtents.Count - 1; c >= 0; c--)
            {
                long extent = plan.ColumnExtents[c];
                offset += column % extent * b.ColumnStrides[c];
                column /= extent;
            }
            return offset;
        }

        // Per-row values: statistics this kernel produced (by NAME) and per-row sources (by
        // addressing key, since one tensor can also be read at a fixed position per row).
        var rowStat = new Dictionary<string, string>(StringComparer.Ordinal);
        string RowOperand(CodegenRowBinding b)
        {
            if (rowStat.TryGetValue(b.Name, out string? stat)) return stat;
            if (rowStat.TryGetValue(b.Key, out string? loaded)) return loaded;
            string v = NextF();
            L($"ld.global.nc.f32 {v}, [{RowBase(b)}];");
            EmittedLoads++;
            return rowStat[b.Key] = v;
        }

        // Loads one step of a Matrix binding (vector registers) or a Feature binding's value,
        // at a byte address expression, caching by key.
        string[] LoadStep(string key, string at, CodegenRowOperandKind kind, Dictionary<string, string[]> cache)
        {
            if (cache.TryGetValue(key, out string[]? have)) return have;
            // An overhanging step loads nothing past the row: its registers read zero instead.
            string load = stepGuard is null ? "ld.global.nc" : $"@{stepGuard} ld.global.nc";
            if (kind == CodegenRowOperandKind.Feature || vector == 1)
            {
                string v = NextF();
                if (stepGuard is not null) L($"mov.f32 {v}, 0f00000000;");
                L($"{load}.f32 {v}, {at};");
                EmittedLoads++;
                var one = new string[vector];
                for (int c = 0; c < vector; c++) one[c] = v;
                return cache[key] = one;
            }
            var parts = new string[vector];
            for (int c = 0; c < vector; c++)
            {
                parts[c] = NextF();
                if (stepGuard is not null) L($"mov.f32 {parts[c]}, 0f00000000;");
            }
            L($"{load}.v{I(vector)}.f32 {{{string.Join(", ", parts)}}}, {at};");
            EmittedLoads++;
            VectorisedLoads++;
            return cache[key] = parts;
        }

        // The value of input `input` of `pass` at element e, through `step` (chunk) access.
        // Set by the held or streamed lowering below before any pass is emitted.
        Func<CodegenKernelSpec, int, int, string> operand =
            (_, _, _) => throw new InvalidOperationException("No operand accessor is set for this lowering.");

        string Term(CodegenKernelSpec pass, int e)
        {
            string product;
            if (pass.Body is not null)
            {
                product = EmitBody(pass.Body, input => operand(pass, input, e));
            }
            else
            {
                product = NextF();
                L($"mov.f32 {product}, {operand(pass, pass.ProductInputs[0], e)};");
                for (int k = 1; k < pass.ProductInputs.Count; k++)
                {
                    string next = NextF();
                    L($"mul.rn.f32 {next}, {product}, {operand(pass, pass.ProductInputs[k], e)};");
                    product = next;
                }
                if (pass.PreBiasInput is { } pb)
                {
                    string pv = operand(pass, pb, e), shifted = NextF();
                    if (pass.PreBiasScale == 1.0) L($"add.rn.f32 {shifted}, {product}, {pv};");
                    else L($"fma.rn.f32 {shifted}, {pv}, {F32(pass.PreBiasScale)}, {product};");
                    product = shifted;
                }
            }
            if (pass.PreReduce == CodegenPreReduceOp.Square)
            {
                string sq = NextF();
                L($"mul.rn.f32 {sq}, {product}, {product};");
                return sq;
            }
            if (pass.PreReduce == CodegenPreReduceOp.Exp)
            {
                string scaled = NextF(), ex = NextF();
                L($"mul.rn.f32 {scaled}, {product}, 0f3FB8AA3B;");
                L($"ex2.approx.f32 {ex}, {scaled};");
                return ex;
            }
            return product;
        }

        // The epilogue in the spec's order: scale, offset, bias, scale, post-scale,
        // post-bias, activation.
        string Epilogue(CodegenKernelSpec pass, string acc, int e)
        {
            void Apply(string op, string value)
            {
                string r = NextF();
                L($"{op} {r}, {acc}, {value};");
                acc = r;
            }
            if (pass.ReduceScale != 1.0) Apply("mul.rn.f32", F32(pass.ReduceScale));
            if (pass.ReduceOffset != 0.0) Apply("add.rn.f32", F32(pass.ReduceOffset));
            if (pass.BiasInput is { } bias) Apply("add.rn.f32", operand(pass, bias, e));
            if (pass.ScaleInput is { } scale) Apply("mul.rn.f32", operand(pass, scale, e));
            if (pass.PostScaleInput is { } postScale) Apply("mul.rn.f32", operand(pass, postScale, e));
            if (pass.PostBiasInput is { } postBias) Apply("add.rn.f32", operand(pass, postBias, e));
            return EmitActivation(pass.Activation, acc);
        }

        // A block row combines its warps through shared memory, one slot per warp and one
        // array per statistic, so no barrier is needed to make a slot reusable.
        int warpsPerRow = blockThreads / Lanes;
        string? laneInWarp = null, warpBytes = null;
        if (blockRow && warpsPerRow > 1)
        {
            string warp = NextR();
            laneInWarp = NextR(); warpBytes = NextRd();
            L($"shr.u32 {warp}, {lanePos}, 5;");
            L($"and.b32 {laneInWarp}, {lanePos}, 31;");
            L($"mul.wide.u32 {warpBytes}, {warp}, 4;");
        }
        int statistic = 0;

        // Finishes one statistic: butterfly, block combine, epilogue, leader store.
        void FinishReduction(CodegenKernelSpec pass, string acc)
        {
            string combine = pass.Reduce == CodegenReduceKind.Max ? "max.f32" : "add.rn.f32";
            for (int offset = Lanes / 2; offset >= 1; offset /= 2)
            {
                string bits = NextR(), otherBits = NextR(), other = NextF(), combined = NextF();
                L($"mov.b32 {bits}, {acc};");
                L($"shfl.sync.bfly.b32 {otherBits}, {bits}, {I(offset)}, 31, 0xffffffff;");
                L($"mov.b32 {other}, {otherBits};");
                L($"{combine} {combined}, {acc}, {other};");
                acc = combined;
            }
            if (laneInWarp is not null)
            {
                // Every warp publishes its total; every thread then folds the totals in warp
                // order, so the result is identical in every thread and run to run.
                string slots = "rowred" + I(statistic);
                _sb.Append("    .shared .align 4 .b8 ").Append(slots)
                   .Append('[').Append(I(warpsPerRow * 4)).Append("];\n");
                string slotBase = NextRd(), slot = NextRd(), first = NextP();
                L($"mov.u64 {slotBase}, {slots};");
                L($"add.s64 {slot}, {slotBase}, {warpBytes};");
                L($"setp.eq.u32 {first}, {laneInWarp}, 0;");
                L($"@{first} st.shared.f32 [{slot}], {acc};");
                L("bar.sync 0;");
                string total = NextF();
                L($"ld.shared.f32 {total}, [{slotBase}];");
                for (int w = 1; w < warpsPerRow; w++)
                {
                    string part = NextF(), next = NextF();
                    L($"ld.shared.f32 {part}, [{slotBase}+{I(w * 4)}];");
                    L($"{combine} {next}, {total}, {part};");
                    total = next;
                }
                acc = total;
            }
            statistic++;
            string value = Epilogue(pass, acc, 0);
            rowStat[pass.Output.Name] = value;
            L($"@{leader} st.global.f32 [{RowBase(plan.BindingFor(pass.Output))}], {value};");
        }

        // A row pass: arithmetic on statistics, in every thread, from registers. It reads only
        // per-row operands, so it uses its own accessor: it can run before any sweep has set one
        // (BatchNorm's shift opens the program) and must not see a finished sweep's loop cache.
        void RowPass(CodegenKernelSpec pass)
        {
            var previous = operand;
            operand = (p, input, _) =>
            {
                var b = plan.BindingFor(p.Inputs[input]);
                if (b.Kind != CodegenRowOperandKind.Row)
                    throw new InvalidOperationException(p.Name + " is a row pass reading the non-row '" + b.Name + "'.");
                return RowOperand(b);
            };
            string value = Epilogue(pass, Term(pass, 0), 0);
            operand = previous;
            rowStat[pass.Output.Name] = value;
            L($"@{leader} st.global.f32 [{RowBase(plan.BindingFor(pass.Output))}], {value};");
        }

        // ---- PERSISTENT row loop (column-fused): accumulators live outside it, in fixed registers.
        var columnAccumulators = new Dictionary<string, string[]>(StringComparer.Ordinal);
        if (columnFused)
        {
            foreach (string name in plan.ColumnPasses)
            {
                var acc = new string[perThread];
                for (int e = 0; e < perThread; e++)
                {
                    acc[e] = NextF();
                    L($"mov.f32 {acc[e]}, 0f00000000;");
                }
                columnAccumulators[name] = acc;
            }
            if (!blockRow)
            {
                string warpOfRow = NextR();
                L($"shr.u32 {warpOfRow}, {tid}, 5;");
                L($"mad.lo.u32 {row}, {ctaid}, {I(rowsPerBlock)}, {warpOfRow};");
            }
            string done = NextP();
            _body.Append("ROWLOOP:\n");
            NoUnroll();
            L($"setp.ge.u32 {done}, {row}, {I(plan.Rows)};");
            L($"@{done} bra ROWDONE;");
            EmitRowValues();
        }

        if (isHeld)
        {
            // ---- HELD: every step at a compile-time offset, sources cached for the kernel.
            var cache = new Dictionary<string, string[]>(StringComparer.Ordinal);
            // A wrapped row's per-step positions, in step order (straight-line code, so each
            // dominates every later use).
            var positions = new List<(string Outer, string Inner)>();
            (string Outer, string Inner) PositionOf(int chunk)
            {
                if (positions.Count == 0) positions.Add(firstPosition);
                while (positions.Count <= chunk)
                {
                    var last = positions[positions.Count - 1];
                    string o = NextR(), i = NextR();
                    AdvanceInto(last.Outer, last.Inner, o, i);
                    positions.Add((o, i));
                }
                return positions[chunk];
            }

            // A ragged row's overhanging steps, each with its predicate (computed once).
            var chunkGuard = new Dictionary<int, string>();
            string? GuardOf(int chunk)
            {
                if (laneColumn is null || (long)(chunk + 1) * step <= plan.Width) return null;
                if (chunkGuard.TryGetValue(chunk, out string? have)) return have;
                string inRow = NextP();
                if (wrapped) L($"setp.lt.u32 {inRow}, {PositionOf(chunk).Outer}, {I(outerCount)};");
                else L($"setp.lt.u32 {inRow}, {laneColumn}, {I(plan.Width - chunk * step)};");
                return chunkGuard[chunk] = inRow;
            }

            // The bracketed address of binding b's step `chunk` (Matrix or Feature).
            var chunkAddress = new Dictionary<string, string>(StringComparer.Ordinal);
            string AddressOf(CodegenRowBinding b, int chunk)
            {
                if (!wrapped) return $"[{RowBase(b)}+{I64(ChunkOffset(b, chunk) * 4)}]";
                string key = b.Key + "@" + I(chunk);
                if (!chunkAddress.TryGetValue(key, out string? at))
                    chunkAddress[key] = at = WrappedAddress(b, PositionOf(chunk));
                return $"[{at}]";
            }

            operand = (pass, input, e) =>
            {
                var b = plan.BindingFor(pass.Inputs[input]);
                if (b.Kind == CodegenRowOperandKind.Row) return RowOperand(b);
                int chunk = e / vector;
                string at = AddressOf(b, chunk);
                stepGuard = GuardOf(chunk);
                return LoadStep(b.Key + "@" + I(chunk), at, b.Kind, cache)[e % vector];
            };

            foreach (var pass in plan.Program.Passes)
            {
                if (plan.RowPasses.Contains(pass.Name)) { RowPass(pass); continue; }
                if (columnAccumulators.TryGetValue(pass.Name, out var columnAcc))
                {
                    // A column pass: this row's term for each held column, into its accumulator.
                    for (int e = 0; e < perThread; e++)
                    {
                        string term = Term(pass, e);
                        stepGuard = GuardOf(e / vector);
                        term = Guarded(pass, term);
                        stepGuard = null;
                        L($"add.rn.f32 {columnAcc[e]}, {columnAcc[e]}, {term};");
                    }
                    continue;
                }
                if (pass.Reduce != CodegenReduceKind.None)
                {
                    bool max = pass.Reduce == CodegenReduceKind.Max;
                    string acc = NextF();
                    L(max ? $"mov.f32 {acc}, 0fFF800000;" : $"mov.f32 {acc}, 0f00000000;");
                    for (int e = 0; e < perThread; e++)
                    {
                        string term = Term(pass, e);
                        stepGuard = GuardOf(e / vector);
                        term = Guarded(pass, term);
                        stepGuard = null;
                        string next = NextF();
                        L($"{(max ? "max.f32" : "add.rn.f32")} {next}, {acc}, {term};");
                        acc = next;
                    }
                    FinishReduction(pass, acc);
                }
                else
                {
                    StorePointwise(pass, perThread, chunk => AddressOf(plan.BindingFor(pass.Output), chunk), GuardOf);
                }
            }

            if (columnFused)
            {
                // A block row's warps share one slot per statistic, written each row: the next
                // row must not overwrite a slot a slower warp is still reading.
                if (laneInWarp is not null) L("bar.sync 0;");
                L($"add.u32 {row}, {row}, {I((int)blocks * rowsPerBlock)};");
                L("bra ROWLOOP;");
                _body.Append("ROWDONE:\n");
                EmitColumnPartials();
            }
        }

        else
        {
            // ---- STREAMED: sweeps over the row with a runtime step counter.
            var sweep = new List<CodegenKernelSpec>();
            var sweepOutputs = new HashSet<string>(StringComparer.Ordinal);
            int sweepIndex = 0;
            foreach (var pass in plan.Program.Passes)
            {
                bool rowPass = plan.RowPasses.Contains(pass.Name);
                bool dependsOnSweep = false;
                foreach (var input in pass.Inputs)
                    if (sweepOutputs.Contains(input.Name)) dependsOnSweep = true;
                if (rowPass || dependsOnSweep)
                {
                    // Only a sweep that has passes takes a label number.
                    if (sweep.Count > 0) EmitSweep(sweep, sweepIndex++);
                    sweep.Clear();
                    sweepOutputs.Clear();
                }
                if (rowPass) { RowPass(pass); continue; }
                sweep.Add(pass);
                if (pass.Reduce != CodegenReduceKind.None) sweepOutputs.Add(pass.Output.Name);
            }
            EmitSweep(sweep, sweepIndex);
        }

        // Each column pass's per-block partial: partial[block, column], column the flattened index
        // chunk * step + lane * vector. Guards are recomputed here from lane-only values: a warp
        // whose first row is past the end ran no iteration, so nothing computed in the loop is set.
        void EmitColumnPartials()
        {
            if (!columnFused) return;
            string laneCol = NextR();
            L($"mul.lo.u32 {laneCol}, {lanePos}, {I(vector)};");
            string? Overhang(int chunk)
            {
                if ((long)(chunk + 1) * step <= plan.Width) return null;
                string inRow = NextP();
                L($"setp.lt.u32 {inRow}, {laneCol}, {I(plan.Width - chunk * step)};");
                return inRow;
            }
            void StoreChunk(string space, string address, string[] acc, int chunk, string? inRow)
            {
                string op = (inRow is null ? "" : "@" + inRow + " ") + "st." + space;
                if (vector == 1) { L($"{op}.f32 [{address}], {acc[chunk]};"); return; }
                var parts = new string[vector];
                for (int c = 0; c < vector; c++) parts[c] = acc[chunk * vector + c];
                L($"{op}.v{I(vector)}.f32 [{address}], {{{string.Join(", ", parts)}}};");
            }

            if (blockRow)
            {
                // One row per block at a time: each thread owns its columns outright.
                foreach (string name in plan.ColumnPasses)
                {
                    var acc = columnAccumulators[name];
                    string blockAt = NextRd(), laneAt = NextRd();
                    L($"mad.wide.u32 {blockAt}, {ctaid}, {I64((long)plan.Width * 4)}, {basePtr[CodegenRowFusionPlan.PartialName(OutputOf(name))]};");
                    L($"add.s64 {laneAt}, {blockAt}, {laneBytes};");
                    for (int chunk = 0; chunk < chunks; chunk++)
                    {
                        string at = NextRd();
                        L($"add.s64 {at}, {laneAt}, {I64((long)chunk * step * 4)};");
                        StoreChunk("global", at, acc, chunk, Overhang(chunk));
                    }
                }
                return;
            }

            // Warp rows: every warp holds the same columns for different rows. Sum them through
            // shared memory, warp 0 first, then one thread per column writes the block's partial.
            int slots = rowsPerBlock * plan.Width;
            _sb.Append("    .shared .align 16 .b8 colred[").Append(I(slots * 4)).Append("];\n");
            string warp = NextR(), smem = NextRd(), warpRow = NextRd(), laneSlot = NextRd(), owner = NextP();
            L($"shr.u32 {warp}, {tid}, 5;");
            L($"mov.u64 {smem}, colred;");
            L($"mad.wide.u32 {warpRow}, {warp}, {I64((long)plan.Width * 4)}, {smem};");
            L($"add.s64 {laneSlot}, {warpRow}, {laneBytes};");
            L($"setp.lt.u32 {owner}, {tid}, {I(plan.Width)};");
            string columnBytes = NextRd();
            L($"mul.wide.u32 {columnBytes}, {tid}, 4;");
            int combineIndex = 0;
            foreach (string name in plan.ColumnPasses)
            {
                var acc = columnAccumulators[name];
                for (int chunk = 0; chunk < chunks; chunk++)
                {
                    string at = NextRd();
                    L($"add.s64 {at}, {laneSlot}, {I64((long)chunk * step * 4)};");
                    StoreChunk("shared", at, acc, chunk, Overhang(chunk));
                }
                L("bar.sync 0;");
                string skip = "COLSKIP" + I(combineIndex++);
                string notOwner = NextP();
                L($"not.pred {notOwner}, {owner};");
                L($"@{notOwner} bra {skip};");
                string column = NextRd(), total = NextF();
                L($"add.s64 {column}, {smem}, {columnBytes};");
                L($"ld.shared.f32 {total}, [{column}];");
                for (int w = 1; w < rowsPerBlock; w++)
                {
                    string part = NextF(), next = NextF();
                    L($"ld.shared.f32 {part}, [{column}+{I64((long)w * plan.Width * 4)}];");
                    L($"add.rn.f32 {next}, {total}, {part};");
                    total = next;
                }
                string blockAt = NextRd(), at2 = NextRd();
                L($"mad.wide.u32 {blockAt}, {ctaid}, {I64((long)plan.Width * 4)}, {basePtr[CodegenRowFusionPlan.PartialName(OutputOf(name))]};");
                L($"add.s64 {at2}, {blockAt}, {columnBytes};");
                L($"st.global.f32 [{at2}], {total};");
                _body.Append(skip).Append(":\n");
                L("bar.sync 0;");
            }
        }

        string OutputOf(string passName)
        {
            foreach (var pass in plan.Program.Passes)
                if (string.Equals(pass.Name, passName, StringComparison.Ordinal)) return pass.Output.Name;
            throw new InvalidOperationException("No pass named '" + passName + "'.");
        }

        void EmitSweep(List<CodegenKernelSpec> passes, int index)
        {
            if (passes.Count == 0) return;
            int innermost = plan.ColumnExtents[plan.ColumnExtents.Count - 1];
            // A ragged row has one column axis, so its run is the whole row, last step partial.
            int stepsPerRun = ragged ? chunks : innermost / step;

            // Accumulators live in FIXED registers, updated in place, because a loop cannot
            // carry renamed registers across its back edge.
            var accumulator = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var pass in passes)
            {
                if (pass.Reduce == CodegenReduceKind.None) continue;
                string acc = NextF();
                L(pass.Reduce == CodegenReduceKind.Max ? $"mov.f32 {acc}, 0fFF800000;" : $"mov.f32 {acc}, 0f00000000;");
                accumulator[pass.Name] = acc;
            }

            // UNROLLED: several steps per trip, so each thread has several loads in flight. One
            // step per trip left a BatchNorm channel at 386 us against a 300 us floor.
            int unroll = 1;
            foreach (int u in new[] { 8, 4, 2 })
                if (chunks % u == 0) { unroll = u; break; }

            string counter = NextR();
            L($"mov.u32 {counter}, 0;");
            // A wrapped sweep carries each lane's position in FIXED registers across the loop.
            string sweepOuter = "", sweepInner = "";
            if (wrapped)
            {
                sweepOuter = NextR();
                sweepInner = NextR();
                L($"mov.u32 {sweepOuter}, {firstPosition.Outer};");
                L($"mov.u32 {sweepInner}, {firstPosition.Inner};");
            }
            string label = "SWEEP" + I(index);
            _body.Append(label).Append(":\n");
            NoUnroll();

            for (int u = 0; u < unroll; u++)
            {
                string stepIndex = counter;
                if (u > 0)
                {
                    stepIndex = NextR();
                    L($"add.u32 {stepIndex}, {counter}, {I(u)};");
                }

                // Column offset of this step, per binding, at run time: the outer column values
                // come from (step / stepsPerRun), the innermost from the remainder. A single column
                // axis has no outer value, and the division is skipped: next to a ragged row's
                // predicated loads, a div/rem by 196 made ptxas spill 2.7 KB a thread (16x50001).
                var outerValue = new string[wrapped ? 0 : plan.ColumnExtents.Count - 1];
                string outer = stepIndex, within = stepIndex;
                if (outerValue.Length > 0)
                {
                    outer = NextR();
                    within = NextR();
                    L($"div.u32 {outer}, {stepIndex}, {I(stepsPerRun)};");
                    L($"rem.u32 {within}, {stepIndex}, {I(stepsPerRun)};");
                }
                string outerRest = outer;
                for (int k = outerValue.Length - 1; k >= 0; k--)
                {
                    if (k == 0) { outerValue[k] = outerRest; break; }
                    string v = NextR(), q = NextR();
                    L($"rem.u32 {v}, {outerRest}, {I(plan.ColumnExtents[k])};");
                    L($"div.u32 {q}, {outerRest}, {I(plan.ColumnExtents[k])};");
                    outerValue[k] = v;
                    outerRest = q;
                }
                string innerColumn = NextR();
                L($"mul.lo.u32 {innerColumn}, {within}, {I(step)};");
                // Every step of a ragged sweep is guarded: which one overhangs is a run-time fact.
                string? sweepGuard = null;
                if (wrapped)
                {
                    sweepGuard = NextP();
                    L($"setp.lt.u32 {sweepGuard}, {sweepOuter}, {I(outerCount)};");
                }
                else if (laneColumn is not null)
                {
                    string column = NextR();
                    sweepGuard = NextP();
                    L($"add.u32 {column}, {innerColumn}, {laneColumn};");
                    L($"setp.lt.u32 {sweepGuard}, {column}, {I(plan.Width)};");
                }

                var address = new Dictionary<string, string>(StringComparer.Ordinal);
                string StepAddress(CodegenRowBinding b)
                {
                    if (address.TryGetValue(b.Key, out string? a)) return a;
                    if (wrapped) return address[b.Key] = WrappedAddress(b, (sweepOuter, sweepInner));
                    string at = RowBase(b);
                    for (int k = 0; k < outerValue.Length; k++)
                    {
                        if (b.ColumnStrides[k] == 0) continue;
                        string next = NextRd();
                        L($"mad.wide.u32 {next}, {outerValue[k]}, {I64(b.ColumnStrides[k] * 4)}, {at};");
                        at = next;
                    }
                    if (b.ColumnStrides[b.ColumnStrides.Count - 1] != 0)
                    {
                        string next = NextRd();
                        L($"mad.wide.u32 {next}, {innerColumn}, 4, {at};");
                        at = next;
                    }
                    return address[b.Key] = at;
                }

                var cache = new Dictionary<string, string[]>(StringComparer.Ordinal);
                operand = (pass, input, e) =>
                {
                    var b = plan.BindingFor(pass.Inputs[input]);
                    if (b.Kind == CodegenRowOperandKind.Row) return RowOperand(b);
                    stepGuard = sweepGuard;
                    return LoadStep(b.Key, $"[{StepAddress(b)}]", b.Kind, cache)[e % vector];
                };

                foreach (var pass in passes)
                {
                    if (pass.Reduce != CodegenReduceKind.None)
                    {
                        string acc = accumulator[pass.Name];
                        string combine = pass.Reduce == CodegenReduceKind.Max ? "max.f32" : "add.rn.f32";
                        for (int e = 0; e < vector; e++)
                        {
                            string term = Term(pass, e);
                            stepGuard = sweepGuard;
                            L($"{combine} {acc}, {acc}, {Guarded(pass, term)};");
                        }
                    }
                    else
                    {
                        var values = new string[vector];
                        for (int e = 0; e < vector; e++) values[e] = Epilogue(pass, Term(pass, e), e);
                        string at = StepAddress(plan.BindingFor(pass.Output));
                        stepGuard = sweepGuard;
                        if (vector == 1) L($"{Store()}.f32 [{at}], {values[0]};");
                        else
                        {
                            L($"{Store()}.v{I(vector)}.f32 [{at}], {{{string.Join(", ", values)}}};");
                            VectorisedStores++;
                        }
                    }
                }
                stepGuard = null;
                if (wrapped) AdvanceInto(sweepOuter, sweepInner, sweepOuter, sweepInner);
            }

            string more = NextP();
            L($"add.u32 {counter}, {counter}, {I(unroll)};");
            L($"setp.lt.u32 {more}, {counter}, {I(chunks)};");
            L($"@{more} bra {label};");

            foreach (var pass in passes)
                if (pass.Reduce != CodegenReduceKind.None) FinishReduction(pass, accumulator[pass.Name]);
        }

        void StorePointwise(CodegenKernelSpec pass, int count, Func<int, string> addressOf, Func<int, string?> guardOf)
        {
            var pending = new string[vector];
            for (int e = 0; e < count; e++)
            {
                string value = Epilogue(pass, Term(pass, e), e);
                pending[e % vector] = value;
                if (e % vector != vector - 1) continue;
                string at = addressOf(e / vector);
                stepGuard = guardOf(e / vector);
                string store = Store();
                stepGuard = null;
                if (vector == 1)
                {
                    L($"{store}.f32 {at}, {value};");
                    continue;
                }
                L($"{store}.v{I(vector)}.f32 {at}, {{{string.Join(", ", pending)}}};");
                VectorisedStores++;
            }
        }

        _body.Append("END:\n    ret;\n}\n");
        return Assemble();
    }

    /// <summary>
    /// Most blocks a column-fused row kernel launches: each writes one partial per column, and
    /// the combine reads all of them per column, so the grid is capped and the blocks loop.
    /// </summary>
    internal const int ColumnPartialBlockCap = 256;

    /// <summary>
    /// The combine kernel of a column-fused <paramref name="plan"/>. A block covers 32 consecutive
    /// columns with 8 warps: warp g sums partials g, g + 8, ... of its 32 columns -- each warp load
    /// one contiguous line -- and the 8 warp sums are added through shared memory, warp 0 first.
    /// Fixed order throughout, so the result is the same every run. Both simpler shapes lost: a
    /// thread per column was latency-bound at 64 columns (64 threads walking 256 partials), a warp
    /// per column uncoalesced at 4000 (its 32 lanes read 32 partial rows). Parameters: every
    /// partial buffer, then every output, in <see cref="CodegenRowFusionPlan.ColumnPasses"/> order.
    /// </summary>
    public string EmitColumnCombine(CodegenRowFusionPlan plan, string kernelName, int partialBlocks,
        int computeMajor, int computeMinor)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));
        if (plan.ColumnPasses.Count == 0) throw new ArgumentException("The plan has no column passes.", nameof(plan));
        if (partialBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(partialBlocks));
        const int Lanes = CodegenWarpReduction.Lanes;
        var passes = new List<CodegenKernelSpec>();
        foreach (string name in plan.ColumnPasses)
            foreach (var pass in plan.Program.Passes)
                if (string.Equals(pass.Name, name, StringComparison.Ordinal)) passes.Add(pass);

        _sb.Clear(); _body.Clear(); _calledFunctions.Clear(); _r = _f = _p = _rd = 0;
        int threads = BlockThreads, groups = threads / Lanes;
        LaunchBlockX = threads; LaunchBlockY = 1; LaunchBlockThreads = threads;
        LaunchBlocks = checked((uint)((plan.Width + Lanes - 1) / Lanes));
        _sb.Append(".version ").Append(PtxIsaVersionFor(computeMajor, computeMinor)).Append('\n')
           .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
           .Append(".address_size 64\n\n")
           .Append("// generated by PtxAffineEmitter.EmitColumnCombine: ").Append(plan.Program.Description)
           .Append(", ").Append(I(partialBlocks)).Append(" partials per column, 32 columns x 8 warps a block\n")
           .Append(".visible .entry ").Append(kernelName).Append("(\n");
        int parameters = passes.Count * 2;
        for (int i = 0; i < parameters; i++)
            _sb.Append("    .param .u64 p").Append(I(i)).Append(i == parameters - 1 ? "\n" : ",\n");
        _sb.Append(")\n.maxntid ").Append(I(threads)).Append(", 1, 1\n{\n");
        _sb.Append("    .shared .align 4 .b8 combine[").Append(I(threads * 4)).Append("];\n");

        string ctaid = NextR(), tid = NextR(), group = NextR(), lane = NextR(), column = NextR();
        L($"mov.u32 {ctaid}, %ctaid.x;");
        L($"mov.u32 {tid}, %tid.x;");
        L($"shr.u32 {group}, {tid}, 5;");
        L($"and.b32 {lane}, {tid}, 31;");
        L($"mad.lo.u32 {column}, {ctaid}, {I(Lanes)}, {lane};");
        // Columns past the width load and store nothing, but every thread reaches the barriers.
        string inWidth = NextP(), first = NextR(), firstBytes = NextRd(), columnBytes = NextRd();
        L($"setp.lt.u32 {inWidth}, {column}, {I(plan.Width)};");
        L($"mad.lo.u32 {first}, {group}, {I(plan.Width)}, {column};");
        L($"mul.wide.u32 {firstBytes}, {first}, 4;");
        L($"mul.wide.u32 {columnBytes}, {column}, 4;");
        string slot = NextRd(), smem = NextRd(), laneSlot = NextRd(), leaderGroup = NextP(), writer = NextP();
        L($"mov.u64 {smem}, combine;");
        L($"mul.wide.u32 {slot}, {tid}, 4;");
        L($"add.s64 {slot}, {smem}, {slot};");
        L($"mul.wide.u32 {laneSlot}, {lane}, 4;");
        L($"add.s64 {laneSlot}, {smem}, {laneSlot};");
        L($"setp.eq.u32 {leaderGroup}, {group}, 0;");
        L($"and.pred {writer}, {leaderGroup}, {inWidth};");
        // Partial b = group + groups * k; the last round may run past the final block.
        int rounds = (partialBlocks + groups - 1) / groups;
        var load = new string[rounds];
        for (int k = 0; k < rounds; k++)
        {
            if ((long)(k + 1) * groups <= partialBlocks) { load[k] = inWidth; continue; }
            string inRange = NextP(), both = NextP();
            L($"setp.lt.u32 {inRange}, {group}, {I(partialBlocks - k * groups)};");
            L($"and.pred {both}, {inRange}, {inWidth};");
            load[k] = both;
        }
        for (int p = 0; p < passes.Count; p++)
        {
            var pass = passes[p];
            string partials = NextRd(), output = NextRd(), at = NextRd();
            L($"ld.param.u64 {partials}, [p{I(p)}];");
            L($"ld.param.u64 {output}, [p{I(passes.Count + p)}];");
            L($"add.s64 {at}, {partials}, {firstBytes};");
            string acc = NextF();
            L($"mov.f32 {acc}, 0f00000000;");
            for (int k = 0; k < rounds; k++)
            {
                string part = NextF(), next = NextF();
                L($"mov.f32 {part}, 0f00000000;");
                L($"@{load[k]} ld.global.f32 {part}, [{at}+{I64((long)k * groups * plan.Width * 4)}];");
                L($"add.rn.f32 {next}, {acc}, {part};");
                acc = next;
            }
            L($"st.shared.f32 [{slot}], {acc};");
            L("bar.sync 0;");
            string total = NextF();
            L($"ld.shared.f32 {total}, [{laneSlot}];");
            for (int g = 1; g < groups; g++)
            {
                string part = NextF(), next = NextF();
                L($"ld.shared.f32 {part}, [{laneSlot}+{I(g * Lanes * 4)}];");
                L($"add.rn.f32 {next}, {total}, {part};");
                total = next;
            }
            string value = total;
            if (pass.ReduceScale != 1.0) { string r = NextF(); L($"mul.rn.f32 {r}, {value}, {F32(pass.ReduceScale)};"); value = r; }
            if (pass.ReduceOffset != 0.0) { string r = NextF(); L($"add.rn.f32 {r}, {value}, {F32(pass.ReduceOffset)};"); value = r; }
            value = EmitActivation(pass.Activation, value);
            string store = NextRd();
            L($"add.s64 {store}, {output}, {columnBytes};");
            L($"@{writer} st.global.f32 [{store}], {value};");
            // The next pass reuses the slots.
            L("bar.sync 0;");
        }
        _body.Append("    ret;\n}\n");
        return Assemble();
    }

    // The emitter unrolls its own loops as far as it means to; ptxas, left to itself, fully unrolls a
    // constant-trip loop (a 256-partial combine became 512 loads of straight-line code).
    private void NoUnroll() => L(".pragma \"nounroll\";");

    private static string I64(long value) => value.ToString(System.Globalization.CultureInfo.InvariantCulture);
}
