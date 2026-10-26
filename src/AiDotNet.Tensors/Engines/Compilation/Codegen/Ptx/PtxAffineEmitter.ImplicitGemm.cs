// Copyright (c) AiDotNet. All rights reserved.
// A tiled implicit-GEMM lowering for any two-operand sum reduction (#890 phase 4b).
//
// The specialised tiled convolution covers stride 1, dilation 1, padding 1 only, and the tiled
// contraction a pointwise stream; everything else -- a stride-2 downsample, a dilated 3x3, the
// exact-division windows of a strided convolution's data gradient -- fell back to one thread per
// output, 0.06-0.44x cuDNN on a GTX 1660 Ti. This lowering needs nothing from the index maps
// beyond what the spec already states:
//
//   M  the parallel axes only operand A reads (a convolution's output channels)
//   N  the parallel axes only operand B reads (its batch and output positions)
//   B  the parallel axes both read (a group), one grid slice each
//   R  every reduction axis, flattened (channels x taps)
//
// A block computes a TM x TN output tile. Each step it GATHERS a KC x TM slab of A and a KC x TN
// slab of B into shared memory through the operands' own maps -- windows, strides, dilations
// and exact-division guards included, through the same EmitLoad every lowering uses -- and each
// thread accumulates a register block of (TM/16) x (TN/16) outputs. Staging is synchronous, so
// it runs on Turing and later.
//
// An operand whose staged-fastest role is one axis running along its unit-stride dimension (a plain
// matrix: a GEMM's A along K or B along N) is gathered four floats at a time with ld.global.v4 --
// the bound buffers must then be 16-byte aligned, which the backend checks before every launch.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

public sealed partial class PtxAffineEmitter
{
    /// <summary>
    /// Lowers an eligible spec as a tiled implicit GEMM (see the file header). Off by default;
    /// the autotuner measures it as the <c>implicit-gemm</c> candidates.
    /// </summary>
    public bool ImplicitGemm { get; set; }

    /// <summary>Output tile edge of the implicit GEMM: 32 or 64 (the default).</summary>
    public int ImplicitGemmTile { get; set; } = 64;

    /// <summary>The M edge of the output tile when it differs from <see cref="ImplicitGemmTile"/> (the N edge):
    /// 128 gives each thread an 8 x 4 micro-tile, twice the FMAs per shared read; 96 (6 x 4) trades some of that for
    /// a block count that fills the last wave. 0 means square.</summary>
    public int ImplicitGemmTileM { get; set; }

    /// <summary>The blocks a multiprocessor must fit (<c>.minnctapersm</c>), which caps registers at 128 a thread for
    /// two; 0 leaves the JIT unbounded, for a shape whose guards need more than the bound and would spill.</summary>
    public int ImplicitGemmMinBlocksPerMultiprocessor { get; set; } = 2;

    private const int ImplicitGemmThreads = 256;
    private const int ImplicitGemmSide = 16;   // threads per tile edge
    private const int ImplicitGemmChunk = 16;  // reduction terms staged per step
    // Threads that share one reduction row when an operand stages position-fastest.
    private const int PositionLanes = ImplicitGemmThreads / ImplicitGemmChunk;

    // Floats of padding after each staged row (rows are multiples of 32 floats): a multiple of 4 keeps rows
    // 16-byte aligned for the vector reads. Reduction-fastest stores walk 16 rows, which a pad of 4 spreads
    // over the banks; position-fastest stores write two rows of 16, which a pad of 16 puts in opposite halves.
    private static int StagingPad(bool reductionFastest) => reductionFastest ? 4 : 16;

    /// <summary>The axis roles of an implicit-GEMM lowering, or null with the reason it does not apply.</summary>
    internal static (int A, int B, int[] M, int[] N, int[] Batch, int[] R)? PlanImplicitGemm(
        CodegenKernelSpec spec, out string reason)
    {
        reason = "";
        if (spec.Algebra != CodegenAlgebra.Real || spec.Reduce != CodegenReduceKind.Sum)
        { reason = "an implicit GEMM is a real sum reduction"; return null; }
        if (spec.ProductInputs.Count != 2 || spec.HasExpressionBody || spec.PreBiasInput.HasValue ||
            spec.PreReduce != CodegenPreReduceOp.None)
        { reason = "an implicit GEMM multiplies exactly two untransformed operands"; return null; }
        if (spec.ExtraOutputs.Count != 0 || spec.SecondaryOutput is not null || spec.HasExtendedEpilogue)
        { reason = "an implicit GEMM writes one output with at most a bias, scale and activation"; return null; }
        if (spec.Output.HasIndirection || spec.Output.ElementType != CodegenElementType.Float32)
        { reason = "an implicit GEMM stores fp32 to a direct address"; return null; }
        foreach (var input in spec.Inputs)
            if (input.HasIndirection || input.ElementType != CodegenElementType.Float32)
            { reason = "an implicit GEMM stages fp32 operands at direct addresses"; return null; }

        int a = spec.ProductInputs[0], b = spec.ProductInputs[1];
        var m = new List<int>(); var n = new List<int>(); var batch = new List<int>();
        foreach (int p in spec.Space.ParallelAxes)
        {
            bool inA = ReferencesAxis(spec.Inputs[a], p), inB = ReferencesAxis(spec.Inputs[b], p);
            if (inA && inB) batch.Add(p);
            else if (inA) m.Add(p);
            else if (inB) n.Add(p);
            else { reason = "parallel axis " + p + " is read by neither operand"; return null; }
        }
        if (m.Count == 0 || n.Count == 0)
        { reason = "an implicit GEMM needs output axes private to each operand"; return null; }
        return (a, b, m.ToArray(), n.ToArray(), batch.ToArray(), spec.Space.ReductionAxes);
    }

    private string EmitImplicitGemm(CodegenKernelSpec spec, int computeMajor, int computeMinor,
        (int A, int B, int[] M, int[] N, int[] Batch, int[] R) roles)
    {
        var axes = spec.Space.Axes;
        long Extent(int[] set) { long e = 1; foreach (int x in set) e *= axes[x].Extent; return e; }
        long mTotal = Extent(roles.M), nTotal = Extent(roles.N), bTotal = Extent(roles.Batch), rTotal = Extent(roles.R);
        if (mTotal > int.MaxValue || nTotal > int.MaxValue || rTotal > int.MaxValue)
            throw new NotSupportedException("An implicit GEMM indexes each role in 32 bits.");
        int tileN = ImplicitGemmTile == 32 ? 32 : ImplicitGemmTile == 128 ? 128 : 64;
        int tileM = ImplicitGemmTileM is 32 or 64 or 96 or 128 ? ImplicitGemmTileM : tileN;
        // Outputs per thread along each edge, contiguous so vector loads read them.
        int perM = tileM / ImplicitGemmSide, perN = tileN / ImplicitGemmSide;
        bool aReductionFastest = StagesReductionFastest(spec, spec.Inputs[roles.A], roles.M, roles.R, tileM);
        bool bReductionFastest = StagesReductionFastest(spec, spec.Inputs[roles.B], roles.N, roles.R, tileN);
        bool aVector = StagesVectors(spec, spec.Inputs[roles.A], roles.M, roles.R, aReductionFastest, tileM);
        bool bVector = StagesVectors(spec, spec.Inputs[roles.B], roles.N, roles.R, bReductionFastest, tileN);
        int rowFloatsA = tileM + StagingPad(aReductionFastest), rowFloatsB = tileN + StagingPad(bReductionFastest);
        int bufferBytesA = ImplicitGemmChunk * rowFloatsA * sizeof(float);   // one operand's chunk
        int bufferBytesB = ImplicitGemmChunk * rowFloatsB * sizeof(float);
        int pairBytes = bufferBytesA + bufferBytesB;
        long mBlocks = (mTotal + tileM - 1) / tileM, nBlocks = (nTotal + tileN - 1) / tileN;
        long blocks = mBlocks * nBlocks * bTotal;
        if (blocks > int.MaxValue) throw new NotSupportedException("Too many tiles for one grid.");
        _sb.Clear(); _body.Clear(); _calledFunctions.Clear(); _r = _f = _p = _rd = 0;
        EmittedLoads = 0; ElidedGuards = 0; IndirectIndexLoads = 0; AtomicStores = 0;
        ExtraOutputStores = 0; VectorisedStores = 0; VectorisedLoads = 0;
        _inputBindings = spec.Inputs;
        axesMeta = axes;   // the guard interval analysis reads every axis extent
        _outOfRangeFill = "0f00000000";
        _stageLabel = 0;
        StagedOperands = spec.Inputs[roles.A].Name + "+" + spec.Inputs[roles.B].Name;
        SharedMemoryBytes = 2 * pairBytes;   // both operands, double buffered
        CoarsenedLanes = perM * perN;
        LoopedAxes = 1; Unrolled = false; WarpReduced = false; UsedTwoDimensionalBlock = false;
        LaunchBlockX = ImplicitGemmThreads; LaunchBlockY = 1; LaunchBlockThreads = ImplicitGemmThreads;
        LaunchBlocks = (uint)blocks;
        TileDescription = "implicit-gemm " + tileM + "x" + tileN + "x" + ImplicitGemmChunk + " double-buffered, staged " +
            (aReductionFastest ? "r" : "m") + (bReductionFastest ? "r" : "n") + "-fastest" +
            (aVector || bVector ? ", v4 gathers " + (aVector ? "A" : "") + (bVector ? "B" : "") : "");
        _sb.Append(".version ").Append(PtxIsaVersionFor(computeMajor, computeMinor)).Append('\n')
           .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
           .Append(".address_size 64\n\n")
           .Append("// generated by PtxAffineEmitter.EmitImplicitGemm: ").Append(TileDescription).Append('\n')
           .Append("// ").Append(spec.Describe().Replace("\n", "\n// ")).Append('\n')
           .Append(".visible .entry ").Append(spec.Name).Append("(\n");
        for (int i = 0; i < spec.ParameterCount; i++)
            _sb.Append("    .param .u64 p").Append(I(i)).Append(i == spec.ParameterCount - 1 ? "\n" : ",\n");
        // Two blocks a SM by default: one block of 256 threads cannot hide a step's barrier, and without the bound the
        // JIT gives the 8 x 4 micro-tile more than the 128 registers that two blocks leave each thread.
        _sb.Append(")\n.maxntid ").Append(I(ImplicitGemmThreads)).Append(", 1, 1\n");
        if (ImplicitGemmMinBlocksPerMultiprocessor > 0)
            _sb.Append(".minnctapersm ").Append(I(ImplicitGemmMinBlocksPerMultiprocessor)).Append('\n');
        _sb.Append("{\n");
        var basePtr = new string[spec.ParameterCount];
        for (int i = 0; i < basePtr.Length; i++)
        {
            basePtr[i] = NextRd();
            L($"ld.param.u64 {basePtr[i]}, [p{I(i)}];");
        }
        _basePointers = basePtr;
        string tid = NextR(), ctaid = NextR(), tx = NextR(), ty = NextR();
        L($"mov.u32 {tid}, %tid.x;");
        L($"mov.u32 {ctaid}, %ctaid.x;");
        L($"rem.u32 {tx}, {tid}, {I(ImplicitGemmSide)};");
        L($"div.u32 {ty}, {tid}, {I(ImplicitGemmSide)};");
        // Block -> (batch slice, M tile, N tile), N tile fastest.
        string nTile = NextR(), rest = NextR(), mTile = NextR(), bIndex = NextR();
        L($"rem.u32 {nTile}, {ctaid}, {I(nBlocks)};");
        L($"div.u32 {rest}, {ctaid}, {I(nBlocks)};");
        L($"rem.u32 {mTile}, {rest}, {I(mBlocks)};");
        L($"div.u32 {bIndex}, {rest}, {I(mBlocks)};");
        string m0 = NextR(), n0 = NextR();
        L($"mul.lo.u32 {m0}, {mTile}, {I(tileM)};");
        L($"mul.lo.u32 {n0}, {nTile}, {I(tileN)};");
        // One register per axis; filled per use by decomposing a flat index, last axis fastest.
        var axisReg = new string[axes.Count];
        void Decompose(int[] set, string flat)
        {
            string remaining = flat;
            for (int k = set.Length - 1; k >= 0; k--)
            {
                string v = NextR();
                if (k == 0) { L($"mov.u32 {v}, {remaining};"); axisReg[set[k]] = v; break; }
                string q = NextR();
                L($"rem.u32 {v}, {remaining}, {I(axes[set[k]].Extent)};");
                L($"div.u32 {q}, {remaining}, {I(axes[set[k]].Extent)};");
                axisReg[set[k]] = v;
                remaining = q;
            }
        }
        if (roles.Batch.Length > 0) Decompose(roles.Batch, bIndex);
        // Shared layout: [A0][B0][A1][B1], each chunk x rowFloats; a step reads one pair and fills the other.
        string sharedBase = NextRd(), current = NextRd();
        L($"mov.u64 {sharedBase}, stageBuf;");
        L($"mov.u64 {current}, 0;");
        var acc = new string[perM, perN];
        for (int i = 0; i < perM; i++)
            for (int j = 0; j < perN; j++)
            {
                acc[i, j] = NextF();
                L($"mov.f32 {acc[i, j]}, 0f00000000;");
            }
        // ---- staging, split so the gather for the next step is in flight while this step computes.
        // Everything that depends only on the thread -- its slots, its own-axis coordinates -- is
        // decomposed once here; a step decomposes only the reduction index.
        ImplicitGemmStage Plan(int input, int[] own, long ownTotal, string origin, int operandOffset, bool reductionFastest,
            int tile, int rowFloats, bool vector)
        {
            // Scalar: one float per slot. Vector: four consecutive floats of the fastest role per slot, so a thread
            // stages a quarter as many slots (tile is a multiple of 64, so the count is whole).
            int perThread = ImplicitGemmChunk * tile / ImplicitGemmThreads / (vector ? VectorWidth : 1);
            var plan = new ImplicitGemmStage(input, own, reductionFastest, operandOffset, perThread, vector);
            for (int e = 0; e < perThread; e++)
            {
                string lin = NextR(), kk = NextR(), pos = NextR();
                L($"add.u32 {lin}, {tid}, {I(e * ImplicitGemmThreads)};");
                if (vector && reductionFastest)
                {
                    // Four threads cover a position's 16 reduction terms, four each: kk = 4 (t % 4), pos = t / 4 + 64 e.
                    L($"rem.u32 {kk}, {tid}, {I(ImplicitGemmChunk / VectorWidth)};");
                    L($"shl.b32 {kk}, {kk}, 2;");
                    L($"div.u32 {pos}, {lin}, {I(ImplicitGemmChunk / VectorWidth)};");
                }
                else if (vector)
                {
                    // Sixteen threads cover a reduction row, four positions each: kk = t / 16, pos = 4 (t % 16) + 64 e.
                    L($"rem.u32 {pos}, {tid}, {I(PositionLanes)};");
                    L($"shl.b32 {pos}, {pos}, 2;");
                    if (e > 0) L($"add.u32 {pos}, {pos}, {I(e * PositionLanes * VectorWidth)};");
                    L($"div.u32 {kk}, {tid}, {I(PositionLanes)};");
                }
                else if (reductionFastest)
                {
                    L($"rem.u32 {kk}, {lin}, {I(ImplicitGemmChunk)};");
                    L($"div.u32 {pos}, {lin}, {I(ImplicitGemmChunk)};");
                }
                else
                {
                    // Position-fastest, but a thread keeps ONE reduction row for all its elements, so a step
                    // decomposes the reduction index once: thread t stages row t / P, positions t % P + P e.
                    L($"rem.u32 {pos}, {tid}, {I(PositionLanes)};");
                    if (e > 0) L($"add.u32 {pos}, {pos}, {I(e * PositionLanes)};");
                    L($"div.u32 {kk}, {tid}, {I(PositionLanes)};");
                }
                string g = NextR();
                L($"add.u32 {g}, {origin}, {pos};");
                string gc = g;
                if (ownTotal % tile != 0)
                {
                    // Clamp so the gathered address stays inside the operand; the value is zeroed below. A vector of
                    // positions starts on a multiple of 4 of an extent that is one, so it is wholly in or out.
                    string inG = NextP();
                    L($"setp.lt.u32 {inG}, {g}, {I(ownTotal)};");
                    plan.OwnInRange[e] = inG;
                    gc = NextR();
                    L($"min.u32 {gc}, {g}, {I(ownTotal - (vector && !reductionFastest ? VectorWidth : 1))};");
                }
                Decompose(own, gc);
                var ownAxes = new string[own.Length];
                for (int k = 0; k < own.Length; k++) ownAxes[k] = axisReg[own[k]];
                plan.OwnAxes[e] = ownAxes;
                string slot = NextR();
                L($"mad.lo.u32 {slot}, {kk}, {I(rowFloats)}, {pos};");
                L($"shl.b32 {slot}, {slot}, 2;");
                plan.Slot[e] = slot;
                plan.Chunk[e] = kk;
            }
            return plan;
        }
        var stageA = Plan(roles.A, roles.M, mTotal, m0, 0, aReductionFastest, tileM, rowFloatsA, aVector);
        var stageB = Plan(roles.B, roles.N, nTotal, n0, bufferBytesA, bReductionFastest, tileN, rowFloatsB, bVector);
        // Gathers one chunk starting at reduction index r0 into registers: a slot's one float, or a vector slot's four.
        string[] Gather(ImplicitGemmStage plan, string r0)
        {
            var binding = spec.Inputs[plan.Input];
            int perThread = plan.Slot.Length, width = plan.Vector ? VectorWidth : 1;
            var values = new string[perThread * width];
            // A vector of reduction terms starts on a multiple of 4 of an extent that is one: wholly in or out.
            long lastStart = rTotal - (plan.Vector && plan.ReductionFastest ? VectorWidth : 1);
            // Both staging orders give every element of a thread the same reduction index (the thread count is
            // a multiple of the chunk), so it is decomposed once.
            const bool sharedChunk = ImplicitGemmThreads % ImplicitGemmChunk == 0;
            string? inR = null;
            for (int e = 0; e < perThread; e++)
            {
                if (e == 0 || !sharedChunk)
                {
                    string r = NextR();
                    L($"add.u32 {r}, {r0}, {plan.Chunk[e]};");
                    string rc = r;
                    inR = null;
                    if (rTotal % ImplicitGemmChunk != 0)
                    {
                        inR = NextP();
                        L($"setp.lt.u32 {inR}, {r}, {I(rTotal)};");
                        rc = NextR();
                        L($"min.u32 {rc}, {r}, {I(lastStart)};");
                    }
                    Decompose(roles.R, rc);
                }
                for (int k = 0; k < plan.Own.Length; k++) axisReg[plan.Own[k]] = plan.OwnAxes[e][k];
                string[] loaded = plan.Vector
                    ? EmitVectorLoad(binding, basePtr[binding.ParameterIndex], axisReg, Array.Empty<int>(), Array.Empty<int>(),
                        VectorWidth, out _)
                    : new[]
                    {
                        EmitLoad(binding, basePtr[binding.ParameterIndex], axisReg, Array.Empty<int>(), Array.Empty<int>(),
                            out _),
                    };
                string? ok = plan.OwnInRange[e];
                if (inR is not null)
                {
                    if (ok is null) ok = inR;
                    else { string both = NextP(); L($"and.pred {both}, {ok}, {inR};"); ok = both; }
                }
                for (int j = 0; j < width; j++)
                {
                    string value = loaded[j];
                    if (ok is not null)
                    {
                        string kept = NextF();
                        L($"selp.f32 {kept}, {value}, 0f00000000, {ok};");
                        value = kept;
                    }
                    values[e * width + j] = value;
                }
            }
            return values;
        }
        void Store(ImplicitGemmStage plan, string[] values, string bufferOffset)
        {
            string operandBase = NextRd();
            L($"add.u64 {operandBase}, {sharedBase}, {bufferOffset};");
            if (plan.OperandOffset != 0) L($"add.u64 {operandBase}, {operandBase}, {I(plan.OperandOffset)};");
            int rowBytes = (plan.Input == roles.A ? rowFloatsA : rowFloatsB) * sizeof(float);
            for (int e = 0; e < plan.Slot.Length; e++)
            {
                string slotBytes = NextRd(), at = NextRd();
                L($"cvt.u64.u32 {slotBytes}, {plan.Slot[e]};");
                L($"add.u64 {at}, {operandBase}, {slotBytes};");
                if (!plan.Vector) L($"st.shared.f32 [{at}], {values[e]};");
                else if (plan.ReductionFastest)
                    // Four reduction terms of one position: one float in each of four consecutive rows.
                    for (int j = 0; j < VectorWidth; j++)
                        L($"st.shared.f32 [{at}+{I(j * rowBytes)}], {values[e * VectorWidth + j]};");
                else
                    // Four positions of one reduction row: adjacent, and 16-byte aligned (rows and starts are multiples of 4).
                    L($"st.shared.v4.f32 [{at}], {{{values[e * VectorWidth]}, {values[e * VectorWidth + 1]}, " +
                      $"{values[e * VectorWidth + 2]}, {values[e * VectorWidth + 3]}}};");
            }
        }
        string r0 = NextR();
        L($"mov.u32 {r0}, 0;");
        Store(stageA, Gather(stageA, r0), current);
        Store(stageB, Gather(stageB, r0), current);
        L("bar.sync 0;");
        _body.Append("IG_STEP:\n");
        // The loop body is straight-line: a gather under a branch is sunk by ptxas to its stores after
        // the compute, which exposes the full global latency every step. The last step therefore
        // re-gathers the final chunk (clamped) and fills a buffer nobody reads.
        string rNext = NextR(), rFetch = NextR();
        L($"add.u32 {rNext}, {r0}, {I(ImplicitGemmChunk)};");
        L($"min.u32 {rFetch}, {rNext}, {I((rTotal - 1) / ImplicitGemmChunk * ImplicitGemmChunk)};");
        var nextA = Gather(stageA, rFetch);
        var nextB = Gather(stageB, rFetch);
        // ---- compute: thread (ty, tx) owns rows ty*perM..+perM and columns tx*perN..+perN of the tile.
        string aRow = NextRd(), bRow = NextRd(), tyBytes = NextRd(), txBytes = NextRd();
        L($"mul.wide.u32 {tyBytes}, {ty}, {I(perM * sizeof(float))};");
        L($"mul.wide.u32 {txBytes}, {tx}, {I(perN * sizeof(float))};");
        L($"add.u64 {aRow}, {sharedBase}, {current};");
        L($"add.u64 {aRow}, {aRow}, {tyBytes};");
        L($"add.u64 {bRow}, {sharedBase}, {current};");
        L($"add.u64 {bRow}, {bRow}, {txBytes};");
        L($"add.u64 {bRow}, {bRow}, {I(bufferBytesA)};");
        // A thread's run of per floats, read 4 (or 2) at a time.
        string[] ReadRun(string row, int count, int byteOffset)
        {
            var values = new string[count];
            for (int v = 0; v < count; v++) values[v] = NextF();
            int width = count % 4 == 0 ? 4 : 2;
            for (int v = 0; v < count; v += width)
            {
                var part = new string[width];
                Array.Copy(values, v, part, 0, width);
                L($"ld.shared.v{I(width)}.f32 {{{string.Join(", ", part)}}}, [{row}+{I(byteOffset + v * sizeof(float))}];");
            }
            return values;
        }
        for (int kk = 0; kk < ImplicitGemmChunk; kk++)
        {
            var av = ReadRun(aRow, perM, kk * rowFloatsA * sizeof(float));
            var bv = ReadRun(bRow, perN, kk * rowFloatsB * sizeof(float));
            for (int i = 0; i < perM; i++)
                for (int j = 0; j < perN; j++)
                    L($"fma.rn.f32 {acc[i, j]}, {av[i]}, {bv[j]}, {acc[i, j]};");
        }
        // The other buffer was last read in the previous step, which ended at a barrier, so it is free.
        string other = NextRd(), hasNext = NextP();
        L($"xor.b64 {other}, {current}, {I(pairBytes)};");
        Store(stageA, nextA, other);
        Store(stageB, nextB, other);
        L("bar.sync 0;");
        L($"mov.u64 {current}, {other};");
        L($"mov.u32 {r0}, {rNext};");
        L($"setp.lt.u32 {hasNext}, {rNext}, {I(rTotal)};");
        L($"@{hasNext} bra IG_STEP;");
        // ---- epilogue and store, guarded per output.
        for (int i = 0; i < perM; i++)
            for (int j = 0; j < perN; j++)
            {
                string mm = NextR(), nn = NextR(), inM = NextP(), inN = NextP(), skip = NextP();
                L($"mad.lo.u32 {mm}, {ty}, {I(perM)}, {m0};");
                if (i > 0) L($"add.u32 {mm}, {mm}, {I(i)};");
                L($"mad.lo.u32 {nn}, {tx}, {I(perN)}, {n0};");
                if (j > 0) L($"add.u32 {nn}, {nn}, {I(j)};");
                L($"setp.lt.u32 {inM}, {mm}, {I(mTotal)};");
                L($"setp.lt.u32 {inN}, {nn}, {I(nTotal)};");
                L($"and.pred {skip}, {inM}, {inN};");
                string label = "IG_SKIP" + I(i * perN + j);
                L($"@!{skip} bra {label};");
                Decompose(roles.M, mm);
                Decompose(roles.N, nn);
                string value = acc[i, j];
                if (spec.ReduceScale != 1.0)
                {
                    string s = NextF();
                    L($"mul.rn.f32 {s}, {value}, {F32(spec.ReduceScale)};");
                    value = s;
                }
                if (spec.BiasInput is { } bias)
                {
                    var bb = spec.Inputs[bias];
                    string bvalue = EmitLoad(bb, basePtr[bb.ParameterIndex], axisReg, Array.Empty<int>(), Array.Empty<int>(), out _);
                    string s = NextF();
                    L($"add.rn.f32 {s}, {value}, {bvalue};");
                    value = s;
                }
                if (spec.ScaleInput is { } scale)
                {
                    var sb = spec.Inputs[scale];
                    string svalue = EmitLoad(sb, basePtr[sb.ParameterIndex], axisReg, Array.Empty<int>(), Array.Empty<int>(), out _);
                    string s = NextF();
                    L($"mul.rn.f32 {s}, {value}, {svalue};");
                    value = s;
                }
                if (spec.Activation != CodegenActivationKind.None)
                {
                    string fresh = NextF();
                    L($"mov.f32 {fresh}, {value};");
                    value = EmitActivation(spec.Activation, fresh);
                }
                string offset = EmitOffset(spec.Output, axisReg, Array.Empty<int>(), Array.Empty<int>(), out string? outPred);
                string bytes = NextRd(), at = NextRd();
                L($"mul.wide.s32 {bytes}, {offset}, 4;");
                L($"add.u64 {at}, {basePtr[spec.Output.ParameterIndex]}, {bytes};");
                L(outPred is null ? $"st.global.f32 [{at}], {value};" : $"@{outPred} st.global.f32 [{at}], {value};");
                _body.Append(label).Append(":\n");
            }

        _body.Append("END:\n    ret;\n}\n");
        return Assemble();
    }

    /// <summary>
    /// Whether staging an operand walks the reduction with consecutive threads (otherwise its own
    /// output axes): the order whose first warp touches fewer 32-byte sectors, so the gather coalesces
    /// whichever way the operand is laid out. A weight tensor is contiguous along the reduction and
    /// strided along output channels; an activation is usually the reverse.
    /// </summary>
    private static bool StagesReductionFastest(
        CodegenKernelSpec spec, CodegenTensorBinding binding, int[] own, int[] reduction, int tile)
    {
        var axes = spec.Space.Axes;
        int Sectors(bool reductionFastest)
        {
            var sectors = new HashSet<long>();
            var value = new long[axes.Count];
            for (int lane = 0; lane < 32; lane++)
            {
                long pos = reductionFastest ? lane / ImplicitGemmChunk : lane % PositionLanes;
                long kk = reductionFastest ? lane % ImplicitGemmChunk : lane / PositionLanes;
                Array.Clear(value, 0, value.Length);
                void Spread(int[] set, long flat)
                {
                    for (int k = set.Length - 1; k >= 0; k--)
                    {
                        value[set[k]] = flat % axes[set[k]].Extent;
                        flat /= axes[set[k]].Extent;
                    }
                }
                Spread(own, pos);
                Spread(reduction, kk);
                long offset = 0;
                for (int d = 0; d < binding.Map.Count; d++)
                {
                    var expr = binding.Map[d];
                    long v = expr.Constant;
                    foreach (var term in expr.Terms) v += (long)term.Coefficient * value[term.Axis];
                    offset += FloorDiv(v, expr.Divisor) * binding.Stride(d);
                }
                sectors.Add(FloorDiv(offset * binding.ElementBytes, 32));
            }
            return sectors.Count;
        }
        static long FloorDiv(long a, long b) => a >= 0 ? a / b : -((-a + b - 1) / b);
        return Sectors(reductionFastest: true) < Sectors(reductionFastest: false);
    }

    /// <summary>
    /// Whether an operand is staged four floats at a time: its staged-fastest role is a single axis that runs along the
    /// operand's unit-stride dimension and spans it (<see cref="IsUnitStrideIn"/>), that extent and every other stride are
    /// multiples of 4 -- so every vector starts 16-byte aligned in a 16-byte-aligned buffer and lies wholly inside or
    /// outside the operand -- every other dimension is indexed in range by construction (non-negative axis terms, no
    /// constant or divisor, at most the dimension's extent: a vector load cannot carry a guard), and the tile is a
    /// multiple of 64, so each thread stages whole vectors.
    /// </summary>
    private bool StagesVectors(CodegenKernelSpec spec, CodegenTensorBinding binding, int[] own, int[] reduction,
        bool reductionFastest, int tile)
    {
        if (!EnableVectorLoads || binding.NeedsConversion || binding.ElementType != CodegenElementType.Float32 ||
            tile % (PositionLanes * VectorWidth) != 0)
            return false;
        int[] fast = reductionFastest ? reduction : own;
        if (fast.Length != 1 || !IsUnitStrideIn(binding, fast[0], spec.Space.Axes) ||
            spec.Space.Axes[fast[0]].Extent % VectorWidth != 0)
            return false;
        var axes = spec.Space.Axes;
        for (int d = 0; d < binding.Map.Count - 1; d++)
        {
            if (binding.Stride(d) % VectorWidth != 0) return false;
            var expr = binding.Map[d];
            if (binding.Indirect[d] is not null || expr.Divisor != 1 || expr.RequiresExactDivision || expr.Constant != 0)
                return false;
            long highest = 0;
            foreach (var term in expr.Terms)
            {
                if (term.Coefficient < 0) return false;
                highest += (long)term.Coefficient * (axes[term.Axis].Extent - 1);
            }
            if (highest >= binding.Shape[d]) return false;
        }
        return true;
    }

    /// <summary>What one thread stages of one implicit-GEMM operand: per slot, its shared offset (bytes), its
    /// reduction offset within the chunk, its own-axis coordinates and whether they are in range. A vector slot holds
    /// four consecutive floats of the staged-fastest role.</summary>
    private sealed class ImplicitGemmStage
    {
        public ImplicitGemmStage(int input, int[] own, bool reductionFastest, int operandOffset, int count, bool vector)
        {
            Input = input; Own = own; ReductionFastest = reductionFastest; OperandOffset = operandOffset; Vector = vector;
            Slot = new string[count]; Chunk = new string[count];
            OwnAxes = new string[count][]; OwnInRange = new string?[count];
        }
        public int Input { get; }
        public int[] Own { get; }
        public bool ReductionFastest { get; }
        public bool Vector { get; }
        public int OperandOffset { get; }
        public string[] Slot { get; }
        public string[] Chunk { get; }
        public string[][] OwnAxes { get; }
        public string?[] OwnInRange { get; }
    }
}
