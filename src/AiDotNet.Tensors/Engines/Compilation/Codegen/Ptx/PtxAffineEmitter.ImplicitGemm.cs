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

    private const int ImplicitGemmThreads = 256;
    private const int ImplicitGemmSide = 16;   // threads per tile edge
    private const int ImplicitGemmChunk = 8;   // reduction terms staged per step

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

        int tile = ImplicitGemmTile == 32 ? 32 : 64;
        int per = tile / ImplicitGemmSide;   // outputs per thread along each edge
        long mBlocks = (mTotal + tile - 1) / tile, nBlocks = (nTotal + tile - 1) / tile;
        long blocks = mBlocks * nBlocks * bTotal;
        if (blocks > int.MaxValue) throw new NotSupportedException("Too many tiles for one grid.");

        _sb.Clear(); _body.Clear(); _r = _f = _p = _rd = 0;
        EmittedLoads = 0; ElidedGuards = 0; IndirectIndexLoads = 0; AtomicStores = 0;
        ExtraOutputStores = 0; VectorisedStores = 0; VectorisedLoads = 0;
        _inputBindings = spec.Inputs;
        axesMeta = axes;   // the guard interval analysis reads every axis extent
        _outOfRangeFill = "0f00000000";
        _stageLabel = 0;
        StagedOperands = spec.Inputs[roles.A].Name + "+" + spec.Inputs[roles.B].Name;
        SharedMemoryBytes = 2 * ImplicitGemmChunk * tile * sizeof(float);
        CoarsenedLanes = per * per;
        LoopedAxes = 1; Unrolled = false; WarpReduced = false; UsedTwoDimensionalBlock = false;
        LaunchBlockX = ImplicitGemmThreads; LaunchBlockY = 1; LaunchBlockThreads = ImplicitGemmThreads;
        LaunchBlocks = (uint)blocks;
        TileDescription = "implicit-gemm " + tile + "x" + tile + "x" + ImplicitGemmChunk;

        _sb.Append(".version ").Append(PtxIsaVersionFor(computeMajor, computeMinor)).Append('\n')
           .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
           .Append(".address_size 64\n\n")
           .Append("// generated by PtxAffineEmitter.EmitImplicitGemm: ").Append(TileDescription).Append('\n')
           .Append("// ").Append(spec.Describe().Replace("\n", "\n// ")).Append('\n')
           .Append(".visible .entry ").Append(spec.Name).Append("(\n");
        for (int i = 0; i < spec.ParameterCount; i++)
            _sb.Append("    .param .u64 p").Append(I(i)).Append(i == spec.ParameterCount - 1 ? "\n" : ",\n");
        _sb.Append(")\n{\n");

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
        L($"mul.lo.u32 {m0}, {mTile}, {I(tile)};");
        L($"mul.lo.u32 {n0}, {nTile}, {I(tile)};");

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

        string sharedA = NextRd(), sharedB = NextRd();
        L($"mov.u64 {sharedA}, stageBuf;");
        L($"add.u64 {sharedB}, {sharedA}, {I(ImplicitGemmChunk * tile * sizeof(float))};");

        var acc = new string[per, per];
        for (int i = 0; i < per; i++)
            for (int j = 0; j < per; j++)
            {
                acc[i, j] = NextF();
                L($"mov.f32 {acc[i, j]}, 0f00000000;");
            }

        string r0 = NextR();
        L($"mov.u32 {r0}, 0;");
        _body.Append("IG_STEP:\n");

        // ---- stage one operand: KC x tile values, gathered through its own map.
        void Stage(int input, int[] own, long ownTotal, string origin, string shared, bool reductionFastest)
        {
            var binding = spec.Inputs[input];
            int count = ImplicitGemmChunk * tile / ImplicitGemmThreads;
            for (int e = 0; e < count; e++)
            {
                string lin = NextR(), kk = NextR(), pos = NextR();
                L($"add.u32 {lin}, {tid}, {I(e * ImplicitGemmThreads)};");
                // A operand: consecutive threads walk the reduction (contiguous in a weight
                // tensor); B: consecutive threads walk N (contiguous along output positions).
                if (reductionFastest)
                {
                    L($"rem.u32 {kk}, {lin}, {I(ImplicitGemmChunk)};");
                    L($"div.u32 {pos}, {lin}, {I(ImplicitGemmChunk)};");
                }
                else
                {
                    L($"rem.u32 {pos}, {lin}, {I(tile)};");
                    L($"div.u32 {kk}, {lin}, {I(tile)};");
                }
                string g = NextR(), r = NextR(), inG = NextP(), inR = NextP(), ok = NextP();
                string gc = NextR(), rc = NextR();
                L($"add.u32 {g}, {origin}, {pos};");
                L($"add.u32 {r}, {r0}, {kk};");
                L($"setp.lt.u32 {inG}, {g}, {I(ownTotal)};");
                L($"setp.lt.u32 {inR}, {r}, {I(rTotal)};");
                L($"and.pred {ok}, {inG}, {inR};");
                // Clamp so the gathered address is always inside the operand; the result is
                // then zeroed when the position is outside the tile's problem.
                L($"min.u32 {gc}, {g}, {I(ownTotal - 1)};");
                L($"min.u32 {rc}, {r}, {I(rTotal - 1)};");
                Decompose(own, gc);
                Decompose(roles.R, rc);
                string value = EmitLoad(binding, basePtr[binding.ParameterIndex], axisReg,
                    Array.Empty<int>(), Array.Empty<int>(), out _);
                string kept = NextF();
                L($"selp.f32 {kept}, {value}, 0f00000000, {ok};");
                string slot = NextR(), slotBytes = NextRd(), at = NextRd();
                L($"mad.lo.u32 {slot}, {kk}, {I(tile)}, {pos};");
                L($"mul.wide.u32 {slotBytes}, {slot}, 4;");
                L($"add.u64 {at}, {shared}, {slotBytes};");
                L($"st.shared.f32 [{at}], {kept};");
            }
        }
        Stage(roles.A, roles.M, mTotal, m0, sharedA, reductionFastest: true);
        Stage(roles.B, roles.N, nTotal, n0, sharedB, reductionFastest: false);
        L("bar.sync 0;");

        // ---- compute: thread (ty, tx) owns rows ty + 16i and columns tx + 16j of the tile.
        string aRow = NextRd(), bRow = NextRd(), tyBytes = NextRd(), txBytes = NextRd();
        L($"mul.wide.u32 {tyBytes}, {ty}, 4;");
        L($"mul.wide.u32 {txBytes}, {tx}, 4;");
        L($"add.u64 {aRow}, {sharedA}, {tyBytes};");
        L($"add.u64 {bRow}, {sharedB}, {txBytes};");
        for (int kk = 0; kk < ImplicitGemmChunk; kk++)
        {
            var av = new string[per];
            var bv = new string[per];
            for (int i = 0; i < per; i++)
            {
                av[i] = NextF();
                L($"ld.shared.f32 {av[i]}, [{aRow}+{I((kk * tile + i * ImplicitGemmSide) * 4)}];");
            }
            for (int j = 0; j < per; j++)
            {
                bv[j] = NextF();
                L($"ld.shared.f32 {bv[j]}, [{bRow}+{I((kk * tile + j * ImplicitGemmSide) * 4)}];");
            }
            for (int i = 0; i < per; i++)
                for (int j = 0; j < per; j++)
                    L($"fma.rn.f32 {acc[i, j]}, {av[i]}, {bv[j]}, {acc[i, j]};");
        }
        L("bar.sync 0;");
        string more = NextP();
        L($"add.u32 {r0}, {r0}, {I(ImplicitGemmChunk)};");
        L($"setp.lt.u32 {more}, {r0}, {I(rTotal)};");
        L($"@{more} bra IG_STEP;");

        // ---- epilogue and store, guarded per output.
        for (int i = 0; i < per; i++)
            for (int j = 0; j < per; j++)
            {
                string mm = NextR(), nn = NextR(), inM = NextP(), inN = NextP(), skip = NextP();
                L($"add.u32 {mm}, {m0}, {ty};");
                if (i > 0) L($"add.u32 {mm}, {mm}, {I(ImplicitGemmSide * i)};");
                L($"add.u32 {nn}, {n0}, {tx};");
                if (j > 0) L($"add.u32 {nn}, {nn}, {I(ImplicitGemmSide * j)};");
                L($"setp.lt.u32 {inM}, {mm}, {I(mTotal)};");
                L($"setp.lt.u32 {inN}, {nn}, {I(nTotal)};");
                L($"and.pred {skip}, {inM}, {inN};");
                string label = "IG_SKIP" + I(i * per + j);
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
}
