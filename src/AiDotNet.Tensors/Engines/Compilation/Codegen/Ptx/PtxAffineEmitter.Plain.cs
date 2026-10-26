// Copyright (c) AiDotNet. All rights reserved.
// The plain lowering (#1441): one thread per output, every reduction a rolled loop, computing in fp32 or fp64. It
// serves the two kinds of spec the tuned path cannot:
//
//   * an fp64 spec -- any tensor stored as Float64 makes the whole spec fp64, end to end. The tuned path hardcodes .f32
//     throughout, and an fp64 operand read through it would compute with half of a double's bits;
//   * a spec with a data-dependent reduction extent (CodegenReductionRange) -- a CSR row's entries, a loop whose bounds
//     are loaded at run time, which no tiling, unrolling or split of the tuned path can express.
//
// Emit hands both here first, and every planner that retiles or splits a spec (row fusion, split and warp reductions,
// implicit GEMM) refuses them. Only the integer index machinery (EmitOffset, with its affine and indirect indices and
// their derived guards) is shared, being type-agnostic. Every operand widens exactly to the compute type on load; a term
// any of whose operands is out of range contributes the reduction's identity, exactly as the interpreter skips it.

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

public sealed partial class PtxAffineEmitter
{
    private int _fd;
    private bool _plainDouble;

    private string NextD() => "%fd" + (_fd++).ToString(CultureInfo.InvariantCulture);

    // A register of the plain lowering's compute type.
    private string NextC() => _plainDouble ? NextD() : NextF();

    private string Ty => _plainDouble ? "f64" : "f32";

    private static string D64(double value) =>
        "0d" + BitConverter.DoubleToInt64Bits(value).ToString("X16", CultureInfo.InvariantCulture);

    private static string F32Bits(double value) =>
        "0f" + AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxCompat.SingleToInt32Bits((float)value).ToString("X8", CultureInfo.InvariantCulture);

    private string Lit(double value) => _plainDouble ? D64(value) : F32Bits(value);

    /// <summary>True when any tensor of <paramref name="spec"/> is stored as Float64: the spec computes in fp64.</summary>
    internal static bool ComputesInDouble(CodegenKernelSpec spec) => spec.HasDoubles;

    /// <summary>True when the plain lowering, not the tuned one, emits <paramref name="spec"/>.</summary>
    internal static bool LowersPlainly(CodegenKernelSpec spec) => spec.HasDoubles || spec.ReductionRange is not null;

    private string EmitPlain(CodegenKernelSpec spec, int computeMajor, int computeMinor)
    {
        _plainDouble = spec.HasDoubles;
        // What this lowering expresses; anything else is refused rather than emitted wrongly.
        if (spec.Algebra != CodegenAlgebra.Real) throw new NotSupportedException("A plain spec must use the real algebra.");
        if (spec.ExtraOutputs.Count != 0)
            throw new NotSupportedException("A plain spec writes one output; " + spec.Name + " has extra outputs.");
        if (spec.Activation != CodegenActivationKind.None || spec.PostScaleInput.HasValue || spec.PostBiasInput.HasValue)
            throw new NotSupportedException(
                "A plain spec reduces terms with at most a pre-bias, pre-reduce, scale, offset, bias and scale tensor; " +
                spec.Name + " asks for more.");
        if (_plainDouble && spec.PreReduce == CodegenPreReduceOp.Exp)
            throw new NotSupportedException("An fp64 spec has no exponential.");
        if (spec.Output.NeedsAtomicStore) throw new NotSupportedException("A plain scatter (atomic) output is not lowered.");
        if (spec.Body is not null) RefusePlainBody(spec.Body);
        foreach (var binding in AllBindings(spec))
            if (binding.NeedsConversion)
                throw new NotSupportedException("'" + binding.Name + "' is 16-bit; a plain spec reads fp64, fp32 or integer values.");

        var space = spec.Space;
        var axes = space.Axes;
        axesMeta = axes;
        int[] parallel = space.ParallelAxes;
        int[] reduction = space.ReductionAxes;
        long threadCount = 1;
        foreach (int p in parallel) threadCount *= axes[p].Extent;
        long blocks = (threadCount + BlockThreads - 1) / BlockThreads;
        CheckLimits(spec, blocks, threadCount);
        LaunchBlocks = (uint)blocks;
        LaunchBlockX = BlockThreads;
        LaunchBlockY = 1;
        LaunchBlockThreads = BlockThreads;
        SharedMemoryBytes = 0;
        Unrolled = reduction.Length == 0;
        LoopedAxes = reduction.Length;
        WarpReduced = false;
        VectorisedLoads = 0;

        _sb.Clear(); _body.Clear(); _calledFunctions.Clear(); _r = _f = _p = _rd = _fd = 0; EmittedLoads = 0; ElidedGuards = 0;
        IndirectIndexLoads = 0; AtomicStores = 0; ExtraOutputStores = 0; VectorisedStores = 0;
        _inputBindings = spec.Inputs;
        _stageLabel = 0;
        _outOfRangeFill = "0f00000000";

        int paramCount = spec.Inputs.Count + 1;
        _sb.Append(".version ").Append(PtxIsaVersionFor(computeMajor, computeMinor)).Append('\n')
           .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
           .Append(".address_size 64\n\n")
           .Append("// generated by PtxAffineEmitter (plain ").Append(Ty).Append(") from CodegenKernelSpec\n")
           .Append("// ").Append(spec.Describe().Replace("\n", "\n// ")).Append('\n')
           .Append(".visible .entry ").Append(spec.Name).Append("(\n");
        for (int i = 0; i < paramCount; i++)
            _sb.Append("    .param .u64 p").Append(I(i)).Append(i == paramCount - 1 ? "\n" : ",\n");
        _sb.Append(")\n{\n");

        var basePtr = new string[paramCount];
        for (int i = 0; i < paramCount; i++)
        {
            basePtr[i] = NextRd();
            L($"ld.param.u64 {basePtr[i]}, [p{I(i)}];");
        }
        _basePointers = basePtr;

        string ctaid = NextR(), tid = NextR(), gid = NextR(), guard = NextP();
        L($"mov.u32 {ctaid}, %ctaid.x;");
        L($"mov.u32 {tid}, %tid.x;");
        L($"mad.lo.u32 {gid}, {ctaid}, {I(BlockThreads)}, {tid};");
        L($"setp.ge.u32 {guard}, {gid}, {I(threadCount)};");
        L($"@{guard} bra END;");

        // The flat id across the parallel axes, last-declared fastest.
        var axisReg = new string[axes.Count];
        string rest = gid;
        for (int k = parallel.Length - 1; k >= 0; k--)
        {
            int ax = parallel[k];
            axisReg[ax] = NextR();
            if (k == 0)
            {
                L($"mov.u32 {axisReg[ax]}, {rest};   // {axes[ax].Name}");
            }
            else
            {
                L($"rem.u32 {axisReg[ax]}, {rest}, {I(axes[ax].Extent)};   // {axes[ax].Name}");
                string next = NextR();
                L($"div.u32 {next}, {rest}, {I(axes[ax].Extent)};");
                rest = next;
            }
        }

        double identity = spec.Reduce switch
        {
            CodegenReduceKind.Max => double.NegativeInfinity,
            CodegenReduceKind.Product => 1.0,
            _ => 0.0,
        };
        var range = spec.ReductionRange;
        string acc = NextC();
        L($"mov.{Ty} {acc}, {Lit(range?.Initial ?? identity)};");

        var noFolding = Array.Empty<int>();
        var noValues = new int[axes.Count];

        // A data-dependent range: its bounds loaded at the range's position and the next, clamped to the axis' extent.
        string? rangeStart = null, rangeEnd = null;
        if (range is not null)
        {
            rangeStart = LoadRangeBound(spec, range, axisReg, noValues, noFolding, plusOne: false);
            rangeEnd = LoadRangeBound(spec, range, axisReg, noValues, noFolding, plusOne: true);
        }

        // Every reduction axis is a rolled loop, its counter the axis register: indices on it stay symbolic. The range
        // axis starts at its loaded bound and skips its body entirely when the range is empty.
        for (int level = 0; level < reduction.Length; level++)
        {
            int ax = reduction[level];
            axisReg[ax] = NextR();
            if (range is not null && ax == range.Axis)
            {
                string empty = NextP();
                L($"mov.u32 {axisReg[ax]}, {rangeStart};   // {axes[ax].Name} (ranged)");
                L($"setp.ge.s32 {empty}, {rangeStart}, {rangeEnd};");
                L($"@{empty} bra PLAINSKIP{I(level)};");
            }
            else
            {
                L($"mov.u32 {axisReg[ax]}, 0;   // {axes[ax].Name}");
            }
            _body.Append("PLAINLOOP").Append(I(level)).Append(":\n");
        }

        // One term: every operand loaded (a guarded read keeps the term's predicate), then the product or the body.
        string? termPred = null;
        var value = new Dictionary<int, string>();
        foreach (int input in spec.Body is not null ? spec.Body.Operands() : ToArray(spec.ProductInputs))
        {
            value[input] = LoadPlain(spec.Inputs[input], axisReg, noValues, noFolding, out string? pred);
            if (pred is not null) termPred = termPred is null ? pred : AndPred(termPred, pred);
        }
        string term;
        if (spec.Body is not null)
        {
            term = LowerPlainBody(spec.Body, value);
        }
        else
        {
            term = value[spec.ProductInputs[0]];
            for (int k = 1; k < spec.ProductInputs.Count; k++)
            {
                string product = NextC();
                L($"mul.rn.{Ty} {product}, {term}, {value[spec.ProductInputs[k]]};");
                term = product;
            }
        }
        if (spec.PreBiasInput is { } preBias)
        {
            string pb = LoadPlain(spec.Inputs[preBias], axisReg, noValues, noFolding, out string? pbPred);
            if (pbPred is not null) termPred = termPred is null ? pbPred : AndPred(termPred, pbPred);
            string shifted = NextC();
            L($"fma.rn.{Ty} {shifted}, {pb}, {Lit(spec.PreBiasScale)}, {term};");
            term = shifted;
        }
        if (spec.PreReduce == CodegenPreReduceOp.Square)
        {
            string squared = NextC();
            L($"mul.rn.{Ty} {squared}, {term}, {term};");
            term = squared;
        }
        else if (spec.PreReduce == CodegenPreReduceOp.Exp)
        {
            string scaled = NextF(), exp = NextF();
            L($"mul.rn.f32 {scaled}, {term}, 0f3FB8AA3B;");
            L($"ex2.approx.f32 {exp}, {scaled};");
            term = exp;
        }
        if (termPred is not null)
        {
            // Any out-of-range operand: the whole term is the identity, as the interpreter skips it.
            string id = NextC(), kept = NextC();
            L($"mov.{Ty} {id}, {Lit(identity)};");
            L($"selp.{Ty} {kept}, {term}, {id}, {termPred};");
            term = kept;
        }
        switch (spec.Reduce)
        {
            case CodegenReduceKind.None: L($"mov.{Ty} {acc}, {term};"); break;
            case CodegenReduceKind.Sum: L($"add.rn.{Ty} {acc}, {acc}, {term};"); break;
            case CodegenReduceKind.Max: L($"max.{Ty} {acc}, {acc}, {term};"); break;
            case CodegenReduceKind.Product: L($"mul.rn.{Ty} {acc}, {acc}, {term};"); break;
            default: throw new NotSupportedException("Unhandled reduction " + spec.Reduce + ".");
        }

        for (int level = reduction.Length - 1; level >= 0; level--)
        {
            int ax = reduction[level];
            bool ranged = range is not null && ax == range.Axis;
            string more = NextP();
            L($"add.s32 {axisReg[ax]}, {axisReg[ax]}, 1;");
            L(ranged
                ? $"setp.lt.s32 {more}, {axisReg[ax]}, {rangeEnd};"
                : $"setp.lt.s32 {more}, {axisReg[ax]}, {I(axes[ax].Extent)};");
            L($"@{more} bra PLAINLOOP{I(level)};");
            if (ranged) _body.Append("PLAINSKIP").Append(I(level)).Append(":\n");
        }

        // The epilogue, in the interpreter's order: scale, offset, bias, scale tensor.
        if (spec.ReduceScale != 1.0) L($"mul.rn.{Ty} {acc}, {acc}, {Lit(spec.ReduceScale)};");
        if (spec.ReduceOffset != 0.0) L($"add.rn.{Ty} {acc}, {acc}, {Lit(spec.ReduceOffset)};");
        foreach (int? operand in new[] { spec.BiasInput, spec.ScaleInput })
        {
            if (operand is not { } at) continue;
            string v = LoadPlain(spec.Inputs[at], axisReg, noValues, noFolding, out string? pred);
            if (pred is not null)
            {
                // An out-of-range bias adds nothing and an out-of-range scale scales by nothing (the interpreter
                // skips both), so the neutral value stands in.
                string neutral = NextC();
                L($"mov.{Ty} {neutral}, {Lit(operand == spec.BiasInput ? 0.0 : 1.0)};");
                L($"selp.{Ty} {v}, {v}, {neutral}, {pred};");
            }
            L(operand == spec.BiasInput ? $"add.rn.{Ty} {acc}, {acc}, {v};" : $"mul.rn.{Ty} {acc}, {acc}, {v};");
        }
        if (range is { EmptyValue: { } emptyValue })
        {
            string isEmpty = NextP(), fill = NextC();
            L($"setp.ge.s32 {isEmpty}, {rangeStart}, {rangeEnd};");
            L($"mov.{Ty} {fill}, {Lit(emptyValue)};");
            L($"selp.{Ty} {acc}, {fill}, {acc}, {isEmpty};");
        }

        // Store at the output's own type.
        string offset = EmitOffset(spec.Output, axisReg, noValues, noFolding, out string? outPred);
        string bytes = NextRd(), addr = NextRd();
        L($"mul.wide.u32 {bytes}, {offset}, {I(spec.Output.ElementBytes)};");
        L($"add.u64 {addr}, {basePtr[spec.Output.ParameterIndex]}, {bytes};");
        string guardText = outPred is null ? "" : "@" + outPred + " ";
        switch (spec.Output.ElementType)
        {
            case CodegenElementType.Float64:
                L($"{guardText}st.global.f64 [{addr}], {acc};");
                break;
            case CodegenElementType.Float32 when _plainDouble:
            {
                string narrow = NextF();
                L($"cvt.rn.f32.f64 {narrow}, {acc};");
                L($"{guardText}st.global.f32 [{addr}], {narrow};");
                break;
            }
            case CodegenElementType.Float32:
                L($"{guardText}st.global.f32 [{addr}], {acc};");
                break;
            case CodegenElementType.Int32 when spec.Output.IsIntegerValue:
            {
                string asInt = NextR();
                L($"cvt.rni.s32.{Ty} {asInt}, {acc};");
                L($"{guardText}st.global.s32 [{addr}], {asInt};");
                break;
            }
            default:
                throw new NotSupportedException("A plain spec stores fp64, fp32 or int32 values; '" + spec.Output.Name + "' is " +
                    spec.Output.ElementType + ".");
        }

        _body.Append("END:\n    ret;\n}\n");
        CalledFunctions = Array.Empty<CodegenLibraryFunction>();
        _sb.Append("    .reg .pred %p<").Append(I(_p + 8)).Append(">;\n")
           .Append("    .reg .b32 %r<").Append(I(_r + 8)).Append(">;\n")
           .Append("    .reg .b64 %rd<").Append(I(_rd + 8)).Append(">;\n")
           .Append("    .reg .f32 %f<").Append(I(_f + 8)).Append(">;\n")
           .Append(_fd > 0 ? "    .reg .f64 %fd<" + I(_fd + 8) + ">;\n" : "")
           .Append(_body);
        return _sb.ToString();
    }

    // One bound of a data-dependent range: indices[position (+ 1)], truncated when float, clamped to [0, extent].
    private string LoadRangeBound(CodegenKernelSpec spec, CodegenReductionRange range, string[] axisReg,
        int[] reductionValues, int[] reductionAxes, bool plusOne)
    {
        var source = spec.Inputs[range.IndexInput];
        var position = plusOne
            ? new CodegenAffineExpr(ToTerms(range.Position), range.Position.Constant + 1)
            : range.Position;
        string? positionPredicate = null;
        string at = EmitAffineIndex(position, axisReg, reductionValues, new HashSet<int>(reductionAxes),
            (int)source.ElementCount, ref positionPredicate);
        string bytes = NextRd(), addr = NextRd(), raw = NextR();
        L($"mul.wide.s32 {bytes}, {at}, 4;");
        L($"add.u64 {addr}, {_basePointers![source.ParameterIndex]}, {bytes};");
        L($"mov.u32 {raw}, 0;");
        string guard = positionPredicate is null ? "" : "@" + positionPredicate + " ";
        if (range.FloatIndices)
        {
            string asFloat = NextF();
            L($"mov.f32 {asFloat}, 0f00000000;");
            L($"{guard}ld.global.nc.f32 {asFloat}, [{addr}];");
            L($"cvt.rzi.s32.f32 {raw}, {asFloat};");
        }
        else
        {
            L($"{guard}ld.global.nc.u32 {raw}, [{addr}];");
        }
        IndirectIndexLoads++;
        string low = NextR(), clamped = NextR();
        L($"max.s32 {low}, {raw}, 0;");
        L($"min.s32 {clamped}, {low}, {I(spec.Space.Axes[range.Axis].Extent)};");
        return clamped;
    }

    private static CodegenAffineTerm[] ToTerms(CodegenAffineExpr expr)
    {
        var terms = new CodegenAffineTerm[expr.Terms.Count];
        for (int i = 0; i < terms.Length; i++) terms[i] = expr.Terms[i];
        return terms;
    }

    // Loads one operand widened to the compute type; a guarded read yields 0 (the caller selects on the predicate).
    private string LoadPlain(CodegenTensorBinding binding, string[] axisReg, int[] reductionValues, int[] reductionAxes,
        out string? predicate)
    {
        if (binding.IsIndexTensor)
            throw new NotSupportedException("'" + binding.Name + "' is an int32 index tensor and cannot be read as a value.");
        string offset = EmitOffset(binding, axisReg, reductionValues, reductionAxes, out predicate);
        string bytes = NextRd(), addr = NextRd(), dst = NextC();
        L($"mul.wide.s32 {bytes}, {offset}, {I(binding.ElementBytes)};");
        L($"add.u64 {addr}, {_basePointers![binding.ParameterIndex]}, {bytes};");
        string guard = predicate is null ? "" : "@" + predicate + " ";
        switch (binding.ElementType)
        {
            case CodegenElementType.Float64:
                L($"mov.f64 {dst}, {D64(0.0)};");
                L($"{guard}ld.global.nc.f64 {dst}, [{addr}];");
                break;
            case CodegenElementType.Float32 when _plainDouble:
            {
                string narrow = NextF();
                L($"mov.f32 {narrow}, 0f00000000;");
                L($"{guard}ld.global.nc.f32 {narrow}, [{addr}];");
                L($"cvt.f64.f32 {dst}, {narrow};");
                break;
            }
            case CodegenElementType.Float32:
                L($"mov.f32 {dst}, 0f00000000;");
                L($"{guard}ld.global.nc.f32 {dst}, [{addr}];");
                break;
            case CodegenElementType.Int32 or CodegenElementType.UInt8 when binding.IsIntegerValue:
            {
                string raw = NextR();
                bool isByte = binding.ElementType == CodegenElementType.UInt8;
                L($"mov.u32 {raw}, 0;");
                L($"{guard}{(isByte ? "ld.global.nc.u8" : "ld.global.nc.s32")} {raw}, [{addr}];");
                L(isByte ? $"cvt.rn.{Ty}.u32 {dst}, {raw};" : $"cvt.rn.{Ty}.s32 {dst}, {raw};");
                break;
            }
            default:
                throw new NotSupportedException("A plain spec cannot read '" + binding.Name + "' (" + binding.ElementType + ").");
        }
        EmittedLoads++;
        return dst;
    }

    private string LowerPlainBody(CodegenExpr body, Dictionary<int, string> operands) =>
        body.Lower(
            input => operands[input],
            constant => Lit(constant),
            (op, a, b) =>
            {
                string d = NextC();
                string typed = _plainDouble ? op.Replace(".f32", ".f64") : op;
                L(b is null ? $"{typed} {d}, {a};" : $"{typed} {d}, {a}, {b};");
                return d;
            });

    private static void RefusePlainBody(CodegenExpr body)
    {
        if (body.Calls().Count != 0)
            throw new NotSupportedException("A plain body cannot call library functions.");
        if (body.ScalarCount() != 0)
            throw new NotSupportedException("A plain body reads no runtime scalars.");
    }

    private static IEnumerable<CodegenTensorBinding> AllBindings(CodegenKernelSpec spec)
    {
        foreach (var input in spec.Inputs) yield return input;
        yield return spec.Output;
    }

    private static int[] ToArray(IReadOnlyList<int> list)
    {
        var array = new int[list.Count];
        for (int i = 0; i < array.Length; i++) array[i] = list[i];
        return array;
    }
}