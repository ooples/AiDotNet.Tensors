// Copyright (c) AiDotNet. All rights reserved.
// Lowers a CodegenKernelSpec to SM86 PTX.
//
// This is the generated-code half of the C#-vs-Rust bake-off. It replaces
// hand-written PTX text for conv-class kernels, and it exists to remove a defect
// class rather than to save typing:
//
//   * the launch grid and the in-kernel bounds guard both read
//     CodegenIterationSpace.TotalThreads, so they cannot disagree;
//   * a load's validity predicate is derived from its index map and the tensor
//     shape, so it cannot be forgotten or hand-recomputed wrongly.
//
// Both of those were real failures in the hand-written #841 kernels.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>
/// Emits SM86 PTX for a <see cref="CodegenKernelSpec"/>.
/// </summary>
public sealed class PtxAffineEmitter
{
    /// <summary>Threads per block. Matches the hand-written kernels so metrics compare like-for-like.</summary>
    public const int BlockThreads = 256;

    /// <summary>
    /// Reduction trip counts at or below this are fully unrolled, which folds the
    /// reduction axes into compile-time constants inside every index expression.
    /// That is the specialisation a general library such as cuDNN cannot take,
    /// and it is what turns a guarded gather into a handful of constant-offset loads.
    /// </summary>
    public const int FullUnrollLimit = 64;

    private readonly StringBuilder _sb = new(16384);

    // The body is built separately from the header so the virtual-register
    // declarations can be written from the ACTUAL counts. They used to be fixed at
    // %f<512>/%r<1024>, which silently became a ceiling: coarsening pushed three
    // kernels past it and ptxas rejected the module with InvalidPtx, because the body
    // referenced registers the header never declared. A declaration derived from usage
    // cannot be outgrown.
    private readonly StringBuilder _body = new(16384);
    private int _r, _f, _p, _rd;
    private IReadOnlyList<CodegenAxis> axesMeta = Array.Empty<CodegenAxis>();

    private string NextR() => "%r" + (_r++).ToString(CultureInfo.InvariantCulture);
    private string NextF() => "%f" + (_f++).ToString(CultureInfo.InvariantCulture);
    private string NextP() => "%p" + (_p++).ToString(CultureInfo.InvariantCulture);
    private string NextRd() => "%rd" + (_rd++).ToString(CultureInfo.InvariantCulture);
    private static string I(long v) => v.ToString(CultureInfo.InvariantCulture);
    private void L(string line) => _body.Append("    ").Append(line).Append('\n');

    /// <summary>Number of SASS-visible global loads the emitter produced (diagnostic).</summary>
    public int EmittedLoads { get; private set; }

    /// <summary>
    /// Global loads ONE THREAD executes, counting loads inside a strip-mined loop once
    /// per trip. <see cref="EmittedLoads"/> counts instructions in the emitted text,
    /// which undercounts a looping kernel by its trip count. Load count is exactly the
    /// quantity the performance model needs, so it has to be the dynamic one.
    /// </summary>
    public long DynamicLoadsPerThread { get; private set; }

    /// <summary>True when the reduction was fully unrolled.</summary>
    public bool Unrolled { get; private set; }

    /// <summary>Bounds guards elided because interval analysis proved them unnecessary.</summary>
    public int ElidedGuards { get; private set; }

    /// <summary>
    /// Emits PTX for <paramref name="spec"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// Thrown for specs this layer deliberately cannot express (data-dependent
    /// indexing). Declining loudly is required -- silently mis-lowering an index
    /// map is exactly the failure this layer exists to prevent.
    /// </exception>
    /// <summary>Number of reduction axes lowered to runtime loops rather than unrolled.</summary>
    public int LoopedAxes { get; private set; }

    /// <summary>Elements per vector load. f32 x4 is the widest global load the ISA offers.</summary>
    private const int VectorWidth = 4;

    /// <summary>
    /// Minimum PTX ISA version that can name a given target. The version was
    /// previously fixed at 7.1 while the target was parameterised, which is
    /// self-contradictory: ptxas rejects <c>.version 7.1</c> paired with
    /// <c>.target sm_90</c>, because sm_89 and sm_90 were not introduced until ISA
    /// 7.8. Emitting for anything past Ampere produced invalid PTX.
    /// </summary>
    /// <remarks>
    /// 7.1 is the floor rather than the exact minimum for older targets, because that
    /// is the version the shipped sm_86 cubins were built with and lowering it would
    /// change their content hash for no benefit.
    /// </remarks>
    internal static string PtxIsaVersionFor(int computeMajor, int computeMinor)
    {
        int capability = computeMajor * 10 + computeMinor;
        if (capability >= 90) return "7.8";
        if (capability >= 89) return "7.8";
        if (capability >= 87) return "7.4";
        return "7.1";   // sm_70 through sm_86
    }

    /// <summary>Number of <c>ld.global.v4.f32</c> instructions emitted.</summary>
    public int VectorisedLoads { get; private set; }

    /// <summary>
    /// Adjacent outputs computed per thread. 1 disables coarsening, which exists so
    /// the two lowerings can be compared inside one process.
    /// </summary>
    public int Coarsening { get; set; } = 4;

    /// <summary>Lanes actually used: <see cref="Coarsening"/>, or 1 if it did not divide.</summary>
    public int CoarsenedLanes { get; private set; } = 1;

    /// <summary>
    /// Blocks the host must launch for the PTX just emitted. Read this rather than
    /// <see cref="GridBlocks"/> when coarsening may be active: the guard inside the
    /// kernel is derived from the same thread count, so the two cannot disagree.
    /// </summary>
    public uint LaunchBlocks { get; private set; }

    /// <summary>
    /// Enables vector loads. Off is not a production setting -- it exists so the two
    /// lowerings can be measured against each other inside ONE process, which is the
    /// only comparison Phase 0.5 showed to be trustworthy.
    /// </summary>
    public bool EnableVectorLoads { get; set; } = true;

    /// <summary>
    /// True when <paramref name="axis"/> indexes the binding's unit-stride dimension
    /// directly and covers all of it -- the condition under which four consecutive
    /// values of that axis are four consecutive, in-range, aligned floats.
    /// </summary>
    private static bool IsUnitStrideIn(
        CodegenTensorBinding binding, int axis, IReadOnlyList<CodegenAxis> axes)
    {
        int last = binding.Map.Count - 1;
        if (last < 0 || binding.Stride(last) != 1) return false;

        var expr = binding.Map[last];
        if (expr.Terms.Count != 1 || expr.Terms[0].Axis != axis ||
            expr.Terms[0].Coefficient != 1 || expr.Constant != 0 || expr.Divisor != 1)
            return false;

        // The axis must span the dimension exactly; otherwise the group could run off
        // the end and the load would need a guard it cannot have.
        return axes[axis].Extent == binding.Shape[last];
    }

    /// <summary>
    /// Emits one <c>ld.global.v4.f32</c> and returns the four component registers.
    /// </summary>
    private string[] EmitVectorLoad(
        CodegenTensorBinding binding, string basePointer, string[] axisReg,
        int[] reductionValues, int[] reductionAxes, int width)
    {
        string offset = EmitOffset(binding, axisReg, reductionValues, reductionAxes, out string? pred);
        if (pred != null)
            throw new InvalidOperationException(
                "A vectorised binding must be provably in range; " + binding.Name + " produced a guard.");

        string byteOffset = NextRd(), address = NextRd();
        L($"mul.wide.u32 {byteOffset}, {offset}, 4;");
        L($"add.u64 {address}, {basePointer}, {byteOffset};");

        var regs = new string[width];
        for (int i = 0; i < width; i++) regs[i] = NextF();
        L($"ld.global.v4.f32 {{{regs[0]}, {regs[1]}, {regs[2]}, {regs[3]}}}, [{address}];");
        VectorisedLoads++;
        EmittedLoads++;
        return regs;
    }

    public string Emit(CodegenKernelSpec spec, int computeMajor, int computeMinor)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));

        var space = spec.Space;
        var axes = space.Axes;
        axesMeta = axes;
        int[] parallel = space.ParallelAxes;
        int[] reduction = space.ReductionAxes;
        long total = space.TotalThreads;           // single source of truth
        long trips = space.ReductionTripCount;

        // STRIP-MINE. Full unroll is the fast path but it cannot be the only path:
        // a dense 3x3 conv over 32 input channels is 288 reduction trips, and
        // unrolling that produces a kernel ptxas cannot allocate. Peel outer
        // reduction axes into runtime loops until the remaining suffix fits the
        // unroll limit, so the inner taps stay unrolled (where the index folding
        // and guard elision pay off) while the channel walk becomes a real loop.
        int split = 0;
        long innerTrips = trips;
        while (innerTrips > FullUnrollLimit && split < reduction.Length)
        {
            innerTrips /= axes[reduction[split]].Extent;
            split++;
        }

        var loopAxes = new int[split];
        Array.Copy(reduction, 0, loopAxes, 0, split);
        var innerAxes = new int[reduction.Length - split];
        Array.Copy(reduction, split, innerAxes, 0, innerAxes.Length);

        Unrolled = split == 0;
        LoopedAxes = split;
        trips = innerTrips;
        reduction = innerAxes;

        // COARSEN. One thread per output element means every operand that does not
        // depend on the output position is re-loaded once per output. The bake-off
        // measured dense convolution at 2% of peak bandwidth and 5% of peak FP32 while
        // cuDNN reached 28%, and the cause was load COUNT, not load cost. Giving each
        // thread several adjacent outputs along the contiguous axis lets one weight
        // load feed several FMAs.
        //
        // The coarsened axis is the last-declared parallel axis, which is the one
        // consecutive threads walk, so a lane group stays contiguous in memory.
        int coarsenAxis = -1;
        int lanes = 1;
        if (Coarsening > 1 && parallel.Length > 0)
        {
            int candidate = parallel[parallel.Length - 1];
            if (axes[candidate].Extent % Coarsening == 0)
            {
                coarsenAxis = candidate;
                lanes = Coarsening;
            }
        }
        CoarsenedLanes = lanes;

        // Threads, grid and in-kernel guard all come from this one number, which is
        // the invariant the whole IR exists to protect.
        long threadCount = total / lanes;
        LaunchBlocks = (uint)((threadCount + BlockThreads - 1) / BlockThreads);

        _sb.Clear(); _body.Clear(); _r = _f = _p = _rd = 0; EmittedLoads = 0; ElidedGuards = 0;

        _sb.Append(".version ").Append(PtxIsaVersionFor(computeMajor, computeMinor)).Append('\n')
           .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
           .Append(".address_size 64\n\n")
           .Append("// generated by PtxAffineEmitter from CodegenKernelSpec\n")
           .Append("// ").Append(spec.Describe().Replace("\n", "\n// ")).Append('\n')
           .Append(".visible .entry ").Append(spec.Name).Append("(\n");

        int paramCount = spec.ParameterCount;
        for (int i = 0; i < paramCount; i++)
            _sb.Append("    .param .u64 p").Append(I(i)).Append(i == paramCount - 1 ? "\n" : ",\n");
        _sb.Append(")\n{\n");

        // Declarations are written AFTER the body, from the counts the body actually
        // used. They were previously fixed at %p<256>/%f<512>, which was a silent
        // ceiling rather than a generous bound: coarsening pushed the transposed
        // convolution to %p256, one past the declared range, and ptxas reported it as
        // "Arguments mismatch for instruction 'setp'" -- an undeclared register, not a
        // malformed instruction. A bound derived from usage cannot be outgrown.

        // Base pointers.
        var basePtr = new string[paramCount];
        for (int i = 0; i < paramCount; i++)
        {
            basePtr[i] = NextRd();
            L($"ld.param.u64 {basePtr[i]}, [p{I(i)}];");
        }

        // Flat thread id.
        string ctaid = NextR(), tid = NextR(), gid = NextR();
        L($"mov.u32 {ctaid}, %ctaid.x;");
        L($"mov.u32 {tid}, %tid.x;");
        L($"mad.lo.u32 {gid}, {ctaid}, {I(BlockThreads)}, {tid};");

        // Bounds guard derived from the SAME TotalThreads the host launches with.
        string guard = NextP();
        L($"setp.ge.u32 {guard}, {gid}, {I(threadCount)};");
        L($"@{guard} bra END;");

        // Decompose the flat id across parallel axes, last-declared fastest, so
        // consecutive threads walk the contiguous tensor axis. When coarsening, the
        // coarsened axis contributes extent/lanes to the decomposition, because one
        // thread now covers `lanes` of its positions.
        var axisReg = new string[axes.Count];
        string rest = gid;
        for (int p = parallel.Length - 1; p >= 0; p--)
        {
            int ax = parallel[p];
            int extent = ax == coarsenAxis ? axes[ax].Extent / lanes : axes[ax].Extent;
            axisReg[ax] = NextR();
            if (p == 0)
            {
                // Outermost axis takes whatever remains; no divide needed.
                L($"mov.u32 {axisReg[ax]}, {rest};   // {axes[ax].Name}");
            }
            else
            {
                L($"rem.u32 {axisReg[ax]}, {rest}, {I(extent)};   // {axes[ax].Name}");
                string nr = NextR();
                L($"div.u32 {nr}, {rest}, {I(extent)};");
                rest = nr;
            }
        }

        // One axis-register view per lane. Only the coarsened axis differs between
        // them: lane l covers position base*lanes + l.
        var laneAxisReg = new string[lanes][];
        for (int l = 0; l < lanes; l++)
        {
            laneAxisReg[l] = (string[])axisReg.Clone();
            if (coarsenAxis < 0) continue;
            string laneReg = NextR();
            if (l == 0) L($"mul.lo.u32 {laneReg}, {axisReg[coarsenAxis]}, {I(lanes)};");
            else L($"mad.lo.u32 {laneReg}, {axisReg[coarsenAxis]}, {I(lanes)}, {I(l)};");
            laneAxisReg[l][coarsenAxis] = laneReg;
        }

        long loadsBeforeLoop = EmittedLoads;

        // One accumulator per lane.
        var accs = new string[lanes];
        for (int l = 0; l < lanes; l++)
        {
            accs[l] = NextF();
            L(spec.Reduce == CodegenReduceKind.Max
                ? $"mov.f32 {accs[l]}, 0fFF800000;   // -inf"
                : $"mov.f32 {accs[l]}, 0f00000000;");
        }

        // Open one runtime loop per peeled axis. Each level emits its counter reset
        // BEFORE its own label, so the reset for level i+1 lands inside level i's
        // body and re-runs on every outer trip.
        var accsFixed = (string[])accs.Clone();
        var loopRegs = new string[loopAxes.Length];
        for (int i = 0; i < loopAxes.Length; i++)
        {
            string reg = NextR();
            loopRegs[i] = reg;
            axisReg[loopAxes[i]] = reg;   // symbolic from here on, not a folded constant

            // The per-lane views were cloned BEFORE this point, so they must be told
            // about the loop counter too. Without this a strip-mined kernel emitted an
            // empty operand ("mad.lo.s32 %r16, , 9, %r15") for the peeled axis, which
            // ptxas rejects -- the lane views had no register for it at all.
            for (int l = 0; l < lanes; l++) laneAxisReg[l][loopAxes[i]] = reg;
            L($"mov.u32 {reg}, 0;   // {axes[loopAxes[i]].Name}");
            // Must go to the BODY, not the header. Written to _sb it landed directly
            // after the opening brace -- ahead of the counter's own initialisation --
            // so the backward branch re-zeroed the counter every iteration and the
            // kernel never terminated.
            _body.Append("LOOP").Append(I(i)).Append(":\n");
        }

        // Decide, before emitting anything, which product operands can be read with a
        // vector load. The condition is exact rather than heuristic: the binding's
        // FASTEST tensor dimension (stride 1) must be indexed by exactly the innermost
        // reduction axis, with that axis covering the whole dimension so no bounds
        // check is needed, and the extent must be a multiple of the vector width.
        int innermost = reduction.Length > 0 ? reduction[reduction.Length - 1] : -1;
        var vectorisable = new bool[spec.Inputs.Count];
        var vectorRegs = new string[spec.Inputs.Count][];
        int vectorGroup = 1;
        if (EnableVectorLoads && innermost >= 0 && axes[innermost].Extent % VectorWidth == 0)
        {
            foreach (int inputIdx in spec.ProductInputs)
                if (IsUnitStrideIn(spec.Inputs[inputIdx], innermost, axes))
                {
                    vectorisable[inputIdx] = true;
                    vectorGroup = VectorWidth;
                }
        }
        VectorisedLoads = 0;

        // Reduction over the axes that remain unrolled: every one of those takes a
        // compile-time value, so each index expression folds to (symbolic terms) +
        // constant. A peeled axis stays symbolic and is simply another term.
        var reductionValues = new int[axes.Count];
        for (long t = 0; t < trips; t++)
        {
            long r = t;
            for (int i = reduction.Length - 1; i >= 0; i--)
            {
                int ax = reduction[i];
                int extent = axes[ax].Extent;
                reductionValues[ax] = (int)(r % extent);
                r /= extent;
            }

            // VECTORISED LOADS. When a binding's fastest tensor dimension is exactly
            // the innermost reduction axis, four consecutive trips read four
            // consecutive floats from it, and one ld.global.v4.f32 replaces four
            // ld.global.f32. The group start is a multiple of four and cudaMalloc
            // aligns to 256 bytes, so the 16-byte alignment the instruction requires
            // is guaranteed rather than hoped for.
            //
            // The vector is fetched on the FIRST trip of each group and the remaining
            // three trips reuse the components already in registers.
            if (vectorGroup > 1 && innermost >= 0)
            {
                int lane = reductionValues[innermost] % vectorGroup;
                if (lane == 0)
                {
                    foreach (int inputIdx in spec.ProductInputs)
                    {
                        if (!vectorisable[inputIdx]) continue;
                        var b = spec.Inputs[inputIdx];
                        vectorRegs[inputIdx] = EmitVectorLoad(
                            b, basePtr[b.ParameterIndex], axisReg, reductionValues, reduction, vectorGroup);
                    }
                }
            }

            // HOIST WHAT DOES NOT VARY ACROSS LANES. This is where coarsening actually
            // pays: a weight operand does not reference the coarsened axis, so ONE load
            // feeds every lane's FMA instead of one load per output. In a dense 3x3 over
            // 32 channels that is 288 weight loads serving `lanes` outputs rather than
            // `lanes` x 288. The bake-off showed dense convolution losing to cuDNN by
            // 4-6x on exactly this: too many loads, not loads that were too slow.
            // VECTORISE THE ACTIVATION OPERAND ACROSS LANES. An operand that is
            // unit-stride in the coarsened axis has its `lanes` values at consecutive
            // addresses, and lane 0 starts at a multiple of `lanes`, so one
            // ld.global.v4.f32 replaces four scalar loads. This is the operand that
            // vectorising along the reduction axis could never reach: reducing thread
            // count by 4 costs latency hiding, and without this the dense 1x1 measured
            // 0.944x -- a regression -- from coarsening alone.
            var laneVec = new string[spec.Inputs.Count][];
            if (lanes == VectorWidth && EnableVectorLoads)
            {
                foreach (int inputIdx in spec.ProductInputs)
                {
                    var b = spec.Inputs[inputIdx];
                    if (!IsUnitStrideIn(b, coarsenAxis, axes)) continue;
                    laneVec[inputIdx] = EmitVectorLoad(
                        b, basePtr[b.ParameterIndex], laneAxisReg[0],
                        reductionValues, reduction, VectorWidth);
                }
            }

            var shared = new string[spec.Inputs.Count];
            foreach (int inputIdx in spec.ProductInputs)
            {
                if (lanes == 1 || ReferencesAxis(spec.Inputs[inputIdx], coarsenAxis)) continue;
                if (vectorGroup > 1 && innermost >= 0 && vectorisable[inputIdx])
                {
                    shared[inputIdx] = vectorRegs[inputIdx]![reductionValues[innermost] % vectorGroup];
                    continue;
                }
                shared[inputIdx] = EmitLoad(
                    spec.Inputs[inputIdx], basePtr[spec.Inputs[inputIdx].ParameterIndex],
                    laneAxisReg[0], reductionValues, reduction, out string? sharedPred);
                if (sharedPred != null)
                    throw new InvalidOperationException(
                        "A lane-invariant operand must not need a lane-dependent guard: " +
                        spec.Inputs[inputIdx].Name + ".");
            }

            for (int l = 0; l < lanes; l++)
            {
                string? product = null;
                string? productPred = null;
                foreach (int inputIdx in spec.ProductInputs)
                {
                    var binding = spec.Inputs[inputIdx];
                    string value;
                    string? pred = null;
                    if (laneVec[inputIdx] != null)
                    {
                        value = laneVec[inputIdx]![l];
                    }
                    else if (shared[inputIdx] != null)
                    {
                        value = shared[inputIdx];
                    }
                    else if (vectorGroup > 1 && innermost >= 0 && vectorisable[inputIdx])
                    {
                        // Already resident from this group's vector load; a vectorisable
                        // binding is by construction in range, so it carries no predicate.
                        value = vectorRegs[inputIdx]![reductionValues[innermost] % vectorGroup];
                    }
                    else
                    {
                        value = EmitLoad(binding, basePtr[binding.ParameterIndex], laneAxisReg[l],
                                         reductionValues, reduction, out pred);
                    }
                    if (product is null) { product = value; productPred = pred; }
                    else
                    {
                        string mul = NextF();
                        L($"mul.rn.f32 {mul}, {product!}, {value};");
                        product = mul;
                        productPred = AndPred(productPred, pred);
                    }
                }

                // An out-of-range tap contributes the additive identity; the guarded
                // load already produced 0, so no extra select is needed for Sum.
                if (spec.Reduce == CodegenReduceKind.Max)
                {
                    L($"max.f32 {accs[l]}, {accs[l]}, {product!};");
                }
                else
                {
                    string na = NextF();
                    L($"add.rn.f32 {na}, {accs[l]}, {product!};");
                    accs[l] = na;
                }
            }
        }

        long loadsInLoopBody = EmittedLoads - loadsBeforeLoop;
        long loopTrips = 1;
        for (int i = 0; i < loopAxes.Length; i++) loopTrips *= axes[loopAxes[i]].Extent;

        // Close the loops. The renamed accumulator must land back in the fixed
        // register before the backward branch: SSA-style renaming cannot cross it.
        if (loopAxes.Length > 0)
        {
            for (int l = 0; l < lanes; l++)
                if (!string.Equals(accs[l], accsFixed[l], StringComparison.Ordinal))
                    L($"mov.f32 {accsFixed[l]}, {accs[l]};");
            for (int i = loopAxes.Length - 1; i >= 0; i--)
            {
                string cont = NextP();
                L($"add.s32 {loopRegs[i]}, {loopRegs[i]}, 1;");
                L($"setp.lt.s32 {cont}, {loopRegs[i]}, {I(axes[loopAxes[i]].Extent)};");
                L($"@{cont} bra LOOP{I(i)};");
            }
            accs = (string[])accsFixed.Clone();
        }

        // Epilogue and store, per lane. Bias and scale are hoisted when they do not
        // reference the coarsened axis, which for a channel-indexed bias is always.
        string? sharedBias = null, sharedScale = null;
        if (lanes > 1 && spec.BiasInput.HasValue &&
            !ReferencesAxis(spec.Inputs[spec.BiasInput.Value], coarsenAxis))
        {
            var b = spec.Inputs[spec.BiasInput.Value];
            sharedBias = EmitLoad(b, basePtr[b.ParameterIndex], laneAxisReg[0],
                                  reductionValues, reduction, out _);
        }
        if (lanes > 1 && spec.ScaleInput.HasValue &&
            !ReferencesAxis(spec.Inputs[spec.ScaleInput.Value], coarsenAxis))
        {
            var s = spec.Inputs[spec.ScaleInput.Value];
            sharedScale = EmitLoad(s, basePtr[s.ParameterIndex], laneAxisReg[0],
                                   reductionValues, reduction, out _);
        }

        for (int l = 0; l < lanes; l++)
        {
            string acc = accs[l];
            if (spec.BiasInput.HasValue)
            {
                var b = spec.Inputs[spec.BiasInput.Value];
                string v = sharedBias ?? EmitLoad(b, basePtr[b.ParameterIndex], laneAxisReg[l],
                                                  reductionValues, reduction, out _);
                string na = NextF();
                L($"add.rn.f32 {na}, {acc}, {v};");
                acc = na;
            }
            if (spec.ScaleInput.HasValue)
            {
                var s = spec.Inputs[spec.ScaleInput.Value];
                string v = sharedScale ?? EmitLoad(s, basePtr[s.ParameterIndex], laneAxisReg[l],
                                                   reductionValues, reduction, out _);
                string na = NextF();
                L($"mul.rn.f32 {na}, {acc}, {v};");
                acc = na;
            }
            if (spec.Activation == CodegenActivationKind.ReLU)
                L($"max.f32 {acc}, {acc}, 0f00000000;");

            string laneOff = EmitOffset(spec.Output, laneAxisReg[l], reductionValues, reduction,
                                       out string? lanePred);
            string laneAddr = NextRd(), laneByte = NextRd();
            L($"mul.wide.u32 {laneByte}, {laneOff}, 4;");
            L($"add.u64 {laneAddr}, {basePtr[spec.Output.ParameterIndex]}, {laneByte};");
            if (lanePred is null) L($"st.global.f32 [{laneAddr}], {acc};");
            else L($"@{lanePred} st.global.f32 [{laneAddr}], {acc};");
        }

        // Loads before the loop run once, loads in its body run once per trip, loads
        // after it (the epilogue) run once.
        long loadsAfterLoop = EmittedLoads - loadsBeforeLoop - loadsInLoopBody;
        DynamicLoadsPerThread = loadsBeforeLoop + loadsInLoopBody * loopTrips + loadsAfterLoop;

        _body.Append("END:\n    ret;\n}\n");

        _sb.Append("    .reg .pred %p<").Append(I(_p + 8)).Append(">;\n")
           .Append("    .reg .b32 %r<").Append(I(_r + 8)).Append(">;\n")
           .Append("    .reg .b64 %rd<").Append(I(_rd + 8)).Append(">;\n")
           .Append("    .reg .f32 %f<").Append(I(_f + 8)).Append(">;\n")
           .Append(_body);
        return _sb.ToString();
    }

    /// <summary>True when a binding's index map reads <paramref name="axis"/>.</summary>
    private static bool ReferencesAxis(CodegenTensorBinding binding, int axis)
    {
        if (axis < 0) return false;
        for (int d = 0; d < binding.Map.Count; d++)
            foreach (var term in binding.Map[d].Terms)
                if (term.Axis == axis) return true;
        return false;
    }

    /// <summary>
    /// Emits the flat element offset for a binding, plus the DERIVED validity
    /// predicate (null when the maps provably cannot leave the tensor).
    /// </summary>
    private string EmitOffset(
        CodegenTensorBinding binding,
        string[] axisReg,
        int[] reductionValues,
        int[] reductionAxes,
        out string? predicate)
    {
        predicate = null;
        var reductionSet = new HashSet<int>(reductionAxes);
        string? offset = null;

        for (int d = 0; d < binding.Map.Count; d++)
        {
            var expr = binding.Map[d];
            int dim = binding.Shape[d];
            long stride = binding.Stride(d);

            // Fold every reduction-axis term into the constant; keep parallel terms symbolic.
            int folded = expr.Constant;
            var symbolic = new List<CodegenAffineTerm>();
            foreach (var term in expr.Terms)
            {
                if (reductionSet.Contains(term.Axis)) folded += term.Coefficient * reductionValues[term.Axis];
                else symbolic.Add(term);
            }

            // Build the numerator.
            string? num;
            if (symbolic.Count == 0)
            {
                num = NextR();
                L($"mov.u32 {num}, {I(folded)};");
            }
            else
            {
                num = null;
                foreach (var term in symbolic)
                {
                    string contribution;
                    if (term.Coefficient == 1) contribution = axisReg[term.Axis];
                    else
                    {
                        contribution = NextR();
                        L($"mul.lo.s32 {contribution}, {axisReg[term.Axis]}, {I(term.Coefficient)};");
                    }
                    if (num is null) num = contribution;
                    else { string s = NextR(); L($"add.s32 {s}, {num}, {contribution};"); num = s; }
                }
                if (folded != 0) { string s = NextR(); L($"add.s32 {s}, {num}, {I(folded)};"); num = s; }
            }

            // Apply the divisor, deriving the exactness predicate when required.
            string idx = num!;
            if (expr.Divisor != 1)
            {
                if (expr.RequiresExactDivision)
                {
                    string rem = NextR(), pe = NextP();
                    L($"rem.s32 {rem}, {num}, {I(expr.Divisor)};");
                    L($"setp.eq.s32 {pe}, {rem}, 0;");
                    predicate = AndPred(predicate, pe);
                }
                idx = NextR();
                L($"div.s32 {idx}, {num}, {I(expr.Divisor)};");
            }

            // Derived bounds predicate: 0 <= idx < dim, emitted only when the folded
            // expression can ACTUALLY leave the tensor.
            //
            // Interval analysis over the parallel axis ranges, not a syntactic guess.
            // The syntactic form is both unsound and wasteful: after a reduction axis is
            // folded away, `oh + kh - 1` becomes `oh - 1`, `oh`, or `oh + 1` depending on
            // the tap, and only the first and last can escape [0, H). Testing the
            // pre-folding constant guards all three; testing the folded range guards
            // exactly the two that need it -- fewer instructions AND no missed case.
            long rangeLo = folded, rangeHi = folded;
            foreach (var term in symbolic)
            {
                long span = (long)term.Coefficient * (axesMeta[term.Axis].Extent - 1);
                if (term.Coefficient >= 0) rangeHi += span; else rangeLo += span;
            }
            bool canEscape = expr.Divisor != 1 || rangeLo < 0 || rangeHi >= dim;
            if (!canEscape) ElidedGuards++;
            if (canEscape)
            {
                string lo = NextP(), hi = NextP(), both = NextP();
                L($"setp.ge.s32 {lo}, {idx}, 0;");
                L($"setp.lt.s32 {hi}, {idx}, {I(dim)};");
                L($"and.pred {both}, {lo}, {hi};");
                predicate = AndPred(predicate, both);
            }

            // offset += idx * stride
            if (stride == 1)
            {
                if (offset is null) offset = idx;
                else { string s = NextR(); L($"add.s32 {s}, {offset}, {idx};"); offset = s; }
            }
            else
            {
                string scaled = NextR();
                if (offset is null) { L($"mul.lo.s32 {scaled}, {idx}, {I(stride)};"); offset = scaled; }
                else { L($"mad.lo.s32 {scaled}, {idx}, {I(stride)}, {offset};"); offset = scaled; }
            }
        }

        return offset!;
    }

    /// <summary>Emits a guarded load, yielding 0 for an out-of-range access (zero padding).</summary>
    private string EmitLoad(
        CodegenTensorBinding binding,
        string basePtr,
        string[] axisReg,
        int[] reductionValues,
        int[] reductionAxes,
        out string? predicate)
    {
        string offset = EmitOffset(binding, axisReg, reductionValues, reductionAxes, out predicate);
        string byteOff = NextRd(), addr = NextRd(), dst = NextF();
        L($"mul.wide.s32 {byteOff}, {offset}, 4;");
        L($"add.u64 {addr}, {basePtr}, {byteOff};");
        if (predicate is null)
        {
            L($"ld.global.nc.f32 {dst}, [{addr}];");
        }
        else
        {
            L($"mov.f32 {dst}, 0f00000000;");
            L($"@{predicate} ld.global.nc.f32 {dst}, [{addr}];");
        }
        EmittedLoads++;
        return dst;
    }

    private string? AndPred(string? a, string? b)
    {
        if (a is null) return b;
        if (b is null) return a;
        string c = NextP();
        L($"and.pred {c}, {a}, {b};");
        return c;
    }

    /// <summary>Blocks needed to cover the spec's iteration space at <see cref="BlockThreads"/>.</summary>
    public static uint GridBlocks(CodegenKernelSpec spec)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        long total = spec.Space.TotalThreads;      // same property the guard uses
        return (uint)((total + BlockThreads - 1) / BlockThreads);
    }
}
