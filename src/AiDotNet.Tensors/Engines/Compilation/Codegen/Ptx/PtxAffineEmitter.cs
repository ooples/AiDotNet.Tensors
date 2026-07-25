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

    /// <summary>Lanes actually used: the product of the chosen tile factors.</summary>
    public int CoarsenedLanes { get; private set; } = 1;

    /// <summary>Upper bound on accumulators per thread, to keep registers in budget.</summary>
    public int MaxTileLanes { get; set; } = 16;

    /// <summary>The tile the reuse analysis chose, e.g. <c>k x4, ow x4</c>.</summary>
    public string TileDescription { get; private set; } = "none";

    /// <summary>Largest grid the CUDA launch API accepts in the X dimension.</summary>
    private const long MaxGridBlocksX = 2147483647L;

    /// <summary>Bytes of kernel parameter space PTX guarantees.</summary>
    private const int MaxParameterBytes = 4096;

    /// <summary>
    /// Emitted global loads above which the kernel is refused rather than handed to
    /// ptxas, which would spend minutes on it before failing.
    /// </summary>
    private const int MaxEmittedLoads = 200_000;

    /// <summary>
    /// Refuses specs this emitter cannot lower correctly, with a message that says which
    /// bound was hit and what to change.
    /// </summary>
    /// <remarks>
    /// Every one of these used to be an implicit ceiling. Register declarations were
    /// fixed at <c>%p&lt;256&gt;</c>, written as a generous bound; coarsening reached
    /// <c>%p256</c> and ptxas reported "Arguments mismatch for instruction 'setp'",
    /// which describes an undeclared register in the language of a malformed
    /// instruction. The PTX version was pinned at 7.1 while the target was
    /// parameterised, so every sm_89 and sm_90 emission was invalid. The unroll limit
    /// refused a 288-trip convolution outright.
    ///
    /// Those three are now derived from the input. What remains are genuine hardware
    /// and format limits, and the rule for them is the same: assert loudly at the point
    /// of violation rather than emit something that fails confusingly later.
    /// </remarks>
    private void CheckLimits(CodegenKernelSpec spec, long blocks, long threadCount)
    {
        if (blocks > MaxGridBlocksX)
            throw new NotSupportedException(
                "Kernel '" + spec.Name + "' needs " + I(blocks) + " blocks, past the " +
                I(MaxGridBlocksX) + " the launch API accepts in X. Raise Coarsening or " +
                "BlockThreads, or give the spec a grid-stride loop.");

        if (threadCount <= 0)
            throw new NotSupportedException(
                "Kernel '" + spec.Name + "' resolved to " + I(threadCount) + " threads. " +
                "A tile factor larger than an axis extent would do this.");

        int parameterBytes = spec.ParameterCount * sizeof(long);
        if (parameterBytes > MaxParameterBytes)
            throw new NotSupportedException(
                "Kernel '" + spec.Name + "' declares " + I(spec.ParameterCount) +
                " pointer parameters (" + I(parameterBytes) + " bytes), past the " +
                I(MaxParameterBytes) + " bytes of PTX parameter space.");

    }

    /// <summary>Checked after emission, when the count actually exists.</summary>
    private void CheckEmittedSize(CodegenKernelSpec spec)
    {
        if (EmittedLoads > MaxEmittedLoads)
            throw new NotSupportedException(
                "Kernel '" + spec.Name + "' emitted " + I(EmittedLoads) + " global loads, " +
                "past the " + I(MaxEmittedLoads) + " this emitter will hand to ptxas. " +
                "Lower FullUnrollLimit or MaxTileLanes so more of the reduction loops.");
    }

    /// <summary>
    /// Chooses which axes to tile and by how much, by minimising loads per MAC.
    /// </summary>
    /// <remarks>
    /// For a tile with factors f over axes A, one trip issues, per operand, the product
    /// of the factors on the axes that operand DEPENDS on -- an operand invariant in a
    /// tiled axis costs nothing extra when that axis grows. MACs per trip is the product
    /// of all factors. So loads/MAC is
    /// <c>sum over operands of (product of dependent factors) / (product of all factors)</c>,
    /// and the best tile is simply the one that minimises it.
    /// </remarks>
    private static void SelectTile(
        CodegenKernelSpec spec, IReadOnlyList<CodegenAxis> axes, int[] parallel,
        int factor, int maxLanes, List<int> tileAxes, List<int> tileFactors)
    {
        int contiguous = parallel[parallel.Length - 1];

        // The contiguous axis is tiled whenever it divides: it is the only axis whose
        // lanes stay adjacent in memory, which is what keeps stores coalesced and makes
        // lane-vectorised loads legal.
        int contiguousFactor = axes[contiguous].Extent % factor == 0 ? factor : 1;

        // A SECOND TILED AXIS IS NOT FREE. It divides the thread count again, which
        // costs occupancy and latency hiding, so it is only worth taking when the load
        // saving is large enough to pay for that. Measured: adding one to the depthwise
        // family -- already at 93% of the DRAM roofline, where fewer loads cannot help
        // because the bytes are fixed -- moved it from 73 us to 100 us. Adding one to
        // dense 3x3, which is load-issue bound, moved it from 126 us to 69 us.
        //
        // So the decision is made on predicted TIME, not on loads per MAC, and only a
        // decisive win is taken. The 20% margin is deliberate slack for the two effects
        // the model does not carry: sector efficiency and occupancy.
        const double RequiredImprovement = 0.80;

        double baseline = PredictedRelativeTime(
            spec, axes, new[] { contiguous }, new[] { contiguousFactor });

        int bestAxis = -1, bestFactor = 1;
        double bestCost = baseline;

        foreach (int candidate in parallel)
        {
            if (candidate == contiguous) continue;
            if (axes[candidate].Extent % factor != 0) continue;
            if (contiguousFactor * factor > maxLanes) continue;

            double cost = PredictedRelativeTime(spec, axes,
                new[] { contiguous, candidate }, new[] { contiguousFactor, factor });
            if (cost < bestCost * RequiredImprovement)
            {
                bestCost = cost;
                bestAxis = candidate;
                bestFactor = factor;
            }
        }

        if (contiguousFactor > 1)
        {
            tileAxes.Add(contiguous);
            tileFactors.Add(contiguousFactor);
        }
        if (bestAxis >= 0)
        {
            tileAxes.Add(bestAxis);
            tileFactors.Add(bestFactor);
        }
    }

    /// <summary>
    /// Relative runtime of a candidate tile: the slowest of its load-issue, DRAM and
    /// compute constraints, in arbitrary but comparable units.
    /// </summary>
    /// <remarks>
    /// The same three-constraint model <c>CodegenPerformanceModel</c> uses, evaluated
    /// here on candidate tiles rather than on a finished kernel. Only the ratios between
    /// candidates matter, so device constants appear as a single ratio: how many bytes of
    /// DRAM traffic cost the same as one warp-level load instruction. On this class of
    /// device that is about 22 bytes (760 GB/s against 35.3 G warp-loads/s).
    /// </remarks>
    private static double PredictedRelativeTime(
        CodegenKernelSpec spec, IReadOnlyList<CodegenAxis> axes, int[] tileAxes, int[] factors)
    {
        const double BytesPerLoadInstruction = 22.0;
        const int WarpWidth = 32;

        long macs = spec.Output.ElementCount * Math.Max(1, spec.Space.ReductionTripCount);
        double loadInstructions = LoadsPerMac(spec, tileAxes, factors) * macs / WarpWidth;

        long bytes = spec.Output.ElementCount;
        for (int i = 0; i < spec.Inputs.Count; i++) bytes += spec.Inputs[i].ElementCount;
        double dramEquivalent = bytes * 4.0 / BytesPerLoadInstruction;

        // Compute never binds for these kernels, so the two memory terms decide it.
        return Math.Max(loadInstructions, dramEquivalent);
    }

    /// <summary>Loads per MAC for a candidate tile, from operand dependence alone.</summary>
    private static double LoadsPerMac(CodegenKernelSpec spec, int[] tileAxes, int[] factors)
    {
        double macs = 1;
        foreach (int f in factors) macs *= f;

        double loads = 0;
        foreach (int inputIndex in spec.ProductInputs)
        {
            double operand = 1;
            for (int t = 0; t < tileAxes.Length; t++)
                if (ReferencesAxis(spec.Inputs[inputIndex], tileAxes[t]))
                    operand *= factors[t];
            loads += operand;
        }
        return loads / macs;
    }

    private static string DescribeTile(
        IReadOnlyList<CodegenAxis> axes, List<int> tileAxes, List<int> tileFactors)
    {
        if (tileAxes.Count == 0) return "none";
        var parts = new List<string>();
        for (int t = 0; t < tileAxes.Count; t++)
            parts.Add(axes[tileAxes[t]].Name + " x" + tileFactors[t].ToString(CultureInfo.InvariantCulture));
        return string.Join(", ", parts);
    }

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

        // TILE, CHOSEN BY REUSE ANALYSIS RATHER THAN HARDCODED.
        //
        // One thread per output element re-loads every operand once per output. Which
        // axes can fix that is not a matter of taste -- it is written in the index maps:
        // an operand that does not reference an axis can share ONE load across every
        // position of it.
        //
        //   input  [n, c, oh+kh-1, ow+kw-1]   independent of k   -> tile K
        //   weights[k, c, kh, kw]             independent of ow  -> tile the spatial axis
        //
        // Tiling only the contiguous axis leaves the input per-lane, so loads/MAC is
        // (Tw+1)/Tw and can never beat 1.0 -- measured, and the reason spatial coarsening
        // plateaued. Tiling BOTH gives (Tw+Tk)/(Tw*Tk), which breaks below 1.0. That is
        // what turns the measured 64x input-load redundancy in dense convolution into
        // reuse instead of traffic.
        var tileAxes = new List<int>();
        var tileFactors = new List<int>();
        if (Coarsening > 1 && parallel.Length > 0)
            SelectTile(spec, axes, parallel, Coarsening, MaxTileLanes, tileAxes, tileFactors);

        int lanes = 1;
        foreach (int f in tileFactors) lanes *= f;
        CoarsenedLanes = lanes;
        TileDescription = DescribeTile(axes, tileAxes, tileFactors);

        // The contiguous axis, when tiled, is the one lane-vectorisation can use.
        int coarsenAxis = tileAxes.Count > 0 && tileAxes[0] == parallel[parallel.Length - 1]
            ? tileAxes[0]
            : -1;
        int contiguousFactor = coarsenAxis >= 0 ? tileFactors[0] : 1;

        // Per-lane offsets on each tiled axis, last tile axis varying fastest.
        var laneOffsets = new int[lanes][];
        for (int l = 0; l < lanes; l++)
        {
            laneOffsets[l] = new int[tileAxes.Count];
            int remaining = l;
            for (int t = tileAxes.Count - 1; t >= 0; t--)
            {
                laneOffsets[l][t] = remaining % tileFactors[t];
                remaining /= tileFactors[t];
            }
        }

        // Threads, grid and in-kernel guard all come from this one number, which is
        // the invariant the whole IR exists to protect.
        long threadCount = total / lanes;
        long blocks = (threadCount + BlockThreads - 1) / BlockThreads;
        CheckLimits(spec, blocks, threadCount);
        LaunchBlocks = (uint)blocks;

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
            int tileSlot = tileAxes.IndexOf(ax);
            int extent = tileSlot >= 0 ? axes[ax].Extent / tileFactors[tileSlot] : axes[ax].Extent;
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

        // One axis-register view per lane. Only the tiled axes differ between them:
        // on tiled axis t, lane l covers position base_t * factor_t + offset_t(l).
        // Registers are shared between lanes that agree on an axis, so a 4x4 tile emits
        // 8 index registers rather than 16.
        var laneAxisReg = new string[lanes][];
        var tileAxisReg = new string[tileAxes.Count][];
        for (int t = 0; t < tileAxes.Count; t++)
        {
            tileAxisReg[t] = new string[tileFactors[t]];
            for (int off = 0; off < tileFactors[t]; off++)
            {
                string reg = NextR();
                if (off == 0) L($"mul.lo.u32 {reg}, {axisReg[tileAxes[t]]}, {I(tileFactors[t])};   // {axes[tileAxes[t]].Name} tile");
                else L($"mad.lo.u32 {reg}, {axisReg[tileAxes[t]]}, {I(tileFactors[t])}, {I(off)};");
                tileAxisReg[t][off] = reg;
            }
        }
        for (int l = 0; l < lanes; l++)
        {
            laneAxisReg[l] = (string[])axisReg.Clone();
            for (int t = 0; t < tileAxes.Count; t++)
                laneAxisReg[l][tileAxes[t]] = tileAxisReg[t][laneOffsets[l][t]];
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

        // TWO SCOPES, because the two vector forms have different lifetimes.
        // A reduction-axis vector covers VectorWidth TRIPS, so it must outlive a trip.
        // A lane vector's address depends on the reduction values, so it must NOT --
        // sharing one cache let trip 1 reuse trip 0's vector and silently corrupted
        // both 1x1 kernels while their neighbours stayed exact.
        var reductionVectorCache = new Dictionary<string, string[]>(StringComparer.Ordinal);

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
            // LOAD ONCE PER DISTINCT DEPENDENCE.
            //
            // This single rule replaces the two special cases it grew out of. An operand
            // is loaded once for each distinct combination of the tile-axis offsets it
            // ACTUALLY REFERENCES:
            //
            //   invariant in every tiled axis  -> 1 load for the whole tile
            //   depends only on the spatial    -> Tw loads
            //   depends only on the reuse axis -> Tk loads
            //
            // which is exactly (Tw + Tk)/(Tw*Tk) loads per MAC. An operand that is
            // unit-stride in the contiguous axis has its Tw values at consecutive
            // aligned addresses, so those collapse further into one ld.global.v4.f32.
            var scalarCache = new Dictionary<string, (string Value, string? Pred)>(StringComparer.Ordinal);
            var laneVectorCache = new Dictionary<string, string[]>(StringComparer.Ordinal);

            for (int l = 0; l < lanes; l++)
            {
                string? product = null;
                string? productPred = null;
                foreach (int inputIdx in spec.ProductInputs)
                {
                    var binding = spec.Inputs[inputIdx];
                    string value;
                    string? pred = null;

                    bool laneVectorisable =
                        EnableVectorLoads && coarsenAxis >= 0 &&
                        contiguousFactor == VectorWidth &&
                        IsUnitStrideIn(binding, coarsenAxis, axes);

                    if (laneVectorisable)
                    {
                        // Key on every referenced tiled axis EXCEPT the contiguous one,
                        // whose Tw positions the vector itself covers.
                        string key = LoadKey(inputIdx, binding, tileAxes, laneOffsets[l], coarsenAxis);
                        if (!laneVectorCache.TryGetValue(key, out string[]? vec))
                        {
                            vec = EmitVectorLoad(binding, basePtr[binding.ParameterIndex],
                                                 laneAxisReg[BaseLaneFor(l, tileAxes, laneOffsets, coarsenAxis)],
                                                 reductionValues, reduction, VectorWidth);
                            laneVectorCache[key] = vec;
                        }
                        int slot = tileAxes.IndexOf(coarsenAxis);
                        value = vec[laneOffsets[l][slot]];
                    }
                    else if (vectorGroup > 1 && innermost >= 0 && vectorisable[inputIdx])
                    {
                        // Vector load along the innermost REDUCTION axis. This was
                        // emitted once from the untiled base registers, which made it
                        // lane-unaware: weights[k, c] is unit-stride in c, so it took
                        // this path, and once k was also tiled every lane received lane
                        // zero's k. Keying it like every other load fixes that.
                        int group = reductionValues[innermost] / vectorGroup;
                        string key = LoadKey(inputIdx, binding, tileAxes, laneOffsets[l], -1) +
                                     "#g" + group.ToString(CultureInfo.InvariantCulture);
                        if (!reductionVectorCache.TryGetValue(key, out string[]? rvec))
                        {
                            rvec = EmitVectorLoad(binding, basePtr[binding.ParameterIndex],
                                                  laneAxisReg[l], reductionValues, reduction, vectorGroup);
                            reductionVectorCache[key] = rvec;
                        }
                        value = rvec[reductionValues[innermost] % vectorGroup];
                    }
                    else
                    {
                        // Scalar path, cached on every referenced tiled axis. Two lanes
                        // that agree on the axes this operand reads share one load.
                        string key = LoadKey(inputIdx, binding, tileAxes, laneOffsets[l], -1);
                        if (scalarCache.TryGetValue(key, out var cached))
                        {
                            value = cached.Value;
                            pred = cached.Pred;
                        }
                        else
                        {
                            value = EmitLoad(binding, basePtr[binding.ParameterIndex], laneAxisReg[l],
                                             reductionValues, reduction, out pred);
                            // GUARDED loads are cached too, together with their predicate.
                            // Refusing to cache them defeated the entire point on the one
                            // kernel that matters: dense convolution's input is a gathered
                            // window and therefore always guarded, so it was re-loaded once
                            // per lane and loads/MAC stayed at 1.167 instead of 0.5. The
                            // predicate is derived from the index, and two lanes with the
                            // same key have the same index, so they have the same predicate.
                            scalarCache[key] = (value, pred);
                        }
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
        // The epilogue caches on the SAME key as the body. Hoisting it to lane 0 whenever
        // it did not reference the contiguous axis was correct while only that axis was
        // tiled, and silently wrong the moment a second axis was: bias[k] does not
        // reference `ow`, so every lane received lane 0's `k` and two kernels returned
        // wrong values while their epilogue-free siblings stayed exact.
        var epilogueCache = new Dictionary<string, string>(StringComparer.Ordinal);

        for (int l = 0; l < lanes; l++)
        {
            string acc = accs[l];
            if (spec.BiasInput.HasValue)
            {
                var b = spec.Inputs[spec.BiasInput.Value];
                string key = LoadKey(spec.BiasInput.Value, b, tileAxes, laneOffsets[l], -1);
                if (!epilogueCache.TryGetValue(key, out string? v))
                {
                    v = EmitLoad(b, basePtr[b.ParameterIndex], laneAxisReg[l],
                                 reductionValues, reduction, out _);
                    epilogueCache[key] = v;
                }
                string na = NextF();
                L($"add.rn.f32 {na}, {acc}, {v};");
                acc = na;
            }
            if (spec.ScaleInput.HasValue)
            {
                var s = spec.Inputs[spec.ScaleInput.Value];
                string key = LoadKey(spec.ScaleInput.Value, s, tileAxes, laneOffsets[l], -1);
                if (!epilogueCache.TryGetValue(key, out string? v))
                {
                    v = EmitLoad(s, basePtr[s.ParameterIndex], laneAxisReg[l],
                                 reductionValues, reduction, out _);
                    epilogueCache[key] = v;
                }
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

        CheckEmittedSize(spec);
        _body.Append("END:\n    ret;\n}\n");

        _sb.Append("    .reg .pred %p<").Append(I(_p + 8)).Append(">;\n")
           .Append("    .reg .b32 %r<").Append(I(_r + 8)).Append(">;\n")
           .Append("    .reg .b64 %rd<").Append(I(_rd + 8)).Append(">;\n")
           .Append("    .reg .f32 %f<").Append(I(_f + 8)).Append(">;\n")
           .Append(_body);
        return _sb.ToString();
    }

    /// <summary>
    /// Identity of a load: the operand plus its offsets on the tiled axes it actually
    /// reads. Two lanes producing the same key need the same value, so one load serves
    /// both -- this is the mechanism that turns reuse analysis into fewer instructions.
    /// </summary>
    private static string LoadKey(
        int inputIndex, CodegenTensorBinding binding, List<int> tileAxes,
        int[] laneOffsets, int excludeAxis)
    {
        var sb = new StringBuilder();
        sb.Append(inputIndex.ToString(CultureInfo.InvariantCulture));
        for (int t = 0; t < tileAxes.Count; t++)
        {
            if (tileAxes[t] == excludeAxis) continue;
            if (!ReferencesAxis(binding, tileAxes[t])) continue;
            sb.Append(':').Append(t.ToString(CultureInfo.InvariantCulture))
              .Append('=').Append(laneOffsets[t].ToString(CultureInfo.InvariantCulture));
        }
        return sb.ToString();
    }

    /// <summary>
    /// The lane that shares <paramref name="lane"/>'s offsets on every tiled axis except
    /// <paramref name="contiguousAxis"/>, where it sits at offset zero. A vector load is
    /// emitted from there so its four components line up with offsets 0..3.
    /// </summary>
    private static int BaseLaneFor(
        int lane, List<int> tileAxes, int[][] laneOffsets, int contiguousAxis)
    {
        int slot = tileAxes.IndexOf(contiguousAxis);
        if (slot < 0) return lane;
        for (int candidate = 0; candidate < laneOffsets.Length; candidate++)
        {
            bool match = laneOffsets[candidate][slot] == 0;
            for (int t = 0; match && t < tileAxes.Count; t++)
                if (t != slot && laneOffsets[candidate][t] != laneOffsets[lane][t]) match = false;
            if (match) return candidate;
        }
        return lane;
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
