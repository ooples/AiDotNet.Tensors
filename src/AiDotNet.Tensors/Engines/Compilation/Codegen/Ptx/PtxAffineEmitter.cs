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
public sealed partial class PtxAffineEmitter
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

    /// <summary>Base-pointer register per kernel parameter, for the current emission.</summary>
    private string[]? _basePointers;

    /// <summary>Input bindings of the spec being emitted, for the current emission.</summary>
    private IReadOnlyList<CodegenTensorBinding>? _inputBindings;

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

    /// <summary>Index loads emitted for data-dependent (gather/scatter) dimensions.</summary>
    public int IndirectIndexLoads { get; private set; }

    /// <summary>Stores emitted as atomic accumulations because the destination may collide.</summary>
    public int AtomicStores { get; private set; }

    /// <summary>Extra-output stores emitted beyond the primary and any argmax.</summary>
    public int ExtraOutputStores { get; private set; }

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

    /// <summary>
    /// Upper bound on accumulators per thread: a REGISTER budget, enforced because the
    /// cost model cannot see register pressure.
    /// </summary>
    /// <remarks>
    /// 16 is measured, not assumed. Raising it let the search pick a 4x8 tile for dense
    /// 3x3, which is genuinely better on the model's terms -- loads/MAC 0.501 -> 0.376 --
    /// and worse on the machine:
    ///
    ///   tile   loads/MAC   registers   blocks   time      run spread
    ///   4x4      0.501         56         98    ~70 us       2-6%
    ///   4x8      0.376        168         49    71.1 us    202.9%
    ///
    /// At 168 registers only about 1.5 blocks fit per SM out of the 65,536 registers an
    /// SM has, so occupancy collapses no matter how many blocks the grid contains, and
    /// the kernel becomes too unstable to measure. The model's occupancy term counts
    /// BLOCKS and is blind to this; a register-pressure term is the next thing it needs.
    ///
    /// The lever that lowers loads/MAC further WITHOUT spending registers is shared
    /// memory, which every kernel here still reports as completely unused (LDS 0/STS 0).
    /// </remarks>
    public int MaxTileLanes { get; set; } = 16;

    /// <summary>The tile the reuse analysis chose, e.g. <c>k x4, ow x4</c>.</summary>
    public string TileDescription { get; private set; } = "none";

    /// <summary>Threads per block this kernel actually launches with.</summary>
    /// <remarks>
    /// Derived per kernel rather than fixed at 256, because shared-memory staging needs
    /// a block to cover exactly one group of the axes the staged operand depends on. A
    /// block that straddles two output-channel groups cannot stage weights: half its
    /// threads would want a different slice.
    /// </remarks>
    public int LaunchBlockThreads { get; private set; } = BlockThreads;

    /// <summary>Bytes of shared memory the launch must reserve.</summary>
    public int SharedMemoryBytes { get; private set; }

    /// <summary>Block width, over the contiguous tile axis.</summary>
    public int LaunchBlockX { get; private set; } = BlockThreads;

    /// <summary>Block height, over the reuse tile axis. 1 for the flat lowering.</summary>
    public int LaunchBlockY { get; private set; } = 1;

    /// <summary>
    /// Selects the two-dimensional lowering, the prerequisite for staging the ACTIVATION
    /// operand. OFF by default: the lowering is correct but not yet profitable.
    /// </summary>
    /// <remarks>
    /// Measured on dense 3x3: the flat lowering with staged weights runs 61.5 us; the
    /// two-dimensional lowering runs 74.3 us. Two reasons, both addressable:
    ///
    /// 1. It gives up weight staging. Under a flat block every thread shares one
    ///    reuse-axis group, so the weights are block-invariant. Under 2D, y varies over
    ///    that axis, so each row needs its own slice -- each operand is invariant in one
    ///    DIMENSION, not in the block, and staging must be indexed by the dimension the
    ///    operand varies in. Staging the block-invariant way under 2D is not merely
    ///    unprofitable, it is wrong: it returned 5.277 and 1.112e1 instead of zero.
    ///
    /// 2. The block is warp-ragged. 7 tiles of ow by 16 tiles of k is 112 threads, which
    ///    is 3.5 warps, so every block wastes half a warp. Padding x to 8 with a guard
    ///    would make it 128, exactly 4 warps.
    ///
    /// With per-dimension staging the arithmetic predicts 0.0078 loads/MAC -- below
    /// cuDNN's measured ~0.03 -- and an LDS-bound ~15 us against cuDNN's 41.0. The
    /// lowering here is the scaffolding for that; it is left off until it earns its way.
    /// </remarks>
    public bool EnableInputStaging { get; set; }

    /// <summary>True when the emitted kernel used the two-dimensional lowering.</summary>
    public bool UsedTwoDimensionalBlock { get; private set; }

    /// <summary>
    /// The activation operand is invariant in the reuse axis, so threads that differ
    /// only in that axis want the SAME input. A flat block cannot exploit that: its
    /// threads walk the spatial axes at one value of the reuse axis, which is the
    /// opposite arrangement. A two-dimensional block fixes it -- x over the contiguous
    /// axis so stores stay coalesced, y over the reuse axis so a staged input row is
    /// shared by every thread in the column.
    /// </summary>
    private static bool CanStageInput(
        CodegenKernelSpec spec, IReadOnlyList<CodegenAxis> axes,
        List<int> tileAxes, List<int> tileFactors, int dataInput,
        out int blockX, out int blockY)
    {
        blockX = 0;
        blockY = 0;
        if (tileAxes.Count < 2) return false;

        int contiguousAxis = tileAxes[0];
        int reuseAxis = tileAxes[1];

        // The data operand must be invariant in the reuse axis -- that invariance IS
        // the reuse -- and must vary along the contiguous one, or there is nothing to
        // stage per column.
        if (ReferencesAxis(spec.Inputs[dataInput], reuseAxis)) return false;
        if (!ReferencesAxis(spec.Inputs[dataInput], contiguousAxis)) return false;

        long x = axes[contiguousAxis].Extent / tileFactors[0];
        long y = axes[reuseAxis].Extent / tileFactors[1];

        // Require the block to cover BOTH tiled axes completely. A partial cover would
        // need the staged tile to carry a base offset, and the halo maths to follow it;
        // full cover keeps the staged row addressed from zero.
        if (x * y > MaxBlockThreads || x * y < 32) return false;
        if (x <= 0 || y <= 1) return false;

        blockX = (int)x;
        blockY = (int)y;
        return true;
    }

    /// <summary>Operand indices staged in shared memory, for reporting.</summary>
    public string StagedOperands { get; private set; } = "none";

    /// <summary>
    /// Enables shared-memory staging. Off restores the fixed 256-thread block and no
    /// staging, so the two lowerings can be compared inside one process.
    /// </summary>
    public bool EnableSharedStaging { get; set; } = true;

    /// <summary>
    /// Cooperatively loads a block-invariant operand into shared memory, once per
    /// strip-mine iteration, and returns the register holding the shared base address.
    /// </summary>
    /// <remarks>
    /// The block is sized so every thread in it agrees on the axes this operand is
    /// indexed by, so all of them want the SAME slice. Measured, 196 threads were each
    /// fetching the same weight, so the slice is fetched once by the first
    /// <c>count</c> threads and read from shared memory thereafter.
    ///
    /// The staged element for (lane offset on the tiled axis, unrolled trip) sits at a
    /// COMPILE-TIME index, so every consumer becomes a constant-offset
    /// <c>ld.shared.f32</c> -- no address arithmetic at the point of use.
    /// </remarks>
    /// <summary>One operand resident in shared memory for the current strip-mine step.</summary>
    /// <param name="Input">Index into the spec's inputs.</param>
    /// <param name="TileSlot">Tile axis the operand varies on, or -1 when block-invariant.</param>
    /// <param name="Dimension">
    /// Block dimension that indexes the slice: 0 for x, 1 for y, or -1 when the slice is
    /// block-invariant and every consumer reads a compile-time constant offset.
    /// </param>
    /// <param name="Count">Elements in the slice.</param>
    /// <param name="ByteOffset">Where the slice starts inside the shared buffer.</param>
    private sealed record StagedSlice(int Input, int TileSlot, int Dimension, int Count, int ByteOffset);

    /// <summary>The slice holding an operand, or null when it is not staged.</summary>
    private static StagedSlice? SliceFor(List<StagedSlice> slices, int input)
    {
        foreach (var slice in slices) if (slice.Input == input) return slice;
        return null;
    }

    private void EmitStageLoad(
        CodegenTensorBinding binding, string basePointer, string sharedBase,
        string tid, int tiledSlot, string tileBaseReg, int tileFactor,
        int[] innerReduction, IReadOnlyList<CodegenAxis> axes, string[] axisRegTemplate,
        int count, int trips, bool coversAxisFromZero = false, int blockThreads = 0)
    {
        // A SLICE CAN BE LARGER THAN THE BLOCK. Requiring one pass per fetch rejected
        // exactly the staging that matters: on dense 3x3 the activation slice is
        // 28 positions x 9 trips = 252 elements against 112 threads, so the one-pass rule
        // refused it and the 2D lowering ran with nothing staged -- which is worse than
        // flat, because it also gives up the block-invariant weight staging (loads/MAC rose
        // from 0.258 to 0.501).
        //
        // The passes are compile-time, so they unroll. Only the last needs a bounds guard.
        if (blockThreads <= 0) blockThreads = count;
        int passes = (count + blockThreads - 1) / blockThreads;
        for (int pass = 0; pass < passes; pass++)
            EmitStagePass(binding, basePointer, sharedBase, tid, tiledSlot, tileBaseReg,
                tileFactor, innerReduction, axes, axisRegTemplate, count, trips,
                coversAxisFromZero, pass * blockThreads);

        L("bar.sync 0;");
    }

    private void EmitStagePass(
        CodegenTensorBinding binding, string basePointer, string sharedBase,
        string tid, int tiledSlot, string tileBaseReg, int tileFactor,
        int[] innerReduction, IReadOnlyList<CodegenAxis> axes, string[] axisRegTemplate,
        int count, int trips, bool coversAxisFromZero, int indexBase)
    {
        string skip = "STAGE_SKIP_" + I(_stageLabel++);

        // The flat index into the slice for this pass.
        string index = tid;
        if (indexBase != 0)
        {
            index = NextR();
            L($"add.u32 {index}, {tid}, {I(indexBase)};");
        }

        string active = NextP();
        L($"setp.ge.u32 {active}, {index}, {I(count)};");
        L($"@{active} bra {skip};");

        // Split the flat stage index into (lane offset on the tiled axis, trip).
        var runtime = (string[])axisRegTemplate.Clone();
        string laneOff = NextR(), trip = NextR();
        L($"div.u32 {laneOff}, {index}, {I(trips)};");
        L($"rem.u32 {trip}, {index}, {I(trips)};");

        if (tiledSlot >= 0 && coversAxisFromZero)
        {
            // PER-DIMENSION SLICE. The block covers this axis completely, so the slice
            // spans the whole extent and the flat index IS the axis coordinate -- no block
            // base to add. That full cover is exactly what CanStageInput requires, and it
            // is what keeps this addressed from zero.
            runtime[tiledSlot] = laneOff;
        }
        else if (tiledSlot >= 0 && tileBaseReg != null)
        {
            string tiledValue = NextR();
            L($"mad.lo.u32 {tiledValue}, {tileBaseReg}, {I(tileFactor)}, {laneOff};");
            runtime[tiledSlot] = tiledValue;
        }

        // Unpack the trip into the unrolled reduction axes, last-declared fastest --
        // the same order the unrolled body enumerates them in.
        string rest = trip;
        for (int i = innerReduction.Length - 1; i >= 0; i--)
        {
            int ax = innerReduction[i];
            int extent = axes[ax].Extent;
            string value = NextR();
            if (i == 0)
            {
                L($"mov.u32 {value}, {rest};");
            }
            else
            {
                L($"rem.u32 {value}, {rest}, {I(extent)};");
                string next = NextR();
                L($"div.u32 {next}, {rest}, {I(extent)};");
                rest = next;
            }
            runtime[ax] = value;
        }

        // Every axis is a register now, so nothing is folded: pass an empty reduction
        // set and let the normal offset path build it symbolically.
        string offset = EmitOffset(binding, runtime, new int[axes.Count],
                                   Array.Empty<int>(), out string? pred);
        string byteOffset = NextRd(), address = NextRd(), value2 = NextF();
        L($"mul.wide.u32 {byteOffset}, {offset}, 4;");
        L($"add.u64 {address}, {basePointer}, {byteOffset};");
        L($"mov.f32 {value2}, 0f00000000;");
        if (pred is null) L($"ld.global.nc.f32 {value2}, [{address}];");
        else L($"@{pred} ld.global.nc.f32 {value2}, [{address}];");
        EmittedLoads++;

        string sharedByte = NextRd(), sharedAddr = NextRd();
        L($"mul.wide.u32 {sharedByte}, {index}, 4;");
        L($"add.u64 {sharedAddr}, {sharedBase}, {sharedByte};");
        L($"st.shared.f32 [{sharedAddr}], {value2};");

        _body.Append(skip).Append(":\n");
    }

    /// <summary>
    /// Emits the epilogue activation, returning the register holding the result.
    /// </summary>
    /// <remarks>
    /// <para>
    /// PTX has no <c>exp</c>: it has <c>ex2.approx.f32</c>, so every exponential is
    /// <c>ex2(x * log2 e)</c> with <c>log2 e = 1.4426950408889634</c>. Reciprocals use
    /// <c>rcp.approx.f32</c> and <c>tanh.approx.f32</c> is native from sm_75, which this
    /// emitter already requires.
    /// </para>
    /// <para>
    /// These are APPROXIMATE instructions — roughly 2 ulp for <c>ex2</c>, 1 ulp for
    /// <c>rcp</c> — so a kernel carrying one of them cannot reach the exact
    /// <c>0.000E+000</c> the affine kernels hit against the fp64 oracle. The deviation is
    /// measured and reported per activation rather than assumed; see
    /// <c>docs/ACTIVATION_ACCURACY.md</c>. ReLU stays exact because <c>max</c> is exact.
    /// </para>
    /// </remarks>
    private string EmitActivation(CodegenActivationKind kind, string x)
    {
        const string Log2E = "0f3FB8AA3B";          // 1.4426950408889634
        const string One = "0f3F800000";
        const string Half = "0f3F000000";
        const string GeluOuter = "0f3F4C422A";      // sqrt(2/pi) = 0.7978845608028654
        const string GeluInner = "0f3D372713";      // 0.044715

        switch (kind)
        {
            case CodegenActivationKind.None:
                return x;

            case CodegenActivationKind.ReLU:
                L($"max.f32 {x}, {x}, 0f00000000;");
                return x;

            case CodegenActivationKind.Tanh:
                return EmitTanh(x, Log2E, One);

            case CodegenActivationKind.Reciprocal:
            {
                string r = NextF();
                L($"rcp.approx.f32 {r}, {x};");
                return r;
            }

            case CodegenActivationKind.Rsqrt:
            {
                string r = NextF();
                L($"rsqrt.approx.f32 {r}, {x};");
                return r;
            }

            case CodegenActivationKind.Sigmoid:
                return EmitSigmoid(x, Log2E, One);

            case CodegenActivationKind.Swish:
            {
                string s = EmitSigmoid(x, Log2E, One);
                string r = NextF();
                L($"mul.rn.f32 {r}, {x}, {s};");
                return r;
            }

            case CodegenActivationKind.Gelu:
            {
                // 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))
                string x2 = NextF(), x3 = NextF(), inner = NextF(), scaled = NextF();
                L($"mul.rn.f32 {x2}, {x}, {x};");
                L($"mul.rn.f32 {x3}, {x2}, {x};");
                L($"fma.rn.f32 {inner}, {x3}, {GeluInner}, {x};");
                L($"mul.rn.f32 {scaled}, {inner}, {GeluOuter};");

                string t = EmitTanh(scaled, Log2E, One);
                string sum = NextF(), half = NextF(), r = NextF();
                L($"add.rn.f32 {sum}, {t}, {One};");
                L($"mul.rn.f32 {half}, {x}, {Half};");
                L($"mul.rn.f32 {r}, {half}, {sum};");
                return r;
            }

            default:
                throw new NotSupportedException(
                    "The PTX emitter has no form for activation " + kind + ".");
        }
    }

    /// <summary>
    /// tanh, built from <c>ex2</c> rather than from <c>tanh.approx.f32</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measured on device against the fp64 oracle, the native instruction is the weakest
    /// link in the whole activation set:
    /// </para>
    /// <code>
    ///   sigmoid  ex2 + rcp            9.574E-008
    ///   swish    ex2 + rcp            8.450E-008
    ///   tanh     tanh.approx.f32      7.268E-006     <- 76x worse
    ///   gelu     tanh.approx.f32      2.437E-006
    /// </code>
    /// <para>
    /// So tanh is derived from the same primitives the accurate two are:
    /// <c>tanh(x) = sign(x) * (1 - 2 / (exp(2|x|) + 1))</c>. Taking the absolute value
    /// first is what makes it stable — for large |x| the exponential saturates to
    /// infinity, the reciprocal underflows to zero, and the result lands on exactly
    /// ±1 with no special case. Evaluating the raw ratio instead would produce inf/inf.
    /// </para>
    /// </remarks>
    private string EmitTanh(string x, string log2E, string one)
    {
        const string Two = "0f40000000";

        string ax = NextF(), scaled = NextF(), e = NextF();
        L($"abs.f32 {ax}, {x};");
        L($"mul.rn.f32 {scaled}, {ax}, {log2E};");
        L($"add.rn.f32 {scaled}, {scaled}, {scaled};");     // 2|x| log2 e
        L($"ex2.approx.f32 {e}, {scaled};");

        string denom = NextF(), inv = NextF(), twice = NextF(), magnitude = NextF(), r = NextF();
        L($"add.rn.f32 {denom}, {e}, {one};");
        L($"rcp.approx.f32 {inv}, {denom};");
        L($"mul.rn.f32 {twice}, {inv}, {Two};");
        L($"sub.rn.f32 {magnitude}, {one}, {twice};");
        L($"copysign.f32 {r}, {x}, {magnitude};");
        return r;
    }

    /// <summary>
    /// A double as a PTX fp32 hex literal, which is how PTX spells a float constant.
    /// </summary>
    /// <remarks>
    /// Emitting the decimal form would let ptxas re-parse and re-round the value; the hex
    /// form names the exact bit pattern, so the constant in the kernel is the constant the
    /// oracle used.
    /// </remarks>
    private static string F32(double value)
    {
        // GetBytes/ToUInt32 rather than SingleToUInt32Bits, which does not exist on
        // net471 -- and this project builds every target framework.
        uint bits = BitConverter.ToUInt32(BitConverter.GetBytes((float)value), 0);
        return "0f" + bits.ToString("X8", CultureInfo.InvariantCulture);
    }

    /// <summary>1 / (1 + exp(-x)), via ex2 and rcp.</summary>
    private string EmitSigmoid(string x, string log2E, string one)
    {
        string negScaled = NextF(), e = NextF(), denom = NextF(), r = NextF();
        L($"mul.rn.f32 {negScaled}, {x}, {log2E};");
        L($"neg.f32 {negScaled}, {negScaled};");
        L($"ex2.approx.f32 {e}, {negScaled};");
        L($"add.rn.f32 {denom}, {e}, {one};");
        L($"rcp.approx.f32 {r}, {denom};");
        return r;
    }

    private int _stageLabel;

    /// <summary>
    /// PTX literal an out-of-range load yields: the identity of the active reduction.
    /// </summary>
    /// <remarks>
    /// <c>0f00000000</c> for a sum, <c>0fFF800000</c> (negative infinity) for a maximum.
    /// Zero is the ADDITIVE identity; under a maximum it is a real candidate, so a padded
    /// max-pool over all-negative inputs returned 0 instead of the true maximum -- in the
    /// oracle and the emitter alike, which is why the agreement gate passed it.
    /// </remarks>
    private string _outOfRangeFill = "0f00000000";

    /// <summary>Fallback when staging is disabled: fixed block, nothing invariant.</summary>
    private static int PassThroughBlock(out HashSet<int> axesVaryingInBlock)
    {
        axesVaryingInBlock = new HashSet<int>();
        return BlockThreads;
    }

    /// <summary>Largest block CUDA accepts.</summary>
    private const int MaxBlockThreads = 1024;

    /// <summary>Block size the derivation aims for; 256 matches the rest of the catalog.</summary>
    private const int TargetBlockThreads = 256;

    /// <summary>
    /// Minimum share of a kernel's loads the staged operand must carry for staging to be
    /// worth its two barriers per step. Measured: at 50% it turned 68.1 us into 61.9;
    /// at 6% it turned 104 us into 131.4.
    /// </summary>
    private const double MinimumStagedShare = 0.25;

    /// <summary>Shared memory a block may use without cutting occupancy hard on sm_86.</summary>
    private const int MaxSharedBytes = 32 * 1024;

    /// <summary>
    /// Chooses a block size whose threads all agree on the slow axes, and reports which
    /// axes vary inside a block.
    /// </summary>
    /// <remarks>
    /// The decomposition assigns the last parallel axis fastest. If a block's thread
    /// count is exactly the product of the fastest axes' extents, then every thread in
    /// the block shares the same value of every slower axis -- which is precisely the
    /// condition under which an operand indexed only by slower axes is constant across
    /// the block and can be fetched once into shared memory instead of once per thread.
    ///
    /// For dense 3x3 this gives 28 x 7 = 196 threads covering all of oh and ow at one
    /// output-channel group, and the measurement that motivates it is that those 196
    /// threads currently fetch the same weight 196 times.
    /// </remarks>
    private static int ChooseBlockThreads(
        IReadOnlyList<CodegenAxis> axes, int[] parallel, List<int> tileAxes, List<int> tileFactors,
        out HashSet<int> axesVaryingInBlock)
    {
        axesVaryingInBlock = new HashSet<int>();
        long threads = 1;

        for (int p = parallel.Length - 1; p >= 0; p--)
        {
            int ax = parallel[p];
            int slot = tileAxes.IndexOf(ax);
            int extent = slot >= 0 ? axes[ax].Extent / tileFactors[slot] : axes[ax].Extent;

            if (threads * extent <= TargetBlockThreads)
            {
                threads *= extent;
                axesVaryingInBlock.Add(ax);
                continue;
            }

            // This axis would overshoot. Taking PART of it is fine as long as the block
            // still covers whole groups of every slower axis, which it does: a partial
            // take only subdivides this axis. Only do it when the partial factor divides
            // the extent, so the decomposition stays exact.
            long room = TargetBlockThreads / threads;
            for (long take = room; take >= 2; take--)
            {
                if (extent % take != 0) continue;
                threads *= take;
                axesVaryingInBlock.Add(ax);
                break;
            }
            break;
        }

        // Too small to be a sensible block: fall back to the fixed size, in which case
        // nothing is treated as block-invariant and no staging happens.
        if (threads < 32)
        {
            axesVaryingInBlock.Clear();
            return BlockThreads;
        }
        return (int)threads;
    }

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

        // SEARCH THE TILE, DO NOT ASSUME IT.
        //
        // This used to try exactly one factor per axis, so 4x4 was the only 2D tile it
        // could construct and raising the lane budget changed nothing. Dense convolution
        // sat at 0.501 loads/MAC while cuDNN reaches roughly 0.03, and the remaining
        // factor of ~16 is available in the tile: (Tw+Tk)/(Tw*Tk) keeps falling as the
        // tile grows. So enumerate factors on both axes and let the cost model choose.
        //
        // The model now carries an occupancy term, which is what makes the search safe:
        // a bigger tile divides the thread count, and past a point the occupancy loss
        // outweighs the load saving. That trade is exactly what the model expresses, so
        // the search finds the turning point instead of running away to a tile that
        // issues almost no loads and leaves the machine idle.
        // PREDICATE-HEAVY OPERANDS COST MORE REGISTERS PER LANE. An exact-division index
        // map (a transposed convolution's (ih + pad - kh)/stride) emits a remainder, a
        // comparison and a predicate for every load, so registers grow much faster with
        // the tile than for a plain gather. Measured: letting the search take 16 lanes
        // for the transposed kernel produced 126 registers and 129.0 us against 40
        // registers and ~104 us at 4 lanes -- better on the model's terms, worse on the
        // machine, exactly like the 4x8 dense case.
        //
        // The cost model has no register term, so express the constraint where it is
        // visible: in the index maps.
        bool predicated = false;
        foreach (int inputIndex in spec.ProductInputs)
            foreach (var expr in spec.Inputs[inputIndex].Map)
                if (expr.Divisor != 1) predicated = true;

        int laneCeiling = predicated ? Math.Min(maxLanes, 4) : maxLanes;
        var factors = new[] { 1, 2, 4, 8, 16 };

        int bestPrimary = 1, bestSecondAxis = -1, bestSecondFactor = 1;
        double bestCost = double.MaxValue;

        // Descending, so that on a TIE the larger contiguous factor wins. Ties are
        // common -- an operand invariant in both candidate axes scores identically
        // either way -- and the cost model cannot see that only the contiguous axis
        // gives coalesced stores and lane-vectorised loads. Ascending order silently
        // picked "ow x1, n x4" over "ow x4" for depthwise, which is equal on paper and
        // worse on the machine.
        Array.Reverse(factors);
        foreach (int tw in factors)
        {
            if (tw > 1 && axes[contiguous].Extent % tw != 0) continue;
            if (tw > laneCeiling) continue;

            // One-dimensional candidate: only the contiguous axis.
            double solo = PredictedRelativeTime(spec, axes, new[] { contiguous }, new[] { tw });
            if (solo < bestCost)
            {
                bestCost = solo;
                bestPrimary = tw;
                bestSecondAxis = -1;
                bestSecondFactor = 1;
            }

            foreach (int candidate in parallel)
            {
                if (candidate == contiguous) continue;
                foreach (int tk in factors)
                {
                    if (tk == 1) continue;
                    if (axes[candidate].Extent % tk != 0) continue;
                    if (tw * tk > laneCeiling) continue;

                    double cost = PredictedRelativeTime(spec, axes,
                        new[] { contiguous, candidate }, new[] { tw, tk });
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        bestPrimary = tw;
                        bestSecondAxis = candidate;
                        bestSecondFactor = tk;
                    }
                }
            }
        }

        if (bestPrimary > 1)
        {
            tileAxes.Add(contiguous);
            tileFactors.Add(bestPrimary);
        }
        if (bestSecondAxis >= 0)
        {
            // The contiguous axis must lead so lane vectorisation can use it.
            if (bestPrimary <= 1)
            {
                tileAxes.Add(contiguous);
                tileFactors.Add(1);
            }
            tileAxes.Add(bestSecondAxis);
            tileFactors.Add(bestSecondFactor);
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
        const int Multiprocessors = 68;
        const double OccupancyCoefficient = 0.5;

        long macs = spec.Output.ElementCount * Math.Max(1, spec.Space.ReductionTripCount);
        double loadInstructions = LoadsPerMac(spec, tileAxes, factors) * macs / WarpWidth;

        long bytes = spec.Output.ElementCount;
        for (int i = 0; i < spec.Inputs.Count; i++) bytes += spec.Inputs[i].ElementCount;
        double dramEquivalent = bytes * 4.0 / BytesPerLoadInstruction;

        // OCCUPANCY, which is what stops the search running away. A larger tile always
        // lowers loads per MAC, so without this term the best tile is always the biggest
        // one -- and at 112 lanes dense 3x3 would run 14 blocks on 68 SMs and stall.
        long lanes = 1;
        foreach (int f in factors) lanes *= f;
        long threads = spec.Output.ElementCount / Math.Max(1, lanes);
        double blocks = Math.Max(1.0, (threads + BlockThreads - 1) / (double)BlockThreads);
        double occupancy = 1.0 + OccupancyCoefficient / Math.Max(blocks / Multiprocessors, 0.05);

        // Compute never binds for these kernels, so the two memory terms decide it.
        return Math.Max(loadInstructions, dramEquivalent) * occupancy;
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
        L($"mul.wide.u32 {byteOffset}, {offset}, {I(binding.ElementBytes)};");
        L($"add.u64 {address}, {basePointer}, {byteOffset};");

        var regs = new string[width];
        for (int i = 0; i < width; i++) regs[i] = NextF();

        if (binding.NeedsConversion)
        {
            // FOUR HALVES ARE TWO WORDS, and two words is v2.u32 -- the same FOUR elements a
            // v4.f32 carries, so the lane model above is unchanged and only the instruction,
            // the byte scale and the unpacking differ.
            //
            // Without this a narrow tensor took the scalar path, and the profile said what
            // that costs: 32 lanes at 2 bytes is a 64-byte request, half a cache line, which
            // measured 2 sectors per request against fp32's 4 and reached only 66.6% of DRAM
            // peak against fp32's 89.1%. The access was perfectly coalesced FOR ITS WIDTH and
            // still wasted half the bus.
            //
            // This is emphatically not the v4.f32 path re-enabled for narrow bindings. That
            // one scales by four bytes an element and would read twice the intended span,
            // returning neighbouring data with no complaint -- which is why it was excluded.
            string lo = NextR(), hi = NextR();
            L($"ld.global.nc.v2.u32 {{{lo}, {hi}}}, [{address}];");

            EmitWiden(binding.ElementType, lo, regs[0]);
            string loHigh = NextR();
            L($"shr.b32 {loHigh}, {lo}, 16;");
            EmitWiden(binding.ElementType, loHigh, regs[1]);

            EmitWiden(binding.ElementType, hi, regs[2]);
            string hiHigh = NextR();
            L($"shr.b32 {hiHigh}, {hi}, 16;");
            EmitWiden(binding.ElementType, hiHigh, regs[3]);
        }
        else
        {
            L($"ld.global.v4.f32 {{{regs[0]}, {regs[1]}, {regs[2]}, {regs[3]}}}, [{address}];");
        }

        VectorisedLoads++;
        EmittedLoads++;
        return regs;
    }

    public string Emit(CodegenKernelSpec spec, int computeMajor, int computeMinor)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));

        // A non-real algebra takes its own path, leaving everything below untouched. See
        // PtxAffineEmitter.Algebra.cs for why that separation is deliberate rather than lazy.
        if (spec.Algebra != CodegenAlgebra.Real)
            return EmitAlgebraic(spec, computeMajor, computeMinor);

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

        // Block size derived so a block's threads all agree on the slow axes, which is
        // the precondition for staging a block-invariant operand in shared memory.
        int blockThreads = EnableSharedStaging
            ? ChooseBlockThreads(axes, parallel, tileAxes, tileFactors, out var varyingAxes)
            : PassThroughBlock(out varyingAxes);

        // An operand indexed only by axes that are CONSTANT across the block is fetched
        // identically by every thread in it. Dense 3x3's weights are the case that
        // matters: 196 threads share one output-channel group, so the same weight is
        // fetched 196 times.
        var stageable = new List<int>();
        if (EnableSharedStaging && varyingAxes.Count > 0)
        {
            foreach (int inputIdx in spec.ProductInputs)
            {
                // The staged slice is addressed in 4-byte units and re-read as f32, so a
                // narrow binding would be staged at the wrong stride. Excluded until the
                // staging path carries an element size of its own.
                if (spec.Inputs[inputIdx].NeedsConversion) continue;

                bool varies = false;
                foreach (int ax in varyingAxes)
                    if (ReferencesAxis(spec.Inputs[inputIdx], ax)) varies = true;
                if (!varies) stageable.Add(inputIdx);
            }
        }
        // Staging pays only if the operand is fetched redundantly across the block, and
        // only if its slice fits. Keep at most one: the shared budget is small and the
        // weights are the measured 196x redundancy.
        int stagedInput = -1;
        int stageCount = 0;
        int stagedTileSlot = -1;
        if (stageable.Count > 0)
        {
            long innerTripsForStage = 1;
            foreach (int ax in reduction) innerTripsForStage *= axes[ax].Extent;

            foreach (int candidate in stageable)
            {
                // Distinct values the block needs per strip-mine step: one per
                // (lane offset on a tiled axis this operand reads) x (unrolled trip).
                int slot = -1;
                for (int t = 0; t < tileAxes.Count; t++)
                    if (ReferencesAxis(spec.Inputs[candidate], tileAxes[t])) slot = t;

                long lanesForOperand = slot >= 0 ? tileFactors[slot] : 1;
                long count = lanesForOperand * innerTripsForStage;

                // The block must be able to fetch the whole slice in one pass, and it
                // must fit the shared budget.
                if (count > blockThreads) continue;
                if (count * 4 > MaxSharedBytes) continue;

                // STAGING IS NOT FREE: it adds two barriers per strip-mine step and the
                // registers to build the cooperative address. It only pays if the operand
                // is a large share of the loads. Measured both ways:
                //
                //   dense 3x3       weights are 50% of loads -> 68.1 us to 61.9 us
                //   conv_transpose  weights are  6% of loads -> 104 us to 131.4 us
                //
                // So require the staged operand to carry a real share of the traffic.
                double totalLoads = LoadsPerMac(spec, tileAxes.ToArray(), tileFactors.ToArray());
                double operandLoads = lanesForOperand;
                double lanesAll = 1;
                foreach (int f in tileFactors) lanesAll *= f;
                double share = totalLoads > 0 ? (operandLoads / lanesAll) / totalLoads : 0;
                if (share < MinimumStagedShare) continue;

                stagedInput = candidate;
                stageCount = (int)count;
                stagedTileSlot = slot;
                break;
            }
        }
        // TWO-DIMENSIONAL LOWERING, which is what makes the ACTIVATION operand stageable.
        // It is invariant in the reuse axis, so threads differing only in that axis want
        // the same input -- but a flat block walks the spatial axes at one value of the
        // reuse axis, exactly the wrong arrangement. x over the contiguous axis keeps
        // stores coalesced; y over the reuse axis makes a staged row serve the column.
        int dataOperand = -1;
        foreach (int inputIdx in spec.ProductInputs)
            if (inputIdx != stagedInput && spec.Inputs[inputIdx].Map.Count > 2) dataOperand = inputIdx;

        bool twoDimensional = false;
        int blockX = blockThreads, blockY = 1;
        if (EnableInputStaging && dataOperand >= 0 &&
            CanStageInput(spec, axes, tileAxes, tileFactors, dataOperand, out int bx, out int by))
        {
            twoDimensional = true;
            blockX = bx;
            blockY = by;
            blockThreads = bx * by;
        }
        UsedTwoDimensionalBlock = twoDimensional;

        // PER-DIMENSION STAGING.
        //
        // A two-dimensional block invalidates the flat staging analysis. That analysis asks
        // which operands are constant across the whole BLOCK; under a flat block the
        // weights are, because every thread shares one reuse-axis group. Under 2D, y varies
        // over that axis, so each row wants a different weight slice and one staged copy is
        // wrong for every row but one -- measured directly, the two dense kernels returned
        // 5.277 and 1.112e1 instead of zero.
        //
        // The fix is not to disable staging but to index it correctly: under 2D each operand
        // is invariant in exactly one DIMENSION, so its slice is keyed by the dimension it
        // varies in and shared along the other. The activation varies in x and is invariant
        // in y, so one row serves the whole column; the weights are the mirror image. Both
        // get staged, which is the point -- the activation half is where the traffic is, and
        // staging weights alone only moved dense 3x3 from 68.1 us to 61.9 us against
        // cuDNN's 41.3.
        //
        // The slice spans the whole axis because CanStageInput requires the block to cover
        // both tiled axes completely, so it is addressed from zero.
        var stagedSlices = new List<StagedSlice>();
        if (twoDimensional)
        {
            stagedInput = -1;
            stagedTileSlot = -1;
            stageCount = 0;

            long tripsPerStep = 1;
            foreach (int ax in reduction) tripsPerStep *= axes[ax].Extent;

            int byteOffset = 0;
            for (int slot = 0; slot < 2 && slot < tileAxes.Count; slot++)
            {
                int tileAxis = tileAxes[slot];
                int other = tileAxes[slot == 0 ? 1 : 0];

                foreach (int inputIdx in spec.ProductInputs)
                {
                    var binding = spec.Inputs[inputIdx];
                    if (binding.NeedsConversion) continue;   // see the flat staging note

                    // Keyed by this dimension only if it varies here and is invariant in
                    // the other -- that invariance is the reuse being exploited.
                    if (!ReferencesAxis(binding, tileAxis)) continue;
                    if (ReferencesAxis(binding, other)) continue;

                    // Every OTHER axis this operand reads is either a reduction axis, which
                    // the fetch unpacks from the trip, or an axis the grid covers, which is
                    // constant across the block and so already correct in the thread's own
                    // register. The two tiled axes are the only ones that vary within a 2D
                    // block, and both have been accounted for above.
                    //
                    // Using the FLAT lowering's varying-axis set here rejected every
                    // candidate: it counts n and oh as varying, when under 2D those are
                    // grid axes and constant within the block.
                    long count = axes[tileAxis].Extent * tripsPerStep;
                    if ((byteOffset + count * 4) > MaxSharedBytes) continue;

                    stagedSlices.Add(new StagedSlice(inputIdx, slot, slot, (int)count, byteOffset));
                    byteOffset += (int)count * 4;
                    break;   // one operand per dimension
                }
            }

            SharedMemoryBytes = byteOffset;
            StagedOperands = stagedSlices.Count == 0
                ? "none"
                : string.Join("+", stagedSlices.ConvertAll(s => spec.Inputs[s.Input].Name));
        }
        else if (stagedInput >= 0)
        {
            // Flat block: the operand is invariant across the WHOLE block, so every
            // consumer reads a compile-time constant offset. Dimension -1 records that.
            stagedSlices.Add(new StagedSlice(stagedInput, stagedTileSlot, -1, stageCount, 0));
        }

        // The derived block size exists ONLY to make staging possible: it is chosen so a
        // block covers whole groups of the staged operand's axes. If nothing ends up
        // staged, that constraint buys nothing and the odd size costs occupancy -- the
        // transposed kernel measured 114.3 us at the derived 196 against ~104 at 256.
        // So fall back to the standard block whenever staging did not apply.
        if (stagedInput < 0 && !twoDimensional)
        {
            blockThreads = BlockThreads;
            blockX = BlockThreads;
            blockY = 1;
        }
        LaunchBlockX = blockX;
        LaunchBlockY = blockY;

        if (!twoDimensional)
        {
            SharedMemoryBytes = stageCount * 4;
            StagedOperands = stagedInput < 0 ? "none" : spec.Inputs[stagedInput].Name;
        }
        LaunchBlockThreads = blockThreads;

        // Threads, grid and in-kernel guard all come from this one number, which is
        // the invariant the whole IR exists to protect.
        long threadCount = total / lanes;
        long blocks;
        if (twoDimensional)
        {
            // The block covers both tiled axes completely, so the grid is exactly the
            // product of the axes it does NOT cover.
            blocks = 1;
            foreach (int ax in parallel)
                if (!tileAxes.Contains(ax)) blocks *= axes[ax].Extent;
        }
        else
        {
            blocks = (threadCount + blockThreads - 1) / blockThreads;
        }
        CheckLimits(spec, blocks, threadCount);
        LaunchBlocks = (uint)blocks;

        _sb.Clear(); _body.Clear(); _r = _f = _p = _rd = 0; EmittedLoads = 0; ElidedGuards = 0;
        IndirectIndexLoads = 0; AtomicStores = 0; ExtraOutputStores = 0;
        _inputBindings = spec.Inputs;
        // Reset with the register counters: a label counter that survives between
        // calls makes the SAME spec emit different text on a second Emit, and cubins
        // are content-addressed on that text.
        _stageLabel = 0;

        // The out-of-range fill is the reduction's identity, not always zero.
        _outOfRangeFill = spec.Reduce == CodegenReduceKind.Max ? "0fFF800000" : "0f00000000";

        // Under a maximum, a product of operands has no single sensible identity: -inf times
        // a positive operand is -inf, but -inf times zero is NaN.
        if (spec.Reduce == CodegenReduceKind.Max && spec.ProductInputs.Count > 1)
            throw new NotSupportedException(
                "A Max reduction over a product of " + spec.ProductInputs.Count +
                " operands has no well-defined padding identity: the out-of-range fill " +
                "would have to be -inf for the product, and -inf times a zero operand is " +
                "NaN. Express the product as a single operand, or add an explicit mask.");

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
        _basePointers = basePtr;

        string ctaid = NextR(), tid = NextR(), gid = NextR();
        L($"mov.u32 {ctaid}, %ctaid.x;");

        if (twoDimensional)
        {
            // x indexes the contiguous tile, y the reuse tile, and the block covers both
            // completely, so a thread's position in them is just its thread index and no
            // guard is needed on those axes. The flat id is kept only so the staging
            // helper has a linear identity for the cooperative fetch.
            string tx = NextR(), ty = NextR();
            L($"mov.u32 {tx}, %tid.x;");
            L($"mov.u32 {ty}, %tid.y;");
            L($"mad.lo.u32 {tid}, {ty}, {I(blockX)}, {tx};");
            L($"mov.u32 {gid}, {tid};   // block-local; slow axes come from ctaid");
        }
        else
        {
            L($"mov.u32 {tid}, %tid.x;");
            L($"mad.lo.u32 {gid}, {ctaid}, {I(blockThreads)}, {tid};");

            // Bounds guard derived from the SAME TotalThreads the host launches with.
            string guard = NextP();
            L($"setp.ge.u32 {guard}, {gid}, {I(threadCount)};");
            L($"@{guard} bra END;");
        }

        // Decompose the flat id across parallel axes, last-declared fastest, so
        // consecutive threads walk the contiguous tensor axis. When coarsening, the
        // coarsened axis contributes extent/lanes to the decomposition, because one
        // thread now covers `lanes` of its positions.
        var axisReg = new string[axes.Count];
        string rest = twoDimensional ? ctaid : gid;
        for (int p = parallel.Length - 1; p >= 0; p--)
        {
            int ax = parallel[p];
            int tileSlot = tileAxes.IndexOf(ax);
            int extent = tileSlot >= 0 ? axes[ax].Extent / tileFactors[tileSlot] : axes[ax].Extent;

            // The two tiled axes are carried by the thread index in this lowering, so
            // they are not part of the block-index decomposition.
            if (twoDimensional && tileSlot >= 0)
            {
                string dim = NextR();
                L($"mov.u32 {dim}, %tid.{(tileSlot == 0 ? "x" : "y")};   // {axes[ax].Name} tile");
                axisReg[ax] = dim;
                continue;
            }
            axisReg[ax] = NextR();
            bool outermost = true;
            for (int q = 0; q < p; q++)
                if (!(twoDimensional && tileAxes.Contains(parallel[q]))) outermost = false;

            if (outermost)
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
        string? sharedBase = null;
        if (stagedInput >= 0)
        {
            sharedBase = NextRd();
            L($"mov.u64 {sharedBase}, stageBuf;");
        }

        // One accumulator per lane.
        var accs = new string[lanes];
        for (int l = 0; l < lanes; l++)
        {
            accs[l] = NextF();
            L(spec.Reduce == CodegenReduceKind.Max
                ? $"mov.f32 {accs[l]}, 0fFF800000;   // -inf"
                : $"mov.f32 {accs[l]}, 0f00000000;");
        }

        // One index register per lane, tracking which reduction term is currently winning.
        string[]? argIndex = null;
        if (spec.SecondaryOutput is not null && spec.SecondaryIndexExpr is not null)
        {
            argIndex = new string[lanes];
            for (int l = 0; l < lanes; l++)
            {
                argIndex[l] = NextR();
                L($"mov.u32 {argIndex[l]}, 0;");
            }
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

        // Stage the block-invariant operand for THIS strip-mine step. Inside a loop this
        // re-runs per iteration, which is required: the staged slice depends on the loop
        // counter.
        if (stagedSlices.Count > 0 && sharedBase != null)
        {
            long innerTripsNow = 1;
            foreach (int ax in reduction) innerTripsNow *= axes[ax].Extent;

            foreach (var slice in stagedSlices)
            {
                var b = spec.Inputs[slice.Input];
                string sliceBase = sharedBase;
                if (slice.ByteOffset != 0)
                {
                    sliceBase = NextRd();
                    L($"add.u64 {sliceBase}, {sharedBase}, {I(slice.ByteOffset)};");
                }

                EmitStageLoad(
                    b, basePtr[b.ParameterIndex], sliceBase, tid,
                    slice.TileSlot >= 0 ? tileAxes[slice.TileSlot] : -1,
                    slice.TileSlot >= 0 ? axisReg[tileAxes[slice.TileSlot]] : null!,
                    slice.TileSlot >= 0 ? tileFactors[slice.TileSlot] : 1,
                    reduction, axes, axisReg, slice.Count, (int)innerTripsNow,
                    coversAxisFromZero: slice.Dimension >= 0,
                    blockThreads: LaunchBlockThreads);
            }
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
                // A narrow binding vectorises through v2.u32 -- four halves, the same four
                // elements a v4.f32 carries -- rather than being excluded. It cannot use the
                // f32 form, which scales by four bytes an element and would read twice the
                // intended span; see EmitVectorLoad.
                //
                // An INDEX tensor is still excluded: it is never an arithmetic operand, so
                // there is nothing to widen it into.
                if (!spec.Inputs[inputIdx].IsIndexTensor &&
                    IsUnitStrideIn(spec.Inputs[inputIdx], innermost, axes))
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

        // Per-thread base for each dimension-keyed slice, hoisted once. The block
        // coordinate is the only runtime term in a staged read, so lifting it here keeps
        // every consumer at a constant offset.
        var sliceBaseCache = new Dictionary<int, string>();
        string SliceBase(StagedSlice slice, long tileTripsForBase)
        {
            if (sliceBaseCache.TryGetValue(slice.Input, out string? cached)) return cached;

            int stride = (int)(tileFactors[slice.TileSlot] * tileTripsForBase * 4);
            string scaled = NextRd();
            L($"mul.wide.u32 {scaled}, {axisReg[tileAxes[slice.TileSlot]]}, {I(stride)};");
            string result = NextRd();
            L($"add.u64 {result}, {sharedBase}, {scaled};");
            if (slice.ByteOffset != 0)
            {
                string shifted = NextRd();
                L($"add.u64 {shifted}, {result}, {I(slice.ByteOffset)};");
                result = shifted;
            }
            sliceBaseCache[slice.Input] = result;
            return result;
        }

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
                    else if (sharedBase != null && SliceFor(stagedSlices, inputIdx) is { } slice)
                    {
                        // Resident in shared memory.
                        int laneOffset = slice.TileSlot >= 0 ? laneOffsets[l][slice.TileSlot] : 0;
                        string loaded = NextF();
                        string sharedAddr = NextRd();

                        if (slice.Dimension < 0)
                        {
                            // Block-invariant: both coordinates are compile-time constants,
                            // so this is a constant-offset ld.shared with no address
                            // arithmetic at the point of use.
                            long slotIndex = laneOffset * trips + t;
                            L($"add.u64 {sharedAddr}, {sharedBase}, {I(slice.ByteOffset + slotIndex * 4)};");
                        }
                        else
                        {
                            // Keyed by a block dimension. The slice spans the whole axis, so
                            // the element is at (blockCoord * tileFactor + lane) * trips + t.
                            // Only the blockCoord term is a runtime value, and it is hoisted
                            // once per thread, so this stays a constant offset from a
                            // per-thread base.
                            string sliceBase = SliceBase(slice, tileTripsForBase: trips);
                            long constant = (laneOffset * (long)trips + t) * 4;
                            L($"add.u64 {sharedAddr}, {sliceBase}, {I(constant)};");
                        }

                        L($"ld.shared.f32 {loaded}, [{sharedAddr}];");
                        value = loaded;
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

                // The pre-reduction slot: shift, then transform, per TERM. An epilogue
                // activation cannot do this -- it runs once on the finished accumulator,
                // while softmax needs exp applied to every term before summing.
                if (spec.PreBiasInput.HasValue)
                {
                    var pb = spec.Inputs[spec.PreBiasInput.Value];
                    string pv = EmitLoad(pb, basePtr[pb.ParameterIndex], laneAxisReg[l],
                                         reductionValues, reduction, out _);
                    string shifted = NextF();
                    if (spec.PreBiasScale == 1.0)
                    {
                        L($"add.rn.f32 {shifted}, {product!}, {pv};");
                    }
                    else
                    {
                        // fma so the scaled add stays one instruction and one rounding.
                        L($"fma.rn.f32 {shifted}, {pv}, {F32(spec.PreBiasScale)}, {product!};");
                    }
                    product = shifted;
                }
                if (spec.PreReduce != CodegenPreReduceOp.None)
                {
                    string transformed = NextF();
                    if (spec.PreReduce == CodegenPreReduceOp.Square)
                    {
                        L($"mul.rn.f32 {transformed}, {product!}, {product!};");
                    }
                    else
                    {
                        // exp(t) = ex2(t * log2 e); PTX has no exp.
                        string scaledT = NextF();
                        L($"mul.rn.f32 {scaledT}, {product!}, 0f3FB8AA3B;");
                        L($"ex2.approx.f32 {transformed}, {scaledT};");
                    }
                    product = transformed;
                }

                // An out-of-range tap contributes the additive identity; the guarded
                // load already produced 0, so no extra select is needed for Sum.
                if (spec.Reduce == CodegenReduceKind.Max)
                {
                    if (argIndex is not null)
                    {
                        // Track WHICH term won, not just the value. Strictly greater keeps
                        // the FIRST maximum on a tie, matching the established kernel --
                        // the backward pass routes a gradient by this index, so a different
                        // tie-break silently sends it to the wrong element.
                        string beats = NextP();
                        L($"setp.gt.f32 {beats}, {product!}, {accs[l]};");

                        string candidate = EmitAffine(
                            spec.SecondaryIndexExpr!, laneAxisReg[l], reductionValues, reduction);
                        string chosen = NextR();
                        L($"selp.b32 {chosen}, {candidate}, {argIndex[l]}, {beats};");
                        argIndex[l] = chosen;
                    }
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
            if (stagedSlices.Count > 0) L("bar.sync 0;");
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

            // The constant scale lands BEFORE the bias: a mean-then-add-bias is the
            // operator, and scaling afterwards would scale the bias too.
            if (spec.ReduceScale != 1.0)
            {
                string scaled = NextF();
                L($"mul.rn.f32 {scaled}, {acc}, {F32(spec.ReduceScale)};");
                acc = scaled;
            }

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
            acc = EmitActivation(spec.Activation, acc);

            string laneOff = EmitOffset(spec.Output, laneAxisReg[l], reductionValues, reduction,
                                       out string? lanePred);
            string laneAddr = NextRd(), laneByte = NextRd();
            L($"mul.wide.u32 {laneByte}, {laneOff}, {I(spec.Output.ElementBytes)};");
            L($"add.u64 {laneAddr}, {basePtr[spec.Output.ParameterIndex]}, {laneByte};");
            if (spec.Output.NeedsAtomicStore)
            {
                // SCATTER. A destination reached through a run-time index cannot be proven
                // unique, so two iterations may target the same element. A plain store would
                // keep whichever warp finished last -- a race that yields a different wrong
                // answer per run and reads as flakiness rather than as a bug. The atomic
                // makes it an accumulation, which is what every scatter operator wants
                // anyway: an embedding backward sums the gradients of repeated tokens.
                //
                // The destination must be zeroed by the caller before launch, since this
                // adds to whatever is already there.
                if (spec.Output.NeedsConversion)
                    throw new NotSupportedException(
                        "A scatter destination must be fp32: there is no atomic add for a " +
                        "16-bit float here, and emulating one with a compare-and-swap loop " +
                        "would silently change the operator's cost.");

                if (lanePred is null) L($"red.global.add.f32 [{laneAddr}], {acc};");
                else L($"@{lanePred} red.global.add.f32 [{laneAddr}], {acc};");
                AtomicStores++;
            }
            else if (spec.Output.NeedsConversion)
            {
                string narrowed = EmitNarrow(spec.Output.ElementType, acc);
                if (lanePred is null) L($"st.global.u16 [{laneAddr}], {narrowed};");
                else L($"@{lanePred} st.global.u16 [{laneAddr}], {narrowed};");
            }
            else if (lanePred is null) L($"st.global.f32 [{laneAddr}], {acc};");
            else L($"@{lanePred} st.global.f32 [{laneAddr}], {acc};");

            // The secondary output holds the argmax position, stored as a 32-bit INTEGER --
            // the consumer is an indices buffer that the backward pass indexes with, not a
            // float. Writing it as f32 would round above 2^24 and silently misroute
            // gradients on a large enough tensor.
            if (argIndex is not null && spec.SecondaryOutput is not null)
            {
                string secondOff = EmitOffset(spec.SecondaryOutput, laneAxisReg[l],
                                              reductionValues, reduction, out string? secondPred);
                string secondAddr = NextRd(), secondByte = NextRd();
                L($"mul.wide.u32 {secondByte}, {secondOff}, 4;");
                L($"add.u64 {secondAddr}, {basePtr[spec.SecondaryOutput.ParameterIndex]}, {secondByte};");
                if (secondPred is null) L($"st.global.u32 [{secondAddr}], {argIndex[l]};");
                else L($"@{secondPred} st.global.u32 [{secondAddr}], {argIndex[l]};");
            }

            // EVERY REMAINING EXTRA OUTPUT, from the SAME accumulator. The count is not
            // capped: an Adam step writes two moment states and the updated parameter, and
            // splitting that across kernels re-reads every operand once per kernel -- which
            // is the exact cost the fusion argument is about.
            //
            // The argmax written just above is skipped here because it has already been
            // emitted through the legacy path; both are entries in the same list.
            for (int e = 0; e < spec.ExtraOutputs.Count; e++)
            {
                var extra = spec.ExtraOutputs[e];
                if (extra.Kind == CodegenExtraOutputKind.ArgMaxIndex) continue;

                // Scale * primary, then optionally + BiasScale * bias. The primary is the
                // FINISHED value -- after the epilogue activation -- because an optimizer's
                // parameter update steps by the state it just computed, not by a partial.
                string value = NextF();
                L($"mul.rn.f32 {value}, {acc}, {F32(extra.Scale)};");

                if (extra.BiasInput.HasValue)
                {
                    var biasBinding = spec.Inputs[extra.BiasInput.Value];
                    string biasValue = EmitLoad(
                        biasBinding, basePtr[biasBinding.ParameterIndex], laneAxisReg[l],
                        reductionValues, reduction, out _);
                    string combined = NextF();
                    L($"fma.rn.f32 {combined}, {biasValue}, {F32(extra.BiasScale)}, {value};");
                    value = combined;
                }

                string extraOff = EmitOffset(extra.Binding, laneAxisReg[l],
                                             reductionValues, reduction, out string? extraPred);
                string extraAddr = NextRd(), extraByte = NextRd();
                L($"mul.wide.u32 {extraByte}, {extraOff}, {I(extra.Binding.ElementBytes)};");
                L($"add.u64 {extraAddr}, {basePtr[extra.Binding.ParameterIndex]}, {extraByte};");

                if (extra.Binding.NeedsConversion)
                {
                    string narrowedExtra = EmitNarrow(extra.Binding.ElementType, value);
                    if (extraPred is null) L($"st.global.u16 [{extraAddr}], {narrowedExtra};");
                    else L($"@{extraPred} st.global.u16 [{extraAddr}], {narrowedExtra};");
                }
                else if (extraPred is null) L($"st.global.f32 [{extraAddr}], {value};");
                else L($"@{extraPred} st.global.f32 [{extraAddr}], {value};");

                ExtraOutputStores++;
            }
        }

        // Loads before the loop run once, loads in its body run once per trip, loads
        // after it (the epilogue) run once.
        long loadsAfterLoop = EmittedLoads - loadsBeforeLoop - loadsInLoopBody;
        DynamicLoadsPerThread = loadsBeforeLoop + loadsInLoopBody * loopTrips + loadsAfterLoop;

        CheckEmittedSize(spec);
        _body.Append("END:\n    ret;\n}\n");
        return Assemble();
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
    /// <summary>
    /// Evaluates a single affine expression into a register.
    /// </summary>
    /// <remarks>
    /// Used for the secondary output's index, which is an expression rather than a tensor
    /// map: it names a POSITION in the input, not an address in a binding. Reduction-axis
    /// terms fold to constants exactly as they do in an address, because the value is
    /// emitted once per unrolled term.
    /// </remarks>
    /// <summary>
    /// Closes the module: register declarations sized from what the body actually used,
    /// then the body.
    /// </summary>
    /// <remarks>
    /// The bound is DERIVED FROM USAGE rather than fixed. It was previously a literal
    /// %p&lt;256&gt;/%f&lt;512&gt;, which is a silent ceiling rather than a generous bound:
    /// coarsening pushed the transposed convolution one past the declared range, and ptxas
    /// reported it as "Arguments mismatch for instruction 'setp'" -- an undeclared register,
    /// not a malformed instruction.
    /// </remarks>
    private string Assemble()
    {
        if (SharedMemoryBytes > 0)
            _sb.Append("    .shared .align 4 .b8 stageBuf[")
               .Append(I(SharedMemoryBytes)).Append("];\n");
        _sb.Append("    .reg .pred %p<").Append(I(_p + 8)).Append(">;\n")
           .Append("    .reg .b32 %r<").Append(I(_r + 8)).Append(">;\n")
           .Append("    .reg .b64 %rd<").Append(I(_rd + 8)).Append(">;\n")
           .Append("    .reg .f32 %f<").Append(I(_f + 8)).Append(">;\n")
           .Append(_body);
        return _sb.ToString();
    }

    private string EmitAffine(
        CodegenAffineExpr expr, string[] axisReg, int[] reductionValues, int[] reductionAxes)
    {
        var reductionSet = new HashSet<int>(reductionAxes);

        int folded = expr.Constant;
        var symbolic = new List<CodegenAffineTerm>();
        foreach (var term in expr.Terms)
        {
            if (reductionSet.Contains(term.Axis)) folded += term.Coefficient * reductionValues[term.Axis];
            else symbolic.Add(term);
        }

        string? value = null;
        foreach (var term in symbolic)
        {
            string contribution;
            if (term.Coefficient == 1) contribution = axisReg[term.Axis];
            else
            {
                contribution = NextR();
                L($"mul.lo.s32 {contribution}, {axisReg[term.Axis]}, {I(term.Coefficient)};");
            }
            if (value is null) value = contribution;
            else { string sum = NextR(); L($"add.s32 {sum}, {value}, {contribution};"); value = sum; }
        }

        if (value is null)
        {
            value = NextR();
            L($"mov.u32 {value}, {I(folded)};");
        }
        else if (folded != 0)
        {
            string sum = NextR();
            L($"add.s32 {sum}, {value}, {I(folded)};");
            value = sum;
        }

        if (expr.Divisor != 1)
        {
            string divided = NextR();
            L($"div.s32 {divided}, {value}, {I(expr.Divisor)};");
            value = divided;
        }
        return value;
    }

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
            int dim = binding.Shape[d];
            long stride = binding.Stride(d);
            string idx;

            var indirect = binding.Indirect[d];
            if (indirect is not null)
            {
                idx = EmitIndirectIndex(indirect, axisReg, reductionValues, reductionSet, ref predicate);
            }
            else
            {
                var expr = binding.Map[d];
                idx = EmitAffineIndex(
                    expr, axisReg, reductionValues, reductionSet, dim, ref predicate);
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

    /// <summary>
    /// Computes one affine dimension's index, contributing its exactness and bounds terms to
    /// <paramref name="predicate"/>.
    /// </summary>
    private string EmitAffineIndex(
        CodegenAffineExpr expr,
        string[] axisReg,
        int[] reductionValues,
        HashSet<int> reductionSet,
        int dim,
        ref string? predicate)
    {
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

        return idx;
    }

    /// <summary>
    /// Loads a dimension's index from another tensor at run time -- the gather/scatter case.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Two guards are emitted and they do different jobs, which is the whole subtlety here.
    /// </para>
    /// <para>
    /// The value is ALWAYS clamped into <c>[0, bound)</c> before it reaches the address
    /// arithmetic. That is not the caller's out-of-range policy -- it is what keeps a
    /// malformed index tensor from forming an address outside the allocation. Predicating
    /// the load alone would not do it: the address is computed whether or not the load
    /// fires, and on a wild index that arithmetic can overflow into somebody else's memory.
    /// </para>
    /// <para>
    /// The POLICY then decides whether the access counts. Under <c>Skip</c> the range test
    /// joins the access predicate, so an out-of-range index contributes the reduction
    /// identity on a gather and performs no write on a scatter -- which is what a padding row
    /// or a -1 sentinel means. Under <c>Clamp</c> no predicate is added and the clamped edge
    /// element is genuinely read or written.
    /// </para>
    /// </remarks>
    private string EmitIndirectIndex(
        CodegenIndirectIndex indirect,
        string[] axisReg,
        int[] reductionValues,
        HashSet<int> reductionSet,
        ref string? predicate)
    {
        var indexBinding = _inputBindings![indirect.IndexInput];

        // Where to read the index from. This position is still affine, which is what keeps
        // the whole mechanism tractable: the emitter knows exactly where the index lives, it
        // just does not know its value until the load returns.
        string? positionPredicate = null;
        string position = EmitAffineIndex(
            indirect.Position, axisReg, reductionValues, reductionSet,
            (int)indexBinding.ElementCount, ref positionPredicate);

        string posBytes = NextRd(), posAddr = NextRd(), raw = NextR();
        L($"mul.wide.s32 {posBytes}, {position}, 4;");
        L($"add.u64 {posAddr}, {_basePointers![indexBinding.ParameterIndex]}, {posBytes};");

        if (positionPredicate is null)
        {
            L($"ld.global.nc.u32 {raw}, [{posAddr}];");
        }
        else
        {
            // A position outside the index tensor reads nothing and yields a sentinel that
            // the range test below will reject.
            L($"mov.u32 {raw}, 0xFFFFFFFF;");
            L($"@{positionPredicate} ld.global.nc.u32 {raw}, [{posAddr}];");
        }
        IndirectIndexLoads++;

        if (indirect.OutOfRange == CodegenIndexOutOfRange.Skip)
        {
            string lo = NextP(), hi = NextP(), both = NextP();
            L($"setp.ge.s32 {lo}, {raw}, 0;");
            L($"setp.lt.s32 {hi}, {raw}, {I(indirect.Bound)};");
            L($"and.pred {both}, {lo}, {hi};");
            predicate = AndPred(predicate, both);
            if (positionPredicate is not null) predicate = AndPred(predicate, positionPredicate);
        }

        // The unconditional clamp described above. Both forms need it.
        string clampedLow = NextR(), clamped = NextR();
        L($"max.s32 {clampedLow}, {raw}, 0;");
        L($"min.s32 {clamped}, {clampedLow}, {I(indirect.Bound - 1)};");
        return clamped;
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
        if (binding.IsIndexTensor)
            throw new NotSupportedException(
                "'" + binding.Name + "' is an int32 index tensor and cannot be read as an " +
                "arithmetic operand. Reading it as one would compute with a bit pattern.");

        string offset = EmitOffset(binding, axisReg, reductionValues, reductionAxes, out predicate);
        string byteOff = NextRd(), addr = NextRd(), dst = NextF();
        L($"mul.wide.s32 {byteOff}, {offset}, {I(binding.ElementBytes)};");
        L($"add.u64 {addr}, {basePtr}, {byteOff};");

        if (binding.NeedsConversion)
        {
            // NARROW STORAGE, WIDE ARITHMETIC. A 16-bit operand is widened the moment it
            // is read and everything downstream stays fp32, which is what mixed precision
            // means: the operands are narrow, the accumulator is not. Accumulating in fp16
            // would lose roughly three decimal digits over a long reduction and is never
            // what a caller wants from a narrow INPUT.
            string raw = NextR();
            if (predicate is null)
            {
                L($"ld.global.nc.u16 {raw}, [{addr}];");
            }
            else
            {
                L($"mov.u32 {raw}, 0;");
                L($"@{predicate} ld.global.nc.u16 {raw}, [{addr}];");
            }
            EmitWiden(binding.ElementType, raw, dst);

            // The out-of-range fill has to be applied AFTER widening: a zero bit pattern is
            // zero in both formats, but the Max identity is negative infinity and its fp16
            // pattern differs from its fp32 one.
            if (predicate is not null && _outOfRangeFill != "0f00000000")
            {
                string fill = NextF();
                L($"mov.f32 {fill}, {_outOfRangeFill};");
                L($"selp.f32 {dst}, {dst}, {fill}, {predicate};");
            }
            EmittedLoads++;
            return dst;
        }

        if (predicate is null)
        {
            L($"ld.global.nc.f32 {dst}, [{addr}];");
        }
        else
        {
            L($"mov.f32 {dst}, {_outOfRangeFill};");
            L($"@{predicate} ld.global.nc.f32 {dst}, [{addr}];");
        }
        EmittedLoads++;
        return dst;
    }

    /// <summary>Widens a 16-bit stored element in a b32 register into an f32 register.</summary>
    /// <remarks>
    /// fp16 has a hardware conversion. bf16 does not need one: it is the TOP 16 bits of the
    /// fp32 pattern, so widening is a shift and narrowing is a truncation. Doing bf16
    /// through <c>cvt</c> would be both slower and wrong, since no such instruction exists
    /// for it on this architecture.
    /// </remarks>
    private void EmitWiden(CodegenElementType type, string rawB32, string destF32)
    {
        if (type == CodegenElementType.Float16)
        {
            L($"cvt.f32.f16 {destF32}, {rawB32};");
            return;
        }

        string shifted = NextR();
        L($"shl.b32 {shifted}, {rawB32}, 16;");
        L($"mov.b32 {destF32}, {shifted};");
    }

    /// <summary>Narrows an f32 register into the low 16 bits of a b32 register.</summary>
    /// <remarks>
    /// bf16 narrowing rounds to nearest even rather than truncating. Truncation is the
    /// obvious implementation and biases every value toward zero, which accumulates into a
    /// systematic drift over a training run rather than showing up as noise in one kernel.
    /// </remarks>
    private string EmitNarrow(CodegenElementType type, string sourceF32)
    {
        string result = NextR();
        if (type == CodegenElementType.Float16)
        {
            L($"cvt.rn.f16.f32 {result}, {sourceF32};");
            return result;
        }

        string bits = NextR(), lsb = NextR(), bias = NextR(), rounded = NextR();
        L($"mov.b32 {bits}, {sourceF32};");
        L($"shr.u32 {lsb}, {bits}, 16;");
        L($"and.b32 {lsb}, {lsb}, 1;");
        L($"add.u32 {bias}, {lsb}, 32767;");
        L($"add.u32 {rounded}, {bits}, {bias};");
        L($"shr.u32 {result}, {rounded}, 16;");
        return result;
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
