// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>
/// Emits a warp-collective matrix-multiply using the tensor cores, for the subset of specs
/// that a <c>wmma</c> tile can express exactly.
/// </summary>
/// <remarks>
/// <para>
/// This is a SEPARATE emitter rather than a flag on <see cref="PtxAffineEmitter"/>, because
/// the two lowerings disagree about what a thread is. The affine emitter gives each thread
/// its own output element and its own address arithmetic; <c>wmma</c> gives a whole WARP one
/// 16x16 tile and does not tell you which lane holds which element -- the fragment layout is
/// deliberately opaque, and correct code must not assume one. Bolting the collective form
/// into the per-thread path would mean threading "is this instruction warp-wide" through
/// every address, guard and epilogue in 2,000 lines of emitter that is correct today.
/// </para>
/// <para>
/// <c>wmma</c> is chosen over <c>mma.sync</c> deliberately. <c>mma.sync</c> is faster on
/// paper and is what a hand-tuned library uses, but it requires the emitter to place each
/// lane's operand elements by hand from the register-layout tables; get one lane wrong and
/// the kernel still runs, still produces plausible magnitudes, and is wrong. <c>wmma</c>
/// hands that mapping to the hardware. The tensor cores are the same either way -- what
/// differs is scheduling freedom, not peak throughput.
/// </para>
/// <para>
/// The reason this exists at all: with no tensor-core path, every generated GEMM competes
/// against cuBLAS using the FP32 pipes while cuBLAS uses the tensor cores. That is not a
/// tuning deficit that better tiling can close -- it is roughly an order of magnitude of
/// arithmetic throughput, and it is why "dense GEMM at large K" is recorded in the blueprint
/// as unwinnable. It is only unwinnable from the scalar pipes.
/// </para>
/// <para>
/// The advantage this path has over cuBLAS is the same one the rest of the campaign trades
/// on: cuBLAS cannot fuse through its own call boundary, so an activation costs it a launch
/// and a full round trip of the output tensor. Here the epilogue is applied to the
/// accumulator fragment while it is still in registers.
/// </para>
/// </remarks>
public sealed partial class PtxTensorCoreEmitter
{
    /// <summary>The <c>wmma</c> tile this emitter uses. m16n16k16 is the shape every
    /// tensor-core generation from sm_70 onward supports for f16 multiplicands.</summary>
    public const int TileM = 16;

    /// <inheritdoc cref="TileM"/>
    public const int TileN = 16;

    /// <inheritdoc cref="TileM"/>
    public const int TileK = 16;

    /// <summary>Registers in an f16 a/b fragment, and in an f32 accumulator fragment.</summary>
    private const int FragmentRegisters = 8;

    /// <summary>Warps per block. Four keeps a block at 128 threads.</summary>
    public int WarpsPerBlock { get; set; } = 4;

    /// <summary>
    /// Whether to use the shared-memory staged lowering where the shape allows it.
    /// </summary>
    /// <remarks>
    /// Settable so the two lowerings can be measured against each other on the same shape.
    /// Turning it off is a measurement device, not a supported configuration.
    /// </remarks>
    public bool EnableStaging { get; set; } = true;

    private readonly StringBuilder _sb = new();

    /// <summary>Number of <c>wmma.mma</c> instructions the last emission produced.</summary>
    public int MmaInstructions { get; private set; }

    /// <summary>Whether the K loop was fully unrolled.</summary>
    public bool Unrolled { get; private set; }

    private static string I(int v) => v.ToString(CultureInfo.InvariantCulture);

    private void L(string text) => _sb.Append("    ").Append(text).Append('\n');

    /// <summary>
    /// The shape a spec must have to be emitted as a tensor-core matmul, together with the
    /// operand orientations recovered from its index maps.
    /// </summary>
    public sealed class Plan
    {
        internal Plan(int m, int n, int k, bool aRowMajor, bool bRowMajor)
        {
            M = m; N = n; K = k; ARowMajor = aRowMajor; BRowMajor = bRowMajor;
        }

        /// <summary>Rows of the output.</summary>
        public int M { get; }

        /// <summary>Columns of the output.</summary>
        public int N { get; }

        /// <summary>Contracted extent.</summary>
        public int K { get; }

        /// <summary>Whether A is stored [M, K] rather than [K, M].</summary>
        public bool ARowMajor { get; }

        /// <summary>Whether B is stored [K, N] rather than [N, K].</summary>
        public bool BRowMajor { get; }

        /// <summary>Output tiles, one warp each.</summary>
        public int TileCount => (M / TileM) * (N / TileN);
    }

    /// <summary>
    /// Decides whether <paramref name="spec"/> can be emitted on the tensor cores, and if
    /// not, says why in terms a caller can act on.
    /// </summary>
    /// <remarks>
    /// Every rejection names the specific property that failed rather than returning a bare
    /// false, because the caller's alternative -- the scalar emitter -- is always correct,
    /// so a silent fallback would look like the tensor cores simply never helping.
    /// </remarks>
    public static bool TryPlan(
        CodegenKernelSpec spec, int computeMajor, int computeMinor,
        out Plan? plan, out string reason)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        plan = null;

        if (computeMajor < 7)
        {
            reason = "tensor cores require sm_70 or later; this device is sm_" +
                I(computeMajor) + I(computeMinor);
            return false;
        }

        var axes = spec.Space.Axes;
        int[] parallel = spec.Space.ParallelAxes;
        int[] reduction = spec.Space.ReductionAxes;

        if (parallel.Length != 2 || reduction.Length != 1)
        {
            reason = "a wmma tile is exactly two parallel axes over one contraction; this " +
                "spec has " + I(parallel.Length) + " parallel and " + I(reduction.Length) +
                " reduction";
            return false;
        }

        if (spec.Reduce != CodegenReduceKind.Sum)
        {
            reason = "the tensor cores compute a sum of products; " + spec.Reduce +
                " is a different reduction";
            return false;
        }

        if (spec.PreReduce != CodegenPreReduceOp.None)
        {
            reason = "a pre-reduction transform cannot be applied inside a wmma tile, " +
                "because the fragment layout does not say which lane holds which element";
            return false;
        }

        if (spec.ProductInputs.Count != 2)
        {
            reason = "a wmma tile multiplies exactly two operands; this spec multiplies " +
                I(spec.ProductInputs.Count);
            return false;
        }

        if (spec.SecondaryOutput is not null)
        {
            reason = "a secondary output would need the fragment's element positions, " +
                "which wmma does not expose";
            return false;
        }

        int m = axes[parallel[0]].Extent, n = axes[parallel[1]].Extent;
        int k = axes[reduction[0]].Extent;

        if (m % TileM != 0 || n % TileN != 0 || k % TileK != 0)
        {
            reason = "wmma tiles are " + I(TileM) + "x" + I(TileN) + "x" + I(TileK) +
                " and this shape is " + I(m) + "x" + I(n) + "x" + I(k) +
                ", which is not a whole number of them";
            return false;
        }

        var a = spec.Inputs[spec.ProductInputs[0]];
        var b = spec.Inputs[spec.ProductInputs[1]];

        if (a.ElementType != CodegenElementType.Float16 ||
            b.ElementType != CodegenElementType.Float16)
        {
            reason = "wmma multiplicands must be fp16; got " + a.ElementType + " and " +
                b.ElementType;
            return false;
        }

        if (spec.Output.ElementType != CodegenElementType.Float32)
        {
            reason = "this path accumulates and stores fp32; the output is " +
                spec.Output.ElementType;
            return false;
        }

        // The index maps decide the orientations. Anything that is not one of the four
        // plain [row, col] forms -- a window, a stride, a broadcast -- is not a matmul.
        if (!TryOrientation(a, parallel[0], reduction[0], out bool aRowMajor))
        {
            reason = "operand A is not indexed as a plain 2-D matrix over the M and K axes";
            return false;
        }

        if (!TryOrientation(b, parallel[1], reduction[0], out bool bColumnOfN))
        {
            reason = "operand B is not indexed as a plain 2-D matrix over the N and K axes";
            return false;
        }

        if (!TryOrientation(spec.Output, parallel[0], parallel[1], out bool outRowMajor) ||
            !outRowMajor)
        {
            reason = "the output is not row-major [M, N]";
            return false;
        }

        // TryOrientation reports "first named axis is the row". For B the axes are handed
        // over as (N, K), so a "row-major" answer means [N, K] -- which is B TRANSPOSED.
        plan = new Plan(m, n, k, aRowMajor, bRowMajor: !bColumnOfN);
        reason = "eligible";
        return true;
    }

    /// <summary>
    /// Recovers whether a rank-2 binding is indexed [<paramref name="first"/>,
    /// <paramref name="second"/>] or the transpose, rejecting any map that is not one of
    /// those two.
    /// </summary>
    private static bool TryOrientation(
        CodegenTensorBinding binding, int first, int second, out bool firstIsRow)
    {
        firstIsRow = false;
        if (binding.Map.Count != 2) return false;

        if (!TryPlainAxis(binding.Map[0], out int rowAxis)) return false;
        if (!TryPlainAxis(binding.Map[1], out int columnAxis)) return false;

        if (rowAxis == first && columnAxis == second) { firstIsRow = true; return true; }
        if (rowAxis == second && columnAxis == first) { firstIsRow = false; return true; }
        return false;
    }

    /// <summary>True when the expression is exactly one axis with unit coefficient.</summary>
    private static bool TryPlainAxis(CodegenAffineExpr expr, out int axis)
    {
        axis = -1;
        if (expr.Terms.Count != 1 || expr.Constant != 0 || expr.Divisor != 1) return false;
        if (expr.Terms[0].Coefficient != 1) return false;
        axis = expr.Terms[0].Axis;
        return true;
    }

    /// <summary>
    /// Threads a block launches for the lowering actually emitted: <see cref="WarpsPerBlock"/>
    /// warps for the naive path, or the fixed <see cref="StagedWarps"/> when the last
    /// <see cref="Emit"/> produced the staged kernel.
    /// </summary>
    /// <remarks>
    /// MUST track the emitted lowering, exactly as <see cref="BlockCount"/> does. EmitStaged
    /// always launches <see cref="StagedWarps"/> (4) warps and derives each warp's row/column
    /// from <c>%r4 &gt;&gt; 1</c> / <c>%r4 &amp; 1</c>, an addressing scheme that is only valid for
    /// exactly four warps. <see cref="WarpsPerBlock"/> is public and settable and only
    /// COINCIDENTALLY defaults to 4, so a caller that tunes it while staging is enabled (also the
    /// default) would otherwise get a thread count that disagrees with the staged geometry.
    /// Launching that way is not merely wrong-but-safe: warps past index 3 compute
    /// warpAOffset/warpBOffset beyond the shared slab (e.g. %r5 * 1024 reaching 2048+ into a
    /// 2048-byte slab), an out-of-bounds shared-memory access on real hardware.
    /// </remarks>
    public int BlockThreads => Staged ? StageThreads : WarpsPerBlock * 32;

    /// <summary>
    /// Picks the largest warp tile whose block tile divides the output, unless the caller
    /// pinned one.
    /// </summary>
    /// <remarks>
    /// A LADDER DERIVED FROM MEASUREMENT, not from a cost model. `--warp-tile-sweep` timed
    /// every candidate at four shapes on an idle GPU, and the bigger tile won wherever it
    /// fits, by 1.28x at 2048^3 and 1.32x at 4096^3, for the reason the profile gave: L1TEX
    /// falls from 92.34% to 61.38% and the tensor pipe rises from 26.79% to 35.74%.
    ///
    /// It is not optimal everywhere. At 1024^3 the measured best was 4x2 at 71.9us against
    /// 4x4's 75.0us -- this rule gives up 4% there. Closing that needs a per-shape autotune
    /// pass (`--kernel-autotune` already exists for the affine kernels), not a cleverer rule:
    /// a static model picked lowerings four times on this branch and lost to the hardware
    /// every time it was checked.
    /// </remarks>
    private void SelectWarpTile(Plan plan)
    {
        if (PinWarpTile) return;

        foreach (var (m, n) in new[] { (4, 4), (4, 2), (2, 4), (2, 2) })
        {
            if (plan.M % (m * 32) == 0 && plan.N % (n * 32) == 0)
            {
                WarpTilesM = m;
                WarpTilesN = n;
                return;
            }
        }

        WarpTilesM = 2;
        WarpTilesN = 2;
    }

    /// <summary>
    /// Keeps the caller's <see cref="WarpTilesM"/>/<see cref="WarpTilesN"/> instead of
    /// selecting one. Used by the sweep, which is measuring the candidates.
    /// </summary>
    public bool PinWarpTile { get; set; }

    /// <summary>Blocks needed to cover a plan, under the lowering this emitter will pick.</summary>
    /// <remarks>
    /// THE TWO LOWERINGS NEED DIFFERENT GRIDS and it is not a small difference: staged, four
    /// warps cover sixteen 16x16 tiles rather than four, so a 64x64 output is ONE block
    /// staged and FOUR naive. Launching the staged kernel on the naive grid would run it four
    /// times over, each pass re-accumulating into the same output.
    /// </remarks>
    public int BlockCount(Plan plan)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));

        if (EnableStaging)
        {
            SelectWarpTile(plan);
            if (CanStage(plan, out _)) return StagedBlockCount(plan);
        }
        return (plan.TileCount + WarpsPerBlock - 1) / WarpsPerBlock;
    }

    /// <summary>Emits the tensor-core kernel for a spec <see cref="TryPlan"/> accepted.</summary>
    public string Emit(CodegenKernelSpec spec, int computeMajor, int computeMinor)
    {
        if (!TryPlan(spec, computeMajor, computeMinor, out var planOrNull, out string reason))
            throw new NotSupportedException(
                "This spec cannot be emitted on the tensor cores: " + reason);

        var plan = planOrNull!;
        _sb.Clear();
        MmaInstructions = 0;
        Staged = false;
        DoubleBuffered = false;
        SharedMemoryBytes = 0;
        LoopBarriers = 0;
        _reg = FixedRegisters;
        _reg64 = FixedRegisters64;
        _pred = FixedPredicates;

        // The staged lowering is preferred wherever it applies. It is not a tuning option:
        // the naive one moves O(M*N*K) operand bytes and collapses from 11.8 to 3.0 TFLOP/s
        // when the reused bands outgrow L2. Shapes it cannot cover fall back, and the naive
        // path stays correct at every shape.
        if (EnableStaging)
        {
            SelectWarpTile(plan);
            if (CanStage(plan, out _))
                return EmitStaged(spec, plan, computeMajor, computeMinor);
        }

        int aIndex = spec.ProductInputs[0], bIndex = spec.ProductInputs[1];
        int outIndex = spec.Inputs.Count;

        // wmma appeared in ISA 6.0; the shipped cubins are built at 7.1, which covers it.
        _sb.Append(".version 7.1\n")
           .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
           .Append(".address_size 64\n\n")
           .Append("// generated by PtxTensorCoreEmitter -- warp-collective wmma\n")
           .Append("// ").Append(spec.Describe().Replace("\n", "\n// ")).Append('\n')
           .Append(".visible .entry ").Append(spec.Name).Append("(\n");

        int paramCount = spec.ParameterCount;
        for (int i = 0; i < paramCount; i++)
            _sb.Append("    .param .u64 p").Append(I(i)).Append(i == paramCount - 1 ? "\n" : ",\n");
        _sb.Append(")\n{\n");

        _sb.Append("    .reg .pred   %p<4>;\n")
           .Append("    .reg .b32    %r<32>;\n")
           .Append("    .reg .f32    %f<32>;\n")
           .Append("    .reg .b64    %rd<24>;\n")
           .Append("    .reg .b32    %fa<").Append(I(FragmentRegisters)).Append(">;\n")
           .Append("    .reg .b32    %fb<").Append(I(FragmentRegisters)).Append(">;\n")
           .Append("    .reg .f32    %fc<").Append(I(FragmentRegisters)).Append(">;\n\n");

        L($"ld.param.u64 %rd0, [p{I(aIndex)}];");
        L($"ld.param.u64 %rd1, [p{I(bIndex)}];");
        L($"ld.param.u64 %rd2, [p{I(outIndex)}];");

        // WARP INDEX, NOT THREAD INDEX. Every lane of a warp must reach the same tile and
        // the same wmma instructions -- these are warp-synchronous, and a divergent warp
        // executing one is undefined. Deriving the tile from tid/32 guarantees that, and it
        // also makes the bounds guard safe: all 32 lanes agree on the comparison.
        L("mov.u32 %r0, %ctaid.x;");
        L("mov.u32 %r1, %tid.x;");
        L("shr.u32 %r2, %r1, 5;                    // lane's warp within the block");
        L($"mad.lo.u32 %r3, %r0, {I(WarpsPerBlock)}, %r2;   // global warp = output tile");

        int tilesN = plan.N / TileN;
        L($"setp.ge.u32 %p0, %r3, {I(plan.TileCount)};");
        L("@%p0 bra END;");

        L($"div.u32 %r4, %r3, {I(tilesN)};          // tile row");
        L($"rem.u32 %r5, %r3, {I(tilesN)};          // tile column");

        // Accumulator starts at zero. wmma.mma reads C and writes D, so the running sum
        // lives in the same fragment across the whole K walk and never touches memory.
        for (int f = 0; f < FragmentRegisters; f++)
            L($"mov.f32 %fc{I(f)}, 0f00000000;");

        // Base addresses of the tile's first K step. A's row block advances by
        // tileRow*16 rows; B's column block by tileCol*16 columns. Two bytes an element.
        int aLeading = plan.ARowMajor ? plan.K : plan.M;
        int bLeading = plan.BRowMajor ? plan.N : plan.K;

        if (plan.ARowMajor)
            L($"mul.lo.u32 %r6, %r4, {I(TileM * aLeading)};   // A row block, elements");
        else
            L($"mul.lo.u32 %r6, %r4, {I(TileM)};              // A column block (transposed)");

        if (plan.BRowMajor)
            L($"mul.lo.u32 %r7, %r5, {I(TileN)};              // B column block, elements");
        else
            L($"mul.lo.u32 %r7, %r5, {I(TileN * bLeading)};   // B row block (transposed)");

        L("mul.wide.u32 %rd3, %r6, 2;");
        L("add.u64 %rd4, %rd0, %rd3;                // A tile base");
        L("mul.wide.u32 %rd5, %r7, 2;");
        L("add.u64 %rd6, %rd1, %rd5;                // B tile base");

        // Per K step, A advances 16 columns and B advances 16 rows -- or the transpose of
        // each. Both are compile-time constants, so the walk is pointer bumps, not
        // recomputed addresses.
        int aStepElements = plan.ARowMajor ? TileK : TileK * aLeading;
        int bStepElements = plan.BRowMajor ? TileK * bLeading : TileK;

        int steps = plan.K / TileK;
        string aLayout = plan.ARowMajor ? "row" : "col";
        string bLayout = plan.BRowMajor ? "row" : "col";
        string fragA = Fragment("%fa"), fragB = Fragment("%fb"), fragC = Fragment("%fc");

        // Unrolling matters more here than on the scalar path: each step is three
        // instructions, so a runtime loop's compare-and-branch is a large fraction of the
        // body, and an unrolled walk lets ptxas overlap the loads of step i+1 with the mma
        // of step i. Long contractions still need a real loop or ptxas runs out of room.
        const int UnrollLimit = 64;
        Unrolled = steps <= UnrollLimit;

        if (!Unrolled)
        {
            L($"mov.u32 %r8, {I(steps)};");
            _sb.Append("KLOOP:\n");
        }

        int emitted = Unrolled ? steps : 1;
        for (int s = 0; s < emitted; s++)
        {
            L($"wmma.load.a.sync.aligned.{aLayout}.m16n16k16.global.f16 {fragA}, [%rd4], {I(aLeading)};");
            L($"wmma.load.b.sync.aligned.{bLayout}.m16n16k16.global.f16 {fragB}, [%rd6], {I(bLeading)};");
            L($"wmma.mma.sync.aligned.{aLayout}.{bLayout}.m16n16k16.f32.f32 {fragC}, {fragA}, {fragB}, {fragC};");
            MmaInstructions++;

            if (s < emitted - 1 || !Unrolled)
            {
                L($"add.u64 %rd4, %rd4, {I(aStepElements * 2)};");
                L($"add.u64 %rd6, %rd6, {I(bStepElements * 2)};");
            }
        }

        if (!Unrolled)
        {
            L("sub.u32 %r8, %r8, 1;");
            L("setp.gt.u32 %p1, %r8, 0;");
            L("@%p1 bra KLOOP;");
        }

        // THE EPILOGUE IS THE POINT. cuBLAS cannot fuse through its own call boundary, so
        // an activation costs it a kernel launch and a full round trip of the output. Here
        // it is applied element-wise to the accumulator fragment, still in registers.
        //
        // Element-wise is exactly what makes this legal without knowing the fragment
        // layout: a function applied to every element does not care which lane holds which.
        // A row-wise epilogue -- a softmax, a per-row bias -- would care, and is refused by
        // TryPlan rather than approximated here.
        if (spec.Activation != CodegenActivationKind.None)
            for (int f = 0; f < FragmentRegisters; f++)
                EmitActivation(spec.Activation, $"%fc{I(f)}");

        if (spec.ReduceScale != 1.0)
            for (int f = 0; f < FragmentRegisters; f++)
                L($"mul.f32 %fc{I(f)}, %fc{I(f)}, {F32(spec.ReduceScale)};");

        L($"mul.lo.u32 %r9, %r4, {I(TileM * plan.N)};");
        L($"mul.lo.u32 %r10, %r5, {I(TileN)};");
        L("add.u32 %r11, %r9, %r10;");
        L("mul.wide.u32 %rd7, %r11, 4;");
        L("add.u64 %rd8, %rd2, %rd7;");
        L($"wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%rd8], {fragC}, {I(plan.N)};");

        _sb.Append("END:\n    ret;\n}\n");
        return _sb.ToString();
    }

    private static string Fragment(string prefix)
    {
        var sb = new StringBuilder("{");
        for (int f = 0; f < FragmentRegisters; f++)
        {
            if (f > 0) sb.Append(", ");
            sb.Append(prefix).Append(I(f));
        }
        return sb.Append('}').ToString();
    }

    /// <summary>
    /// A hexadecimal fp32 literal. PTX will not take a decimal one, and net471 has no
    /// <c>BitConverter.SingleToUInt32Bits</c>.
    /// </summary>
    private static string F32(double value)
    {
        byte[] bits = BitConverter.GetBytes((float)value);
        uint raw = (uint)(bits[0] | (bits[1] << 8) | (bits[2] << 16) | (bits[3] << 24));
        return "0f" + raw.ToString("X8", CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// The element-wise epilogues, written against a single accumulator register.
    /// </summary>
    /// <remarks>
    /// These mirror <see cref="PtxAffineEmitter"/>'s forms deliberately -- the same
    /// approximation choices, so a kernel does not change its numerical answer depending on
    /// which lowering happened to be picked. In particular tanh is built from
    /// <c>ex2.approx</c> rather than <c>tanh.approx.f32</c>, which is only available from
    /// sm_75 and would silently restrict this path to newer devices than wmma needs.
    /// </remarks>
    private void EmitActivation(CodegenActivationKind kind, string reg)
    {
        switch (kind)
        {
            case CodegenActivationKind.ReLU:
                L($"max.f32 {reg}, {reg}, 0f00000000;");
                break;

            case CodegenActivationKind.Sigmoid:
                EmitSigmoid(reg, reg);
                break;

            case CodegenActivationKind.Tanh:
                // tanh(x) = 2*sigmoid(2x) - 1
                L($"mul.f32 %f20, {reg}, 0f40000000;");
                EmitSigmoid("%f20", "%f21");
                L($"fma.rn.f32 {reg}, %f21, 0f40000000, 0fBF800000;");
                break;

            case CodegenActivationKind.Swish:
                EmitSigmoid(reg, "%f22");
                L($"mul.f32 {reg}, {reg}, %f22;");
                break;

            case CodegenActivationKind.Reciprocal:
                L($"rcp.approx.f32 {reg}, {reg};");
                break;

            case CodegenActivationKind.Rsqrt:
                L($"rsqrt.approx.f32 {reg}, {reg};");
                break;

            case CodegenActivationKind.Gelu:
                // The tanh approximation, matching the affine emitter exactly.
                L($"mul.f32 %f23, {reg}, {reg};");
                L($"mul.f32 %f24, %f23, {reg};");
                L($"fma.rn.f32 %f25, %f24, {F32(0.044715)}, {reg};");
                L($"mul.f32 %f26, %f25, {F32(0.7978845608028654)};");
                L("mul.f32 %f27, %f26, 0f40000000;");
                EmitSigmoid("%f27", "%f28");
                L("fma.rn.f32 %f29, %f28, 0f40000000, 0fBF800000;");
                L("add.f32 %f30, %f29, 0f3F800000;");
                L($"mul.f32 %f31, {reg}, 0f3F000000;");
                L($"mul.f32 {reg}, %f31, %f30;");
                break;

            default:
                throw new NotSupportedException(
                    "Activation " + kind + " has no tensor-core epilogue form.");
        }
    }

    /// <summary>sigmoid(x) = 1 / (1 + exp(-x)), via the hardware base-2 exponential.</summary>
    private void EmitSigmoid(string source, string destination)
    {
        L($"mul.f32 %f16, {source}, {F32(-1.4426950408889634)};");   // -x * log2(e)
        L("ex2.approx.f32 %f17, %f16;");
        L("add.f32 %f18, %f17, 0f3F800000;");
        L($"rcp.approx.f32 {destination}, %f18;");
    }
}
