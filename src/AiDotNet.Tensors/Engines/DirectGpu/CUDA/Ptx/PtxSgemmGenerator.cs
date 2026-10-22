using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// The genome of a generated FP32 GEMM (#1262, phase B of #377): every field changes the emitted PTX or its launch
/// geometry, so <see cref="PtxSgemmGenerator"/> plus a configuration is a complete kernel and an evolutionary search
/// (#1263) can explore the space directly.
/// </summary>
/// <param name="BlockM">Rows of C per thread block.</param>
/// <param name="BlockN">Columns of C per thread block.</param>
/// <param name="BlockK">Depth of each K tile staged in shared memory.</param>
/// <param name="ThreadM">Rows of C each thread accumulates in registers.</param>
/// <param name="ThreadN">Columns of C each thread accumulates in registers.</param>
/// <param name="VectorWidth">Elements per global load / store: 1 (any shape) or 4 (16-byte, needs aligned shapes).</param>
/// <param name="SharedPad">Extra floats per shared-memory row, to spread bank accesses.</param>
/// <param name="DoubleBuffer">Prefetch the next K tile into registers while computing the current one.</param>
/// <param name="SplitTile">Split each thread's TM x TN tile into 4 x 4 fragments spread across the block (fragment g of
/// a thread sits g * Threads * 4 rows or columns from its first), so a warp's 16-byte shared reads cover consecutive
/// banks and its stores consecutive addresses, instead of each thread owning one contiguous run.</param>
/// <param name="Swizzle">Grouped block rasterization: consecutive blocks walk <c>Swizzle</c> block rows before moving
/// to the next block column, so they share A and B tiles in L2. 1 is plain row order.</param>
/// <param name="MinBlocksPerSM">Blocks each multiprocessor must be able to hold (<c>.minnctapersm</c>): 2 caps the
/// registers ptxas may allocate so two blocks are resident; 1 leaves the register count to ptxas.</param>
/// <param name="AsyncStages">0 stages K tiles through registers (any compute capability); 2 to 4 streams them straight
/// into that many shared-memory stages with <c>cp.async</c> (SM 8.0+), with no staging registers: 16-byte copies for
/// an operand contiguous along M/N, 4-byte copies (transposing on the way in) for one contiguous along K.</param>
/// <param name="RegisterTranspose">With <c>AsyncStages</c>, an operand contiguous along K is not copied element by
/// element: each thread loads 16-byte vectors into registers while the current stage computes and stores them
/// transposed into the stage being refilled afterwards (as CUTLASS does for a K-contiguous SIMT operand).</param>
/// <param name="FragmentPipeline">With <c>AsyncStages</c>, the shared-memory fragments are double-buffered in
/// registers: each k step's fragments load while the previous step's FMAs run, and the K tile's wait and barrier sit
/// before its last step's FMAs, so the next tile's first fragments load under them (CUTLASS's warp-level pipeline).
/// Without it every tile starts with the barrier and then waits for its first fragments.</param>
internal readonly record struct PtxGemmConfiguration(
    int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN, int VectorWidth, int SharedPad, bool DoubleBuffer,
    bool SplitTile = false, int Swizzle = 1, int MinBlocksPerSM = 1, int AsyncStages = 0, bool RegisterTranspose = false,
    bool FragmentPipeline = false)
{
    internal int ThreadsN => BlockN / ThreadN;
    internal int ThreadsM => BlockM / ThreadM;
    internal int Threads => ThreadsM * ThreadsN;
    internal int PitchA => BlockM + SharedPad;
    internal int PitchB => BlockN + SharedPad;
    internal int Stages => AsyncStages > 0 ? AsyncStages : DoubleBuffer ? 2 : 1;
    internal bool IsAsync => AsyncStages > 0;
    internal int SharedBytes => checked(Stages * BlockK * (PitchA + PitchB) * sizeof(float));

    /// <summary>A short stable name, used in kernel identities and evolution records.</summary>
    internal string Name => string.Format(CultureInfo.InvariantCulture, "b{0}x{1}x{2}-t{3}x{4}-v{5}-p{6}-{7}{8}{9}{10}{11}",
        BlockM, BlockN, BlockK, ThreadM, ThreadN, VectorWidth, SharedPad,
        IsAsync ? "as" + AsyncStages.ToString(CultureInfo.InvariantCulture) + (RegisterTranspose ? "r" : "") : DoubleBuffer ? "db" : "sb",
        SplitTile ? "-split" : "", Swizzle > 1 ? "-g" + Swizzle.ToString(CultureInfo.InvariantCulture) : "",
        MinBlocksPerSM > 1 ? "-o" + MinBlocksPerSM.ToString(CultureInfo.InvariantCulture) : "", FragmentPipeline ? "-fp" : "");

    // The thread tile's fragment granularity: 4 x 4 fragments when split, one TM x TN run otherwise.
    internal int FragmentM => SplitTile ? 4 : ThreadM;
    internal int FragmentN => SplitTile ? 4 : ThreadN;

    // Offset, in rows of the block tile, of a thread's i-th accumulator row from its first (likewise columns).
    internal int RowOffset(int i) => i / FragmentM * ThreadsM * FragmentM + i % FragmentM;
    internal int ColumnOffset(int j) => j / FragmentN * ThreadsN * FragmentN + j % FragmentN;

    /// <summary>
    /// The emit-able configurations over the genome's axes: the domain the evolutionary search (#1263) explores.
    /// </summary>
    internal static IEnumerable<PtxGemmConfiguration> SearchSpace()
    {
        foreach (int bm in new[] { 32, 64, 128, 256 })
        foreach (int bn in new[] { 32, 64, 128, 256 })
        foreach (int bk in new[] { 8, 16, 32 })
        foreach (int tm in new[] { 4, 8 })
        foreach (int tn in new[] { 4, 8 })
        foreach (int vw in new[] { 1, 4 })
        foreach (int pad in new[] { 0, 4 })
        foreach (bool db in new[] { false, true })
        foreach (bool split in new[] { false, true })
        foreach (int swizzle in Swizzles)
        foreach (int occupancy in new[] { 1, 2 })
        foreach (int async in AsyncStageCounts)
        foreach (bool transpose in new[] { false, true })
        {
            var config = new PtxGemmConfiguration(bm, bn, bk, tm, tn, vw, pad, db, split, swizzle, occupancy, async, transpose);
            if (config.Invalidity() is null) yield return config;
        }
    }

    internal static readonly int[] Swizzles = { 1, 4, 8, 16 };
    internal static readonly int[] AsyncStageCounts = { 0, 2, 3, 4 };

    // Register-staged kernels stay within the 48 KiB every supported device gives a block statically; cp.async kernels
    // (SM 8.0+) take dynamic shared memory up to the 99 KiB an SM 8.6 block may opt into.
    internal const int MaxStaticSharedBytes = 48 * 1024;
    internal const int MaxDynamicSharedBytes = 99 * 1024;

    /// <summary>Why this configuration cannot be emitted, or null when it can.</summary>
    internal string? Invalidity()
    {
        if (BlockM <= 0 || BlockN <= 0 || BlockK <= 0 || ThreadM <= 0 || ThreadN <= 0) return "non-positive dimension";
        if (VectorWidth is not (1 or 4)) return "vector width must be 1 or 4";
        if (SharedPad < 0 || SharedPad % 4 != 0) return "shared pad must be a non-negative multiple of 4";
        if (BlockM % ThreadM != 0 || BlockN % ThreadN != 0) return "thread tile must divide the block tile";
        if (ThreadM % 4 != 0 || ThreadN % 4 != 0) return "thread tile must be a multiple of 4 (vector shared loads)";
        if (Threads < 32 || Threads > 1024 || Threads % 32 != 0) return "thread count must be a multiple of 32 in [32, 1024]";
        if (BlockK % VectorWidth != 0) return "vector width must divide the K tile";
        if (BlockM * BlockK % (Threads * VectorWidth) != 0 || BlockN * BlockK % (Threads * VectorWidth) != 0)
            return "each thread must load a whole number of vectors of both tiles";
        if (Array.IndexOf(AsyncStageCounts, AsyncStages) < 0) return "async stages must be 0, 2, 3 or 4";
        if (IsAsync && DoubleBuffer) return "the cp.async pipeline replaces register double buffering";
        if (IsAsync && VectorWidth != 4) return "cp.async needs the 16-byte-aligned (vector width 4) admission";
        if (RegisterTranspose && !IsAsync) return "register transposition is a cp.async pipeline option";
        if (FragmentPipeline && !IsAsync) return "the fragment pipeline is a cp.async pipeline option";
        if (FragmentPipeline && BlockK % 2 != 0) return "the fragment pipeline alternates two register sets over an even K tile";
        if (SharedBytes > (IsAsync ? MaxDynamicSharedBytes : MaxStaticSharedBytes))
            return IsAsync ? "shared memory exceeds 99 KiB" : "shared memory exceeds 48 KiB";
        if (ThreadM * ThreadN > 128) return "more than 128 accumulators per thread";
        if (Array.IndexOf(Swizzles, Swizzle) < 0) return "swizzle must be 1, 4, 8 or 16";
        if (SplitTile && ThreadM == 4 && ThreadN == 4) return "a 4 x 4 thread tile is already one fragment";
        if (MinBlocksPerSM is not (1 or 2)) return "min blocks per multiprocessor must be 1 or 2";
        // An SM 8.6/8.9 multiprocessor holds 100 KiB of shared memory, less 1 KiB the driver reserves per block.
        if (MinBlocksPerSM > 1 && (Threads * MinBlocksPerSM > 1024 || (SharedBytes + 1024) * MinBlocksPerSM > 100 * 1024))
            return "two blocks of this size do not fit one multiprocessor";
        return null;
    }
}

/// <summary>
/// Operand, output and arithmetic types of a generated GEMM (#1265: the cuBLAS-free fallback covers every cuBLAS GEMM the
/// backend issues). FP32 is the tuned path; the others are register-staged with scalar global loads. FP16 operands are
/// widened to FP32 on load and accumulate in FP32 (cuBLAS's <c>CUBLAS_COMPUTE_32F</c>), FP64 runs entirely in FP64.
/// </summary>
internal enum PtxGemmPrecision
{
    /// <summary>FP32 operands, FP32 output, FP32 arithmetic.</summary>
    Fp32 = 0,
    /// <summary>FP16 operands, FP32 output, FP32 arithmetic.</summary>
    Fp16InFp32Out = 1,
    /// <summary>FP16 operands, FP16 output, FP32 arithmetic.</summary>
    Fp16InFp16Out = 2,
    /// <summary>FP64 operands, FP64 output, FP64 arithmetic.</summary>
    Fp64 = 3,
    /// <summary>BF16 operands, FP32 output, FP32 arithmetic.</summary>
    Bf16InFp32Out = 4,
}

internal static class PtxGemmPrecisionFacts
{
    internal static int OperandBytes(this PtxGemmPrecision precision) => precision switch
    {
        PtxGemmPrecision.Fp16InFp32Out or PtxGemmPrecision.Fp16InFp16Out or PtxGemmPrecision.Bf16InFp32Out => 2,
        PtxGemmPrecision.Fp64 => 8,
        _ => 4,
    };

    internal static int OutputBytes(this PtxGemmPrecision precision) => precision switch
    {
        PtxGemmPrecision.Fp16InFp16Out => 2,
        PtxGemmPrecision.Fp64 => 8,
        _ => 4,
    };

    // Shared-memory elements and registers: FP64 for FP64, FP32 otherwise.
    internal static int ComputeBytes(this PtxGemmPrecision precision) => precision == PtxGemmPrecision.Fp64 ? 8 : 4;

    internal static int SharedBytes(this PtxGemmPrecision precision, PtxGemmConfiguration configuration) =>
        checked(configuration.SharedBytes / sizeof(float) * precision.ComputeBytes());

    /// <summary>Why a configuration cannot run at this precision, or null when it can.</summary>
    internal static string? Invalidity(this PtxGemmPrecision precision, PtxGemmConfiguration configuration)
    {
        if (!Enum.IsDefined(typeof(PtxGemmPrecision), precision)) return "unknown precision";
        if (precision == PtxGemmPrecision.Fp32) return null;
        if (configuration.IsAsync) return "only FP32 runs the cp.async pipeline";
        if (configuration.VectorWidth != 1) return "only FP32 loads vectors";
        if (precision.SharedBytes(configuration) > PtxGemmConfiguration.MaxStaticSharedBytes)
            return "shared memory exceeds 48 KiB at this precision";
        return null;
    }
}

/// <summary>The operand layouts and the K-contiguity they imply.</summary>
internal readonly record struct PtxGemmLayout(bool TransposeA, bool TransposeB)
{
    // A is M x K (row-major, contiguous along K) unless transposed (stored K x M, contiguous along M).
    internal bool AContiguousAlongK => !TransposeA;
    // B is K x N (row-major, contiguous along N) unless transposed (stored N x K, contiguous along K).
    internal bool BContiguousAlongK => TransposeB;
    internal string Name => (TransposeA ? "t" : "n") + (TransposeB ? "t" : "n");
}

/// <summary>
/// Emits a register-blocked FP32 GEMM, <c>C = alpha * op(A) @ op(B) + beta * C</c>, batched over grid z, with M, N,
/// K, the leading dimensions and the batch strides as kernel arguments, so one compiled configuration serves every
/// shape. Each block stages BK-deep tiles of A and B in shared memory (both K-major, so the inner product reads
/// vectors), each thread accumulates a TM x TN register tile, and edges are predicated: loads outside the matrices
/// read zero and stores outside are skipped. C is never read when beta is zero (cuBLAS semantics).
/// </summary>
internal static class PtxSgemmGenerator
{
    internal const string EntryPoint = "aidotnet_sgemm";

    // Kernel parameters, in launch order.
    internal static readonly string[] ParameterNames =
    {
        "a_ptr", "b_ptr", "c_ptr", "m", "n", "k", "lda", "ldb", "ldc", "alpha", "beta", "stride_a", "stride_b", "stride_c",
    };

    internal static string Emit(PtxGemmConfiguration config, PtxGemmLayout layout, int ccMajor, int ccMinor,
        PtxGemmPrecision precision = PtxGemmPrecision.Fp32)
    {
        string? invalid = config.Invalidity() ?? precision.Invalidity(config);
        if (invalid is not null) throw new ArgumentException("Invalid GEMM configuration " + config.Name + ": " + invalid, nameof(config));
        if (config.IsAsync && ccMajor < 8)
            throw new PlatformNotSupportedException("cp.async needs compute capability 8.0 or newer.");
        return new Emitter(config, layout, precision).Build(ccMajor, ccMinor);
    }

    /// <summary>
    /// Whether a configuration may run a problem: a 16-byte vector configuration needs every vector it loads or stores
    /// to be wholly inside or wholly outside the matrix and 16-byte aligned.
    /// </summary>
    internal static bool Admits(PtxGemmConfiguration config, PtxGemmLayout layout, int m, int n, int k,
        int lda, int ldb, int ldc, long strideA, long strideB, long strideC, PtxGemmPrecision precision = PtxGemmPrecision.Fp32)
    {
        if (config.Invalidity() is not null || precision.Invalidity(config) is not null || m <= 0 || n <= 0 || k < 0)
            return false;
        if (config.VectorWidth == 1) return true;
        int aContiguousExtent = layout.AContiguousAlongK ? k : m;
        int bContiguousExtent = layout.BContiguousAlongK ? k : n;
        return aContiguousExtent % 4 == 0 && bContiguousExtent % 4 == 0 && n % 4 == 0 &&
               lda % 4 == 0 && ldb % 4 == 0 && ldc % 4 == 0 &&
               strideA % 4 == 0 && strideB % 4 == 0 && strideC % 4 == 0;
    }

    private sealed class Emitter
    {
        private readonly PtxGemmConfiguration _c;
        private readonly PtxGemmLayout _layout;
        private readonly PtxGemmPrecision _p;
        private readonly int _ob, _cb, _outb;  // operand, compute (shared and register) and output element bytes
        private readonly bool _f64;
        private readonly StringBuilder _body = new(64_000);
        private int _r32, _r64, _pred, _f, _fd, _h;

        internal Emitter(PtxGemmConfiguration config, PtxGemmLayout layout, PtxGemmPrecision precision)
        {
            _c = config;
            _layout = layout;
            _p = precision;
            _ob = precision.OperandBytes();
            _cb = precision.ComputeBytes();
            _outb = precision.OutputBytes();
            _f64 = precision == PtxGemmPrecision.Fp64;
        }

        // The compute type: its PTX suffix, a fresh register and its zero literal.
        private string Ct => _f64 ? "f64" : "f32";
        private string CF() => _f64 ? "%fd" + (_fd++).ToString(CultureInfo.InvariantCulture) : F();
        private string Zero => _f64 ? "0d0000000000000000" : "0f00000000";
        private string H() => "%h" + (_h++).ToString(CultureInfo.InvariantCulture);
        private static int Log2(int bytes) => bytes switch { 2 => 1, 4 => 2, 8 => 3, _ => throw new ArgumentOutOfRangeException(nameof(bytes)) };

        private string R() => "%r" + (_r32++).ToString(CultureInfo.InvariantCulture);
        private string Rd() => "%rd" + (_r64++).ToString(CultureInfo.InvariantCulture);
        private string P() => "%p" + (_pred++).ToString(CultureInfo.InvariantCulture);
        private string F() => "%f" + (_f++).ToString(CultureInfo.InvariantCulture);
        private void L(string line) => _body.Append("    ").AppendLine(line);
        private void Label(string label) => _body.Append(label).AppendLine(":");
        private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);
        private static string I(long value) => value.ToString(CultureInfo.InvariantCulture);

        // One operand's per-thread load slots: each slot is VW contiguous elements at a fixed (k, mn) tile offset.
        private sealed class Operand
        {
            internal string Name = "";
            internal bool ContiguousAlongK;
            internal int MnTile;
            internal int Pitch;
            internal string SharedSymbol = "";
            internal string[] SlotPointer = Array.Empty<string>();  // u64 global address of the slot for the current tile
            internal string[] SlotMnInBounds = Array.Empty<string>();  // predicate: the slot's mn row/column exists
            internal int[] SlotK = Array.Empty<int>();
            internal int[] SlotMn = Array.Empty<int>();
            internal string[][] Staged = Array.Empty<string[]>();  // registers holding the slot's VW loaded values
            internal string Step = "";  // u64 bytes to advance every slot pointer by one K tile
            internal bool ViaRegisters;  // staged through registers (always, or a transposed operand of a cp.async kernel)
            // Affine slots: slot s sits a fixed (DeltaK, DeltaMn) * s from slot 0 in the tile, so only slot 0 keeps
            // registers (pointer, k and mn offsets, shared offset); slot s's global address is slot 0's plus s * Stride
            // (a u64 register: ld enters it), its k offset and shared offset slot 0's plus constants.
            internal bool Affine;
            internal int DeltaK, DeltaMn;
            internal string Stride = "";
        }

        // The global address of `op`'s slot s for the current tile: a register (slot 0, or a non-affine slot), or one
        // computed into a fresh register from slot 0's.
        private string SlotAddress(Operand op, int s)
        {
            if (!op.Affine || s == 0) return op.SlotPointer[s];
            string address = Rd();
            L($"mad.lo.u64 {address}, {op.Stride}, {I(s)}, {op.SlotPointer[0]};");
            return address;
        }

        // k0 + the k offset of `op`'s slot s, into a fresh register.
        private string SlotKGlobal(Operand op, int s, string k0)
        {
            string kGlobal = R();
            int slot = op.Affine ? 0 : s;
            L($"add.u32 {kGlobal}, {k0}, {_slotK[(op.Name, slot)]};");
            if (op.Affine && s * op.DeltaK != 0) L($"add.u32 {kGlobal}, {kGlobal}, {I(s * op.DeltaK)};");
            return kGlobal;
        }

        // The register holding a slot-relative offset (k, mn or shared bytes) and the constant to add for slot s.
        private (string Register, int Constant) SlotOffset(Dictionary<(string, int), string> offsets, Operand op, int s, int perSlot) =>
            op.Affine ? (offsets[(op.Name, 0)], s * perSlot) : (offsets[(op.Name, s)], 0);

        internal string Build(int ccMajor, int ccMinor)
        {
            int threads = _c.Threads;
            // Kernel arguments.
            string aPtr = Rd(), bPtr = Rd(), cPtr = Rd(), strideA = Rd(), strideB = Rd(), strideC = Rd();
            string m = R(), n = R(), k = R(), lda = R(), ldb = R(), ldc = R();
            string alpha = CF(), beta = CF();
            L($"ld.param.u64 {aPtr}, [a_ptr];");
            L($"ld.param.u64 {bPtr}, [b_ptr];");
            L($"ld.param.u64 {cPtr}, [c_ptr];");
            L($"ld.param.u32 {m}, [m];");
            L($"ld.param.u32 {n}, [n];");
            L($"ld.param.u32 {k}, [k];");
            L($"ld.param.u32 {lda}, [lda];");
            L($"ld.param.u32 {ldb}, [ldb];");
            L($"ld.param.u32 {ldc}, [ldc];");
            L($"ld.param.{Ct} {alpha}, [alpha];");
            L($"ld.param.{Ct} {beta}, [beta];");
            L($"ld.param.u64 {strideA}, [stride_a];");
            L($"ld.param.u64 {strideB}, [stride_b];");
            L($"ld.param.u64 {strideC}, [stride_c];");

            // Thread and block coordinates.
            string tid = R(), tx = R(), ty = R(), bm = R(), bn = R(), batch = R();
            L($"mov.u32 {tid}, %tid.x;");
            L($"rem.u32 {tx}, {tid}, {I(_c.ThreadsN)};");
            L($"div.u32 {ty}, {tid}, {I(_c.ThreadsN)};");
            string ctaX = R(), ctaY = R();
            L($"mov.u32 {ctaX}, %ctaid.x;");
            L($"mov.u32 {ctaY}, %ctaid.y;");
            L($"mov.u32 {batch}, %ctaid.z;");
            if (_c.Swizzle > 1)
            {
                // Grouped rasterization over the launch's linear block index: groups of Swizzle block rows, walked
                // column by column, so blocks that run together share A rows and B columns in L2.
                string gridX = R(), gridY = R(), linear = R(), perGroup = R(), group = R(), firstRow = R(), rows = R(),
                    inGroup = R();
                L($"mov.u32 {gridX}, %nctaid.x;");
                L($"mov.u32 {gridY}, %nctaid.y;");
                L($"mad.lo.u32 {linear}, {ctaY}, {gridX}, {ctaX};");
                L($"mul.lo.u32 {perGroup}, {gridX}, {I(_c.Swizzle)};");
                L($"div.u32 {group}, {linear}, {perGroup};");
                L($"mul.lo.u32 {firstRow}, {group}, {I(_c.Swizzle)};");
                L($"sub.u32 {rows}, {gridY}, {firstRow};");
                L($"min.u32 {rows}, {rows}, {I(_c.Swizzle)};");
                L($"rem.u32 {inGroup}, {linear}, {perGroup};");
                L($"rem.u32 {ctaY}, {inGroup}, {rows};");
                L($"add.u32 {ctaY}, {ctaY}, {firstRow};");
                L($"div.u32 {ctaX}, {inGroup}, {rows};");
            }
            L($"mul.lo.u32 {bn}, {ctaX}, {I(_c.BlockN)};");
            L($"mul.lo.u32 {bm}, {ctaY}, {I(_c.BlockM)};");

            // Batch offsets (element strides -> bytes).
            string batch64 = Rd();
            L($"cvt.u64.u32 {batch64}, {batch};");
            foreach (var (ptr, stride, bytes) in new[] { (aPtr, strideA, _ob), (bPtr, strideB, _ob), (cPtr, strideC, _outb) })
            {
                string off = Rd();
                L($"mul.lo.u64 {off}, {batch64}, {stride};");
                L($"shl.b64 {off}, {off}, {I(Log2(bytes))};");
                L($"add.u64 {ptr}, {ptr}, {off};");
            }

            var a = SetUpOperand("a", _layout.AContiguousAlongK, _c.BlockM, _c.PitchA, aPtr, lda, bm, m, tid);
            var b = SetUpOperand("b", _layout.BContiguousAlongK, _c.BlockN, _c.PitchB, bPtr, ldb, bn, n, tid);

            // Shared buffers (32-bit shared-window addresses) and the double-buffer selector. A cp.async kernel's
            // stages live in dynamic shared memory: every A stage, then every B stage.
            int tileBytesA = _c.BlockK * _c.PitchA * _cb, tileBytesB = _c.BlockK * _c.PitchB * _cb;
            string curA = R(), curB = R(), nxtA = R(), nxtB = R();
            if (_c.IsAsync)
            {
                L($"mov.u32 {curA}, smem;");
                L($"add.u32 {curB}, {curA}, {I(_c.AsyncStages * tileBytesA)};");
            }
            else
            {
                L($"mov.u32 {curA}, a_tile;");
                L($"mov.u32 {curB}, b_tile;");
            }
            L($"add.u32 {nxtA}, {curA}, {I(tileBytesA)};");
            L($"add.u32 {nxtB}, {curB}, {I(tileBytesB)};");

            // Accumulators.
            int accumulators = _c.ThreadM * _c.ThreadN;
            var acc = new string[accumulators];
            for (int i = 0; i < accumulators; i++)
            {
                acc[i] = CF();
                L($"mov.{Ct} {acc[i]}, {Zero};");
            }
            var fa = new string[_c.ThreadM];
            var fb = new string[_c.ThreadN];
            for (int i = 0; i < fa.Length; i++) fa[i] = CF();
            for (int i = 0; i < fb.Length; i++) fb[i] = CF();
            // The fragment pipeline's second register set.
            var fa2 = new string[_c.FragmentPipeline ? _c.ThreadM : 0];
            var fb2 = new string[_c.FragmentPipeline ? _c.ThreadN : 0];
            for (int i = 0; i < fa2.Length; i++) fa2[i] = CF();
            for (int i = 0; i < fb2.Length; i++) fb2[i] = CF();

            // Per-thread shared read offsets for the inner product: the thread's first A row ty*FM and B column tx*FN
            // (bytes); its other fragments sit at compile-time offsets (RowOffset / ColumnOffset).
            string aRead = R(), bRead = R();
            L($"mul.lo.u32 {aRead}, {ty}, {I(_c.FragmentM * _cb)};");
            L($"mul.lo.u32 {bRead}, {tx}, {I(_c.FragmentN * _cb)};");

            // Tile count; K == 0 skips the main loop (C = beta * C).
            string tiles = R(), tile = R(), k0 = R();
            L($"add.u32 {tiles}, {k}, {I(_c.BlockK - 1)};");
            L($"div.u32 {tiles}, {tiles}, {I(_c.BlockK)};");
            L($"mov.u32 {tile}, 0;");
            L($"mov.u32 {k0}, 0;");
            string empty = P();
            L($"setp.eq.u32 {empty}, {tiles}, 0;");
            L($"@{empty} bra EPILOGUE;");

            if (_c.IsAsync)
            {
                // Prologue: tiles 0 .. Stages-2 in flight, one commit group per tile (empty past the last tile).
                for (int s = 0; s < _c.AsyncStages - 1; s++)
                {
                    string issue = P();
                    L($"setp.gt.u32 {issue}, {tiles}, {I(s)};");
                    IssueTile(a, curA, I(s * tileBytesA), k0, k, issue);
                    IssueTile(b, curB, I(s * tileBytesB), k0, k, issue);
                    L("cp.async.commit_group;");
                    FinishTile(a, curA, I(s * tileBytesA));
                    FinishTile(b, curB, I(s * tileBytesB));
                    if (s < _c.AsyncStages - 2)
                    {
                        Advance(a);
                        Advance(b);
                        L($"add.u32 {k0}, {k0}, {I(_c.BlockK)};");
                    }
                }
                if (_c.FragmentPipeline)
                {
                    // Tile 0 lands, and its first fragments load ahead of the loop.
                    L($"cp.async.wait_group {I(_c.AsyncStages - 2)};");
                    L("bar.sync 0;");
                    string firstA = R(), firstB = R();
                    L($"add.u32 {firstA}, {curA}, {aRead};");
                    L($"add.u32 {firstB}, {curB}, {bRead};");
                    LoadFragments(fa, fb, firstA, firstB, 0);
                }
                Label("K_LOOP");
                // Tile `tile` has landed once at most Stages-2 younger groups are pending; the barrier also retires
                // every thread's reads of the stage the next issue overwrites (computed in the previous iteration).
                // The fragment pipeline waits at the end of the previous iteration instead (see ComputePipelined).
                if (!_c.FragmentPipeline)
                {
                    L($"cp.async.wait_group {I(_c.AsyncStages - 2)};");
                    L("bar.sync 0;");
                }
                string next = R(), issueStage = R(), hasNext = P(), nextK0 = R(), offA = R(), offB = R();
                L($"add.u32 {next}, {tile}, {I(_c.AsyncStages - 1)};");
                L($"setp.lt.u32 {hasNext}, {next}, {tiles};");
                L($"rem.u32 {issueStage}, {next}, {I(_c.AsyncStages)};");
                L($"mul.lo.u32 {nextK0}, {next}, {I(_c.BlockK)};");
                Advance(a);
                Advance(b);
                L($"mul.lo.u32 {offA}, {issueStage}, {I(tileBytesA)};");
                L($"mul.lo.u32 {offB}, {issueStage}, {I(tileBytesB)};");
                IssueTile(a, curA, offA, nextK0, k, hasNext);
                IssueTile(b, curB, offB, nextK0, k, hasNext);
                L("cp.async.commit_group;");
                string computeStage = R(), stageA = R(), stageB = R();
                L($"rem.u32 {computeStage}, {tile}, {I(_c.AsyncStages)};");
                L($"mad.lo.u32 {stageA}, {computeStage}, {I(tileBytesA)}, {curA};");
                L($"mad.lo.u32 {stageB}, {computeStage}, {I(tileBytesB)}, {curB};");
                if (_c.FragmentPipeline)
                {
                    string nextStage = R(), nextA = R(), nextB = R();
                    L($"add.u32 {nextStage}, {tile}, 1;");
                    L($"rem.u32 {nextStage}, {nextStage}, {I(_c.AsyncStages)};");
                    L($"mad.lo.u32 {nextA}, {nextStage}, {I(tileBytesA)}, {curA};");
                    L($"mad.lo.u32 {nextB}, {nextStage}, {I(tileBytesB)}, {curB};");
                    ComputePipelined(stageA, stageB, nextA, nextB, aRead, bRead, new[] { fa, fa2 }, new[] { fb, fb2 }, acc, () =>
                    {
                        // Before the barrier: the refilled stage's transposed stores (read after the barrier, next
                        // iteration at the earliest).
                        FinishTile(a, curA, offA);
                        FinishTile(b, curB, offB);
                    });
                }
                else
                {
                    Compute(stageA, stageB, aRead, bRead, fa, fb, acc);
                    // The refilled stage was last read in the previous iteration (the barrier above retired it) and is
                    // next read Stages-1 iterations on, after at least one more barrier.
                    FinishTile(a, curA, offA);
                    FinishTile(b, curB, offB);
                }
                string more = P();
                L($"add.u32 {tile}, {tile}, 1;");
                L($"setp.lt.u32 {more}, {tile}, {tiles};");
                L($"@{more} bra K_LOOP;");
                L("cp.async.wait_group 0;");
            }
            else if (_c.DoubleBuffer)
            {
                LoadTile(a, k0, k);
                LoadTile(b, k0, k);
                StoreTile(a, curA);
                StoreTile(b, curB);
                L("bar.sync 0;");
                Label("K_LOOP");
                string next = R(), nextK0 = R(), hasNext = P();
                L($"add.u32 {next}, {tile}, 1;");
                L($"add.u32 {nextK0}, {k0}, {I(_c.BlockK)};");
                L($"setp.lt.u32 {hasNext}, {next}, {tiles};");
                L($"@!{hasNext} bra NO_PREFETCH;");
                Advance(a);
                Advance(b);
                LoadTile(a, nextK0, k);
                LoadTile(b, nextK0, k);
                Label("NO_PREFETCH");
                Compute(curA, curB, aRead, bRead, fa, fb, acc);
                L($"@!{hasNext} bra TILE_DONE;");
                StoreTile(a, nxtA);
                StoreTile(b, nxtB);
                Label("TILE_DONE");
                L("bar.sync 0;");
                // Swap the buffers; advance the tile.
                string swap = R();
                L($"mov.u32 {swap}, {curA};");
                L($"mov.u32 {curA}, {nxtA};");
                L($"mov.u32 {nxtA}, {swap};");
                L($"mov.u32 {swap}, {curB};");
                L($"mov.u32 {curB}, {nxtB};");
                L($"mov.u32 {nxtB}, {swap};");
                L($"mov.u32 {tile}, {next};");
                L($"mov.u32 {k0}, {nextK0};");
                L($"@{hasNext} bra K_LOOP;");
            }
            else
            {
                string first = P();
                Label("K_LOOP");
                L($"setp.eq.u32 {first}, {tile}, 0;");
                L($"@{first} bra NO_ADVANCE;");
                Advance(a);
                Advance(b);
                Label("NO_ADVANCE");
                LoadTile(a, k0, k);
                LoadTile(b, k0, k);
                StoreTile(a, curA);
                StoreTile(b, curB);
                L("bar.sync 0;");
                Compute(curA, curB, aRead, bRead, fa, fb, acc);
                L("bar.sync 0;");
                string more = P();
                L($"add.u32 {tile}, {tile}, 1;");
                L($"add.u32 {k0}, {k0}, {I(_c.BlockK)};");
                L($"setp.lt.u32 {more}, {tile}, {tiles};");
                L($"@{more} bra K_LOOP;");
            }

            Label("EPILOGUE");
            Epilogue(cPtr, m, n, ldc, alpha, beta, bm, bn, tx, ty, acc);
            L("ret;");

            var ptx = new StringBuilder(_body.Length + 2_048);
            ptx.AppendLine(".version 7.1");
            ptx.Append(".target sm_").Append(I(ccMajor)).AppendLine(I(ccMinor));
            ptx.AppendLine(".address_size 64");
            ptx.Append("// sgemm ").Append(_c.Name).Append(' ').Append(_layout.Name).Append(' ').AppendLine(_p.ToString());
            if (_c.IsAsync) ptx.AppendLine(".extern .shared .align 16 .b8 smem[];");
            ptx.Append(".visible .entry ").Append(EntryPoint).AppendLine("(");
            ptx.AppendLine("    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 c_ptr,");
            ptx.AppendLine("    .param .u32 m, .param .u32 n, .param .u32 k, .param .u32 lda, .param .u32 ldb, .param .u32 ldc,");
            ptx.Append("    .param .").Append(Ct).Append(" alpha, .param .").Append(Ct).AppendLine(" beta,");
            ptx.AppendLine("    .param .u64 stride_a, .param .u64 stride_b, .param .u64 stride_c)");
            ptx.Append(".maxntid ").Append(I(threads)).AppendLine(", 1, 1");
            if (_c.MinBlocksPerSM > 1) ptx.Append(".minnctapersm ").AppendLine(I(_c.MinBlocksPerSM));
            ptx.AppendLine("{");
            ptx.Append("    .reg .pred %p<").Append(I(Math.Max(1, _pred))).AppendLine(">;");
            ptx.Append("    .reg .b32 %r<").Append(I(Math.Max(1, _r32))).AppendLine(">;");
            ptx.Append("    .reg .b64 %rd<").Append(I(Math.Max(1, _r64))).AppendLine(">;");
            ptx.Append("    .reg .f32 %f<").Append(I(Math.Max(1, _f))).AppendLine(">;");
            ptx.Append("    .reg .f64 %fd<").Append(I(Math.Max(1, _fd))).AppendLine(">;");
            ptx.Append("    .reg .b16 %h<").Append(I(Math.Max(1, _h))).AppendLine(">;");
            if (!_c.IsAsync)
            {
                ptx.Append("    .shared .align 16 .b8 a_tile[").Append(I(_c.Stages * tileBytesA)).AppendLine("];");
                ptx.Append("    .shared .align 16 .b8 b_tile[").Append(I(_c.Stages * tileBytesB)).AppendLine("];");
            }
            ptx.Append(_body);
            ptx.AppendLine("}");
            return ptx.ToString();
        }

        // Fixes each load slot's tile offset, its global pointer for tile 0 and the per-tile pointer step.
        private Operand SetUpOperand(string name, bool contiguousAlongK, int mnTile, int pitch, string basePtr, string ld,
            string blockMn, string mnLimit, string tid)
        {
            // A cp.async slot is 16 bytes along M/N, or one element along K (the copy transposes it into the K-major
            // tile); a register slot is the configuration's vector width either way.
            bool viaRegisters = !_c.IsAsync || (contiguousAlongK && _c.RegisterTranspose);
            int vw = viaRegisters ? _c.VectorWidth : contiguousAlongK ? 1 : 4;
            int slots = mnTile * _c.BlockK / (_c.Threads * vw);
            var op = new Operand
            {
                Name = name,
                ViaRegisters = viaRegisters,
                ContiguousAlongK = contiguousAlongK,
                MnTile = mnTile,
                Pitch = pitch,
                SharedSymbol = name + "_tile",
                SlotPointer = new string[slots],
                SlotMnInBounds = new string[slots],
                SlotK = new int[slots],
                SlotMn = new int[slots],
                Staged = new string[slots][],
            };
            string ld64 = Rd();
            L($"cvt.u64.u32 {ld64}, {ld};");
            // Consecutive slots are Threads * vw elements apart along the contiguous dimension; when that is a whole
            // number of rows of it, they differ by constant (k, mn) steps and share slot 0's registers.
            int slotStep = _c.Threads * vw, contiguous = contiguousAlongK ? _c.BlockK : mnTile;
            op.Affine = slots > 1 && slotStep % contiguous == 0;
            if (op.Affine)
            {
                op.DeltaK = contiguousAlongK ? 0 : slotStep / mnTile;
                op.DeltaMn = contiguousAlongK ? slotStep / _c.BlockK : 0;
                op.Stride = Rd();
                // Bytes from slot s to s + 1: DeltaMn rows of ld along K, DeltaK rows of ld along MN.
                L($"mul.lo.u64 {op.Stride}, {ld64}, {I((long)(contiguousAlongK ? op.DeltaMn : op.DeltaK) * _ob)};");
            }
            for (int s = 0; s < slots; s++)
            {
                op.Staged[s] = new string[viaRegisters ? vw : 0];
                for (int v = 0; v < op.Staged[s].Length; v++) op.Staged[s][v] = CF();
                op.SlotK[s] = s;
                op.SlotMn[s] = s;
                if (op.Affine && s > 0)
                {
                    // Only the bound: slot 0's mn plus s * DeltaMn must exist.
                    string mnSlot = R(), slotIn = P();
                    L($"add.u32 {mnSlot}, {blockMn}, {_slotMn[(name, 0)]};");
                    if (s * op.DeltaMn != 0) L($"add.u32 {mnSlot}, {mnSlot}, {I(s * op.DeltaMn)};");
                    L($"setp.lt.u32 {slotIn}, {mnSlot}, {mnLimit};");
                    op.SlotMnInBounds[s] = slotIn;
                    continue;
                }
                // Element index of the slot's first value in the tile, walked along the contiguous dimension. The
                // thread id enters at run time; the slot index at emit time.
                string linear = R();
                L($"mad.lo.u32 {linear}, {tid}, {I(vw)}, {I(s * _c.Threads * vw)};");
                string kOff = R(), mnOff = R();
                if (contiguousAlongK)
                {
                    L($"div.u32 {mnOff}, {linear}, {I(_c.BlockK)};");
                    L($"rem.u32 {kOff}, {linear}, {I(_c.BlockK)};");
                }
                else
                {
                    L($"div.u32 {kOff}, {linear}, {I(mnTile)};");
                    L($"rem.u32 {mnOff}, {linear}, {I(mnTile)};");
                }
                string mnGlobal = R();
                L($"add.u32 {mnGlobal}, {blockMn}, {mnOff};");
                string inBounds = P();
                L($"setp.lt.u32 {inBounds}, {mnGlobal}, {mnLimit};");
                // Element offset: (mn * ld + k) along K, (k * ld + mn) along MN; then bytes.
                string mn64 = Rd(), k64 = Rd(), offset = Rd(), ptr = Rd();
                L($"cvt.u64.u32 {mn64}, {mnGlobal};");
                L($"cvt.u64.u32 {k64}, {kOff};");
                if (contiguousAlongK) L($"mad.lo.u64 {offset}, {mn64}, {ld64}, {k64};");
                else L($"mad.lo.u64 {offset}, {k64}, {ld64}, {mn64};");
                L($"shl.b64 {offset}, {offset}, {I(Log2(_ob))};");
                L($"add.u64 {ptr}, {basePtr}, {offset};");
                op.SlotPointer[s] = ptr;
                op.SlotMnInBounds[s] = inBounds;
                // Keep the k offset and the mn offset as registers too (shared store address and the K bound).
                _slotK[(name, s)] = kOff;
                _slotMn[(name, s)] = mnOff;
                if (!viaRegisters)
                {
                    // The slot's byte offset within a stage of the K-major tile.
                    string shared = R();
                    L($"mad.lo.u32 {shared}, {kOff}, {I(pitch)}, {mnOff};");
                    L($"shl.b32 {shared}, {shared}, 2;");
                    _slotShared[(name, s)] = shared;
                    _slotBytes[name] = vw * sizeof(float);
                }
            }
            // One K tile further: BK elements along K, or BK rows of ld elements.
            op.Step = Rd();
            if (contiguousAlongK) L($"mov.u64 {op.Step}, {I((long)_c.BlockK * _ob)};");
            else L($"mul.lo.u64 {op.Step}, {ld64}, {I((long)_c.BlockK * _ob)};");
            return op;
        }

        private readonly Dictionary<(string, int), string> _slotK = new();
        private readonly Dictionary<(string, int), string> _slotMn = new();
        private readonly Dictionary<(string, int), string> _slotShared = new();
        private readonly Dictionary<string, int> _slotBytes = new();

        // A register-staged operand of a cp.async kernel: stores the values IssueTile loaded into buffer + stageOffset.
        private void FinishTile(Operand op, string buffer, string stageOffset)
        {
            if (!op.ViaRegisters) return;
            string stage = R();
            L($"add.u32 {stage}, {buffer}, {stageOffset};");
            StoreTile(op, stage);
        }

        // cp.async of every slot of the tile starting at k0 into the stage at buffer + stageOffset (a register or an
        // immediate). A slot outside the matrix, or past the last tile, copies zero bytes and zero-fills; a
        // register-staged operand loads instead (FinishTile stores it).
        private void IssueTile(Operand op, string buffer, string stageOffset, string k0, string k, string issue)
        {
            if (op.ViaRegisters)
            {
                // Loads only; out-of-range (or past-the-end) values read zero. FinishTile stores them.
                LoadTile(op, k0, k);
                return;
            }
            int bytes = _slotBytes[op.Name];
            for (int s = 0; s < op.SlotPointer.Length; s++)
            {
                string ok = P(), size = R(), dst = R();
                string kGlobal = SlotKGlobal(op, s, k0);
                L($"setp.lt.and.u32 {ok}, {kGlobal}, {k}, {op.SlotMnInBounds[s]};");
                L($"and.pred {ok}, {ok}, {issue};");
                L($"selp.u32 {size}, {I(bytes)}, 0, {ok};");
                var (shared, sharedConstant) = SlotOffset(_slotShared, op, s, (op.DeltaK * op.Pitch + op.DeltaMn) * sizeof(float));
                L($"add.u32 {dst}, {buffer}, {shared};");
                L($"add.u32 {dst}, {dst}, {stageOffset};");
                string source = SlotAddress(op, s);
                string at = sharedConstant != 0 ? dst + "+" + I(sharedConstant) : dst;
                // 16-byte copies may bypass L1 (.cg); 4-byte copies must go through it (.ca).
                L(bytes == 16
                    ? $"cp.async.cg.shared.global [{at}], [{source}], 16, {size};"
                    : $"cp.async.ca.shared.global [{at}], [{source}], {I(bytes)}, {size};");
            }
        }

        private void Advance(Operand op)
        {
            for (int s = 0; s < (op.Affine ? 1 : op.SlotPointer.Length); s++)
                L($"add.u64 {op.SlotPointer[s]}, {op.SlotPointer[s]}, {op.Step};");
        }

        // Global -> registers for the tile starting at k0; values outside the matrix are zero.
        private void LoadTile(Operand op, string k0, string k)
        {
            int vw = _c.VectorWidth;
            for (int s = 0; s < op.SlotPointer.Length; s++)
            {
                string ok = P();
                string kGlobal = SlotKGlobal(op, s, k0);
                L($"setp.lt.and.u32 {ok}, {kGlobal}, {k}, {op.SlotMnInBounds[s]};");
                string pointer = SlotAddress(op, s);
                foreach (string v in op.Staged[s]) L($"mov.{Ct} {v}, {Zero};");
                if (vw == 4)
                    L($"@{ok} ld.global.v4.f32 {{{op.Staged[s][0]}, {op.Staged[s][1]}, {op.Staged[s][2]}, {op.Staged[s][3]}}}, [{pointer}];");
                else if (_p == PtxGemmPrecision.Bf16InFp32Out)
                {
                    // BF16 operand: its bits are an FP32's upper half, so widening is a 16-bit shift.
                    string bits = R();
                    L($"mov.u32 {bits}, 0;");
                    L($"@{ok} ld.global.u16 {bits}, [{pointer}];");
                    L($"shl.b32 {bits}, {bits}, 16;");
                    L($"mov.b32 {op.Staged[s][0]}, {bits};");
                }
                else if (_ob == 2)
                {
                    // FP16 operand: widen to FP32 at load, so shared memory and the inner product stay FP32.
                    string half = H();
                    L($"@{ok} ld.global.b16 {half}, [{pointer}];");
                    L($"@{ok} cvt.f32.f16 {op.Staged[s][0]}, {half};");
                }
                else
                    L($"@{ok} ld.global.{Ct} {op.Staged[s][0]}, [{pointer}];");
            }
        }

        // Registers -> shared, K-major: element (k, mn) at (k * pitch + mn) * 4 bytes from the buffer base.
        private void StoreTile(Operand op, string buffer)
        {
            int vw = _c.VectorWidth;
            for (int s = 0; s < op.SlotPointer.Length; s++)
            {
                var (kOff, _) = SlotOffset(_slotK, op, s, op.DeltaK);
                var (mnOff, _) = SlotOffset(_slotMn, op, s, op.DeltaMn);
                string address = R();
                L($"mad.lo.u32 {address}, {kOff}, {I(op.Pitch)}, {mnOff};");
                L($"shl.b32 {address}, {address}, {I(Log2(_cb))};");
                L($"add.u32 {address}, {address}, {buffer};");
                // An affine slot's element sits (DeltaK * pitch + DeltaMn) * s elements past slot 0's.
                if (op.Affine && s > 0) L($"add.u32 {address}, {address}, {I((op.DeltaK * op.Pitch + op.DeltaMn) * s * _cb)};");
                if (op.ContiguousAlongK)
                {
                    // The vector runs along K: a column of the K-major tile, one row apart per element.
                    for (int v = 0; v < vw; v++)
                        L($"st.shared.{Ct} [{address}+{I(v * op.Pitch * _cb)}], {op.Staged[s][v]};");
                }
                else if (vw == 4)
                {
                    L($"st.shared.v4.f32 [{address}], {{{op.Staged[s][0]}, {op.Staged[s][1]}, {op.Staged[s][2]}, {op.Staged[s][3]}}};");
                }
                else
                {
                    L($"st.shared.{Ct} [{address}], {op.Staged[s][0]};");
                }
            }
        }

        // acc[i][j] += A[kk][row i] * B[kk][col j] over the K tile, reading 16-byte vectors from shared memory.
        private void Compute(string curA, string curB, string aRead, string bRead, string[] fa, string[] fb, string[] acc)
        {
            string aBase = R(), bBase = R();
            L($"add.u32 {aBase}, {curA}, {aRead};");
            L($"add.u32 {bBase}, {curB}, {bRead};");
            for (int kk = 0; kk < _c.BlockK; kk++)
            {
                int aRow = kk * _c.PitchA * _cb, bRow = kk * _c.PitchB * _cb;
                // Four consecutive elements per read: one 16-byte vector in FP32, two in FP64.
                for (int i = 0; i < fa.Length; i += 4)
                    LoadFragment(fa, i, aBase, aRow + _c.RowOffset(i) * _cb);
                for (int j = 0; j < fb.Length; j += 4)
                    LoadFragment(fb, j, bBase, bRow + _c.ColumnOffset(j) * _cb);
                for (int i = 0; i < fa.Length; i++)
                    for (int j = 0; j < fb.Length; j++)
                    {
                        string a = acc[i * fb.Length + j];
                        L($"fma.rn.{Ct} {a}, {fa[i]}, {fb[j]}, {a};");
                    }
            }
        }

        // One k step's fragments (row kk of the stage) into the given register set.
        private void LoadFragments(string[] fa, string[] fb, string aBase, string bBase, int kk)
        {
            int aRow = kk * _c.PitchA * _cb, bRow = kk * _c.PitchB * _cb;
            for (int i = 0; i < fa.Length; i += 4) LoadFragment(fa, i, aBase, aRow + _c.RowOffset(i) * _cb);
            for (int j = 0; j < fb.Length; j += 4) LoadFragment(fb, j, bBase, bRow + _c.ColumnOffset(j) * _cb);
        }

        // Compute with fragments double-buffered in registers: step kk's FMAs use set kk % 2 while step kk + 1's
        // fragments load into the other. Before the last step's FMAs, `beforeBarrier` runs, then the wait for the next
        // tile and the barrier, then the next tile's first fragments load (into set 0: the K tile is even) under the
        // last step's FMAs. The barrier retires every read of this tile's stage (its last fragments loaded before it),
        // so the next iteration's copy may refill it; on the last tile the extra fragment loads read a stale stage and
        // are never used.
        private void ComputePipelined(string curA, string curB, string nextA, string nextB, string aRead, string bRead,
            string[][] fa, string[][] fb, string[] acc, Action beforeBarrier)
        {
            string aBase = R(), bBase = R();
            L($"add.u32 {aBase}, {curA}, {aRead};");
            L($"add.u32 {bBase}, {curB}, {bRead};");
            for (int kk = 0; kk < _c.BlockK; kk++)
            {
                var curFa = fa[kk % 2];
                var curFb = fb[kk % 2];
                if (kk + 1 < _c.BlockK)
                {
                    LoadFragments(fa[(kk + 1) % 2], fb[(kk + 1) % 2], aBase, bBase, kk + 1);
                }
                else
                {
                    beforeBarrier();
                    L($"cp.async.wait_group {I(_c.AsyncStages - 2)};");
                    L("bar.sync 0;");
                    string nextABase = R(), nextBBase = R();
                    L($"add.u32 {nextABase}, {nextA}, {aRead};");
                    L($"add.u32 {nextBBase}, {nextB}, {bRead};");
                    LoadFragments(fa[0], fb[0], nextABase, nextBBase, 0);
                }
                for (int i = 0; i < curFa.Length; i++)
                    for (int j = 0; j < curFb.Length; j++)
                    {
                        string a = acc[i * curFb.Length + j];
                        L($"fma.rn.{Ct} {a}, {curFa[i]}, {curFb[j]}, {a};");
                    }
            }
        }

        private void LoadFragment(string[] registers, int first, string baseRegister, int offset)
        {
            if (_f64)
            {
                L($"ld.shared.v2.f64 {{{registers[first]}, {registers[first + 1]}}}, [{baseRegister}+{I(offset)}];");
                L($"ld.shared.v2.f64 {{{registers[first + 2]}, {registers[first + 3]}}}, [{baseRegister}+{I(offset + 16)}];");
            }
            else
            {
                L($"ld.shared.v4.f32 {{{registers[first]}, {registers[first + 1]}, {registers[first + 2]}, {registers[first + 3]}}}, [{baseRegister}+{I(offset)}];");
            }
        }

        // C = alpha * acc + beta * C, predicated on the matrix bounds; C is not read when beta == 0.
        private void Epilogue(string cPtr, string m, string n, string ldc, string alpha, string beta, string bm, string bn,
            string tx, string ty, string[] acc)
        {
            string useBeta = P();
            L($"setp.neu.{Ct} {useBeta}, {beta}, {Zero};");
            string row0 = R(), col0 = R(), ldc64 = Rd();
            L($"mad.lo.u32 {row0}, {ty}, {I(_c.FragmentM)}, {bm};");
            L($"mad.lo.u32 {col0}, {tx}, {I(_c.FragmentN)}, {bn};");
            L($"cvt.u64.u32 {ldc64}, {ldc};");
            bool vector = _c.VectorWidth == 4;
            for (int i = 0; i < _c.ThreadM; i++)
            {
                string row = R(), rowOk = P(), row64 = Rd(), rowPtr = Rd();
                L($"add.u32 {row}, {row0}, {I(_c.RowOffset(i))};");
                L($"setp.lt.u32 {rowOk}, {row}, {m};");
                L($"cvt.u64.u32 {row64}, {row};");
                L($"mul.lo.u64 {rowPtr}, {row64}, {ldc64};");
                L($"shl.b64 {rowPtr}, {rowPtr}, {I(Log2(_outb))};");
                L($"add.u64 {rowPtr}, {rowPtr}, {cPtr};");
                int step = vector ? 4 : 1;
                for (int j = 0; j < _c.ThreadN; j += step)
                {
                    string col = R(), ok = P(), col64 = Rd(), ptr = Rd(), okBeta = P();
                    L($"add.u32 {col}, {col0}, {I(_c.ColumnOffset(j))};");
                    L($"setp.lt.and.u32 {ok}, {col}, {n}, {rowOk};");
                    L($"cvt.u64.u32 {col64}, {col};");
                    L($"shl.b64 {col64}, {col64}, {I(Log2(_outb))};");
                    L($"add.u64 {ptr}, {rowPtr}, {col64};");
                    L($"and.pred {okBeta}, {ok}, {useBeta};");
                    var outs = new string[step];
                    var olds = new string[step];
                    for (int v = 0; v < step; v++)
                    {
                        outs[v] = CF();
                        olds[v] = CF();
                        L($"mul.rn.{Ct} {outs[v]}, {alpha}, {acc[i * _c.ThreadN + j + v]};");
                        L($"mov.{Ct} {olds[v]}, {Zero};");
                    }
                    if (vector)
                    {
                        L($"@{okBeta} ld.global.v4.f32 {{{olds[0]}, {olds[1]}, {olds[2]}, {olds[3]}}}, [{ptr}];");
                        for (int v = 0; v < step; v++) L($"@{okBeta} fma.rn.f32 {outs[v]}, {beta}, {olds[v]}, {outs[v]};");
                        L($"@{ok} st.global.v4.f32 [{ptr}], {{{outs[0]}, {outs[1]}, {outs[2]}, {outs[3]}}};");
                    }
                    else if (_outb == 2)
                    {
                        // FP16 output: C is read and written as FP16, the blend is FP32.
                        string oldHalf = H(), outHalf = H();
                        L($"@{okBeta} ld.global.b16 {oldHalf}, [{ptr}];");
                        L($"@{okBeta} cvt.f32.f16 {olds[0]}, {oldHalf};");
                        L($"@{okBeta} fma.rn.f32 {outs[0]}, {beta}, {olds[0]}, {outs[0]};");
                        L($"cvt.rn.f16.f32 {outHalf}, {outs[0]};");
                        L($"@{ok} st.global.b16 [{ptr}], {outHalf};");
                    }
                    else
                    {
                        L($"@{okBeta} ld.global.{Ct} {olds[0]}, [{ptr}];");
                        L($"@{okBeta} fma.rn.{Ct} {outs[0]}, {beta}, {olds[0]}, {outs[0]};");
                        L($"@{ok} st.global.{Ct} [{ptr}], {outs[0]};");
                    }
                }
            }
        }
    }
}
