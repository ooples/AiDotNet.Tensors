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
internal readonly record struct PtxGemmConfiguration(
    int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN, int VectorWidth, int SharedPad, bool DoubleBuffer)
{
    internal int ThreadsN => BlockN / ThreadN;
    internal int ThreadsM => BlockM / ThreadM;
    internal int Threads => ThreadsM * ThreadsN;
    internal int PitchA => BlockM + SharedPad;
    internal int PitchB => BlockN + SharedPad;
    internal int Stages => DoubleBuffer ? 2 : 1;
    internal int SharedBytes => checked(Stages * BlockK * (PitchA + PitchB) * sizeof(float));

    /// <summary>A short stable name, used in kernel identities and evolution records.</summary>
    internal string Name => string.Create(CultureInfo.InvariantCulture,
        $"b{BlockM}x{BlockN}x{BlockK}-t{ThreadM}x{ThreadN}-v{VectorWidth}-p{SharedPad}-{(DoubleBuffer ? "db" : "sb")}");

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
        {
            var config = new PtxGemmConfiguration(bm, bn, bk, tm, tn, vw, pad, db);
            if (config.Invalidity() is null) yield return config;
        }
    }

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
        if (SharedBytes > 48 * 1024) return "shared memory exceeds 48 KiB";
        if (ThreadM * ThreadN > 128) return "more than 128 accumulators per thread";
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

    internal static string Emit(PtxGemmConfiguration config, PtxGemmLayout layout, int ccMajor, int ccMinor)
    {
        string? invalid = config.Invalidity();
        if (invalid is not null) throw new ArgumentException("Invalid GEMM configuration " + config.Name + ": " + invalid, nameof(config));
        return new Emitter(config, layout).Build(ccMajor, ccMinor);
    }

    /// <summary>
    /// Whether a configuration may run a problem: a 16-byte vector configuration needs every vector it loads or stores
    /// to be wholly inside or wholly outside the matrix and 16-byte aligned.
    /// </summary>
    internal static bool Admits(PtxGemmConfiguration config, PtxGemmLayout layout, int m, int n, int k,
        int lda, int ldb, int ldc, long strideA, long strideB, long strideC)
    {
        if (config.Invalidity() is not null || m <= 0 || n <= 0 || k < 0) return false;
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
        private readonly StringBuilder _body = new(64_000);
        private int _r32, _r64, _pred, _f;

        internal Emitter(PtxGemmConfiguration config, PtxGemmLayout layout)
        {
            _c = config;
            _layout = layout;
        }

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
        }

        internal string Build(int ccMajor, int ccMinor)
        {
            int threads = _c.Threads;
            // Kernel arguments.
            string aPtr = Rd(), bPtr = Rd(), cPtr = Rd(), strideA = Rd(), strideB = Rd(), strideC = Rd();
            string m = R(), n = R(), k = R(), lda = R(), ldb = R(), ldc = R();
            string alpha = F(), beta = F();
            L($"ld.param.u64 {aPtr}, [a_ptr];");
            L($"ld.param.u64 {bPtr}, [b_ptr];");
            L($"ld.param.u64 {cPtr}, [c_ptr];");
            L($"ld.param.u32 {m}, [m];");
            L($"ld.param.u32 {n}, [n];");
            L($"ld.param.u32 {k}, [k];");
            L($"ld.param.u32 {lda}, [lda];");
            L($"ld.param.u32 {ldb}, [ldb];");
            L($"ld.param.u32 {ldc}, [ldc];");
            L($"ld.param.f32 {alpha}, [alpha];");
            L($"ld.param.f32 {beta}, [beta];");
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
            L($"mul.lo.u32 {bn}, {ctaX}, {I(_c.BlockN)};");
            L($"mul.lo.u32 {bm}, {ctaY}, {I(_c.BlockM)};");

            // Batch offsets (element strides -> bytes).
            string batch64 = Rd();
            L($"cvt.u64.u32 {batch64}, {batch};");
            foreach (var (ptr, stride) in new[] { (aPtr, strideA), (bPtr, strideB), (cPtr, strideC) })
            {
                string off = Rd();
                L($"mul.lo.u64 {off}, {batch64}, {stride};");
                L($"shl.b64 {off}, {off}, 2;");
                L($"add.u64 {ptr}, {ptr}, {off};");
            }

            var a = SetUpOperand("a", _layout.AContiguousAlongK, _c.BlockM, _c.PitchA, aPtr, lda, bm, m, tid);
            var b = SetUpOperand("b", _layout.BContiguousAlongK, _c.BlockN, _c.PitchB, bPtr, ldb, bn, n, tid);

            // Shared buffers (32-bit shared-window addresses) and the double-buffer selector.
            int tileBytesA = _c.BlockK * _c.PitchA * sizeof(float), tileBytesB = _c.BlockK * _c.PitchB * sizeof(float);
            string curA = R(), curB = R(), nxtA = R(), nxtB = R();
            L($"mov.u32 {curA}, a_tile;");
            L($"mov.u32 {curB}, b_tile;");
            L($"add.u32 {nxtA}, {curA}, {I(tileBytesA)};");
            L($"add.u32 {nxtB}, {curB}, {I(tileBytesB)};");

            // Accumulators.
            int accumulators = _c.ThreadM * _c.ThreadN;
            var acc = new string[accumulators];
            for (int i = 0; i < accumulators; i++)
            {
                acc[i] = F();
                L($"mov.f32 {acc[i]}, 0f00000000;");
            }
            var fa = new string[_c.ThreadM];
            var fb = new string[_c.ThreadN];
            for (int i = 0; i < fa.Length; i++) fa[i] = F();
            for (int i = 0; i < fb.Length; i++) fb[i] = F();

            // Per-thread shared read offsets for the inner product: A row ty*TM, B column tx*TN (bytes).
            string aRead = R(), bRead = R();
            L($"mul.lo.u32 {aRead}, {ty}, {I(_c.ThreadM * sizeof(float))};");
            L($"mul.lo.u32 {bRead}, {tx}, {I(_c.ThreadN * sizeof(float))};");

            // Tile count; K == 0 skips the main loop (C = beta * C).
            string tiles = R(), tile = R(), k0 = R();
            L($"add.u32 {tiles}, {k}, {I(_c.BlockK - 1)};");
            L($"div.u32 {tiles}, {tiles}, {I(_c.BlockK)};");
            L($"mov.u32 {tile}, 0;");
            L($"mov.u32 {k0}, 0;");
            string empty = P();
            L($"setp.eq.u32 {empty}, {tiles}, 0;");
            L($"@{empty} bra EPILOGUE;");

            if (_c.DoubleBuffer)
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
            ptx.Append("// sgemm ").Append(_c.Name).Append(' ').AppendLine(_layout.Name);
            ptx.Append(".visible .entry ").Append(EntryPoint).AppendLine("(");
            ptx.AppendLine("    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 c_ptr,");
            ptx.AppendLine("    .param .u32 m, .param .u32 n, .param .u32 k, .param .u32 lda, .param .u32 ldb, .param .u32 ldc,");
            ptx.AppendLine("    .param .f32 alpha, .param .f32 beta,");
            ptx.AppendLine("    .param .u64 stride_a, .param .u64 stride_b, .param .u64 stride_c)");
            ptx.Append(".maxntid ").Append(I(threads)).AppendLine(", 1, 1");
            ptx.AppendLine("{");
            ptx.Append("    .reg .pred %p<").Append(I(Math.Max(1, _pred))).AppendLine(">;");
            ptx.Append("    .reg .b32 %r<").Append(I(Math.Max(1, _r32))).AppendLine(">;");
            ptx.Append("    .reg .b64 %rd<").Append(I(Math.Max(1, _r64))).AppendLine(">;");
            ptx.Append("    .reg .f32 %f<").Append(I(Math.Max(1, _f))).AppendLine(">;");
            ptx.Append("    .shared .align 16 .b8 a_tile[").Append(I(_c.Stages * tileBytesA)).AppendLine("];");
            ptx.Append("    .shared .align 16 .b8 b_tile[").Append(I(_c.Stages * tileBytesB)).AppendLine("];");
            ptx.Append(_body);
            ptx.AppendLine("}");
            return ptx.ToString();
        }

        // Fixes each load slot's tile offset, its global pointer for tile 0 and the per-tile pointer step.
        private Operand SetUpOperand(string name, bool contiguousAlongK, int mnTile, int pitch, string basePtr, string ld,
            string blockMn, string mnLimit, string tid)
        {
            int vw = _c.VectorWidth;
            int slots = mnTile * _c.BlockK / (_c.Threads * vw);
            var op = new Operand
            {
                Name = name,
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
            for (int s = 0; s < slots; s++)
            {
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
                L($"shl.b64 {offset}, {offset}, 2;");
                L($"add.u64 {ptr}, {basePtr}, {offset};");
                op.SlotPointer[s] = ptr;
                op.SlotMnInBounds[s] = inBounds;
                // Keep the k offset and the mn offset as registers too (shared store address and the K bound).
                op.SlotK[s] = s;
                op.SlotMn[s] = s;
                op.Staged[s] = new string[vw];
                for (int v = 0; v < vw; v++) op.Staged[s][v] = F();
                _slotK[(name, s)] = kOff;
                _slotMn[(name, s)] = mnOff;
            }
            // One K tile further: BK elements along K, or BK rows of ld elements.
            op.Step = Rd();
            if (contiguousAlongK) L($"mov.u64 {op.Step}, {I((long)_c.BlockK * sizeof(float))};");
            else L($"mul.lo.u64 {op.Step}, {ld64}, {I((long)_c.BlockK * sizeof(float))};");
            return op;
        }

        private readonly Dictionary<(string, int), string> _slotK = new();
        private readonly Dictionary<(string, int), string> _slotMn = new();

        private void Advance(Operand op)
        {
            foreach (string ptr in op.SlotPointer) L($"add.u64 {ptr}, {ptr}, {op.Step};");
        }

        // Global -> registers for the tile starting at k0; values outside the matrix are zero.
        private void LoadTile(Operand op, string k0, string k)
        {
            int vw = _c.VectorWidth;
            for (int s = 0; s < op.SlotPointer.Length; s++)
            {
                string kGlobal = R(), ok = P();
                L($"add.u32 {kGlobal}, {k0}, {_slotK[(op.Name, s)]};");
                L($"setp.lt.and.u32 {ok}, {kGlobal}, {k}, {op.SlotMnInBounds[s]};");
                foreach (string v in op.Staged[s]) L($"mov.f32 {v}, 0f00000000;");
                if (vw == 4)
                    L($"@{ok} ld.global.v4.f32 {{{op.Staged[s][0]}, {op.Staged[s][1]}, {op.Staged[s][2]}, {op.Staged[s][3]}}}, [{op.SlotPointer[s]}];");
                else
                    L($"@{ok} ld.global.f32 {op.Staged[s][0]}, [{op.SlotPointer[s]}];");
            }
        }

        // Registers -> shared, K-major: element (k, mn) at (k * pitch + mn) * 4 bytes from the buffer base.
        private void StoreTile(Operand op, string buffer)
        {
            int vw = _c.VectorWidth;
            for (int s = 0; s < op.SlotPointer.Length; s++)
            {
                string kOff = _slotK[(op.Name, s)], mnOff = _slotMn[(op.Name, s)];
                string address = R();
                L($"mad.lo.u32 {address}, {kOff}, {I(op.Pitch)}, {mnOff};");
                L($"shl.b32 {address}, {address}, 2;");
                L($"add.u32 {address}, {address}, {buffer};");
                if (op.ContiguousAlongK)
                {
                    // The vector runs along K: a column of the K-major tile, one row apart per element.
                    for (int v = 0; v < vw; v++)
                        L($"st.shared.f32 [{address}+{I(v * op.Pitch * sizeof(float))}], {op.Staged[s][v]};");
                }
                else if (vw == 4)
                {
                    L($"st.shared.v4.f32 [{address}], {{{op.Staged[s][0]}, {op.Staged[s][1]}, {op.Staged[s][2]}, {op.Staged[s][3]}}};");
                }
                else
                {
                    L($"st.shared.f32 [{address}], {op.Staged[s][0]};");
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
                int aRow = kk * _c.PitchA * sizeof(float), bRow = kk * _c.PitchB * sizeof(float);
                for (int i = 0; i < fa.Length; i += 4)
                    L($"ld.shared.v4.f32 {{{fa[i]}, {fa[i + 1]}, {fa[i + 2]}, {fa[i + 3]}}}, [{aBase}+{I(aRow + i * sizeof(float))}];");
                for (int j = 0; j < fb.Length; j += 4)
                    L($"ld.shared.v4.f32 {{{fb[j]}, {fb[j + 1]}, {fb[j + 2]}, {fb[j + 3]}}}, [{bBase}+{I(bRow + j * sizeof(float))}];");
                for (int i = 0; i < fa.Length; i++)
                    for (int j = 0; j < fb.Length; j++)
                    {
                        string a = acc[i * fb.Length + j];
                        L($"fma.rn.f32 {a}, {fa[i]}, {fb[j]}, {a};");
                    }
            }
        }

        // C = alpha * acc + beta * C, predicated on the matrix bounds; C is not read when beta == 0.
        private void Epilogue(string cPtr, string m, string n, string ldc, string alpha, string beta, string bm, string bn,
            string tx, string ty, string[] acc)
        {
            string useBeta = P();
            L($"setp.neu.f32 {useBeta}, {beta}, 0f00000000;");
            string row0 = R(), col0 = R(), ldc64 = Rd();
            L($"mad.lo.u32 {row0}, {ty}, {I(_c.ThreadM)}, {bm};");
            L($"mad.lo.u32 {col0}, {tx}, {I(_c.ThreadN)}, {bn};");
            L($"cvt.u64.u32 {ldc64}, {ldc};");
            bool vector = _c.VectorWidth == 4;
            for (int i = 0; i < _c.ThreadM; i++)
            {
                string row = R(), rowOk = P(), row64 = Rd(), rowPtr = Rd();
                L($"add.u32 {row}, {row0}, {I(i)};");
                L($"setp.lt.u32 {rowOk}, {row}, {m};");
                L($"cvt.u64.u32 {row64}, {row};");
                L($"mul.lo.u64 {rowPtr}, {row64}, {ldc64};");
                L($"shl.b64 {rowPtr}, {rowPtr}, 2;");
                L($"add.u64 {rowPtr}, {rowPtr}, {cPtr};");
                int step = vector ? 4 : 1;
                for (int j = 0; j < _c.ThreadN; j += step)
                {
                    string col = R(), ok = P(), col64 = Rd(), ptr = Rd(), okBeta = P();
                    L($"add.u32 {col}, {col0}, {I(j)};");
                    L($"setp.lt.and.u32 {ok}, {col}, {n}, {rowOk};");
                    L($"cvt.u64.u32 {col64}, {col};");
                    L($"shl.b64 {col64}, {col64}, 2;");
                    L($"add.u64 {ptr}, {rowPtr}, {col64};");
                    L($"and.pred {okBeta}, {ok}, {useBeta};");
                    var outs = new string[step];
                    var olds = new string[step];
                    for (int v = 0; v < step; v++)
                    {
                        outs[v] = F();
                        olds[v] = F();
                        L($"mul.rn.f32 {outs[v]}, {alpha}, {acc[i * _c.ThreadN + j + v]};");
                        L($"mov.f32 {olds[v]}, 0f00000000;");
                    }
                    if (vector)
                    {
                        L($"@{okBeta} ld.global.v4.f32 {{{olds[0]}, {olds[1]}, {olds[2]}, {olds[3]}}}, [{ptr}];");
                        for (int v = 0; v < step; v++) L($"@{okBeta} fma.rn.f32 {outs[v]}, {beta}, {olds[v]}, {outs[v]};");
                        L($"@{ok} st.global.v4.f32 [{ptr}], {{{outs[0]}, {outs[1]}, {outs[2]}, {outs[3]}}};");
                    }
                    else
                    {
                        L($"@{okBeta} ld.global.f32 {olds[0]}, [{ptr}];");
                        L($"@{okBeta} fma.rn.f32 {outs[0]}, {beta}, {olds[0]}, {outs[0]};");
                        L($"@{ok} st.global.f32 [{ptr}], {outs[0]};");
                    }
                }
            }
        }
    }
}
