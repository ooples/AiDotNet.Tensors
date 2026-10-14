using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// The genome of a generated Tensor-Core FP32-in / FP32-out GEMM computed in TF32 (#1264, phase D of #377): the math
/// cuBLAS uses for FP32 GEMM by default on Ampere and newer. Every field changes the emitted PTX or its launch geometry.
/// </summary>
/// <param name="BlockM">Rows of C per thread block.</param>
/// <param name="BlockN">Columns of C per thread block.</param>
/// <param name="BlockK">Depth of each K tile staged in shared memory (a multiple of the MMA's k = 8).</param>
/// <param name="WarpM">Rows of C per warp (a multiple of the MMA's m = 16).</param>
/// <param name="WarpN">Columns of C per warp (a multiple of the MMA's n = 8).</param>
/// <param name="Stages">K tiles in flight in the cp.async pipeline (2 to 4).</param>
/// <param name="SharedPad">Extra floats per shared-memory row (0, 4 or 8), to keep fragment loads bank-conflict free.</param>
internal readonly record struct PtxTensorCoreGemmConfiguration(
    int BlockM, int BlockN, int BlockK, int WarpM, int WarpN, int Stages, int SharedPad)
{
    internal const int MmaM = 16, MmaN = 8, MmaK = 8;
    internal const int MaximumSharedBytes = 99 * 1024;

    internal int WarpsM => BlockM / WarpM;
    internal int WarpsN => BlockN / WarpN;
    internal int Threads => WarpsM * WarpsN * 32;
    internal int TilesM => WarpM / MmaM;
    internal int TilesN => WarpN / MmaN;
    internal int Accumulators => TilesM * TilesN * 4;

    // Shared pitches (floats) per operand layout: the global-contiguous dimension stays contiguous in shared memory.
    internal int PitchA(PtxGemmLayout layout) => (layout.AContiguousAlongK ? BlockK : BlockM) + SharedPad;
    internal int PitchB(PtxGemmLayout layout) => (layout.BContiguousAlongK ? BlockK : BlockN) + SharedPad;
    internal int StageFloatsA(PtxGemmLayout layout) => (layout.AContiguousAlongK ? BlockM : BlockK) * PitchA(layout);
    internal int StageFloatsB(PtxGemmLayout layout) => (layout.BContiguousAlongK ? BlockN : BlockK) * PitchB(layout);
    internal int SharedBytes(PtxGemmLayout layout) =>
        checked(Stages * (StageFloatsA(layout) + StageFloatsB(layout)) * sizeof(float));

    internal string Name => string.Format(CultureInfo.InvariantCulture, "tc-b{0}x{1}x{2}-w{3}x{4}-s{5}-p{6}",
        BlockM, BlockN, BlockK, WarpM, WarpN, Stages, SharedPad);

    /// <summary>Why this configuration cannot be emitted for <paramref name="layout"/>, or null when it can.</summary>
    internal string? Invalidity(PtxGemmLayout layout)
    {
        if (BlockM <= 0 || BlockN <= 0 || BlockK <= 0 || WarpM <= 0 || WarpN <= 0) return "non-positive dimension";
        if (WarpM % MmaM != 0 || WarpN % MmaN != 0) return "the warp tile must be whole m16n8 MMA tiles";
        if (BlockM % WarpM != 0 || BlockN % WarpN != 0) return "the warp tile must divide the block tile";
        if (BlockK % MmaK != 0 || BlockK % 4 != 0) return "the K tile must be whole k8 steps";
        if (Stages is < 2 or > 4) return "stages must be 2 to 4";
        if (SharedPad is not (0 or 4 or 8)) return "shared pad must be 0, 4 or 8";
        if (Threads < 32 || Threads > 1024) return "thread count must be in [32, 1024]";
        if (BlockM * BlockK % (Threads * 4) != 0 || BlockN * BlockK % (Threads * 4) != 0)
            return "each thread must copy a whole number of 16-byte chunks of both tiles";
        if (Accumulators > 128) return "more than 128 accumulators per thread";
        if (SharedBytes(layout) > MaximumSharedBytes) return "shared memory exceeds 99 KiB";
        return null;
    }

    /// <summary>The emit-able configurations over the genome's axes, for <paramref name="layout"/>.</summary>
    internal static IEnumerable<PtxTensorCoreGemmConfiguration> SearchSpace(PtxGemmLayout layout)
    {
        foreach (int bm in new[] { 64, 128, 256 })
        foreach (int bn in new[] { 64, 128, 256 })
        foreach (int bk in new[] { 16, 32 })
        foreach (int wm in new[] { 32, 64 })
        foreach (int wn in new[] { 32, 64 })
        foreach (int stages in new[] { 2, 3, 4 })
        foreach (int pad in new[] { 0, 4, 8 })
        {
            var config = new PtxTensorCoreGemmConfiguration(bm, bn, bk, wm, wn, stages, pad);
            if (config.Invalidity(layout) is null) yield return config;
        }
    }
}

/// <summary>
/// Emits <c>C = alpha * op(A) @ op(B) + beta * C</c> on Tensor Cores: FP32 operands rounded to TF32 at fragment load,
/// <c>mma.sync.m16n8k8</c> with FP32 accumulation, batched over grid z, with M, N, K, the leading dimensions and the
/// batch strides as arguments (one module per configuration and layout, any shape). Global tiles stream into shared
/// memory through a <c>cp.async</c> pipeline of <c>Stages</c> K tiles; a copy outside the matrix uses the zero-fill form
/// (<c>src-size = 0</c>), so edges need no separate path. C is never read when beta is zero. Needs SM 8.0+ and 16-byte
/// aligned operands (see <see cref="Admits"/>).
/// </summary>
internal static class PtxTensorCoreGemmGenerator
{
    internal const string EntryPoint = "aidotnet_tc_sgemm";

    internal static string Emit(PtxTensorCoreGemmConfiguration config, PtxGemmLayout layout, int ccMajor, int ccMinor)
    {
        if (ccMajor < 8) throw new PlatformNotSupportedException("TF32 MMA and cp.async need compute capability 8.0+.");
        string? invalid = config.Invalidity(layout);
        if (invalid is not null)
            throw new ArgumentException("Invalid Tensor-Core GEMM configuration " + config.Name + ": " + invalid, nameof(config));
        return new Emitter(config, layout).Build(ccMajor, ccMinor);
    }

    /// <summary>Whether the kernel may run a problem: every 16-byte copy must be wholly inside or outside, and aligned.</summary>
    internal static bool Admits(PtxTensorCoreGemmConfiguration config, PtxGemmLayout layout, int m, int n, int k,
        int lda, int ldb, int ldc, long strideA, long strideB, long strideC)
    {
        if (config.Invalidity(layout) is not null || m <= 0 || n <= 0 || k < 0) return false;
        int aContiguous = layout.AContiguousAlongK ? k : m;
        int bContiguous = layout.BContiguousAlongK ? k : n;
        return aContiguous % 4 == 0 && bContiguous % 4 == 0 && n % 4 == 0 &&
               lda % 4 == 0 && ldb % 4 == 0 && ldc % 4 == 0 &&
               strideA % 4 == 0 && strideB % 4 == 0 && strideC % 4 == 0;
    }

    private sealed class Emitter
    {
        private readonly PtxTensorCoreGemmConfiguration _c;
        private readonly PtxGemmLayout _layout;
        private readonly StringBuilder _body = new(128_000);
        private int _r32, _r64, _pred, _f;

        internal Emitter(PtxTensorCoreGemmConfiguration config, PtxGemmLayout layout)
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
        private static string I(long value) => value.ToString(CultureInfo.InvariantCulture);

        // One operand's cp.async chunks for a stage: each is 4 floats contiguous in global and in shared memory.
        private sealed class Operand
        {
            internal bool ContiguousAlongK;
            internal int MnTile;
            internal int Pitch;
            internal int StageBytes;
            internal string[] ChunkPointer = Array.Empty<string>();   // u64 global source for the current tile
            internal string[] ChunkShared = Array.Empty<string>();    // u32 offset within a stage, in bytes
            internal string[] ChunkK = Array.Empty<string>();         // the chunk's k offset within the tile
            internal string[] ChunkMnOk = Array.Empty<string>();      // predicate: its row/column exists
            internal string Step = "";                               // u64 bytes per K tile
            internal string SharedBase = "";                         // u32 shared address of stage 0
        }

        internal string Build(int ccMajor, int ccMinor)
        {
            // Arguments.
            string aPtr = Rd(), bPtr = Rd(), cPtr = Rd(), strideA = Rd(), strideB = Rd(), strideC = Rd();
            string m = R(), n = R(), k = R(), lda = R(), ldb = R(), ldc = R();
            string alpha = F(), beta = F();
            foreach (var (reg, name) in new[] { (aPtr, "a_ptr"), (bPtr, "b_ptr"), (cPtr, "c_ptr"),
                         (strideA, "stride_a"), (strideB, "stride_b"), (strideC, "stride_c") })
                L($"ld.param.u64 {reg}, [{name}];");
            foreach (var (reg, name) in new[] { (m, "m"), (n, "n"), (k, "k"), (lda, "lda"), (ldb, "ldb"), (ldc, "ldc") })
                L($"ld.param.u32 {reg}, [{name}];");
            L($"ld.param.f32 {alpha}, [alpha];");
            L($"ld.param.f32 {beta}, [beta];");

            // Coordinates: block origin, batch, warp origin within the block, lane group / thread-in-group.
            string tid = R(), warp = R(), lane = R(), gid = R(), tig = R(), wm = R(), wn = R();
            string ctaX = R(), ctaY = R(), batch = R(), bm = R(), bn = R(), warpM0 = R(), warpN0 = R();
            L($"mov.u32 {tid}, %tid.x;");
            L($"shr.u32 {warp}, {tid}, 5;");
            L($"and.b32 {lane}, {tid}, 31;");
            L($"shr.u32 {gid}, {lane}, 2;");
            L($"and.b32 {tig}, {lane}, 3;");
            L($"div.u32 {wm}, {warp}, {I(_c.WarpsN)};");
            L($"rem.u32 {wn}, {warp}, {I(_c.WarpsN)};");
            L($"mul.lo.u32 {warpM0}, {wm}, {I(_c.WarpM)};");
            L($"mul.lo.u32 {warpN0}, {wn}, {I(_c.WarpN)};");
            L($"mov.u32 {ctaX}, %ctaid.x;");
            L($"mov.u32 {ctaY}, %ctaid.y;");
            L($"mov.u32 {batch}, %ctaid.z;");
            L($"mul.lo.u32 {bn}, {ctaX}, {I(_c.BlockN)};");
            L($"mul.lo.u32 {bm}, {ctaY}, {I(_c.BlockM)};");
            string batch64 = Rd();
            L($"cvt.u64.u32 {batch64}, {batch};");
            foreach (var (ptr, stride) in new[] { (aPtr, strideA), (bPtr, strideB), (cPtr, strideC) })
            {
                string off = Rd();
                L($"mul.lo.u64 {off}, {batch64}, {stride};");
                L($"shl.b64 {off}, {off}, 2;");
                L($"add.u64 {ptr}, {ptr}, {off};");
            }

            // Dynamic shared memory: A stages, then B stages.
            string smem = R();
            L($"mov.u32 {smem}, smem;");
            int stageBytesA = _c.StageFloatsA(_layout) * sizeof(float);
            int stageBytesB = _c.StageFloatsB(_layout) * sizeof(float);
            var a = SetUpOperand(_layout.AContiguousAlongK, _c.BlockM, _c.PitchA(_layout), stageBytesA, aPtr, lda, bm, m, tid);
            var b = SetUpOperand(_layout.BContiguousAlongK, _c.BlockN, _c.PitchB(_layout), stageBytesB, bPtr, ldb, bn, n, tid);
            a.SharedBase = smem;
            b.SharedBase = R();
            L($"add.u32 {b.SharedBase}, {smem}, {I((long)_c.Stages * stageBytesA)};");

            // Per-thread fragment bases within a stage (bytes); every fragment load adds a compile-time offset.
            string aFrag = FragmentBase(a, _layout.AContiguousAlongK, warpM0, gid, tig);
            string bFrag = FragmentBase(b, _layout.BContiguousAlongK, warpN0, gid, tig);

            // Accumulators.
            var acc = new string[_c.Accumulators];
            for (int i = 0; i < acc.Length; i++)
            {
                acc[i] = F();
                L($"mov.f32 {acc[i]}, 0f00000000;");
            }

            string tiles = R(), tile = R(), k0 = R();
            L($"add.u32 {tiles}, {k}, {I(_c.BlockK - 1)};");
            L($"div.u32 {tiles}, {tiles}, {I(_c.BlockK)};");
            string empty = P();
            L($"setp.eq.u32 {empty}, {tiles}, 0;");
            L($"@{empty} bra EPILOGUE;");

            // Prologue: tiles 0 .. Stages-2 in flight (a group per tile, empty past the last tile).
            L($"mov.u32 {k0}, 0;");
            for (int s = 0; s < _c.Stages - 1; s++)
            {
                string issue = P();
                L($"setp.gt.u32 {issue}, {tiles}, {I(s)};");
                IssueTile(a, I(s * stageBytesA), k0, k, issue);
                IssueTile(b, I(s * stageBytesB), k0, k, issue);
                L("cp.async.commit_group;");
                if (s < _c.Stages - 2)
                {
                    Advance(a);
                    Advance(b);
                    L($"add.u32 {k0}, {k0}, {I(_c.BlockK)};");
                }
            }
            // The next tile to issue is Stages-1; its pointers are one step beyond the last issued tile.
            L($"mov.u32 {tile}, 0;");

            Label("K_LOOP");
            L($"cp.async.wait_group {I(_c.Stages - 2)};");
            L("bar.sync 0;");
            {
                // Issue tile (tile + Stages - 1) into stage (tile + Stages - 1) % Stages, then compute stage tile % Stages.
                string next = R(), issueStage = R(), issue = P(), nextK0 = R();
                L($"add.u32 {next}, {tile}, {I(_c.Stages - 1)};");
                L($"setp.lt.u32 {issue}, {next}, {tiles};");
                L($"rem.u32 {issueStage}, {next}, {I(_c.Stages)};");
                L($"mul.lo.u32 {nextK0}, {next}, {I(_c.BlockK)};");
                Advance(a);
                Advance(b);
                string offA = R(), offB = R();
                L($"mul.lo.u32 {offA}, {issueStage}, {I(stageBytesA)};");
                L($"mul.lo.u32 {offB}, {issueStage}, {I(stageBytesB)};");
                IssueTile(a, offA, nextK0, k, issue);
                IssueTile(b, offB, nextK0, k, issue);
                L("cp.async.commit_group;");

                string computeStage = R(), stageA = R(), stageB = R();
                L($"rem.u32 {computeStage}, {tile}, {I(_c.Stages)};");
                L($"mad.lo.u32 {stageA}, {computeStage}, {I(stageBytesA)}, {aFrag};");
                L($"mad.lo.u32 {stageB}, {computeStage}, {I(stageBytesB)}, {bFrag};");
                Compute(a, b, stageA, stageB, acc);
            }
            string more = P();
            L($"add.u32 {tile}, {tile}, 1;");
            L($"setp.lt.u32 {more}, {tile}, {tiles};");
            L($"@{more} bra K_LOOP;");
            L("cp.async.wait_group 0;");

            Label("EPILOGUE");
            Epilogue(cPtr, m, n, ldc, alpha, beta, bm, bn, warpM0, warpN0, gid, tig, acc);
            L("ret;");

            var ptx = new StringBuilder(_body.Length + 2_048);
            ptx.AppendLine(".version 7.1");
            ptx.Append(".target sm_").Append(I(ccMajor)).AppendLine(I(ccMinor));
            ptx.AppendLine(".address_size 64");
            ptx.Append("// tc-sgemm (tf32 mma) ").Append(_c.Name).Append(' ').AppendLine(_layout.Name);
            ptx.AppendLine(".extern .shared .align 16 .b8 smem[];");
            ptx.Append(".visible .entry ").Append(EntryPoint).AppendLine("(");
            ptx.AppendLine("    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 c_ptr,");
            ptx.AppendLine("    .param .u32 m, .param .u32 n, .param .u32 k, .param .u32 lda, .param .u32 ldb, .param .u32 ldc,");
            ptx.AppendLine("    .param .f32 alpha, .param .f32 beta,");
            ptx.AppendLine("    .param .u64 stride_a, .param .u64 stride_b, .param .u64 stride_c)");
            ptx.Append(".maxntid ").Append(I(_c.Threads)).AppendLine(", 1, 1");
            ptx.AppendLine("{");
            ptx.Append("    .reg .pred %p<").Append(I(Math.Max(1, _pred))).AppendLine(">;");
            ptx.Append("    .reg .b32 %r<").Append(I(Math.Max(1, _r32))).AppendLine(">;");
            ptx.Append("    .reg .b64 %rd<").Append(I(Math.Max(1, _r64))).AppendLine(">;");
            ptx.Append("    .reg .f32 %f<").Append(I(Math.Max(1, _f))).AppendLine(">;");
            ptx.Append(_body);
            ptx.AppendLine("}");
            return ptx.ToString();
        }

        // Fixes each 16-byte chunk's tile position, its global source for tile 0 and its shared offset in a stage.
        private Operand SetUpOperand(bool contiguousAlongK, int mnTile, int pitch, int stageBytes, string basePtr,
            string ld, string blockMn, string mnLimit, string tid)
        {
            int chunks = mnTile * _c.BlockK / (_c.Threads * 4);
            var op = new Operand
            {
                ContiguousAlongK = contiguousAlongK,
                MnTile = mnTile,
                Pitch = pitch,
                StageBytes = stageBytes,
                ChunkPointer = new string[chunks],
                ChunkShared = new string[chunks],
                ChunkK = new string[chunks],
                ChunkMnOk = new string[chunks],
            };
            string ld64 = Rd();
            L($"cvt.u64.u32 {ld64}, {ld};");
            int perRow = (contiguousAlongK ? _c.BlockK : mnTile) / 4;  // chunks along the contiguous dimension
            for (int s = 0; s < chunks; s++)
            {
                string chunk = R(), row = R(), column = R();
                L($"add.u32 {chunk}, {tid}, {I(s * _c.Threads)};");
                L($"div.u32 {row}, {chunk}, {I(perRow)};");
                L($"rem.u32 {column}, {chunk}, {I(perRow)};");
                L($"shl.b32 {column}, {column}, 2;");
                // Contiguous along K: row = mn, column = k. Otherwise row = k, column = mn.
                string kOff = contiguousAlongK ? column : row;
                string mnOff = contiguousAlongK ? row : column;
                string mnGlobal = R(), ok = P();
                L($"add.u32 {mnGlobal}, {blockMn}, {mnOff};");
                L($"setp.lt.u32 {ok}, {mnGlobal}, {mnLimit};");
                string mn64 = Rd(), k64 = Rd(), offset = Rd(), ptr = Rd();
                L($"cvt.u64.u32 {mn64}, {mnGlobal};");
                L($"cvt.u64.u32 {k64}, {kOff};");
                if (contiguousAlongK) L($"mad.lo.u64 {offset}, {mn64}, {ld64}, {k64};");
                else L($"mad.lo.u64 {offset}, {k64}, {ld64}, {mn64};");
                L($"shl.b64 {offset}, {offset}, 2;");
                L($"add.u64 {ptr}, {basePtr}, {offset};");
                string shared = R();
                L($"mad.lo.u32 {shared}, {row}, {I(pitch)}, {column};");
                L($"shl.b32 {shared}, {shared}, 2;");
                op.ChunkPointer[s] = ptr;
                op.ChunkShared[s] = shared;
                op.ChunkK[s] = kOff;
                op.ChunkMnOk[s] = ok;
            }
            op.Step = Rd();
            if (contiguousAlongK) L($"mov.u64 {op.Step}, {I((long)_c.BlockK * sizeof(float))};");
            else L($"mul.lo.u64 {op.Step}, {ld64}, {I((long)_c.BlockK * sizeof(float))};");
            return op;
        }

        private void Advance(Operand op)
        {
            foreach (string ptr in op.ChunkPointer) L($"add.u64 {ptr}, {ptr}, {op.Step};");
        }

        // cp.async of every chunk of the tile starting at k0 into the stage at stageOffset (a register or an immediate);
        // a chunk outside the matrix (or past the last tile) copies zero bytes and zero-fills its 16 bytes.
        private void IssueTile(Operand op, string stageOffset, string k0, string k, string issue)
        {
            for (int s = 0; s < op.ChunkPointer.Length; s++)
            {
                string kGlobal = R(), inBounds = P(), bytes = R(), dst = R();
                L($"add.u32 {kGlobal}, {k0}, {op.ChunkK[s]};");
                L($"setp.lt.and.u32 {inBounds}, {kGlobal}, {k}, {op.ChunkMnOk[s]};");
                L($"and.pred {inBounds}, {inBounds}, {issue};");
                L($"selp.u32 {bytes}, 16, 0, {inBounds};");
                L($"add.u32 {dst}, {op.SharedBase}, {op.ChunkShared[s]};");
                L($"add.u32 {dst}, {dst}, {stageOffset};");
                L($"cp.async.cg.shared.global [{dst}], [{op.ChunkPointer[s]}], 16, {bytes};");
            }
        }

        // This thread's fragment address base within stage 0 (bytes, including the operand's shared base).
        private string FragmentBase(Operand op, bool contiguousAlongK, string warpMn0, string gid, string tig)
        {
            // A / B fragments read element (mn = warpMn0 + gid [+ 8 / + tile offsets], k = tig [+ 4 / + kk * 8]).
            string mnThread = R(), offset = R();
            L($"add.u32 {mnThread}, {warpMn0}, {gid};");
            if (contiguousAlongK) L($"mad.lo.u32 {offset}, {mnThread}, {I(op.Pitch)}, {tig};");
            else L($"mad.lo.u32 {offset}, {tig}, {I(op.Pitch)}, {mnThread};");
            L($"shl.b32 {offset}, {offset}, 2;");
            L($"add.u32 {offset}, {offset}, {op.SharedBase};");
            return offset;
        }

        // Byte offset of fragment element (mn, k) relative to the thread's fragment base.
        private static int FragmentOffset(Operand op, int mn, int kIndex) =>
            (op.ContiguousAlongK ? mn * op.Pitch + kIndex : kIndex * op.Pitch + mn) * sizeof(float);

        private void Compute(Operand a, Operand b, string stageA, string stageB, string[] acc)
        {
            int tilesM = _c.TilesM, tilesN = _c.TilesN;
            var fa = new string[tilesM, 4];
            var fb = new string[tilesN, 2];
            for (int i = 0; i < tilesM; i++) for (int j = 0; j < 4; j++) fa[i, j] = R();
            for (int i = 0; i < tilesN; i++) for (int j = 0; j < 2; j++) fb[i, j] = R();
            string scratch = F();
            for (int kk = 0; kk < _c.BlockK / PtxTensorCoreGemmConfiguration.MmaK; kk++)
            {
                int k8 = kk * PtxTensorCoreGemmConfiguration.MmaK;
                // A (16 x 8, row): a0 (gid, tig), a1 (gid + 8, tig), a2 (gid, tig + 4), a3 (gid + 8, tig + 4).
                for (int mt = 0; mt < tilesM; mt++)
                {
                    int row = mt * 16;
                    int[] offsets =
                    {
                        FragmentOffset(a, row, k8), FragmentOffset(a, row + 8, k8),
                        FragmentOffset(a, row, k8 + 4), FragmentOffset(a, row + 8, k8 + 4),
                    };
                    for (int j = 0; j < 4; j++)
                    {
                        L($"ld.shared.f32 {scratch}, [{stageA}+{I(offsets[j])}];");
                        L($"cvt.rna.tf32.f32 {fa[mt, j]}, {scratch};");
                    }
                }
                // B (8 x 8, col): b0 (k = tig, n = gid), b1 (k = tig + 4, n = gid).
                for (int nt = 0; nt < tilesN; nt++)
                {
                    int column = nt * 8;
                    int[] offsets = { FragmentOffset(b, column, k8), FragmentOffset(b, column, k8 + 4) };
                    for (int j = 0; j < 2; j++)
                    {
                        L($"ld.shared.f32 {scratch}, [{stageB}+{I(offsets[j])}];");
                        L($"cvt.rna.tf32.f32 {fb[nt, j]}, {scratch};");
                    }
                }
                for (int mt = 0; mt < tilesM; mt++)
                    for (int nt = 0; nt < tilesN; nt++)
                    {
                        int c = (mt * tilesN + nt) * 4;
                        string d = $"{{{acc[c]}, {acc[c + 1]}, {acc[c + 2]}, {acc[c + 3]}}}";
                        L($"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {d}, " +
                          $"{{{fa[mt, 0]}, {fa[mt, 1]}, {fa[mt, 2]}, {fa[mt, 3]}}}, {{{fb[nt, 0]}, {fb[nt, 1]}}}, {d};");
                    }
            }
        }

        // D (16 x 8): d0 (gid, 2 tig), d1 (gid, 2 tig + 1), d2 (gid + 8, 2 tig), d3 (gid + 8, 2 tig + 1).
        private void Epilogue(string cPtr, string m, string n, string ldc, string alpha, string beta, string bm, string bn,
            string warpM0, string warpN0, string gid, string tig, string[] acc)
        {
            string useBeta = P(), ldc64 = Rd(), rowBase = R(), colBase = R();
            L($"setp.neu.f32 {useBeta}, {beta}, 0f00000000;");
            L($"cvt.u64.u32 {ldc64}, {ldc};");
            L($"add.u32 {rowBase}, {bm}, {warpM0};");
            L($"add.u32 {rowBase}, {rowBase}, {gid};");
            L($"add.u32 {colBase}, {bn}, {warpN0};");
            string twoTig = R();
            L($"shl.b32 {twoTig}, {tig}, 1;");
            L($"add.u32 {colBase}, {colBase}, {twoTig};");
            for (int mt = 0; mt < _c.TilesM; mt++)
                for (int half = 0; half < 2; half++)
                {
                    string row = R(), rowOk = P(), row64 = Rd(), rowPtr = Rd();
                    L($"add.u32 {row}, {rowBase}, {I(mt * 16 + half * 8)};");
                    L($"setp.lt.u32 {rowOk}, {row}, {m};");
                    L($"cvt.u64.u32 {row64}, {row};");
                    L($"mul.lo.u64 {rowPtr}, {row64}, {ldc64};");
                    L($"shl.b64 {rowPtr}, {rowPtr}, 2;");
                    L($"add.u64 {rowPtr}, {rowPtr}, {cPtr};");
                    for (int nt = 0; nt < _c.TilesN; nt++)
                    {
                        // N is a multiple of 4 (Admits) and the column even, so the pair is wholly in or out.
                        string col = R(), ok = P(), okBeta = P(), col64 = Rd(), ptr = Rd();
                        L($"add.u32 {col}, {colBase}, {I(nt * 8)};");
                        L($"setp.lt.and.u32 {ok}, {col}, {n}, {rowOk};");
                        L($"and.pred {okBeta}, {ok}, {useBeta};");
                        L($"cvt.u64.u32 {col64}, {col};");
                        L($"shl.b64 {col64}, {col64}, 2;");
                        L($"add.u64 {ptr}, {rowPtr}, {col64};");
                        int c = (mt * _c.TilesN + nt) * 4 + half * 2;
                        string o0 = F(), o1 = F(), c0 = F(), c1 = F();
                        L($"mul.rn.f32 {o0}, {alpha}, {acc[c]};");
                        L($"mul.rn.f32 {o1}, {alpha}, {acc[c + 1]};");
                        L($"mov.f32 {c0}, 0f00000000;");
                        L($"mov.f32 {c1}, 0f00000000;");
                        L($"@{okBeta} ld.global.v2.f32 {{{c0}, {c1}}}, [{ptr}];");
                        L($"@{okBeta} fma.rn.f32 {o0}, {beta}, {c0}, {o0};");
                        L($"@{okBeta} fma.rn.f32 {o1}, {beta}, {c1}, {o1};");
                        L($"@{ok} st.global.v2.f32 [{ptr}], {{{o0}, {o1}}};");
                    }
                }
        }
    }
}
