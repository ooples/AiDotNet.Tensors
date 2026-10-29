// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// FlashAttention-2 style forward for FP16 Q/K/V on Ampere+ tensor cores (#1549): <c>O = softmax(scale * Q K^T) V</c>
/// for any query and key length, head dimension 64 or 128, causal or not, grouped-query heads. One block of four warps
/// owns 64 query rows (one m16 tile per warp, Q fragments held in registers, the Q block staged through the V buffer);
/// key/value tiles of 64 rows stream through one K and one V buffer (FlashAttention-2 order: K_{j+1} loads during the
/// softmax and P V of tile j, V_{j+1} during the next Q K^T) into padded shared rows (conflict-free <c>ldmatrix</c>).
/// Scores and the
/// output accumulate on <c>mma.sync.m16n8k16</c>; the online softmax (running row max and sum) stays FP32 in registers,
/// and no score or probability matrix touches memory.
/// <para>
/// FP16 accumulation (the GeForce full-rate form): each tile's Q K^T and P V products accumulate in FP16 and the P V
/// partials are promoted into FP32 output accumulators. An FP16 overflow (a score or tile partial past 65504) surfaces
/// as a non-finite value: every thread keeps a sticky flag over its scores and final outputs, the block votes after the
/// loop, and a block that saw one recomputes its rows with FP32 accumulation - the result never depends on magnitude.
/// </para>
/// </summary>
internal static class PtxFlashAttentionGenerator
{
    internal const string EntryPoint = "flash_attention_fp16";
    internal const int BlockN = 64;

    /// <summary>
    /// Warps per block, each owning one m16 tile of query rows. Eight (128 rows sharing each K/V tile, half the K/V
    /// traffic per row) at head dimension 128 and for causal attention; four for non-causal head dimension 64, where
    /// the smaller block keeps more blocks per SM in flight (RTX 3080, B2 H16 S4096: D64 1489 against 1575 us; D64
    /// causal 765 against 824 us; D128 3109 against 3271 us).
    /// </summary>
    internal static int Warps(int headDimension, bool isCausal) => headDimension == 128 || isCausal ? 8 : 4;

    internal static int Threads(int headDimension, bool isCausal) => 32 * Warps(headDimension, isCausal);

    /// <summary>Query rows per warp: one m16 tile (two would share every K and V fragment, but spill registers at both
    /// head dimensions).</summary>
    internal static int RowsPerWarp(int headDimension) => 16;

    /// <summary>Query rows per block.</summary>
    internal static int BlockM(int headDimension, bool isCausal) => Warps(headDimension, isCausal) * RowsPerWarp(headDimension);

    internal static bool SupportsHeadDimension(int headDimension) => headDimension is 64 or 128;

    /// <summary>Halves per shared-memory row: the head dimension padded by 8 so ldmatrix rows hit distinct banks.</summary>
    internal static int Pitch(int headDimension) => headDimension + 8;

    /// <summary>Dynamic shared memory: one K and one V tile of 64 padded rows (the Q block is staged through V). Small
    /// enough for four blocks per SM at head dimension 64, where registers, not shared memory, bound occupancy.</summary>
    internal static int SharedBytes(int headDimension) => 2 * BlockN * Pitch(headDimension) * 2;

    internal static string Emit(int ccMajor, int ccMinor, int headDimension, bool isCausal, bool fp16Accumulate,
        bool emitStats)
    {
        if (ccMajor < 8) throw new PlatformNotSupportedException("Tensor-core FP16 MMA and cp.async need compute capability 8.0+.");
        if (!SupportsHeadDimension(headDimension))
            throw new ArgumentOutOfRangeException(nameof(headDimension), "Flash attention supports head dimension 64 or 128.");
        return new Emitter(headDimension, isCausal, fp16Accumulate, emitStats).Build(ccMajor, ccMinor);
    }

    private sealed class Emitter
    {
        private readonly int _d;
        private readonly bool _causal;
        private readonly bool _fp16;
        private readonly bool _stats;
        private readonly StringBuilder _body = new();
        private int _r, _f, _p, _rd, _h;

        // Shared and lane state set up once.
        private string _tid = "", _warp = "", _lane = "", _g = "", _t = "";
        private string _sq = "", _sk = "", _scaleLog2 = "", _offset = "", _m0 = "";
        private string _qGlobal = "", _kGlobal = "", _vGlobal = "", _oGlobal = "", _lseGlobal = "", _bh = "";
        private string _qShared = "", _kShared = "", _vShared = "";
        private string _kLane = "", _vLane = "", _qLane = "";
        private string[] _chunkRow = Array.Empty<string>();
        private string[] _chunkShared = Array.Empty<string>();
        private string[] _chunkGlobal = Array.Empty<string>();
        private string[,,] _qf = new string[0, 0, 0];
        private string[,,] _o = new string[0, 0, 0];      // [m tile, d n8 tile, 4]
        private string[,] _rowMax = new string[0, 0];     // [m tile, row half (g, g + 8)]
        private string[,] _rowSum = new string[0, 0];
        private string _tiles = "", _bad = "";

        internal Emitter(int d, bool causal, bool fp16, bool stats)
        {
            _d = d;
            _causal = causal;
            _fp16 = fp16;
            _stats = stats;
        }

        private int Mt => RowsPerWarp(_d) / 16;   // m16 tiles per warp
        private int TileBytes => BlockN * Pitch(_d) * 2;
        private int RowBytes => Pitch(_d) * 2;
        private int KSteps => _d / 16;          // k16 steps over the head dimension
        private int DTiles => _d / 8;           // n8 output tiles over the head dimension
        private int ThreadCount => Threads(_d, _causal);
        private int ChunksPerThread => BlockN * (_d / 8) / ThreadCount;   // per 64-row tile

        private string R() => "%r" + I(_r++);
        private string F() => "%f" + I(_f++);
        private string P() => "%p" + I(_p++);
        private string RD() => "%rd" + I(_rd++);
        private string H() => "%h" + I(_h++);
        private void L(string line) => _body.Append("    ").AppendLine(line);
        private void Label(string name) => _body.Append(name).AppendLine(":");
        private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);
        internal string Build(int ccMajor, int ccMinor)
        {
            Setup();
            string tilesZero = P();
            L($"setp.eq.u32 {tilesZero}, {_tiles}, 0;");
            if (_fp16)
            {
                _bad = P();
                L($"setp.ne.u32 {_bad}, 0, 0;");
                L($"@{tilesZero} bra EPILOGUE;");
                MainLoop("_F16", half: true);
                // The block's vote: any thread whose scores or outputs went non-finite redoes the block in FP32.
                string finalBad = OutputsNonFinite();
                L($"or.pred {_bad}, {_bad}, {finalBad};");
                string any = P();
                L($"bar.red.or.pred {any}, 0, {_bad};");
                L($"@!{any} bra EPILOGUE;");
                ResetState();
                // Restart the stream: K tile 0 resident, V tile 0 in flight, as the loop expects.
                L("cp.async.wait_group 0;");
                L("bar.sync 0;");
                IssueKeys("0");
                L("cp.async.commit_group;");
                IssueValues("0");
                L("cp.async.commit_group;");
                L("cp.async.wait_group 1;");
                L("bar.sync 0;");
                MainLoop("_F32", half: false);
            }
            else
            {
                L($"@{tilesZero} bra EPILOGUE;");
                MainLoop("", half: false);
            }
            Label("EPILOGUE");
            L("cp.async.wait_group 0;");
            Epilogue();
            L("ret;");

            var ptx = new StringBuilder();
            ptx.AppendLine("//");
            ptx.Append("// flash-attention fp16 d").Append(I(_d)).Append(_causal ? " causal" : "")
                .Append(_fp16 ? " (fp16 accumulate per tile, fp32 redo on overflow)" : " (fp32 accumulate)")
                .AppendLine(_stats ? " + logsumexp" : "");
            ptx.AppendLine("//");
            ptx.AppendLine(".version 7.1");
            ptx.Append(".target ").AppendLine(PtxIsa71Target.For(ccMajor, ccMinor));
            ptx.AppendLine(".address_size 64");
            ptx.AppendLine(".extern .shared .align 16 .b8 smem[];");
            ptx.Append(".visible .entry ").Append(EntryPoint).AppendLine("(");
            ptx.AppendLine("    .param .u64 q_ptr, .param .u64 k_ptr, .param .u64 v_ptr, .param .u64 o_ptr, .param .u64 lse_ptr,");
            ptx.AppendLine("    .param .u32 sq, .param .u32 sk, .param .u32 q_heads, .param .u32 group,");
            ptx.AppendLine("    .param .f32 scale_log2, .param .s32 causal_offset)");
            ptx.AppendLine("{");
            ptx.Append("    .reg .b32 %r<").Append(I(Math.Max(_r, 1))).AppendLine(">;");
            ptx.Append("    .reg .f32 %f<").Append(I(Math.Max(_f, 1))).AppendLine(">;");
            ptx.Append("    .reg .pred %p<").Append(I(Math.Max(_p, 1))).AppendLine(">;");
            ptx.Append("    .reg .b64 %rd<").Append(I(Math.Max(_rd, 1))).AppendLine(">;");
            ptx.Append("    .reg .b16 %h<").Append(I(Math.Max(_h, 1))).AppendLine(">;");
            ptx.Append(_body);
            ptx.AppendLine("}");
            return ptx.ToString();
        }

        private void Setup()
        {
            _tid = R(); _warp = R(); _lane = R(); _g = R(); _t = R();
            L($"mov.u32 {_tid}, %tid.x;");
            L($"shr.u32 {_warp}, {_tid}, 5;");
            L($"and.b32 {_lane}, {_tid}, 31;");
            L($"shr.u32 {_g}, {_lane}, 2;");
            L($"and.b32 {_t}, {_lane}, 3;");

            _sq = R(); _sk = R(); _scaleLog2 = F(); _offset = R();
            string qHeads = R(), group = R();
            L($"ld.param.u32 {_sq}, [sq];");
            L($"ld.param.u32 {_sk}, [sk];");
            L($"ld.param.u32 {qHeads}, [q_heads];");
            L($"ld.param.u32 {group}, [group];");
            L($"ld.param.f32 {_scaleLog2}, [scale_log2];");
            L($"ld.param.s32 {_offset}, [causal_offset];");

            // Head mapping: bh = b * Hq + h; the K/V head is h / group of b's Hq / group heads.
            _bh = R();
            string b = R(), h = R(), kvHeads = R(), kvHead = R(), kvIndex = R();
            L($"mov.u32 {_bh}, %ctaid.y;");
            L($"div.u32 {b}, {_bh}, {qHeads};");
            L($"mul.lo.u32 {h}, {b}, {qHeads};");
            L($"sub.u32 {h}, {_bh}, {h};");
            L($"div.u32 {kvHeads}, {qHeads}, {group};");
            L($"div.u32 {kvHead}, {h}, {group};");
            L($"mad.lo.u32 {kvIndex}, {b}, {kvHeads}, {kvHead};");

            _qGlobal = RD(); _kGlobal = RD(); _vGlobal = RD(); _oGlobal = RD(); _lseGlobal = RD();
            string qElems = RD(), kvElems = RD(), tmp = RD();
            L($"ld.param.u64 {_qGlobal}, [q_ptr];");
            L($"ld.param.u64 {_kGlobal}, [k_ptr];");
            L($"ld.param.u64 {_vGlobal}, [v_ptr];");
            L($"ld.param.u64 {_oGlobal}, [o_ptr];");
            L($"ld.param.u64 {_lseGlobal}, [lse_ptr];");
            // Element offsets of this (b, h) slice: bh * sq * D and kvIndex * sk * D, as 64-bit.
            L($"mul.wide.u32 {qElems}, {_bh}, {_sq};");
            L($"mul.lo.u64 {qElems}, {qElems}, {I(_d)};");
            L($"mul.wide.u32 {kvElems}, {kvIndex}, {_sk};");
            L($"mul.lo.u64 {kvElems}, {kvElems}, {I(_d)};");
            L($"shl.b64 {tmp}, {qElems}, 1;");
            L($"add.u64 {_qGlobal}, {_qGlobal}, {tmp};");
            L($"shl.b64 {tmp}, {qElems}, 2;");
            L($"add.u64 {_oGlobal}, {_oGlobal}, {tmp};");
            L($"shl.b64 {tmp}, {kvElems}, 1;");
            L($"add.u64 {_kGlobal}, {_kGlobal}, {tmp};");
            L($"add.u64 {_vGlobal}, {_vGlobal}, {tmp};");

            _m0 = R();
            L($"mov.u32 {_m0}, %ctaid.x;");
            L($"mul.lo.u32 {_m0}, {_m0}, {I(BlockM(_d, _causal))};");

            // Shared regions: Q, K stage 0/1, V stage 0/1.
            _qShared = R(); _kShared = R(); _vShared = R();
            // K, then V (adjacent); the Q block (64 or 128 rows) is staged through the end of them before the first tiles.
            L($"mov.u32 {_kShared}, smem;");
            L($"add.u32 {_vShared}, {_kShared}, {I(TileBytes)};");
            L($"add.u32 {_qShared}, {_kShared}, {I((2 - BlockM(_d, _causal) / BlockN) * TileBytes)};");

            // This thread's cp.async chunks of a 64-row tile: chunk c = tid + i * 128 is row c / (D / 8), column c % (D / 8).
            int chunksPerRow = _d / 8;
            _chunkRow = new string[ChunksPerThread];
            _chunkShared = new string[ChunksPerThread];
            _chunkGlobal = new string[ChunksPerThread];
            for (int i = 0; i < ChunksPerThread; i++)
            {
                string c = R(), col = R();
                _chunkRow[i] = R(); _chunkShared[i] = R(); _chunkGlobal[i] = RD();
                L($"add.u32 {c}, {_tid}, {I(i * ThreadCount)};");
                L($"shr.u32 {_chunkRow[i]}, {c}, {I(Log2(chunksPerRow))};");
                L($"and.b32 {col}, {c}, {I(chunksPerRow - 1)};");
                L($"mul.lo.u32 {_chunkShared[i]}, {_chunkRow[i]}, {I(RowBytes)};");
                L($"mad.lo.u32 {_chunkShared[i]}, {col}, 16, {_chunkShared[i]};");
                string rowBytes = R();
                L($"mul.lo.u32 {rowBytes}, {_chunkRow[i]}, {I(_d * 2)};");
                L($"mad.lo.u32 {rowBytes}, {col}, 16, {rowBytes};");
                L($"cvt.u64.u32 {_chunkGlobal[i]}, {rowBytes};");
            }
            // ldmatrix lane addresses (lane = 8 * matrix + row): Q as the A operand (x4: rows (m & 1) * 8, k (m >> 1) * 8);
            // K as the B operand of S = Q K^T, two key n8 tiles per x4 (keys (m >> 1) * 8, k (m & 1) * 8); V as the B
            // operand of P V through .trans, two d n8 tiles per x4 (keys (m & 1) * 8, d (m >> 1) * 8).
            string matrix = R(), row = R(), lo = R(), hi = R();
            L($"shr.u32 {matrix}, {_lane}, 3;");
            L($"and.b32 {row}, {_lane}, 7;");
            L($"and.b32 {lo}, {matrix}, 1;");
            L($"shr.u32 {hi}, {matrix}, 1;");
            _qLane = R(); _kLane = R(); _vLane = R();
            string tmpRow = R();
            // Q: row = warp * rows-per-warp + lo * 8 + row, col = hi * 8 (m tile t adds 16 t rows).
            L($"mul.lo.u32 {tmpRow}, {_warp}, {I(RowsPerWarp(_d))};");
            L($"mad.lo.u32 {tmpRow}, {lo}, 8, {tmpRow};");
            L($"add.u32 {tmpRow}, {tmpRow}, {row};");
            L($"mul.lo.u32 {_qLane}, {tmpRow}, {I(RowBytes)};");
            L($"mad.lo.u32 {_qLane}, {hi}, 16, {_qLane};");
            L($"add.u32 {_qLane}, {_qLane}, {_qShared};");
            // K: key = hi * 8 + row, col = lo * 8.
            L($"mad.lo.u32 {tmpRow}, {hi}, 8, {row};");
            L($"mul.lo.u32 {_kLane}, {tmpRow}, {I(RowBytes)};");
            L($"mad.lo.u32 {_kLane}, {lo}, 16, {_kLane};");
            // V: key = lo * 8 + row, col = hi * 8.
            L($"mad.lo.u32 {tmpRow}, {lo}, 8, {row};");
            L($"mul.lo.u32 {_vLane}, {tmpRow}, {I(RowBytes)};");
            L($"mad.lo.u32 {_vLane}, {hi}, 16, {_vLane};");

            // Key tiles: ceil(sk / 64); causal: up to the tile of the block's last key, (min(m0 + BlockM - 1, sq - 1) + offset).
            _tiles = R();
            L($"add.u32 {_tiles}, {_sk}, 63;");
            L($"shr.u32 {_tiles}, {_tiles}, 6;");
            if (_causal)
            {
                string lastRow = R(), lastKey = R(), causalTiles = R(), none = P();
                L($"add.u32 {lastRow}, {_m0}, {I(BlockM(_d, _causal) - 1)};");
                L($"sub.u32 {lastKey}, {_sq}, 1;");
                L($"min.u32 {lastRow}, {lastRow}, {lastKey};");
                L($"add.s32 {lastKey}, {lastRow}, {_offset};");
                L($"setp.lt.s32 {none}, {lastKey}, 0;");
                L($"shr.s32 {causalTiles}, {lastKey}, 6;");
                L($"add.s32 {causalTiles}, {causalTiles}, 1;");
                L($"min.u32 {_tiles}, {_tiles}, {causalTiles};");
                L($"selp.u32 {_tiles}, 0, {_tiles}, {none};");
            }

            // Output accumulators, running max (log2 units) and sum for rows g and g + 8.
            _o = new string[Mt, DTiles, 4];
            _rowMax = new string[Mt, 2];
            _rowSum = new string[Mt, 2];
            for (int m = 0; m < Mt; m++)
            {
                for (int n = 0; n < DTiles; n++) for (int e = 0; e < 4; e++) _o[m, n, e] = F();
                for (int r = 0; r < 2; r++) { _rowMax[m, r] = F(); _rowSum[m, r] = F(); }
            }
            ResetState();

            // The Q block (64-row tiles), then K/V tile 0, as the first cp.async group.
            for (int q = 0; q < BlockM(_d, _causal) / BlockN; q++)
            {
                string rowStart = R(), shared = R();
                L($"add.u32 {rowStart}, {_m0}, {I(q * BlockN)};");
                L($"add.u32 {shared}, {_qShared}, {I(q * TileBytes)};");
                IssueTile(_qGlobal, shared, rowStart, _sq);
            }
            _qf = new string[Mt, KSteps, 4];
            for (int m = 0; m < Mt; m++) for (int s = 0; s < KSteps; s++) for (int e = 0; e < 4; e++) _qf[m, s, e] = R();
            // Q fragments to registers, then K tile 0 resident and V tile 0 in flight into the freed buffers.
            L("cp.async.commit_group;");
            L("cp.async.wait_group 0;");
            L("bar.sync 0;");
            LoadQueryFragments();
            L("bar.sync 0;");
            IssueKeys("0");
            L("cp.async.commit_group;");
            IssueValues("0");
            L("cp.async.commit_group;");
            L("cp.async.wait_group 1;");
            L("bar.sync 0;");
        }

        private void ResetState()
        {
            for (int m = 0; m < Mt; m++)
            {
                for (int n = 0; n < DTiles; n++) for (int e = 0; e < 4; e++) L($"mov.f32 {_o[m, n, e]}, 0f00000000;");
                for (int r = 0; r < 2; r++)
                {
                    L($"mov.f32 {_rowMax[m, r]}, 0fFF800000;");
                    L($"mov.f32 {_rowSum[m, r]}, 0f00000000;");
                }
            }
        }

        // cp.async of a 64-row tile starting at global row `rowStart` (rows at or past `rowLimit` zero-fill).
        private void IssueTile(string globalBase, string shared, string rowStart, string rowLimit)
        {
            string rowOffset = RD();
            L($"mul.wide.u32 {rowOffset}, {rowStart}, {I(_d * 2)};");
            for (int i = 0; i < ChunksPerThread; i++)
            {
                string rowGlobal = R(), inBounds = P(), bytes = R(), dst = R(), src = RD();
                L($"add.u32 {rowGlobal}, {rowStart}, {_chunkRow[i]};");
                L($"setp.lt.u32 {inBounds}, {rowGlobal}, {rowLimit};");
                L($"selp.u32 {bytes}, 16, 0, {inBounds};");
                L($"add.u64 {src}, {globalBase}, {rowOffset};");
                L($"add.u64 {src}, {src}, {_chunkGlobal[i]};");
                // A zero-byte copy reads nothing, but keep its address inside the slice.
                L($"selp.b64 {src}, {src}, {globalBase}, {inBounds};");
                L($"add.u32 {dst}, {shared}, {_chunkShared[i]};");
                L($"cp.async.cg.shared.global [{dst}], [{src}], 16, {bytes};");
            }
        }

        private void LoadQueryFragments()
        {
            for (int m = 0; m < Mt; m++)
                for (int s = 0; s < KSteps; s++)
                    L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{_qf[m, s, 0]}, {_qf[m, s, 1]}, {_qf[m, s, 2]}, {_qf[m, s, 3]}}}, " +
                      $"[{_qLane}+{I(m * 16 * RowBytes + s * 32)}];");
        }

        // K or V tile `tile` (a register or "0") into its one buffer.
        private void IssueKeys(string tile) => IssueOne(_kGlobal, _kShared, tile);

        private void IssueValues(string tile) => IssueOne(_vGlobal, _vShared, tile);

        private void IssueOne(string global, string shared, string tile)
        {
            string rowStart = R();
            if (tile == "0") L($"mov.u32 {rowStart}, 0;");
            else L($"shl.b32 {rowStart}, {tile}, 6;");
            IssueTile(global, shared, rowStart, _sk);
        }

        // One m tile at a time - promote (FP16), mask, softmax, pack to P fragments - so only one tile's FP32 scores are
        // live at once. Returns the P fragments [m, k16 step, 4].
        private string[,,] SoftmaxAndPack(string[,,] raw, string j, bool half, string suffix)
        {
            string needMask = NeedMask(j, out string n0, out string warpRow);
            var pa = new string[Mt, 4, 4];
            for (int m = 0; m < Mt; m++)
            {
                string[] tileMax;
                string[,] tile;
                if (half)
                {
                    // Packed FP16 scores: overflow check, masking and the raw row max on half2 registers, then FP32.
                    CheckHalfScores(raw, m);
                    MaskHalf(raw, m, needMask, n0, warpRow, suffix);
                    tileMax = new[] { HalfRowMax(raw, m, 0), HalfRowMax(raw, m, 1) };
                    tile = new string[8, 4];
                    for (int n = 0; n < 8; n++)
                        for (int e = 0; e < 2; e++)
                        {
                            string h0 = H(), h1 = H();
                            tile[n, 2 * e] = F();
                            tile[n, 2 * e + 1] = F();
                            L($"mov.b32 {{{h0}, {h1}}}, {raw[m, n, e]};");
                            L($"cvt.f32.f16 {tile[n, 2 * e]}, {h0};");
                            L($"cvt.f32.f16 {tile[n, 2 * e + 1]}, {h1};");
                        }
                }
                else
                {
                    tile = Slice(raw, m);
                    Mask(tile, m, needMask, n0, warpRow, suffix);
                    tileMax = new[] { FloatRowMax(tile, 0), FloatRowMax(tile, 1) };
                }
                string[] alpha = { RowExponentials(tile, m, 0, tileMax[0]), RowExponentials(tile, m, 1, tileMax[1]) };
                for (int k = 0; k < 4; k++)
                {
                    pa[m, k, 0] = Pack(tile[2 * k, 0], tile[2 * k, 1]);
                    pa[m, k, 1] = Pack(tile[2 * k, 2], tile[2 * k, 3]);
                    pa[m, k, 2] = Pack(tile[2 * k + 1, 0], tile[2 * k + 1, 1]);
                    pa[m, k, 3] = Pack(tile[2 * k + 1, 2], tile[2 * k + 1, 3]);
                }
                // Row sums from the packed P the P V product uses (a0, a2 hold row g; a1, a3 row g + 8).
                for (int r = 0; r < 2; r++)
                {
                    string sum2 = R();
                    L($"add.rn.f16x2 {sum2}, {pa[m, 0, r]}, {pa[m, 0, r + 2]};");
                    for (int k = 1; k < 4; k++)
                    {
                        L($"add.rn.f16x2 {sum2}, {sum2}, {pa[m, k, r]};");
                        L($"add.rn.f16x2 {sum2}, {sum2}, {pa[m, k, r + 2]};");
                    }
                    string h0 = H(), h1 = H(), x = F(), y = F();
                    L($"mov.b32 {{{h0}, {h1}}}, {sum2};");
                    L($"cvt.f32.f16 {x}, {h0};");
                    L($"cvt.f32.f16 {y}, {h1};");
                    L($"add.f32 {x}, {x}, {y};");
                    L($"fma.rn.f32 {_rowSum[m, r]}, {_rowSum[m, r]}, {alpha[r]}, {x};");
                }
                RescaleOutputs(m, alpha);
            }
            return pa;
        }

        // One K and one V buffer: entering tile j, K_j is resident and V_j in flight. S = Q K_j; K_{j+1} streams in during
        // the softmax and P V; V_{j+1} during the next tile's Q K^T.
        private void MainLoop(string suffix, bool half)
        {
            string j = R();
            L($"mov.u32 {j}, 0;");
            Label("TILE" + suffix);
            string next = R(), hasNext = P();
            L($"add.u32 {next}, {j}, 1;");
            L($"setp.lt.u32 {hasNext}, {next}, {_tiles};");
            string kLane = R(), vLane = R();
            L($"add.u32 {kLane}, {_kShared}, {_kLane};");
            L($"add.u32 {vLane}, {_vShared}, {_vLane};");

            string[,,] raw = Scores(kLane, half);
            L("bar.sync 0;");                                   // every warp is done with K_j
            L($"@!{hasNext} bra NOK{suffix};");
            IssueKeys(next);
            Label("NOK" + suffix);
            L("cp.async.commit_group;");

            var pa = SoftmaxAndPack(raw, j, half, suffix);

            L("cp.async.wait_group 1;");                        // V_j landed (K_{j+1} may still be in flight)
            L("bar.sync 0;");
            ProbabilitiesTimesValues(pa, vLane, half);
            L("bar.sync 0;");                                   // every warp is done with V_j
            L($"@!{hasNext} bra NOV{suffix};");
            IssueValues(next);
            Label("NOV" + suffix);
            L("cp.async.commit_group;");
            L("cp.async.wait_group 1;");                        // K_{j+1} landed (V_{j+1} may still be in flight)
            L("bar.sync 0;");

            L($"add.u32 {j}, {j}, 1;");
            string more = P();
            L($"setp.lt.u32 {more}, {j}, {_tiles};");
            L($"@{more} bra TILE{suffix};");
        }

        // S (per m tile, 16 x 64) = Q K^T over eight n8 key tiles, every K fragment pair feeding each of the warp's m
        // tiles: FP32 accumulators [m, n, 4] (c0, c1 row g; c2, c3 row g + 8), or packed FP16 pairs [m, n, 2] when half.
        private string[,,] Scores(string kStage, bool half)
        {
            var acc = new string[Mt, 8, half ? 2 : 4];
            for (int m = 0; m < Mt; m++)
                for (int n = 0; n < 8; n++)
                    for (int e = 0; e < acc.GetLength(2); e++)
                    {
                        if (half)
                        {
                            acc[m, n, e] = R();
                            L($"mov.b32 {acc[m, n, e]}, 0;");
                        }
                        else
                        {
                            acc[m, n, e] = F();
                            L($"mov.f32 {acc[m, n, e]}, 0f00000000;");
                        }
                    }
            for (int pair = 0; pair < 4; pair++)
            {
                for (int k = 0; k < KSteps; k++)
                {
                    string b0 = R(), b1 = R(), b2 = R(), b3 = R();
                    L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{b0}, {b1}, {b2}, {b3}}}, [{kStage}+{I(pair * 16 * RowBytes + k * 32)}];");
                    for (int m = 0; m < Mt; m++)
                    {
                        string a = $"{{{_qf[m, k, 0]}, {_qf[m, k, 1]}, {_qf[m, k, 2]}, {_qf[m, k, 3]}}}";
                        for (int sub = 0; sub < 2; sub++)
                        {
                            int n = pair * 2 + sub;
                            string b = sub == 0 ? $"{{{b0}, {b1}}}" : $"{{{b2}, {b3}}}";
                            string c = half
                                ? $"{{{acc[m, n, 0]}, {acc[m, n, 1]}}}"
                                : $"{{{acc[m, n, 0]}, {acc[m, n, 1]}, {acc[m, n, 2]}, {acc[m, n, 3]}}}";
                            L(half
                                ? $"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {c}, {a}, {b}, {c};"
                                : $"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {c}, {a}, {b}, {c};");
                        }
                    }
                }
            }
            return acc;
        }

        private static string[,] Slice(string[,,] acc, int m)
        {
            var tile = new string[8, 4];
            for (int n = 0; n < 8; n++) for (int e = 0; e < 4; e++) tile[n, e] = acc[m, n, e];
            return tile;
        }

        // Overflow check on one m tile's packed FP16 scores: max |s| over every half (NaN propagating); an infinite
        // (overflowed) or NaN score raises the thread's sticky flag.
        private void CheckHalfScores(string[,,] acc, int m)
        {
            string absMax = R();
            L($"abs.f16x2 {absMax}, {acc[m, 0, 0]};");
            for (int n = 0; n < 8; n++)
                for (int e = 0; e < 2; e++)
                {
                    if (n == 0 && e == 0) continue;
                    string a = R();
                    L($"abs.f16x2 {a}, {acc[m, n, e]};");
                    L($"max.NaN.f16x2 {absMax}, {absMax}, {a};");
                }
            FlagNonFiniteHalf2(absMax);
        }

        // Raises the sticky flag when either half of a packed |x| max is infinite or NaN (bits at or above 0x7C00).
        private void FlagNonFiniteHalf2(string absMax)
        {
            string lo = R(), hi = R(), nonFinite = P(), other = P();
            L($"and.b32 {lo}, {absMax}, 0x7FFF;");
            L($"shr.u32 {hi}, {absMax}, 16;");
            L($"and.b32 {hi}, {hi}, 0x7FFF;");
            L($"setp.ge.u32 {nonFinite}, {lo}, 0x7C00;");
            L($"setp.ge.u32 {other}, {hi}, 0x7C00;");
            L($"or.pred {_bad}, {_bad}, {nonFinite};");
            L($"or.pred {_bad}, {_bad}, {other};");
        }

        // The raw (unscaled) max of one row of an m tile's packed scores ({c0, c1} = row g, {c2, c3} = row g + 8), across
        // the quad, as FP32.
        private string HalfRowMax(string[,,] acc, int m, int rowHalf)
        {
            string max2 = R();
            L($"max.f16x2 {max2}, {acc[m, 0, rowHalf]}, {acc[m, 1, rowHalf]};");
            for (int n = 2; n < 8; n++) L($"max.f16x2 {max2}, {max2}, {acc[m, n, rowHalf]};");
            string h0 = H(), h1 = H(), x = F(), y = F();
            L($"mov.b32 {{{h0}, {h1}}}, {max2};");
            L($"cvt.f32.f16 {x}, {h0};");
            L($"cvt.f32.f16 {y}, {h1};");
            L($"max.f32 {x}, {x}, {y};");
            QuadReduce("max.f32", x);
            return x;
        }

        private string FloatRowMax(string[,] tile, int rowHalf)
        {
            int e0 = rowHalf * 2;
            string tileMax = F();
            L($"max.f32 {tileMax}, {tile[0, e0]}, {tile[0, e0 + 1]};");
            for (int n = 1; n < 8; n++)
            {
                L($"max.f32 {tileMax}, {tileMax}, {tile[n, e0]};");
                L($"max.f32 {tileMax}, {tileMax}, {tile[n, e0 + 1]};");
            }
            QuadReduce("max.f32", tileMax);
            return tileMax;
        }

        // Masking on packed FP16 scores: a masked half becomes -inf (0xFC00).
        private void MaskHalf(string[,,] acc, int m, string needMask, string n0, string warpRow, string suffix)
        {
            string label = "UNMASKED" + suffix + "_" + I(m);
            L($"@!{needMask} bra {label};");
            string keyBase = R();
            L($"shl.b32 {keyBase}, {_t}, 1;");
            L($"add.u32 {keyBase}, {keyBase}, {n0};");
            string[] limit = RowLimits(m, warpRow);
            string negInf = H();
            L($"mov.b16 {negInf}, 0xFC00;");
            for (int n = 0; n < 8; n++)
            {
                string key = R();
                L($"add.s32 {key}, {keyBase}, {I(n * 8)};");
                for (int r = 0; r < 2; r++)
                {
                    string lo = H(), hi = H(), maskLo = P(), maskHi = P(), next = R();
                    L($"add.s32 {next}, {key}, 1;");
                    L($"setp.ge.s32 {maskLo}, {key}, {limit[r]};");
                    L($"setp.ge.s32 {maskHi}, {next}, {limit[r]};");
                    L($"mov.b32 {{{lo}, {hi}}}, {acc[m, n, r]};");
                    L($"selp.b16 {lo}, {negInf}, {lo}, {maskLo};");
                    L($"selp.b16 {hi}, {negInf}, {hi}, {maskHi};");
                    L($"mov.b32 {acc[m, n, r]}, {{{lo}, {hi}}};");
                }
            }
            Label(label);
        }

        // Row r's key limit (exclusive) for m tile m: sk, or (causal) min(sk, row + offset + 1).
        private string[] RowLimits(int m, string warpRow)
        {
            var limit = new string[2];
            for (int r = 0; r < 2; r++)
            {
                limit[r] = R();
                if (_causal)
                {
                    string row = R();
                    L($"add.u32 {row}, {warpRow}, {_g};");
                    L($"add.u32 {row}, {row}, {I(m * 16 + r * 8)};");
                    L($"add.s32 {limit[r]}, {row}, {_offset};");
                    L($"add.s32 {limit[r]}, {limit[r]}, 1;");
                    L($"min.s32 {limit[r]}, {limit[r]}, {_sk};");
                }
                else
                {
                    L($"mov.u32 {limit[r]}, {_sk};");
                }
            }
            return limit;
        }

        // Whether this key tile needs masking for this warp: keys past sk, or (causal) keys after the warp's first row.
        // Warp-uniform, so the full tiles skip the masking code.
        private string NeedMask(string j, out string n0, out string warpRow)
        {
            n0 = R();
            string tileEnd = R(), needMask = P();
            L($"shl.b32 {n0}, {j}, 6;");
            L($"add.u32 {tileEnd}, {n0}, 64;");
            L($"setp.gt.u32 {needMask}, {tileEnd}, {_sk};");
            warpRow = R();
            L($"mul.lo.u32 {warpRow}, {_warp}, {I(RowsPerWarp(_d))};");
            L($"add.u32 {warpRow}, {warpRow}, {_m0};");
            if (_causal)
            {
                string warpLimit = R(), lastKey = R(), causalMask = P();
                L($"add.s32 {warpLimit}, {warpRow}, {_offset};");
                L($"add.s32 {lastKey}, {n0}, 63;");
                L($"setp.gt.s32 {causalMask}, {lastKey}, {warpLimit};");
                L($"or.pred {needMask}, {needMask}, {causalMask};");
            }
            return needMask;
        }

        private void Mask(string[,] tile, int m, string needMask, string n0, string warpRow, string suffix)
        {
            string label = "UNMASKED" + suffix + "_" + I(m);
            L($"@!{needMask} bra {label};");
            string keyBase = R();
            L($"shl.b32 {keyBase}, {_t}, 1;");
            L($"add.u32 {keyBase}, {keyBase}, {n0};");
            // Allowed keys: key < sk and key <= row + offset, i.e. key < min(sk, row + offset + 1) (signed).
            string[] limit = RowLimits(m, warpRow);
            for (int n = 0; n < 8; n++)
                for (int e = 0; e < 2; e++)
                {
                    string key = R(), m0 = P(), m1 = P();
                    L($"add.s32 {key}, {keyBase}, {I(n * 8 + e)};");
                    L($"setp.ge.s32 {m0}, {key}, {limit[0]};");
                    L($"setp.ge.s32 {m1}, {key}, {limit[1]};");
                    L($"@{m0} mov.f32 {tile[n, e]}, 0fFF800000;");
                    L($"@{m1} mov.f32 {tile[n, 2 + e]}, 0fFF800000;");
                }
            Label(label);
        }

        // Online softmax for one of the thread's rows (c0, c1 = row g; c2, c3 = row g + 8 of m tile m), given the row's
        // raw tile max. Scores stay unscaled: the max is scaled once (scale is positive) and each exponent is
        // ex2(fma(s, scale log2 e, -max)). Updates the running max (log2 units), leaves the exponentials in place of the
        // scores, and returns alpha = ex2(old max - new max), the factor for the row's sum and outputs.
        private string RowExponentials(string[,] tile, int m, int rowHalf, string tileMax)
        {
            int e0 = rowHalf * 2;
            string rowMax = _rowMax[m, rowHalf];
            string newMax = F(), use = F(), negUse = F(), allMasked = P(), alpha = F();
            L($"mul.f32 {tileMax}, {tileMax}, {_scaleLog2};");
            L($"max.f32 {newMax}, {rowMax}, {tileMax};");
            // A row with every key masked so far keeps max -inf; exponentiate against 0 so it yields zeros, not NaN.
            L($"setp.eq.f32 {allMasked}, {newMax}, 0fFF800000;");
            L($"selp.f32 {use}, 0f00000000, {newMax}, {allMasked};");
            L($"sub.f32 {alpha}, {rowMax}, {use};");
            L($"ex2.approx.ftz.f32 {alpha}, {alpha};");
            L($"mov.f32 {rowMax}, {newMax};");
            L($"neg.f32 {negUse}, {use};");
            for (int n = 0; n < 8; n++)
                for (int e = e0; e < e0 + 2; e++)
                {
                    L($"fma.rn.f32 {tile[n, e]}, {tile[n, e]}, {_scaleLog2}, {negUse};");
                    L($"ex2.approx.ftz.f32 {tile[n, e]}, {tile[n, e]};");
                }
            return alpha;
        }

        // O *= alpha per row, every tile. A warp vote that skipped the multiplies once the row maxes settle (alpha
        // exactly 1) measured slower at D = 128: the branch splits the tile body and stops ptxas interleaving the
        // softmax with the P V MMAs.
        private void RescaleOutputs(int m, string[] alpha)
        {
            for (int r = 0; r < 2; r++)
                for (int n = 0; n < DTiles; n++)
                {
                    L($"mul.f32 {_o[m, n, 2 * r]}, {_o[m, n, 2 * r]}, {alpha[r]};");
                    L($"mul.f32 {_o[m, n, 2 * r + 1]}, {_o[m, n, 2 * r + 1]}, {alpha[r]};");
                }
        }

        private void QuadReduce(string op, string value)
        {
            foreach (int delta in new[] { 1, 2 })
            {
                string other = F();
                L($"shfl.sync.bfly.b32 {other}, {value}, {I(delta)}, 31, 0xffffffff;");
                L($"{op} {value}, {value}, {other};");
            }
        }

        // O += P V: P (the exponentials packed to FP16 A fragments [m, k16 step, 4]: the score layout is the A layout)
        // times the V tile; every V fragment pair feeds each m tile.
        private void ProbabilitiesTimesValues(string[,,] pa, string vStage, bool half)
        {
            var acc = new string[Mt, DTiles, 2];
            if (half)
                for (int m = 0; m < Mt; m++)
                    for (int n = 0; n < DTiles; n++)
                        for (int e = 0; e < 2; e++)
                        {
                            acc[m, n, e] = R();
                            L($"mov.b32 {acc[m, n, e]}, 0;");
                        }
            for (int k = 0; k < 4; k++)
            {
                for (int pair = 0; pair < DTiles / 2; pair++)
                {
                    string b0 = R(), b1 = R(), b2 = R(), b3 = R();
                    L($"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{b0}, {b1}, {b2}, {b3}}}, [{vStage}+{I(k * 16 * RowBytes + pair * 32)}];");
                    for (int m = 0; m < Mt; m++)
                    {
                        string a = $"{{{pa[m, k, 0]}, {pa[m, k, 1]}, {pa[m, k, 2]}, {pa[m, k, 3]}}}";
                        for (int sub = 0; sub < 2; sub++)
                        {
                            int n = pair * 2 + sub;
                            string b = sub == 0 ? $"{{{b0}, {b1}}}" : $"{{{b2}, {b3}}}";
                            if (half)
                            {
                                string c = $"{{{acc[m, n, 0]}, {acc[m, n, 1]}}}";
                                L($"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {c}, {a}, {b}, {c};");
                            }
                            else
                            {
                                string c = $"{{{_o[m, n, 0]}, {_o[m, n, 1]}, {_o[m, n, 2]}, {_o[m, n, 3]}}}";
                                L($"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {c}, {a}, {b}, {c};");
                            }
                        }
                    }
                }
            }
            if (half)
                for (int m = 0; m < Mt; m++)
                    for (int n = 0; n < DTiles; n++)
                        for (int e = 0; e < 2; e++)
                        {
                            string h0 = H(), h1 = H(), x = F(), y = F();
                            L($"mov.b32 {{{h0}, {h1}}}, {acc[m, n, e]};");
                            L($"cvt.f32.f16 {x}, {h0};");
                            L($"cvt.f32.f16 {y}, {h1};");
                            L($"add.f32 {_o[m, n, 2 * e]}, {_o[m, n, 2 * e]}, {x};");
                            L($"add.f32 {_o[m, n, 2 * e + 1]}, {_o[m, n, 2 * e + 1]}, {y};");
                        }
        }

        // Two FP32 values to one f16x2 register in a single conversion (the first operand lands in the upper half).
        private string Pack(string low, string high)
        {
            string packed = R();
            L($"cvt.rn.f16x2.f32 {packed}, {high}, {low};");
            return packed;
        }

        // Any of this thread's output accumulators or row sums non-finite (an FP16 P V partial overflowed).
        private string OutputsNonFinite()
        {
            string sum = F();
            L($"mov.f32 {sum}, 0f00000000;");
            for (int m = 0; m < Mt; m++)
            {
                for (int r = 0; r < 2; r++) L($"add.f32 {sum}, {sum}, {_rowSum[m, r]};");
                for (int n = 0; n < DTiles; n++) for (int e = 0; e < 4; e++) L($"add.f32 {sum}, {sum}, {_o[m, n, e]};");
            }
            string abs = F(), nonFinite = P();
            L($"abs.f32 {abs}, {sum};");
            L($"setp.gtu.f32 {nonFinite}, {abs}, 0f7F7FFFFF;");
            return nonFinite;
        }

        private void Epilogue()
        {
            string warpRow = R();
            L($"mul.lo.u32 {warpRow}, {_warp}, {I(RowsPerWarp(_d))};");
            L($"add.u32 {warpRow}, {warpRow}, {_m0};");
            L($"add.u32 {warpRow}, {warpRow}, {_g};");
            for (int m = 0; m < Mt; m++)
                for (int r = 0; r < 2; r++)
                {
                    QuadReduce("add.f32", _rowSum[m, r]);
                    string row = R();
                    L($"add.u32 {row}, {warpRow}, {I(m * 16 + r * 8)};");
                    StoreRow(row, m, r);
                }
        }

        private void StoreRow(string row, int m, int rowHalf)
        {
            int e0 = rowHalf * 2;
            string rowSum = _rowSum[m, rowHalf], rowMax = _rowMax[m, rowHalf];
            string skip = P(), done = "ROWDONE" + I(m) + "_" + I(rowHalf);
            L($"setp.ge.u32 {skip}, {row}, {_sq};");
            L($"@{skip} bra {done};");
            // A row with no key (sum 0) stores zeros.
            string inv = F(), empty = P();
            L($"setp.eq.f32 {empty}, {rowSum}, 0f00000000;");
            L($"rcp.rn.f32 {inv}, {rowSum};");
            L($"selp.f32 {inv}, 0f00000000, {inv}, {empty};");
            string elems = RD(), address = RD(), colBytes = RD(), tcol = R();
            L($"mul.wide.u32 {elems}, {row}, {I(_d)};");
            L($"shl.b64 {address}, {elems}, 2;");
            L($"add.u64 {address}, {address}, {_oGlobal};");
            L($"shl.b32 {tcol}, {_t}, 3;");
            L($"cvt.u64.u32 {colBytes}, {tcol};");
            L($"add.u64 {address}, {address}, {colBytes};");
            for (int n = 0; n < DTiles; n++)
            {
                string x = F(), y = F();
                L($"mul.f32 {x}, {_o[m, n, e0]}, {inv};");
                L($"mul.f32 {y}, {_o[m, n, e0 + 1]}, {inv};");
                L($"st.global.v2.f32 [{address}+{I(n * 32)}], {{{x}, {y}}};");
            }
            if (_stats)
            {
                // logsumexp (natural) of the row's scaled scores: ln 2 * (max + log2 sum); -inf for an empty row.
                string notLead = P(), lse = F(), lg = F(), lseAddress = RD(), index = RD();
                L($"setp.ne.u32 {notLead}, {_t}, 0;");
                L($"@{notLead} bra {done};");
                L($"lg2.approx.f32 {lg}, {rowSum};");
                L($"add.f32 {lse}, {rowMax}, {lg};");
                L($"mul.f32 {lse}, {lse}, 0f3F317218;");
                L($"selp.f32 {lse}, 0fFF800000, {lse}, {empty};");
                L($"mul.wide.u32 {index}, {_bh}, {_sq};");
                L($"cvt.u64.u32 {lseAddress}, {row};");
                L($"add.u64 {index}, {index}, {lseAddress};");
                L($"shl.b64 {index}, {index}, 2;");
                L($"add.u64 {lseAddress}, {_lseGlobal}, {index};");
                L($"st.global.f32 [{lseAddress}], {lse};");
            }
            Label(done);
        }

        private static int Log2(int value)
        {
            int log = 0;
            while ((1 << log) < value) log++;
            return log;
        }
    }
}