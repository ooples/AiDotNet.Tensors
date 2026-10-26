// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// FlashAttention-2 style forward for FP16 Q/K/V on Ampere+ tensor cores (#1549): <c>O = softmax(scale * Q K^T) V</c>
/// for any query and key length, head dimension 64 or 128, causal or not, grouped-query heads. One block of four warps
/// owns 64 query rows (16 per warp, Q fragments held in registers); key/value tiles of 64 rows stream through a
/// double-buffered <c>cp.async</c> pipeline into padded shared rows (conflict-free <c>ldmatrix</c>). Scores and the
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
    internal const int BlockM = 64;
    internal const int BlockN = 64;
    internal const int Threads = 128;

    internal static bool SupportsHeadDimension(int headDimension) => headDimension is 64 or 128;

    /// <summary>Halves per shared-memory row: the head dimension padded by 8 so ldmatrix rows hit distinct banks.</summary>
    internal static int Pitch(int headDimension) => headDimension + 8;

    /// <summary>Dynamic shared memory: Q plus two K and two V stages of 64 padded rows.</summary>
    internal static int SharedBytes(int headDimension) => 5 * BlockM * Pitch(headDimension) * 2;

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
        private string[,] _qf = new string[0, 0];
        private string[,] _o = new string[0, 0];
        private string _rowMax0 = "", _rowMax1 = "", _rowSum0 = "", _rowSum1 = "";
        private string _tiles = "", _bad = "";

        internal Emitter(int d, bool causal, bool fp16, bool stats)
        {
            _d = d;
            _causal = causal;
            _fp16 = fp16;
            _stats = stats;
        }

        private int TileBytes => BlockM * Pitch(_d) * 2;
        private int RowBytes => Pitch(_d) * 2;
        private int KSteps => _d / 16;          // k16 steps over the head dimension
        private int DTiles => _d / 8;           // n8 output tiles over the head dimension
        private int ChunksPerThread => BlockM * (_d / 8) / Threads;

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
                IssueKeyValue("0", stage: 0);
                L("cp.async.commit_group;");
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
            ptx.Append(".target sm_").Append(I(ccMajor)).AppendLine(I(ccMinor));
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
            L($"shl.b32 {_m0}, {_m0}, 6;");

            // Shared regions: Q, K stage 0/1, V stage 0/1.
            _qShared = R(); _kShared = R(); _vShared = R();
            L($"mov.u32 {_qShared}, smem;");
            L($"add.u32 {_kShared}, {_qShared}, {I(TileBytes)};");
            L($"add.u32 {_vShared}, {_kShared}, {I(2 * TileBytes)};");

            // This thread's cp.async chunks of a 64-row tile: chunk c = tid + i * 128 is row c / (D / 8), column c % (D / 8).
            int chunksPerRow = _d / 8;
            _chunkRow = new string[ChunksPerThread];
            _chunkShared = new string[ChunksPerThread];
            _chunkGlobal = new string[ChunksPerThread];
            for (int i = 0; i < ChunksPerThread; i++)
            {
                string c = R(), col = R();
                _chunkRow[i] = R(); _chunkShared[i] = R(); _chunkGlobal[i] = RD();
                L($"add.u32 {c}, {_tid}, {I(i * Threads)};");
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
            // Q: row = warp * 16 + lo * 8 + row, col = hi * 8.
            L($"shl.b32 {tmpRow}, {_warp}, 4;");
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

            // Key tiles: ceil(sk / 64); causal: up to the tile of the block's last key, (min(m0 + 63, sq - 1) + offset).
            _tiles = R();
            L($"add.u32 {_tiles}, {_sk}, 63;");
            L($"shr.u32 {_tiles}, {_tiles}, 6;");
            if (_causal)
            {
                string lastRow = R(), lastKey = R(), causalTiles = R(), none = P();
                L($"add.u32 {lastRow}, {_m0}, 63;");
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
            _o = new string[DTiles, 4];
            for (int n = 0; n < DTiles; n++) for (int e = 0; e < 4; e++) _o[n, e] = F();
            _rowMax0 = F(); _rowMax1 = F(); _rowSum0 = F(); _rowSum1 = F();
            ResetState();

            // Q tile, then K/V tile 0, as the first cp.async group.
            IssueTile(_qGlobal, _qShared, _m0, _sq);
            IssueKeyValue("0", stage: 0);
            L("cp.async.commit_group;");

            _qf = new string[KSteps, 4];
            for (int s = 0; s < KSteps; s++) for (int e = 0; e < 4; e++) _qf[s, e] = R();
        }

        private void ResetState()
        {
            for (int n = 0; n < DTiles; n++) for (int e = 0; e < 4; e++) L($"mov.f32 {_o[n, e]}, 0f00000000;");
            L($"mov.f32 {_rowMax0}, 0fFF800000;");
            L($"mov.f32 {_rowMax1}, 0fFF800000;");
            L($"mov.f32 {_rowSum0}, 0f00000000;");
            L($"mov.f32 {_rowSum1}, 0f00000000;");
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

        // K and V tile `tile` (a register or immediate tile index) into stage `stage` (0 or 1, or a register offset).
        private void IssueKeyValue(string tile, int stage) => IssueKeyValue(tile, I(stage * TileBytes));

        private void IssueKeyValue(string tile, string stageOffset)
        {
            string rowStart = R(), kDst = R(), vDst = R();
            if (tile == "0") L($"mov.u32 {rowStart}, 0;");
            else L($"shl.b32 {rowStart}, {tile}, 6;");
            L($"add.u32 {kDst}, {_kShared}, {stageOffset};");
            L($"add.u32 {vDst}, {_vShared}, {stageOffset};");
            IssueTile(_kGlobal, kDst, rowStart, _sk);
            IssueTile(_vGlobal, vDst, rowStart, _sk);
        }

        private void MainLoop(string suffix, bool half)
        {
            string j = R();
            L($"mov.u32 {j}, 0;");
            Label("TILE" + suffix);
            // Prefetch tile j + 1 into the other stage (an empty group past the last tile keeps the wait count uniform).
            string next = R(), hasNext = P(), nextStage = R();
            L($"add.u32 {next}, {j}, 1;");
            L($"setp.lt.u32 {hasNext}, {next}, {_tiles};");
            L($"and.b32 {nextStage}, {next}, 1;");
            L($"mul.lo.u32 {nextStage}, {nextStage}, {I(TileBytes)};");
            L($"@!{hasNext} bra NOPREFETCH{suffix};");
            IssueKeyValue(next, nextStage);
            Label("NOPREFETCH" + suffix);
            L("cp.async.commit_group;");
            L("cp.async.wait_group 1;");
            L("bar.sync 0;");

            // Q fragments: loaded once, on the first tile (the Q tile arrived with K/V tile 0).
            string first = P();
            L($"setp.eq.u32 {first}, {j}, 0;");
            L($"@!{first} bra QREADY{suffix};");
            for (int s = 0; s < KSteps; s++)
                L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{_qf[s, 0]}, {_qf[s, 1]}, {_qf[s, 2]}, {_qf[s, 3]}}}, [{_qLane}+{I(s * 32)}];");
            Label("QREADY" + suffix);

            string stage = R(), kStage = R(), vStage = R();
            L($"and.b32 {stage}, {j}, 1;");
            L($"mul.lo.u32 {stage}, {stage}, {I(TileBytes)};");
            L($"add.u32 {kStage}, {_kShared}, {stage};");
            L($"add.u32 {kStage}, {kStage}, {_kLane};");
            L($"add.u32 {vStage}, {_vShared}, {stage};");
            L($"add.u32 {vStage}, {vStage}, {_vLane};");

            string[,] s4 = Scores(kStage, half);
            Softmax(s4, j, half, suffix);
            ProbabilitiesTimesValues(s4, vStage, half);

            L("bar.sync 0;");
            L($"add.u32 {j}, {j}, 1;");
            string more = P();
            L($"setp.lt.u32 {more}, {j}, {_tiles};");
            L($"@{more} bra TILE{suffix};");
        }
        // S (16 x 64 per warp) = Q K^T: eight n8 key tiles, four FP32 values each (c0, c1 row g; c2, c3 row g + 8).
        private string[,] Scores(string kStage, bool half)
        {
            var s = new string[8, 4];
            for (int n = 0; n < 8; n++) for (int e = 0; e < 4; e++) s[n, e] = F();
            var acc = new string[8, 2];
            if (half)
            {
                for (int n = 0; n < 8; n++) for (int e = 0; e < 2; e++)
                {
                    acc[n, e] = R();
                    L($"mov.b32 {acc[n, e]}, 0;");
                }
            }
            else
            {
                for (int n = 0; n < 8; n++) for (int e = 0; e < 4; e++) L($"mov.f32 {s[n, e]}, 0f00000000;");
            }
            for (int pair = 0; pair < 4; pair++)
            {
                for (int k = 0; k < KSteps; k++)
                {
                    string b0 = R(), b1 = R(), b2 = R(), b3 = R();
                    L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{b0}, {b1}, {b2}, {b3}}}, [{kStage}+{I(pair * 16 * RowBytes + k * 32)}];");
                    string a = $"{{{_qf[k, 0]}, {_qf[k, 1]}, {_qf[k, 2]}, {_qf[k, 3]}}}";
                    for (int sub = 0; sub < 2; sub++)
                    {
                        int n = pair * 2 + sub;
                        string b = sub == 0 ? $"{{{b0}, {b1}}}" : $"{{{b2}, {b3}}}";
                        if (half)
                        {
                            string c = $"{{{acc[n, 0]}, {acc[n, 1]}}}";
                            L($"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {c}, {a}, {b}, {c};");
                        }
                        else
                        {
                            string c = $"{{{s[n, 0]}, {s[n, 1]}, {s[n, 2]}, {s[n, 3]}}}";
                            L($"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {c}, {a}, {b}, {c};");
                        }
                    }
                }
            }
            if (half)
            {
                // Promote, and fold every score into a sum: an overflowed (infinite) or NaN score makes it non-finite.
                string sum = F();
                L($"mov.f32 {sum}, 0f00000000;");
                for (int n = 0; n < 8; n++)
                    for (int e = 0; e < 2; e++)
                    {
                        string h0 = H(), h1 = H();
                        L($"mov.b32 {{{h0}, {h1}}}, {acc[n, e]};");
                        L($"cvt.f32.f16 {s[n, 2 * e]}, {h0};");
                        L($"cvt.f32.f16 {s[n, 2 * e + 1]}, {h1};");
                        L($"add.f32 {sum}, {sum}, {s[n, 2 * e]};");
                        L($"add.f32 {sum}, {sum}, {s[n, 2 * e + 1]};");
                    }
                string abs = F(), nonFinite = P();
                L($"abs.f32 {abs}, {sum};");
                L($"setp.gtu.f32 {nonFinite}, {abs}, 0f7F7FFFFF;");
                L($"or.pred {_bad}, {_bad}, {nonFinite};");
            }
            return s;
        }

        private void Softmax(string[,] s, string j, bool half, string suffix)
        {
            for (int n = 0; n < 8; n++) for (int e = 0; e < 4; e++) L($"mul.f32 {s[n, e]}, {s[n, e]}, {_scaleLog2};");

            // Mask keys past sk and (causal) keys after a row's query position; a warp-uniform test skips full tiles.
            string n0 = R(), tileEnd = R(), needMask = P();
            L($"shl.b32 {n0}, {j}, 6;");
            L($"add.u32 {tileEnd}, {n0}, 64;");
            L($"setp.gt.u32 {needMask}, {tileEnd}, {_sk};");
            string row0 = R(), row1 = R();
            L($"shl.b32 {row0}, {_warp}, 4;");
            L($"add.u32 {row0}, {row0}, {_m0};");
            if (_causal)
            {
                string warpLimit = R(), lastKey = R(), causalMask = P();
                L($"add.s32 {warpLimit}, {row0}, {_offset};");
                L($"add.s32 {lastKey}, {n0}, 63;");
                L($"setp.gt.s32 {causalMask}, {lastKey}, {warpLimit};");
                L($"or.pred {needMask}, {needMask}, {causalMask};");
            }
            L($"add.u32 {row0}, {row0}, {_g};");
            L($"add.u32 {row1}, {row0}, 8;");
            L($"@!{needMask} bra UNMASKED{suffix};");
            string limit0 = R(), limit1 = R();
            if (_causal)
            {
                // Allowed keys: key < sk and key <= row + offset, i.e. key < min(sk, row + offset + 1) (signed).
                L($"add.s32 {limit0}, {row0}, {_offset};");
                L($"add.s32 {limit0}, {limit0}, 1;");
                L($"min.s32 {limit0}, {limit0}, {_sk};");
                L($"add.s32 {limit1}, {row1}, {_offset};");
                L($"add.s32 {limit1}, {limit1}, 1;");
                L($"min.s32 {limit1}, {limit1}, {_sk};");
            }
            else
            {
                L($"mov.u32 {limit0}, {_sk};");
                L($"mov.u32 {limit1}, {_sk};");
            }
            string keyBase = R();
            L($"shl.b32 {keyBase}, {_t}, 1;");
            L($"add.u32 {keyBase}, {keyBase}, {n0};");
            for (int n = 0; n < 8; n++)
                for (int e = 0; e < 2; e++)
                {
                    string key = R(), m0 = P(), m1 = P();
                    L($"add.s32 {key}, {keyBase}, {I(n * 8 + e)};");
                    L($"setp.ge.s32 {m0}, {key}, {limit0};");
                    L($"setp.ge.s32 {m1}, {key}, {limit1};");
                    L($"@{m0} mov.f32 {s[n, e]}, 0fFF800000;");
                    L($"@{m1} mov.f32 {s[n, 2 + e]}, 0fFF800000;");
                }
            Label("UNMASKED" + suffix);

            RowUpdate(s, rowHalf: 0, _rowMax0, _rowSum0);
            RowUpdate(s, rowHalf: 1, _rowMax1, _rowSum1);
        }

        // Online softmax for one of the thread's two rows (c0, c1 = row g; c2, c3 = row g + 8): new running max, rescale
        // of the sum and the output row, and the exponentials in place of the scores.
        private void RowUpdate(string[,] s, int rowHalf, string rowMax, string rowSum)
        {
            int e0 = rowHalf * 2;
            string tileMax = F();
            L($"max.f32 {tileMax}, {s[0, e0]}, {s[0, e0 + 1]};");
            for (int n = 1; n < 8; n++)
            {
                L($"max.f32 {tileMax}, {tileMax}, {s[n, e0]};");
                L($"max.f32 {tileMax}, {tileMax}, {s[n, e0 + 1]};");
            }
            QuadReduce("max.f32", tileMax);
            string newMax = F(), use = F(), allMasked = P(), alpha = F();
            L($"max.f32 {newMax}, {rowMax}, {tileMax};");
            // A row with every key masked so far keeps max -inf; exponentiate against 0 so it yields zeros, not NaN.
            L($"setp.eq.f32 {allMasked}, {newMax}, 0fFF800000;");
            L($"selp.f32 {use}, 0f00000000, {newMax}, {allMasked};");
            L($"sub.f32 {alpha}, {rowMax}, {use};");
            L($"ex2.approx.ftz.f32 {alpha}, {alpha};");
            L($"mov.f32 {rowMax}, {newMax};");
            string tileSum = F();
            L($"mov.f32 {tileSum}, 0f00000000;");
            for (int n = 0; n < 8; n++)
                for (int e = e0; e < e0 + 2; e++)
                {
                    L($"sub.f32 {s[n, e]}, {s[n, e]}, {use};");
                    L($"ex2.approx.ftz.f32 {s[n, e]}, {s[n, e]};");
                    L($"add.f32 {tileSum}, {tileSum}, {s[n, e]};");
                }
            L($"fma.rn.f32 {rowSum}, {rowSum}, {alpha}, {tileSum};");
            for (int n = 0; n < DTiles; n++)
            {
                L($"mul.f32 {_o[n, e0]}, {_o[n, e0]}, {alpha};");
                L($"mul.f32 {_o[n, e0 + 1]}, {_o[n, e0 + 1]}, {alpha};");
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
        // O += P V: P (the exponentials, packed to FP16 A fragments: the score layout is the A layout) times the V tile.
        private void ProbabilitiesTimesValues(string[,] s, string vStage, bool half)
        {
            var pa = new string[4, 4];
            for (int k = 0; k < 4; k++)
            {
                pa[k, 0] = Pack(s[2 * k, 0], s[2 * k, 1]);
                pa[k, 1] = Pack(s[2 * k, 2], s[2 * k, 3]);
                pa[k, 2] = Pack(s[2 * k + 1, 0], s[2 * k + 1, 1]);
                pa[k, 3] = Pack(s[2 * k + 1, 2], s[2 * k + 1, 3]);
            }
            var acc = new string[DTiles, 2];
            if (half)
                for (int n = 0; n < DTiles; n++) for (int e = 0; e < 2; e++)
                {
                    acc[n, e] = R();
                    L($"mov.b32 {acc[n, e]}, 0;");
                }
            for (int k = 0; k < 4; k++)
            {
                string a = $"{{{pa[k, 0]}, {pa[k, 1]}, {pa[k, 2]}, {pa[k, 3]}}}";
                for (int pair = 0; pair < DTiles / 2; pair++)
                {
                    string b0 = R(), b1 = R(), b2 = R(), b3 = R();
                    L($"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{b0}, {b1}, {b2}, {b3}}}, [{vStage}+{I(k * 16 * RowBytes + pair * 32)}];");
                    for (int sub = 0; sub < 2; sub++)
                    {
                        int n = pair * 2 + sub;
                        string b = sub == 0 ? $"{{{b0}, {b1}}}" : $"{{{b2}, {b3}}}";
                        if (half)
                        {
                            string c = $"{{{acc[n, 0]}, {acc[n, 1]}}}";
                            L($"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {c}, {a}, {b}, {c};");
                        }
                        else
                        {
                            string c = $"{{{_o[n, 0]}, {_o[n, 1]}, {_o[n, 2]}, {_o[n, 3]}}}";
                            L($"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {c}, {a}, {b}, {c};");
                        }
                    }
                }
            }
            if (half)
                for (int n = 0; n < DTiles; n++)
                    for (int e = 0; e < 2; e++)
                    {
                        string h0 = H(), h1 = H(), x = F(), y = F();
                        L($"mov.b32 {{{h0}, {h1}}}, {acc[n, e]};");
                        L($"cvt.f32.f16 {x}, {h0};");
                        L($"cvt.f32.f16 {y}, {h1};");
                        L($"add.f32 {_o[n, 2 * e]}, {_o[n, 2 * e]}, {x};");
                        L($"add.f32 {_o[n, 2 * e + 1]}, {_o[n, 2 * e + 1]}, {y};");
                    }
        }

        private string Pack(string low, string high)
        {
            string h0 = H(), h1 = H(), packed = R();
            L($"cvt.rn.f16.f32 {h0}, {low};");
            L($"cvt.rn.f16.f32 {h1}, {high};");
            L($"mov.b32 {packed}, {{{h0}, {h1}}};");
            return packed;
        }

        // Any of this thread's output accumulators or row sums non-finite (an FP16 P V partial overflowed).
        private string OutputsNonFinite()
        {
            string sum = F();
            L($"add.f32 {sum}, {_rowSum0}, {_rowSum1};");
            for (int n = 0; n < DTiles; n++) for (int e = 0; e < 4; e++) L($"add.f32 {sum}, {sum}, {_o[n, e]};");
            string abs = F(), nonFinite = P();
            L($"abs.f32 {abs}, {sum};");
            L($"setp.gtu.f32 {nonFinite}, {abs}, 0f7F7FFFFF;");
            return nonFinite;
        }

        private void Epilogue()
        {
            QuadReduce("add.f32", _rowSum0);
            QuadReduce("add.f32", _rowSum1);
            string row0 = R(), row1 = R();
            L($"shl.b32 {row0}, {_warp}, 4;");
            L($"add.u32 {row0}, {row0}, {_m0};");
            L($"add.u32 {row0}, {row0}, {_g};");
            L($"add.u32 {row1}, {row0}, 8;");
            StoreRow(row0, 0, _rowSum0, _rowMax0);
            StoreRow(row1, 2, _rowSum1, _rowMax1);
        }

        private void StoreRow(string row, int e0, string rowSum, string rowMax)
        {
            string skip = P(), done = "ROWDONE" + I(e0);
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
                L($"mul.f32 {x}, {_o[n, e0]}, {inv};");
                L($"mul.f32 {y}, {_o[n, e0 + 1]}, {inv};");
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