// Copyright (c) AiDotNet. All rights reserved.
// The stride-2 input gradients of CUDA/Kernels/CudaConvDgradS2Kernels.cs (#1476, #1488, #1494) -- the 3 x 3 four-phase
// kernel, the 1 x 1 kernel and the projection-shortcut pair -- written once in WinogradF23KernelTemplate's body
// conventions and emitted as OpenCL C, HIP C++, Vulkan GLSL and Metal Shading Language (#1479, #1480, #1481, #1482).
// WGSL keeps its own copy (WebGpu/WebGpuConvDgradS2Kernels.cs, #1483).
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The input gradient of a 3 x 3, stride-2, padding-1 convolution whose input is exactly twice its output, all four
/// sub-pixel phases in one kernel (see <c>CudaConvDgradS2Kernels</c> for the phase algebra); the 1 x 1, stride-2,
/// padding-0 input gradient written straight into dX with its three zero phases; and the pair of both reading one input
/// (the compiled plan's projection shortcut), whose 1 x 1 term lands in phase (0, 0).
/// </summary>
/// <remarks>
/// <para>
/// A group of 256 owns CUDA's block: 32 dY cells (whole rows: the output width divides 32) x 32 input channels. Thread t
/// owns the four cells 4 (t &amp; 7) .. + 3 of one dY row and channel t &gt;&gt; 3, so its 2 x 2 dX blocks are 16
/// accumulators; per output channel it reads its cells' two five-wide dY rows and its channel's nine taps from shared
/// memory for 36 FMAs. CUDA's 64-thread blocks give a thread four channels and add in-block output-channel groups; here
/// the group is fixed at 256, so the channels go to separate threads and no group sum is needed.
/// </para>
/// <para>
/// Shared memory is one staging buffer (barrier, compute, barrier per eight output channels), 11.5 KB for the 3 x 3
/// kernel and 14.5 KB for the pair: under WebGPU's guaranteed 16 KB. The output-channel reduction may split across
/// groups into partial slices of dX's layout, which <see cref="Reduce"/> sums in split order, so every split count is
/// deterministic. The grid is 1-D: group grp is (split bz, channel tile by, cell tile bx) with
/// grp = (bz * gy + by) * gx + bx.
/// </para>
/// </remarks>
internal static class ConvDgradS2KernelTemplate
{
    internal const string K3 = "conv2d_dgrad_s2p_k3";
    internal const string K3Pair = "conv2d_dgrad_s2p_k3sc";
    internal const string K1 = "conv2d_dgrad_s2p_k1";
    internal const string Reduce = "conv2d_dgrad_s2p_reduce";

    /// <summary>Every kernel's group size: the thread roles assume exactly 256.</summary>
    internal const int GroupSize = 256;

    /// <summary>Output channels staged per step; split slices are whole steps.</summary>
    internal const int CoTile = 8;

    /// <summary>dY cells per group (whole rows of the 3 x 3 kernel's dY).</summary>
    internal const int CellTile = 32;

    /// <summary>Input-gradient channels per group.</summary>
    internal const int ChannelTile = 32;

    // Shared-memory layout, in floats: a staged chunk's dY rows (eight output channels x 72: (32 / OW + 1) rows of
    // OW + 4, the last four a zero pad so the fifth cell of a row's last quad reads zero), its weights [8][9 taps][32
    // channels], and the pair's shortcut dY and weights [8][32].
    private const int YChunk = 8 * 72, WChunk = 8 * 9 * 32, W2Chunk = 8 * 32;

    private static readonly string[] Ints3 = { "N", "C", "K", "OH", "OW", "coPerSplit" };

    private static readonly string[] IntsPair = { "N", "C", "K", "OH", "OW", "coPerSplit", "K2", "co2PerSplit" };

    // The group's tile and split from grp, and the reduction range of this split. gx counts cell tiles of RB dY rows.
    private const string K3Prologue = @"
    int tid = lid;
    int RB = 32 / OW, YS = OW + 4;
    int rows = N * OH, H = 2 * OH, W = 2 * OW;
    int gx = (rows + RB - 1) / RB, gy = (C + 31) / 32;
    int bz = grp / (gx * gy);
    int rem = grp - bz * (gx * gy);
    int by = rem / gx, bx = rem - by * gx;
    int R0 = bx * RB, c0 = by * 32;
    int coBegin = bz * coPerSplit, coEnd = min(K, coBegin + coPerSplit);
    int dxBase = bz * N * C * H * W;
    int cg = tid & 7, ch = tid >> 3;
    int l = (cg * 4) / OW, b0 = cg * 4 - l * OW;
    int R = R0 + l;
    int aRow = R % OH;
    bool lastRow = aRow == OH - 1;
    int c = c0 + ch;
    float acc[16];
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    int steps = coBegin < coEnd ? (coEnd - coBegin + 7) / 8 : 0;
";

    // A chunk's dY into sm[base ..]: slot s is (channel kl, staged row r, column col); rows past the image's last dY row,
    // the pad columns and channels past the split's end read zero.
    private static string StageY(string buffer, string channels, string first, string end, int smBase) => $@"
        for (int s = tid; s < {YChunk}; s += 256)
        {{
            int kl = s / 72, o = s - kl * 72, r = o / YS, col = o - r * YS;
            int rr = R0 + r, co = {first} + kl;
            float v = 0.0f;
            if (r <= RB && col < OW && rr < rows && co < {end})
            {{
                int nn = rr / OH, a = rr - nn * OH;
                v = {buffer}[((nn * {channels} + co) * OH + a) * OW + col];
            }}
            sm[{smBase} + s] = v;
        }}
";

    // The nine taps of a chunk's eight output channels x 32 input channels: read in the weights' own order (a channel's
    // nine taps are contiguous), stored [kl][tap][channel] so a thread's nine reads are one per tap row.
    private const string StageW = @"
        for (int s = tid; s < 2304; s += 256)
        {
            int kl = s / 288, o = s - kl * 288, cl = o / 9, tap = o - cl * 9;
            int co = co0 + kl;
            float v = 0.0f;
            if (co < coEnd && c0 + cl < C) v = wt[(co * C + c0 + cl) * 9 + tap];
            sm[576 + kl * 288 + tap * 32 + cl] = v;
        }
";

    // The 3 x 3 kernel's arithmetic, CUDA's per (cell j, output channel): phase p of dX block j, in CUDA's fma order.
    private const string ComputeK3 = @"
        for (int k = 0; k < 8; k++)
        {
            int yb = k * 72 + l * YS + b0;
            float ya[5];
            float yb1[5];
            for (int j = 0; j < 5; j++)
            {
                ya[j] = sm[yb + j];
                yb1[j] = lastRow ? 0.0f : sm[yb + YS + j];
            }
            float wq[9];
            for (int t = 0; t < 9; t++) wq[t] = sm[576 + k * 288 + t * 32 + ch];
            for (int j = 0; j < 4; j++)
            {
                acc[j * 4] = FMA(ya[j], wq[4], acc[j * 4]);
                acc[j * 4 + 1] = FMA(ya[j + 1], wq[3], FMA(ya[j], wq[5], acc[j * 4 + 1]));
                acc[j * 4 + 2] = FMA(yb1[j], wq[1], FMA(ya[j], wq[7], acc[j * 4 + 2]));
                acc[j * 4 + 3] = FMA(yb1[j + 1], wq[0], FMA(yb1[j], wq[2], FMA(ya[j + 1], wq[6], FMA(ya[j], wq[8], acc[j * 4 + 3]))));
            }
        }
";

    // The four cells' 2 x 2 dX blocks of this split's slice.
    private const string StoreK3 = @"
    if (R < rows && c < C)
    {
        int nn = R / OH;
        int o0 = dxBase + ((nn * C + c) * H + 2 * aRow) * W + 2 * b0;
        for (int j = 0; j < 4; j++)
        {
            dx[o0 + 2 * j] = acc[j * 4];
            dx[o0 + 2 * j + 1] = acc[j * 4 + 1];
            dx[o0 + W + 2 * j] = acc[j * 4 + 2];
            dx[o0 + W + 2 * j + 1] = acc[j * 4 + 3];
        }
    }
";

    private static string K3Body() => K3Prologue + @"
    for (int st = 0; st < steps; st++)
    {
        int co0 = coBegin + st * 8;
" + StageY("dy", "K", "co0", "coEnd", 0) + StageW + @"
        BARRIER();
" + ComputeK3 + @"
        BARRIER();
    }
" + StoreK3;

    // The pair: each step also stages a chunk of the shortcut's reduction (co2PerSplit per split, its own range) and adds
    // its 1 x 1 term to phase (0, 0); the loop runs max(steps of either) steps.
    private static string K3PairBody() => K3Prologue + @"
    int co2Begin = bz * co2PerSplit, co2End = min(K2, co2Begin + co2PerSplit);
    int steps2 = co2Begin < co2End ? (co2End - co2Begin + 7) / 8 : 0;
    int stepsAll = max(steps, steps2);
    for (int st = 0; st < stepsAll; st++)
    {
        int co0 = coBegin + st * 8, co20 = co2Begin + st * 8;
" + StageY("dy", "K", "co0", "coEnd", 0) + StageW + StageY("dy2", "K2", "co20", "co2End", YChunk + WChunk) + $@"
        for (int s = tid; s < {W2Chunk}; s += 256)
        {{
            int kl = s >> 5, cl = s & 31;
            int co = co20 + kl;
            sm[{YChunk + WChunk + YChunk} + s] = (co < co2End && c0 + cl < C) ? ws[co * C + c0 + cl] : 0.0f;
        }}
        BARRIER();
" + ComputeK3 + $@"
        for (int k = 0; k < 8; k++)
        {{
            int yb = {YChunk + WChunk} + k * 72 + l * YS + b0;
            float w2 = sm[{YChunk + WChunk + YChunk} + k * 32 + ch];
            for (int j = 0; j < 4; j++) acc[j * 4] = FMA(sm[yb + j], w2, acc[j * 4]);
        }}
        BARRIER();
    }}
" + StoreK3;

    // The 1 x 1 kernel: a GEMM over the group's 32 consecutive dY cells (the output width is a multiple of four, so a
    // thread's four cells share a row) x 32 channels, written as phase (0, 0) with the other three phases zero.
    private const string K1Body = @"
    int tid = lid;
    int HW = OH * OW, M = N * HW, H = 2 * OH, W = 2 * OW;
    int gx = (M + 31) / 32, gy = (C + 31) / 32;
    int bz = grp / (gx * gy);
    int rem = grp - bz * (gx * gy);
    int by = rem / gx, bx = rem - by * gx;
    int m0 = bx * 32, c0 = by * 32;
    int coBegin = bz * coPerSplit, coEnd = min(K, coBegin + coPerSplit);
    int dxBase = bz * N * C * H * W;
    int cg = tid & 7, ch = tid >> 3;
    int c = c0 + ch;
    float acc[4];
    for (int i = 0; i < 4; i++) acc[i] = 0.0f;
    int steps = coBegin < coEnd ? (coEnd - coBegin + 7) / 8 : 0;
    for (int st = 0; st < steps; st++)
    {
        int co0 = coBegin + st * 8;
        int kl = tid >> 5, ci = tid & 31;
        int co = co0 + kl, m = m0 + ci;
        float v = 0.0f;
        if (m < M && co < coEnd)
        {
            int nn = m / HW;
            v = dy[(nn * K + co) * HW + (m - nn * HW)];
        }
        sm[tid] = v;
        sm[256 + tid] = (co < coEnd && c0 + ci < C) ? wt[co * C + c0 + ci] : 0.0f;
        BARRIER();
        for (int k = 0; k < 8; k++)
        {
            float w1 = sm[256 + k * 32 + ch];
            for (int j = 0; j < 4; j++) acc[j] = FMA(sm[k * 32 + cg * 4 + j], w1, acc[j]);
        }
        BARRIER();
    }
    int m4 = m0 + cg * 4;
    if (m4 < M && c < C)
    {
        int nn = m4 / HW, rm = m4 - nn * HW, a = rm / OW, b0 = rm - a * OW;
        int o0 = dxBase + ((nn * C + c) * H + 2 * a) * W + 2 * b0;
        for (int j = 0; j < 4; j++)
        {
            dx[o0 + 2 * j] = acc[j];
            dx[o0 + 2 * j + 1] = 0.0f;
            dx[o0 + W + 2 * j] = 0.0f;
            dx[o0 + W + 2 * j + 1] = 0.0f;
        }
    }
";

    // dX = the split slices summed in split order.
    private const string ReduceBody = @"
    if (gid < total)
    {
        float s = part[gid];
        for (int i = 1; i < splits; i++) s += part[i * total + gid];
        dx[gid] = s;
    }
";

    /// <summary>Every kernel, in this template's conventions (dX is binding 1, written; every other buffer is read).</summary>
    internal static readonly WinogradF23KernelTemplate.Spec[] Kernels =
    {
        new(K3, new[] { "dy", "dx", "wt" }, Ints3, K3Body(), YChunk + WChunk),
        new(K3Pair, new[] { "dy", "dx", "wt", "dy2", "ws" }, IntsPair, K3PairBody(), YChunk + WChunk + YChunk + W2Chunk),
        new(K1, new[] { "dy", "dx", "wt" }, Ints3, K1Body, 2 * W2Chunk),
        new(Reduce, new[] { "part", "dx" }, new[] { "total", "splits" }, ReduceBody),
    };

    internal static string[] GetKernelNames() => new[] { K3, K3Pair, K1, Reduce };

    /// <summary>Every kernel in one OpenCL C, HIP C++ or Metal Shading Language module.</summary>
    internal static string GetModuleSource(WinogradKernelDialect dialect)
        => WinogradF23KernelTemplate.EmitModule(Kernels, dialect);

    /// <summary>One kernel as a Vulkan compute shader (bindings in buffer order, the ints as push constants).</summary>
    internal static string GetGlslSource(string name)
    {
        foreach (var spec in Kernels)
            if (spec.Name == name) return WinogradF23KernelTemplate.EmitGlsl(spec);
        throw new KeyNotFoundException(name);
    }
}
