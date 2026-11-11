// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The jagged-rows grouped GEMM of IGroupedGemmBackend (torch grouped_mm, 2-D a x 3-D b), written once in
// WinogradF23KernelTemplate's body conventions and emitted as OpenCL C, HIP C++, Vulkan GLSL and Metal Shading Language.
// WGSL keeps its own copy (WebGpu/WebGpuGroupedGemmKernels.cs).
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Two kernels over 256-wide groups, each owning a 64 x 64 output tile (4 x 4 values per thread, operands staged in
/// shared memory 16 deep along the reduction):
/// <list type="bullet">
/// <item><see cref="Forward"/>: <c>y[r, :] = a[r, :] op(b[g(r)]) (+ bias)</c>, where row r belongs to group g when
/// <c>ends[g-1] &lt;= r &lt; ends[g]</c>. A row tile that straddles group boundaries runs once per group it meets, its
/// other rows masked to zero, so every group's rows use that group's matrix and nothing is padded to the largest group.
/// Rows past the last offset are zero. <c>transB</c> reads b as [G, N, K].</item>
/// <item><see cref="WeightGrad"/>: <c>w[g] = a_g^T c_g</c> ([G, K, N]), the rows of group g reduced in chunks of 16;
/// an empty group's tile is written as zero.</item>
/// </list>
/// The offsets ride in a float buffer (exact below 2^24 rows; <see cref="PortableGroupedGemm"/> refuses larger) and the
/// group's index range is uniform across the work-group, so every barrier sits in uniform control flow.
/// <c>tileBase</c> offsets the group index, so a grid larger than one dispatch's 65535 groups runs as several launches.
/// </summary>
internal static class GroupedGemmKernelTemplate
{
    /// <summary>Threads per group; the tile roles assume exactly this.</summary>
    internal const int GroupSize = 256;

    /// <summary>Output tile edge (rows and columns) per group.</summary>
    internal const int Tile = 64;

    internal const string Forward = "grouped_rows_gemm";
    internal const string WeightGrad = "grouped_rows_wgrad";

    private static readonly string[] ForwardInts = { "M", "N", "K", "G", "tilesN", "transB", "biasMode", "tileBase" };
    private static readonly string[] WeightGradInts = { "M", "K", "N", "G", "tilesK", "tilesN", "tileBase" };

    // Shared memory: the a tile [64 rows][16] at 0, the b tile [16][64 cols] at 1024.
    private const string ForwardBody = @"
    int tid = lid;
    int t = grp + tileBase;
    int rt = t / tilesN;
    int ct = t - rt * tilesN;
    int row0 = rt * 64, col0 = ct * 64;
    int rowHi = row0 + 64 < M ? row0 + 64 : M;
    int tr = tid / 16, tc = tid - (tid / 16) * 16;
    float acc[16];
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    int lo = 0, hi = G;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        if (TOINT(ends[mid]) > row0) hi = mid; else lo = mid + 1;
    }
    int g = lo;
    while (g < G)
    {
        int gs = g == 0 ? 0 : TOINT(ends[g - 1]);
        if (gs >= rowHi) break;
        int ge = TOINT(ends[g]);
        int r0 = gs > row0 ? gs : row0;
        int r1 = ge < rowHi ? ge : rowHi;
        if (r1 > r0)
        {
            for (int k0 = 0; k0 < K; k0 += 16)
            {
                for (int q = 0; q < 4; q++)
                {
                    int idx = tid + q * 256;
                    int ar = idx / 16;
                    int ak = idx - ar * 16;
                    int arow = row0 + ar, acol = k0 + ak;
                    sm[idx] = (arow >= r0 && arow < r1 && acol < K) ? xa[arow * K + acol] : 0.0f;
                    int bc = 0, bk = 0;
                    if (transB != 0) { bc = idx / 16; bk = idx - bc * 16; }
                    else { bk = idx / 64; bc = idx - bk * 64; }
                    int bcol = col0 + bc, brow = k0 + bk;
                    float bv = 0.0f;
                    if (bcol < N && brow < K)
                        bv = transB != 0 ? wb[(g * N + bcol) * K + brow] : wb[(g * K + brow) * N + bcol];
                    sm[1024 + bk * 64 + bc] = bv;
                }
                BARRIER();
                for (int kk = 0; kk < 16; kk++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        float av = sm[(tr * 4 + i) * 16 + kk];
                        for (int j = 0; j < 4; j++)
                            acc[i * 4 + j] = FMA(av, sm[1024 + kk * 64 + tc * 4 + j], acc[i * 4 + j]);
                    }
                }
                BARRIER();
            }
        }
        g++;
    }
    int last = G > 0 ? TOINT(ends[G - 1]) : 0;
    for (int i = 0; i < 4; i++)
    {
        int row = row0 + tr * 4 + i;
        if (row < M)
        {
            int rg = 0;
            if (biasMode == 2 && row < last)
            {
                int l2 = 0, h2 = G;
                while (l2 < h2)
                {
                    int m2 = (l2 + h2) / 2;
                    if (TOINT(ends[m2]) > row) h2 = m2; else l2 = m2 + 1;
                }
                rg = l2;
            }
            for (int j = 0; j < 4; j++)
            {
                int col = col0 + tc * 4 + j;
                if (col < N)
                {
                    float v = acc[i * 4 + j];
                    if (biasMode == 1 && row < last) v += bias[col];
                    if (biasMode == 2 && row < last) v += bias[rg * N + col];
                    yc[row * N + col] = v;
                }
            }
        }
    }
";

    // Shared memory: the a rows [16][64 k] at 0, the c rows [16][64 n] at 1024.
    private const string WeightGradBody = @"
    int tid = lid;
    int t = grp + tileBase;
    int per = tilesK * tilesN;
    int g = t / per;
    int rem = t - g * per;
    int kt = rem / tilesN;
    int nt = rem - kt * tilesN;
    int k0 = kt * 64, n0 = nt * 64;
    int gs = g == 0 ? 0 : TOINT(ends[g - 1]);
    int ge = TOINT(ends[g]);
    int tr = tid / 16, tc = tid - (tid / 16) * 16;
    float acc[16];
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    for (int r0 = gs; r0 < ge; r0 += 16)
    {
        for (int q = 0; q < 4; q++)
        {
            int idx = tid + q * 256;
            int rr = idx / 64;
            int cc = idx - rr * 64;
            int row = r0 + rr;
            sm[idx] = (row < ge && k0 + cc < K) ? xa[row * K + k0 + cc] : 0.0f;
            sm[1024 + idx] = (row < ge && n0 + cc < N) ? cg[row * N + n0 + cc] : 0.0f;
        }
        BARRIER();
        for (int rr = 0; rr < 16; rr++)
        {
            for (int i = 0; i < 4; i++)
            {
                float av = sm[rr * 64 + tr * 4 + i];
                for (int j = 0; j < 4; j++)
                    acc[i * 4 + j] = FMA(av, sm[1024 + rr * 64 + tc * 4 + j], acc[i * 4 + j]);
            }
        }
        BARRIER();
    }
    for (int i = 0; i < 4; i++)
    {
        int k = k0 + tr * 4 + i;
        for (int j = 0; j < 4; j++)
        {
            int n = n0 + tc * 4 + j;
            if (k < K && n < N) wout[(g * K + k) * N + n] = acc[i * 4 + j];
        }
    }
";

    /// <summary>
    /// Forward buffers: a [M, K], y [M, N] (written), b ([G, K, N], or [G, N, K] with transB), ends [G] (floats), bias
    /// ([N] with biasMode 1, [G, N] with biasMode 2; any buffer with 0). Weight-gradient buffers: a [M, K], w [G, K, N]
    /// (written), c [M, N], ends [G].
    /// </summary>
    internal static readonly WinogradF23KernelTemplate.Spec[] Kernels =
    {
        new(Forward, new[] { "xa", "yc", "wb", "ends", "bias" }, ForwardInts, ForwardBody, 2048, new[] { 1 }),
        new(WeightGrad, new[] { "xa", "wout", "cg", "ends" }, WeightGradInts, WeightGradBody, 2048, new[] { 1 }),
    };

    /// <summary>The kernel names: the forward and the weight gradient.</summary>
    internal static string[] GetKernelNames() => new[] { Forward, WeightGrad };

    internal static string GetModuleSource(WinogradKernelDialect dialect) => WinogradF23KernelTemplate.EmitModule(Kernels, dialect);

    /// <summary>The standalone GLSL compute shader for one kernel (Vulkan compiles one pipeline per kernel).</summary>
    internal static string GetGlslSource(string name)
    {
        foreach (var spec in Kernels)
            if (spec.Name == name) return WinogradF23KernelTemplate.EmitGlsl(spec);
        throw new System.ArgumentException($"No grouped GEMM kernel named '{name}'.", nameof(name));
    }
}
