#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IFusedAttentionBackend"/> on every non-CUDA backend (#1179) against a double-precision reference of the
/// CUDA kernels' contract: scores (q * scale) . k, soft-cap cap * tanh(s / cap), causal j &lt;= i + offset, an int32
/// mask through broadcast strides, lse = -inf (output and gradients 0) for a row with no allowed key. q is stored
/// [B, S, H, D] and k, v, o [B, H, S, D], so the strides are exercised.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class FusedAttentionBackendParityTests
{
    private readonly record struct Case(int B, int H, int Sq, int Sk, int D, bool Causal, int Offset, bool Mask, float Softcap);

    private static readonly Case[] Cases =
    {
        new(2, 3, 5, 7, 16, false, 0, false, 0f),
        new(1, 2, 6, 6, 5, true, 0, false, 0f),
        new(2, 1, 4, 9, 8, true, 5, true, 0f),
        new(1, 2, 7, 5, 3, false, 0, true, 2.5f),   // the mask leaves query row 0 with no allowed key
    };

    private static float[] Random(int count, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }

    private static int[] MaskValues(Case c)
    {
        var mask = new int[c.Sq * c.Sk];
        for (int i = 0; i < c.Sq; i++)
            for (int j = 0; j < c.Sk; j++)
                mask[i * c.Sk + j] = i == 0 ? 0 : ((i + 2 * j) % 3 != 0 ? 1 : 0);
        return mask;
    }

    private static void Check(IDirectGpuBackend backend, IFusedAttentionBackend? fusedOverride = null)
    {
        var fused = fusedOverride ?? Assert.IsAssignableFrom<IFusedAttentionBackend>(backend);
        foreach (var c in Cases)
        {
            Assert.True(fused.SupportsFusedAttention(c.D));
            int qCount = c.B * c.Sq * c.H * c.D, kCount = c.B * c.H * c.Sk * c.D, oCount = c.B * c.H * c.Sq * c.D;
            float[] q = Random(qCount, 11 + c.D), k = Random(kCount, 12 + c.D), v = Random(kCount, 13 + c.D);
            float[] dO = Random(oCount, 14 + c.D);
            int[] mask = MaskValues(c);
            float scale = 1f / MathF.Sqrt(c.D);
            int QIndex(int b, int h, int i, int d) => ((b * c.Sq + i) * c.H + h) * c.D + d;
            int KIndex(int b, int h, int j, int d) => ((b * c.H + h) * c.Sk + j) * c.D + d;
            int OIndex(int b, int h, int i, int d) => ((b * c.H + h) * c.Sq + i) * c.D + d;
            bool Allowed(int i, int j) => !(c.Causal && j > i + c.Offset) && !(c.Mask && mask[i * c.Sk + j] == 0);

            var o = new double[oCount];
            var dq = new double[qCount];
            var dk = new double[kCount];
            var dv = new double[kCount];
            var lse = new double[c.B * c.H * c.Sq];
            for (int b = 0; b < c.B; b++)
                for (int h = 0; h < c.H; h++)
                    for (int i = 0; i < c.Sq; i++)
                    {
                        var s = new double[c.Sk];
                        var slope = new double[c.Sk];
                        double max = double.NegativeInfinity;
                        for (int j = 0; j < c.Sk; j++)
                        {
                            double raw = 0;
                            for (int d = 0; d < c.D; d++) raw += (double)q[QIndex(b, h, i, d)] * scale * k[KIndex(b, h, j, d)];
                            slope[j] = 1;
                            if (c.Softcap > 0) { double t = Math.Tanh(raw / c.Softcap); slope[j] = 1 - t * t; raw = c.Softcap * t; }
                            s[j] = raw;
                            if (Allowed(i, j)) max = Math.Max(max, raw);
                        }
                        int row = (b * c.H + h) * c.Sq + i;
                        if (double.IsNegativeInfinity(max)) { lse[row] = double.NegativeInfinity; continue; }
                        double l = 0;
                        for (int j = 0; j < c.Sk; j++) if (Allowed(i, j)) l += Math.Exp(s[j] - max);
                        lse[row] = max + Math.Log(l);
                        var p = new double[c.Sk];
                        for (int j = 0; j < c.Sk; j++) p[j] = Allowed(i, j) ? Math.Exp(s[j] - lse[row]) : 0;
                        for (int d = 0; d < c.D; d++)
                        {
                            double acc = 0;
                            for (int j = 0; j < c.Sk; j++) acc += p[j] * v[KIndex(b, h, j, d)];
                            o[OIndex(b, h, i, d)] = acc;
                        }
                        double delta = 0;
                        for (int d = 0; d < c.D; d++) delta += dO[OIndex(b, h, i, d)] * o[OIndex(b, h, i, d)];
                        for (int j = 0; j < c.Sk; j++)
                        {
                            if (p[j] == 0) continue;
                            double dp = 0;
                            for (int d = 0; d < c.D; d++) dp += dO[OIndex(b, h, i, d)] * v[KIndex(b, h, j, d)];
                            double dRaw = p[j] * (dp - delta) * slope[j];
                            for (int d = 0; d < c.D; d++)
                            {
                                dq[QIndex(b, h, i, d)] += dRaw * k[KIndex(b, h, j, d)] * scale;
                                dk[KIndex(b, h, j, d)] += dRaw * q[QIndex(b, h, i, d)] * scale;
                                dv[KIndex(b, h, j, d)] += p[j] * dO[OIndex(b, h, i, d)];
                            }
                        }
                    }

            var prm = new FusedAttentionParams
            {
                B = c.B, H = c.H, Sq = c.Sq, Sk = c.Sk,
                Qb = (long)c.Sq * c.H * c.D, Qh = c.D, Qr = (long)c.H * c.D,
                Kb = (long)c.H * c.Sk * c.D, Kh = (long)c.Sk * c.D, Kr = c.D,
                Vb = (long)c.H * c.Sk * c.D, Vh = (long)c.Sk * c.D, Vr = c.D,
                Ob = (long)c.H * c.Sq * c.D, Oh = (long)c.Sq * c.D, Or = c.D,
                Scale = scale, Softcap = c.Softcap, Causal = c.Causal ? 1 : 0, CausalOffset = c.Offset,
                HasMask = c.Mask ? 1 : 0, Mb = 0, Mh = 0, Mi = c.Sk, Mj = 1,
            };
            var maskWords = new float[mask.Length];
            for (int i = 0; i < mask.Length; i++) maskWords[i] = BitConverter.Int32BitsToSingle(mask[i]);
            using var qBuf = backend.AllocateBuffer(q);
            using var kBuf = backend.AllocateBuffer(k);
            using var vBuf = backend.AllocateBuffer(v);
            using var dOBuf = backend.AllocateBuffer(dO);
            using var maskBuf = backend.AllocateBuffer(maskWords);
            using var oBuf = backend.AllocateBuffer(Filled(oCount));
            using var lseBuf = backend.AllocateBuffer(Filled(lse.Length));
            using var dqBuf = backend.AllocateBuffer(Filled(qCount));
            using var dkBuf = backend.AllocateBuffer(Filled(kCount));
            using var dvBuf = backend.AllocateBuffer(Filled(kCount));
            IGpuBuffer? maskArg = c.Mask ? maskBuf : null;
            fused.FusedAttentionForward(qBuf, kBuf, vBuf, oBuf, lseBuf, maskArg, prm, c.D);
            fused.FusedAttentionBackward(qBuf, kBuf, vBuf, oBuf, dOBuf, lseBuf, maskArg, dqBuf, dkBuf, dvBuf, prm, c.D);

            float[] gotLse = backend.DownloadBuffer(lseBuf);
            for (int r = 0; r < lse.Length; r++)
            {
                if (double.IsNegativeInfinity(lse[r])) Assert.True(float.IsNegativeInfinity(gotLse[r]), $"{c} lse[{r}]: {gotLse[r]}");
                else AssertClose(lse[r], gotLse[r], $"{c} lse[{r}]");
            }
            Compare(o, backend.DownloadBuffer(oBuf), $"{c} o");
            Compare(dq, backend.DownloadBuffer(dqBuf), $"{c} dq");
            Compare(dk, backend.DownloadBuffer(dkBuf), $"{c} dk");
            Compare(dv, backend.DownloadBuffer(dvBuf), $"{c} dv");
        }
    }

    // A pattern in every output buffer, so a kernel that skips an element (instead of writing 0) shows.
    private static float[] Filled(int count)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = 7.25f;
        return values;
    }

    private static void AssertClose(double expected, float actual, string what)
        => Assert.True(Math.Abs(expected - actual) <= 2e-5 * (1 + Math.Abs(expected)), $"{what}: expected {expected}, got {actual}");

    private static void Compare(double[] expected, float[] actual, string what)
    {
        for (int i = 0; i < expected.Length; i++) AssertClose(expected[i], actual[i], $"{what}[{i}]");
    }

    /// <summary>
    /// The packed layout MultiHeadAttentionPacked runs: q, k and v as views at column offsets 0, W and 2W of one row-major
    /// [B * S, 3W] projection (row stride 3W), and dQ, dK, dV as views of a second such buffer (dQSpan) pre-filled with a
    /// pattern. Every element of that gradient buffer must come out equal to the dense interleaved run's gradients, which
    /// the reference check above already holds the backend to.
    /// </summary>
    private static void CheckPacked(IDirectGpuBackend backend)
    {
        var fused = Assert.IsAssignableFrom<IFusedAttentionBackend>(backend);
        var views = Assert.IsAssignableFrom<IGpuBufferViews>(backend);
        foreach (var (b, h, s, d, causal) in new[] { (2, 2, 9, 16, false), (1, 3, 6, 32, true) })
        {
            Assert.True(fused.SupportsFusedAttention(d));
            int w = h * d, rowWidth = 3 * w, rows = b * s;
            float[] x = Random(rows * rowWidth, 21 + d), dO = Random(rows * w, 22 + d);
            float[] q = new float[rows * w], k = new float[rows * w], v = new float[rows * w];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < w; c++)
                {
                    q[r * w + c] = x[r * rowWidth + c];
                    k[r * w + c] = x[r * rowWidth + w + c];
                    v[r * w + c] = x[r * rowWidth + 2 * w + c];
                }
            var dense = new FusedAttentionParams
            {
                B = b, H = h, Sq = s, Sk = s,
                Qb = (long)s * w, Qh = d, Qr = w, Kb = (long)s * w, Kh = d, Kr = w,
                Vb = (long)s * w, Vh = d, Vr = w, Ob = (long)s * w, Oh = d, Or = w,
                Scale = 1f / MathF.Sqrt(d), Causal = causal ? 1 : 0,
            };
            var packed = dense;
            packed.Qb = packed.Kb = packed.Vb = (long)s * rowWidth;
            packed.Qr = packed.Kr = packed.Vr = rowWidth;

            using var qBuf = backend.AllocateBuffer(q);
            using var kBuf = backend.AllocateBuffer(k);
            using var vBuf = backend.AllocateBuffer(v);
            using var dOBuf = backend.AllocateBuffer(dO);
            using var oDense = backend.AllocateBuffer(Filled(rows * w));
            using var lseDense = backend.AllocateBuffer(Filled(b * h * s));
            using var dqDense = backend.AllocateBuffer(Filled(rows * w));
            using var dkDense = backend.AllocateBuffer(Filled(rows * w));
            using var dvDense = backend.AllocateBuffer(Filled(rows * w));
            fused.FusedAttentionForward(qBuf, kBuf, vBuf, oDense, lseDense, null, dense, d);
            fused.FusedAttentionBackward(qBuf, kBuf, vBuf, oDense, dOBuf, lseDense, null, dqDense, dkDense, dvDense, dense, d);

            using var xBuf = backend.AllocateBuffer(x);
            using var gBuf = backend.AllocateBuffer(Filled(rows * rowWidth));
            using var oPacked = backend.AllocateBuffer(Filled(rows * w));
            using var lsePacked = backend.AllocateBuffer(Filled(b * h * s));
            IGpuBuffer View(IGpuBuffer whole, int start) => views.TryCreateView(whole, start, whole.Size - start)
                ?? throw new InvalidOperationException("view refused");
            using var qv = View(xBuf, 0);
            using var kv = View(xBuf, w);
            using var vv = View(xBuf, 2 * w);
            using var dqv = View(gBuf, 0);
            using var dkv = View(gBuf, w);
            using var dvv = View(gBuf, 2 * w);
            fused.FusedAttentionForward(qv, kv, vv, oPacked, lsePacked, null, packed, d);
            fused.FusedAttentionBackward(qv, kv, vv, oPacked, dOBuf, lsePacked, null, dqv, dkv, dvv, packed, d, gBuf);

            string what = $"packed B{b} H{h} S{s} D{d} causal={causal}";
            float[] o = backend.DownloadBuffer(oDense), oP = backend.DownloadBuffer(oPacked);
            for (int i = 0; i < o.Length; i++) AssertClose(o[i], oP[i], $"{what} o[{i}]");
            float[] dq = backend.DownloadBuffer(dqDense), dk = backend.DownloadBuffer(dkDense), dv = backend.DownloadBuffer(dvDense);
            float[] g = backend.DownloadBuffer(gBuf);
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < w; c++)
                {
                    AssertClose(dq[r * w + c], g[r * rowWidth + c], $"{what} dq[{r},{c}]");
                    AssertClose(dk[r * w + c], g[r * rowWidth + w + c], $"{what} dk[{r},{c}]");
                    AssertClose(dv[r * w + c], g[r * rowWidth + 2 * w + c], $"{what} dv[{r},{c}]");
                }
        }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Cuda_PackedAttention_MatchesTheDenseLayout(bool deterministic)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        bool prior = CpuParallelSettings.DeterministicReductions;
        try
        {
            CpuParallelSettings.DeterministicReductions = deterministic;
            CheckPacked(backend);
        }
        finally
        {
            CpuParallelSettings.DeterministicReductions = prior;
        }
    }

    // The packed node runs on CUDA and HIP only (pointer-offset views); HIP runs the shared portable kernels, which this
    // checks on OpenCL, whose sub-buffers take these offsets.
    [SkippableFact]
    public void OpenCl_PackedAttention_MatchesTheDenseLayout()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        CheckPacked(backend);
    }

    [SkippableFact]
    public void OpenCl_FusedAttention_MatchesReference()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_FusedAttention_MatchesReference()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_FusedAttention_MatchesReference()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }

    // The shared fa_p_ kernels through a launcher (the WebGPU WGSL ports on Dawn), with buffers on the host backend.
    private sealed class LaunchedFusedAttention : IFusedAttentionBackend
    {
        private readonly IDirectGpuBackend _host;
        private readonly PortableGenericFft.Launcher _launch;
        public LaunchedFusedAttention(IDirectGpuBackend host, PortableGenericFft.Launcher launch) { _host = host; _launch = launch; }
        public bool SupportsFusedAttention(int headDim) => PortableFusedAttention.Supports(headDim);
        public void FusedAttentionForward(IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer o, IGpuBuffer lse, IGpuBuffer? mask,
            in FusedAttentionParams p, int headDim)
            => PortableFusedAttention.Forward(_host, _launch, q, k, v, o, lse, mask, p, headDim);
        public void FusedAttentionBackward(IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer o, IGpuBuffer dO, IGpuBuffer lse,
            IGpuBuffer? mask, IGpuBuffer dQ, IGpuBuffer dK, IGpuBuffer dV, in FusedAttentionParams p, int headDim,
            IGpuBuffer? dQSpan = null)
            => PortableFusedAttention.Backward(_host, _launch, q, k, v, o, dO, lse, mask, dQ, dK, dV, p, headDim);
    }

    [SkippableFact]
    public void WebGpuWgsl_FusedAttention_MatchesReference_OnDawn()
    {
        using var dawn = DawnWgslLauncher.TryCreate();
        Skip.If(dawn is null, "Set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu (npm install webgpu).");
        using var host = new OpenClBackend();
        Skip.IfNot(host.IsAvailable, "OpenCL hosts the buffers and is not available.");
        Check(host, new LaunchedFusedAttention(host, dawn!.For(host)));
    }}
#endif
