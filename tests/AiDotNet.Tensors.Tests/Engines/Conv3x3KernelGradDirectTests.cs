#if NET5_0_OR_GREATER
using System;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Small 3x3 stride-1 kernel gradients run on a direct FMA kernel (one task per (oc, ic) pair over every image)
/// instead of im2col plus a native GEMM per image. These pin it against a double-precision reference over shapes
/// that hit every edge of its tiling (maps narrower than one 8-wide chunk, a masked last chunk, paddings 0/1/2,
/// odd channel counts, one image), the accumulate contract, and thread-count independence.
/// </summary>
[Collection("Conv3x3KernelSerial")]
public sealed class Conv3x3KernelGradDirectTests
{
    private static Tensor<float> Rnd(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    [SkippableTheory]
    [InlineData(64, 1, 28, 28, 16, 1)]   // parity CNN conv1
    [InlineData(4, 16, 14, 14, 32, 1)]   // parity CNN conv2 (fewer images): masked 6-wide last chunk
    [InlineData(3, 3, 5, 6, 5, 1)]       // narrower than one chunk, odd output channels
    [InlineData(2, 4, 9, 17, 3, 0)]      // pad 0: 7x15 output
    [InlineData(1, 5, 6, 11, 7, 2)]      // pad 2, one image
    [InlineData(2, 2, 1, 1, 2, 1)]       // 1x1 map: only the centre tap sees data
    public void MatchesDoubleReference(int batch, int ic, int h, int w, int oc, int pad)
    {
        Skip.IfNot(Avx2.IsSupported && Fma.IsSupported, "the direct kernel gradient is the AVX2/FMA route");
        var engine = new CpuEngine();
        var x = Rnd(new[] { batch, ic, h, w }, 1);
        int oh = h + 2 * pad - 2, ow = w + 2 * pad - 2;
        var dy = Rnd(new[] { batch, oc, oh, ow }, 2);
        var dk = new Tensor<float>(new[] { oc, ic, 3, 3 });
        for (int i = 0; i < dk.Length; i++) dk[i] = float.NaN;   // overwrite mode must replace every element
        engine.Conv2DBackwardKernelInto(dk, dy, x, dk._shape, new[] { 1, 1 }, new[] { pad, pad }, new[] { 1, 1 }, false);

        var prior = Rnd(dk._shape, 3);
        var dkAcc = new Tensor<float>(dk._shape);
        prior.AsSpan().CopyTo(dkAcc.AsWritableSpan());
        engine.Conv2DBackwardKernelInto(dkAcc, dy, x, dk._shape, new[] { 1, 1 }, new[] { pad, pad }, new[] { 1, 1 }, true);

        for (int o = 0; o < oc; o++)
        for (int i = 0; i < ic; i++)
        for (int kh = 0; kh < 3; kh++)
        for (int kw = 0; kw < 3; kw++)
        {
            double sum = 0, mag = 0;
            for (int b = 0; b < batch; b++)
            for (int r = 0; r < oh; r++)
            for (int c = 0; c < ow; c++)
            {
                int ih = r + kh - pad, iw = c + kw - pad;
                if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                double t = (double)dy[((b * oc + o) * oh + r) * ow + c] * x[((b * ic + i) * h + ih) * w + iw];
                sum += t; mag += Math.Abs(t);
            }
            int idx = ((o * ic + i) * 3 + kh) * 3 + kw;
            Assert.True(Math.Abs(dk[idx] - sum) <= 2e-5 * mag + 1e-6, $"dK[{o},{i},{kh},{kw}]: got {dk[idx]:R}, reference {sum:R}");
            Assert.Equal(BitConverter.SingleToInt32Bits(prior[idx] + dk[idx]), BitConverter.SingleToInt32Bits(dkAcc[idx]));
        }
    }

    /// <summary>
    /// The kernel's documented order, emulated in scalar code: for each (oc, ic) and tap, eight lane accumulators
    /// each take one fused multiply-add per image, row and 8-wide chunk (lanes past the row multiply a zero gradient),
    /// then the lanes are summed 0..7. The task split groups (oc, ic) pairs into blocks whose shape depends on the
    /// thread count; whatever the blocks, every dK element must come out bit-identical to this.
    /// </summary>
    [SkippableTheory]
    [InlineData(5, 16, 14, 14, 32, 1)]   // conv2-like: full blocks
    [InlineData(3, 7, 9, 11, 13, 1)]     // odd channel counts: partial blocks on both axes
    [InlineData(2, 3, 6, 5, 9, 0)]       // narrower than a chunk, pad 0
    public void MatchesLaneOrderEmulationAtEveryThreadCount(int batch, int ic, int h, int w, int oc, int pad)
    {
        Skip.IfNot(Avx2.IsSupported && Fma.IsSupported, "the direct kernel gradient is the AVX2/FMA route");
        var engine = new CpuEngine();
        var x = Rnd(new[] { batch, ic, h, w }, 21);
        int oh = h + 2 * pad - 2, ow = w + 2 * pad - 2;
        var dy = Rnd(new[] { batch, oc, oh, ow }, 22);
        int chunks = (ow + 7) / 8;
        var expected = new float[oc * ic * 9];
        for (int o = 0; o < oc; o++)
        for (int i = 0; i < ic; i++)
        for (int kh = 0; kh < 3; kh++)
        for (int kw = 0; kw < 3; kw++)
        {
            var lanes = new float[8];
            for (int b = 0; b < batch; b++)
            for (int r = 0; r < oh; r++)
            for (int c = 0; c < chunks; c++)
            for (int l = 0; l < 8; l++)
            {
                int col = c * 8 + l;
                float g = col < ow ? dy[((b * oc + o) * oh + r) * ow + col] : 0f;
                int ih = r + kh - pad, iw = col + kw - pad;
                // Padded cells and columns past the padded row (only under a zero gradient) read as zero.
                float xv = ih >= 0 && ih < h && iw >= 0 && iw < w && col < ow ? x[((b * ic + i) * h + ih) * w + iw] : 0f;
                lanes[l] = MathF.FusedMultiplyAdd(g, xv, lanes[l]);
            }
            float s = lanes[0];
            for (int l = 1; l < 8; l++) s += lanes[l];
            expected[((o * ic + i) * 3 + kh) * 3 + kw] = s;
        }
        int prior = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            foreach (int dop in new[] { 1, 2, 3, Math.Max(4, Environment.ProcessorCount) })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = dop;
                var dk = new Tensor<float>(new[] { oc, ic, 3, 3 });
                engine.Conv2DBackwardKernelInto(dk, dy, x, dk._shape, new[] { 1, 1 }, new[] { pad, pad }, new[] { 1, 1 }, false);
                for (int k = 0; k < expected.Length; k++)
                    Assert.True(BitConverter.SingleToInt32Bits(expected[k]) == BitConverter.SingleToInt32Bits(dk[k]),
                        $"threads {dop}, dK[{k}]: expected {expected[k]:R} got {dk[k]:R}");
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = prior;
        }
    }

    [SkippableFact]
    public void ResultIsIndependentOfThreadCount()
    {
        Skip.IfNot(Avx2.IsSupported && Fma.IsSupported, "the direct kernel gradient is the AVX2/FMA route");
        var engine = new CpuEngine();
        var x = Rnd(new[] { 8, 16, 14, 14 }, 4);
        var dy = Rnd(new[] { 8, 32, 14, 14 }, 5);
        int prior = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            var serial = new Tensor<float>(new[] { 32, 16, 3, 3 });
            engine.Conv2DBackwardKernelInto(serial, dy, x, serial._shape, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, false);
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount);
            var parallel = new Tensor<float>(serial._shape);
            engine.Conv2DBackwardKernelInto(parallel, dy, x, serial._shape, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, false);
            for (int i = 0; i < serial.Length; i++)
                Assert.Equal(BitConverter.SingleToInt32Bits(serial[i]), BitConverter.SingleToInt32Bits(parallel[i]));
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = prior;
        }
    }
}
#endif