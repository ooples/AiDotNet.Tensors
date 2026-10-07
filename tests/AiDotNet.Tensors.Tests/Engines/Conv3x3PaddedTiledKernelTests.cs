#if NET5_0_OR_GREATER
using System;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>Serialised: these tests mutate the process-wide conv variant and parallelism settings.</summary>
[CollectionDefinition("Conv3x3KernelSerial", DisableParallelization = true)]
public sealed class Conv3x3KernelSerialCollection { }

/// <summary>
/// The default float 3x3 stride-1 conv runs every output position (borders and the partial last column chunk
/// included) as vector FMAs over a zero-padded copy of the image. These pin its numerics against a double
/// reference over shapes that hit every edge of the tiling: maps narrower than one 8-wide chunk, widths that
/// leave a masked tail, heights that leave a row remainder below the 4-row block, odd output-channel counts,
/// and paddings 0/1/2.
/// </summary>
[Collection("Conv3x3KernelSerial")]
public sealed unsafe class Conv3x3PaddedTiledKernelTests : IDisposable
{
    private readonly int _priorMaxDop = CpuParallelSettings.MaxDegreeOfParallelism;
    private readonly SimdConvHelper.Conv3x3Variant _priorVariant = SimdConvHelper.ActiveConv3x3Variant;

    public Conv3x3PaddedTiledKernelTests()
    {
        SimdConvHelper.ActiveConv3x3Variant = SimdConvHelper.Conv3x3Variant.Auto;
    }

    public void Dispose()
    {
        CpuParallelSettings.MaxDegreeOfParallelism = _priorMaxDop;
        SimdConvHelper.ActiveConv3x3Variant = _priorVariant;
    }

    private static float[] Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static float[] Run(float[] x, float[] k, int batch, int ic, int h, int w, int oc, int pad)
    {
        int oh = h + 2 * pad - 2, ow = w + 2 * pad - 2;
        // Poisoned so an output position the kernel never writes cannot pass as a zero.
        var y = new float[batch * oc * oh * ow];
        Array.Fill(y, float.NaN);
        fixed (float* px = x) fixed (float* pk = k) fixed (float* py = y)
            SimdConvHelper.Conv3x3Stride1(px, pk, py, batch, ic, h, w, oc, pad, pad, 1, 1);
        return y;
    }

    [SkippableTheory]
    [InlineData(2, 1, 28, 28, 16, 1)]   // CNN conv1
    [InlineData(3, 16, 14, 14, 32, 1)]  // CNN conv2: masked 6-wide tail, 2-row remainder
    [InlineData(1, 32, 14, 14, 16, 1)]  // conv2 input-gradient (flipped kernel)
    [InlineData(2, 3, 5, 6, 5, 1)]      // narrower than one chunk, odd output channels
    [InlineData(1, 2, 3, 3, 1, 1)]      // single output channel, 3x3 map
    [InlineData(2, 4, 9, 17, 3, 0)]     // pad 0: 7x15 output
    [InlineData(1, 5, 6, 11, 7, 2)]     // pad 2: 8x13 output
    [InlineData(1, 1, 1, 1, 2, 1)]      // 1x1 map, every tap but the centre is padding
    public void MatchesDoubleReference(int batch, int ic, int h, int w, int oc, int pad)
    {
        Skip.IfNot(Fma.IsSupported, "the padded-tiled kernel is the FMA route");
        var x = Rand(batch * ic * h * w, 1);
        var k = Rand(oc * ic * 9, 2);
        var y = Run(x, k, batch, ic, h, w, oc, pad);
        int oh = h + 2 * pad - 2, ow = w + 2 * pad - 2;
        for (int b = 0; b < batch; b++)
        for (int o = 0; o < oc; o++)
        for (int r = 0; r < oh; r++)
        for (int c = 0; c < ow; c++)
        {
            double sum = 0, mag = 0;
            for (int i = 0; i < ic; i++)
            for (int kh = 0; kh < 3; kh++)
            for (int kw = 0; kw < 3; kw++)
            {
                int ih = r + kh - pad, iw = c + kw - pad;
                if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                double t = (double)x[((b * ic + i) * h + ih) * w + iw] * k[(o * ic + i) * 9 + kh * 3 + kw];
                sum += t; mag += Math.Abs(t);
            }
            float got = y[((b * oc + o) * oh + r) * ow + c];
            Assert.True(Math.Abs(got - sum) <= 2e-5 * mag + 1e-6,
                $"b={b} oc={o} ({r},{c}): got {got:R}, reference {sum:R}");
        }
    }

    /// <summary>
    /// Interior outputs keep the legacy OcBlock4 kernel's reduction order exactly (input channel, kernel row,
    /// kernel column, one FMA per tap), so on the positions that kernel vectorized the two agree to the bit.
    /// </summary>
    [SkippableFact]
    public void InteriorIsBitIdenticalToLegacyBlock4()
    {
        Skip.IfNot(Fma.IsSupported, "the padded-tiled kernel is the FMA route");
        const int batch = 2, ic = 16, h = 20, w = 21, oc = 8;
        var x = Rand(batch * ic * h * w, 3);
        var k = Rand(oc * ic * 9, 4);
        var tiled = Run(x, k, batch, ic, h, w, oc, 1);
        SimdConvHelper.ActiveConv3x3Variant = SimdConvHelper.Conv3x3Variant.Block4;
        var legacy = Run(x, k, batch, ic, h, w, oc, 1);
        int simdEnd = 1 + ((w - 2) & ~7);   // the legacy kernel's vectorized interior columns: [1, simdEnd)
        int compared = 0;
        for (int b = 0; b < batch; b++)
        for (int o = 0; o < oc; o++)
        for (int r = 1; r < h - 1; r++)
        for (int c = 1; c < simdEnd; c++)
        {
            int idx = ((b * oc + o) * h + r) * w + c;
            Assert.Equal(BitConverter.SingleToInt32Bits(legacy[idx]), BitConverter.SingleToInt32Bits(tiled[idx]));
            compared++;
        }
        Assert.True(compared > 0);
    }

    /// <summary>Each output element is written by exactly one task, so the thread budget cannot change a bit.</summary>
    [SkippableFact]
    public void ResultIsIndependentOfThreadCount()
    {
        Skip.IfNot(Fma.IsSupported, "the padded-tiled kernel is the FMA route");
        const int batch = 8, ic = 16, h = 14, w = 14, oc = 32;
        var x = Rand(batch * ic * h * w, 5);
        var k = Rand(oc * ic * 9, 6);
        CpuParallelSettings.MaxDegreeOfParallelism = 1;
        var serial = Run(x, k, batch, ic, h, w, oc, 1);
        CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount);
        var parallel = Run(x, k, batch, ic, h, w, oc, 1);
        for (int i = 0; i < serial.Length; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(serial[i]), BitConverter.SingleToInt32Bits(parallel[i]));
    }
}
#endif