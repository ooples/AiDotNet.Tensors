using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using System;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

#if NET5_0_OR_GREATER
/// <summary>
/// The direct blocked conv kernels against a naive double-precision reference, called directly so every geometry is
/// covered whatever the routing predicates currently send to them (the engine-level conv suites cover the routed
/// shapes).
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class DirectConvAvx2Tests
{
    private static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, (float)(rng.NextDouble() * 2 - 1));
        return t;
    }

    private static int OutSize(int size, int k, int stride, int pad, int dilation) => (size + 2 * pad - dilation * (k - 1) - 1) / stride + 1;

    private static void AssertClose(double[] expected, Tensor<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double e = expected[i], a = actual.GetFlat(i);
            Assert.True(Math.Abs(e - a) <= 1e-4 * Math.Max(1.0, Math.Abs(e)), $"{what}[{i}]: expected {e}, got {a}");
        }
    }

    // Shapes: batch, inC, outC, size, kernel, stride, pad, dilation. 3x3 and 1x1, strides 1 and 2, padding 0-2,
    // dilation 2, and output-position counts that leave a 1- and 2-position tail of the 3-position tile.
    [SkippableTheory]
    [InlineData(2, 8, 32, 7, 3, 1, 1, 1)]
    [InlineData(3, 16, 64, 6, 3, 2, 1, 1)]
    [InlineData(2, 8, 32, 9, 1, 2, 0, 1)]
    [InlineData(2, 16, 32, 8, 3, 1, 2, 1)]
    [InlineData(2, 8, 32, 9, 3, 1, 2, 2)]
    [InlineData(4, 24, 32, 5, 3, 1, 0, 1)]
    [InlineData(2, 3, 32, 7, 3, 1, 1, 1)]     // a 3-channel image stem: input channels zero-padded to one block
    [InlineData(2, 13, 32, 6, 3, 2, 1, 1)]    // a full block plus a partial one
    public void Forward_MatchesANaiveReference(int batch, int inC, int outC, int size, int k, int stride, int pad, int dilation)
    {
        int outSize = OutSize(size, k, stride, pad, dilation);
        Skip.IfNot(DirectConvAvx2.IsSupported, "needs AVX2 and FMA");
        var x = Random(1, batch, inC, size, size);
        var w = Random(2, outC, inC, k, k);
        var y = new Tensor<float>(new[] { batch, outC, outSize, outSize });
        for (int i = 0; i < y.Length; i++) y.SetFlat(i, float.NaN);

        DirectConvAvx2.Forward(x.GetDataArray(), 0, w.GetDataArray(), 0, y.GetDataArray(), 0, accumulate: false,
            batch, inC, size, size, outC, k, k, stride, stride, pad, pad, dilation, dilation, outSize, outSize);

        var expected = new double[batch * outC * outSize * outSize];
        for (int b = 0; b < batch; b++)
        for (int o = 0; o < outC; o++)
        for (int oh = 0; oh < outSize; oh++)
        for (int ow = 0; ow < outSize; ow++)
        {
            double acc = 0;
            for (int c = 0; c < inC; c++)
            for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
            {
                int ih = oh * stride + i * dilation - pad, iw = ow * stride + j * dilation - pad;
                if (ih < 0 || ih >= size || iw < 0 || iw >= size) continue;
                acc += (double)w[o, c, i, j] * x[b, c, ih, iw];
            }
            expected[((b * outC + o) * outSize + oh) * outSize + ow] = acc;
        }
        AssertClose(expected, y, "output");
    }

    // #653: the forward runs the input channels in passes whose kernel slice fits a cache budget, later passes
    // accumulating onto the first. A small budget forces several passes (one block each at 9 KB for 3x3), an uneven
    // split, a single image on the staged contiguous store, a batch on the per-position store, 1x1, and accumulate.
    [SkippableTheory]
    [InlineData(1, 64, 32, 34, 3, 1, 1, false, 9 * 1024)]     // staged single image, 8 passes of one block
    [InlineData(1, 72, 64, 33, 3, 1, 1, true, 30 * 1024)]     // staged, accumulate, 9 blocks in passes of 3
    [InlineData(2, 40, 32, 7, 3, 1, 1, false, 20 * 1024)]     // batched per-position store, 5 blocks split 3 + 2
    [InlineData(2, 48, 32, 6, 3, 2, 0, true, 10 * 1024)]      // strided, no padding, accumulate
    [InlineData(2, 64, 32, 5, 1, 1, 0, false, 2 * 1024)]      // 1x1: 2 blocks per pass
    public void Forward_InInputChannelPasses_MatchesANaiveReference(int batch, int inC, int outC, int size, int k, int stride,
        int pad, bool accumulate, int passBudgetBytes)
    {
        Skip.IfNot(DirectConvAvx2.IsSupported, "needs AVX2 and FMA");
        int outSize = OutSize(size, k, stride, pad, 1);
        int inBlocks = (inC + 7) / 8;
        int before = DirectConvAvx2.s_forwardPassKernelBytes;
        try
        {
            DirectConvAvx2.s_forwardPassKernelBytes = passBudgetBytes;
            Assert.True(DirectConvAvx2.ForwardInputBlocksPerPass(inBlocks, k * k) < inBlocks, "the budget must force passes");

            var x = Random(11, batch, inC, size, size);
            var w = Random(12, outC, inC, k, k);
            var prior = Random(13, batch, outC, outSize, outSize);
            var y = new Tensor<float>(new[] { batch, outC, outSize, outSize });
            for (int i = 0; i < y.Length; i++) y.SetFlat(i, accumulate ? prior.GetFlat(i) : float.NaN);

            DirectConvAvx2.Forward(x.GetDataArray(), 0, w.GetDataArray(), 0, y.GetDataArray(), 0, accumulate,
                batch, inC, size, size, outC, k, k, stride, stride, pad, pad, 1, 1, outSize, outSize);

            var expected = new double[batch * outC * outSize * outSize];
            for (int b = 0; b < batch; b++)
            for (int o = 0; o < outC; o++)
            for (int oh = 0; oh < outSize; oh++)
            for (int ow = 0; ow < outSize; ow++)
            {
                int index = ((b * outC + o) * outSize + oh) * outSize + ow;
                double acc = accumulate ? prior.GetFlat(index) : 0;
                for (int c = 0; c < inC; c++)
                for (int i = 0; i < k; i++)
                for (int j = 0; j < k; j++)
                {
                    int ih = oh * stride + i - pad, iw = ow * stride + j - pad;
                    if (ih < 0 || ih >= size || iw < 0 || iw >= size) continue;
                    acc += (double)w[o, c, i, j] * x[b, c, ih, iw];
                }
                expected[index] = acc;
            }
            AssertClose(expected, y, "output");
        }
        finally
        {
            DirectConvAvx2.s_forwardPassKernelBytes = before;
        }
    }

#if NET9_0_OR_GREATER
    // The 512-bit forward tile (six positions x two 16-channel vectors) computes every output element with the same
    // FMA chain as the 256-bit tile, so a conv may not depend on the host's vector width. Forced on, it runs in software
    // where the CPU lacks AVX-512. Shapes: position counts leaving every tail of the 6-position step, a single image on
    // the staged store, several input passes, accumulate, stride 2, dilation 2, and the backward-input conv.
    [SkippableTheory]
    [InlineData(2, 16, 32, 7, 3, 1, 1, 1, false, 0)]        // 49 positions per image: a 1-position tail
    [InlineData(1, 64, 64, 34, 3, 1, 1, 1, false, 0)]       // single image, 1156 positions: staged store
    [InlineData(1, 64, 32, 33, 3, 1, 1, 1, true, 9216)]     // staged, accumulate, one input block per pass
    [InlineData(3, 24, 32, 9, 3, 2, 1, 1, false, 0)]        // stride 2: 25 positions, a 1-position tail
    [InlineData(2, 8, 32, 10, 3, 1, 2, 2, true, 0)]         // dilation 2, accumulate
    [InlineData(2, 40, 64, 5, 1, 1, 0, 1, false, 2048)]     // 1x1, two passes
    public void Vector512ForwardTile_IsBitIdenticalTo256Bit(int batch, int inC, int outC, int size, int k, int stride,
        int pad, int dilation, bool accumulate, int passBudgetBytes)
    {
        Skip.IfNot(DirectConvAvx2.IsSupported, "needs AVX2 and FMA");
        int outSize = OutSize(size, k, stride, pad, dilation);
        var x = Random(21, batch, inC, size, size);
        var w = Random(22, outC, inC, k, k);
        var prior = Random(23, batch, outC, outSize, outSize);
        float[] Run(bool wide)
        {
            var y = new Tensor<float>(new[] { batch, outC, outSize, outSize });
            for (int i = 0; i < y.Length; i++) y.SetFlat(i, accumulate ? prior.GetFlat(i) : float.NaN);
            bool? before = SimdGemm.s_forceVector512Direct;
            int budget = DirectConvAvx2.s_forwardPassKernelBytes;
            try
            {
                SimdGemm.s_forceVector512Direct = wide;
                if (passBudgetBytes > 0) DirectConvAvx2.s_forwardPassKernelBytes = passBudgetBytes;
                DirectConvAvx2.Forward(x.GetDataArray(), 0, w.GetDataArray(), 0, y.GetDataArray(), 0, accumulate,
                    batch, inC, size, size, outC, k, k, stride, stride, pad, pad, dilation, dilation, outSize, outSize);
            }
            finally
            {
                SimdGemm.s_forceVector512Direct = before;
                DirectConvAvx2.s_forwardPassKernelBytes = budget;
            }
            return y.ToArray();
        }
        var narrow = Run(false);
        var wide = Run(true);
        for (int i = 0; i < narrow.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(narrow[i]) == BitConverter.SingleToInt32Bits(wide[i]),
                $"output[{i}]: 256-bit {narrow[i]:G9}, 512-bit {wide[i]:G9}");
    }

    [SkippableFact]
    public void Vector512ForwardTile_BackwardInputIsBitIdenticalTo256Bit()
    {
        Skip.IfNot(DirectConvAvx2.IsSupported, "needs AVX2 and FMA");
        const int batch = 2, inC = 32, outC = 16, size = 9, k = 3;
        var g = Random(31, batch, outC, size, size);
        var w = Random(32, outC, inC, k, k);
        float[] Run(bool wide)
        {
            var dx = new Tensor<float>(new[] { batch, inC, size, size });
            for (int i = 0; i < dx.Length; i++) dx.SetFlat(i, float.NaN);
            bool? before = SimdGemm.s_forceVector512Direct;
            try
            {
                SimdGemm.s_forceVector512Direct = wide;
                DirectConvAvx2.BackwardInput(g.GetDataArray(), 0, w.GetDataArray(), 0, dx.GetDataArray(), 0, false,
                    batch, inC, size, size, outC, k, k, 1, 1, 1, 1, size, size);
            }
            finally { SimdGemm.s_forceVector512Direct = before; }
            return dx.ToArray();
        }
        var narrow = Run(false);
        var wide = Run(true);
        for (int i = 0; i < narrow.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(narrow[i]) == BitConverter.SingleToInt32Bits(wide[i]),
                $"dx[{i}]: 256-bit {narrow[i]:G9}, 512-bit {wide[i]:G9}");
    }
#endif
    [SkippableTheory]
    [InlineData(2, 32, 8, 7, 3, 1, 1, false)]
    [InlineData(3, 64, 16, 4, 3, 1, 1, true)]
    [InlineData(2, 32, 16, 6, 3, 1, 0, false)]
    [InlineData(2, 32, 8, 5, 1, 1, 0, true)]
    // Strided: each dX phase is a stride-1 correlation with a sub-kernel. Even and odd sizes, a 1x1 kernel whose odd
    // phases receive no tap (zeroed, or left alone when accumulating), stride 3, and padding 0.
    [InlineData(2, 32, 16, 8, 3, 2, 1, false)]
    [InlineData(3, 32, 8, 9, 3, 2, 1, true)]
    [InlineData(2, 32, 16, 8, 1, 2, 0, false)]
    [InlineData(2, 32, 16, 9, 1, 2, 0, true)]
    [InlineData(2, 32, 8, 10, 3, 3, 1, false)]
    [InlineData(2, 64, 16, 7, 3, 2, 0, false)]
    public void BackwardInput_MatchesANaiveReference(int batch, int inC, int outC, int size, int k, int stride, int pad, bool accumulate)
    {
        int outSize = OutSize(size, k, stride, pad, 1);
        Skip.IfNot(DirectConvAvx2.IsSupported, "needs AVX2 and FMA");
        var w = Random(3, outC, inC, k, k);
        var g = Random(4, batch, outC, outSize, outSize);
        var prior = Random(5, batch, inC, size, size);
        var dx = new Tensor<float>(prior.Shape.ToArray());
        for (int i = 0; i < dx.Length; i++) dx.SetFlat(i, accumulate ? prior.GetFlat(i) : float.NaN);

        DirectConvAvx2.BackwardInput(g.GetDataArray(), 0, w.GetDataArray(), 0, dx.GetDataArray(), 0, accumulate,
            batch, inC, size, size, outC, k, k, stride, stride, pad, pad, outSize, outSize);

        var expected = new double[dx.Length];
        for (int b = 0; b < batch; b++)
        for (int c = 0; c < inC; c++)
        for (int ih = 0; ih < size; ih++)
        for (int iw = 0; iw < size; iw++)
        {
            double acc = accumulate ? prior[b, c, ih, iw] : 0;
            for (int o = 0; o < outC; o++)
            for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
            {
                int nh = ih + pad - i, nw = iw + pad - j;
                if (nh < 0 || nw < 0 || nh % stride != 0 || nw % stride != 0) continue;
                int oh = nh / stride, ow = nw / stride;
                if (oh >= outSize || ow >= outSize) continue;
                acc += (double)w[o, c, i, j] * g[b, o, oh, ow];
            }
            expected[((b * inC + c) * size + ih) * size + iw] = acc;
        }
        AssertClose(expected, dx, "input gradient");
    }

#if NET9_0_OR_GREATER
    // The 512-bit weight-gradient tile (two input blocks per 16-lane vector) keeps every lane's FMA chain and the
    // 256-bit path's image splits, so dW may not depend on the host's vector width. Shapes: the batch split across
    // partial gradients, one task per block pair, 5 kernel columns (a 3-column group and a 2-column one), accumulate.
    [SkippableTheory]
    [InlineData(7, 64, 64, 5, 3, 1, 1, false)]    // split into partial gradients
    [InlineData(2, 128, 128, 4, 3, 1, 1, false)]  // enough block pairs that the batch is not split
    [InlineData(2, 32, 32, 6, 5, 1, 2, true)]     // 5 columns, accumulate
    [InlineData(3, 16, 64, 8, 3, 2, 1, true)]     // stride 2
    public void Vector512KernelGradient_IsBitIdenticalTo256Bit(int batch, int inC, int outC, int size, int k, int stride, int pad, bool accumulate)
    {
        Skip.IfNot(DirectConvAvx2.IsSupported, "needs AVX2 and FMA");
        int outSize = OutSize(size, k, stride, pad, 1);
        var x = Random(41, batch, inC, size, size);
        var g = Random(42, batch, outC, outSize, outSize);
        var prior = Random(43, outC, inC, k, k);
        float[] Run(bool wide)
        {
            var dw = new Tensor<float>(new[] { outC, inC, k, k });
            for (int i = 0; i < dw.Length; i++) dw.SetFlat(i, accumulate ? prior.GetFlat(i) : float.NaN);
            bool? before = SimdGemm.s_forceVector512Direct;
            try
            {
                SimdGemm.s_forceVector512Direct = wide;
                DirectConvAvx2.BackwardKernel(x.GetDataArray(), 0, g.GetDataArray(), 0, dw.GetDataArray(), 0, accumulate,
                    batch, inC, size, size, outC, k, k, stride, stride, pad, pad, 1, 1, outSize, outSize);
            }
            finally { SimdGemm.s_forceVector512Direct = before; }
            return dw.ToArray();
        }
        var narrow = Run(false);
        var wide = Run(true);
        for (int i = 0; i < narrow.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(narrow[i]) == BitConverter.SingleToInt32Bits(wide[i]),
                $"dW[{i}]: 256-bit {narrow[i]:G9}, 512-bit {wide[i]:G9}");
    }
#endif
    [SkippableTheory]
    [InlineData(2, 32, 32, 7, 3, 1, 1, 1, false)]
    [InlineData(3, 16, 64, 8, 3, 2, 1, 1, true)]
    [InlineData(2, 32, 32, 9, 1, 2, 0, 1, false)]
    [InlineData(2, 32, 32, 9, 3, 1, 2, 2, false)]
    [InlineData(2, 32, 32, 6, 5, 1, 2, 1, true)]   // 5 kernel columns: a full 3-column group and a 2-column tail
    [InlineData(7, 64, 64, 5, 3, 1, 1, 1, true)]   // the batch split unevenly (4 + 3 images) across partial gradients
    [InlineData(2, 128, 128, 4, 3, 1, 1, 1, false)] // enough block pairs that the batch is not split
    public void BackwardKernel_MatchesANaiveReference(int batch, int inC, int outC, int size, int k, int stride, int pad, int dilation, bool accumulate)
    {
        int outSize = OutSize(size, k, stride, pad, dilation);
        Skip.IfNot(DirectConvAvx2.IsSupported, "needs AVX2 and FMA");
        var x = Random(6, batch, inC, size, size);
        var g = Random(7, batch, outC, outSize, outSize);
        var prior = Random(8, outC, inC, k, k);
        var dw = new Tensor<float>(prior.Shape.ToArray());
        for (int i = 0; i < dw.Length; i++) dw.SetFlat(i, accumulate ? prior.GetFlat(i) : float.NaN);

        DirectConvAvx2.BackwardKernel(x.GetDataArray(), 0, g.GetDataArray(), 0, dw.GetDataArray(), 0, accumulate,
            batch, inC, size, size, outC, k, k, stride, stride, pad, pad, dilation, dilation, outSize, outSize);

        var expected = new double[dw.Length];
        for (int o = 0; o < outC; o++)
        for (int c = 0; c < inC; c++)
        for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
        {
            double acc = accumulate ? prior[o, c, i, j] : 0;
            for (int b = 0; b < batch; b++)
            for (int oh = 0; oh < outSize; oh++)
            for (int ow = 0; ow < outSize; ow++)
            {
                int ih = oh * stride + i * dilation - pad, iw = ow * stride + j * dilation - pad;
                if (ih < 0 || ih >= size || iw < 0 || iw >= size) continue;
                acc += (double)g[b, o, oh, ow] * x[b, c, ih, iw];
            }
            expected[((o * inC + c) * k + i) * k + j] = acc;
        }
        AssertClose(expected, dw, "kernel gradient");
    }

    [SkippableFact]
    public void EngineRoutes_ReadAndWriteOffsetViewsInPlace()
    {
        // Operands that are contiguous views at a nonzero storage offset (an arena slab, a batch slice): every pass the
        // engine sends to the direct kernels must address them at their offset, against a copy at offset 0.
        Skip.IfNot(DirectConvAvx2.IsSupported, "needs AVX2 and FMA");
        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        int[] stride = { 2, 2 }, pad = { 1, 1 }, dil = { 1, 1 };
        Assert.True(DirectConvAvx2.TryChoose(new DirectConvShape(DirectConvPass.Forward, 4, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1), out _));
        Assert.True(DirectConvAvx2.TryChoose(new DirectConvShape(DirectConvPass.BackwardInput, 4, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1), out _));
        Assert.True(DirectConvAvx2.TryChoose(new DirectConvShape(DirectConvPass.BackwardKernel, 4, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1), out _));

        var xBig = Random(11, 6, 32, 16, 16); var x = xBig.Slice(0, 1, 5);
        var w = Random(12, 32, 32, 3, 3);
        var gBig = Random(13, 6, 32, 8, 8); var g = gBig.Slice(0, 2, 6);
        var xFlat = x.Contiguous().Clone(); var gFlat = g.Contiguous().Clone();
        Assert.True(x.IsContiguous && g.IsContiguous);

        var y = engine.Conv2D(x, w, stride, pad, dil);
        var yRef = engine.Conv2D(xFlat, w, stride, pad, dil);
        Assert.Equal(yRef.AsSpan().ToArray(), y.AsSpan().ToArray());

        var w32 = Random(14, 32, 32, 3, 3);
        var dxBig = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 6, 32, 16, 16 });
        var dx = dxBig.Slice(0, 1, 5);
        var dxRef = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 4, 32, 16, 16 });
        engine.Conv2DBackwardInputInto(dx, g, w32, new[] { 4, 32, 16, 16 }, stride, pad, dil, accumulate: false);
        engine.Conv2DBackwardInputInto(dxRef, gFlat, w32, new[] { 4, 32, 16, 16 }, stride, pad, dil, accumulate: false);
        Assert.Equal(dxRef.AsSpan().ToArray(), dx.Contiguous().AsSpan().ToArray());
        Assert.All(dxBig.Slice(0, 0, 1).AsSpan().ToArray(), v => Assert.Equal(0f, v));   // nothing written outside the view

        var dw = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 32, 32, 3, 3 });
        var dwRef = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 32, 32, 3, 3 });
        engine.Conv2DBackwardKernelInto(dw, g, x, new[] { 32, 32, 3, 3 }, stride, pad, dil, accumulate: false);
        engine.Conv2DBackwardKernelInto(dwRef, gFlat, xFlat, new[] { 32, 32, 3, 3 }, stride, pad, dil, accumulate: false);
        Assert.Equal(dwRef.AsSpan().ToArray(), dw.AsSpan().ToArray());
    }
}
#endif