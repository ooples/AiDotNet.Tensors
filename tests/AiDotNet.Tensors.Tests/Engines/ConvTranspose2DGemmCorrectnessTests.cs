using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

// Validates the new GEMM-based ConvTranspose2D forward fast path against
// the existing naive 7-nested-loop reference. The naive path stays as the
// fallback when BLAS isn't available, so we exercise BOTH paths and assert
// they produce numerically equivalent output. Shapes cover DCGAN's
// generator stack (4×4 stride 2 padding 1, channels 512→256→128→64→3) and a
// few small shapes to catch indexing edge cases.
public class ConvTranspose2DGemmCorrectnessTests
{
    private static Tensor<double> MakeRandomTensor(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var rng = new Random(seed);
        var span = t.Data.Span;
        for (int i = 0; i < span.Length; i++) span[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static double MaxAbsDiff(Tensor<double> a, Tensor<double> b)
    {
        var sa = a.Data.Span;
        var sb = b.Data.Span;
        double max = 0;
        for (int i = 0; i < sa.Length; i++)
        {
            double d = Math.Abs(sa[i] - sb[i]);
            if (d > max) max = d;
        }
        return max;
    }

    // Naive reference identical to the pre-BLAS CpuEngine.ConvTranspose2D
    // inner loop — extracted here so we can compare against the BLAS path
    // without depending on whether the engine's naive branch is reachable.
    private static double[] NaiveConvTranspose2DDouble(
        double[] input, double[] kernel,
        int batch, int inChannels, int height, int width,
        int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW,
        int outputHeight, int outputWidth)
    {
        var output = new double[batch * outChannels * outputHeight * outputWidth];
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        double sum = 0d;
                        for (int kh = 0; kh < kernelHeight; kh++)
                        {
                            int numerH = oh + padH - kh;
                            if (numerH < 0 || numerH % strideH != 0) continue;
                            int ih = numerH / strideH;
                            if (ih >= height) continue;
                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                int numerW = ow + padW - kw;
                                if (numerW < 0 || numerW % strideW != 0) continue;
                                int iw = numerW / strideW;
                                if (iw >= width) continue;
                                for (int ic = 0; ic < inChannels; ic++)
                                {
                                    int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                    int kernelIdx = ((ic * outChannels + oc) * kernelHeight + kh) * kernelWidth + kw;
                                    sum += input[inputIdx] * kernel[kernelIdx];
                                }
                            }
                        }
                        int outputIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                        output[outputIdx] = sum;
                    }
                }
            }
        }
        return output;
    }

    [Theory]
    // DCGAN generator stack:
    [InlineData(1, 512, 4, 4, 256, 4, 4, 2, 1)]   // 4×4 → 8×8
    [InlineData(1, 256, 8, 8, 128, 4, 4, 2, 1)]   // 8×8 → 16×16
    [InlineData(1, 128, 16, 16, 64, 4, 4, 2, 1)]  // 16×16 → 32×32
    [InlineData(1, 64, 32, 32, 3, 4, 4, 2, 1)]    // 32×32 → 64×64 (final Tanh)
    // Edge: stride 1 (identity-ish)
    [InlineData(2, 8, 4, 4, 16, 3, 3, 1, 1)]
    // Edge: stride 2 padding 0 (output grows extra)
    [InlineData(1, 4, 3, 3, 8, 3, 3, 2, 0)]
    // Edge: kernel == stride (no overlap)
    [InlineData(1, 8, 4, 4, 8, 2, 2, 2, 0)]
    // Edge: small batch=2 to catch per-batch loop bugs
    [InlineData(2, 4, 4, 4, 8, 4, 4, 2, 1)]
    public void GemmPath_MatchesNaiveReference_Double(
        int batch, int inChannels, int inH, int inW,
        int outChannels, int kH, int kW, int stride, int padding)
    {
        var inputT = MakeRandomTensor(new[] { batch, inChannels, inH, inW }, seed: 42);
        var kernelT = MakeRandomTensor(new[] { inChannels, outChannels, kH, kW }, seed: 43);

        int outH = (inH - 1) * stride - 2 * padding + kH;
        int outW = (inW - 1) * stride - 2 * padding + kW;

        // Naive reference
        var inputArr = inputT.Data.ToArray();
        var kernelArr = kernelT.Data.ToArray();
        var expected = NaiveConvTranspose2DDouble(
            inputArr, kernelArr,
            batch, inChannels, inH, inW,
            outChannels, kH, kW,
            stride, stride, padding, padding,
            outH, outW);

        // GEMM path
        var actual = new double[batch * outChannels * outH * outW];
        bool used = Im2ColHelper.TryConvTranspose2DWithGemm(
            inputArr, kernelArr, actual,
            batch, inChannels, inH, inW,
            outChannels, kH, kW,
            stride, stride, padding, padding,
            outH, outW);

        // If BLAS isn't available on this host the GEMM path returns false;
        // we can't validate it there. Skip with a no-op assertion in that case.
        if (!used)
        {
            return;
        }

        // L∞ tolerance: GEMM order-of-summation differs from naive, so we
        // get bit-noise on the order of n_reductions × machine_epsilon.
        // For double at K=inChannels ≤ 512 that's well under 1e-9.
        var actualT = new Tensor<double>(new[] { batch, outChannels, outH, outW }, new Vector<double>(actual));
        var expectedT = new Tensor<double>(new[] { batch, outChannels, outH, outW }, new Vector<double>(expected));
        double maxDiff = MaxAbsDiff(actualT, expectedT);
        Assert.True(maxDiff < 1e-9,
            $"GEMM vs naive ConvTranspose2D drift: maxDiff={maxDiff:E3} for shape "
            + $"[B={batch},Ci={inChannels},H={inH},W={inW}] → [Co={outChannels}] k={kH}x{kW} s={stride} p={padding}");
    }

    [Fact]
    public void DcganL2Shape_FatASmallNFastPath_BeatsOpenBlasBudget()
    {
        // Issue #358 phase-2 perf gate: the DCGAN L2 ConvTranspose2D shape
        // [B=1, Cin=512, 4, 4] → [Co=256, 8, 8] hits a documented MKL/OpenBLAS
        // perf cliff at M=4096, N=16, K=512, transA=true (215 ms OpenBLAS /
        // 559 ms MKL vs ~1 ms theoretical peak). The packed-A + Mc-blocked
        // AVX2 kernel must come in well under the OpenBLAS budget.
        //
        // Budget = 50 ms — generous over the expected ~3-10 ms steady-state
        // on a 4-core AVX2 runner with code coverage instrumentation, but
        // still 4-10x under the pre-fix OpenBLAS measurement so any
        // regression that re-introduces the cliff (e.g., disabling the
        // dispatch gate, or breaking the packed layout) trips the test.
        const int batch = 1, inChannels = 512, inH = 4, inW = 4;
        const int outChannels = 256, kH = 4, kW = 4;
        const int stride = 2, padding = 1;
        int outH = (inH - 1) * stride - 2 * padding + kH;
        int outW = (inW - 1) * stride - 2 * padding + kW;

        var inputT = MakeRandomTensor(new[] { batch, inChannels, inH, inW }, seed: 7);
        var kernelT = MakeRandomTensor(new[] { inChannels, outChannels, kH, kW }, seed: 11);
        var inputArr = inputT.Data.ToArray();
        var kernelArr = kernelT.Data.ToArray();
        var actual = new double[batch * outChannels * outH * outW];

        Im2ColHelper._smallNTransAAvx2Calls = 0;
        Im2ColHelper._smallNTransAScalarCalls = 0;

        // Warmup — first call pays JIT + ArrayPool warmup.
        Im2ColHelper.TryConvTranspose2DWithGemm(
            inputArr, kernelArr, actual,
            batch, inChannels, inH, inW, outChannels, kH, kW,
            stride, stride, padding, padding, outH, outW);
        Assert.True(Im2ColHelper._smallNTransAAvx2Calls > 0 || Im2ColHelper._smallNTransAScalarCalls > 0,
            "Phase-2 dispatch did not fire on warmup — gate (hw == 16 && kmkn >= 8*inChannels) " +
            "may not match L2 shape.");

        // Two additional warmups to settle thermal / branch-predictor /
        // ArrayPool state. The first warmup-after-JIT can still pay for
        // the pool's slab allocation; the second + third let the kernel
        // run cleanly on the hot data path.
        for (int w = 0; w < 2; w++)
        {
            Im2ColHelper.TryConvTranspose2DWithGemm(
                inputArr, kernelArr, actual,
                batch, inChannels, inH, inW, outChannels, kH, kW,
                stride, stride, padding, padding, outH, outW);
        }

        const int iters = 10;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            Im2ColHelper.TryConvTranspose2DWithGemm(
                inputArr, kernelArr, actual,
                batch, inChannels, inH, inW, outChannels, kH, kW,
                stride, stride, padding, padding, outH, outW);
        }
        sw.Stop();
        double msPerCall = sw.Elapsed.TotalMilliseconds / iters;

        Assert.True(msPerCall < 50.0,
            $"L2 ConvTranspose2D took {msPerCall:F1} ms/call — exceeds 50 ms phase-2 budget. " +
            $"AVX2 calls: {Im2ColHelper._smallNTransAAvx2Calls}, scalar calls: {Im2ColHelper._smallNTransAScalarCalls}. " +
            "Pre-fix OpenBLAS measured 215 ms / MKL 559 ms at this shape. Phase-2 packed-A " +
            "kernel should produce well under the 50 ms budget with the BLIS-style Mc=64, Mr=2 blocking.");
    }

    // Same shape suite for float — float has tighter precision tolerance
    // and BLAS path uses SGEMM whose accumulation order differs from naive.
    [Theory]
    [InlineData(1, 512, 4, 4, 256, 4, 4, 2, 1)]
    [InlineData(1, 256, 8, 8, 128, 4, 4, 2, 1)]
    [InlineData(2, 4, 4, 4, 8, 4, 4, 2, 1)]
    [InlineData(1, 8, 4, 4, 8, 2, 2, 2, 0)]
    public void GemmPath_MatchesNaiveReference_Float(
        int batch, int inChannels, int inH, int inW,
        int outChannels, int kH, int kW, int stride, int padding)
    {
        var rng = new Random(42);
        int outH = (inH - 1) * stride - 2 * padding + kH;
        int outW = (inW - 1) * stride - 2 * padding + kW;

        var inputArr = new float[batch * inChannels * inH * inW];
        var kernelArr = new float[inChannels * outChannels * kH * kW];
        for (int i = 0; i < inputArr.Length; i++) inputArr[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kernelArr.Length; i++) kernelArr[i] = (float)(rng.NextDouble() * 2 - 1);

        // Naive reference (compute in double for higher reference precision)
        var inputD = new double[inputArr.Length];
        var kernelD = new double[kernelArr.Length];
        for (int i = 0; i < inputArr.Length; i++) inputD[i] = inputArr[i];
        for (int i = 0; i < kernelArr.Length; i++) kernelD[i] = kernelArr[i];
        var expectedD = NaiveConvTranspose2DDouble(
            inputD, kernelD,
            batch, inChannels, inH, inW,
            outChannels, kH, kW,
            stride, stride, padding, padding,
            outH, outW);

        var actual = new float[batch * outChannels * outH * outW];
        bool used = Im2ColHelper.TryConvTranspose2DWithGemm(
            inputArr, kernelArr, actual,
            batch, inChannels, inH, inW,
            outChannels, kH, kW,
            stride, stride, padding, padding,
            outH, outW);
        if (!used) return;

        double maxDiff = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double d = Math.Abs(actual[i] - expectedD[i]);
            if (d > maxDiff) maxDiff = d;
        }
        // SGEMM accumulation in float has wider rounding than double; bound by
        // n_reductions × machine_epsilon × max(magnitude).
        Assert.True(maxDiff < 1e-3,
            $"GEMM vs naive float ConvTranspose2D drift: maxDiff={maxDiff:E3}");
    }
}
