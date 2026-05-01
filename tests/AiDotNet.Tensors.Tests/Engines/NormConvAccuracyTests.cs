using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Numerical-accuracy regression tests for LayerNorm, BatchNorm, and Conv2D.
/// These tests compute the operation in scalar double-precision as the
/// reference and compare the float SIMD output element-by-element at a
/// tight relative tolerance. They guard the upcoming #209 close-parity
/// kernel rewrites (register-resident LayerNorm, output-channel-blocked
/// Conv2D, etc.) — every micro-change has to keep these green.
///
/// Tolerance choice: 1e-5 absolute + 1e-4 relative. Picked to allow
/// FMA-order rearrangements (which can change the lowest-bit float
/// rounding by ~1 ULP) but reject any algorithmic bug.
/// </summary>
public class NormConvAccuracyTests
{
    private readonly CpuEngine E = new();

    // Tighter than MathInvariantTests' 1e-4 since these are point-by-point
    // checks against double-precision references, not algebraic invariants.
    private const float AbsTol = 1e-5f;
    private const float RelTol = 1e-4f;

    private static float[] DeterministicData(int count, int seed)
    {
        var rng = new Random(seed);
        var d = new float[count];
        for (int i = 0; i < count; i++)
            d[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return d;
    }

    // Hot-path equality check called millions of times by the per-element
    // loops. Build the failure string only on mismatch — eager interpolation
    // would allocate one string per element (a 32k×64 LayerNorm = 2M strings).
    private static void AssertCloseFloat(float expected, float actual, string ctx)
    {
        float diff = Math.Abs(expected - actual);
        float scale = Math.Max(1f, Math.Max(Math.Abs(expected), Math.Abs(actual)));
        if (diff > AbsTol + RelTol * scale)
        {
            Assert.Fail($"{ctx}: expected={expected:G9}, actual={actual:G9}, diff={diff:G9}");
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // LayerNorm — normalizes over the trailing dims that match gamma's shape.
    // Reference: scalar double-precision, two-pass (mean → centered variance).
    // ─────────────────────────────────────────────────────────────────

    private static void LayerNormReference(
        float[] input, float[] gamma, float[] beta,
        int batchSize, int featureSize, double epsilon,
        float[] outputRef)
    {
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * featureSize;
            // Pass 1: mean (double accumulator)
            double sum = 0;
            for (int f = 0; f < featureSize; f++) sum += input[off + f];
            double mean = sum / featureSize;

            // Pass 2: centered variance (double accumulator)
            double sumSq = 0;
            for (int f = 0; f < featureSize; f++)
            {
                double d = input[off + f] - mean;
                sumSq += d * d;
            }
            double variance = sumSq / featureSize;
            double invStd = 1.0 / Math.Sqrt(variance + epsilon);

            // Pass 3: normalize + affine
            for (int f = 0; f < featureSize; f++)
            {
                double normalized = (input[off + f] - mean) * invStd;
                outputRef[off + f] = (float)(gamma[f] * normalized + beta[f]);
            }
        }
    }

    [Theory]
    [InlineData(32768, 64)]   // BDN benchmark shape
    [InlineData(1, 768)]      // BERT [1, 768]
    [InlineData(32, 256)]     // medium transformer
    [InlineData(8, 4096)]     // LLaMA-style large hidden
    [InlineData(1, 1)]        // edge: 1-element row
    [InlineData(1, 7)]        // edge: prime feature size (tail handling)
    public void LayerNorm_FloatMatchesDoubleReference(int batchSize, int featureSize)
    {
        var inputData = DeterministicData(batchSize * featureSize, seed: 13_001);
        var gammaData = DeterministicData(featureSize, seed: 13_002);
        var betaData  = DeterministicData(featureSize, seed: 13_003);

        var input = new Tensor<float>(inputData, new[] { batchSize, featureSize });
        var gamma = new Tensor<float>(gammaData, new[] { featureSize });
        var beta  = new Tensor<float>(betaData,  new[] { featureSize });

        var actualResult = E.LayerNorm(input, gamma, beta, 1e-5, out _, out _);
        var actual = actualResult.GetDataArray();

        var expected = new float[batchSize * featureSize];
        LayerNormReference(inputData, gammaData, betaData, batchSize, featureSize, 1e-5, expected);

        for (int b = 0; b < batchSize; b++)
            for (int f = 0; f < featureSize; f++)
            {
                int idx = b * featureSize + f;
                AssertCloseFloat(expected[idx], actual[idx], $"[b={b}, f={f}]");
            }
    }

    [Fact]
    public void LayerNorm_GammaBetaApplied_AffineCheck()
    {
        // gamma=2, beta=3 → output_i = 2 * normalized_i + 3.
        // After subtracting beta and dividing by gamma we should recover
        // the canonical (mean=0, var=1) normalized values.
        const int batch = 4, fs = 16;
        var inputData = DeterministicData(batch * fs, seed: 13_010);
        var gammaData = Enumerable.Repeat(2f, fs).ToArray();
        var betaData  = Enumerable.Repeat(3f, fs).ToArray();

        var input = new Tensor<float>(inputData, new[] { batch, fs });
        var gamma = new Tensor<float>(gammaData, new[] { fs });
        var beta  = new Tensor<float>(betaData,  new[] { fs });

        var output = E.LayerNorm(input, gamma, beta, 1e-5, out _, out _).GetDataArray();
        for (int b = 0; b < batch; b++)
        {
            double sum = 0, sumSq = 0;
            for (int f = 0; f < fs; f++)
            {
                double n = (output[b * fs + f] - 3.0) / 2.0; // recover normalized
                sum += n;
                sumSq += n * n;
            }
            double mean = sum / fs;
            double variance = sumSq / fs;
            Assert.True(Math.Abs(mean) < 1e-4, $"row {b}: mean={mean:G9} (expected 0)");
            Assert.True(Math.Abs(variance - 1.0) < 1e-3, $"row {b}: var={variance:G9} (expected 1)");
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // BatchNorm 4D — normalizes per channel across (batch, height, width).
    // Reference: scalar double-precision per channel.
    // ─────────────────────────────────────────────────────────────────

    private static void BatchNorm4DReference(
        float[] input, float[] gamma, float[] beta,
        int batch, int channels, int height, int width, double epsilon,
        float[] outputRef)
    {
        int spatial = height * width;
        int elementsPerChannel = batch * spatial;
        for (int c = 0; c < channels; c++)
        {
            // Mean
            double sum = 0;
            for (int n = 0; n < batch; n++)
            {
                int off = n * channels * spatial + c * spatial;
                for (int s = 0; s < spatial; s++) sum += input[off + s];
            }
            double mean = sum / elementsPerChannel;

            // Variance
            double sumSq = 0;
            for (int n = 0; n < batch; n++)
            {
                int off = n * channels * spatial + c * spatial;
                for (int s = 0; s < spatial; s++)
                {
                    double d = input[off + s] - mean;
                    sumSq += d * d;
                }
            }
            double variance = sumSq / elementsPerChannel;
            double invStd = 1.0 / Math.Sqrt(variance + epsilon);

            // Normalize + affine
            for (int n = 0; n < batch; n++)
            {
                int off = n * channels * spatial + c * spatial;
                for (int s = 0; s < spatial; s++)
                {
                    double normalized = (input[off + s] - mean) * invStd;
                    outputRef[off + s] = (float)(gamma[c] * normalized + beta[c]);
                }
            }
        }
    }

    [Theory]
    [InlineData(32, 64, 32, 32)]  // BDN benchmark shape
    [InlineData(1, 16, 8, 8)]     // small
    [InlineData(2, 3, 4, 4)]      // tiny edge
    [InlineData(8, 32, 16, 16)]   // medium
    public void BatchNorm4D_FloatMatchesDoubleReference(int batch, int channels, int height, int width)
    {
        int total = batch * channels * height * width;
        var inputData = DeterministicData(total, seed: 14_001);
        var gammaData = DeterministicData(channels, seed: 14_002);
        var betaData  = DeterministicData(channels, seed: 14_003);

        var input = new Tensor<float>(inputData, new[] { batch, channels, height, width });
        var gamma = new Tensor<float>(gammaData, new[] { channels });
        var beta  = new Tensor<float>(betaData,  new[] { channels });

        var actualResult = E.BatchNorm(input, gamma, beta, 1e-5, out _, out _);
        var actual = actualResult.GetDataArray();

        var expected = new float[total];
        BatchNorm4DReference(inputData, gammaData, betaData, batch, channels, height, width, 1e-5, expected);

        for (int i = 0; i < total; i++)
            AssertCloseFloat(expected[i], actual[i], $"i={i}");
    }

    // ─────────────────────────────────────────────────────────────────
    // Conv2D 3×3 stride=1 — reference: scalar nested loops.
    // ─────────────────────────────────────────────────────────────────

    private static void Conv2DReference(
        float[] input, float[] kernel,
        int batch, int inChannels, int height, int width,
        int outChannels, int kernelH, int kernelW,
        int stride, int padding, int dilation,
        int outHeight, int outWidth,
        float[] outputRef)
    {
        int effKernelH = dilation * (kernelH - 1) + 1;
        int effKernelW = dilation * (kernelW - 1) + 1;
        for (int b = 0; b < batch; b++)
        for (int oc = 0; oc < outChannels; oc++)
        for (int oh = 0; oh < outHeight; oh++)
        for (int ow = 0; ow < outWidth; ow++)
        {
            double acc = 0;
            for (int ic = 0; ic < inChannels; ic++)
            for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++)
            {
                int ih = oh * stride + kh * dilation - padding;
                int iw = ow * stride + kw * dilation - padding;
                if (ih < 0 || ih >= height || iw < 0 || iw >= width) continue;
                int inIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                int kIdx = ((oc * inChannels + ic) * kernelH + kh) * kernelW + kw;
                acc += (double)input[inIdx] * kernel[kIdx];
            }
            int outIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
            outputRef[outIdx] = (float)acc;
        }
    }

    [Theory]
    [InlineData(1, 16, 64, 64, 32, 3, 1, 1, 1)]   // BDN benchmark shape
    [InlineData(1, 3, 32, 32, 8, 3, 1, 1, 1)]     // small ResNet
    [InlineData(2, 4, 16, 16, 6, 3, 1, 1, 1)]     // batched
    [InlineData(1, 1, 8, 8, 1, 3, 1, 1, 1)]       // single-channel edge
    [InlineData(1, 4, 16, 16, 4, 1, 1, 0, 1)]     // 1×1 conv (different fast path)
    public void Conv2D_FloatMatchesDoubleReference(
        int batch, int inChannels, int height, int width,
        int outChannels, int kernelSize, int stride, int padding, int dilation)
    {
        int effKernel = dilation * (kernelSize - 1) + 1;
        int outHeight = (height + 2 * padding - effKernel) / stride + 1;
        int outWidth  = (width  + 2 * padding - effKernel) / stride + 1;

        var inputData = DeterministicData(batch * inChannels * height * width, seed: 15_001);
        var kernelData = DeterministicData(outChannels * inChannels * kernelSize * kernelSize, seed: 15_002);

        var input = new Tensor<float>(inputData, new[] { batch, inChannels, height, width });
        var kernel = new Tensor<float>(kernelData, new[] { outChannels, inChannels, kernelSize, kernelSize });

        var actualResult = E.Conv2D(input, kernel, stride, padding, dilation);
        var actual = actualResult.GetDataArray();

        var expected = new float[batch * outChannels * outHeight * outWidth];
        Conv2DReference(inputData, kernelData,
            batch, inChannels, height, width,
            outChannels, kernelSize, kernelSize,
            stride, padding, dilation,
            outHeight, outWidth, expected);

        for (int i = 0; i < expected.Length; i++)
        {
            // Conv2D accumulates more terms than LayerNorm/BatchNorm so the
            // tolerance loosens slightly (more rounding error compounding).
            // Build the failure message only on mismatch (lazy) — see
            // AssertCloseFloat for rationale.
            float diff = Math.Abs(expected[i] - actual[i]);
            float scale = Math.Max(1f, Math.Max(Math.Abs(expected[i]), Math.Abs(actual[i])));
            if (diff > 1e-4f + 1e-3f * scale)
            {
                Assert.Fail($"i={i}: expected={expected[i]:G9}, actual={actual[i]:G9}, diff={diff:G9}");
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Idempotence + determinism: same inputs must always produce same outputs.
    // Catches race conditions in the parallel dispatch.
    // ─────────────────────────────────────────────────────────────────

    [Fact]
    public void LayerNorm_IsDeterministic_AcrossManyCalls()
    {
        var x = new Tensor<float>(DeterministicData(32 * 64, 16_001), new[] { 32, 64 });
        var g = new Tensor<float>(DeterministicData(64, 16_002), new[] { 64 });
        var b = new Tensor<float>(DeterministicData(64, 16_003), new[] { 64 });
        var first = E.LayerNorm(x, g, b, 1e-5, out _, out _).GetDataArray();
        for (int trial = 0; trial < 10; trial++)
        {
            var r = E.LayerNorm(x, g, b, 1e-5, out _, out _).GetDataArray();
            for (int i = 0; i < first.Length; i++)
                Assert.Equal(first[i], r[i]);
        }
    }

    [Fact]
    public void BatchNorm_IsDeterministic_AcrossManyCalls()
    {
        var x = new Tensor<float>(DeterministicData(4 * 8 * 4 * 4, 17_001), new[] { 4, 8, 4, 4 });
        var g = new Tensor<float>(DeterministicData(8, 17_002), new[] { 8 });
        var b = new Tensor<float>(DeterministicData(8, 17_003), new[] { 8 });
        var first = E.BatchNorm(x, g, b, 1e-5, out _, out _).GetDataArray();
        for (int trial = 0; trial < 10; trial++)
        {
            var r = E.BatchNorm(x, g, b, 1e-5, out _, out _).GetDataArray();
            for (int i = 0; i < first.Length; i++)
                Assert.Equal(first[i], r[i]);
        }
    }

    [Fact]
    public void Conv2D_IsDeterministic_AcrossManyCalls()
    {
        var x = new Tensor<float>(DeterministicData(1 * 4 * 8 * 8, 18_001), new[] { 1, 4, 8, 8 });
        var k = new Tensor<float>(DeterministicData(8 * 4 * 3 * 3, 18_002), new[] { 8, 4, 3, 3 });
        var first = E.Conv2D(x, k, 1, 1, 1).GetDataArray();
        for (int trial = 0; trial < 10; trial++)
        {
            var r = E.Conv2D(x, k, 1, 1, 1).GetDataArray();
            for (int i = 0; i < first.Length; i++)
                Assert.Equal(first[i], r[i]);
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Softmax (double) — guards the new SoftmaxRowDoubleUnsafe SIMD kernel.
    // Reference: scalar double-precision max → exp(x-max) → divide.
    // ─────────────────────────────────────────────────────────────────

    private static double[] DeterministicDoubleData(int count, int seed)
    {
        var rng = new Random(seed);
        var d = new double[count];
        for (int i = 0; i < count; i++) d[i] = rng.NextDouble() * 4.0 - 2.0; // [-2, 2]
        return d;
    }

    private static void SoftmaxDoubleReference(double[] input, int rows, int cols, double[] output)
    {
        for (int r = 0; r < rows; r++)
        {
            int off = r * cols;
            double maxVal = double.NegativeInfinity;
            for (int j = 0; j < cols; j++) if (input[off + j] > maxVal) maxVal = input[off + j];
            double sumExp = 0;
            for (int j = 0; j < cols; j++) { double e = Math.Exp(input[off + j] - maxVal); output[off + j] = e; sumExp += e; }
            if (sumExp == 0.0) continue;
            double invSum = 1.0 / sumExp;
            for (int j = 0; j < cols; j++) output[off + j] *= invSum;
        }
    }

    [Theory]
    [InlineData(512, 1024)]   // BDN benchmark shape
    [InlineData(1, 16)]       // edge: 1 row
    [InlineData(1, 1)]        // edge: 1×1 (sums to 1, single output is 1.0)
    [InlineData(4, 13)]       // edge: prime axis size (tail handling)
    [InlineData(64, 256)]     // medium
    public void Softmax_Double_FloatMatchesReference(int rows, int cols)
    {
        var inputData = DeterministicDoubleData(rows * cols, seed: 19_001);
        var input = new Tensor<double>(inputData, new[] { rows, cols });
        var actualResult = E.Softmax(input, axis: -1);
        var actual = actualResult.GetDataArray();

        var expected = new double[rows * cols];
        SoftmaxDoubleReference(inputData, rows, cols, expected);

        // Tighter tolerance for double-precision reference (1e-12 absolute, 1e-10 relative)
        // since FastExpDouble256 is accurate to ~1e-12 ULP per element.
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(expected[i] - actual[i]);
            double scale = Math.Max(1.0, Math.Max(Math.Abs(expected[i]), Math.Abs(actual[i])));
            if (diff > 1e-12 + 1e-10 * scale)
            {
                Assert.Fail($"i={i}: expected={expected[i]:G17}, actual={actual[i]:G17}, diff={diff:G17}");
            }
        }

        // Sanity: each row sums to 1.0 (within numerical precision)
        for (int r = 0; r < rows; r++)
        {
            double rowSum = 0;
            for (int j = 0; j < cols; j++) rowSum += actual[r * cols + j];
            Assert.True(Math.Abs(rowSum - 1.0) < 1e-10, $"row {r}: sum={rowSum:G17}");
        }
    }
}
