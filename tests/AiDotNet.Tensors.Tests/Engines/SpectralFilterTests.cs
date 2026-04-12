using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tests for NativeSpectralFilter and NativeSpectralFilterBatch.
/// Covers Issue #150: fused spectral filter ops.
/// Verifies correctness against the manual FFT2D → multiply → IFFT2D pipeline,
/// plus mathematical invariants (identity filter, zero filter, linearity).
/// </summary>
public class SpectralFilterTests
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = new CpuEngine();

    public SpectralFilterTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // ================================================================
    // NativeSpectralFilter: correctness vs manual pipeline
    // ================================================================

    [Theory]
    [InlineData(8, 8)]
    [InlineData(16, 16)]
    [InlineData(8, 16)]
    public void SpectralFilter_MatchesManualPipeline(int h, int w)
    {
        var rng = new Random(42);
        var input = new Tensor<double>([h, w]);
        for (int i = 0; i < h * w; i++) input[i] = rng.NextDouble() * 2 - 1;

        // Random complex filter
        var filter = new Tensor<Complex<double>>([h, w]);
        for (int i = 0; i < h * w; i++)
            filter[i] = new Complex<double>(rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1);

        // Manual pipeline: FFT2D → multiply → IFFT2D
        var spectrum = _engine.NativeComplexFFT2D(input);
        var filtered = _engine.NativeComplexMultiply(spectrum, filter);
        var expected = _engine.NativeComplexIFFT2DReal(filtered);

        // Fused single call
        var actual = _engine.NativeSpectralFilter(input, filter);

        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(expected[i] - actual[i]);
            Assert.True(diff < 1e-10,
                $"SpectralFilter [{h},{w}] mismatch at [{i}]: expected={expected[i]:F10}, actual={actual[i]:F10}, diff={diff:E2}");
        }
    }

    [Fact]
    public void SpectralFilter_IdentityFilter_PreservesInput()
    {
        // All-ones filter in frequency domain should preserve the signal
        int h = 8, w = 8;
        var rng = new Random(123);
        var input = new Tensor<double>([h, w]);
        for (int i = 0; i < h * w; i++) input[i] = rng.NextDouble() * 2 - 1;

        // Identity filter: (1+0i) everywhere
        var filter = new Tensor<Complex<double>>([h, w]);
        for (int i = 0; i < h * w; i++)
            filter[i] = new Complex<double>(1.0, 0.0);

        var result = _engine.NativeSpectralFilter(input, filter);

        for (int i = 0; i < input.Length; i++)
        {
            double diff = Math.Abs(input[i] - result[i]);
            Assert.True(diff < 1e-10,
                $"Identity filter should preserve input. Mismatch at [{i}]: input={input[i]:F10}, result={result[i]:F10}");
        }
    }

    [Fact]
    public void SpectralFilter_ZeroFilter_ProducesZeros()
    {
        int h = 8, w = 8;
        var rng = new Random(456);
        var input = new Tensor<double>([h, w]);
        for (int i = 0; i < h * w; i++) input[i] = rng.NextDouble() * 2 - 1;

        // Zero filter
        var filter = new Tensor<Complex<double>>([h, w]);

        var result = _engine.NativeSpectralFilter(input, filter);

        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(Math.Abs(result[i]) < 1e-10,
                $"Zero filter should produce zeros. Got {result[i]:E2} at [{i}]");
        }
    }

    [Fact]
    public void SpectralFilter_Linearity_ScaledFilterScalesOutput()
    {
        // SpectralFilter(input, 2*filter) should equal 2 * SpectralFilter(input, filter)
        int h = 8, w = 8;
        var rng = new Random(789);
        var input = new Tensor<double>([h, w]);
        for (int i = 0; i < h * w; i++) input[i] = rng.NextDouble() * 2 - 1;

        var filter1 = new Tensor<Complex<double>>([h, w]);
        for (int i = 0; i < h * w; i++)
            filter1[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        var filter2 = new Tensor<Complex<double>>([h, w]);
        for (int i = 0; i < h * w; i++)
            filter2[i] = new Complex<double>(filter1[i].Real * 2, filter1[i].Imaginary * 2);

        var result1 = _engine.NativeSpectralFilter(input, filter1);
        var result2 = _engine.NativeSpectralFilter(input, filter2);

        for (int i = 0; i < result1.Length; i++)
        {
            double diff = Math.Abs(result1[i] * 2 - result2[i]);
            Assert.True(diff < 1e-8,
                $"Linearity violated at [{i}]: 2*result1={result1[i] * 2:F10}, result2={result2[i]:F10}");
        }
    }

    [Fact]
    public void SpectralFilter_BatchedInput_BroadcastsFilter()
    {
        // [B, H, W] input with [H, W] filter should work
        int b = 3, h = 8, w = 8;
        var rng = new Random(101);
        var input = new Tensor<double>([b, h, w]);
        for (int i = 0; i < b * h * w; i++) input[i] = rng.NextDouble() * 2 - 1;

        var filter = new Tensor<Complex<double>>([h, w]);
        for (int i = 0; i < h * w; i++)
            filter[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        var result = _engine.NativeSpectralFilter(input, filter);
        Assert.Equal(new[] { b, h, w }, result.Shape.ToArray());

        // Verify each batch slice matches single-slice result
        for (int bi = 0; bi < b; bi++)
        {
            var singleSlice = new Tensor<double>([h, w]);
            for (int i = 0; i < h * w; i++)
                singleSlice[i] = input[bi * h * w + i];

            var singleResult = _engine.NativeSpectralFilter(singleSlice, filter);

            for (int i = 0; i < h * w; i++)
            {
                double diff = Math.Abs(result[bi * h * w + i] - singleResult[i]);
                Assert.True(diff < 1e-8,
                    $"Batch [{bi}] mismatch at [{i}]: batched={result[bi * h * w + i]:F10}, single={singleResult[i]:F10}");
            }
        }
    }

    // ================================================================
    // NativeSpectralFilterBatch: correctness
    // ================================================================

    [Fact]
    public void SpectralFilterBatch_SharedFilter_MatchesPerSlice()
    {
        // [B, C, H, W] with shared [H, W] filter
        int batch = 2, channels = 3, h = 8, w = 8;
        var rng = new Random(42);
        var input = new Tensor<double>([batch, channels, h, w]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;

        var filter = new Tensor<Complex<double>>([h, w]);
        for (int i = 0; i < h * w; i++)
            filter[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        var result = _engine.NativeSpectralFilterBatch(input, filter);
        Assert.Equal(new[] { batch, channels, h, w }, result.Shape.ToArray());

        // Verify each (b,c) slice matches single-slice spectral filter
        int sliceSize = h * w;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                var singleSlice = new Tensor<double>([h, w]);
                int srcOff = (b * channels + c) * sliceSize;
                for (int i = 0; i < sliceSize; i++)
                    singleSlice[i] = input[srcOff + i];

                var singleResult = _engine.NativeSpectralFilter(singleSlice, filter);

                for (int i = 0; i < sliceSize; i++)
                {
                    double diff = Math.Abs(result[srcOff + i] - singleResult[i]);
                    Assert.True(diff < 1e-8,
                        $"Batch shared filter (b={b},c={c}) mismatch at [{i}]: " +
                        $"batched={result[srcOff + i]:F10}, single={singleResult[i]:F10}");
                }
            }
        }
    }

    [Fact]
    public void SpectralFilterBatch_PerChannelFilter_MatchesPerSlice()
    {
        // [B, C, H, W] with per-channel [C, H, W] filter
        int batch = 2, channels = 4, h = 8, w = 8;
        var rng = new Random(99);
        var input = new Tensor<double>([batch, channels, h, w]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;

        var filter = new Tensor<Complex<double>>([channels, h, w]);
        for (int i = 0; i < filter.Length; i++)
            filter[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        var result = _engine.NativeSpectralFilterBatch(input, filter);
        Assert.Equal(new[] { batch, channels, h, w }, result.Shape.ToArray());

        // Verify each (b,c) slice matches single-slice spectral filter with channel-specific filter
        int sliceSize = h * w;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                var singleSlice = new Tensor<double>([h, w]);
                int srcOff = (b * channels + c) * sliceSize;
                for (int i = 0; i < sliceSize; i++)
                    singleSlice[i] = input[srcOff + i];

                // Extract per-channel filter
                var channelFilter = new Tensor<Complex<double>>([h, w]);
                int filterOff = c * sliceSize;
                for (int i = 0; i < sliceSize; i++)
                    channelFilter[i] = filter[filterOff + i];

                var singleResult = _engine.NativeSpectralFilter(singleSlice, channelFilter);

                for (int i = 0; i < sliceSize; i++)
                {
                    double diff = Math.Abs(result[srcOff + i] - singleResult[i]);
                    Assert.True(diff < 1e-8,
                        $"Batch per-channel filter (b={b},c={c}) mismatch at [{i}]: " +
                        $"batched={result[srcOff + i]:F10}, single={singleResult[i]:F10}");
                }
            }
        }
    }

    [Fact]
    public void SpectralFilterBatch_IdentityFilter_PreservesInput()
    {
        int batch = 2, channels = 3, h = 8, w = 8;
        var rng = new Random(55);
        var input = new Tensor<double>([batch, channels, h, w]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;

        var filter = new Tensor<Complex<double>>([h, w]);
        for (int i = 0; i < h * w; i++)
            filter[i] = new Complex<double>(1.0, 0.0);

        var result = _engine.NativeSpectralFilterBatch(input, filter);

        for (int i = 0; i < input.Length; i++)
        {
            double diff = Math.Abs(input[i] - result[i]);
            Assert.True(diff < 1e-8,
                $"Identity filter should preserve input. Mismatch at [{i}]: input={input[i]:F10}, result={result[i]:F10}");
        }
    }

    [Fact]
    public void SpectralFilterBatch_ZeroFilter_ProducesZeros()
    {
        int batch = 2, channels = 3, h = 8, w = 8;
        var rng = new Random(77);
        var input = new Tensor<double>([batch, channels, h, w]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;

        var filter = new Tensor<Complex<double>>([h, w]);

        var result = _engine.NativeSpectralFilterBatch(input, filter);

        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(Math.Abs(result[i]) < 1e-10,
                $"Zero filter should produce zeros. Got {result[i]:E2} at [{i}]");
        }
    }

    // ================================================================
    // Validation tests
    // ================================================================

    [Fact]
    public void SpectralFilter_ThrowsOnMismatchedFilterDims()
    {
        var input = new Tensor<double>([8, 8]);
        var wrongFilter = new Tensor<Complex<double>>([4, 4]);

        Assert.Throws<ArgumentException>(() => _engine.NativeSpectralFilter(input, wrongFilter));
    }

    [Fact]
    public void SpectralFilter_Rank3Filter_BroadcastsPerChannel()
    {
        // [C,H,W] filter with [B,C,H,W] input should broadcast per-channel across batches
        int batch = 2, channels = 3, h = 8, w = 8;
        var rng = new Random(42);
        var input = new Tensor<double>([batch, channels, h, w]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;

        var filter = new Tensor<Complex<double>>([channels, h, w]);
        for (int i = 0; i < filter.Length; i++)
            filter[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        var result = _engine.NativeSpectralFilter(input, filter);
        Assert.Equal(new[] { batch, channels, h, w }, result.Shape.ToArray());

        // Verify each (b,c) slice matches single-slice spectral filter with the channel's filter
        int sliceSize = h * w;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                var singleSlice = new Tensor<double>([h, w]);
                int srcOff = (b * channels + c) * sliceSize;
                for (int i = 0; i < sliceSize; i++) singleSlice[i] = input[srcOff + i];

                var channelFilter = new Tensor<Complex<double>>([h, w]);
                int fOff = c * sliceSize;
                for (int i = 0; i < sliceSize; i++) channelFilter[i] = filter[fOff + i];

                var singleResult = _engine.NativeSpectralFilter(singleSlice, channelFilter);
                for (int i = 0; i < sliceSize; i++)
                {
                    double diff = Math.Abs(result[srcOff + i] - singleResult[i]);
                    Assert.True(diff < 1e-8,
                        $"Rank-3 filter (b={b},c={c}) mismatch at [{i}]: " +
                        $"batched={result[srcOff + i]:F10}, single={singleResult[i]:F10}");
                }
            }
        }
    }

    [Fact]
    public void SpectralFilter_FullRankFilter_DirectMultiply()
    {
        // [B,C,H,W] filter with [B,C,H,W] input — no broadcast, direct 1:1 multiply
        int batch = 2, channels = 3, h = 8, w = 8;
        var rng = new Random(99);
        var input = new Tensor<double>([batch, channels, h, w]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;

        var filter = new Tensor<Complex<double>>([batch, channels, h, w]);
        for (int i = 0; i < filter.Length; i++)
            filter[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        var result = _engine.NativeSpectralFilter(input, filter);
        Assert.Equal(new[] { batch, channels, h, w }, result.Shape.ToArray());

        // Verify against manual pipeline
        var spectrum = _engine.NativeComplexFFT2D(input);
        var filtered = _engine.NativeComplexMultiply(spectrum, filter);
        var expected = _engine.NativeComplexIFFT2DReal(filtered);

        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(expected[i] - result[i]);
            Assert.True(diff < 1e-8,
                $"Full-rank filter mismatch at [{i}]: expected={expected[i]:F10}, actual={result[i]:F10}");
        }
    }

    [Fact]
    public void SpectralFilter_ThrowsOnFilterLargerThanInput()
    {
        var input = new Tensor<double>([8, 8]);
        var bigFilter = new Tensor<Complex<double>>([2, 8, 8]); // filter larger than input

        Assert.Throws<ArgumentException>(() => _engine.NativeSpectralFilter(input, bigFilter));
    }

    [Fact]
    public void SpectralFilterBatch_ThrowsOnNon4DInput()
    {
        var input = new Tensor<double>([8, 8]);
        var filter = new Tensor<Complex<double>>([8, 8]);

        Assert.Throws<ArgumentException>(() => _engine.NativeSpectralFilterBatch(input, filter));
    }

    [Fact]
    public void SpectralFilterBatch_ThrowsOnFilterLargerThanInput()
    {
        var input = new Tensor<double>([2, 3, 8, 8]);
        var bigFilter = new Tensor<Complex<double>>([4, 3, 8, 8]); // larger than input

        Assert.Throws<ArgumentException>(() => _engine.NativeSpectralFilterBatch(input, bigFilter));
    }

    [Fact]
    public void SpectralFilterBatch_Rank4Filter_DirectMultiply()
    {
        // Full-rank [B,C,H,W] filter — no broadcast, direct 1:1 match
        int batch = 2, channels = 3, h = 8, w = 8;
        var rng = new Random(77);
        var input = new Tensor<double>([batch, channels, h, w]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;

        var filter = new Tensor<Complex<double>>([batch, channels, h, w]);
        for (int i = 0; i < filter.Length; i++)
            filter[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        var result = _engine.NativeSpectralFilterBatch(input, filter);
        Assert.Equal(new[] { batch, channels, h, w }, result.Shape.ToArray());

        // Verify against manual pipeline
        var spectrum = _engine.NativeComplexFFT2D(input);
        var filtered = _engine.NativeComplexMultiply(spectrum, filter);
        var expected = _engine.NativeComplexIFFT2DReal(filtered);

        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(expected[i] - result[i]);
            Assert.True(diff < 1e-8,
                $"Rank-4 batch filter mismatch at [{i}]: expected={expected[i]:F10}, actual={result[i]:F10}");
        }
    }

    // ================================================================
    // Performance: fused vs manual pipeline
    // ================================================================

    [Theory]
    [Trait("Category", "Performance")]
    [InlineData(16, 4, 32, 32)]
    public void SpectralFilterBatch_FasterThanManualLoop(int batch, int channels, int h, int w)
    {
        var rng = new Random(42);
        var input = new Tensor<double>([batch, channels, h, w]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;

        var filter = new Tensor<Complex<double>>([channels, h, w]);
        for (int i = 0; i < filter.Length; i++)
            filter[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        int sliceSize = h * w;

        // Pre-allocate reusable buffers (not counted in timing)
        var sliceBuf = new Tensor<double>([h, w]);
        var channelFilters = new Tensor<Complex<double>>[channels];
        for (int c = 0; c < channels; c++)
        {
            channelFilters[c] = new Tensor<Complex<double>>([h, w]);
            int fOff = c * sliceSize;
            for (int i = 0; i < sliceSize; i++) channelFilters[c][i] = filter[fOff + i];
        }

        // Warmup both paths
        _engine.NativeSpectralFilterBatch(input, filter);
        for (int i = 0; i < sliceSize; i++) sliceBuf[i] = input[i];
        _engine.NativeSpectralFilter(sliceBuf, channelFilters[0]);

        // Time fused batched path
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int iters = 5;
        for (int iter = 0; iter < iters; iter++)
            _engine.NativeSpectralFilterBatch(input, filter);
        sw.Stop();
        double fusedMs = sw.Elapsed.TotalMilliseconds / iters;

        // Time manual per-slice loop (reuses pre-allocated slice buffer)
        sw.Restart();
        for (int iter = 0; iter < iters; iter++)
        {
            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    int off = (b * channels + c) * sliceSize;
                    for (int i = 0; i < sliceSize; i++) sliceBuf[i] = input[off + i];
                    _engine.NativeSpectralFilter(sliceBuf, channelFilters[c]);
                }
            }
        }
        sw.Stop();
        double manualMs = sw.Elapsed.TotalMilliseconds / iters;

        double speedup = manualMs / fusedMs;
        _output.WriteLine($"SpectralFilterBatch [{batch},{channels},{h},{w}]:");
        _output.WriteLine($"  Fused batched: {fusedMs:F2}ms");
        _output.WriteLine($"  Manual per-slice loop: {manualMs:F2}ms");
        _output.WriteLine($"  Speedup: {speedup:F1}x");

        Assert.True(speedup >= 1.0,
            $"Fused path should not be slower than manual loop. Got {speedup:F1}x " +
            $"(fused={fusedMs:F2}ms, manual={manualMs:F2}ms)");
    }

    // ================================================================
    // Float type tests (verify both float and double paths work)
    // ================================================================

    [Fact]
    public void SpectralFilter_FloatType_MatchesManualPipeline()
    {
        int h = 8, w = 8;
        var rng = new Random(42);
        var input = new Tensor<float>([h, w]);
        for (int i = 0; i < h * w; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var filter = new Tensor<Complex<float>>([h, w]);
        for (int i = 0; i < h * w; i++)
            filter[i] = new Complex<float>((float)(rng.NextDouble() * 2 - 1), (float)(rng.NextDouble() * 2 - 1));

        var spectrum = _engine.NativeComplexFFT2D(input);
        var filtered = _engine.NativeComplexMultiply(spectrum, filter);
        var expected = _engine.NativeComplexIFFT2DReal(filtered);

        var actual = _engine.NativeSpectralFilter(input, filter);

        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(expected[i] - actual[i]);
            Assert.True(diff < 1e-4,
                $"Float SpectralFilter mismatch at [{i}]: expected={expected[i]:F6}, actual={actual[i]:F6}");
        }
    }

    [Fact]
    public void SpectralFilterBatch_FloatType_IdentityPreservesInput()
    {
        int batch = 2, channels = 3, h = 8, w = 8;
        var rng = new Random(55);
        var input = new Tensor<float>([batch, channels, h, w]);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var filter = new Tensor<Complex<float>>([h, w]);
        for (int i = 0; i < h * w; i++)
            filter[i] = new Complex<float>(1.0f, 0.0f);

        var result = _engine.NativeSpectralFilterBatch(input, filter);

        for (int i = 0; i < input.Length; i++)
        {
            double diff = Math.Abs(input[i] - result[i]);
            Assert.True(diff < 1e-4,
                $"Float identity filter mismatch at [{i}]: input={input[i]:F6}, result={result[i]:F6}");
        }
    }
}
