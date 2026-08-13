using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Pins the N-D GPU FFT against a direct DFT, at the sizes and shapes that used to be wrong.
/// </summary>
/// <remarks>
/// <para>
/// The N-D path is the only consumer of <c>batched_bit_reverse</c> and <c>batched_fft_butterfly</c>;
/// FFT2D drives its own row/column kernels and the 1-D entry points drive the single-signal kernel.
/// That is why every other FFT parity test passed while these two failed, and why the bug survived
/// several rounds of reading the shared-looking kernel source.
/// </para>
/// <para>
/// The defect was an in-place swap in the bit-reversal permutation. Instrumenting the kernel showed
/// the index math and the <c>i &lt; j</c> guard were both correct — exactly one work item per pair
/// fired — but only the first half of the swap landed. For input 1..8 the permutation produced
/// [1,5,3,7,5,6,7,8]: slots 1 and 3 took their partner's value while slots 4 and 6 kept their
/// originals. n=2 passed throughout because it is the one size where the permutation is the
/// identity, which made "small transforms are fine" a misleading signal.
/// </para>
/// <para>
/// These assert against a direct O(n²) DFT rather than against the CPU engine. A parity test can
/// only say the two disagree; it cannot say which is wrong. When this was failing, that distinction
/// mattered — the CPU agreed with the DFT to 3E-008 and the GPU was off by 6.9E-001.
/// </para>
/// </remarks>
[Collection("DirectGpuSerial")]
public class FftNdCorrectnessTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _cpu = new();
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly bool _available;

    public FftNdCorrectnessTests(ITestOutputHelper o)
    {
        _out = o;
        try { _gpu = new DirectGpuTensorEngine(); _available = _gpu.IsGpuAvailable; }
        catch { _available = false; }
    }

    public void Dispose() { _gpu?.Dispose(); GC.SuppressFinalize(this); }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    /// <summary>Direct O(n²) DFT of one row — no algorithmic structure to get wrong.</summary>
    private static (double Re, double Im) Dft(Tensor<float> x, int row, int n, int k)
    {
        double re = 0, im = 0;
        for (int t = 0; t < n; t++)
        {
            double ang = -2.0 * Math.PI * k * t / n;
            double v = x.Rank == 1 ? x[t] : x[row, t];
            re += v * Math.Cos(ang);
            im += v * Math.Sin(ang);
        }
        return (re, im);
    }

    /// <summary>
    /// The exact case that was wrong: a known input whose transform can be checked by hand.
    /// </summary>
    [SkippableFact]
    public void KnownInput_MatchesTheHandComputedTransform()
    {
        Skip.If(!_available, "GPU backend not available");

        var x = new Tensor<float>([4]);
        x[0] = 1; x[1] = 2; x[2] = 3; x[3] = 4;

        var gpu = _gpu!.NativeComplexFFTND(x, [0]);

        // DFT of [1,2,3,4] by hand: 10, -2+2i, -2, -2-2i.
        var expected = new[] { (10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0) };
        for (int k = 0; k < 4; k++)
        {
            Assert.True(Math.Abs(gpu[k].Real - expected[k].Item1) < 1e-4,
                $"X[{k}].re = {gpu[k].Real} expected {expected[k].Item1}");
            Assert.True(Math.Abs(gpu[k].Imaginary - expected[k].Item2) < 1e-4,
                $"X[{k}].im = {gpu[k].Imaginary} expected {expected[k].Item2}");
        }
    }

    /// <summary>
    /// Transform length sweep. n=2 was the ONLY size that used to pass, because it is the only one
    /// whose bit-reversal permutation is the identity.
    /// </summary>
    [SkippableTheory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(32)]
    [InlineData(64)]
    public void TransformLength_MatchesDirectDft(int n)
    {
        Skip.If(!_available, "GPU backend not available");

        var x = Rand([4, n], 11);
        var gpu = _gpu!.NativeComplexFFTND(x, [1]);

        double worst = 0;
        for (int row = 0; row < 4; row++)
            for (int k = 0; k < n; k++)
            {
                var (re, im) = Dft(x, row, n, k);
                var got = gpu[row, k];
                worst = Math.Max(worst, Math.Max(Math.Abs(got.Real - re), Math.Abs(got.Imaginary - im)));
            }

        _out.WriteLine($"n={n} max_abs_err={worst:E3}");
        Assert.True(worst < 1e-3, $"n={n}: GPU N-D FFT differs from a direct DFT by {worst:E3}");
    }

    /// <summary>
    /// Batch sweep with n fixed. Batch was never the variable — batch=1 failed exactly as batch=16
    /// did — but the batched kernels index by a flattened global id, so it stays pinned.
    /// </summary>
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(8)]
    [InlineData(16)]
    public void BatchCount_MatchesDirectDft(int batch)
    {
        Skip.If(!_available, "GPU backend not available");
        const int n = 8;

        var x = Rand([batch, n], 13);
        var gpu = _gpu!.NativeComplexFFTND(x, [1]);

        double worst = 0;
        for (int row = 0; row < batch; row++)
            for (int k = 0; k < n; k++)
            {
                var (re, im) = Dft(x, row, n, k);
                var got = gpu[row, k];
                worst = Math.Max(worst, Math.Max(Math.Abs(got.Real - re), Math.Abs(got.Imaginary - im)));
            }

        Assert.True(worst < 1e-3, $"batch={batch}: GPU N-D FFT differs from a direct DFT by {worst:E3}");
    }

    /// <summary>
    /// Transforming a non-final axis routes through permute/unpermute; transforming the last axis
    /// does not. Both were wrong before, for the same underlying reason.
    /// </summary>
    [SkippableFact]
    public void EveryAxisSelection_AgreesWithTheCpu()
    {
        Skip.If(!_available, "GPU backend not available");

        foreach (var shape in new[] { new[] { 8, 8 }, new[] { 16, 32 } })
        {
            var x = Rand(shape, 17);
            foreach (var axes in new[] { new[] { 1 }, new[] { 0 }, new[] { 0, 1 } })
            {
                var cpu = _cpu.NativeComplexFFTND(x, axes);
                var gpu = _gpu!.NativeComplexFFTND(x, axes);

                double worst = 0;
                for (int i = 0; i < cpu.Length; i++)
                    worst = Math.Max(worst,
                        Math.Max(Math.Abs(cpu[i].Real - gpu[i].Real), Math.Abs(cpu[i].Imaginary - gpu[i].Imaginary)));

                Assert.True(worst < 1e-2,
                    $"shape [{string.Join(",", shape)}] axes [{string.Join(",", axes)}]: " +
                    $"GPU differs from CPU by {worst:E3}");
            }
        }
    }
}
