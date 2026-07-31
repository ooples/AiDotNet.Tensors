using System;
using System.Diagnostics;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Bit-exactness + allocation/wall-time coverage for the blittable NativeComplexFFT /
/// NativeComplexIFFTReal fast path (float/double, contiguous). The fast path runs the
/// tested SIMD radix-2 kernel (FftKernels) over a [ThreadStatic] interleaved double[]
/// scratch instead of a per-call Complex&lt;T&gt;[] buffer. Each test A/Bs the fast path
/// against the retained legacy generic Complex&lt;T&gt;[] path IN THE SAME PROCESS
/// (NativeComplexFFTLegacyGeneric / NativeComplexIFFTRealLegacyGeneric), so results are
/// immune to cross-build / machine-load noise. Runs on net10.0 + net471.
/// </summary>
public class NativeComplexFftBlittableTests
{
    private readonly ITestOutputHelper _output;
    public NativeComplexFftBlittableTests(ITestOutputHelper output) => _output = output;

    private static CpuEngine Cpu() => (AiDotNetEngine.Current as CpuEngine) ?? new CpuEngine();

    private static Tensor<double> RandD(int b, int n, int seed)
    {
        var t = new Tensor<double>([b, n]);
        var sp = t.AsWritableSpan();
        uint s = (uint)seed | 1u;
        for (int i = 0; i < sp.Length; i++) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; sp[i] = (s / (double)uint.MaxValue) - 0.5; }
        return t;
    }

    private static Tensor<float> RandF(int b, int n, int seed)
    {
        var t = new Tensor<float>([b, n]);
        var sp = t.AsWritableSpan();
        uint s = (uint)seed | 1u;
        for (int i = 0; i < sp.Length; i++) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; sp[i] = (float)((s / (double)uint.MaxValue) - 0.5); }
        return t;
    }

    // ---- Bit-exactness: fast path vs retained legacy generic path -------------

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void NativeComplexFFT_Double_MatchesLegacyGeneric(int n)
    {
        int b = 8;
        var cpu = Cpu();
        var x = RandD(b, n, 1234 + n);
        var fast = cpu.NativeComplexFFT(x);
        var legacy = cpu.NativeComplexFFTLegacyGeneric(x);
        double maxDiff = 0;
        for (int i = 0; i < fast.Length; i++)
        {
            maxDiff = Math.Max(maxDiff, Math.Abs(fast[i].Real - legacy[i].Real));
            maxDiff = Math.Max(maxDiff, Math.Abs(fast[i].Imaginary - legacy[i].Imaginary));
        }
        _output.WriteLine($"double FFT n={n} max|fast-legacy|={maxDiff:E3}");
        Assert.True(maxDiff < 1e-9, $"double FFT fast vs legacy maxDiff {maxDiff:E3} exceeds 1e-9 (n={n})");
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void NativeComplexIFFTReal_Double_MatchesLegacyGeneric(int n)
    {
        int b = 8;
        var cpu = Cpu();
        var real = RandD(b, n, 77 + n);
        var spec = cpu.NativeComplexFFT(real); // shared contiguous spectrum
        var fast = cpu.NativeComplexIFFTReal(spec);
        var legacy = cpu.NativeComplexIFFTRealLegacyGeneric(spec);
        double maxDiff = 0;
        for (int i = 0; i < fast.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(fast[i] - legacy[i]));
        _output.WriteLine($"double IFFTReal n={n} max|fast-legacy|={maxDiff:E3}");
        Assert.True(maxDiff < 1e-9, $"double IFFTReal fast vs legacy maxDiff {maxDiff:E3} exceeds 1e-9 (n={n})");
    }

    [Theory]
    [InlineData(256)]
    [InlineData(1024)]
    public void NativeComplexFFT_Float_MatchesLegacyGeneric(int n)
    {
        // Float fast path uses a double-precision intermediate (reuses the double
        // SIMD kernel), so it is MORE accurate than the all-float legacy butterfly;
        // the two differ only by float rounding of the legacy path, not a bug.
        int b = 8;
        var cpu = Cpu();
        var x = RandF(b, n, 4242 + n);
        var fast = cpu.NativeComplexFFT(x);
        var legacy = cpu.NativeComplexFFTLegacyGeneric(x);
        double maxAbs = 0, maxRel = 0;
        for (int i = 0; i < fast.Length; i++)
        {
            double d = Math.Abs((double)fast[i].Real - legacy[i].Real);
            double d2 = Math.Abs((double)fast[i].Imaginary - legacy[i].Imaginary);
            double denom = 1.0 + Math.Abs(legacy[i].Real) + Math.Abs(legacy[i].Imaginary);
            maxAbs = Math.Max(maxAbs, Math.Max(d, d2));
            maxRel = Math.Max(maxRel, Math.Max(d, d2) / denom);
        }
        _output.WriteLine($"float FFT n={n} max|fast-legacy|={maxAbs:E3} maxRel={maxRel:E3}");
        Assert.True(maxRel < 1e-3, $"float FFT fast vs legacy maxRel {maxRel:E3} exceeds 1e-3 (n={n})");
    }

    // ---- Absolute correctness: fast path vs independent O(n^2) DFT ------------

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    public void NativeComplexFFT_Double_MatchesNaiveDft(int n)
    {
        var cpu = Cpu();
        var x = RandD(1, n, 909 + n);
        var fast = cpu.NativeComplexFFT(x);
        double maxDiff = 0;
        for (int k = 0; k < n; k++)
        {
            double re = 0, im = 0;
            for (int t = 0; t < n; t++)
            {
                double ang = -2.0 * Math.PI * k * t / n;
                re += x[t] * Math.Cos(ang);
                im += x[t] * Math.Sin(ang);
            }
            maxDiff = Math.Max(maxDiff, Math.Abs(fast[k].Real - re));
            maxDiff = Math.Max(maxDiff, Math.Abs(fast[k].Imaginary - im));
        }
        _output.WriteLine($"double FFT n={n} max|fast-naiveDFT|={maxDiff:E3}");
        Assert.True(maxDiff < 1e-8, $"double FFT vs naive DFT maxDiff {maxDiff:E3} exceeds 1e-8 (n={n})");
    }

    [Fact]
    public void FFT_IFFTReal_RoundTrip_Double_Batched()
    {
        int b = 16, n = 1024;
        var cpu = Cpu();
        var x = RandD(b, n, 5150);
        var recovered = cpu.NativeComplexIFFTReal(cpu.NativeComplexFFT(x));
        double maxDiff = 0;
        for (int i = 0; i < x.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(x[i] - recovered[i]));
        _output.WriteLine($"roundtrip [b={b},n={n}] max|x-IFFT(FFT(x))|={maxDiff:E3}");
        Assert.True(maxDiff < 1e-9, $"round-trip maxDiff {maxDiff:E3} exceeds 1e-9");
    }

    // ---- Allocation + wall-time A/B (opt-in) ---------------------------------
    // HE_FFT_ALLOC_BENCH=1 : prints GC.GetAllocatedBytesForCurrentThread and median
    // ms/call for the blittable fast path vs the legacy generic path on [128,1024],
    // same process. This is the trustworthy before/after (no cross-build noise).

    [Fact]
    public void NativeComplexFFT_AllocAndTime_Bench()
    {
        if (Environment.GetEnvironmentVariable("HE_FFT_ALLOC_BENCH") != "1") return;
        int b = 128, n = 1024, WARM = 5, REP = 50;
        var cpu = Cpu();
        var x = RandD(b, n, 20260721);
        var spec = cpu.NativeComplexFFT(x);
        var sb = new StringBuilder();
        sb.AppendLine($"NativeComplexFFT / NativeComplexIFFTReal blittable-vs-legacy on [{b},{n}] (REP={REP}) engine={cpu.GetType().Name}");
        sb.AppendLine("  op                       path      bytes/call     median_ms");

        void Bench(string op, string path, Func<long> oneCallReturningBytesUnused, Action call)
        {
            for (int i = 0; i < WARM; i++) call();
            // allocation: measure a single call delta after warmup (steady state)
            long bytes;
#if NET5_0_OR_GREATER
            long a0 = GC.GetAllocatedBytesForCurrentThread();
            call();
            long a1 = GC.GetAllocatedBytesForCurrentThread();
            bytes = a1 - a0;
#else
            call();
            bytes = -1; // GC.GetAllocatedBytesForCurrentThread unavailable on net471
#endif
            var times = new double[REP];
            for (int i = 0; i < REP; i++) { var sw = Stopwatch.StartNew(); call(); sw.Stop(); times[i] = sw.Elapsed.TotalMilliseconds; }
            Array.Sort(times);
            sb.AppendLine($"  {op,-24} {path,-8} {bytes,12}     {times[REP / 2],9:F3}");
        }

        Bench("NativeComplexFFT", "legacy", () => 0, () => { _ = cpu.NativeComplexFFTLegacyGeneric(x); });
        Bench("NativeComplexFFT", "fast", () => 0, () => { _ = cpu.NativeComplexFFT(x); });
        Bench("NativeComplexIFFTReal", "legacy", () => 0, () => { _ = cpu.NativeComplexIFFTRealLegacyGeneric(spec); });
        Bench("NativeComplexIFFTReal", "fast", () => 0, () => { _ = cpu.NativeComplexIFFTReal(spec); });

        var outp = sb.ToString();
        _output.WriteLine(outp);
        try { System.IO.File.WriteAllText(@"C:\Users\cheat\Temp\he-m2-audit\localpkgs\fft_alloc_bench.txt", outp); } catch { /* print is the record */ }
    }
}
