using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

// NOT namespace ...LinearAlgebra.Fft — that segment shadows the public static class Fft that the sibling
// suites call as Fft.RFft(...), breaking their compilation. The existing files use FftTests for exactly
// this reason.
namespace AiDotNet.Tensors.Tests.LinearAlgebra.FftTests;

/// <summary>
/// Wall-clock and allocation for <see cref="IEngine.RFFT{T}"/> / <see cref="IEngine.IRFFT{T}"/> on the CPU
/// engine, across the batch/length shapes real spectral work actually issues.
///
/// WHY: RFFT and IRFFT were routed through FFTCore, which recomputed Math.Cos/Math.Sin INSIDE the innermost
/// butterfly loop even though the twiddle for a given (size, k) is identical across every block, and pushed
/// every value through numOps.ToDouble/FromDouble. A length-256 transform therefore cost ~2048 trig calls and
/// ~6100 generic conversions PER SIGNAL, plus ~6 Vector&lt;T&gt; allocations. An allocation-free, twiddle-cached,
/// float/double-specialised core (NativeFFTInPlace) already existed in the same file serving the
/// NativeComplex* ops — which are the NON-differentiable ones. The fast kernel was on the ops that cannot
/// train and the slow kernel on the ops that can.
///
/// HOW TO A/B: run this, then `git stash` the CpuEngine routing change, rebuild, run again, `git stash pop`.
/// Same box, same shapes, same harness — the only variable is which core RFFT dispatches to.
/// </summary>
public class RfftPerformanceBenchmark
{
    private readonly ITestOutputHelper _out;
    public RfftPerformanceBenchmark(ITestOutputHelper output) => _out = output;

    private static Tensor<double> Signal(int batch, int n, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>([batch, n]);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    [Fact(Timeout = 600000)]
    public async Task Benchmark_rfft_irfft()
    {
        await Task.Yield();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var engine = AiDotNetEngine.Current;
            const int Warmup = 3, Iters = 10;

            _out.WriteLine("RFFT / IRFFT — CpuEngine, double");
            _out.WriteLine($"warmup={Warmup}, iters={Iters}, median of {Iters}");
            _out.WriteLine("");
            _out.WriteLine("  batch      n   nFft   RFFT ms   RFFT alloc KB   IRFFT ms   IRFFT alloc KB   trig calls saved/call");
            _out.WriteLine("  " + new string('-', 104));

            foreach (var (batch, n) in new[]
                     {
                         (256, 128), (1024, 128), (4096, 192), (4096, 256), (8192, 256), (1024, 1024),
                     })
            {
                var x = Signal(batch, n, seed: 7);

                int nFft = 1;
                while (nFft < n) nFft <<= 1;

                double Time(Func<Tensor<double>> f, out long bytes)
                {
                    for (int i = 0; i < Warmup; i++) _ = f();
                    var times = new double[Iters];
                    long before = GC.GetTotalAllocatedBytes(precise: false);
                    for (int i = 0; i < Iters; i++)
                    {
                        var sw = Stopwatch.StartNew();
                        _ = f();
                        sw.Stop();
                        times[i] = sw.Elapsed.TotalMilliseconds;
                    }
                    bytes = (GC.GetTotalAllocatedBytes(precise: false) - before) / Iters;
                    Array.Sort(times);
                    return times[Iters / 2];
                }

                double tR = Time(() => engine.RFFT(x), out long aR);
                var spec = engine.RFFT(x);
                double tI = Time(() => engine.IRFFT(spec, n), out long aI);

                // What the old core recomputed per call: (nFft/2)*log2(nFft) butterflies x 2 transcendentals,
                // once per signal. The cached-twiddle core computes each (size, k) pair once per (n, inverse).
                long trig = (long)batch * (nFft / 2) * (long)Math.Log2(nFft) * 2;

                _out.WriteLine(
                    $"  {batch,5}  {n,5}  {nFft,5}  {tR,8:F3}  {aR / 1024.0,14:F1}  {tI,9:F3}  {aI / 1024.0,15:F1}  {trig / 1e6,15:F1} M");
            }

            _out.WriteLine("");
            _out.WriteLine("Round-trip correctness (IRFFT(RFFT(x)) == x) is covered by FftApiTests/FftParityTests;");
            _out.WriteLine("this file measures only, and asserts nothing about speed so it cannot fail a build.");
        }
        finally { AiDotNetEngine.Current = prior; }
    }
}
