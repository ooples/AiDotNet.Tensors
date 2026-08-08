using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Broadcasting;

/// <summary>
/// Measures the BatchNorm-shaped broadcast that the TensorBroadcast* SIMD fast paths exist for.
/// </summary>
/// <remarks>
/// Not an assertion of a threshold — a probe that prints numbers, so the decision about whether the
/// implicit path needs those fast paths is made from measurement rather than from reading the code.
/// </remarks>
public class BroadcastPerfProbe
{
    private readonly ITestOutputHelper _out;
    public BroadcastPerfProbe(ITestOutputHelper output) => _out = output;

    [Fact]
    public void CompareImplicitBroadcastAgainstTheExplicitFastPath()
    {
        var engine = new CpuEngine();

        // The shape the fast-path comment calls out: NCHW activations plus per-channel scale.
        var x = new Tensor<float>([1, 64, 112, 112]);
        for (int i = 0; i < x.Length; i++) x[i] = i % 7 * 0.1f;
        var scale = new Tensor<float>([1, 64, 1, 1]);
        for (int i = 0; i < scale.Length; i++) scale[i] = 1.0f + i % 3;

        // Warm up both paths (JIT, tiering, pools).
        for (int i = 0; i < 50; i++)
        {
            _ = engine.TensorAdd(x, scale);
            _ = engine.TensorBroadcastAdd(x, scale);
        }

        // Report the MINIMUM over many trials, not the mean. Each call allocates a 3.2 MB result,
        // so GC and thermal noise land as large positive outliers; earlier runs of this probe read
        // 10.8, 4.5 and 8.1 ms for the same unchanged code, which is too wide to optimize against.
        // The floor is the stable quantity.
        double Best(Func<Tensor<float>> call, int trials = 30)
        {
            double best = double.MaxValue;
            for (int t = 0; t < trials; t++)
            {
                var sw = Stopwatch.StartNew();
                _ = call();
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
            return best;
        }

        // Decompose the implicit path so the cost lands on a specific stage rather than a guess.
        var preExpanded = scale.ExpandTo([1, 64, 112, 112]);      // view built once, outside the timer
        var dense = preExpanded.Contiguous();                      // same shape, fully contiguous

        double implicitMs = Best(() => engine.TensorAdd(x, scale));
        double explicitMs = Best(() => engine.TensorBroadcastAdd(x, scale));
        double preExpandedMs = Best(() => engine.TensorAdd(x, preExpanded));
        double contiguousMs = Best(() => engine.TensorAdd(x, dense));

        _out.WriteLine("shape [1,64,112,112] + [1,64,1,1], min of 30 trials");
        _out.WriteLine($"  implicit TensorAdd        : {implicitMs:F3} ms/call");
        _out.WriteLine($"  explicit BroadcastAdd     : {explicitMs:F3} ms/call");
        _out.WriteLine($"  TensorAdd(pre-expanded)   : {preExpandedMs:F3} ms/call   <- strided SIMD kernel alone");
        _out.WriteLine($"  TensorAdd(contiguous)     : {contiguousMs:F3} ms/call   <- floor, no broadcast at all");
        _out.WriteLine($"  ratio (implicit/explicit) : {implicitMs / explicitMs:F2}x");

        // Values must agree regardless of which path ran.
        var viaImplicit = engine.TensorAdd(x, scale);
        var viaExplicit = engine.TensorBroadcastAdd(x, scale);
        for (int i = 0; i < viaExplicit.Length; i += 997)
            Assert.Equal(viaExplicit[i], viaImplicit[i], 5);
    }
}
