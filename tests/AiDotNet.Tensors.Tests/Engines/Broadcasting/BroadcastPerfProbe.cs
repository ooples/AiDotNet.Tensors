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

        // Warm up both paths (JIT, pools).
        for (int i = 0; i < 3; i++)
        {
            _ = engine.TensorAdd(x, scale);
            _ = engine.TensorBroadcastAdd(x, scale);
        }

        const int reps = 10;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < reps; i++) _ = engine.TensorAdd(x, scale);
        sw.Stop();
        double implicitMs = sw.Elapsed.TotalMilliseconds / reps;

        sw.Restart();
        for (int i = 0; i < reps; i++) _ = engine.TensorBroadcastAdd(x, scale);
        sw.Stop();
        double explicitMs = sw.Elapsed.TotalMilliseconds / reps;

        _out.WriteLine($"shape [1,64,112,112] + [1,64,1,1], {reps} reps");
        _out.WriteLine($"  implicit TensorAdd      : {implicitMs:F3} ms/call");
        _out.WriteLine($"  explicit BroadcastAdd   : {explicitMs:F3} ms/call");
        _out.WriteLine($"  ratio (implicit/explicit): {implicitMs / explicitMs:F2}x");

        // Values must agree regardless of which path ran.
        var viaImplicit = engine.TensorAdd(x, scale);
        var viaExplicit = engine.TensorBroadcastAdd(x, scale);
        for (int i = 0; i < viaExplicit.Length; i += 997)
            Assert.Equal(viaExplicit[i], viaImplicit[i], 5);
    }
}
