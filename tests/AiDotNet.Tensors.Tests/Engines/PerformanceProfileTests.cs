using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Granular profiling to find exact bottlenecks for each operation vs PyTorch.
/// Run manually to identify optimization targets.
/// </summary>
public class PerformanceProfileTests
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public PerformanceProfileTests(ITestOutputHelper output) => _output = output;

    [Fact(Skip = "Profile - run manually")]
    public void Profile_BatchMatMul_Breakdown()
    {
        // PyTorch: 0.022ms for [4,32,64]@[4,64,32]
        var a = Tensor<float>.CreateRandom([4, 32, 64]);
        var b = Tensor<float>.CreateRandom([4, 64, 32]);
        int iters = 100;

        for (int w = 0; w < 10; w++) _engine.BatchMatMul(a, b);

        // A. Full BatchMatMul
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) _engine.BatchMatMul(a, b);
        sw.Stop();
        _output.WriteLine($"A. BatchMatMul [4,32,64]: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // B. 4x individual TensorMatMul (what batch should cost)
        var slices_a = new Tensor<float>[4];
        var slices_b = new Tensor<float>[4];
        for (int s = 0; s < 4; s++)
        {
            slices_a[s] = Tensor<float>.CreateRandom([32, 64]);
            slices_b[s] = Tensor<float>.CreateRandom([64, 32]);
        }
        sw.Restart();
        for (int i = 0; i < iters; i++)
            for (int s = 0; s < 4; s++)
                _engine.TensorMatMul(slices_a[s], slices_b[s]);
        sw.Stop();
        _output.WriteLine($"B. 4x TensorMatMul [32,64]: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // C. Raw BLAS 4x (pre-allocated)
        var af = new float[4 * 32 * 64];
        var bf = new float[4 * 64 * 32];
        var cf = new float[4 * 32 * 32];
        var rng = new Random(42);
        for (int j = 0; j < af.Length; j++) af[j] = (float)(rng.NextDouble() - 0.5);
        sw.Restart();
        for (int i = 0; i < iters; i++)
            for (int s = 0; s < 4; s++)
                BlasProvider.TryGemm(32, 32, 64, af, s * 32 * 64, 64, bf, s * 64 * 32, 32, cf, s * 32 * 32, 32);
        sw.Stop();
        _output.WriteLine($"C. 4x raw BLAS [32,64]: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // D. Single large matmul (same total FLOPs: 128x64 @ 64x32)
        var bigA = Tensor<float>.CreateRandom([128, 64]);
        var bigB = Tensor<float>.CreateRandom([64, 32]);
        sw.Restart();
        for (int i = 0; i < iters; i++) _engine.TensorMatMul(bigA, bigB);
        sw.Stop();
        _output.WriteLine($"D. Single 128x64 @ 64x32: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");
    }

    [Fact(Skip = "Profile - run manually")]
    public void Profile_Elementwise_Breakdown()
    {
        // PyTorch: Add 0.015ms, Multiply 0.013ms for 100K
        int size = 100000;
        var a = Tensor<float>.CreateRandom([size]);
        var b = Tensor<float>.CreateRandom([size]);
        int iters = 200;

        for (int w = 0; w < 20; w++) { _engine.TensorAdd(a, b); _engine.TensorMultiply(a, b); }

        // A. Full TensorAdd
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) _engine.TensorAdd(a, b);
        sw.Stop();
        _output.WriteLine($"A. TensorAdd 100K: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // B. Full TensorMultiply
        sw.Restart();
        for (int i = 0; i < iters; i++) _engine.TensorMultiply(a, b);
        sw.Stop();
        _output.WriteLine($"B. TensorMultiply 100K: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // C. Raw array add (no tensor overhead) via Buffer.BlockCopy baseline
        var af = (float[])(object)a.GetDataArray();
        var bf = (float[])(object)b.GetDataArray();
        var cf = new float[size];
        for (int w = 0; w < 10; w++)
            for (int j = 0; j < size; j++) cf[j] = af[j] + bf[j];
        sw.Restart();
        for (int i = 0; i < iters; i++)
            for (int j = 0; j < size; j++) cf[j] = af[j] + bf[j];
        sw.Stop();
        _output.WriteLine($"C. Raw scalar add (baseline): {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // D. TensorAllocator.Rent overhead
        sw.Restart();
        for (int i = 0; i < iters; i++)
            TensorAllocator.Rent<float>([size]);
        sw.Stop();
        _output.WriteLine($"D. TensorAllocator.Rent [{size}]: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // E. With arena
        using (var arena = TensorArena.Create())
        {
            TensorAllocator.Rent<float>([size]);
            arena.Reset();
            sw.Restart();
            for (int i = 0; i < iters; i++)
            {
                arena.Reset();
                _engine.TensorAdd(a, b);
            }
            sw.Stop();
            _output.WriteLine($"E. TensorAdd 100K (arena): {sw.Elapsed.TotalMilliseconds / iters:F4}ms");
        }

#if NET5_0_OR_GREATER
        // F. Allocation bytes per Add
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10; i++) _engine.TensorAdd(a, b);
        long after = GC.GetAllocatedBytesForCurrentThread();
        _output.WriteLine($"F. Bytes per TensorAdd: {(after - before) / 10:N0}");
#endif
    }

    [Fact(Skip = "Profile - run manually")]
    public void Profile_FusedLinear_Double_Breakdown()
    {
        // PyTorch: 0.068ms
        var input = new Tensor<double>(Enumerable.Range(0, 32 * 256).Select(i => (double)i / 1000).ToArray(), [32, 256]);
        var weights = new Tensor<double>(Enumerable.Range(0, 256 * 256).Select(i => (double)i / 100000).ToArray(), [256, 256]);
        var bias = new Tensor<double>(Enumerable.Range(0, 256).Select(i => 0.01 * i).ToArray(), [1, 256]);
        int iters = 100;

        for (int w = 0; w < 10; w++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);

        // A. Full FusedLinear double
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        sw.Stop();
        _output.WriteLine($"A. FusedLinear double 32x256+ReLU: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // B. Raw DGEMM
        var ad = (double[])(object)input.GetDataArray();
        var wd = (double[])(object)weights.GetDataArray();
        var cd = new double[32 * 256];
        BlasProvider.TryGemm(32, 256, 256, ad, 0, 256, wd, 0, 256, cd, 0, 256);
        sw.Restart();
        for (int i = 0; i < iters; i++)
            BlasProvider.TryGemm(32, 256, 256, ad, 0, 256, wd, 0, 256, cd, 0, 256);
        sw.Stop();
        _output.WriteLine($"B. Raw DGEMM (pre-alloc): {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // C. TensorMatMul double only
        sw.Restart();
        for (int i = 0; i < iters; i++)
            _engine.TensorMatMul(input, weights);
        sw.Stop();
        _output.WriteLine($"C. TensorMatMul double: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");
    }
}
