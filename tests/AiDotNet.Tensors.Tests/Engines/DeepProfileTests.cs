using System.Diagnostics;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

public class DeepProfileTests
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public DeepProfileTests(ITestOutputHelper output) => _output = output;

    [Fact(Skip = "Profiling test — run manually with --filter DeepProfile")]
    public void Profile_BatchMatMul_WhereWeLose()
    {
        // We: 0.117ms, PyTorch: 0.016ms — 7.3x gap
        // Hypothesis: per-slice BLAS call overhead (4 calls vs 1 strided call)
        int B = 4, M = 32, K = 64, N = 32;
        var a = Tensor<float>.CreateRandom([B, M, K]);
        var b = Tensor<float>.CreateRandom([B, K, N]);
        int iters = 200;

        for (int w = 0; w < 20; w++) _engine.BatchMatMul(a, b);

        // A. Full BatchMatMul
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) _engine.BatchMatMul(a, b);
        sw.Stop();
        _output.WriteLine($"A. Full BatchMatMul: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // B. Just GetDataArray (is data extraction slow?)
        sw.Restart();
        for (int i = 0; i < iters; i++)
        {
            var _ = a.GetDataArray();
            var __ = b.GetDataArray();
        }
        sw.Stop();
        _output.WriteLine($"B. GetDataArray x2: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // C. TensorAllocator.RentUninitialized for output
        sw.Restart();
        for (int i = 0; i < iters; i++)
            TensorAllocator.RentUninitialized<float>(new[] { B, M, N });
        sw.Stop();
        _output.WriteLine($"C. RentUninitialized [{B},{M},{N}]: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // D. 4x raw BLAS calls only (pre-pinned)
        var af = (float[])(object)a.GetDataArray();
        var bf = (float[])(object)b.GetDataArray();
        var cf = new float[B * M * N];
        sw.Restart();
        for (int i = 0; i < iters; i++)
            for (int s = 0; s < B; s++)
                BlasProvider.TryGemm(M, N, K, af, s * M * K, K, bf, s * K * N, N, cf, s * M * N, N);
        sw.Stop();
        _output.WriteLine($"D. 4x BLAS (pre-alloc): {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // E. Single BLAS (reshape to 2D: [128,64]@[64,32])
        var a2d = new float[B * M * K];
        Array.Copy(af, a2d, af.Length);
        var b2d = new float[K * N]; // Just first slice for single call
        Array.Copy(bf, b2d, K * N);
        sw.Restart();
        for (int i = 0; i < iters; i++)
            BlasProvider.TryGemm(B * M, N, K, a2d, 0, K, b2d, 0, N, cf, 0, N);
        sw.Stop();
        _output.WriteLine($"E. Single BLAS [{B * M},{K}]@[{K},{N}]: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // F. BatchMatMul with arena (training simulation)
        using (var arena2 = TensorArena.Create())
        {
            using (var tape = new GradientTape<float>())
                _engine.BatchMatMul(a, b);
            arena2.Reset();
            sw.Restart();
            for (int i = 0; i < iters; i++)
            {
                arena2.Reset();
                using var tape = new GradientTape<float>();
                _engine.BatchMatMul(a, b);
            }
            sw.Stop();
            _output.WriteLine($"F. BatchMatMul (arena+tape): {sw.Elapsed.TotalMilliseconds / iters:F4}ms");
        }

        // G. With arena
        using (var arena = TensorArena.Create())
        {
            _engine.BatchMatMul(a, b);
            arena.Reset();
            sw.Restart();
            for (int i = 0; i < iters; i++)
            {
                arena.Reset();
                _engine.BatchMatMul(a, b);
            }
            sw.Stop();
            _output.WriteLine($"G. BatchMatMul (arena): {sw.Elapsed.TotalMilliseconds / iters:F4}ms");
        }
    }

    [Fact(Skip = "Profiling test — run manually with --filter DeepProfile")]
    public void Profile_TensorMatMul_WhereWeLose()
    {
        // We: 0.077ms, PyTorch: 0.029ms — 2.7x gap
        var a = Tensor<float>.CreateRandom([32, 256]);
        var b = Tensor<float>.CreateRandom([256, 256]);
        int iters = 200;

        for (int w = 0; w < 20; w++) _engine.TensorMatMul(a, b);

        // A. Full TensorMatMul
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) _engine.TensorMatMul(a, b);
        sw.Stop();
        _output.WriteLine($"A. TensorMatMul: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // B. Raw BLAS
        var af = (float[])(object)a.GetDataArray();
        var bf = (float[])(object)b.GetDataArray();
        var cf = new float[32 * 256];
        BlasProvider.TryGemm(32, 256, 256, af, 0, 256, bf, 0, 256, cf, 0, 256);
        sw.Restart();
        for (int i = 0; i < iters; i++)
            BlasProvider.TryGemm(32, 256, 256, af, 0, 256, bf, 0, 256, cf, 0, 256);
        sw.Stop();
        _output.WriteLine($"B. Raw BLAS: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // C. Contiguous checks
        sw.Restart();
        for (int i = 0; i < iters; i++)
        {
            var _ = a.IsContiguous;
            var __ = b.IsContiguous;
        }
        sw.Stop();
        _output.WriteLine($"C. IsContiguous x2: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // D. RentUninitialized
        sw.Restart();
        for (int i = 0; i < iters; i++)
            TensorAllocator.RentUninitialized<float>(new[] { 32, 256 });
        sw.Stop();
        _output.WriteLine($"D. RentUninitialized: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // E. With arena
        using (var arena = TensorArena.Create())
        {
            _engine.TensorMatMul(a, b);
            arena.Reset();
            sw.Restart();
            for (int i = 0; i < iters; i++)
            {
                arena.Reset();
                _engine.TensorMatMul(a, b);
            }
            sw.Stop();
            _output.WriteLine($"E. TensorMatMul (arena): {sw.Elapsed.TotalMilliseconds / iters:F4}ms");
        }

        // F. Tensor.MatrixMultiply (instance method, routes through engine)
        sw.Restart();
        for (int i = 0; i < iters; i++) a.MatrixMultiply(b);
        sw.Stop();
        _output.WriteLine($"F. Tensor.MatrixMultiply: {sw.Elapsed.TotalMilliseconds / iters:F4}ms");
    }

    [Fact(Skip = "Profiling test — run manually with --filter DeepProfile")]
    public void Profile_FusedLinear_WhereWeLose()
    {
        // We: 0.141ms, PyTorch: 0.043ms — 3.3x gap
        var input = Tensor<float>.CreateRandom([32, 256]);
        var weights = Tensor<float>.CreateRandom([256, 256]);
        var bias = Tensor<float>.CreateRandom([1, 256]);
        int iters = 200;

        for (int w = 0; w < 20; w++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);

        // A. Full FusedLinear (with tape because tape is active during training)
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        sw.Stop();
        _output.WriteLine($"A. FusedLinear (no tape): {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // B. FusedLinear WITH tape (what training actually does)
        using (var tape = new GradientTape<float>())
        {
            // warmup
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        }
        sw.Restart();
        for (int i = 0; i < iters; i++)
        {
            using var tape = new GradientTape<float>();
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        }
        sw.Stop();
        _output.WriteLine($"B. FusedLinear (with tape): {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // C. Decomposed: MatMul + BroadcastAdd + ReLU (what tape path does)
        sw.Restart();
        for (int i = 0; i < iters; i++)
        {
            var mm = _engine.TensorMatMul(input, weights);
            var added = _engine.TensorBroadcastAdd(mm, bias);
            var activated = _engine.ReLU(added);
        }
        sw.Stop();
        _output.WriteLine($"C. Decomposed (MatMul+Add+ReLU): {sw.Elapsed.TotalMilliseconds / iters:F4}ms");

        // D. With arena
        using (var arena = TensorArena.Create())
        {
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
            arena.Reset();
            sw.Restart();
            for (int i = 0; i < iters; i++)
            {
                arena.Reset();
                _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
            }
            sw.Stop();
            _output.WriteLine($"D. FusedLinear (arena, no tape): {sw.Elapsed.TotalMilliseconds / iters:F4}ms");
        }

        // E. With arena AND tape (realistic training)
        using (var arena = TensorArena.Create())
        {
            using (var tape = new GradientTape<float>())
                _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
            arena.Reset();
            sw.Restart();
            for (int i = 0; i < iters; i++)
            {
                arena.Reset();
                using var tape = new GradientTape<float>();
                _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
            }
            sw.Stop();
            _output.WriteLine($"E. FusedLinear (arena+tape): {sw.Elapsed.TotalMilliseconds / iters:F4}ms");
        }
    }
}
