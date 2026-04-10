using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A/B tests for eager CpuEngine ops vs PyTorch BDN reference times.
/// Measures BEFORE and AFTER each fix with proper warmup and best-of-N.
/// </summary>
[Trait("Category", "Benchmark")]
public class EagerVsPyTorchABTest
{
    private readonly ITestOutputHelper _output;
    public EagerVsPyTorchABTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void LogSoftmax_256x256_EagerVsPyTorch()
    {
        // PyTorch BDN reference: 176us (median from Run 2)
        const double pytorchMs = 0.176;
        int rows = 256, cols = 256;
        int length = rows * cols;
        var rng = new Random(42);
        var input = new float[length];
        for (int i = 0; i < length; i++) input[i] = (float)(rng.NextDouble() * 10 - 5);
        var output = new float[length];

        var hIn = GCHandle.Alloc(input, GCHandleType.Pinned);
        var hOut = GCHandle.Alloc(output, GCHandleType.Pinned);
        float* pIn = (float*)hIn.AddrOfPinnedObject();
        float* pOut = (float*)hOut.AddrOfPinnedObject();

        // Warmup
        for (int w = 0; w < 100; w++)
            for (int r = 0; r < rows; r++)
                SimdKernels.FusedLogSoftmaxRow(pIn + r * cols, pOut + r * cols, cols);

        // A/B: FusedLogSoftmaxRow (per-row call, no parallel)
        double bestDirect = double.MaxValue;
        for (int trial = 0; trial < 10; trial++)
        {
            var sw = Stopwatch.StartNew();
            for (int iter = 0; iter < 500; iter++)
                for (int r = 0; r < rows; r++)
                    SimdKernels.FusedLogSoftmaxRow(pIn + r * cols, pOut + r * cols, cols);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds / 500;
            if (ms < bestDirect) bestDirect = ms;
        }

        // B: Full CpuEngine eager path (includes allocation, pin, tape check)
        var engine = new CpuEngine();
        var tensor = new Tensor<float>(input, new[] { rows, cols });
        for (int w = 0; w < 50; w++)
            engine.TensorLogSoftmax(tensor, -1);

        double bestEager = double.MaxValue;
        for (int trial = 0; trial < 10; trial++)
        {
            var sw = Stopwatch.StartNew();
            for (int iter = 0; iter < 200; iter++)
                engine.TensorLogSoftmax(tensor, -1);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds / 200;
            if (ms < bestEager) bestEager = ms;
        }

        hIn.Free(); hOut.Free();

        _output.WriteLine($"LogSoftmax 256x256 (best of 10):");
        _output.WriteLine($"  SIMD direct:  {bestDirect:F4}ms");
        _output.WriteLine($"  Eager engine: {bestEager:F4}ms");
        _output.WriteLine($"  PyTorch BDN:  {pytorchMs}ms");
        _output.WriteLine($"  Direct ratio: {bestDirect / pytorchMs:F3}x");
        _output.WriteLine($"  Eager ratio:  {bestEager / pytorchMs:F3}x");
    }

    [Fact]
    public unsafe void MatMul_512_EagerVsPyTorch()
    {
        const double pytorchMs = 0.564;
        var engine = new CpuEngine();
        int n = 512;
        var rng = new Random(42);
        var aData = new float[n * n];
        var bData = new float[n * n];
        for (int i = 0; i < aData.Length; i++) { aData[i] = (float)(rng.NextDouble() * 2 - 1); bData[i] = (float)(rng.NextDouble() * 2 - 1); }
        var a = new Tensor<float>(aData, new[] { n, n });
        var b = new Tensor<float>(bData, new[] { n, n });

        for (int w = 0; w < 20; w++)
            engine.TensorMatMul(a, b);

        double bestEager = double.MaxValue;
        for (int trial = 0; trial < 10; trial++)
        {
            var sw = Stopwatch.StartNew();
            for (int iter = 0; iter < 50; iter++)
                engine.TensorMatMul(a, b);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds / 50;
            if (ms < bestEager) bestEager = ms;
        }

        _output.WriteLine($"MatMul 512x512 (best of 10):");
        _output.WriteLine($"  Eager:       {bestEager:F4}ms");
        _output.WriteLine($"  PyTorch BDN: {pytorchMs}ms");
        _output.WriteLine($"  Ratio:       {bestEager / pytorchMs:F3}x");
    }

    [Fact]
    public unsafe void DoubleAdd_1M_EagerVsPyTorch()
    {
        const double pytorchMs = 0.758;
        var engine = new CpuEngine();
        int n = 1_000_000;
        var rng = new Random(42);
        var aData = new double[n];
        var bData = new double[n];
        for (int i = 0; i < n; i++) { aData[i] = rng.NextDouble() * 2 - 1; bData[i] = rng.NextDouble() * 2 - 1; }
        var a = new Tensor<double>(aData, new[] { n });
        var b = new Tensor<double>(bData, new[] { n });

        for (int w = 0; w < 20; w++)
            engine.TensorAdd(a, b);

        double bestEager = double.MaxValue;
        for (int trial = 0; trial < 10; trial++)
        {
            var sw = Stopwatch.StartNew();
            for (int iter = 0; iter < 100; iter++)
                engine.TensorAdd(a, b);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds / 100;
            if (ms < bestEager) bestEager = ms;
        }

        _output.WriteLine($"Double Add 1M (best of 10):");
        _output.WriteLine($"  Eager:       {bestEager:F4}ms");
        _output.WriteLine($"  PyTorch BDN: {pytorchMs}ms");
        _output.WriteLine($"  Ratio:       {bestEager / pytorchMs:F3}x");
    }

    [Fact]
    public unsafe void DoubleExp_1M_EagerVsPyTorch()
    {
        const double pytorchMs = 0.282; // BDN Run 1 (Run 2 was noisy at 11.5ms)
        var engine = new CpuEngine();
        int n = 1_000_000;
        var rng = new Random(42);
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = rng.NextDouble() * 20 - 10;
        var tensor = new Tensor<double>(data, new[] { n });

        for (int w = 0; w < 20; w++)
            engine.TensorExp(tensor);

        double bestEager = double.MaxValue;
        for (int trial = 0; trial < 10; trial++)
        {
            var sw = Stopwatch.StartNew();
            for (int iter = 0; iter < 100; iter++)
                engine.TensorExp(tensor);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds / 100;
            if (ms < bestEager) bestEager = ms;
        }

        _output.WriteLine($"Double Exp 1M (best of 10):");
        _output.WriteLine($"  Eager:       {bestEager:F4}ms");
        _output.WriteLine($"  PyTorch BDN: {pytorchMs}ms");
        _output.WriteLine($"  Ratio:       {bestEager / pytorchMs:F3}x");
    }

    [Fact]
    public unsafe void DoubleTanh_1M_EagerVsPyTorch()
    {
        const double pytorchMs = 1.025;
        var engine = new CpuEngine();
        int n = 1_000_000;
        var rng = new Random(42);
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = rng.NextDouble() * 10 - 5;
        var tensor = new Tensor<double>(data, new[] { n });

        for (int w = 0; w < 20; w++)
            engine.TensorTanh(tensor);

        double bestEager = double.MaxValue;
        for (int trial = 0; trial < 10; trial++)
        {
            var sw = Stopwatch.StartNew();
            for (int iter = 0; iter < 100; iter++)
                engine.TensorTanh(tensor);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds / 100;
            if (ms < bestEager) bestEager = ms;
        }

        _output.WriteLine($"Double Tanh 1M (best of 10):");
        _output.WriteLine($"  Eager:       {bestEager:F4}ms");
        _output.WriteLine($"  PyTorch BDN: {pytorchMs}ms");
        _output.WriteLine($"  Ratio:       {bestEager / pytorchMs:F3}x");
    }

    [Fact]
    public unsafe void DoubleGELU_1M_EagerVsPyTorch()
    {
        const double pytorchMs = 1.393;
        var engine = new CpuEngine();
        int n = 1_000_000;
        var rng = new Random(42);
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = rng.NextDouble() * 10 - 5;
        var tensor = new Tensor<double>(data, new[] { n });

        for (int w = 0; w < 20; w++)
            engine.TensorGELU(tensor);

        double bestEager = double.MaxValue;
        for (int trial = 0; trial < 10; trial++)
        {
            var sw = Stopwatch.StartNew();
            for (int iter = 0; iter < 100; iter++)
                engine.TensorGELU(tensor);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds / 100;
            if (ms < bestEager) bestEager = ms;
        }

        _output.WriteLine($"Double GELU 1M (best of 10):");
        _output.WriteLine($"  Eager:       {bestEager:F4}ms");
        _output.WriteLine($"  PyTorch BDN: {pytorchMs}ms");
        _output.WriteLine($"  Ratio:       {bestEager / pytorchMs:F3}x");
    }
}
