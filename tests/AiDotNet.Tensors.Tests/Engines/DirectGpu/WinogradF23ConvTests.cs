using System;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Winograd F(2x2, 3x3) forward convolution against the implicit-GEMM engine on the ResNet-18 (CIFAR, batch 8) stage
/// shapes plus odd sizes and channel counts: same result to fp32 Winograd rounding, and timed kernel-only (50 launches,
/// one sync), the protocol the torch per-conv numbers use. Timing is reported, not gated.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class WinogradF23ConvTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly ITestOutputHelper _output;

    public WinogradF23ConvTests(DirectGpuTensorEngineTestFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    private static float[] Rand(int n, int seed, float scale)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
    }

    [SkippableTheory]
    [InlineData(8, 64, 64, 32, 1)]
    [InlineData(8, 128, 128, 16, 1)]
    [InlineData(8, 256, 256, 8, 1)]
    [InlineData(8, 512, 512, 4, 1)]
    [InlineData(3, 5, 7, 9, 1)]      // odd everything: tile edges, non-multiple-of-8 channels
    [InlineData(2, 16, 24, 11, 0)]   // no padding: output 9x9
    public void WinogradF23_matches_the_engine(int batch, int inC, int outC, int size, int pad)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = (DirectGpuTensorEngine)_fixture.Engine!;
        var cuda = gpu.GetBackend() as CudaBackend;
        Skip.If(cuda is null, "CUDA only.");
        Skip.IfNot(cuda!.EnsureWinogradModule(), "Winograd kernels did not compile.");

        int outSize = size + 2 * pad - 2;
        using var x = cuda.AllocateBuffer(Rand(batch * inC * size * size, 1, 1f));
        using var w = cuda.AllocateBuffer(Rand(outC * inC * 9, 2, 1f / MathF.Sqrt(inC * 9)));
        using var yRef = cuda.AllocateBuffer(batch * outC * outSize * outSize);
        using var yWin = cuda.AllocateBuffer(batch * outC * outSize * outSize);
        var refArgs = new CudaConv2DArgs(x, w, yRef, batch, inC, size, size, outC, outSize, outSize, 3, 3, 1, 1, pad, pad, 1, 1);
        var winArgs = new CudaConv2DArgs(x, w, yWin, batch, inC, size, size, outC, outSize, outSize, 3, 3, 1, 1, pad, pad, 1, 1);
        Assert.True(cuda.TryConv2DForwardEngine(refArgs), "engine refused the reference shape");
        Assert.True(cuda.TryConv2DForwardWinogradF23(winArgs), "Winograd refused an admitted shape");

        var expected = cuda.DownloadBuffer(yRef);
        var actual = cuda.DownloadBuffer(yWin);
        double maxErr = 0;
        for (int i = 0; i < expected.Length; i++)
            maxErr = Math.Max(maxErr, Math.Abs(expected[i] - actual[i]) / Math.Max(1.0, Math.Abs(expected[i])));
        _output.WriteLine($"[N{batch} C{inC}->K{outC} {size}x{size} pad{pad}] max rel err vs engine {maxErr:E2}");
        Assert.True(maxErr < 1e-4, $"Winograd F(2,3) disagrees with the engine: max rel err {maxErr:E2}");

        double Time(Func<bool> launch)
        {
            for (int i = 0; i < 10; i++) launch();
            gpu.SynchronizeStream();
            double best = double.MaxValue;
            for (int r = 0; r < 7; r++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < 50; i++) launch();
                gpu.SynchronizeStream();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000 / 50);
            }
            return best;
        }
        double engineUs = Time(() => cuda.TryConv2DForwardEngine(refArgs));
        double winUs = Time(() => cuda.TryConv2DForwardWinogradF23(winArgs));
        _output.WriteLine($"[N{batch} C{inC}->K{outC} {size}x{size}] engine {engineUs:F1} us, winograd F(2,3) {winUs:F1} us ({engineUs / winUs:F2}x)");
    }

    // Input gradient through the flipped-filter Winograd pipeline against the backend's tuned input-gradient route.
    [SkippableTheory]
    [InlineData(8, 64, 64, 32, 1)]
    [InlineData(8, 128, 128, 16, 1)]
    [InlineData(8, 256, 256, 8, 1)]
    [InlineData(3, 5, 7, 9, 1)]
    [InlineData(2, 16, 24, 11, 0)]
    public void WinogradF23_input_gradient_matches_the_backend(int batch, int inC, int outC, int size, int pad)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = (DirectGpuTensorEngine)_fixture.Engine!;
        var cuda = gpu.GetBackend() as CudaBackend;
        Skip.If(cuda is null, "CUDA only.");
        Skip.IfNot(cuda!.EnsureWinogradModule(), "Winograd kernels did not compile.");

        int outSize = size + 2 * pad - 2;
        using var gy = cuda.AllocateBuffer(Rand(batch * outC * outSize * outSize, 3, 1f));
        using var w = cuda.AllocateBuffer(Rand(outC * inC * 9, 4, 1f / MathF.Sqrt(outC * 9)));
        using var dxRef = cuda.AllocateBuffer(batch * inC * size * size);
        using var dxWin = cuda.AllocateBuffer(batch * inC * size * size);
        void Reference() => cuda.Conv2DBackwardInput(gy, w, dxRef, batch, inC, size, size, outC, outSize, outSize, 3, 3, 1, 1, pad, pad, 1, 1);
        var winArgs = new CudaConv2DBackwardArgs(gy, w, dxWin, false, batch, inC, size, size, outC, outSize, outSize, 3, 3, 1, 1, pad, pad, 1, 1);
        Reference();
        Assert.True(cuda.TryConv2DBackwardDataWinogradF23(winArgs), "Winograd refused an admitted input-gradient shape");

        var expected = cuda.DownloadBuffer(dxRef);
        var actual = cuda.DownloadBuffer(dxWin);
        double maxErr = 0;
        for (int i = 0; i < expected.Length; i++)
            maxErr = Math.Max(maxErr, Math.Abs(expected[i] - actual[i]) / Math.Max(1.0, Math.Abs(expected[i])));
        _output.WriteLine($"[dX N{batch} C{inC}<-K{outC} {size}x{size} pad{pad}] max rel err vs backend {maxErr:E2}");
        Assert.True(maxErr < 1e-4, $"Winograd F(2,3) input gradient disagrees: max rel err {maxErr:E2}");

        double Time(Action launch)
        {
            for (int i = 0; i < 10; i++) launch();
            gpu.SynchronizeStream();
            double best = double.MaxValue;
            for (int r = 0; r < 7; r++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < 50; i++) launch();
                gpu.SynchronizeStream();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000 / 50);
            }
            return best;
        }
        double backendUs = Time(Reference);
        double winUs = Time(() => cuda.TryConv2DBackwardDataWinogradF23(winArgs));
        _output.WriteLine($"[dX N{batch} C{inC}<-K{outC} {size}x{size}] backend {backendUs:F1} us, winograd F(2,3) {winUs:F1} us ({backendUs / winUs:F2}x)");
    }
}
