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

    // The wiring: both slots register the Winograd candidate, never as the reference, and the forward slot's measured
    // choice for ResNet-18's 128@16x16 layer (where Winograd is ~1.5x faster) is Winograd.
    [SkippableFact]
    public void The_conv_slots_register_Winograd_and_pick_it_for_a_mid_resnet_layer()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = (DirectGpuTensorEngine)_fixture.Engine!;
        var cuda = gpu.GetBackend() as CudaBackend;
        Skip.If(cuda is null, "CUDA only.");
        Skip.IfNot(cuda!.EnsureWinogradModule(), "Winograd kernels did not compile.");

        var forwardIds = cuda.Conv2DForwardSlot.CandidateIds;
        var dataIds = cuda.Conv2DBackwardDataSlot.CandidateIds;
        Assert.Contains("nvrtc.conv2d.winograd_f23", forwardIds);
        Assert.Contains("nvrtc.conv2d_dx.winograd_f23", dataIds);
        Assert.NotEqual("nvrtc.conv2d.winograd_f23", forwardIds[0]);
        Assert.NotEqual("nvrtc.conv2d_dx.winograd_f23", dataIds[0]);
        var filterIds = cuda.Conv2DBackwardFilterSlot.CandidateIds;
        Assert.Contains("nvrtc.conv2d_dw.winograd_f32", filterIds);
        Assert.NotEqual("nvrtc.conv2d_dw.winograd_f32", filterIds[0]);

        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_KERNEL_REGISTRY_PERSIST") != "0",
            "The measured choice is checked with persisted tuning off (AIDOTNET_KERNEL_REGISTRY_PERSIST=0).");
        int batch = 8, c = 128, s = 16;
        using var x = cuda.AllocateBuffer(Rand(batch * c * s * s, 1, 1f));
        using var w = cuda.AllocateBuffer(Rand(c * c * 9, 2, 1f / MathF.Sqrt(c * 9)));
        using var y = cuda.AllocateBuffer(batch * c * s * s);
        var args = new CudaConv2DArgs(x, w, y, batch, c, s, s, c, s, s, 3, 3, 1, 1, 1, 1, 1, 1);
        // The per-process tuning budget is shared with every earlier test in this process; once spent the slot serves
        // its reference without measuring. Give these decisions room to measure, then restore the budget.
        // The slot's gate rejects a candidate whose min-pair ratio is inside the run's noise band even when its median
        // wins (by design: a noisy win is not shipped), and a ~70 us kernel can lose one such round on a shared GPU. So
        // the decision gets up to three fresh measurements and Winograd must win one of them.
        double budget = AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.BudgetMilliseconds;
        AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.BudgetMilliseconds =
            AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.SpentMilliseconds + 60_000;
        AiDotNet.Tensors.Helpers.Autotune.TunedKernels.ITunedKernelCandidate<CudaConv2DArgs>? chosen = null;
        try
        {
            for (int attempt = 1; attempt <= 3 && chosen?.Id != "nvrtc.conv2d.winograd_f23"; attempt++)
            {
                // An earlier test (or attempt) may have cached a decision for this shape.
                cuda.Conv2DForwardSlot.InvalidateDecisions();
                chosen = cuda.Conv2DForwardSlot.Resolve(args.Shape(), args);
                var decision = AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelRegistry.Decisions
                    .LastOrDefault(d => d.Op == AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelOp.Conv2DForward
                                        && d.Shape.Equals(args.Shape()));
                _output.WriteLine($"attempt {attempt}: forward slot picked {chosen?.Id ?? "<none>"} for N8 C128 16x16 (reason {decision?.Reason}, " +
                                  $"reference {decision?.ReferenceMilliseconds:F3} ms, chosen {decision?.CandidateMilliseconds:F3} ms, " +
                                  $"max rel err {decision?.MaxRelativeError:E2}) among [{string.Join(", ", forwardIds)}]; " +
                                  $"rejected [{string.Join("; ", decision?.Rejected ?? Array.Empty<string>())}]");
            }
        }
        finally
        {
            AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.BudgetMilliseconds = budget;
        }
        Assert.Equal("nvrtc.conv2d.winograd_f23", chosen?.Id);
    }

    // Weight gradient through Winograd F(3x3, 2x2) against the backend's tuned weight-gradient route. The reduction runs over
    // every output pixel (batch * outH * outW), so the comparison allows the longer sum's rounding.
    [SkippableTheory]
    [InlineData(8, 64, 64, 32, 1)]
    [InlineData(8, 128, 128, 16, 1)]
    [InlineData(8, 256, 256, 8, 1)]
    [InlineData(8, 512, 512, 4, 1)]
    [InlineData(3, 5, 7, 9, 1)]
    [InlineData(2, 16, 24, 11, 0)]
    public void WinogradF32_weight_gradient_matches_the_backend(int batch, int inC, int outC, int size, int pad)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = (DirectGpuTensorEngine)_fixture.Engine!;
        var cuda = gpu.GetBackend() as CudaBackend;
        Skip.If(cuda is null, "CUDA only.");
        Skip.IfNot(cuda!.EnsureWinogradModule(), "Winograd kernels did not compile.");

        int outSize = size + 2 * pad - 2;
        using var x = cuda.AllocateBuffer(Rand(batch * inC * size * size, 5, 1f));
        using var gy = cuda.AllocateBuffer(Rand(batch * outC * outSize * outSize, 6, 1f));
        using var dwRef = cuda.AllocateBuffer(outC * inC * 9);
        using var dwWin = cuda.AllocateBuffer(outC * inC * 9);
        void Reference() => cuda.Conv2DBackwardKernel(x, gy, dwRef, batch, inC, size, size, outC, outSize, outSize, 3, 3, 1, 1, pad, pad, 1, 1);
        var winArgs = new CudaConv2DBackwardArgs(x, gy, dwWin, true, batch, inC, size, size, outC, outSize, outSize, 3, 3, 1, 1, pad, pad, 1, 1);
        Reference();
        Assert.True(cuda.TryConv2DBackwardFilterWinogradF32(winArgs), "Winograd refused an admitted weight-gradient shape");

        var expected = cuda.DownloadBuffer(dwRef);
        var actual = cuda.DownloadBuffer(dwWin);
        double scale = expected.Max(v => Math.Abs((double)v)), maxErr = 0;
        for (int i = 0; i < expected.Length; i++)
            maxErr = Math.Max(maxErr, Math.Abs(expected[i] - actual[i]) / Math.Max(1.0, scale));
        _output.WriteLine($"[dW N{batch} C{inC}->K{outC} {size}x{size} pad{pad}] max err / max|dW| vs backend {maxErr:E2}");
        Assert.True(maxErr < 1e-5, $"Winograd F(3,2) weight gradient disagrees: max err / max|dW| {maxErr:E2}");

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
        double winUs = Time(() => cuda.TryConv2DBackwardFilterWinogradF32(winArgs));
        _output.WriteLine($"[dW N{batch} C{inC}->K{outC} {size}x{size}] backend {backendUs:F1} us, winograd F(3,2) {winUs:F1} us ({backendUs / winUs:F2}x)");
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
