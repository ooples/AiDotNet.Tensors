using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmarks GPU convolution kernels: Conv2D forward, backward, depthwise.
/// Default: N=4, C=64, H=56, K=3x3 (ResNet-like configuration).
/// </summary>
public static class GpuConvolutionBenchmark
{
    public static void Run()
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("GPU CONVOLUTION BENCHMARK");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        DirectGpuEngine? engine = null;
        try
        {
            engine = new DirectGpuEngine();
            if (!engine.IsAvailable)
            {
                Console.WriteLine("[SKIP] DirectGpu not available.");
                return;
            }

            Console.WriteLine($"Backend: {engine.BackendName}");
            Console.WriteLine($"Device:  {engine.DeviceName}");
            Console.WriteLine();

            var backend = engine.Backend;
            if (backend == null)
            {
                Console.WriteLine("[ERROR] Could not access backend.");
                return;
            }

            Console.WriteLine($"{"Config",-45} {"Time(ms)",10} {"GFLOPS",10}");
            Console.WriteLine(new string('-', 67));

            var configs = new (int N, int C, int H, int W, int outC, int kH, int kW, int stride, int pad, string desc)[]
            {
                (4, 64, 56, 56, 64, 3, 3, 1, 1, "ResNet conv3x3 s1p1"),
                (4, 64, 56, 56, 64, 3, 3, 1, 0, "Conv3x3 no-pad"),
                (4, 64, 56, 56, 128, 1, 1, 1, 0, "1x1 projection"),
                (4, 128, 28, 28, 128, 3, 3, 1, 1, "ResNet stage2"),
                (4, 256, 14, 14, 256, 3, 3, 1, 1, "ResNet stage3"),
                (4, 512, 7, 7, 512, 3, 3, 1, 1, "ResNet stage4"),
                (16, 64, 56, 56, 64, 3, 3, 1, 1, "Batch=16 conv3x3"),
            };

            foreach (var (N, C, H, W, outC, kH, kW, stride, pad, desc) in configs)
            {
                BenchmarkConv2DForward(backend, N, C, H, W, outC, kH, kW, stride, pad, desc);
            }

            Console.WriteLine();

            Console.WriteLine("--- Conv2D Backward ---");
            Console.WriteLine($"{"Config",-45} {"Time(ms)",10} {"GFLOPS",10}");
            Console.WriteLine(new string('-', 67));

            BenchmarkConv2DBackward(backend, 4, 64, 56, 56, 64, 3, 3, 1, 1, "Backward input");
            BenchmarkConv2DBackwardKernel(backend, 4, 64, 56, 56, 64, 3, 3, 1, 1, "Backward kernel");

            Console.WriteLine();

            Console.WriteLine("--- Depthwise Conv2D ---");
            Console.WriteLine($"{"Config",-45} {"Time(ms)",10} {"GFLOPS",10}");
            Console.WriteLine(new string('-', 67));

            BenchmarkDepthwiseConv2D(backend, 4, 64, 56, 56, 3, 3, 1, 1, "DW conv3x3");
            BenchmarkDepthwiseConv2D(backend, 4, 128, 28, 28, 3, 3, 1, 1, "DW conv3x3 stage2");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {ex.Message}");
        }
        finally
        {
            engine?.Dispose();
        }
    }

    private static void BenchmarkConv2DForward(IDirectGpuBackend backend, int N, int C, int H, int W,
        int outC, int kH, int kW, int stride, int pad, string desc)
    {
        int outH = (H + 2 * pad - kH) / stride + 1;
        int outW = (W + 2 * pad - kW) / stride + 1;
        int inputSize = N * C * H * W;
        int kernelSize = outC * C * kH * kW;
        int outputSize = N * outC * outH * outW;

        var rand = new Random(42);
        var input = CreateRandom(inputSize, rand);
        var kernel = CreateRandom(kernelSize, rand, 0.1f);

        using var bufIn = backend.AllocateBuffer(input);
        using var bufKernel = backend.AllocateBuffer(kernel);
        using var bufOut = backend.AllocateBuffer(outputSize);

        // Warmup
        for (int i = 0; i < 3; i++)
            backend.Conv2D(bufIn, bufKernel, bufOut, N, C, H, W, outC, outH, outW, kH, kW, stride, stride, pad, pad, 1, 1);
        backend.Synchronize();

        int runs = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.Conv2D(bufIn, bufKernel, bufOut, N, C, H, W, outC, outH, outW, kH, kW, stride, stride, pad, pad, 1, 1);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double flops = 2.0 * N * outC * outH * outW * C * kH * kW;
        double gflops = flops / (avgMs * 1e6);

        Console.WriteLine($"{desc + $" ({outH}x{outW})",-45} {avgMs,10:F3} {gflops,10:F1}");
    }

    private static void BenchmarkConv2DBackward(IDirectGpuBackend backend, int N, int C, int H, int W,
        int outC, int kH, int kW, int stride, int pad, string desc)
    {
        int outH = (H + 2 * pad - kH) / stride + 1;
        int outW = (W + 2 * pad - kW) / stride + 1;
        int gradOutSize = N * outC * outH * outW;
        int kernelSize = outC * C * kH * kW;
        int gradInSize = N * C * H * W;

        var rand = new Random(42);
        var gradOut = CreateRandom(gradOutSize, rand);
        var kernel = CreateRandom(kernelSize, rand, 0.1f);

        using var bufGradOut = backend.AllocateBuffer(gradOut);
        using var bufKernel = backend.AllocateBuffer(kernel);
        using var bufGradIn = backend.AllocateBuffer(gradInSize);

        for (int i = 0; i < 3; i++)
            backend.Conv2DBackwardInput(bufGradOut, bufKernel, bufGradIn, N, C, H, W, outC, outH, outW, kH, kW, stride, stride, pad, pad, 1, 1);
        backend.Synchronize();

        int runs = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.Conv2DBackwardInput(bufGradOut, bufKernel, bufGradIn, N, C, H, W, outC, outH, outW, kH, kW, stride, stride, pad, pad, 1, 1);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double flops = 2.0 * N * outC * outH * outW * C * kH * kW;
        double gflops = flops / (avgMs * 1e6);

        Console.WriteLine($"{desc,-45} {avgMs,10:F3} {gflops,10:F1}");
    }

    private static void BenchmarkConv2DBackwardKernel(IDirectGpuBackend backend, int N, int C, int H, int W,
        int outC, int kH, int kW, int stride, int pad, string desc)
    {
        int outH = (H + 2 * pad - kH) / stride + 1;
        int outW = (W + 2 * pad - kW) / stride + 1;
        int inputSize = N * C * H * W;
        int gradOutSize = N * outC * outH * outW;
        int gradKernelSize = outC * C * kH * kW;

        var rand = new Random(42);
        var input = CreateRandom(inputSize, rand);
        var gradOut = CreateRandom(gradOutSize, rand);

        using var bufIn = backend.AllocateBuffer(input);
        using var bufGradOut = backend.AllocateBuffer(gradOut);
        using var bufGradKernel = backend.AllocateBuffer(gradKernelSize);

        for (int i = 0; i < 3; i++)
            backend.Conv2DBackwardKernel(bufIn, bufGradOut, bufGradKernel, N, C, H, W, outC, outH, outW, kH, kW, stride, stride, pad, pad, 1, 1);
        backend.Synchronize();

        int runs = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.Conv2DBackwardKernel(bufIn, bufGradOut, bufGradKernel, N, C, H, W, outC, outH, outW, kH, kW, stride, stride, pad, pad, 1, 1);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double flops = 2.0 * N * outC * outH * outW * C * kH * kW;
        double gflops = flops / (avgMs * 1e6);

        Console.WriteLine($"{desc,-45} {avgMs,10:F3} {gflops,10:F1}");
    }

    private static void BenchmarkDepthwiseConv2D(IDirectGpuBackend backend, int N, int C, int H, int W,
        int kH, int kW, int stride, int pad, string desc)
    {
        int outH = (H + 2 * pad - kH) / stride + 1;
        int outW = (W + 2 * pad - kW) / stride + 1;
        int inputSize = N * C * H * W;
        int kernelSize = C * kH * kW; // depthwise: 1 filter per channel
        int outputSize = N * C * outH * outW;

        var rand = new Random(42);
        var input = CreateRandom(inputSize, rand);
        var kernel = CreateRandom(kernelSize, rand, 0.1f);

        using var bufIn = backend.AllocateBuffer(input);
        using var bufKernel = backend.AllocateBuffer(kernel);
        using var bufOut = backend.AllocateBuffer(outputSize);

        for (int i = 0; i < 3; i++)
            backend.DepthwiseConv2D(bufIn, bufKernel, bufOut, N, C, H, W, outH, outW, kH, kW, stride, stride, pad, pad);
        backend.Synchronize();

        int runs = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.DepthwiseConv2D(bufIn, bufKernel, bufOut, N, C, H, W, outH, outW, kH, kW, stride, stride, pad, pad);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        // Depthwise FLOPs: 2 * N * C * outH * outW * kH * kW (no cross-channel)
        double flops = 2.0 * N * C * outH * outW * kH * kW;
        double gflops = flops / (avgMs * 1e6);

        Console.WriteLine($"{desc + $" ({outH}x{outW})",-45} {avgMs,10:F3} {gflops,10:F1}");
    }

    private static float[] CreateRandom(int size, Random rand, float scale = 1.0f)
    {
        var arr = new float[size];
        for (int i = 0; i < size; i++)
            arr[i] = (float)(rand.NextDouble() - 0.5) * 2 * scale;
        return arr;
    }
}
