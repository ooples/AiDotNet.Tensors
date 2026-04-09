using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public class PreciseSigmoidTest
{
    private readonly ITestOutputHelper _output;
    public PreciseSigmoidTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void Sigmoid_1M_HighPrecisionTiming()
    {
        int length = 1_000_000;
        var rng = new Random(42);
        var input = new float[length];
        for (int i = 0; i < length; i++) input[i] = (float)(rng.NextDouble() * 20 - 10);
        var output1 = new float[length];
        var output2 = new float[length];

        var hIn = GCHandle.Alloc(input, GCHandleType.Pinned);
        var hOut1 = GCHandle.Alloc(output1, GCHandleType.Pinned);
        var hOut2 = GCHandle.Alloc(output2, GCHandleType.Pinned);
        float* pIn = (float*)hIn.AddrOfPinnedObject();
        float* pOut1 = (float*)hOut1.AddrOfPinnedObject();
        float* pOut2 = (float*)hOut2.AddrOfPinnedObject();

        // Set high priority and GC collect before measurement
        var oldPriority = Thread.CurrentThread.Priority;
        Thread.CurrentThread.Priority = ThreadPriority.Highest;
        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true);

        // Very heavy warmup (50 iters)
        for (int w = 0; w < 50; w++)
        {
            PadeSigmoid.SigmoidArray(pIn, pOut1, length);
            SimdKernels.SigmoidUnsafe(pIn, pOut2, length);
        }

        // Measure 10 trials of 200 iterations, take best
        double bestDirect = double.MaxValue;
        double bestUnsafe = double.MaxValue;
        const int iters = 200;
        for (int trial = 0; trial < 10; trial++)
        {
            var sw1 = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                PadeSigmoid.SigmoidArray(pIn, pOut1, length);
            sw1.Stop();
            double ms1 = sw1.Elapsed.TotalMilliseconds / iters;
            if (ms1 < bestDirect) bestDirect = ms1;

            var sw2 = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                SimdKernels.SigmoidUnsafe(pIn, pOut2, length);
            sw2.Stop();
            double ms2 = sw2.Elapsed.TotalMilliseconds / iters;
            if (ms2 < bestUnsafe) bestUnsafe = ms2;
        }

        Thread.CurrentThread.Priority = oldPriority;

        // Accuracy
        float maxErr = 0f;
        for (int i = 0; i < length; i++)
        {
            float exact = 1f / (1f + MathF.Exp(-input[i]));
            maxErr = MathF.Max(maxErr, MathF.Abs(output1[i] - exact));
        }

        hIn.Free(); hOut1.Free(); hOut2.Free();

        _output.WriteLine($"Sigmoid 1M (200 iters x 10 trials, best of 10):");
        _output.WriteLine($"  Pade direct:    {bestDirect:F4}ms  (max err: {maxErr:E3})");
        _output.WriteLine($"  SigmoidUnsafe:  {bestUnsafe:F4}ms");
        _output.WriteLine($"  PyTorch BDN:    0.4880ms");
        _output.WriteLine($"  Direct ratio:   {bestDirect / 0.488:F3}x");
        _output.WriteLine($"  Unsafe ratio:   {bestUnsafe / 0.488:F3}x");
    }

    [Fact]
    public unsafe void Sum_1M_8WayAccumulation()
    {
        int length = 1_000_000;
        var rng = new Random(42);
        var input = new float[length];
        for (int i = 0; i < length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var hIn = GCHandle.Alloc(input, GCHandleType.Pinned);
        float* pIn = (float*)hIn.AddrOfPinnedObject();

        var oldPriority = Thread.CurrentThread.Priority;
        Thread.CurrentThread.Priority = ThreadPriority.Highest;
        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true);

        // Warmup
        for (int w = 0; w < 50; w++)
            SimdKernels.SumUnsafe(pIn, length);

        // 10 trials of 1000 iters, best
        double bestSum = double.MaxValue;
        for (int trial = 0; trial < 10; trial++)
        {
            var sw = Stopwatch.StartNew();
            float s = 0f;
            for (int i = 0; i < 1000; i++)
                s = SimdKernels.SumUnsafe(pIn, length);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds / 1000;
            if (ms < bestSum) bestSum = ms;
        }

        Thread.CurrentThread.Priority = oldPriority;

        // Accuracy
        double exactSum = 0;
        for (int i = 0; i < length; i++) exactSum += input[i];
        float simdSum = SimdKernels.SumUnsafe(pIn, length);
        double relErr = Math.Abs(simdSum - exactSum) / Math.Abs(exactSum);

        hIn.Free();

        double bwGBs = (length * 4.0 / 1e9) / (bestSum / 1000.0);
        _output.WriteLine($"Sum 1M (1000 iters x 10 trials, best of 10):");
        _output.WriteLine($"  8-way SIMD:   {bestSum:F4}ms");
        _output.WriteLine($"  PyTorch BDN:  0.089ms (Mean includes /N)");
        _output.WriteLine($"  Ratio:        {bestSum / 0.089:F3}x");
        _output.WriteLine($"  Rel error:    {relErr:E3}");
        _output.WriteLine($"  Bandwidth:    {bwGBs:F1} GB/s");
    }
}
