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
        var output = new float[length];

        var hIn = GCHandle.Alloc(input, GCHandleType.Pinned);
        var hOut = GCHandle.Alloc(output, GCHandleType.Pinned);
        float* pIn = (float*)hIn.AddrOfPinnedObject();
        float* pOut = (float*)hOut.AddrOfPinnedObject();

        var output2 = new float[length];
        var output3 = new float[length];
        var hOut2 = GCHandle.Alloc(output2, GCHandleType.Pinned);
        var hOut3 = GCHandle.Alloc(output3, GCHandleType.Pinned);
        float* pOut2 = (float*)hOut2.AddrOfPinnedObject();
        float* pOut3 = (float*)hOut3.AddrOfPinnedObject();

        // Heavy warmup both paths
        for (int w = 0; w < 20; w++)
        {
            SimdKernels.SigmoidUnsafe(pIn, pOut, length);
            TableDrivenSigmoid.SigmoidArray(pIn, pOut2, length);
        }

        // Path A: current exp+divide sigmoid
        var swA = Stopwatch.StartNew();
        for (int i = 0; i < 100; i++)
            SimdKernels.SigmoidUnsafe(pIn, pOut, length);
        swA.Stop();
        double msA = swA.Elapsed.TotalMilliseconds / 100;

        // Path B: table-driven sigmoid
        var swB = Stopwatch.StartNew();
        for (int i = 0; i < 100; i++)
            TableDrivenSigmoid.SigmoidArray(pIn, pOut2, length);
        swB.Stop();
        double msB = swB.Elapsed.TotalMilliseconds / 100;

        // Accuracy check
        float maxErr = 0f;
        for (int i = 0; i < length; i++)
        {
            float exact = 1f / (1f + MathF.Exp(-input[i]));
            maxErr = MathF.Max(maxErr, MathF.Abs(output2[i] - exact));
        }

        // Path C: Padé [2,2] sigmoid (fused exp+divide → single divide)
        for (int w = 0; w < 20; w++)
            PadeSigmoid.SigmoidArray(pIn, pOut3, length);
        var swC = Stopwatch.StartNew();
        for (int i = 0; i < 100; i++)
            PadeSigmoid.SigmoidArray(pIn, pOut3, length);
        swC.Stop();
        double msC = swC.Elapsed.TotalMilliseconds / 100;

        // Accuracy check for Padé
        float maxErrPade = 0f;
        for (int i = 0; i < length; i++)
        {
            float exact = 1f / (1f + MathF.Exp(-input[i]));
            maxErrPade = MathF.Max(maxErrPade, MathF.Abs(output3[i] - exact));
        }

        hIn.Free(); hOut.Free(); hOut2.Free(); hOut3.Free();

        _output.WriteLine($"Sigmoid 1M (100 iters, heavy warmup):");
        _output.WriteLine($"  Path A (Estrin exp+div): {msA:F4}ms");
        _output.WriteLine($"  Path B (table+quad):     {msB:F4}ms");
        _output.WriteLine($"  Path C (Pade fused):     {msC:F4}ms  (max err: {maxErrPade:E3})");
        _output.WriteLine($"  PyTorch BDN: 0.4880ms");
        _output.WriteLine($"  Path A ratio: {msA / 0.488:F3}x");
        _output.WriteLine($"  Path C ratio: {msC / 0.488:F3}x");
        _output.WriteLine($"  Pade vs Estrin speedup: {msA / msC:F2}x");
    }
}
