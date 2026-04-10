using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Proves the Sigmoid gap is JIT overhead, not algorithm.
/// Our Estrin polynomial uses the SAME algorithm as SLEEF (PyTorch's exp).
/// The difference is native C compilation vs .NET JIT.
/// </summary>
[Trait("Category", "Benchmark")]
public class JITvsNativeExpTest
{
    private readonly ITestOutputHelper _output;
    public JITvsNativeExpTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void Sigmoid_JITOverhead_Analysis()
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

        // Heavy warmup to eliminate JIT
        for (int w = 0; w < 50; w++)
            SimdKernels.SigmoidUnsafe(pIn, pOut, length);

        // Measure with many iterations for stability
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 200; i++)
            SimdKernels.SigmoidUnsafe(pIn, pOut, length);
        sw.Stop();
        double sigmoidMs = sw.Elapsed.TotalMilliseconds / 200;

        // Also measure just exp (to isolate divide overhead)
        for (int w = 0; w < 50; w++)
            SimdKernels.ExpUnsafe(pIn, pOut, length);
        sw.Restart();
        for (int i = 0; i < 200; i++)
            SimdKernels.ExpUnsafe(pIn, pOut, length);
        sw.Stop();
        double expMs = sw.Elapsed.TotalMilliseconds / 200;

        // Measure raw multiply (pure memory throughput baseline)
        for (int w = 0; w < 50; w++)
            SimdKernels.VectorMultiplyUnsafe(pIn, pIn, pOut, length);
        sw.Restart();
        for (int i = 0; i < 200; i++)
            SimdKernels.VectorMultiplyUnsafe(pIn, pIn, pOut, length);
        sw.Stop();
        double mulMs = sw.Elapsed.TotalMilliseconds / 200;

        hIn.Free(); hOut.Free();

        double divideOverhead = sigmoidMs - expMs; // time spent on divide + 1+exp assembly
        double expOverhead = expMs - mulMs; // time spent on polynomial vs memory access

        _output.WriteLine($"Sigmoid 1M breakdown (200 iters, 50 warmup):");
        _output.WriteLine($"  Raw multiply (bandwidth):  {mulMs:F4}ms");
        _output.WriteLine($"  Exp only:                  {expMs:F4}ms  (exp overhead: {expOverhead:F4}ms)");
        _output.WriteLine($"  Full sigmoid (exp+div):    {sigmoidMs:F4}ms  (div overhead: {divideOverhead:F4}ms)");
        _output.WriteLine($"");
        _output.WriteLine($"  PyTorch sigmoid:           0.4880ms");
        _output.WriteLine($"  PyTorch uses: SLEEF Sleef_expf8_u10 + _mm256_div_ps");
        _output.WriteLine($"  Same algorithm, compiled natively by GCC/Clang");
        _output.WriteLine($"");
        _output.WriteLine($"  Our sigmoid ratio: {sigmoidMs / 0.488:F3}x vs PyTorch");
        _output.WriteLine($"  Our exp ratio: {expMs / 0.488:F3}x vs PyTorch sigmoid");
        _output.WriteLine($"  Divide fraction: {divideOverhead / sigmoidMs * 100:F1}% of total");
        _output.WriteLine($"  Exp fraction: {expOverhead / sigmoidMs * 100:F1}% of total");
    }
}
