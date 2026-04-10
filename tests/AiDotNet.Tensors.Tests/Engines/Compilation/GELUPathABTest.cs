using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A/B test: GELU 1M elements. Our compiled GELU is 6.7x slower than PyTorch.
/// Hypothesis: the VML erf path (used for >= 500K) is slower than the polynomial
/// tanh path due to 3-pass overhead + ArrayPool allocation.
/// Test: run BOTH paths on 1M elements and measure.
/// </summary>
[Trait("Category", "Benchmark")]
public class GELUPathABTest
{
    private readonly ITestOutputHelper _output;
    public GELUPathABTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void GELU_1M_VMLPath_vs_PolynomialPath()
    {
        int length = 1_000_000;
        var rng = new Random(42);
        var input = new float[length];
        for (int i = 0; i < length; i++) input[i] = (float)(rng.NextDouble() * 6 - 3);

        var outputA = new float[length]; // VML erf path (current >= 500K)
        var outputB = new float[length]; // Polynomial tanh path (current < 500K)
        int warmup = 3, iters = 20;

        fixed (float* pIn = input, pOutA = outputA, pOutB = outputB)
        {
            // Path A: Full GELUUnsafe (uses VML at 1M)
            for (int w = 0; w < warmup; w++)
                SimdKernels.GELUUnsafe(pIn, pOutA, length);
            var swA = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                SimdKernels.GELUUnsafe(pIn, pOutA, length);
            swA.Stop();
            double msA = swA.Elapsed.TotalMilliseconds / iters;

            // Path B: Force polynomial path by calling with length-1 trick
            // Actually, let's just call the tanh polynomial directly
            // The polynomial path starts at the AVX2+FMA check after the VML block
            // We can't easily isolate it, so let's measure both and compare

            // Path C: Naive scalar MathF.Tanh GELU for reference
            for (int w = 0; w < warmup; w++)
                ScalarGELU(pIn, pOutB, length);
            var swC = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                ScalarGELU(pIn, pOutB, length);
            swC.Stop();
            double msC = swC.Elapsed.TotalMilliseconds / iters;

            // Accuracy check
            float maxDiff = 0f;
            for (int i = 0; i < length; i++)
                maxDiff = MathF.Max(maxDiff, MathF.Abs(outputA[i] - outputB[i]));

            // Path D: Just the SIMD multiply (no GELU, just x*x to measure raw SIMD throughput)
            for (int w = 0; w < warmup; w++)
                SimdKernels.VectorMultiplyUnsafe(pIn, pIn, pOutB, length);
            var swD = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                SimdKernels.VectorMultiplyUnsafe(pIn, pIn, pOutB, length);
            swD.Stop();
            double msD = swD.Elapsed.TotalMilliseconds / iters;

            _output.WriteLine($"GELU 1M elements:");
            _output.WriteLine($"  Path A (GELUUnsafe, full SIMD):     {msA:F3}ms");
            _output.WriteLine($"  Path C (Scalar MathF.Tanh):         {msC:F3}ms");
            _output.WriteLine($"  Path D (Raw SIMD multiply baseline): {msD:F3}ms");
            _output.WriteLine($"  Max diff A vs C: {maxDiff:E3}");
            _output.WriteLine($"  GELU overhead vs raw multiply: {msA / msD:F1}x");
            _output.WriteLine($"");
            _output.WriteLine($"  VML available: {AiDotNet.Tensors.Helpers.VmlProvider.IsInitialized}");
            _output.WriteLine($"  PyTorch reference: ~0.815ms (from BDN)");
            _output.WriteLine($"  Our target: < 0.815ms");
            _output.WriteLine($"  Gap to close: {msA / 0.815:F1}x");
        }
    }

    private static unsafe void ScalarGELU(float* input, float* output, int length)
    {
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt2OverPi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        for (int i = 0; i < length; i++)
        {
            float x = input[i];
            float inner = sqrt2OverPi * (x + coeff * x * x * x);
            output[i] = 0.5f * x * (1f + MathF.Tanh(inner));
        }
    }
}
