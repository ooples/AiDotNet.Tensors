using System;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// The cached-B GEMMs and the fused bias+GELU epilogue on a CPU without AVX2 or FMA (#1542).
/// </summary>
/// <remarks>
/// <para>
/// Both used to call AVX2/FMA intrinsics with no check of their own, so on such a CPU they threw
/// PlatformNotSupportedException instead of taking a portable path. Whether the JIT reports an instruction set is
/// fixed per process, so no test can switch it off in place: build.yml runs this class a second time in a host
/// started with <c>DOTNET_EnableAVX2=0</c> and <c>DOTNET_EnableFMA=0</c>, where these exercise the fallbacks. In the
/// ordinary run they check the vector paths against the same references.
/// </para>
/// </remarks>
public class InstructionSetFallbackTests
{
    private readonly ITestOutputHelper _output;

    public InstructionSetFallbackTests(ITestOutputHelper output)
    {
        _output = output;
        _output.WriteLine($"Avx={Avx.IsSupported} Avx2={Avx2.IsSupported} Fma={Fma.IsSupported}");
    }

    private static float[] Values(int length, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (float)(random.NextDouble() * 2 - 1);
        return values;
    }

    private static double[] Reference(float[] a, float[] b, int m, int k, int n)
    {
        var c = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++) sum += (double)a[i * k + p] * b[p * n + j];
                c[i * n + j] = sum;
            }
        return c;
    }

    [Theory]
    [InlineData(32, 48, 64)]
    [InlineData(7, 33, 129)]   // ragged: a tail in every dimension
    public void CachedBGemm_MatchesTheReference(int m, int k, int n)
    {
        var a = Values(m * k, 1);
        var b = Values(k * n, 2);
        var c = new float[m * n];

        SimdGemm.SgemmWithCachedB(a, b, c, m, k, n);

        var expected = Reference(a, b, m, k, n);
        for (int i = 0; i < c.Length; i++)
            Assert.True(Math.Abs(c[i] - expected[i]) <= 1e-4 * (1 + Math.Abs(expected[i])),
                $"C[{i}] = {c[i]}, expected {expected[i]}");
    }

    [Fact]
    public void Int8CachedBGemm_MatchesTheReferenceWithinQuantization()
    {
        const int m = 16, k = 64, n = 96;
        var a = Values(m * k, 3);
        var b = Values(k * n, 4);
        var c = new float[m * n];

        SimdGemm.SgemmWithInt8CachedB(a, b, c, m, k, n);

        // B is quantized to 8 bits: each product carries up to half a step of its column's scale, so the error grows
        // with k. 5% of the largest magnitude bounds it with room to spare, and is far below what a wrong kernel gives.
        var expected = Reference(a, b, m, k, n);
        double largest = 0;
        foreach (double value in expected) largest = Math.Max(largest, Math.Abs(value));
        for (int i = 0; i < c.Length; i++)
            Assert.True(Math.Abs(c[i] - expected[i]) <= 0.05 * largest, $"C[{i}] = {c[i]}, expected {expected[i]}");
    }

    [Theory]
    [InlineData(64, 128)]    // the vector epilogue, serial
    [InlineData(128, 512)]   // the vector epilogue, split across threads
    public void FusedBiasGelu_MatchesTheReference(int m, int n)
    {
        var output = Values(m * n, 5);
        for (int i = 0; i < output.Length; i++) output[i] *= 3;
        var bias = Values(n, 6);
        var expected = new double[output.Length];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double x = output[i * n + j] + bias[j];
                expected[i * n + j] = 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
            }

        CpuFusedOperations.ApplyBiasActivationInPlace(output, bias, m, n, FusedActivationType.GELU);

        // The vector kernels evaluate the tanh form through a fast sigmoid; 2e-3 covers that approximation.
        for (int i = 0; i < output.Length; i++)
            Assert.True(Math.Abs(output[i] - expected[i]) <= 2e-3, $"out[{i}] = {output[i]}, expected {expected[i]}");
    }
}
