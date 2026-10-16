#if NET6_0_OR_GREATER
using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #377 F: measures hipBLAS SGEMM against the GEMM HipBackend runs without it (CLBlast's indirect kernel or the
/// native kernels), on the same buffers, so the decision to drop the vendor path rests on numbers from the device.
/// Needs a HIP device whose rocBLAS has Tensile kernels for it, so it runs only when asked.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class HipVendorGemmComparisonTests
{
    private const string ProofEnvironmentVariable = "AIDOTNET_RUN_HIP_VENDOR_GEMM_COMPARISON";
    private const string GemmImplEnvironmentVariable = "AIDOTNET_GPU_GEMM_IMPL";
    private readonly ITestOutputHelper _output;

    public HipVendorGemmComparisonTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    [Trait("Category", "HardwareProof")]
    public void HipBlas_VersusBuiltInGemm_Measured()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable(ProofEnvironmentVariable), "1", StringComparison.Ordinal),
            $"Set {ProofEnvironmentVariable}=1 to run the hipBLAS comparison.");

        string? previousImpl = Environment.GetEnvironmentVariable(GemmImplEnvironmentVariable);
        try
        {
            using var backend = new HipBackend();
            Skip.IfNot(backend.IsAvailable && backend.IsHipBlasAvailable,
                "A HIP device with a working hipBLAS/rocBLAS for its architecture is required.");

            int[][] shapes =
            {
                new[] { 128, 128, 128 }, new[] { 256, 256, 256 }, new[] { 512, 512, 512 }, new[] { 1024, 1024, 1024 },
                new[] { 2048, 2048, 2048 }, new[] { 64, 4096, 1024 }, new[] { 4096, 64, 1024 }, new[] { 256, 256, 2304 },
                new[] { 1024, 4096, 1024 }, new[] { 1024, 1024, 4096 }, new[] { 200, 500, 300 }, new[] { 32, 1024, 1024 }
            };

            _output.WriteLine("M x N x K | hipBLAS ms | built-in ms | built-in / hipBLAS");
            foreach (var s in shapes)
            {
                int m = s[0], n = s[1], k = s[2];
                float[] a = Values(m * k, 11), b = Values(k * n, 23);
                float[] reference = CpuGemm(a, b, m, n, k);
                using var ga = backend.AllocateBuffer(a);
                using var gb = backend.AllocateBuffer(b);
                using var gc = backend.AllocateBuffer(new float[m * n]);

                Environment.SetEnvironmentVariable(GemmImplEnvironmentVariable, "hipblas");
                double vendorMs = Time(() => Assert.True(backend.TryExecuteHipBlasGemm(ga, gb, gc, m, n, k, 1f, 0f)));
                AssertClose(reference, backend.DownloadBuffer(gc), k, $"hipBLAS {m}x{n}x{k}");

                Environment.SetEnvironmentVariable(GemmImplEnvironmentVariable, "builtin");
                double builtInMs = Time(() => backend.Gemm(ga, gb, gc, m, n, k));
                AssertClose(reference, backend.DownloadBuffer(gc), k, $"built-in {m}x{n}x{k}");

                _output.WriteLine($"{m} x {n} x {k} | {vendorMs:F3} | {builtInMs:F3} | {builtInMs / vendorMs:F2}");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(GemmImplEnvironmentVariable, previousImpl);
        }
    }

    private static double Time(Action gemm)
    {
        for (int i = 0; i < 3; i++) gemm();
        var samples = new double[15];
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            gemm();
            samples[i] = (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
        }
        Array.Sort(samples);
        return samples[samples.Length / 2];
    }

    private static float[] Values(int count, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        return values;
    }

    private static float[] CpuGemm(float[] a, float[] b, int m, int n, int k)
    {
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++)
            {
                double av = a[i * k + p];
                for (int j = 0; j < n; j++) c[i * n + j] += (float)(av * b[p * n + j]);
            }
        return c;
    }

    private static void AssertClose(float[] expected, float[] actual, int k, string label)
    {
        // Values are uniform in [-1, 1], so a K-term dot product has magnitude ~sqrt(K/3); fp32 accumulation error
        // grows ~sqrt(K) * eps relative to that.
        double tolerance = 1e-4 * Math.Sqrt(k) * Math.Max(1.0, Math.Sqrt(k / 3.0));
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"{label}: index {i} expected {expected[i]} got {actual[i]} (tolerance {tolerance:E2})");
    }
}
#endif
