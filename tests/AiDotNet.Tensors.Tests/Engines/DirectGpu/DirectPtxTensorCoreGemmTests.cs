#if NETCOREAPP
using System;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The generated Tensor-Core (TF32) GEMM (#1264): every search-space configuration emits, and on an SM 8.0+ device
/// sampled configurations match an fp64 oracle to TF32 accuracy over all layouts, edge shapes, alpha/beta, batches,
/// beta = 0 with a NaN-filled C, and K = 0, with no local memory.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DirectPtxTensorCoreGemmTests
{
    private readonly ITestOutputHelper _output;

    public DirectPtxTensorCoreGemmTests(ITestOutputHelper output) => _output = output;

    private static readonly PtxGemmLayout[] Layouts =
    {
        new(false, false), new(true, false), new(false, true), new(true, true),
    };

    private static readonly PtxTensorCoreGemmConfiguration[] Sampled =
    {
        new(64, 64, 16, 32, 32, 2, 4),
        new(128, 128, 32, 64, 32, 3, 4),
        new(128, 64, 16, 32, 32, 4, 8),
        new(64, 128, 32, 32, 64, 2, 0),
    };

    // (M, N, K): every vector-aligned (N and the contiguous extents multiples of 4) so the TF32 kernel admits them.
    private static readonly (int M, int N, int K)[] Shapes =
    {
        (4, 4, 4), (12, 20, 28), (128, 128, 128), (200, 132, 36), (64, 256, 96), (40, 44, 1000), (260, 4, 72),
    };

    [Fact]
    public void SearchSpace_EveryConfigurationEmits()
    {
        foreach (var layout in Layouts)
        {
            var space = PtxTensorCoreGemmConfiguration.SearchSpace(layout).ToList();
            Assert.True(space.Count > 100, $"{layout.Name}: the search space shrank to {space.Count}");
            foreach (var config in space)
            {
                string ptx = PtxTensorCoreGemmGenerator.Emit(config, layout, 8, 6);
                Assert.Contains("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", ptx);
                Assert.Contains("cp.async.cg.shared.global", ptx);
            }
        }
        foreach (var config in Sampled)
            foreach (var layout in Layouts)
                Assert.Null(config.Invalidity(layout));
    }

    [Fact]
    public void PreAmpereDevices_AreRefused()
    {
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxTensorCoreGemmGenerator.Emit(Sampled[0], Layouts[0], 7, 5));
    }

    private static DirectPtxRuntime RequireAmpere()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var runtime = new DirectPtxRuntime();
        if (runtime.ComputeCapabilityMajor < 8)
        {
            runtime.Dispose();
            Skip.If(true, "TF32 Tensor Cores need compute capability 8.0+.");
        }
        return runtime;
    }

    private static float[] Values(int count, int seed)
    {
        var random = new Random(seed);
        var values = new float[Math.Max(1, count)];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() * 2 - 1);
        return values;
    }

    private static void Check(DirectPtxRuntime runtime, PtxTensorCoreGemmKernel kernel, int batch, int m, int n, int k,
        float alpha, float beta, bool poisonC)
    {
        var layout = kernel.Layout;
        int lda = layout.TransposeA ? m : k, ldb = layout.TransposeB ? k : n;
        float[] a = Values(batch * m * k, m * 31 + k), b = Values(batch * k * n, n * 17 + k);
        float[] c = poisonC ? Enumerable.Repeat(float.NaN, batch * m * n).ToArray() : Values(batch * m * n, m + n);
        var expected = new double[batch * m * n];
        for (int z = 0; z < batch; z++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int p = 0; p < k; p++)
                    {
                        double av = layout.TransposeA ? a[z * m * k + p * lda + i] : a[z * m * k + i * lda + p];
                        double bv = layout.TransposeB ? b[z * k * n + j * ldb + p] : b[z * k * n + p * ldb + j];
                        sum += av * bv;
                    }
                    int index = z * m * n + i * n + j;
                    expected[index] = alpha * sum + (beta == 0 ? 0 : beta * (double)c[index]);
                }
        using var da = runtime.AllocateBytes((nuint)(a.Length * sizeof(float)));
        using var db = runtime.AllocateBytes((nuint)(b.Length * sizeof(float)));
        using var dc = runtime.AllocateBytes((nuint)(c.Length * sizeof(float)));
        da.Upload<float>(a);
        db.Upload<float>(b);
        dc.Upload<float>(c);
        kernel.Launch(da.Pointer, db.Pointer, dc.Pointer, m, n, k, lda, ldb, n, alpha, beta, batch,
            (long)m * k, (long)k * n, (long)m * n);
        runtime.Synchronize();
        var actual = new float[c.Length];
        dc.Download<float>(actual);
        // TF32 keeps 10 mantissa bits: each product is within ~2^-10 relative, and |a|, |b| <= 1.
        double bound = 1.5e-3 * Math.Abs(alpha) * Math.Max(1, k) + 1e-5 * (1 + Math.Abs(beta));
        string what = $"{kernel.Configuration.Name} {layout.Name} b{batch} M{m} N{n} K{k} alpha{alpha} beta{beta}";
        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= bound, $"{what}: [{i}] got {actual[i]}, expected {expected[i]}");
    }

    [SkippableFact]
    public void SampledConfigurations_MatchTheOracle_OverLayoutsShapesAndScalars()
    {
        using var runtime = RequireAmpere();
        foreach (var config in Sampled)
            foreach (var layout in Layouts)
            {
                using var kernel = new PtxTensorCoreGemmKernel(runtime, config, layout);
                Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
                foreach (var (m, n, k) in Shapes)
                {
                    Assert.True(PtxTensorCoreGemmGenerator.Admits(config, layout, m, n, k,
                        layout.TransposeA ? m : k, layout.TransposeB ? k : n, n, (long)m * k, (long)k * n, (long)m * n),
                        $"{config.Name} {layout.Name} M{m} N{n} K{k} not admitted");
                    Check(runtime, kernel, 1, m, n, k, 1f, 0f, poisonC: false);
                    Check(runtime, kernel, 1, m, n, k, 0.5f, 2f, poisonC: false);
                }
            }
    }

    [SkippableFact]
    public void Batched_BetaZeroAndKZero()
    {
        using var runtime = RequireAmpere();
        foreach (var layout in Layouts)
        {
            using var kernel = new PtxTensorCoreGemmKernel(runtime, Sampled[1], layout);
            Check(runtime, kernel, 3, 40, 36, 52, -1.5f, 0.25f, poisonC: false);
            Check(runtime, kernel, 1, 64, 64, 64, 1f, 0f, poisonC: true);
            Check(runtime, kernel, 1, 64, 64, 0, 1f, 3f, poisonC: false);
        }
    }

    /// <summary>
    /// Opt-in seconds-long timing probe (AIDOTNET_RUN_TC_GEMM_PROBE=1): one large shape, the sampled configurations
    /// against cuBLAS's default (TF32) GEMM on the same device. A development check, not evidence.
    /// </summary>
    [SkippableFact]
    public void Probe_LargeShapeAgainstCublasTf32()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_TC_GEMM_PROBE") == "1", "Set AIDOTNET_RUN_TC_GEMM_PROBE=1.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is unavailable.");
        DirectPtxRuntime runtime = backend.GetDirectPtxRuntime();
        const int m = 4096, n = 4096, k = 4096;
        using var a = backend.AllocateBuffer(Values(m * k, 1));
        using var b = backend.AllocateBuffer(Values(k * n, 2));
        using var c = backend.AllocateBuffer(m * n);
        double flops = 2d * m * n * k;
        double Time(Action launch)
        {
            for (int i = 0; i < 3; i++) launch();
            backend.Synchronize();
            var watch = Stopwatch.StartNew();
            const int reps = 20;
            for (int i = 0; i < reps; i++) launch();
            backend.Synchronize();
            return watch.Elapsed.TotalMilliseconds / reps;
        }
        double cublas = Time(() => backend.Gemm(a, b, c, m, n, k));
        _output.WriteLine($"cuBLAS default (TF32): {cublas:F3} ms, {flops / cublas / 1e9:F1} TFLOP/s");
        var layout = new PtxGemmLayout(false, false);
        foreach (var config in PtxTensorCoreGemmConfiguration.SearchSpace(layout).Where(x => x.Stages == 3 && x.SharedPad == 4 &&
                     x.BlockK == 32 && x.BlockM >= 128 && x.BlockN >= 128))
        {
            using var kernel = new PtxTensorCoreGemmKernel(runtime, config, layout);
            double ms = Time(() => kernel.Launch(a.Handle, b.Handle, c.Handle, m, n, k, k, n, n, 1f, 0f));
            _output.WriteLine($"{config.Name}: {ms:F3} ms, {flops / ms / 1e9:F1} TFLOP/s ({cublas / ms:F2}x cuBLAS)");
        }
    }
}
#endif
