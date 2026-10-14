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
        new(128, 128, 32, 64, 32, 2, 4),
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

    // FP16: every contiguous extent and N a multiple of 8 (one 16-byte chunk of halves).
    private static readonly (int M, int N, int K)[] Fp16Shapes =
    {
        (8, 8, 8), (16, 24, 40), (128, 128, 128), (200, 136, 40), (64, 256, 96), (40, 48, 1000), (264, 8, 72),
    };

    private static void CheckFp16(DirectPtxRuntime runtime, PtxTensorCoreGemmKernel kernel, int batch, int m, int n, int k,
        float alpha, float beta, bool poisonC)
    {
        var layout = kernel.Layout;
        bool halfOut = kernel.HalfOutput;
        int lda = layout.TransposeA ? m : k, ldb = layout.TransposeB ? k : n;
        Half[] a = Values(batch * m * k, m * 31 + k).Select(v => (Half)v).ToArray();
        Half[] b = Values(batch * k * n, n * 17 + k).Select(v => (Half)v).ToArray();
        float[] cFloat = poisonC ? Enumerable.Repeat(float.NaN, batch * m * n).ToArray() : Values(batch * m * n, m + n);
        // The oracle reads C as the kernel will: rounded to FP16 when the output is FP16.
        double[] cSeen = cFloat.Select(v => halfOut ? (double)(float)(Half)v : v).ToArray();
        var expected = new double[batch * m * n];
        for (int z = 0; z < batch; z++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int p = 0; p < k; p++)
                    {
                        double av = (double)(layout.TransposeA ? a[z * m * k + p * lda + i] : a[z * m * k + i * lda + p]);
                        double bv = (double)(layout.TransposeB ? b[z * k * n + j * ldb + p] : b[z * k * n + p * ldb + j]);
                        sum += av * bv;
                    }
                    int index = z * m * n + i * n + j;
                    expected[index] = alpha * sum + (beta == 0 ? 0 : beta * cSeen[index]);
                }
        using var da = runtime.AllocateBytes((nuint)(Math.Max(1, a.Length) * 2));
        using var db = runtime.AllocateBytes((nuint)(Math.Max(1, b.Length) * 2));
        using var dc = runtime.AllocateBytes((nuint)(cFloat.Length * (halfOut ? 2 : 4)));
        if (a.Length > 0) da.Upload<Half>(a);
        if (b.Length > 0) db.Upload<Half>(b);
        if (halfOut) dc.Upload<Half>(cFloat.Select(v => (Half)v).ToArray());
        else dc.Upload<float>(cFloat);
        kernel.Launch(da.Pointer, db.Pointer, dc.Pointer, m, n, k, lda, ldb, n, alpha, beta, batch,
            (long)m * k, (long)k * n, (long)m * n);
        runtime.Synchronize();
        double[] actual;
        if (halfOut)
        {
            var raw = new Half[cFloat.Length];
            dc.Download<Half>(raw);
            actual = raw.Select(v => (double)(float)v).ToArray();
        }
        else
        {
            var raw = new float[cFloat.Length];
            dc.Download<float>(raw);
            actual = raw.Select(v => (double)v).ToArray();
        }
        // FP16 operands are exact; products accumulate in FP32. An FP16 output adds one rounding (2^-11 relative).
        double bound = 1e-4 * Math.Abs(alpha) * Math.Max(1, k) + 1e-5 * (1 + Math.Abs(beta));
        string what = $"{kernel.Configuration.Name} {layout.Name} fp16{(halfOut ? "->fp16" : "->fp32")} b{batch} M{m} N{n} K{k} alpha{alpha} beta{beta}";
        for (int i = 0; i < actual.Length; i++)
        {
            double allowed = bound + (halfOut ? Math.Abs(expected[i]) * 1e-3 : 0);
            Assert.True(Math.Abs(actual[i] - expected[i]) <= allowed, $"{what}: [{i}] got {actual[i]}, expected {expected[i]}");
        }
    }

    [SkippableFact]
    public void Fp16_SampledConfigurations_MatchTheOracle_BothOutputs()
    {
        using var runtime = RequireAmpere();
        var configs = Sampled.Where(c => Layouts.All(l => c.Invalidity(l, PtxTensorCorePrecision.Fp16) is null)).ToList();
        Assert.True(configs.Count >= 2, "too few sampled configurations are valid for FP16");
        foreach (var config in configs)
            foreach (var layout in Layouts)
                foreach (bool halfOut in new[] { false, true })
                {
                    using var kernel = new PtxTensorCoreGemmKernel(runtime, config, layout, PtxTensorCorePrecision.Fp16, halfOut);
                    Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
                    foreach (var (m, n, k) in Fp16Shapes)
                    {
                        CheckFp16(runtime, kernel, 1, m, n, k, 1f, 0f, poisonC: false);
                        CheckFp16(runtime, kernel, 1, m, n, k, 0.5f, 2f, poisonC: false);
                    }
                    CheckFp16(runtime, kernel, 3, 40, 48, 56, -1.5f, 0.25f, poisonC: false);
                    CheckFp16(runtime, kernel, 1, 64, 64, 64, 1f, 0f, poisonC: true);
                    CheckFp16(runtime, kernel, 1, 64, 64, 0, 1f, 3f, poisonC: false);
                }
    }

    [Fact]
    public void Fp16_SearchSpaceEmits_AndHalfOutputNeedsFp16Operands()
    {
        foreach (var layout in Layouts)
        {
            var space = PtxTensorCoreGemmConfiguration.SearchSpace(layout, PtxTensorCorePrecision.Fp16).ToList();
            Assert.True(space.Count > 50, $"{layout.Name}: the FP16 search space shrank to {space.Count}");
            foreach (var config in space.Take(40))
                Assert.Contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
                    PtxTensorCoreGemmGenerator.Emit(config, layout, 8, 6, PtxTensorCorePrecision.Fp16, halfOutput: true));
        }
        Assert.Throws<ArgumentException>(() =>
            PtxTensorCoreGemmGenerator.Emit(Sampled[0], Layouts[0], 8, 6, PtxTensorCorePrecision.Tf32, halfOutput: true));
    }

    private sealed class MemoryStore : AiDotNet.Tensors.Helpers.Autotune.IKernelTuningStore<PtxTensorCoreGemmConfiguration>
    {
        public bool TryLoad(AiDotNet.Tensors.Helpers.Autotune.KernelTuningIdentity identity,
            AiDotNet.Evolution.IEvolutionGenomeCodec<PtxTensorCoreGemmConfiguration> codec,
            out AiDotNet.Tensors.Helpers.Autotune.KernelTuningDeploymentSnapshot<PtxTensorCoreGemmConfiguration>? snapshot)
        {
            snapshot = null;
            return false;
        }

        public bool TryStore(AiDotNet.Tensors.Helpers.Autotune.KernelTuningDeploymentSnapshot<PtxTensorCoreGemmConfiguration> snapshot,
            AiDotNet.Evolution.IEvolutionGenomeCodec<PtxTensorCoreGemmConfiguration> codec) => true;
    }

    [SkippableFact]
    public async System.Threading.Tasks.Task SmallBudgetTuning_DeploysAValidatedWinner_Tf32AndFp16()
    {
        using var runtime = RequireAmpere();
        foreach (var precision in new[] { PtxTensorCorePrecision.Tf32, PtxTensorCorePrecision.Fp16 })
        {
            var options = new AiDotNet.Evolution.EvolutionEngineOptions
            {
                RunId = "tc-gemm-evolution-test-" + precision, Seed = 1264, MaxEvaluationAttempts = 8, MaxProposals = 16,
                MaxGenerations = 2, ProposalBatchSize = 4, MaxDegreeOfParallelism = 1, IslandCount = 1,
                MigrationInterval = 0, MigrantsPerIsland = 1,
            };
            var result = await PtxTensorCoreGemmEvolution.TuneAsync(runtime, Layouts[0], 256, 256, 256, precision,
                engineOptions: options,
                deploymentRegistry: new AiDotNet.Tensors.Helpers.Autotune.KernelTuningDeploymentRegistry<PtxTensorCoreGemmConfiguration>(),
                store: new MemoryStore(),
                onTrial: (c, t) => _output.WriteLine($"{precision} {c.Name}: {t.Status} {t.Measurement?.ThroughputGflops:F0} GFLOP/s {t.Diagnostic}"));
            Assert.True(result.Run.Counters.EvaluationAttempts >= 3);
            Assert.Equal(AiDotNet.Tensors.Helpers.Autotune.KernelTuningValidationScope.Output,
                result.ProposedWinner.Measurement.Correctness.Scope);
            Assert.True(result.ProposedWinner.Measurement.ThroughputGflops > 0);
            Assert.True(PtxTensorCoreGemmEvolution.Admits(result.ProposedWinner.Configuration, Layouts[0], 256, 256, 256, precision));
        }
    }

    /// <summary>
    /// Opt-in seconds-long timing probe (AIDOTNET_RUN_TC_GEMM_PROBE=1): one large shape, the large-tile configurations
    /// against cuBLAS's default (TF32) GEMM on the same device and stream, timed with CUDA events; each PTX output is also
    /// compared with cuBLAS's (both TF32), so a fast-but-wrong kernel fails. A development check, not evidence.
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
        using var reference = backend.AllocateBuffer(m * n);
        using var c = backend.AllocateBuffer(m * n);
        double flops = 2d * m * n * k;
        double Time(Action launch)
        {
            for (int i = 0; i < 3; i++) launch();
            backend.Synchronize();
            using var start = backend.CreateEvent(enableTiming: true);
            using var end = backend.CreateEvent(enableTiming: true);
            const int reps = 20;
            backend.RecordEvent(start, backend.DefaultStream);
            for (int i = 0; i < reps; i++) launch();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            return backend.GetEventElapsedTime(start, end) / reps;
        }
        double cublas = Time(() => backend.Gemm(a, b, reference, m, n, k));
        float[] expected = backend.DownloadBuffer(reference);
        double scale = expected.Max(v => Math.Abs((double)v));
        _output.WriteLine($"cuBLAS default (TF32): {cublas:F3} ms, {flops / cublas / 1e9:F1} TFLOP/s, |C| max {scale:F1}");
        var layout = new PtxGemmLayout(false, false);
        foreach (var config in PtxTensorCoreGemmConfiguration.SearchSpace(layout).Where(x => x.BlockM >= 128 && x.BlockN >= 128))
        {
            PtxTensorCoreGemmKernel kernel;
            try
            {
                kernel = new PtxTensorCoreGemmKernel(runtime, config, layout);
            }
            catch (InvalidOperationException rejected)
            {
                // The loader enforces zero local memory: a configuration that spills is refused, as tuning would.
                _output.WriteLine($"{config.Name}: rejected ({rejected.Message})");
                continue;
            }
            using var _ = kernel;
            double ms = Time(() => kernel.Launch(a.Handle, b.Handle, c.Handle, m, n, k, k, n, n, 1f, 0f));
            float[] actual = backend.DownloadBuffer(c);
            double diff = 0;
            for (int i = 0; i < actual.Length; i += 97) diff = Math.Max(diff, Math.Abs(actual[i] - expected[i]));
            _output.WriteLine($"{config.Name}: {ms:F3} ms, {flops / ms / 1e9:F1} TFLOP/s ({cublas / ms:F2}x cuBLAS), " +
                $"max |PTX - cuBLAS| / max|C| = {diff / scale:E1}");
            Assert.True(diff / scale < 5e-3, $"{config.Name} disagrees with cuBLAS TF32: {diff / scale:E2}");
        }

        // FP16 operands, FP32 output, against cuBLAS's GemmFp16In32fOut (cublasGemmEx) on the same operands.
        static byte[] HalfBytes(float[] values)
        {
            var bytes = new byte[values.Length * 2];
            for (int i = 0; i < values.Length; i++)
            {
                ushort bits = BitConverter.HalfToUInt16Bits((Half)values[i]);
                bytes[2 * i] = (byte)bits;
                bytes[2 * i + 1] = (byte)(bits >> 8);
            }
            return bytes;
        }
        using var ah = backend.AllocateByteBuffer(m * k * 2);
        using var bh = backend.AllocateByteBuffer(k * n * 2);
        backend.UploadBytes(ah, HalfBytes(Values(m * k, 3)));
        backend.UploadBytes(bh, HalfBytes(Values(k * n, 4)));
        double cublasHalf = Time(() => backend.GemmFp16In32fOut(ah, bh, reference, m, n, k));
        float[] expectedHalf = backend.DownloadBuffer(reference);
        double scaleHalf = expectedHalf.Max(v => Math.Abs((double)v));
        _output.WriteLine($"cuBLAS FP16 -> FP32: {cublasHalf:F3} ms, {flops / cublasHalf / 1e9:F1} TFLOP/s");
        foreach (var config in PtxTensorCoreGemmConfiguration.SearchSpace(layout, PtxTensorCorePrecision.Fp16)
                     .Where(x => x.BlockM >= 128 && x.BlockN >= 128))
        {
            PtxTensorCoreGemmKernel kernel;
            try
            {
                kernel = new PtxTensorCoreGemmKernel(runtime, config, layout, PtxTensorCorePrecision.Fp16);
            }
            catch (InvalidOperationException rejected)
            {
                _output.WriteLine($"fp16 {config.Name}: rejected ({rejected.Message})");
                continue;
            }
            using var _ = kernel;
            double ms = Time(() => kernel.Launch(ah.Handle, bh.Handle, c.Handle, m, n, k, k, n, n, 1f, 0f));
            float[] actual = backend.DownloadBuffer(c);
            double diff = 0;
            for (int i = 0; i < actual.Length; i += 97) diff = Math.Max(diff, Math.Abs(actual[i] - expectedHalf[i]));
            _output.WriteLine($"fp16 {config.Name}: {ms:F3} ms, {flops / ms / 1e9:F1} TFLOP/s ({cublasHalf / ms:F2}x cuBLAS), " +
                $"max |PTX - cuBLAS| / max|C| = {diff / scaleHalf:E1}");
            Assert.True(diff / scaleHalf < 1e-3, $"fp16 {config.Name} disagrees with cuBLAS: {diff / scaleHalf:E2}");
        }
    }
}
#endif
