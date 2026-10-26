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
        var absProducts = new double[batch * m * n];
        for (int z = 0; z < batch; z++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double sum = 0, abs = 0;
                    for (int p = 0; p < k; p++)
                    {
                        double av = (double)(layout.TransposeA ? a[z * m * k + p * lda + i] : a[z * m * k + i * lda + p]);
                        double bv = (double)(layout.TransposeB ? b[z * k * n + j * ldb + p] : b[z * k * n + p * ldb + j]);
                        sum += av * bv;
                        abs += Math.Abs(av * bv);
                    }
                    int index = z * m * n + i * n + j;
                    expected[index] = alpha * sum + (beta == 0 ? 0 : beta * cSeen[index]);
                    absProducts[index] = abs;
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
        // FP16 accumulation (#1549): each K tile's partial sum is rounded to FP16 once per MMA step of the tile
        // (BlockK / 16 steps) and once more when promoted, each at most 2^-11 of the partial's magnitude, which is at
        // most the tile's sum of |a b|. Summed over tiles: (BlockK / 16 + 1) * 2^-11 * sum |a b|.
        double accumulateUlps = kernel.Fp16Accumulate ? (kernel.Configuration.BlockK / 16 + 1) * Math.Pow(2, -11) : 0;
        string what = $"{kernel.Configuration.Name} {layout.Name} fp16{(halfOut ? "->fp16" : "->fp32")}" +
            $"{(kernel.Fp16Accumulate ? " f16acc" : "")} b{batch} M{m} N{n} K{k} alpha{alpha} beta{beta}";
        for (int i = 0; i < actual.Length; i++)
        {
            double allowed = bound + (halfOut ? Math.Abs(expected[i]) * 1e-3 : 0)
                + accumulateUlps * Math.Abs(alpha) * absProducts[i];
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

    // #1549: FP16 accumulation per K tile, promoted into an FP32 total, over the same sampled configurations, layouts,
    // shapes and scalars, within the bound CheckFp16 derives from the per-tile FP16 roundings.
    [SkippableFact]
    public void Fp16Accumulate_SampledConfigurations_MatchTheOracle_BothOutputs()
    {
        using var runtime = RequireAmpere();
        var configs = Sampled.Where(c => Layouts.All(l => c.Invalidity(l, PtxTensorCorePrecision.Fp16) is null)).ToList();
        foreach (var config in configs)
            foreach (var layout in Layouts)
                foreach (bool halfOut in new[] { false, true })
                {
                    using var kernel = new PtxTensorCoreGemmKernel(runtime, config, layout, PtxTensorCorePrecision.Fp16, halfOut,
                        fp16Accumulate: true);
                    Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
                    Assert.Contains("f16.f16.f16.f16", kernel.Ptx);
                    Assert.DoesNotContain("f32.f16.f16.f32", kernel.Ptx);
                    foreach (var (m, n, k) in Fp16Shapes)
                    {
                        CheckFp16(runtime, kernel, 1, m, n, k, 1f, 0f, poisonC: false);
                        CheckFp16(runtime, kernel, 1, m, n, k, 0.5f, 2f, poisonC: false);
                    }
                    CheckFp16(runtime, kernel, 3, 40, 48, 56, -1.5f, 0.25f, poisonC: false);
                    CheckFp16(runtime, kernel, 1, 64, 64, 64, 1f, 0f, poisonC: true);
                    CheckFp16(runtime, kernel, 1, 64, 64, 0, 1f, 3f, poisonC: false);
                    CheckFp16(runtime, kernel, 1, 64, 64, 4096, 1f, 0f, poisonC: false);   // many promoted tiles
                }
    }

    // The documented limit of FP16 accumulation: a K tile's partial sum must stay below 65504. 32 products of
    // 64 x 64 = 4096 each make a 131072 partial in a 32-deep tile, which FP16 cannot hold, while FP32 accumulation of the
    // same problem is exact. Callers that cannot bound their operands must keep FP32 accumulation.
    [SkippableFact]
    public void Fp16Accumulate_TilePartialAbove65504_Overflows_WhereFp32AccumulationDoesNot()
    {
        using var runtime = RequireAmpere();
        var config = new PtxTensorCoreGemmConfiguration(64, 64, 32, 32, 32, 2, 4);
        var layout = new PtxGemmLayout(false, false);
        const int m = 64, n = 64, k = 32;
        Half[] values = Enumerable.Repeat((Half)64f, m * k).ToArray();
        using var da = runtime.AllocateBytes((nuint)(m * k * 2));
        using var db = runtime.AllocateBytes((nuint)(k * n * 2));
        using var dc = runtime.AllocateBytes((nuint)(m * n * 4));
        da.Upload<Half>(values);
        db.Upload<Half>(values);
        float Run(bool fp16Accumulate)
        {
            using var kernel = new PtxTensorCoreGemmKernel(runtime, config, layout, PtxTensorCorePrecision.Fp16,
                fp16Accumulate: fp16Accumulate);
            kernel.Launch(da.Pointer, db.Pointer, dc.Pointer, m, n, k, k, n, n, 1f, 0f, 1, (long)m * k, (long)k * n, (long)m * n);
            runtime.Synchronize();
            var c = new float[m * n];
            dc.Download<float>(c);
            return c[0];
        }
        Assert.Equal(131072f, Run(fp16Accumulate: false));
        Assert.True(float.IsInfinity(Run(fp16Accumulate: true)), "a 131072 FP16 partial sum should overflow to infinity");
    }

    // Opt-in speed probe (AIDOTNET_RUN_TC_GEMM_PROBE=1): FP16 operands and FP32 output, FP32 accumulation against FP16
    // accumulation with per-tile promotion (#1549), same configurations, CUDA events, and cuBLAS FP16 -> FP32 for scale.
    // The operands are in [-1, 1), so each partial sum is far below 65504. A development check, not evidence.
    [SkippableFact]
    public void Probe_Fp16AccumulateAgainstFp32Accumulate()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_TC_GEMM_PROBE") == "1", "Set AIDOTNET_RUN_TC_GEMM_PROBE=1.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is unavailable.");
        DirectPtxRuntime runtime = backend.GetDirectPtxRuntime();
        var layout = new PtxGemmLayout(false, false);
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
        foreach (var (m, n, k) in new[] { (4096, 4096, 4096), (1024, 3072, 768), (1024, 768, 3072), (3136, 64, 576), (12544, 128, 1152) })
        {
            using var ah = backend.AllocateByteBuffer(m * k * 2);
            using var bh = backend.AllocateByteBuffer(k * n * 2);
            using var reference = backend.AllocateBuffer(m * n);
            using var c = backend.AllocateBuffer(m * n);
            backend.UploadBytes(ah, HalfBytes(Values(m * k, 3)));
            backend.UploadBytes(bh, HalfBytes(Values(k * n, 4)));
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
            double cublas = Time(() => backend.GemmFp16In32fOut(ah, bh, reference, m, n, k));
            float[] expected = backend.DownloadBuffer(reference);
            double scale = expected.Max(v => Math.Abs((double)v));
            double best32 = double.MaxValue, best16 = double.MaxValue;
            string best32Name = "", best16Name = "";
            foreach (var config in PtxTensorCoreGemmConfiguration.SearchSpace(layout, PtxTensorCorePrecision.Fp16)
                         .Where(x => x.BlockM >= 64 && x.BlockN >= 64 && x.Stages >= 3 && x.SharedPad == 8))
            {
                foreach (bool fp16Accumulate in new[] { false, true })
                {
                    PtxTensorCoreGemmKernel kernel;
                    try
                    {
                        kernel = new PtxTensorCoreGemmKernel(runtime, config, layout, PtxTensorCorePrecision.Fp16,
                            fp16Accumulate: fp16Accumulate);
                    }
                    catch (InvalidOperationException)
                    {
                        continue;
                    }
                    using var _ = kernel;
                    if (!PtxTensorCoreGemmGenerator.Admits(config, layout, m, n, k, k, n, n, 0, 0, 0, PtxTensorCorePrecision.Fp16)) continue;
                    double ms = Time(() => kernel.Launch(ah.Handle, bh.Handle, c.Handle, m, n, k, k, n, n, 1f, 0f));
                    float[] actual = backend.DownloadBuffer(c);
                    double diff = 0;
                    for (int i = 0; i < actual.Length; i += 97) diff = Math.Max(diff, Math.Abs(actual[i] - expected[i]));
                    Assert.True(diff / scale < 5e-3, $"{config.Name} f16acc={fp16Accumulate} disagrees with cuBLAS: {diff / scale:E2}");
                    if (fp16Accumulate && ms < best16) { best16 = ms; best16Name = config.Name; }
                    if (!fp16Accumulate && ms < best32) { best32 = ms; best32Name = config.Name; }
                }
            }
            _output.WriteLine($"M{m} N{n} K{k}: cuBLAS FP16->FP32 {flops / cublas / 1e9:F1} TFLOP/s; " +
                $"best FP32-acc {flops / best32 / 1e9:F1} TFLOP/s ({best32Name}); " +
                $"best FP16-acc {flops / best16 / 1e9:F1} TFLOP/s ({best16Name}); FP16-acc / FP32-acc {best32 / best16:F2}x, " +
                $"FP16-acc / cuBLAS {cublas / best16:F2}x");
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

    /// <summary>
    /// Opt-in routing probe (AIDOTNET_RUN_TC_GEMM_PROBE=1): over the #1261 grid shapes, the default (untuned) generated
    /// kernels against the cuBLAS mode they would replace: TF32 Tensor-Core vs cuBLAS TF32, CUDA-core FP32 vs cuBLAS strict
    /// FP32. Seconds long; it decides where #1265 may route by default, not evidence.
    /// </summary>
    [SkippableFact]
    public void Probe_GridShapes_DefaultKernelsAgainstCublas()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_TC_GEMM_PROBE") == "1", "Set AIDOTNET_RUN_TC_GEMM_PROBE=1.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is unavailable.");
        DirectPtxRuntime runtime = backend.GetDirectPtxRuntime();
        var layout = new PtxGemmLayout(false, false);
        int sms = PtxGemmEvolution.MultiprocessorCount(runtime);
        (string, int, int, int)[] shapes =
        {
            ("sq128", 128, 128, 128), ("sq256", 256, 256, 256), ("sq512", 512, 512, 512), ("sq1024", 1024, 1024, 1024),
            ("sq2048", 2048, 2048, 2048), ("sq4096", 4096, 4096, 4096), ("m1-4096", 1, 4096, 4096), ("m16-4096", 16, 4096, 4096),
            ("m1-mlp", 1, 4096, 11008), ("m16-mlp", 16, 4096, 11008), ("m128-qkv", 128, 4096, 4096), ("m2048-qkv", 2048, 4096, 4096),
            ("m2048-up", 2048, 4096, 11008), ("m2048-down", 2048, 11008, 4096), ("vis-c64", 3136, 576, 64), ("vis-c128", 784, 1152, 128),
            ("odd-1000", 1000, 1000, 1000), ("odd-127", 127, 513, 255),
        };
        double Time(Action launch, double work)
        {
            int reps = (int)Math.Clamp(Math.Ceiling(5e10 / work), 3, 200);
            for (int i = 0; i < 3; i++) launch();
            backend.Synchronize();
            using var start = backend.CreateEvent(enableTiming: true);
            using var end = backend.CreateEvent(enableTiming: true);
            backend.RecordEvent(start, backend.DefaultStream);
            for (int i = 0; i < reps; i++) launch();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            return backend.GetEventElapsedTime(start, end) * 1000.0 / reps;
        }
        bool previous = AiDotNet.Tensors.Engines.AiDotNetEngine.DeterministicMode;
        try
        {
            foreach (var (name, m, k, n) in shapes)
            {
                double work = 2d * m * n * k;
                using var a = backend.AllocateBuffer(Values(m * k, 1));
                using var b = backend.AllocateBuffer(Values(k * n, 2));
                using var c = backend.AllocateBuffer(m * n);
                string line = $"{name,-11} M{m} K{k} N{n}: ";
                AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(false);
                double tf32 = Time(() => backend.Gemm(a, b, c, m, n, k), work);
                if (PtxTensorCoreGemmEvolution.Admits(new PtxTensorCoreGemmConfiguration(64, 64, 16, 32, 32, 3, 4), layout, m, n, k, PtxTensorCorePrecision.Tf32))
                {
                    var config = PtxTensorCoreGemmEvolution.Seeds(layout, m, n, k, PtxTensorCorePrecision.Tf32, sms)[0];
                    using var kernel = new PtxTensorCoreGemmKernel(runtime, config, layout);
                    double ptx = Time(() => kernel.Launch(a.Handle, b.Handle, c.Handle, m, n, k, k, n, n, 1f, 0f), work);
                    line += $"TF32 cuBLAS {tf32,9:F1} us / PTX {ptx,9:F1} us ({tf32 / ptx:F2}x, {config.Name})   ";
                }
                else
                {
                    line += $"TF32 cuBLAS {tf32,9:F1} us / PTX n/a (unaligned)   ";
                }
                AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(true);
                double strict = Time(() => backend.Gemm(a, b, c, m, n, k), work);
                var sconfig = PtxGemmEvolution.Seeds(layout, m, n, k, sms)[0];
                using var skernel = new PtxSgemmKernel(runtime, sconfig, layout);
                double sptx = Time(() => skernel.Launch(a.Handle, b.Handle, c.Handle, m, n, k, k, n, n, 1f, 0f), work);
                line += $"| FP32 cuBLAS {strict,9:F1} us / PTX {sptx,9:F1} us ({strict / sptx:F2}x, {sconfig.Name})";
                _output.WriteLine(line);
            }
        }
        finally
        {
            AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(previous);
        }
    }

    /// <summary>
    /// Opt-in interleaved A/B (AIDOTNET_RUN_TC_GEMM_PROBE=1): 4096^3 TF32, cuBLAS and the generated kernel alternating in
    /// short rounds (ABBA), each round timed on CUDA events; reports the per-round ratio's median and range, so clock or
    /// thermal drift and background load hit both arms equally.
    /// </summary>
    [SkippableFact]
    public void Probe_Interleaved4096Tf32()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_TC_GEMM_PROBE") == "1", "Set AIDOTNET_RUN_TC_GEMM_PROBE=1.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is unavailable.");
        DirectPtxRuntime runtime = backend.GetDirectPtxRuntime();
        const int m = 4096, n = 4096, k = 4096;
        var layout = new PtxGemmLayout(false, false);
        using var a = backend.AllocateBuffer(Values(m * k, 1));
        using var b = backend.AllocateBuffer(Values(k * n, 2));
        using var c = backend.AllocateBuffer(m * n);
        using var kernel = new PtxTensorCoreGemmKernel(runtime, new PtxTensorCoreGemmConfiguration(128, 128, 16, 64, 64, 2, 8), layout);
        Action cublas = () => backend.Gemm(a, b, c, m, n, k);
        Action ptx = () => kernel.Launch(a.Handle, b.Handle, c.Handle, m, n, k, k, n, n, 1f, 0f);
        double Round(Action launch)
        {
            using var start = backend.CreateEvent(enableTiming: true);
            using var end = backend.CreateEvent(enableTiming: true);
            backend.RecordEvent(start, backend.DefaultStream);
            for (int i = 0; i < 5; i++) launch();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            return backend.GetEventElapsedTime(start, end) / 5;
        }
        for (int i = 0; i < 5; i++) { cublas(); ptx(); }
        backend.Synchronize();
        void Phase(string label)
        {
            var ratios = new System.Collections.Generic.List<double>();
            for (int r = 0; r < 8; r++)
            {
                // ABBA within each round.
                double c1 = Round(cublas), p1 = Round(ptx), p2 = Round(ptx), c2 = Round(cublas);
                double ratio = (c1 + c2) / (p1 + p2);
                ratios.Add(ratio);
                _output.WriteLine($"{label} round {r}: cuBLAS {(c1 + c2) / 2:F3} ms, PTX {(p1 + p2) / 2:F3} ms, cuBLAS/PTX {ratio:F3}");
            }
            ratios.Sort();
            _output.WriteLine($"{label}: cuBLAS/PTX median {ratios[ratios.Count / 2]:F3}, range {ratios[0]:F3} .. {ratios[^1]:F3}");
        }
        // The same process measures before and after a deterministic-mode round trip (as the grid-shape probe does), so a
        // cuBLAS state change across the toggle shows up as a step between the phases rather than across processes.
        float[] Output()
        {
            cublas();
            backend.Synchronize();
            return backend.DownloadBuffer(c);
        }
        Phase("as-created");
        float[] created = Output();
        AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(true);
        Phase("strict-fp32");
        float[] strict = Output();
        AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(false);
        for (int i = 0; i < 5; i++) cublas();
        backend.Synchronize();
        Phase("after-toggle");
        float[] toggled = Output();
        double MaxDiff(float[] x, float[] y)
        {
            double worst = 0;
            for (int i = 0; i < x.Length; i++) worst = Math.Max(worst, Math.Abs(x[i] - y[i]));
            return worst;
        }
        _output.WriteLine($"max |as-created - strict| {MaxDiff(created, strict):G3}, max |after-toggle - strict| {MaxDiff(toggled, strict):G3}");
    }

    /// <summary>
    /// Opt-in strict-FP32 probe (AIDOTNET_RUN_TC_GEMM_PROBE=1): CUDA-core generated SGEMM variants against cuBLAS in the
    /// library's default deterministic mode (strict FP32), cuBLAS and each variant alternating per round on CUDA events.
    /// Seconds long; it steers the kernel work, it is not evidence.
    /// </summary>
    [SkippableFact]
    public void Probe_StrictFp32VariantsAgainstCublas()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_TC_GEMM_PROBE") == "1", "Set AIDOTNET_RUN_TC_GEMM_PROBE=1.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is unavailable.");
        DirectPtxRuntime runtime = backend.GetDirectPtxRuntime();
        var layout = new PtxGemmLayout(false, false);
        PtxGemmConfiguration[] variants =
        {
            new(128, 128, 16, 8, 8, 4, 4, false, true, 8, 2),
            new(128, 128, 16, 8, 8, 4, 4, false, true, 8, 1, 3),
            new(128, 128, 16, 8, 8, 4, 4, false, true, 8, 2, 2, true),
            new(128, 128, 16, 8, 8, 4, 4, false, true, 8, 1, 2, true),
            new(128, 128, 16, 8, 8, 4, 4, false, true, 8, 1, 3, true),
            new(128, 128, 8, 8, 8, 4, 4, false, true, 8, 2, 3, true),
            new(128, 128, 8, 8, 8, 4, 4, false, true, 8, 2, 4, true),
            new(128, 128, 32, 8, 8, 4, 4, false, true, 8, 1, 2, true),
        };
        variants = Array.FindAll(variants, v => v.Invalidity() is null);
        (int M, int N, int K)[] shapes = { (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (2048, 4096, 4096) };
        double Round(Action launch, int reps)
        {
            using var start = backend.CreateEvent(enableTiming: true);
            using var end = backend.CreateEvent(enableTiming: true);
            backend.RecordEvent(start, backend.DefaultStream);
            for (int i = 0; i < reps; i++) launch();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            return backend.GetEventElapsedTime(start, end) * 1000.0 / reps;
        }
        bool previous = AiDotNet.Tensors.Engines.AiDotNetEngine.DeterministicMode;
        AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(true);
        try
        {
            foreach (var (m, n, k) in shapes)
            {
                using var a = backend.AllocateBuffer(Values(m * k, 1));
                using var b = backend.AllocateBuffer(Values(k * n, 2));
                using var c = backend.AllocateBuffer(m * n);
                int reps = (int)Math.Clamp(Math.Ceiling(2e10 / (2d * m * n * k)), 2, 50);
                Action cublas = () => backend.Gemm(a, b, c, m, n, k);
                for (int i = 0; i < 3; i++) cublas();
                foreach (var config in variants)
                {
                    PtxSgemmKernel loaded;
                    try
                    {
                        loaded = new PtxSgemmKernel(runtime, config, layout);
                    }
                    catch (InvalidOperationException rejected)
                    {
                        // The loader enforces zero local memory: a configuration that spills is refused, as tuning would.
                        _output.WriteLine($"{config.Name}: rejected ({rejected.Message})");
                        continue;
                    }
                    using var kernel = loaded;
                    Action ptx = () => kernel.Launch(a.Handle, b.Handle, c.Handle, m, n, k, k, n, n, 1f, 0f);
                    for (int i = 0; i < 3; i++) ptx();
                    var ratios = new System.Collections.Generic.List<double>();
                    double bestPtx = double.MaxValue, bestCublas = double.MaxValue;
                    for (int r = 0; r < 4; r++)
                    {
                        double c1 = Round(cublas, reps), p1 = Round(ptx, reps), p2 = Round(ptx, reps), c2 = Round(cublas, reps);
                        ratios.Add((c1 + c2) / (p1 + p2));
                        bestPtx = Math.Min(bestPtx, Math.Min(p1, p2));
                        bestCublas = Math.Min(bestCublas, Math.Min(c1, c2));
                    }
                    ratios.Sort();
                    _output.WriteLine($"M{m} N{n} K{k} {config.Name,-40} regs {kernel.Audit.Function.RegistersPerThread,3}: " +
                        $"cuBLAS {bestCublas,9:F1} us, PTX {bestPtx,9:F1} us ({2d * m * n * k / bestPtx / 1e6:F1} TFLOP/s), " +
                        $"cuBLAS/PTX median {ratios[ratios.Count / 2]:F3} ({ratios[0]:F3} .. {ratios[^1]:F3})");
                }
            }
        }
        finally
        {
            AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(previous);
        }
    }
}
#endif
