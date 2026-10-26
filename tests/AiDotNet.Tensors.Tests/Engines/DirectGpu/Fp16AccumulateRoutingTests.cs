#if NETCOREAPP
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1549: <see cref="CudaBackend.GemmFp16In32fOut"/> routes a large shape to the FP16-accumulate GEMM when that measures
/// faster than cuBLAS. Whether or not a shape is routed, its output must agree with cuBLAS to FP16-accumulation accuracy;
/// and on a routed shape, operands large enough to overflow an FP16 partial sum must still give the exact
/// FP32-accumulated result, because the kernel redoes an overflowing block with FP32 accumulation.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class Fp16AccumulateRoutingTests
{
    private readonly ITestOutputHelper _output;

    public Fp16AccumulateRoutingTests(ITestOutputHelper output) => _output = output;

    private static byte[] HalfBytes(Half[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            ushort bits = BitConverter.HalfToUInt16Bits(values[i]);
            bytes[2 * i] = (byte)bits;
            bytes[2 * i + 1] = (byte)(bits >> 8);
        }
        return bytes;
    }

    private static CudaBackend RequireAmpereBackend()
    {
        var backend = new CudaBackend();
        if (!backend.IsAvailable || backend.GetDirectPtxRuntime().ComputeCapabilityMajor < 8)
        {
            backend.Dispose();
            Skip.If(true, "Needs a CUDA device of compute capability 8.0+.");
        }
        return backend;
    }

    // One disturbed round must not keep cuBLAS for the process: a winning median with a losing round is measured again,
    // and the second measurement decides on its own; a clear win or a losing median is final.
    [Fact]
    public void Verdict_WinningMedianWithALosingRound_IsReMeasuredOnce()
    {
        static CudaBackend.GemmRouteVerdict V(double median, double minimum) =>
            new(median >= 1.02 && minimum >= 1.0, median, minimum, 0, 0, 8, $"{median}/{minimum}");
        CudaBackend.GemmRouteVerdict Run(params CudaBackend.GemmRouteVerdict[] sequence)
        {
            int calls = 0;
            var result = CudaBackend.ConfirmFp16AccumulateVerdict(() => sequence[calls++]);
            Assert.Equal(sequence.Length, calls);
            return result;
        }
        Assert.True(Run(V(1.2, 1.1)).Wins);                          // clear win: measured once
        Assert.False(Run(V(0.99, 0.9)).Wins);                        // losing median: measured once
        var retried = Run(V(1.2, 0.85), V(1.19, 1.05));              // outlier round: the re-measurement wins
        Assert.True(retried.Wins);
        Assert.Contains("re-measured", retried.Reason);
        Assert.False(Run(V(1.2, 0.85), V(1.15, 0.95)).Wins);         // the re-measurement must pass on its own
        Assert.False(Run(V(1.2, 0.85), V(1.01, 1.0)).Wins);
    }

    [SkippableFact]
    public void LargeShape_IsEvaluated_AndAgreesWithTheFp64Reference()
    {
        using var backend = RequireAmpereBackend();
        const int m = 1024, n = 1024, k = 1024;   // 2^30 multiply-adds: past the route's minimum work
        var rng = new Random(11);
        Half[] a = Enumerable.Range(0, m * k).Select(_ => (Half)(rng.NextDouble() * 2 - 1)).ToArray();
        Half[] b = Enumerable.Range(0, k * n).Select(_ => (Half)(rng.NextDouble() * 2 - 1)).ToArray();
        using var da = backend.AllocateByteBuffer(m * k * 2);
        using var db = backend.AllocateByteBuffer(k * n * 2);
        using var dc = backend.AllocateBuffer(m * n);
        backend.UploadBytes(da, HalfBytes(a));
        backend.UploadBytes(db, HalfBytes(b));

        backend.GemmFp16In32fOut(da, db, dc, m, n, k);
        var verdict = backend.Fp16AccumulateVerdict(m, n, k);
        Assert.True(verdict.HasValue, "the first eager call of a large shape must evaluate the route");
        _output.WriteLine($"routed: {verdict!.Value.Routed}; {verdict.Value.Reason}");
        float[] c = backend.DownloadBuffer(dc);

        // fp64 reference on a sample of outputs; FP16 accumulation's bound (BlockK/16 + 1) * 2^-11 * sum|ab| with BlockK <= 32.
        for (int s = 0; s < 400; s++)
        {
            int i = rng.Next(m), j = rng.Next(n);
            double sum = 0, abs = 0;
            for (int p = 0; p < k; p++)
            {
                double prod = (double)a[i * k + p] * (double)b[p * n + j];
                sum += prod;
                abs += Math.Abs(prod);
            }
            Assert.True(Math.Abs(c[i * n + j] - sum) <= 3 * Math.Pow(2, -11) * abs + 1e-3,
                $"C[{i},{j}] = {c[i * n + j]}, expected {sum} (sum|ab| {abs})");
        }

    }

    // A routed shape (forced, so the check does not depend on which route measures faster on this device): small operands
    // go through the FP16-accumulate path and agree with cuBLAS; operands large enough to overflow an FP16 partial sum
    // (64 x 64 x BlockK 32 = 131072) make every block redo its tile with FP32 accumulation, giving the exact result.
    [SkippableFact]
    public void RoutedShape_SmallOperandsAgree_AndOverflowSizedOperandsAreRedoneExactly()
    {
        using var backend = RequireAmpereBackend();
        const int m = 1024, n = 1024, k = 1024;
        backend.ForceFp16AccumulateRouteForTests(m, n, k,
            new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmConfiguration(64, 128, 32, 64, 32, 3, 8));
        var rng = new Random(13);
        Half[] a = Enumerable.Range(0, m * k).Select(_ => (Half)(rng.NextDouble() * 2 - 1)).ToArray();
        Half[] b = Enumerable.Range(0, k * n).Select(_ => (Half)(rng.NextDouble() * 2 - 1)).ToArray();
        using var da = backend.AllocateByteBuffer(m * k * 2);
        using var db = backend.AllocateByteBuffer(k * n * 2);
        using var dc = backend.AllocateBuffer(m * n);
        backend.UploadBytes(da, HalfBytes(a));
        backend.UploadBytes(db, HalfBytes(b));
        long before = backend.Fp16AccumulateDispatchCount;
        backend.GemmFp16In32fOut(da, db, dc, m, n, k);
        Assert.Equal(before + 1, backend.Fp16AccumulateDispatchCount);
        float[] c = backend.DownloadBuffer(dc);
        for (int s = 0; s < 400; s++)
        {
            int i = rng.Next(m), j = rng.Next(n);
            double sum = 0, abs = 0;
            for (int p = 0; p < k; p++)
            {
                double prod = (double)a[i * k + p] * (double)b[p * n + j];
                sum += prod;
                abs += Math.Abs(prod);
            }
            Assert.True(Math.Abs(c[i * n + j] - sum) <= 3 * Math.Pow(2, -11) * abs + 1e-3,
                $"C[{i},{j}] = {c[i * n + j]}, expected {sum} (sum|ab| {abs})");
        }

        Half[] big = Enumerable.Repeat((Half)64f, m * k).ToArray();
        backend.UploadBytes(da, HalfBytes(big));
        backend.UploadBytes(db, HalfBytes(big));
        backend.GemmFp16In32fOut(da, db, dc, m, n, k);
        Assert.Equal(before + 2, backend.Fp16AccumulateDispatchCount);
        float[] exact = backend.DownloadBuffer(dc);
        Assert.All(exact, v => Assert.Equal(64f * 64f * k, v));
    }

    // Opt-in (AIDOTNET_RUN_TC_GEMM_PROBE=1): per shape, cuBLAS against the best of a few configurations with FP32
    // accumulation and with FP16 accumulation (operands in [-1, 1): no block redoes), and the FP16-accumulate kernel when
    // every block overflows and redoes (all operands 64), the worst case of the in-kernel recovery.
    [SkippableFact]
    public void Probe_RecoveryCostBreakdown()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_TC_GEMM_PROBE") == "1", "Set AIDOTNET_RUN_TC_GEMM_PROBE=1.");
        using var backend = RequireAmpereBackend();
        var runtime = backend.GetDirectPtxRuntime();
        var layout = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxGemmLayout(false, false);
        var configs = new[]
        {
            new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmConfiguration(128, 128, 16, 64, 64, 2, 8),
            new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmConfiguration(64, 128, 32, 64, 32, 3, 8),
            new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmConfiguration(64, 64, 32, 32, 32, 2, 8),
            new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmConfiguration(64, 64, 64, 32, 32, 2, 8),
        };
        foreach (var (m, n, k) in new[] { (4096, 4096, 4096), (4096, 768, 3072), (4096, 3072, 768), (1024, 768, 3072),
                     (12544, 128, 1152), (1024, 1024, 1024) })
        {
            var rng = new Random(3);
            using var da = backend.AllocateByteBuffer(m * k * 2);
            using var db = backend.AllocateByteBuffer(k * n * 2);
            using var dbig = backend.AllocateByteBuffer(Math.Max(m * k, k * n) * 2);
            using var dc = backend.AllocateBuffer(m * n);
            backend.UploadBytes(da, HalfBytes(Enumerable.Range(0, m * k).Select(_ => (Half)(rng.NextDouble() * 2 - 1)).ToArray()));
            backend.UploadBytes(db, HalfBytes(Enumerable.Range(0, k * n).Select(_ => (Half)(rng.NextDouble() * 2 - 1)).ToArray()));
            backend.UploadBytes(dbig, HalfBytes(Enumerable.Repeat((Half)64f, Math.Max(m * k, k * n)).ToArray()));
            double Time(Action launch)
            {
                for (int i = 0; i < 3; i++) launch();
                backend.Synchronize();
                double best = double.MaxValue;
                for (int round = 0; round < 3; round++)
                {
                    using var start = backend.CreateEvent(enableTiming: true);
                    using var end = backend.CreateEvent(enableTiming: true);
                    backend.RecordEvent(start, backend.DefaultStream);
                    for (int i = 0; i < 10; i++) launch();
                    backend.RecordEvent(end, backend.DefaultStream);
                    end.Synchronize();
                    best = Math.Min(best, backend.GetEventElapsedTime(start, end) * 1000 / 10);
                }
                return best;
            }
            double cublas = Time(() => backend.CublasGemmFp16In32fOutForTests(da, db, dc, m, n, k));
            double best32 = double.MaxValue, best16 = double.MaxValue, bestRedo = double.MaxValue;
            foreach (var config in configs)
            {
                if (!AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmGenerator.Admits(config, layout, m, n, k, k, n, n, 0, 0, 0,
                        AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCorePrecision.Fp16)) continue;
                using var f32 = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmKernel(runtime, config, layout,
                    AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCorePrecision.Fp16);
                AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmKernel f16;
                try
                {
                    f16 = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmKernel(runtime, config, layout,
                        AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCorePrecision.Fp16, fp16Accumulate: true);
                }
                catch (InvalidOperationException rejected)
                {
                    _output.WriteLine($"  {config.Name}: FP16-acc rejected ({rejected.Message.Split('.')[0]})");
                    continue;
                }
                using var _f16 = f16;
                best32 = Math.Min(best32, Time(() => f32.Launch(da.Handle, db.Handle, dc.Handle, m, n, k, k, n, n, 1f, 0f)));
                best16 = Math.Min(best16, Time(() => f16.Launch(da.Handle, db.Handle, dc.Handle, m, n, k, k, n, n, 1f, 0f)));
                bestRedo = Math.Min(bestRedo, Time(() => f16.Launch(dbig.Handle, dbig.Handle, dc.Handle, m, n, k, k, n, n, 1f, 0f)));
            }
            _output.WriteLine($"M{m} N{n} K{k}: cuBLAS {cublas:F1} us | ours FP32-acc {best32:F1} | FP16-acc {best16:F1} " +
                $"({cublas / best16:F2}x cuBLAS) | every block redoes {bestRedo:F1} ({cublas / bestRedo:F2}x)");
        }
    }

    // Opt-in (AIDOTNET_RUN_TC_GEMM_PROBE=1): per-shape evolution of the FP16-accumulate kernel (TuneFp16AccumulateRouteAsync,
    // not persisted) and its verdict against cuBLAS, on the shapes the fixed candidates lost.
    [SkippableFact]
    public async System.Threading.Tasks.Task Probe_EvolvedRouteVerdicts()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_TC_GEMM_PROBE") == "1", "Set AIDOTNET_RUN_TC_GEMM_PROBE=1.");
        using var backend = RequireAmpereBackend();
        foreach (var (m, n, k) in new[] { (2048, 2048, 2048), (4096, 768, 3072), (4096, 3072, 768), (1024, 768, 3072),
                     (12544, 128, 1152), (1024, 1024, 1024), (4096, 4096, 4096) })
        {
            var options = new AiDotNet.Evolution.EvolutionEngineOptions
            {
                RunId = $"fp16acc-route-probe-{m}x{n}x{k}", Seed = 1549, MaxEvaluationAttempts = 48, MaxProposals = 96,
                MaxGenerations = 6, ProposalBatchSize = 8, MaxDegreeOfParallelism = 1, IslandCount = 1,
                MigrationInterval = 0, MigrantsPerIsland = 1,
            };
            var started = System.Diagnostics.Stopwatch.StartNew();
            var verdict = await backend.TuneFp16AccumulateRouteAsync(m, n, k, options, persist: false);
            _output.WriteLine($"M{m} N{n} K{k}: routed {verdict.Routed}, {verdict.Kernel}, median {verdict.MedianSpeedup:F3}x, " +
                $"tuned in {started.Elapsed.TotalSeconds:F0} s; {verdict.Reason}");
        }
    }

    // Opt-in (AIDOTNET_RUN_TC_GEMM_PROBE=1): the route's verdict on representative large shapes on this device.
    [SkippableFact]
    public void Probe_RouteVerdicts()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_TC_GEMM_PROBE") == "1", "Set AIDOTNET_RUN_TC_GEMM_PROBE=1.");
        using var backend = RequireAmpereBackend();
        foreach (var (m, n, k) in new[] { (4096, 4096, 4096), (2048, 2048, 2048), (1024, 768, 3072), (4096, 768, 3072),
                     (4096, 3072, 768), (12544, 128, 1152), (1024, 1024, 1024) })
        {
            var rng = new Random(m + n + k);
            using var da = backend.AllocateByteBuffer(m * k * 2);
            using var db = backend.AllocateByteBuffer(k * n * 2);
            using var dc = backend.AllocateBuffer(m * n);
            backend.UploadBytes(da, HalfBytes(Enumerable.Range(0, m * k).Select(_ => (Half)(rng.NextDouble() * 2 - 1)).ToArray()));
            backend.UploadBytes(db, HalfBytes(Enumerable.Range(0, k * n).Select(_ => (Half)(rng.NextDouble() * 2 - 1)).ToArray()));
            backend.GemmFp16In32fOut(da, db, dc, m, n, k);
            var verdict = backend.Fp16AccumulateVerdict(m, n, k);
            _output.WriteLine($"M{m} N{n} K{k}: " + (verdict is { } v ? $"routed {v.Routed}, median {v.MedianSpeedup:F3}x; {v.Reason}" : "not evaluated"));
        }
    }
}
#endif