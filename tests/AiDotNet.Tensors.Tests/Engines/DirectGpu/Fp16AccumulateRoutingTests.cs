#if NETCOREAPP
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1549: <see cref="CudaBackend.GemmFp16In32fOut"/> routes a large shape to the guarded FP16-accumulate GEMM when that
/// measures faster than cuBLAS. Whether or not a shape is routed, its output must agree with cuBLAS to FP16-accumulation
/// accuracy; and on a routed shape, operands large enough to overflow an FP16 partial sum must still give the exact
/// FP32-accumulated result, because the device-side guard falls back.
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

    [SkippableFact]
    public void LargeShape_AgreesWithCublas_AndARoutedShapeFallsBackPastTheBound()
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
    // (64 x 64 x BlockK 32 = 131072) make the device-side guard fall back to FP32 accumulation, giving the exact result.
    [SkippableFact]
    public void RoutedShape_SmallOperandsAgree_AndOverflowSizedOperandsFallBackExactly()
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

    // Opt-in (AIDOTNET_RUN_TC_GEMM_PROBE=1): where the guarded route's time goes at 4096^3 - cuBLAS, the FP32-accumulate,
    // FP16-accumulate and guarded FP16-accumulate kernels of one configuration, and the two max-abs reductions alone.
    [SkippableFact]
    public void Probe_GuardCostBreakdown()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_TC_GEMM_PROBE") == "1", "Set AIDOTNET_RUN_TC_GEMM_PROBE=1.");
        using var backend = RequireAmpereBackend();
        var runtime = backend.GetDirectPtxRuntime();
        const int m = 4096, n = 4096, k = 4096;
        var rng = new Random(3);
        using var da = backend.AllocateByteBuffer(m * k * 2);
        using var db = backend.AllocateByteBuffer(k * n * 2);
        using var dc = backend.AllocateBuffer(m * n);
        using var amax = backend.AllocateBuffer(2);
        backend.UploadBytes(da, HalfBytes(Enumerable.Range(0, m * k).Select(_ => (Half)(rng.NextDouble() * 2 - 1)).ToArray()));
        backend.UploadBytes(db, HalfBytes(Enumerable.Range(0, k * n).Select(_ => (Half)(rng.NextDouble() * 2 - 1)).ToArray()));
        double Time(Action launch)
        {
            for (int i = 0; i < 3; i++) launch();
            backend.Synchronize();
            using var start = backend.CreateEvent(enableTiming: true);
            using var end = backend.CreateEvent(enableTiming: true);
            backend.RecordEvent(start, backend.DefaultStream);
            for (int i = 0; i < 20; i++) launch();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            return backend.GetEventElapsedTime(start, end) / 20;
        }
        var layout = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxGemmLayout(false, false);
        double flops = 2d * m * n * k;
        string Tf(double ms) => $"{ms:F3} ms ({flops / ms / 1e9:F1} TFLOP/s)";
        backend.GemmRoutingEnabled = false;
        _output.WriteLine($"cuBLAS FP16->FP32: {Tf(Time(() => backend.CublasGemmFp16In32fOutForTests(da, db, dc, m, n, k)))}");
        _output.WriteLine($"max-abs of A and B: {Time(() => { backend.AbsMaxFp16Bits(da, m * k, amax.Handle); backend.AbsMaxFp16Bits(db, k * n, amax.Handle + 4); }):F3} ms");
        foreach (var config in new[] { new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmConfiguration(64, 128, 32, 64, 32, 3, 8),
                     new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmConfiguration(128, 128, 16, 64, 64, 3, 8) })
        {
            using var f32 = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmKernel(runtime, config, layout,
                AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCorePrecision.Fp16);
            using var f16 = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmKernel(runtime, config, layout,
                AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCorePrecision.Fp16, fp16Accumulate: true);
            using var guarded = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCoreGemmKernel(runtime, config, layout,
                AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx.PtxTensorCorePrecision.Fp16, fp16Accumulate: true, fp16AccumulateGuard: true);
            backend.AbsMaxFp16Bits(da, m * k, amax.Handle);
            backend.AbsMaxFp16Bits(db, k * n, amax.Handle + 4);
            _output.WriteLine($"{config.Name}: FP32 acc {Tf(Time(() => f32.Launch(da.Handle, db.Handle, dc.Handle, m, n, k, k, n, n, 1f, 0f)))}; " +
                $"FP16 acc {Tf(Time(() => f16.Launch(da.Handle, db.Handle, dc.Handle, m, n, k, k, n, n, 1f, 0f)))}; " +
                $"guarded (bounds ready) {Tf(Time(() => guarded.LaunchGuarded(da.Handle, db.Handle, dc.Handle, m, n, k, k, n, n, 1f, 0f, amax.Handle, amax.Handle + 4)))}; " +
                $"regs f32 {f32.Audit.Function.RegistersPerThread} f16 {f16.Audit.Function.RegistersPerThread} guarded {guarded.Audit.Function.RegistersPerThread}");
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