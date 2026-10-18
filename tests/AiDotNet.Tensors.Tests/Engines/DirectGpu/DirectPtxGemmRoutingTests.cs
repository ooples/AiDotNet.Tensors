#if NETCOREAPP
using System;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Tuned per-shape GEMM routing (#1265): an installed route runs its generated kernel through the public
/// <see cref="CudaBackend.Gemm"/> only in the math mode, and the launch mode (eager or graph capture), it won in, and
/// reproduces cuBLAS there; a tuning run's verdicts persist and activation follows them exactly.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DirectPtxGemmRoutingTests
{
    private readonly ITestOutputHelper _output;

    public DirectPtxGemmRoutingTests(ITestOutputHelper output) => _output = output;

    private static CudaBackend RequireAmpereBackend()
    {
        var backend = new CudaBackend();
        if (!backend.IsAvailable || backend.GetDirectPtxRuntime().ComputeCapabilityMajor < 8)
        {
            backend.Dispose();
            Skip.If(true, "Needs a CUDA device of compute capability 8.0 or newer.");
        }
        return backend;
    }

    private static float[] Values(int count, int seed)
    {
        var random = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() * 2 - 1);
        return values;
    }

    private static double MaxRelativeDifference(float[] expected, float[] actual)
    {
        double largest = 1e-30, worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            largest = Math.Max(largest, Math.Abs(expected[i]));
            worst = Math.Max(worst, Math.Abs((double)actual[i] - expected[i]));
        }
        return worst / largest;
    }

    [SkippableFact]
    public void InstalledRoute_RunsTheGeneratedKernel_OnlyInItsMathMode_AndMatchesCublas()
    {
        using var backend = RequireAmpereBackend();
        bool previous = AiDotNetEngine.DeterministicMode;
        try
        {
            AiDotNetEngine.SetDeterministicMode(true);
            const int m = 192, n = 256, k = 320;
            float[] hostC = Values(m * n, 3);
            using var a = backend.AllocateBuffer(Values(m * k, 1));
            using var b = backend.AllocateBuffer(Values(k * n, 2));
            using var c = backend.AllocateBuffer(hostC);

            // cuBLAS reference (no route installed), with alpha and beta so the epilogue's C read is exercised too.
            backend.Gemm(a, b, c, m, n, k, 0.5f, 2f);
            Assert.Equal(CudaGemmDispatchPath.Cublas, backend.LastGemmDispatchPath);
            float[] expected = backend.DownloadBuffer(c);

            backend.InstallSgemmRouteForTesting(PtxGemmEvolution.Seeds(new PtxGemmLayout(false, false), m, n, k)[0], m, n, k);
            using var c2 = backend.AllocateBuffer(hostC);
            long before = backend.GemmRouteDispatchCount;
            backend.Gemm(a, b, c2, m, n, k, 0.5f, 2f);
            Assert.Equal(CudaGemmDispatchPath.DirectPtxSgemm, backend.LastGemmDispatchPath);
            Assert.Equal(before + 1, backend.GemmRouteDispatchCount);
            double relative = MaxRelativeDifference(expected, backend.DownloadBuffer(c2));
            _output.WriteLine($"routed vs cuBLAS strict: max relative difference {relative:G3}");
            Assert.True(relative <= 1e-5, $"routed output differs from cuBLAS by {relative:G3}");

            // Another shape stays on cuBLAS.
            using var c3 = backend.AllocateBuffer(m * m);
            backend.Gemm(a, a, c3, m, m, m);
            Assert.Equal(CudaGemmDispatchPath.Cublas, backend.LastGemmDispatchPath);

            // The route was measured in strict FP32: with determinism off (TF32 cuBLAS) it must not be used.
            AiDotNetEngine.SetDeterministicMode(false);
            backend.Gemm(a, b, c2, m, n, k);
            Assert.Equal(CudaGemmDispatchPath.Cublas, backend.LastGemmDispatchPath);

            AiDotNetEngine.SetDeterministicMode(true);
            backend.ClearGemmRoutes();
            backend.Gemm(a, b, c2, m, n, k);
            Assert.Equal(CudaGemmDispatchPath.Cublas, backend.LastGemmDispatchPath);
            Assert.Equal(before + 1, backend.GemmRouteDispatchCount);
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(previous);
        }
    }

    [SkippableFact]
    public async Task TunedRoute_PersistsItsVerdict_AndActivationFollowsIt()
    {
        using var backend = RequireAmpereBackend();
        const string cacheVariable = "AIDOTNET_AUTOTUNE_CACHE_PATH";
        string? priorCache = Environment.GetEnvironmentVariable(cacheVariable);
        string cache = Path.Combine(Path.GetTempPath(), "aidotnet-gemm-route-" + Guid.NewGuid().ToString("N"));
        bool previous = AiDotNetEngine.DeterministicMode;
        Environment.SetEnvironmentVariable(cacheVariable, cache);
        try
        {
            AiDotNetEngine.SetDeterministicMode(true);
            const int m = 256, n = 256, k = 256;
            Assert.False(backend.TryActivatePersistedGemmRoute(m, n, k));   // nothing persisted yet

            var options = new AiDotNet.Evolution.EvolutionEngineOptions
            {
                RunId = "gemm-route-test", Seed = 1265, MaxEvaluationAttempts = 6, MaxProposals = 12, MaxGenerations = 2,
                ProposalBatchSize = 3, MaxDegreeOfParallelism = 1, IslandCount = 1, MigrationInterval = 0, MigrantsPerIsland = 1,
            };
            CudaGemmRouteDecision decision = await backend.TuneGemmRouteAsync(m, n, k, options);
            _output.WriteLine($"{decision.Kernel}: {decision.Reason}; cuBLAS {decision.CublasMicroseconds:F1} us, " +
                $"PTX {decision.DirectPtxMicroseconds:F1} us, max relative difference {decision.MaxRelativeDifference:G3}");
            Assert.True(decision.StrictFp32);
            Assert.True(decision.MaxRelativeDifference <= 1e-4);
            Assert.Equal(GemmRoutingRule(decision), decision.UsesDirectPtx);
            _output.WriteLine($"graph: {decision.GraphReason}; cuBLAS {decision.GraphCublasMicroseconds:F1} us, " +
                $"PTX {decision.GraphDirectPtxMicroseconds:F1} us");
            Assert.Equal(GraphRoutingRule(decision), decision.UsesDirectPtxInGraph);
            Assert.True(decision.GraphCublasMicroseconds > 0 && decision.GraphDirectPtxMicroseconds > 0,
                "the graph verdict was not measured: " + decision.GraphReason);

            // The tuning run installs exactly what it decided. Without a generated-kernel verdict the shape takes its
            // untuned path: here (256^3) the narrow skinny_gemm, which comes after a tuned route and before cuBLAS.
            var untuned = CudaGemmDispatchPath.NarrowGemm;
            using var a = backend.AllocateBuffer(Values(m * k, 1));
            using var b = backend.AllocateBuffer(Values(k * n, 2));
            using var c = backend.AllocateBuffer(m * n);
            backend.Gemm(a, b, c, m, n, k);
            Assert.Equal(decision.UsesDirectPtx ? CudaGemmDispatchPath.DirectPtxSgemm : untuned, backend.LastGemmDispatchPath);

            // A later activation (here: after clearing) follows the persisted verdict, in this math mode only.
            backend.ClearGemmRoutes();
            Assert.Equal(decision.UsesDirectPtx || decision.UsesDirectPtxInGraph, backend.TryActivatePersistedGemmRoute(m, n, k));
            backend.Gemm(a, b, c, m, n, k);
            Assert.Equal(decision.UsesDirectPtx ? CudaGemmDispatchPath.DirectPtxSgemm : untuned, backend.LastGemmDispatchPath);

            // A new backend (a later process) activates the verdict on the shape's first GEMM, with no explicit call.
            using (var fresh = new CudaBackend())
            {
                using var fa = fresh.AllocateBuffer(Values(m * k, 1));
                using var fb = fresh.AllocateBuffer(Values(k * n, 2));
                using var fc = fresh.AllocateBuffer(m * n);
                fresh.Gemm(fa, fb, fc, m, n, k);
                Assert.Equal(decision.UsesDirectPtx ? CudaGemmDispatchPath.DirectPtxSgemm : untuned, fresh.LastGemmDispatchPath);
            }

            AiDotNetEngine.SetDeterministicMode(false);
            Assert.False(backend.TryActivatePersistedGemmRoute(m, n, k));   // no TF32 verdict was measured
            Assert.False(backend.TryActivatePersistedGemmRoute(m, n, 2 * k));
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(previous);
            Environment.SetEnvironmentVariable(cacheVariable, priorCache);
            try { Directory.Delete(cache, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    // The documented rule: a median speedup of at least 2% and no round slower than cuBLAS.
    private static bool GemmRoutingRule(CudaGemmRouteDecision decision) =>
        decision.MedianSpeedup >= 1.02 && decision.MinimumSpeedup >= 1.0;

    private static bool GraphRoutingRule(CudaGemmRouteDecision decision) =>
        decision.GraphMedianSpeedup >= 1.02 && decision.GraphMinimumSpeedup >= 1.0;

    [SkippableFact]
    public void InstalledRoute_FollowsItsOwnVerdict_ForEagerLaunchesAndGraphCaptures()
    {
        using var backend = RequireAmpereBackend();
        bool previous = AiDotNetEngine.DeterministicMode;
        try
        {
            AiDotNetEngine.SetDeterministicMode(true);
            const int m = 128, n = 128, k = 128;
            using var a = backend.AllocateBuffer(Values(m * k, 1));
            using var b = backend.AllocateBuffer(Values(k * n, 2));
            using var c = backend.AllocateBuffer(m * n);
            backend.Gemm(a, b, c, m, n, k);
            float[] expected = backend.DownloadBuffer(c);
            var configuration = PtxGemmEvolution.Seeds(new PtxGemmLayout(false, false), m, n, k)[0];

            // Returns the path the captured GEMM took, and checks the replay reproduces cuBLAS.
            CudaGemmDispatchPath Captured()
            {
                using var captured = backend.AllocateBuffer(m * n);
                CudaGemmDispatchPath path = CudaGemmDispatchPath.None;
                IntPtr graph = backend.CaptureGraph(() =>
                {
                    backend.Gemm(a, b, captured, m, n, k);
                    path = backend.LastGemmDispatchPath;
                });
                Assert.NotEqual(IntPtr.Zero, graph);
                try
                {
                    backend.LaunchCapturedGraph(graph);
                    double relative = MaxRelativeDifference(expected, backend.DownloadBuffer(captured));
                    Assert.True(relative <= 1e-5, $"graph replay differs from cuBLAS by {relative:G3} ({path})");
                }
                finally
                {
                    backend.DestroyCapturedGraph(graph);
                }
                return path;
            }

            // Wins eagerly only (launch overhead): eager launches route, captures keep cuBLAS.
            backend.InstallSgemmRouteForTesting(configuration, m, n, k, eager: true, inGraph: false);
            backend.Gemm(a, b, c, m, n, k);
            Assert.Equal(CudaGemmDispatchPath.DirectPtxSgemm, backend.LastGemmDispatchPath);
            Assert.Equal(CudaGemmDispatchPath.Cublas, Captured());

            // Wins in graphs only: the reverse.
            backend.InstallSgemmRouteForTesting(configuration, m, n, k, eager: false, inGraph: true);
            backend.Gemm(a, b, c, m, n, k);
            Assert.Equal(CudaGemmDispatchPath.Cublas, backend.LastGemmDispatchPath);
            Assert.Equal(CudaGemmDispatchPath.DirectPtxSgemm, Captured());

            // Both.
            backend.InstallSgemmRouteForTesting(configuration, m, n, k);
            backend.Gemm(a, b, c, m, n, k);
            Assert.Equal(CudaGemmDispatchPath.DirectPtxSgemm, backend.LastGemmDispatchPath);
            Assert.Equal(CudaGemmDispatchPath.DirectPtxSgemm, Captured());
        }
        finally
        {
            backend.ClearGemmRoutes();
            AiDotNetEngine.SetDeterministicMode(previous);
        }
    }
}
#endif
