using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>The route the most recent <see cref="CudaBackend.Gemm"/> call took.</summary>
public enum CudaGemmDispatchPath
{
    /// <summary>No GEMM has been dispatched yet.</summary>
    None = 0,
    /// <summary>cuBLAS (<c>cublasSgemm</c>) in the backend's current math mode.</summary>
    Cublas = 1,
    /// <summary>A tuned, generated CUDA-core FP32 GEMM (strict FP32, the deterministic default).</summary>
    DirectPtxSgemm = 2,
    /// <summary>A tuned, generated TF32 Tensor-Core GEMM (determinism off, TF32 allowed).</summary>
    DirectPtxTensorCore = 3,
}

/// <summary>
/// The verdict of <see cref="CudaBackend.TuneGemmRouteAsync"/> for one shape and math mode: the evolved generated kernel
/// against cuBLAS in paired, alternating rounds on CUDA events, after checking both produce the same output. It is
/// measured twice: on eager launches, and as replays of a captured CUDA graph, where launch overhead is gone and only
/// device time counts. A shape can win one and lose the other, so each launch mode routes on its own verdict.
/// </summary>
public sealed class CudaGemmRouteDecision
{
    internal CudaGemmRouteDecision(int m, int n, int k, bool strictFp32, string kernel, double maxRelativeDifference,
        CudaBackend.GemmRouteVerdict eager, CudaBackend.GemmRouteVerdict graph)
    {
        M = m;
        N = n;
        K = k;
        StrictFp32 = strictFp32;
        Kernel = kernel;
        UsesDirectPtx = eager.Wins;
        MedianSpeedup = eager.MedianSpeedup;
        MinimumSpeedup = eager.MinimumSpeedup;
        CublasMicroseconds = eager.CublasMicroseconds;
        DirectPtxMicroseconds = eager.DirectPtxMicroseconds;
        Rounds = eager.Rounds;
        Reason = eager.Reason;
        UsesDirectPtxInGraph = graph.Wins;
        GraphMedianSpeedup = graph.MedianSpeedup;
        GraphMinimumSpeedup = graph.MinimumSpeedup;
        GraphCublasMicroseconds = graph.CublasMicroseconds;
        GraphDirectPtxMicroseconds = graph.DirectPtxMicroseconds;
        GraphReason = graph.Reason;
        MaxRelativeDifference = maxRelativeDifference;
    }

    /// <summary>Rows of A and C.</summary>
    public int M { get; }
    /// <summary>Columns of B and C.</summary>
    public int N { get; }
    /// <summary>Inner dimension.</summary>
    public int K { get; }
    /// <summary>True for the strict-FP32 (deterministic) mode, false for TF32.</summary>
    public bool StrictFp32 { get; }
    /// <summary>The evolved kernel's configuration name.</summary>
    public string Kernel { get; }
    /// <summary>Whether <see cref="CudaBackend.Gemm"/> now routes this shape's eager launches to the generated kernel.</summary>
    public bool UsesDirectPtx { get; }
    /// <summary>Median over rounds of cuBLAS time / generated-kernel time (above 1: the generated kernel is faster).</summary>
    public double MedianSpeedup { get; }
    /// <summary>Smallest per-round speedup.</summary>
    public double MinimumSpeedup { get; }
    /// <summary>Fastest cuBLAS round, microseconds per GEMM.</summary>
    public double CublasMicroseconds { get; }
    /// <summary>Fastest generated-kernel round, microseconds per GEMM.</summary>
    public double DirectPtxMicroseconds { get; }
    /// <summary>Paired rounds measured.</summary>
    public int Rounds { get; }
    /// <summary>Largest |generated - cuBLAS| over the output, relative to the largest |cuBLAS| output.</summary>
    public double MaxRelativeDifference { get; }
    /// <summary>Why the eager route was or was not taken.</summary>
    public string Reason { get; }
    /// <summary>Whether a GEMM of this shape recorded into a CUDA graph capture uses the generated kernel.</summary>
    public bool UsesDirectPtxInGraph { get; }
    /// <summary>As <see cref="MedianSpeedup"/>, timing replays of captured graphs.</summary>
    public double GraphMedianSpeedup { get; }
    /// <summary>As <see cref="MinimumSpeedup"/>, timing replays of captured graphs.</summary>
    public double GraphMinimumSpeedup { get; }
    /// <summary>Fastest cuBLAS graph-replay round, microseconds per GEMM.</summary>
    public double GraphCublasMicroseconds { get; }
    /// <summary>Fastest generated-kernel graph-replay round, microseconds per GEMM.</summary>
    public double GraphDirectPtxMicroseconds { get; }
    /// <summary>Why the graph route was or was not taken.</summary>
    public string GraphReason { get; }
}

/// <summary>
/// Tuned per-shape routing of <see cref="Gemm"/> between cuBLAS and the generated GEMMs (#1265, phase E of #377). A shape
/// routes to a generated kernel only after (1) the evolutionary search found a validated winner for this device, shape
/// and math mode, (2) that winner produced cuBLAS's output, and (3) it beat cuBLAS in paired alternating rounds. The
/// verdict persists in the autotune cache, so a later process activates it without measuring; everything else stays
/// on cuBLAS.
/// </summary>
public sealed partial class CudaBackend
{
    // A route must win, not tie: median speedup at least 2%, and no round slower than cuBLAS.
    private const double GemmRouteMedianMargin = 1.02;
    private const double GemmRouteMinimumSpeedup = 1.0;
    private const int GemmRouteRounds = 8;
    private const string GemmRouteVariant = "cuda-gemm-route-v1";
    private static readonly KernelId GemmRouteKernelId = new("gemm", "cuda-ptx-route");

    private readonly ConcurrentDictionary<GemmRouteKey, GemmRoute> _gemmRoutes = new();
    // Shapes whose persisted verdict was already looked up in this process (hit or miss): one lookup per shape and mode.
    private readonly ConcurrentDictionary<GemmRouteKey, byte> _gemmRouteLookups = new();

    // AIDOTNET_GEMM_ROUTES=0 keeps every GEMM on cuBLAS: no persisted verdict is activated.
    private static readonly bool GemmRouteActivationEnabled =
        Environment.GetEnvironmentVariable("AIDOTNET_GEMM_ROUTES") != "0";
    private int _gemmRouteCount;
    private int _lastGemmDispatchPath;
    private long _gemmRouteDispatchCount;

    private readonly record struct GemmRouteKey(int M, int N, int K, bool Strict);

    // One launch mode's paired-round comparison (eager launches or graph replays).
    internal readonly record struct GemmRouteVerdict(bool Wins, double MedianSpeedup, double MinimumSpeedup,
        double CublasMicroseconds, double DirectPtxMicroseconds, int Rounds, string Reason)
    {
        internal static GemmRouteVerdict Rejected(string reason) => new(false, 0, 0, 0, 0, 0, reason);
    }

    private sealed class GemmRoute : IDisposable
    {
        internal GemmRoute(IDisposable owner, CudaGemmDispatchPath path, Action<IntPtr, IntPtr, IntPtr, float, float> launch)
        {
            Owner = owner;
            Path = path;
            Launch = launch;
        }

        internal IDisposable Owner { get; }
        internal CudaGemmDispatchPath Path { get; }
        internal Action<IntPtr, IntPtr, IntPtr, float, float> Launch { get; }
        // The launch modes in which it measured faster than cuBLAS; set before the route is installed.
        internal bool Eager { get; set; } = true;
        internal bool InGraph { get; set; } = true;

        public void Dispose() => Owner.Dispose();
    }

    /// <summary>The route the most recent <see cref="Gemm"/> call took.</summary>
    public CudaGemmDispatchPath LastGemmDispatchPath => (CudaGemmDispatchPath)Volatile.Read(ref _lastGemmDispatchPath);

    internal long GemmRouteDispatchCount => Interlocked.Read(ref _gemmRouteDispatchCount);

    /// <summary>
    /// Benchmarks: false pins <see cref="Gemm"/> to cuBLAS on this backend (installed routes stay installed and nothing
    /// is activated), so a lane labelled cuBLAS measures cuBLAS even where a route exists.
    /// </summary>
    internal bool GemmRoutingEnabled { get; set; } = true;

    // Strict FP32 under determinism (the default) or when TF32 is unavailable or disabled; TF32 otherwise. The same rule
    // picks cuBLAS's math mode (ApplyDeterministicGemmMathMode), so a route always competes with the cuBLAS it replaces.
    private bool GemmRouteIsStrict =>
        GpuDeterminism.IsActive || _initGemmMathMode != CuBlasNative.CUBLAS_TF32_TENSOR_OP_MATH;

    /// <summary>
    /// Tunes the generated GEMM for <c>C[M,N] = A[M,K] @ B[K,N]</c> in the current math mode (strict FP32 when
    /// deterministic, TF32 otherwise), checks it reproduces cuBLAS's output, times it against cuBLAS in paired
    /// alternating rounds, and routes <see cref="Gemm"/> for this shape to it only when it wins. Eager launches and
    /// launches recorded into a CUDA graph capture are timed and routed separately. The verdicts, either way, are
    /// persisted (unless <paramref name="persist"/> is false) for <see cref="TryActivatePersistedGemmRoute"/>.
    /// </summary>
    public async Task<CudaGemmRouteDecision> TuneGemmRouteAsync(int m, int n, int k,
        EvolutionEngineOptions? engineOptions = null, KernelTuningOptions? tuningOptions = null, bool persist = true,
        CancellationToken cancellationToken = default)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (!IsAvailable) throw new GpuCapabilityUnavailableException("CUDA backend is not available.");
        DirectPtxRuntime runtime = GetDirectPtxRuntime();
        bool strict = GemmRouteIsStrict;
        var layout = new PtxGemmLayout(false, false);
        string genome, codecId, codecVersion, name;
        GemmRoute candidate;
        if (strict)
        {
            var result = await PtxGemmEvolution.TuneAsync(runtime, layout, m, n, k, engineOptions, tuningOptions,
                cancellationToken: cancellationToken).ConfigureAwait(false);
            PtxGemmConfiguration winner = result.ProposedWinner.Configuration;
            candidate = CreateSgemmRoute(runtime, winner, m, n, k);
            genome = PtxGemmEvolution.GenomeCodec.Serialize(winner);
            codecId = PtxGemmEvolution.GenomeCodec.Id;
            codecVersion = PtxGemmEvolution.GenomeCodec.VersionHash;
            name = winner.Name;
        }
        else
        {
            var result = await PtxTensorCoreGemmEvolution.TuneAsync(runtime, layout, m, n, k, PtxTensorCorePrecision.Tf32,
                engineOptions, tuningOptions, cancellationToken: cancellationToken).ConfigureAwait(false);
            PtxTensorCoreGemmConfiguration winner = result.ProposedWinner.Configuration;
            candidate = CreateTensorCoreRoute(runtime, winner, m, n, k);
            genome = PtxTensorCoreGemmEvolution.GenomeCodec.Serialize(winner);
            codecId = PtxTensorCoreGemmEvolution.GenomeCodec.Id;
            codecVersion = PtxTensorCoreGemmEvolution.GenomeCodec.VersionHash;
            name = winner.Name;
        }

        CudaGemmRouteDecision decision;
        try
        {
            decision = CompareWithCublas(candidate, name, m, n, k, strict);
        }
        catch
        {
            candidate.Dispose();
            throw;
        }
        if (decision.UsesDirectPtx || decision.UsesDirectPtxInGraph)
        {
            candidate.Eager = decision.UsesDirectPtx;
            candidate.InGraph = decision.UsesDirectPtxInGraph;
            InstallGemmRoute(new GemmRouteKey(m, n, k, strict), candidate);
        }
        else
        {
            candidate.Dispose();
            RemoveGemmRoute(new GemmRouteKey(m, n, k, strict));
        }
        if (persist) PersistGemmRoute(runtime, decision, genome, codecId, codecVersion);
        return decision;
    }

    /// <summary>
    /// Activates a persisted route for this shape in the current math mode without measuring: true when a verdict for
    /// this device says the generated kernel wins on eager launches, inside graph captures, or both, and it compiled;
    /// false (cuBLAS stays) for no verdict, cuBLAS verdicts in both modes, another device, or a stale genome. A verdict
    /// persisted before graph verdicts were measured routes graph captures to cuBLAS.
    /// </summary>
    public bool TryActivatePersistedGemmRoute(int m, int n, int k)
    {
        if (m <= 0 || n <= 0 || k <= 0 || !IsAvailable) return false;
        bool strict = GemmRouteIsStrict;
        KernelChoice? choice;
        try
        {
            choice = AutotuneCache.Lookup(GemmRouteKernelId, RouteShape(m, n, k, strict));
        }
        catch
        {
            return false;
        }
        if (choice is null || !string.Equals(choice.Variant, GemmRouteVariant, StringComparison.Ordinal) ||
            choice.Parameters is null) return false;
        Dictionary<string, string> p = choice.Parameters;
        DirectPtxRuntime runtime = GetDirectPtxRuntime();
        bool eager = Matches(p, "route", "ptx"), inGraph = Matches(p, "graph-route", "ptx");
        if (!Matches(p, "device", PtxGemmEvolution.Fingerprint(runtime).LocalKey) || !(eager || inGraph) ||
            !p.TryGetValue("genome", out string? genome)) return false;
        try
        {
            GemmRoute route;
            if (strict)
            {
                if (!Matches(p, "codec-id", PtxGemmEvolution.GenomeCodec.Id) ||
                    !Matches(p, "codec-version", PtxGemmEvolution.GenomeCodec.VersionHash)) return false;
                PtxGemmConfiguration configuration = PtxGemmEvolution.GenomeCodec.Deserialize(genome);
                route = CreateSgemmRoute(runtime, configuration, m, n, k);
            }
            else
            {
                if (!Matches(p, "codec-id", PtxTensorCoreGemmEvolution.GenomeCodec.Id) ||
                    !Matches(p, "codec-version", PtxTensorCoreGemmEvolution.GenomeCodec.VersionHash)) return false;
                PtxTensorCoreGemmConfiguration configuration = PtxTensorCoreGemmEvolution.GenomeCodec.Deserialize(genome);
                route = CreateTensorCoreRoute(runtime, configuration, m, n, k);
            }
            route.Eager = eager;
            route.InGraph = inGraph;
            InstallGemmRoute(new GemmRouteKey(m, n, k, strict), route);
            return true;
        }
        catch (Exception exception) when (exception is InvalidOperationException or ArgumentException or
                                             PlatformNotSupportedException or System.IO.InvalidDataException)
        {
            return false;
        }
    }

    /// <summary>Drops every installed route; <see cref="Gemm"/> uses cuBLAS for all shapes again.</summary>
    public void ClearGemmRoutes()
    {
        foreach (var key in _gemmRoutes.Keys) RemoveGemmRoute(key);
    }

    // Tests: installs a strict-FP32 route without tuning or measuring, to exercise the dispatch wiring itself.
    internal void InstallSgemmRouteForTesting(PtxGemmConfiguration configuration, int m, int n, int k, bool eager = true,
        bool inGraph = true)
    {
        GemmRoute route = CreateSgemmRoute(GetDirectPtxRuntime(), configuration, m, n, k);
        route.Eager = eager;
        route.InGraph = inGraph;
        InstallGemmRoute(new GemmRouteKey(m, n, k, Strict: true), route);
    }

    private static bool Matches(Dictionary<string, string> parameters, string key, string expected) =>
        parameters.TryGetValue(key, out string? value) && string.Equals(value, expected, StringComparison.Ordinal);

    private static ShapeProfile RouteShape(int m, int n, int k, bool strict) => new(m, n, k, strict ? 1 : 0);

    // The route launches the kernel on the problem's row-major operands: lda = K, ldb = N, ldc = N.
    private static GemmRoute CreateSgemmRoute(DirectPtxRuntime runtime, PtxGemmConfiguration configuration, int m, int n, int k)
    {
        var layout = new PtxGemmLayout(false, false);
        if (!PtxGemmEvolution.Admits(configuration, layout, m, n, k))
            throw new ArgumentException($"GEMM configuration {configuration.Name} does not admit M={m} N={n} K={k}.");
        var kernel = new PtxSgemmKernel(runtime, configuration, layout);
        return new GemmRoute(kernel, CudaGemmDispatchPath.DirectPtxSgemm,
            (a, b, c, alpha, beta) => kernel.Launch(a, b, c, m, n, k, k, n, n, alpha, beta));
    }

    private static GemmRoute CreateTensorCoreRoute(DirectPtxRuntime runtime, PtxTensorCoreGemmConfiguration configuration,
        int m, int n, int k)
    {
        var layout = new PtxGemmLayout(false, false);
        if (!PtxTensorCoreGemmEvolution.Admits(configuration, layout, m, n, k, PtxTensorCorePrecision.Tf32))
            throw new ArgumentException($"Tensor-Core GEMM {configuration.Name} does not admit M={m} N={n} K={k}.");
        var kernel = new PtxTensorCoreGemmKernel(runtime, configuration, layout);
        return new GemmRoute(kernel, CudaGemmDispatchPath.DirectPtxTensorCore,
            (a, b, c, alpha, beta) => kernel.Launch(a, b, c, m, n, k, k, n, n, alpha, beta));
    }

    private void InstallGemmRoute(GemmRouteKey key, GemmRoute route)
    {
        GemmRoute? previous = null;
        _gemmRoutes.AddOrUpdate(key, route, (_, old) =>
        {
            previous = old;
            return route;
        });
        if (previous is null) Interlocked.Increment(ref _gemmRouteCount);
        else if (!ReferenceEquals(previous, route)) DisposeGemmRouteAfterQuiesce(previous);
    }

    private void RemoveGemmRoute(GemmRouteKey key)
    {
        if (!_gemmRoutes.TryRemove(key, out GemmRoute? removed)) return;
        Interlocked.Decrement(ref _gemmRouteCount);
        DisposeGemmRouteAfterQuiesce(removed);
    }

    // A replaced route's module may still have launches queued: unloading it is safe only once they completed.
    private void DisposeGemmRouteAfterQuiesce(GemmRoute route)
    {
        lock (GpuDispatchLock)
        {
            Synchronize();
            route.Dispose();
        }
    }

    private void DisposeGemmRoutes()
    {
        foreach (var key in _gemmRoutes.Keys)
            if (_gemmRoutes.TryRemove(key, out GemmRoute? route)) route.Dispose();
        Volatile.Write(ref _gemmRouteCount, 0);
    }

    /// <summary>
    /// Runs an installed route for this shape and math mode. False (the caller uses cuBLAS) when none is installed, its
    /// verdict for the current launch mode (eager or graph capture) is cuBLAS, an operand is not 16-byte aligned, or a
    /// buffer's extent differs from the problem's.
    /// </summary>
    private bool TryRoutedGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k, float alpha, float beta)
    {
        if (!GemmRouteActivationEnabled || !GemmRoutingEnabled) return false;
        var key = new GemmRouteKey(m, n, k, GemmRouteIsStrict);
        // First GEMM of a shape in this mode: activate its persisted verdict, if any. Never inside a stream capture,
        // where loading a module is not allowed; the shape is then looked up on its first uncaptured call.
        if (!_gemmRouteLookups.ContainsKey(key) && !IsStreamCapturing() && _gemmRouteLookups.TryAdd(key, 0))
            TryActivatePersistedGemmRoute(m, n, k);
        if (Volatile.Read(ref _gemmRouteCount) == 0) return false;
        if (!_gemmRoutes.TryGetValue(key, out GemmRoute? route)) return false;
        // Each launch mode follows its own verdict: a kernel can win on launch overhead alone, which a graph removes.
        if (!(route.Eager && route.InGraph) && (IsStreamCapturingCurrentContext() ? !route.InGraph : !route.Eager))
            return false;
        if (((long)a.Handle & 15) != 0 || ((long)b.Handle & 15) != 0 || ((long)c.Handle & 15) != 0) return false;
        if (a.SizeInBytes < (long)m * k * sizeof(float) || b.SizeInBytes < (long)k * n * sizeof(float) ||
            c.SizeInBytes < (long)m * n * sizeof(float)) return false;
        lock (GpuDispatchLock) route.Launch(a.Handle, b.Handle, c.Handle, alpha, beta);
        Interlocked.Increment(ref _gemmRouteDispatchCount);
        Volatile.Write(ref _lastGemmDispatchPath, (int)route.Path);
        return true;
    }

    // Output check, then paired rounds: cuBLAS, the candidate, the candidate, cuBLAS (ABBA), each timed on CUDA events,
    // so clock, thermal and background-load drift hit both arms alike; the speedup is per round. The rounds run twice:
    // as eager launches, and as replays of graphs that captured the same launches.
    private CudaGemmRouteDecision CompareWithCublas(GemmRoute candidate, string name, int m, int n, int k, bool strict)
    {
        using var context = PushContext();
        var random = new Random(1265);
        var hostA = new float[checked(m * k)];
        var hostB = new float[checked(k * n)];
        for (int i = 0; i < hostA.Length; i++) hostA[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < hostB.Length; i++) hostB[i] = (float)(random.NextDouble() * 2 - 1);
        using IGpuBuffer a = AllocateBuffer(hostA);
        using IGpuBuffer b = AllocateBuffer(hostB);
        using IGpuBuffer c = AllocateBuffer(checked(m * n));
        Action cublas = () => CublasGemm(a, b, c, m, n, k, 1f, 0f);
        Action ptx = () => candidate.Launch(a.Handle, b.Handle, c.Handle, 1f, 0f);

        cublas();
        float[] expected = DownloadBuffer(c);
        ptx();
        float[] actual = DownloadBuffer(c);
        double largest = 1e-30, worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            largest = Math.Max(largest, Math.Abs(expected[i]));
            double difference = Math.Abs((double)actual[i] - expected[i]);
            if (double.IsNaN(difference)) difference = double.PositiveInfinity;
            worst = Math.Max(worst, difference);
        }
        double relative = worst / largest;
        // Both arms sum K products in FP32 in different orders; TF32 also rounds the operands (on both arms, though not
        // necessarily the same way).
        double tolerance = strict ? 1e-4 : 1e-2;
        if (!(relative <= tolerance))
        {
            var rejected = GemmRouteVerdict.Rejected(
                $"output differs from cuBLAS by {relative:G3} of the largest output (tolerance {tolerance:G3})");
            return new CudaGemmRouteDecision(m, n, k, strict, name, relative, rejected, rejected);
        }

        double work = 2d * m * n * k;
        int reps = (int)Math.Max(1, Math.Min(200, Math.Ceiling(2e10 / work)));
        for (int i = 0; i < 3; i++)
        {
            cublas();
            ptx();
        }
        Synchronize();
        GemmRouteVerdict eager = PairedRounds(() => { for (int i = 0; i < reps; i++) cublas(); },
            () => { for (int i = 0; i < reps; i++) ptx(); }, reps);
        GemmRouteVerdict graph = GraphRounds(cublas, ptx, reps);
        return new CudaGemmRouteDecision(m, n, k, strict, name, relative, eager, graph);
    }

    // Each arm captured as one graph of `reps` launches; the replays are then compared in paired rounds.
    private GemmRouteVerdict GraphRounds(Action cublas, Action ptx, int reps)
    {
        IntPtr cublasGraph = IntPtr.Zero, ptxGraph = IntPtr.Zero;
        try
        {
            cublasGraph = CaptureGraph(() => { for (int i = 0; i < reps; i++) cublas(); });
            ptxGraph = CaptureGraph(() => { for (int i = 0; i < reps; i++) ptx(); });
            if (cublasGraph == IntPtr.Zero || ptxGraph == IntPtr.Zero)
                return GemmRouteVerdict.Rejected("graph capture failed; graphs keep cuBLAS");
            IntPtr cublasExec = cublasGraph, ptxExec = ptxGraph;
            for (int i = 0; i < 3; i++)
            {
                EnqueueCapturedGraph(cublasExec);
                EnqueueCapturedGraph(ptxExec);
            }
            Synchronize();
            return PairedRounds(() => EnqueueCapturedGraph(cublasExec), () => EnqueueCapturedGraph(ptxExec), reps);
        }
        // A capture that cannot be set up or replayed costs only the graph verdict: the eager verdict already measured
        // stands. A sticky fault (the GPU circuit breaker tripped) is not recoverable and propagates.
        catch (InvalidOperationException exception) when (!AiDotNetEngine.GpuCircuitBroken)
        {
            return GemmRouteVerdict.Rejected($"graph measurement failed ({exception.Message}); graphs keep cuBLAS");
        }
        finally
        {
            if (cublasGraph != IntPtr.Zero) DestroyCapturedGraph(cublasGraph);
            if (ptxGraph != IntPtr.Zero) DestroyCapturedGraph(ptxGraph);
        }
    }

    // ABBA rounds of `cublas` against `ptx`, each running `reps` GEMMs; microseconds per GEMM from CUDA events.
    private GemmRouteVerdict PairedRounds(Action cublas, Action ptx, int reps)
    {
        double Round(Action launch)
        {
            using IGpuEvent start = CreateEvent(enableTiming: true);
            using IGpuEvent end = CreateEvent(enableTiming: true);
            RecordEvent(start, DefaultStream);
            launch();
            RecordEvent(end, DefaultStream);
            end.Synchronize();
            return GetEventElapsedTime(start, end) * 1000.0 / reps;
        }
        var speedups = new double[GemmRouteRounds];
        double bestCublas = double.MaxValue, bestPtx = double.MaxValue;
        for (int r = 0; r < GemmRouteRounds; r++)
        {
            double c1 = Round(cublas), p1 = Round(ptx), p2 = Round(ptx), c2 = Round(cublas);
            speedups[r] = (c1 + c2) / (p1 + p2);
            bestCublas = Math.Min(bestCublas, Math.Min(c1, c2));
            bestPtx = Math.Min(bestPtx, Math.Min(p1, p2));
        }
        Array.Sort(speedups);
        double median = (speedups[(GemmRouteRounds - 1) / 2] + speedups[GemmRouteRounds / 2]) / 2;
        double minimum = speedups[0];
        bool wins = median >= GemmRouteMedianMargin && minimum >= GemmRouteMinimumSpeedup;
        string reason = wins
            ? $"faster than cuBLAS: median {median:F3}x, every round at least {minimum:F3}x"
            : $"not faster than cuBLAS: median {median:F3}x (needs {GemmRouteMedianMargin:F2}), minimum {minimum:F3}x (needs {GemmRouteMinimumSpeedup:F2})";
        return new GemmRouteVerdict(wins, median, minimum, bestCublas, bestPtx, GemmRouteRounds, reason);
    }

    private static void PersistGemmRoute(DirectPtxRuntime runtime, CudaGemmRouteDecision decision, string genome,
        string codecId, string codecVersion)
    {
        var choice = new KernelChoice
        {
            Variant = GemmRouteVariant,
            MeasuredTimeMs = (decision.UsesDirectPtx ? decision.DirectPtxMicroseconds : decision.CublasMicroseconds) / 1000.0,
            MeasuredGflops = decision.CublasMicroseconds > 0 || decision.DirectPtxMicroseconds > 0
                ? 2d * decision.M * decision.N * decision.K /
                  ((decision.UsesDirectPtx ? decision.DirectPtxMicroseconds : decision.CublasMicroseconds) * 1000.0)
                : 0,
            Parameters = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["device"] = PtxGemmEvolution.Fingerprint(runtime).LocalKey,
                ["route"] = decision.UsesDirectPtx ? "ptx" : "cublas",
                ["family"] = decision.StrictFp32 ? "sgemm" : "tf32",
                ["kernel"] = decision.Kernel,
                ["genome"] = genome,
                ["codec-id"] = codecId,
                ["codec-version"] = codecVersion,
                ["median-speedup"] = decision.MedianSpeedup.ToString("R", CultureInfo.InvariantCulture),
                ["minimum-speedup"] = decision.MinimumSpeedup.ToString("R", CultureInfo.InvariantCulture),
                ["max-relative-difference"] = decision.MaxRelativeDifference.ToString("R", CultureInfo.InvariantCulture),
                ["reason"] = decision.Reason,
                ["graph-route"] = decision.UsesDirectPtxInGraph ? "ptx" : "cublas",
                ["graph-median-speedup"] = decision.GraphMedianSpeedup.ToString("R", CultureInfo.InvariantCulture),
                ["graph-minimum-speedup"] = decision.GraphMinimumSpeedup.ToString("R", CultureInfo.InvariantCulture),
                ["graph-reason"] = decision.GraphReason,
            },
        };
        AutotuneCache.TryStore(GemmRouteKernelId,
            RouteShape(decision.M, decision.N, decision.K, decision.StrictFp32), choice);
    }
}
