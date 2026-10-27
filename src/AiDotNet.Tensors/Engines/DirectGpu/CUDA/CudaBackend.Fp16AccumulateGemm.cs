// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// FP16-accumulate routing of <see cref="GemmFp16In32fOut"/> and <see cref="GemmFp16HalfOut"/> (#1549; the FP16 conv
/// path runs on the latter). On GeForce cards FP16-accumulate tensor-core MMA
/// issues at full rate while FP32-accumulate issues at half rate, so a GEMM that accumulates each K tile in FP16 and
/// promotes it into an FP32 total can run up to twice as fast as cuBLAS's FP32-accumulate GEMM. Overflow is handled in
/// the kernel: a partial sum past 65504 becomes infinite (finite FP16 products cannot overflow the FP32 total, so a
/// non-finite accumulator after the loop means an FP16 partial overflowed or an operand was already non-finite), and a
/// block that sees one recomputes its tile with FP32 accumulation, so the output never depends on operand magnitude.
/// The first eager call of a large shape times candidates against cuBLAS in paired rounds and routes the shape only
/// when the best one wins; the verdict lives for the process. Before the first such measurement the device's MMA issue
/// rates are probed once (<see cref="PtxMmaRateProbe"/>): where FP32 accumulation already issues at the FP16 rate (a
/// datacenter or workstation part, or a GeForce card unlocked by a speed-select module) nothing can double and no shape
/// is tuned. AIDOTNET_FP16_ACCUMULATE=0 turns the route off; =1 tunes even at full rate.
/// </summary>
public sealed partial class CudaBackend
{
    private static readonly bool s_fp16AccumulateEnabled =
        Environment.GetEnvironmentVariable("AIDOTNET_FP16_ACCUMULATE") != "0";
    private static readonly bool s_fp16AccumulateForced =
        Environment.GetEnvironmentVariable("AIDOTNET_FP16_ACCUMULATE") == "1";

    // Below this much work the GEMM is bound by memory or latency, not the MMA rate, so FP16 accumulation buys nothing
    // (RTX 3080 probe: 3136x64x576 0.99x), and timing it against cuBLAS would cost more than it can return.
    private const long Fp16AccumulateMinWork = 1L << 29;

    // The fastest configurations of the #1549 probes across the measured shapes (the first two: the evolved winners of
    // 7 shapes out of 7 on an RTX 3080).
    private static readonly PtxTensorCoreGemmConfiguration[] s_fp16AccumulateCandidates =
    {
        new(128, 128, 16, 64, 64, 2, 8),
        new(64, 64, 32, 32, 32, 2, 8),
        new(64, 128, 32, 64, 32, 3, 8),
        new(128, 128, 16, 64, 64, 3, 8),
        new(128, 64, 16, 64, 32, 4, 8),
        new(64, 64, 32, 32, 32, 4, 8),
        new(64, 64, 16, 32, 32, 3, 8),
    };

    private const string Fp16AccumulateRouteVariant = "cuda-fp16acc-route-v3";
    private static readonly KernelId Fp16AccumulateRouteKernelId = new("gemm", "cuda-ptx-fp16acc-route");

    private sealed class Fp16AccumulateRoute : IDisposable
    {
        internal Fp16AccumulateRoute(PtxTensorCoreGemmKernel? kernel, string reason, double medianSpeedup)
        {
            Kernel = kernel;
            Reason = reason;
            MedianSpeedup = medianSpeedup;
        }

        internal PtxTensorCoreGemmKernel? Kernel { get; }
        internal string Reason { get; }
        internal double MedianSpeedup { get; }
        public void Dispose() => Kernel?.Dispose();
    }

    // One routed problem: C[M, N] = A[M, K] @ B for each of Batch problems. Batch 1 is the plain row-major GEMM; above one
    // it is the FP16 im2col conv's per-image GEMM (IFp16ConvBatchedGemmBackend): B is image b's slice of col[K, Batch * N]
    // (ld Batch * N, next image N columns on), A the shared weights, C image b's NCHW block (ld N, next image M * N on).
    private readonly record struct Fp16AccumulateProblem(int M, int N, int K, bool HalfOutput, int Batch)
    {
        internal int Ldb => Batch * N;
        internal long StrideB => N;
        internal long StrideC => (long)M * N;
        internal long OutputElements => (long)Batch * M * N;
        internal long Work => (long)Batch * M * N * K;
    }

    private readonly ConcurrentDictionary<Fp16AccumulateProblem, Fp16AccumulateRoute> _fp16AccumulateRoutes = new();
    private readonly object _fp16AccumulateTuneLock = new();
    // Routes replaced by TuneFp16AccumulateRouteAsync (or the test hook). A GEMM may have read one on the lock-free fast
    // path and still be about to launch its kernel, so they are disposed only with the backend.
    private readonly List<Fp16AccumulateRoute> _retiredFp16AccumulateRoutes = new();

    /// <summary>
    /// The FP16-accumulate check's tolerance, relative to max |C|: each K tile's partial sum is rounded to FP16 (2^-11)
    /// before it reaches the FP32 total. Shared by the route's cuBLAS check and the evolution's acceptance check.
    /// </summary>
    internal const double Fp16AccumulateRelativeTolerance = 5e-3;
    private PtxMmaRateProbe.Result? _mmaRate;   // measured once, under _fp16AccumulateTuneLock
    private long _fp16AccumulateDispatchCount;

    /// <summary>Routed FP16-accumulate GEMM launches on this backend (test and diagnostics hook).</summary>
    internal long Fp16AccumulateDispatchCount => Interlocked.Read(ref _fp16AccumulateDispatchCount);

    /// <summary>The verdict for one shape, or null when it was never evaluated (test and diagnostics hook).</summary>
    internal (bool Routed, string Reason, double MedianSpeedup)? Fp16AccumulateVerdict(int m, int n, int k,
        bool halfOutput = false, int batch = 1) =>
        _fp16AccumulateRoutes.TryGetValue(new Fp16AccumulateProblem(m, n, k, halfOutput, batch), out var route)
            ? (route.Kernel is not null, route.Reason, route.MedianSpeedup)
            : null;

    /// <summary>
    /// Tests: routes a shape to the FP16-accumulate kernel of <paramref name="config"/> without timing it, so the
    /// routed path (including its in-kernel FP32 redo) can be checked independently of which route measures faster.
    /// </summary>
    internal void ForceFp16AccumulateRouteForTests(int m, int n, int k, PtxTensorCoreGemmConfiguration config,
        bool halfOutput = false, int batch = 1)
    {
        var kernel = new PtxTensorCoreGemmKernel(GetDirectPtxRuntime(), config, new PtxGemmLayout(false, false),
            PtxTensorCorePrecision.Fp16, halfOutput, fp16Accumulate: true);
        var problem = new Fp16AccumulateProblem(m, n, k, halfOutput, batch);
        ReplaceFp16AccumulateRoute(problem, new Fp16AccumulateRoute(kernel, "forced for tests", 0));
    }

    /// <summary>
    /// Runs <c>C[M,N] = A[M,K] @ B[K,N]</c> (FP16 in, FP32 or FP16 out, row-major) through the FP16-accumulate kernel when
    /// this shape is routed; false leaves the call to cuBLAS.
    /// </summary>
    internal bool TryFp16AccumulateGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k,
        bool halfOutput = false) =>
        TryFp16Accumulate(new Fp16AccumulateProblem(m, n, k, halfOutput, 1), a, b, c);

    /// <summary>
    /// The FP16 im2col conv's per-image GEMM (<see cref="GemmFp16ConvBatched"/>) through the FP16-accumulate kernel, batched
    /// over images with strides, when this shape is routed; false leaves it to cuBLAS's strided-batched GEMM.
    /// </summary>
    internal bool TryFp16AccumulateConvBatched(IGpuBuffer weights, IGpuBuffer col, IGpuBuffer output, int m, int pixels,
        int k, int batch, bool halfOutput) =>
        TryFp16Accumulate(new Fp16AccumulateProblem(m, pixels, k, halfOutput, batch), weights, col, output);

    private void ReplaceFp16AccumulateRoute(Fp16AccumulateProblem problem, Fp16AccumulateRoute route)
    {
        lock (_fp16AccumulateTuneLock)
        {
            if (_fp16AccumulateRoutes.TryGetValue(problem, out var previous)) _retiredFp16AccumulateRoutes.Add(previous);
            _fp16AccumulateRoutes[problem] = route;
        }
    }

    private bool TryFp16Accumulate(Fp16AccumulateProblem problem, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c)
    {
        if (!s_fp16AccumulateEnabled || _ccMajor < 8 || problem.Work < Fp16AccumulateMinWork) return false;
        if (!_fp16AccumulateRoutes.TryGetValue(problem, out var route))
        {
            // Tuning measures with synchronizing events, which a stream capture cannot record: capture keeps cuBLAS until
            // an eager call of the shape has decided.
            if (IsStreamCapturing()) return false;
            lock (_fp16AccumulateTuneLock)
            {
                if (!_fp16AccumulateRoutes.TryGetValue(problem, out route))
                {
                    // Where FP32 accumulation already issues at full rate nothing can double: no shape is tuned. A
                    // verdict persisted by TuneFp16AccumulateRouteAsync (an evolved kernel for this device and rate)
                    // wins over the quick pick among a few fixed candidates.
                    route = Fp16AccumulateRateRejection() ??
                            TryLoadPersistedFp16AccumulateRoute(problem) ??
                            TuneAndRecordFp16AccumulateRoute(a, b, c, problem);
                    _fp16AccumulateRoutes[problem] = route;
                }
            }
        }
        if (route.Kernel is null) return false;
        LaunchFp16Accumulate(route.Kernel, a, b, c, problem);
        Interlocked.Increment(ref _fp16AccumulateDispatchCount);
        return true;
    }

    /// <summary>
    /// Measures this device's tensor-core issue rate for FP16 MMA with FP32 accumulation and with FP16 accumulation
    /// (#1549). On GeForce cards the FP32-accumulate form issues at a fraction of the FP16-accumulate rate (RTX 3080:
    /// about 62 against 123 TFLOP/s); datacenter and workstation parts, and GeForce cards unlocked by a speed-select
    /// module, issue both at the same rate. Takes a few hundred milliseconds; needs compute capability 8.0+.
    /// </summary>
    public (double Fp32AccumulateTflops, double Fp16AccumulateTflops, bool Fp32AccumulateIsFullRate)
        MeasureTensorCoreAccumulateRates()
    {
        if (!IsAvailable) throw new GpuCapabilityUnavailableException("CUDA backend is not available.");
        var result = PtxMmaRateProbe.Measure(GetDirectPtxRuntime());
        return (result.Fp32AccumulateTflops, result.Fp16AccumulateTflops, result.Fp32AccumulateIsFullRate);
    }

    // The device's rate class, measured on first use (null when the probe cannot run here).
    private PtxMmaRateProbe.Result? MmaRate()
    {
        if (_mmaRate is null)
        {
            try
            {
                _mmaRate = PtxMmaRateProbe.Measure(GetDirectPtxRuntime());
            }
            catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
            {
                return null;
            }
        }
        return _mmaRate;
    }

    /// <summary>Tests: the rate the route believes this device has (as if measured), e.g. an unlocked card's.</summary>
    internal void ForceMmaRateForTests(double fp32AccumulateTflops, double fp16AccumulateTflops) =>
        _mmaRate = new PtxMmaRateProbe.Result(fp32AccumulateTflops, fp16AccumulateTflops);

    // A route that keeps cuBLAS because FP32 accumulation already runs at full rate, or null to go on.
    private Fp16AccumulateRoute? Fp16AccumulateRateRejection()
    {
        if (s_fp16AccumulateForced || MmaRate() is not { Fp32AccumulateIsFullRate: true } rate) return null;
        return new Fp16AccumulateRoute(null,
            $"FP32 accumulation already issues at the FP16 rate on this device ({rate.Fp32AccumulateTflops:F0} against " +
            $"{rate.Fp16AccumulateTflops:F0} TFLOP/s): FP16 accumulation cannot double it", 0);
    }

    private void LaunchFp16Accumulate(PtxTensorCoreGemmKernel kernel, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        Fp16AccumulateProblem p)
    {
        using var _ = PushContext();
        lock (GpuDispatchLock)
            kernel.Launch(a.Handle, b.Handle, c.Handle, p.M, p.N, p.K, p.K, p.Ldb, p.N, 1f, 0f,
                p.Batch, 0, p.StrideB, p.StrideC, stream: _stream);
    }

    private void CublasFp16Accumulate(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, Fp16AccumulateProblem p)
    {
        if (p.Batch == 1) CublasFp16Gemm(a, b, c, p.M, p.N, p.K, p.HalfOutput);
        else CublasGemmFp16ConvBatched(a, b, c, p.M, p.N, p.K, p.Batch, p.HalfOutput);
    }

    private static bool AdmitsFp16Accumulate(PtxTensorCoreGemmConfiguration config, PtxGemmLayout layout, Fp16AccumulateProblem p) =>
        PtxTensorCoreGemmGenerator.Admits(config, layout, p.M, p.N, p.K, p.K, p.Ldb, p.N,
            0, p.Batch > 1 ? p.StrideB : 0, p.Batch > 1 ? p.StrideC : 0, PtxTensorCorePrecision.Fp16);

    // A first call's measured verdict, recorded so every later process applies the same one. That recording is what keeps
    // GpuDeterminism (on by default) as cuBLASLt's recorded algorithm rank does: a verdict that could not be recorded
    // might be measured differently by the next process, so under determinism it keeps cuBLAS instead.
    private Fp16AccumulateRoute TuneAndRecordFp16AccumulateRoute(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        Fp16AccumulateProblem problem)
    {
        Fp16AccumulateRoute route = TuneFp16AccumulateRoute(a, b, c, problem);
        bool recorded = PersistFp16AccumulateRoute(problem, route, route.Kernel?.Configuration);
        if (recorded || route.Kernel is null || !GpuDeterminism.IsActive) return route;
        route.Dispose();
        return new Fp16AccumulateRoute(null,
            $"deterministic mode: the verdict ({route.Reason}) could not be recorded, so cuBLAS runs", route.MedianSpeedup);
    }

    private static ShapeProfile Fp16AccumulateShape(Fp16AccumulateProblem p) => p.Batch == 1
        ? new ShapeProfile(p.M, p.N, p.K, p.HalfOutput ? 1 : 0)
        : new ShapeProfile(p.M, p.N, p.K, p.HalfOutput ? 1 : 0, p.Batch);

    /// <summary>Tests: the verdict cache is unavailable (nothing loads, every record is refused).</summary>
    internal bool Fp16AccumulateCacheUnavailableForTests { get; set; }

    // Records a verdict (the kernel's genome when it routes) for this device and MMA rate; false when the cache refuses.
    private bool PersistFp16AccumulateRoute(Fp16AccumulateProblem problem, Fp16AccumulateRoute route,
        PtxTensorCoreGemmConfiguration? configuration)
    {
        if (Fp16AccumulateCacheUnavailableForTests) return false;
        try
        {
            var parameters = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["device"] = PtxGemmEvolution.Fingerprint(GetDirectPtxRuntime()).LocalKey,
                ["mma-rate"] = MmaRate()?.RateClass ?? "unknown",
                ["route"] = route.Kernel is not null ? "ptx" : "cublas",
                ["median-speedup"] = route.MedianSpeedup.ToString("R", CultureInfo.InvariantCulture),
                ["reason"] = route.Reason,
            };
            if (configuration is { } chosen)
            {
                parameters["kernel"] = chosen.Name;
                parameters["genome"] = PtxTensorCoreGemmEvolution.GenomeCodec.Serialize(chosen);
                parameters["codec-id"] = PtxTensorCoreGemmEvolution.GenomeCodec.Id;
                parameters["codec-version"] = PtxTensorCoreGemmEvolution.GenomeCodec.VersionHash;
            }
            return AutotuneCache.TryStore(Fp16AccumulateRouteKernelId, Fp16AccumulateShape(problem),
                new KernelChoice { Variant = Fp16AccumulateRouteVariant, Parameters = parameters });
        }
        catch (Exception ex) when (ex is InvalidOperationException or System.IO.IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    // Picks the fastest of the fixed candidates (and `extra`, first) that loads and reproduces cuBLAS, then times it
    // against cuBLAS.
    private Fp16AccumulateRoute TuneFp16AccumulateRoute(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, Fp16AccumulateProblem problem,
        PtxTensorCoreGemmConfiguration? extra = null)
    {
        bool halfOutput = problem.HalfOutput;
        int outputCount = checked((int)problem.OutputElements);
        DirectPtxRuntime runtime;
        try
        {
            runtime = GetDirectPtxRuntime();
        }
        catch (Exception ex)
        {
            return new Fp16AccumulateRoute(null, $"direct PTX unavailable ({ex.GetType().Name})", 0);
        }
        var layout = new PtxGemmLayout(false, false);
        PtxTensorCoreGemmKernel? best = null;
        double bestMs = double.MaxValue;
        try
        {
            // cuBLAS's output, the reference every candidate must reproduce (to FP16-accumulation accuracy).
            CublasFp16Accumulate(a, b, c, problem);
            float[] reference = DownloadGemmOutput(c, outputCount, halfOutput);
            // Finite outputs only: a NaN operand or an FP32 overflow in the caller's data must not reject every candidate
            // (the kernel's FP32 redo handles those inputs); non-finite outputs must agree exactly.
            double scale = 0;
            foreach (float v in reference)
                if (!float.IsNaN(v) && !float.IsInfinity(v)) scale = Math.Max(scale, Math.Abs((double)v));

            var candidates = new List<PtxTensorCoreGemmConfiguration>();
            if (extra is { } evolved) candidates.Add(evolved);
            foreach (var config in s_fp16AccumulateCandidates)
                if (!candidates.Contains(config)) candidates.Add(config);
            foreach (var config in candidates)
            {
                if (!AdmitsFp16Accumulate(config, layout, problem))
                    continue;
                PtxTensorCoreGemmKernel kernel;
                try
                {
                    kernel = new PtxTensorCoreGemmKernel(runtime, config, layout, PtxTensorCorePrecision.Fp16,
                        halfOutput, fp16Accumulate: true);
                }
                catch (InvalidOperationException)
                {
                    continue;   // spills or does not load on this device
                }
                LaunchFp16Accumulate(kernel, a, b, c, problem);
                float[] actual = DownloadGemmOutput(c, outputCount, halfOutput);
                double diff = 0;
                for (int i = 0; i < actual.Length; i++)
                {
                    float x = actual[i], r = reference[i];
                    if (float.IsNaN(r) || float.IsInfinity(r))
                    {
                        if (!(float.IsNaN(r) ? float.IsNaN(x) : x == r)) diff = double.PositiveInfinity;
                        continue;
                    }
                    diff = Math.Max(diff, Math.Abs((double)x - r));
                }
                if (!(diff <= Fp16AccumulateRelativeTolerance * Math.Max(scale, 1e-30)))
                {
                    kernel.Dispose();
                    continue;
                }
                double ms = TimeFp16Gemm(() => LaunchFp16Accumulate(kernel, a, b, c, problem), 10);
                if (ms < bestMs)
                {
                    best?.Dispose();
                    best = kernel;
                    bestMs = ms;
                }
                else
                {
                    kernel.Dispose();
                }
            }
            if (best is null) return new Fp16AccumulateRoute(null, "no candidate admits, loads and reproduces cuBLAS", 0);
            return VerdictAgainstCublas(best, a, b, c, problem);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            best?.Dispose();
            return new Fp16AccumulateRoute(null, $"tuning failed ({ex.GetType().Name}: {ex.Message})", 0);
        }
    }

    // Paired ABBA rounds of cuBLAS against the kernel. A verdict whose median wins but one round lost (one disturbed round
    // would otherwise keep cuBLAS for the process) is measured once more, and that second set of rounds must pass on its
    // own. Takes ownership of the kernel: kept in the route on a win, disposed otherwise.
    private Fp16AccumulateRoute VerdictAgainstCublas(PtxTensorCoreGemmKernel kernel, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        Fp16AccumulateProblem problem)
    {
        const int reps = 3;
        GemmRouteVerdict verdict;
        try
        {
            verdict = ConfirmFp16AccumulateVerdict(() => PairedRounds(
                () => { for (int i = 0; i < reps; i++) CublasFp16Accumulate(a, b, c, problem); },
                () => { for (int i = 0; i < reps; i++) LaunchFp16Accumulate(kernel, a, b, c, problem); },
                reps));
        }
        catch
        {
            kernel.Dispose();
            throw;
        }
        if (!verdict.Wins)
        {
            kernel.Dispose();
            return new Fp16AccumulateRoute(null, verdict.Reason, verdict.MedianSpeedup);
        }
        return new Fp16AccumulateRoute(kernel, $"{kernel.Configuration.Name}: {verdict.Reason}", verdict.MedianSpeedup);
    }

    // The first verdict, or, when its median won but a round lost, a second measurement's verdict on its own.
    internal static GemmRouteVerdict ConfirmFp16AccumulateVerdict(Func<GemmRouteVerdict> measure)
    {
        GemmRouteVerdict first = measure();
        if (first.Wins || first.MedianSpeedup < GemmRouteMedianMargin) return first;
        GemmRouteVerdict second = measure();
        return second with
        {
            Reason = $"{second.Reason} (re-measured: the first rounds' median {first.MedianSpeedup:F3}x won but one " +
                     $"round was {first.MinimumSpeedup:F3}x)",
        };
    }

    /// <summary>
    /// Evolves the FP16-accumulate tensor-core GEMM for <c>C[M,N] = A[M,K] @ B[K,N]</c> (FP16 in, FP32 out) on this device,
    /// takes the fastest of the winner and the fixed candidates that reproduces cuBLAS's output on operands in [-1, 1),
    /// times it against cuBLAS in paired rounds, and routes <see cref="GemmFp16In32fOut"/> for this shape to it only on a
    /// win (#1549). The verdict, either way, is persisted (unless <paramref name="persist"/> is false), so a later process
    /// applies it on the shape's first call without measuring.
    /// </summary>
    public async Task<(bool Routed, string Kernel, double MedianSpeedup, string Reason)> TuneFp16AccumulateRouteAsync(
        int m, int n, int k, EvolutionEngineOptions? engineOptions = null, KernelTuningOptions? tuningOptions = null,
        bool persist = true, bool halfOutput = false, CancellationToken cancellationToken = default)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (!IsAvailable) throw new GpuCapabilityUnavailableException("CUDA backend is not available.");
        DirectPtxRuntime runtime = GetDirectPtxRuntime();
        var layout = new PtxGemmLayout(false, false);
        var result = await PtxTensorCoreGemmEvolution.TuneAsync(runtime, layout, m, n, k, PtxTensorCorePrecision.Fp16,
            engineOptions, tuningOptions, cancellationToken: cancellationToken, fp16Accumulate: true).ConfigureAwait(false);
        PtxTensorCoreGemmConfiguration winner = result.ProposedWinner.Configuration;

        using IGpuBuffer a = AllocateByteBuffer(checked(m * k * 2));
        using IGpuBuffer b = AllocateByteBuffer(checked(k * n * 2));
        using IGpuBuffer c = halfOutput ? AllocateByteBuffer(checked(m * n * 2)) : AllocateBuffer(checked(m * n));
        var rng = new Random(m * 31 + n * 17 + k);
        UploadBytes(a, RandomHalfBytes(m * k, rng));
        UploadBytes(b, RandomHalfBytes(k * n, rng));

        // The search's winner competes with the fixed candidates on the same operands: its ranking is a short measurement,
        // so it must also beat them here to be kept.
        var problem = new Fp16AccumulateProblem(m, n, k, halfOutput, 1);
        Fp16AccumulateRoute route = TuneFp16AccumulateRoute(a, b, c, problem, winner);
        PtxTensorCoreGemmConfiguration chosen = route.Kernel?.Configuration ?? winner;
        ReplaceFp16AccumulateRoute(problem, route);

        if (persist) PersistFp16AccumulateRoute(problem, route, chosen);
        return (route.Kernel is not null, chosen.Name, route.MedianSpeedup, route.Reason);
    }

    // A verdict recorded for this device, MMA rate and problem (by a first call or TuneFp16AccumulateRouteAsync), or null
    // when there is none (or it is stale or does not load), in which case the caller measures.
    private Fp16AccumulateRoute? TryLoadPersistedFp16AccumulateRoute(Fp16AccumulateProblem problem)
    {
        if (Fp16AccumulateCacheUnavailableForTests) return null;
        KernelChoice? choice;
        try
        {
            choice = AutotuneCache.Lookup(Fp16AccumulateRouteKernelId, Fp16AccumulateShape(problem));
        }
        catch
        {
            return null;
        }
        if (choice?.Parameters is not { } p || !string.Equals(choice.Variant, Fp16AccumulateRouteVariant, StringComparison.Ordinal))
            return null;
        try
        {
            DirectPtxRuntime runtime = GetDirectPtxRuntime();
            if (!Matches(p, "device", PtxGemmEvolution.Fingerprint(runtime).LocalKey)) return null;
            // A verdict measured at another MMA rate (the card was unlocked or relocked since) is stale.
            if (!Matches(p, "mma-rate", MmaRate()?.RateClass ?? "unknown")) return null;
            string reason = p.TryGetValue("reason", out string? r) ? "persisted: " + r : "persisted";
            double median = p.TryGetValue("median-speedup", out string? ms) &&
                double.TryParse(ms, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) ? parsed : 0;
            if (Matches(p, "route", "cublas")) return new Fp16AccumulateRoute(null, reason, median);
            if (!Matches(p, "route", "ptx") || !p.TryGetValue("genome", out string? genome) ||
                !Matches(p, "codec-id", PtxTensorCoreGemmEvolution.GenomeCodec.Id) ||
                !Matches(p, "codec-version", PtxTensorCoreGemmEvolution.GenomeCodec.VersionHash)) return null;
            PtxTensorCoreGemmConfiguration configuration = PtxTensorCoreGemmEvolution.GenomeCodec.Deserialize(genome);
            var layout = new PtxGemmLayout(false, false);
            if (!AdmitsFp16Accumulate(configuration, layout, problem)) return null;
            var kernel = new PtxTensorCoreGemmKernel(runtime, configuration, layout, PtxTensorCorePrecision.Fp16,
                problem.HalfOutput, fp16Accumulate: true);
            return new Fp16AccumulateRoute(kernel, reason, median);
        }
        catch (Exception exception) when (exception is InvalidOperationException or ArgumentException or
                                             PlatformNotSupportedException or System.IO.InvalidDataException)
        {
            return null;
        }
    }

    // FP16 operands q / 256 for q in [-63, 63], exact in FP16, built from bits (no Half type: netstandard2.0 and net462
    // lack it): with p the position of |q|'s leading bit, the exponent is p - 8 and the fraction |q|'s remaining bits.
    private static byte[] RandomHalfBytes(int count, Random rng)
    {
        var bytes = new byte[count * 2];
        for (int i = 0; i < count; i++)
        {
            int q = rng.Next(-63, 64);
            ushort bits = 0;
            if (q != 0)
            {
                int magnitude = Math.Abs(q), leading = 0;
                while ((magnitude >> (leading + 1)) != 0) leading++;
                int fraction = (magnitude - (1 << leading)) << (10 - leading);
                bits = (ushort)((q < 0 ? 0x8000 : 0) | ((leading - 8 + 15) << 10) | fraction);
            }
            bytes[2 * i] = (byte)bits;
            bytes[2 * i + 1] = (byte)(bits >> 8);
        }
        return bytes;
    }

    // Milliseconds per launch of `launch`, after one warm-up, from CUDA events.
    private double TimeFp16Gemm(Action launch, int reps)
    {
        launch();
        using IGpuEvent start = CreateEvent(enableTiming: true);
        using IGpuEvent end = CreateEvent(enableTiming: true);
        RecordEvent(start, DefaultStream);
        for (int i = 0; i < reps; i++) launch();
        RecordEvent(end, DefaultStream);
        end.Synchronize();
        return GetEventElapsedTime(start, end) / reps;
    }

    private void CublasFp16Gemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k, bool halfOutput)
    {
        if (halfOutput) CublasGemmFp16HalfOut(a, b, c, m, n, k, 1f, 0f);
        else CublasGemmFp16In32fOut(a, b, c, m, n, k);
    }

    // The GEMM's output as floats: FP32 as stored, FP16 decoded from its bits.
    private float[] DownloadGemmOutput(IGpuBuffer c, int count, bool halfOutput)
    {
        if (!halfOutput) return DownloadBuffer(c);
        byte[] bytes = DownloadByteBuffer(c, count * 2);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = HalfBitsToSingle((ushort)(bytes[2 * i] | (bytes[2 * i + 1] << 8)));
        return values;
    }

    // IEEE binary16 bits to float, without a Half type (netstandard2.0 and net462 lack it).
    private static float HalfBitsToSingle(ushort bits)
    {
        int sign = bits >> 15, exponent = (bits >> 10) & 0x1F, fraction = bits & 0x3FF;
        float magnitude = exponent == 0x1F
            ? (fraction == 0 ? float.PositiveInfinity : float.NaN)
            : exponent == 0
                ? fraction * (1f / 16777216f)   // subnormal: fraction * 2^-24
                : (1f + fraction / 1024f) * (float)Math.Pow(2, exponent - 15);
        return sign != 0 ? -magnitude : magnitude;
    }

    /// <summary>Tests and probes: the cuBLAS FP16-in / FP32-out GEMM, bypassing every route.</summary>
    internal void CublasGemmFp16In32fOutForTests(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
        => CublasGemmFp16In32fOut(a, b, c, m, n, k);

    private void DisposeFp16AccumulateRoutes()
    {
        foreach (var route in _fp16AccumulateRoutes.Values) route.Dispose();
        _fp16AccumulateRoutes.Clear();
        lock (_fp16AccumulateTuneLock)
        {
            foreach (var route in _retiredFp16AccumulateRoutes) route.Dispose();
            _retiredFp16AccumulateRoutes.Clear();
        }
    }
}