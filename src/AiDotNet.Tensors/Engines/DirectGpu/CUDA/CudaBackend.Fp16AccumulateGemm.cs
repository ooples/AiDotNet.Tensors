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
/// FP16-accumulate routing of <see cref="GemmFp16In32fOut"/> (#1549). On GeForce cards FP16-accumulate tensor-core MMA
/// issues at full rate while FP32-accumulate issues at half rate, so a GEMM that accumulates each K tile in FP16 and
/// promotes it into an FP32 total can run up to twice as fast as cuBLAS's FP32-accumulate GEMM. Every routed launch is
/// guarded on the device: max|A| and max|B| are reduced first, and the kernel accumulates in FP16 only when
/// <c>max|A| * max|B| * BlockK &lt; 65504</c> (no tile's partial sum can overflow), otherwise in FP32. The first eager
/// call of a large shape times guarded candidates against cuBLAS in paired rounds (the reductions included) and routes
/// the shape only when the best one wins; the verdict lives for the process. AIDOTNET_FP16_ACCUMULATE=0 turns it off.
/// </summary>
public sealed partial class CudaBackend
{
    private static readonly bool s_fp16AccumulateEnabled =
        Environment.GetEnvironmentVariable("AIDOTNET_FP16_ACCUMULATE") != "0";

    // Below this much work the GEMM is bound by memory or latency, not the MMA rate, so FP16 accumulation buys nothing
    // (RTX 3080 probe: 3136x64x576 0.99x) while the guard's reductions still cost a pass over the operands.
    private const long Fp16AccumulateMinWork = 1L << 29;

    // The fastest configurations of the #1549 probe across the measured shapes.
    private static readonly PtxTensorCoreGemmConfiguration[] s_fp16AccumulateCandidates =
    {
        new(64, 128, 32, 64, 32, 3, 8),
        new(128, 128, 16, 64, 64, 3, 8),
        new(128, 64, 16, 64, 32, 4, 8),
        new(64, 64, 32, 32, 32, 4, 8),
        new(64, 64, 16, 32, 32, 3, 8),
    };

    private const string Fp16AccumulateRouteVariant = "cuda-fp16acc-route-v1";
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

    private readonly ConcurrentDictionary<(int M, int N, int K), Fp16AccumulateRoute> _fp16AccumulateRoutes = new();
    private readonly object _fp16AccumulateTuneLock = new();
    private IGpuBuffer? _fp16AccumulateAmax;   // two u32 words: max|A|, max|B| as FP16 bit patterns
    private long _fp16AccumulateDispatchCount;

    /// <summary>Routed FP16-accumulate GEMM launches on this backend (test and diagnostics hook).</summary>
    internal long Fp16AccumulateDispatchCount => Interlocked.Read(ref _fp16AccumulateDispatchCount);

    /// <summary>The verdict for one shape, or null when it was never evaluated (test and diagnostics hook).</summary>
    internal (bool Routed, string Reason, double MedianSpeedup)? Fp16AccumulateVerdict(int m, int n, int k) =>
        _fp16AccumulateRoutes.TryGetValue((m, n, k), out var route)
            ? (route.Kernel is not null, route.Reason, route.MedianSpeedup)
            : null;

    /// <summary>
    /// Tests: routes a shape to the guarded FP16-accumulate kernel of <paramref name="config"/> without timing it, so the
    /// routed path (including its device-side fallback) can be checked independently of which route measures faster.
    /// </summary>
    internal void ForceFp16AccumulateRouteForTests(int m, int n, int k, PtxTensorCoreGemmConfiguration config)
    {
        var kernel = new PtxTensorCoreGemmKernel(GetDirectPtxRuntime(), config, new PtxGemmLayout(false, false),
            PtxTensorCorePrecision.Fp16, fp16Accumulate: true, fp16AccumulateGuard: true);
        if (_fp16AccumulateRoutes.TryGetValue((m, n, k), out var previous)) previous.Dispose();
        _fp16AccumulateRoutes[(m, n, k)] = new Fp16AccumulateRoute(kernel, "forced for tests", 0);
    }

    /// <summary>
    /// Runs <c>C[M,N] = A[M,K] @ B[K,N]</c> (FP16 in, FP32 out, row-major) through the guarded FP16-accumulate kernel when
    /// this shape is routed; false leaves the call to cuBLAS. <paramref name="amaxA"/> / <paramref name="amaxB"/>, when not
    /// zero, are device words already holding max|A| / max|B| (a cached weight bound), so that reduction is skipped.
    /// </summary>
    internal bool TryFp16AccumulateGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k,
        IntPtr amaxA = default, IntPtr amaxB = default)
    {
        if (!s_fp16AccumulateEnabled || _ccMajor < 8 || (long)m * n * k < Fp16AccumulateMinWork) return false;
        if (!_fp16AccumulateRoutes.TryGetValue((m, n, k), out var route))
        {
            // Tuning measures with synchronizing events, which a stream capture cannot record: capture keeps cuBLAS until
            // an eager call of the shape has decided.
            if (IsStreamCapturing()) return false;
            lock (_fp16AccumulateTuneLock)
            {
                if (!_fp16AccumulateRoutes.TryGetValue((m, n, k), out route))
                {
                    // A verdict persisted by TuneFp16AccumulateRouteAsync (an evolved kernel for this device) wins over
                    // the quick pick among a few fixed candidates.
                    route = TryLoadPersistedFp16AccumulateRoute(m, n, k) ?? TuneFp16AccumulateRoute(a, b, c, m, n, k);
                    _fp16AccumulateRoutes[(m, n, k)] = route;
                }
            }
        }
        if (route.Kernel is null) return false;
        LaunchFp16Accumulate(route.Kernel, a, b, c, m, n, k, amaxA, amaxB);
        Interlocked.Increment(ref _fp16AccumulateDispatchCount);
        return true;
    }

    private void LaunchFp16Accumulate(PtxTensorCoreGemmKernel kernel, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int m, int n, int k, IntPtr amaxA, IntPtr amaxB)
    {
        IGpuBuffer amax = _fp16AccumulateAmax ??= AllocateBuffer(2);
        if (amaxA == IntPtr.Zero)
        {
            amaxA = amax.Handle;
            AbsMaxFp16Bits(a, m * k, amaxA);
        }
        if (amaxB == IntPtr.Zero)
        {
            amaxB = amax.Handle + 4;
            AbsMaxFp16Bits(b, k * n, amaxB);
        }
        using var _ = PushContext();
        lock (GpuDispatchLock)
            kernel.LaunchGuarded(a.Handle, b.Handle, c.Handle, m, n, k, k, n, n, 1f, 0f, amaxA, amaxB, stream: _stream);
    }

    private Fp16AccumulateRoute TuneFp16AccumulateRoute(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
    {
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
            CublasGemmFp16In32fOut(a, b, c, m, n, k);
            float[] reference = DownloadBuffer(c);
            double scale = 0;
            foreach (float v in reference) scale = Math.Max(scale, Math.Abs((double)v));

            foreach (var config in s_fp16AccumulateCandidates)
            {
                if (!PtxTensorCoreGemmGenerator.Admits(config, layout, m, n, k, k, n, n, 0, 0, 0, PtxTensorCorePrecision.Fp16))
                    continue;
                PtxTensorCoreGemmKernel kernel;
                try
                {
                    kernel = new PtxTensorCoreGemmKernel(runtime, config, layout, PtxTensorCorePrecision.Fp16,
                        fp16Accumulate: true, fp16AccumulateGuard: true);
                }
                catch (InvalidOperationException)
                {
                    continue;   // spills or does not load on this device
                }
                LaunchFp16Accumulate(kernel, a, b, c, m, n, k, IntPtr.Zero, IntPtr.Zero);
                float[] actual = DownloadBuffer(c);
                double diff = 0;
                for (int i = 0; i < actual.Length; i++) diff = Math.Max(diff, Math.Abs((double)actual[i] - reference[i]));
                if (!(diff <= 5e-3 * Math.Max(scale, 1e-30)))
                {
                    kernel.Dispose();
                    continue;
                }
                double ms = TimeFp16Gemm(() => LaunchFp16Accumulate(kernel, a, b, c, m, n, k, IntPtr.Zero, IntPtr.Zero), 3);
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
            return VerdictAgainstCublas(best, a, b, c, m, n, k);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            best?.Dispose();
            return new Fp16AccumulateRoute(null, $"tuning failed ({ex.GetType().Name}: {ex.Message})", 0);
        }
    }

    // Paired ABBA rounds of cuBLAS against the guarded kernel (its max-abs reductions included). Takes ownership of the
    // kernel: kept in the route on a win, disposed otherwise.
    private Fp16AccumulateRoute VerdictAgainstCublas(PtxTensorCoreGemmKernel kernel, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int m, int n, int k)
    {
        const int reps = 3;
        GemmRouteVerdict verdict;
        try
        {
            verdict = PairedRounds(
                () => { for (int i = 0; i < reps; i++) CublasGemmFp16In32fOut(a, b, c, m, n, k); },
                () => { for (int i = 0; i < reps; i++) LaunchFp16Accumulate(kernel, a, b, c, m, n, k, IntPtr.Zero, IntPtr.Zero); },
                reps);
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

    /// <summary>
    /// Evolves the FP16-accumulate tensor-core GEMM for <c>C[M,N] = A[M,K] @ B[K,N]</c> (FP16 in, FP32 out) on this device,
    /// checks the guarded winner reproduces cuBLAS's output on operands in [-1, 1), times it against cuBLAS in paired
    /// rounds (its max-abs reductions included), and routes <see cref="GemmFp16In32fOut"/> for this shape to it only on a
    /// win (#1549). The verdict, either way, is persisted (unless <paramref name="persist"/> is false), so a later process
    /// applies it on the shape's first call without measuring.
    /// </summary>
    public async Task<(bool Routed, string Kernel, double MedianSpeedup, string Reason)> TuneFp16AccumulateRouteAsync(
        int m, int n, int k, EvolutionEngineOptions? engineOptions = null, KernelTuningOptions? tuningOptions = null,
        bool persist = true, CancellationToken cancellationToken = default)
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
        using IGpuBuffer c = AllocateBuffer(checked(m * n));
        var rng = new Random(m * 31 + n * 17 + k);
        UploadBytes(a, RandomHalfBytes(m * k, rng));
        UploadBytes(b, RandomHalfBytes(k * n, rng));

        var kernel = new PtxTensorCoreGemmKernel(runtime, winner, layout, PtxTensorCorePrecision.Fp16,
            fp16Accumulate: true, fp16AccumulateGuard: true);
        CublasGemmFp16In32fOut(a, b, c, m, n, k);
        float[] reference = DownloadBuffer(c);
        LaunchFp16Accumulate(kernel, a, b, c, m, n, k, IntPtr.Zero, IntPtr.Zero);
        float[] actual = DownloadBuffer(c);
        double scale = 0, diff = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs((double)reference[i]));
            diff = Math.Max(diff, Math.Abs((double)actual[i] - reference[i]));
        }
        Fp16AccumulateRoute route = diff <= 5e-3 * Math.Max(scale, 1e-30)
            ? VerdictAgainstCublas(kernel, a, b, c, m, n, k)
            : DisposeAndReject(kernel, $"{winner.Name} disagrees with cuBLAS: max |diff| / max |C| = {diff / scale:E2}");
        if (_fp16AccumulateRoutes.TryGetValue((m, n, k), out var previous)) previous.Dispose();
        _fp16AccumulateRoutes[(m, n, k)] = route;

        if (persist)
        {
            var choice = new KernelChoice
            {
                Variant = Fp16AccumulateRouteVariant,
                Parameters = new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["device"] = PtxGemmEvolution.Fingerprint(runtime).LocalKey,
                    ["route"] = route.Kernel is not null ? "ptx" : "cublas",
                    ["kernel"] = winner.Name,
                    ["genome"] = PtxTensorCoreGemmEvolution.GenomeCodec.Serialize(winner),
                    ["codec-id"] = PtxTensorCoreGemmEvolution.GenomeCodec.Id,
                    ["codec-version"] = PtxTensorCoreGemmEvolution.GenomeCodec.VersionHash,
                    ["median-speedup"] = route.MedianSpeedup.ToString("R", CultureInfo.InvariantCulture),
                    ["reason"] = route.Reason,
                },
            };
            AutotuneCache.TryStore(Fp16AccumulateRouteKernelId, new ShapeProfile(m, n, k, 0), choice);
        }
        return (route.Kernel is not null, winner.Name, route.MedianSpeedup, route.Reason);

        static Fp16AccumulateRoute DisposeAndReject(PtxTensorCoreGemmKernel kernel, string reason)
        {
            kernel.Dispose();
            return new Fp16AccumulateRoute(null, reason, 0);
        }
    }

    // A verdict TuneFp16AccumulateRouteAsync persisted for this device and shape, or null when there is none (or it is
    // stale or does not load), in which case the caller measures.
    private Fp16AccumulateRoute? TryLoadPersistedFp16AccumulateRoute(int m, int n, int k)
    {
        KernelChoice? choice;
        try
        {
            choice = AutotuneCache.Lookup(Fp16AccumulateRouteKernelId, new ShapeProfile(m, n, k, 0));
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
            string reason = p.TryGetValue("reason", out string? r) ? "persisted: " + r : "persisted";
            double median = p.TryGetValue("median-speedup", out string? ms) &&
                double.TryParse(ms, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) ? parsed : 0;
            if (Matches(p, "route", "cublas")) return new Fp16AccumulateRoute(null, reason, median);
            if (!Matches(p, "route", "ptx") || !p.TryGetValue("genome", out string? genome) ||
                !Matches(p, "codec-id", PtxTensorCoreGemmEvolution.GenomeCodec.Id) ||
                !Matches(p, "codec-version", PtxTensorCoreGemmEvolution.GenomeCodec.VersionHash)) return null;
            PtxTensorCoreGemmConfiguration configuration = PtxTensorCoreGemmEvolution.GenomeCodec.Deserialize(genome);
            var layout = new PtxGemmLayout(false, false);
            if (!PtxTensorCoreGemmGenerator.Admits(configuration, layout, m, n, k, k, n, n, 0, 0, 0, PtxTensorCorePrecision.Fp16))
                return null;
            var kernel = new PtxTensorCoreGemmKernel(runtime, configuration, layout, PtxTensorCorePrecision.Fp16,
                fp16Accumulate: true, fp16AccumulateGuard: true);
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

    /// <summary>Tests and probes: the cuBLAS FP16-in / FP32-out GEMM, bypassing every route.</summary>
    internal void CublasGemmFp16In32fOutForTests(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
        => CublasGemmFp16In32fOut(a, b, c, m, n, k);

    private void DisposeFp16AccumulateRoutes()
    {
        foreach (var route in _fp16AccumulateRoutes.Values) route.Dispose();
        _fp16AccumulateRoutes.Clear();
        _fp16AccumulateAmax?.Dispose();
        _fp16AccumulateAmax = null;
    }
}