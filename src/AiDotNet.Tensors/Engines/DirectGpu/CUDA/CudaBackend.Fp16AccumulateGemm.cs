// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Threading;
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
                    route = TuneFp16AccumulateRoute(a, b, c, m, n, k);
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

            const int reps = 3;
            var winner = best;
            GemmRouteVerdict verdict = PairedRounds(
                () => { for (int i = 0; i < reps; i++) CublasGemmFp16In32fOut(a, b, c, m, n, k); },
                () => { for (int i = 0; i < reps; i++) LaunchFp16Accumulate(winner, a, b, c, m, n, k, IntPtr.Zero, IntPtr.Zero); },
                reps);
            if (!verdict.Wins)
            {
                best.Dispose();
                return new Fp16AccumulateRoute(null, verdict.Reason, verdict.MedianSpeedup);
            }
            return new Fp16AccumulateRoute(best, $"{best.Configuration.Name}: {verdict.Reason}", verdict.MedianSpeedup);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            best?.Dispose();
            return new Fp16AccumulateRoute(null, $"tuning failed ({ex.GetType().Name}: {ex.Message})", 0);
        }
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