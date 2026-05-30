using System;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors;

/// <summary>
/// Process-wide CPU inference tuning knobs. The one that matters for latency is
/// the native-BLAS thread count: all-core BLAS saturates large training GEMMs but
/// <b>oversubscribes</b> the small-batch, wide-but-short GEMMs typical of inference
/// (a classifier head's 784→512 layer, a ResNet 1×1, etc.), where the cost of
/// spinning up every core outweighs the work.
///
/// <para><b>Measured (AiDotNet.Tensors Phase-7F probe, 16-core box):</b> running the
/// wide AIsEval MLP layers on <c>ProcessorCount/2</c> BLAS threads beats all-core by
/// ~1.3× at every batch (e.g. 784→512 at M=1: 53 vs 68 µs; 512→128 at M=32: 39 vs
/// 58 µs). The win is only realizable when the thread count is <b>pinned once</b> —
/// flipping it per call is catastrophic because <c>openblas_set_num_threads</c>
/// rebuilds the thread pool (~700 µs) on every change. Hence this is a
/// <b>startup</b> knob, not a per-op one.</para>
///
/// <para><b>For Beginners:</b> call <see cref="PinBlasThreadsForLatency"/> once when
/// your serving process starts (before handling requests). It tells the math library
/// to use about half your CPU cores for each matrix multiply, which — perhaps
/// counter-intuitively — makes small-batch prediction <i>faster</i> by avoiding the
/// overhead of waking every core for a tiny amount of work. Leave it alone for
/// throughput-oriented batch/training workloads, which benefit from all cores.</para>
/// </summary>
public static class CpuInferenceConfig
{
    /// <summary>
    /// Pins the native-BLAS (OpenBLAS) thread count for the lifetime of the process —
    /// the recommended one-time startup call for low-latency CPU inference. Call this
    /// once at process startup; do <b>not</b> call it per request (see the type remarks
    /// — changing the count rebuilds the BLAS thread pool).
    /// </summary>
    /// <param name="threads">
    /// Desired BLAS thread count. When <c>null</c> (the default), uses the latency-tuned
    /// default of <c>max(1, Environment.ProcessorCount / 2)</c>. Pass an explicit value
    /// to override (e.g. <c>1</c> for fully deterministic single-threaded GEMM, or the
    /// full core count to restore throughput-oriented behavior).
    /// </param>
    /// <returns>The thread count that was applied.</returns>
    /// <remarks>
    /// No-op (returns the requested count) when no native BLAS is available — the managed
    /// SIMD GEMM path manages its own parallelism via the engine's work thresholds.
    /// Idempotent: re-applying the same count does not rebuild the pool.
    /// </remarks>
    public static int PinBlasThreadsForLatency(int? threads = null)
    {
        int target = threads ?? Math.Max(1, Environment.ProcessorCount / 2);
        BlasProvider.TrySetOpenBlasThreads(target);
        return target;
    }
}
