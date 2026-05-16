using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Issue #338 Phase C.1: parallelization primitive for backward op
/// implementations.
///
/// <para>
/// Wraps <see cref="CpuParallelSettings.ParallelForOrSerial"/> with two
/// additional guarantees that matter for backward kernels:
/// </para>
/// <para>
/// 1. <b>BLAS thread-pool isolation:</b> when a backward function
///    internally calls a BLAS GEMM (which uses its own native thread
///    pool), parallelizing the outer loop too creates two competing
///    thread pools that oversubscribe the cores. <see cref="ForRows"/>
///    is for op bodies that DO NOT call BLAS; for those that do, the
///    outer parallelism is skipped and the BLAS internal threading
///    handles the work.
/// </para>
/// <para>
/// 2. <b>Recursion guard:</b> when a backward function is called from
///    inside another already-parallelized backward (Hvp-style higher-
///    order AD), nested parallelism doubles thread-pool pressure. The
///    <c>[ThreadStatic]</c> <see cref="_inBackwardParallel"/> flag
///    disables inner parallelism when an outer pass is active.
/// </para>
/// <para>
/// Use <see cref="MinWorkForParallel"/> as the gate before dispatching:
/// per-element work × element count must exceed the threshold to make
/// parallelism win over its own overhead. Empirically 64K MACs is the
/// crossover on 16-core x64.
/// </para>
/// </summary>
internal static class BackwardParallel
{
    /// <summary>
    /// Crossover threshold (total scalar work) above which parallel
    /// dispatch beats serial execution. Below this, the
    /// <see cref="Parallel.For"/> setup cost dominates the body work.
    /// </summary>
    internal const long MinWorkForParallel = 64L * 1024;

    /// <summary>
    /// Max parallelism for backward operations. Capped lower than
    /// the global <see cref="CpuParallelSettings.MaxDegreeOfParallelism"/>
    /// because most backward ops have diminishing returns past 8 threads
    /// at d=128 transformer shapes (per-row work is small, pool overhead
    /// dominates). Tuneable via env var <c>AIDOTNET_BWD_MAX_DOP</c>.
    /// </summary>
    internal static readonly int MaxDegreeOfParallelism = ReadMaxDop();

    private static int ReadMaxDop()
    {
        var raw = Environment.GetEnvironmentVariable("AIDOTNET_BWD_MAX_DOP");
        if (int.TryParse(raw, out int n) && n > 0)
            return Math.Min(n, Environment.ProcessorCount);
        return Math.Min(8, Math.Max(1, Environment.ProcessorCount));
    }

    [ThreadStatic]
    private static bool _inBackwardParallel;

    /// <summary>
    /// Parallel-for over rows, with BLAS-safe / recursion-safe gates.
    /// Falls back to serial when work is below
    /// <see cref="MinWorkForParallel"/> OR when an outer backward
    /// parallel pass is already in flight on this thread.
    /// </summary>
    /// <param name="rows">Total rows to iterate.</param>
    /// <param name="workPerRow">Estimated scalar work per row body.</param>
    /// <param name="body">Row index → body. MUST NOT call BLAS — use
    /// <see cref="ForRowsSerialOnBlasPath"/> when the body invokes
    /// <c>BlasProvider.TryGemm</c> or similar.</param>
    internal static void ForRows(int rows, long workPerRow, Action<int> body)
    {
        if (rows <= 0) return;
        long totalWork = rows * Math.Max(workPerRow, 1);

        // Recursion gate: if we're already inside a parallel backward
        // pass, nested parallelism is a net loss.
        if (_inBackwardParallel
            || totalWork < MinWorkForParallel
            || MaxDegreeOfParallelism <= 1
            || rows == 1)
        {
            for (int i = 0; i < rows; i++) body(i);
            return;
        }

        _inBackwardParallel = true;
        try
        {
            var po = new ParallelOptions { MaxDegreeOfParallelism = MaxDegreeOfParallelism };
            Parallel.For(0, rows, po, body);
        }
        finally
        {
            _inBackwardParallel = false;
        }
    }

    /// <summary>
    /// Variant for op bodies that internally call BLAS — always runs
    /// serially. BLAS handles its own parallelism via its native thread
    /// pool; outer parallelism here would oversubscribe cores. Exposed
    /// to make the choice explicit at call sites.
    /// </summary>
    internal static void ForRowsSerialOnBlasPath(int rows, Action<int> body)
    {
        for (int i = 0; i < rows; i++) body(i);
    }

    /// <summary>
    /// Runs two independent actions in parallel when both have
    /// sufficient work AND no outer pass is active. Used for the
    /// common backward pattern "compute dA and dB independently".
    /// Falls back to serial otherwise.
    /// </summary>
    internal static void Invoke2(long workA, long workB, Action a, Action b)
    {
        bool canParallel = !_inBackwardParallel
            && MaxDegreeOfParallelism >= 2
            && workA >= MinWorkForParallel
            && workB >= MinWorkForParallel;

        if (!canParallel)
        {
            a();
            b();
            return;
        }

        _inBackwardParallel = true;
        try
        {
            Parallel.Invoke(
                new ParallelOptions { MaxDegreeOfParallelism = 2 },
                a, b);
        }
        finally
        {
            _inBackwardParallel = false;
        }
    }
}
