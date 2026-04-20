using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Simd;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// Tensors-internal kernel inventory — the catalog entries that
/// <see cref="AutotuneCache.WarmupCommonKernelsAsync"/> populates when no
/// external registrations are present.
///
/// <para>Registered at module init so consumers who only call the new
/// public warmup API don't need to know internal KernelIds or variant
/// spaces. Today: one GEMM family (parallel-vs-sequential). Each new
/// tunable kernel (Conv2D variants, SDPA block sizes, …) plugs in here
/// as its integration lands.</para>
///
/// <para><b>Acceptance contract (issue #200):</b> "After
/// WarmupCommonKernelsAsync completes, <c>Lookup(id, shape)</c> returns
/// a non-null KernelChoice for every common kernel at every supplied
/// shape." Empty catalog trivially satisfies this; this class makes the
/// claim load-bearing by actually registering a kernel that benchmarks
/// and stores.</para>
///
/// <para><b>Concurrency contract — run warmup BEFORE serving traffic.</b>
/// The <c>SGEMM</c> benchmark below toggles the process-wide
/// <see cref="SimdGemm.UseParallelGemm"/> flag to measure each variant and
/// restores it before returning. This is safe only when no other thread is
/// calling <c>SimdGemm.Sgemm</c> concurrently with the warmup; a concurrent
/// GEMM caller would observe the wrong parallelism choice for the duration
/// of the benchmark, producing correct results but degraded (or inflated)
/// throughput attributable to the wrong variant. In practice warmup is
/// a startup-time operation run before the application handles requests,
/// so this window is empty. Future follow-up: introduce a scoped override
/// on <see cref="SimdGemm"/> (e.g.
/// <c>SimdGemm.WithParallelGemm(bool, Action)</c>) and route the benchmark
/// through it so concurrent callers are never perturbed.</para>
/// </summary>
internal static class BuiltInCatalog
{
    public static readonly KernelId SGEMM = new("gemm", "cpu-simd-sgemm");

    // Registration latch MUST be set only AFTER Register() successfully
    // completes. The previous Interlocked.Exchange-then-Register ordering
    // had two failure modes: (a) a concurrent caller could observe
    // _registered == 1 while Register() was still in flight and call
    // AutotuneCache.Lookup before the catalog entry was installed, and
    // (b) if Register() threw, _registered stayed at 1 forever and the
    // registration could never be retried. Fix via double-checked locking
    // with Volatile.Write after the successful registration completes.
    private static readonly object _registerGate = new();
    private static int _registered;

    /// <summary>Idempotently registers the built-in catalog entries.
    /// Called from <see cref="AutotuneCache"/>'s module init so the
    /// registry is warm before any Warmup call.</summary>
    public static void EnsureRegistered()
    {
        if (Volatile.Read(ref _registered) == 1) return;
        lock (_registerGate)
        {
            if (_registered == 1) return;
            AutotuneKernelCatalog.Register(new AutotuneCatalogEntry(
                SGEMM,
                variants: SgemmVariants,
                benchmarkVariant: BenchmarkSgemmVariant));
            // Publish only after Register() succeeds. If Register throws,
            // _registered stays 0 and the next caller retries.
            Volatile.Write(ref _registered, 1);
        }
    }

    /// <summary>Test hook — forces re-registration on the next EnsureRegistered
    /// call. Needed by AutotuneKernelCatalog.Clear tests so the module-init
    /// registrations don't shadow a Clear.</summary>
    public static void ResetRegistrationForTests() => Volatile.Write(ref _registered, 0);

    private static IEnumerable<string> SgemmVariants(ShapeProfile shape)
    {
        // Shape[0]*[1]*[2] = M*N*K — effective work in FMAs. Parallel dispatch
        // only makes sense when work justifies the thread-pool overhead; we
        // still ENUMERATE both so the benchmark can record that sequential
        // won at small shapes (consumers may still want that data).
        yield return "sequential";
        yield return "parallel";
    }

    private static Task<double> BenchmarkSgemmVariant(ShapeProfile shape, string variant, CancellationToken ct)
    {
        // Shape profile for GEMM is (M, N, K). Other ranks fall back to a
        // square-ish heuristic — pick the geometric mean as the side length
        // so the benchmark still runs something meaningful. Returning 0
        // would make the warmup skip this variant at that shape, which is
        // also acceptable.
        int[] dims = shape.Dimensions;
        int m, n, k;
        if (dims.Length >= 3)
        {
            m = dims[0]; n = dims[1]; k = dims[2];
        }
        else if (dims.Length == 2)
        {
            // [M, features] — treat features as N=K for square-ish GEMM.
            m = dims[0]; n = dims[1]; k = dims[1];
        }
        else
        {
            return Task.FromResult(0.0);
        }
        if (m <= 0 || n <= 0 || k <= 0) return Task.FromResult(0.0);

        // Compute allocation sizes in long so we can detect int-overflow
        // from large shapes BEFORE calling `new float[...]`. Overflowing
        // sizes would either wrap to a small/negative allocation (wrong
        // benchmark output) or push memory pressure into the tens of GB.
        // Return 0.0 (the agreed "not applicable" signal) so the warmup
        // driver skips this variant at this shape instead of throwing.
        long aLen = (long)m * k;
        long bLen = (long)k * n;
        long cLen = (long)m * n;
        if (aLen > int.MaxValue || bLen > int.MaxValue || cLen > int.MaxValue)
            return Task.FromResult(0.0);

        var a = new float[(int)aLen];
        var b = new float[(int)bLen];
        var c = new float[(int)cLen];
        var rng = new Random(0x515EE + variant.GetHashCode());
        for (int i = 0; i < a.Length; i++) a[i] = (float)rng.NextDouble();
        for (int i = 0; i < b.Length; i++) b[i] = (float)rng.NextDouble();

        // Warm up — let JIT compile and L1 prime.
        SimdGemm.Sgemm(a, b, c, m, k, n);
        SimdGemm.Sgemm(a, b, c, m, k, n);
        ct.ThrowIfCancellationRequested();

        // Measure. Toggle the process-wide parallel switch per variant.
        // SimdGemm exposes only the global flag today — save/restore bounds
        // the mutation to this benchmark. CALLER CONTRACT (see class
        // docstring): warmup must run when no other thread is calling
        // SimdGemm.Sgemm; a concurrent caller during this window would see
        // the wrong parallelism choice (correct results, wrong throughput
        // for that one call). The warmup driver calls BenchmarkVariant
        // sequentially so variants don't fight each other; a future scoped
        // SimdGemm.WithParallelGemm(...) helper will make this robust
        // against concurrent application callers too.
        bool savedParallel = SimdGemm.UseParallelGemm;
        SimdGemm.UseParallelGemm = variant == "parallel";
        const int iters = 5;
        var sw = Stopwatch.StartNew();
        try
        {
            for (int i = 0; i < iters; i++)
            {
                ct.ThrowIfCancellationRequested();
                SimdGemm.Sgemm(a, b, c, m, k, n);
            }
        }
        finally
        {
            SimdGemm.UseParallelGemm = savedParallel;
        }
        sw.Stop();
        double secsPerIter = sw.Elapsed.TotalSeconds / iters;
        if (secsPerIter <= 0) return Task.FromResult(0.0);
        double flops = 2.0 * m * k * n;
        return Task.FromResult(flops / 1e9 / secsPerIter);
    }
}
