using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using AiDotNet.Tensors.Helpers;
using Xunit;
using ManagedBlas = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using BMode = AiDotNet.Tensors.Engines.BlasManaged.BlasMode;
using PMode = AiDotNet.Tensors.Engines.BlasManaged.PackingMode;
using BOptF = AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float>;
using BOptD = AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<double>;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

// Drop-native audit (perf/managed-blas-vs-native-audit): quantify the managed-vs-native GEMM
// gap on the real hot shapes so we can decide whether the managed kernel can fully replace
// native OpenBLAS. Writes results to a FILE (xUnit ITestOutputHelper isn't captured for
// passing tests). Native gets ALL cores (deterministic mode otherwise pins OpenBLAS to 1).
// Category=Performance so CI (Category!=Performance) skips it.
// Joined to BlasManaged-Stats-Serial because Audit_ManagedVsNative_AcrossHotShapes calls
// BlasProvider.TrySetOpenBlasThreads (mutating the PROCESS-GLOBAL OpenBLAS thread count). Left
// uncollected it raced the bit-match/determinism tests in that collection under xUnit's parallel
// collections, flipping their reduction order. Serializing it here is the same remedy applied to
// the other global-BLAS-state mutators (see the collection definition in BlasManagedCollections).
[Collection("BlasManaged-Stats-Serial")]
public class ManagedVsNativeGemmAudit
{
    private static readonly (int M, int K, int N, string Note)[] Shapes =
    {
        (512, 512, 512,   "square-med / attn-proj"),
        (640, 640, 640,   "square-mid (microkernel→packed crossover)"),
        (1024, 1024, 1024,"square-large"),
        (2048, 2048, 2048,"square-xl"),
        (512, 512, 2048,  "ffn-up (d->4d)"),
        (512, 2048, 512,  "ffn-down (4d->d)"),
        (256, 784, 128,   "mlp hidden"),
        (128, 64, 256,    "lstm-small"),
        (4096, 512, 16,   "thin-N (dcgan-ish)"),
        (64, 4096, 64,    "thin-M wide-K"),
        (4096, 64, 64,    "tall-M thin-K"),
    };

    private static readonly string ResultFile =
        Path.Combine(Path.GetTempPath(), "blas-managed-vs-native-audit.txt");

    [Trait("Category", "Performance")]
    [Fact]
    public unsafe void Audit_ManagedVsNative_AcrossHotShapes()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# Managed (BlasManaged.Gemm) vs native (OpenBLAS cblas) — cores={Environment.ProcessorCount}");
        sb.AppendLine($"# native available (HasRawSgemm)={BlasProvider.HasRawSgemm}");
        sb.AppendLine($"# GFLOP/s = 2*M*N*K / time; higher is better. ratio = managed/native (>1 = managed wins)");
        sb.AppendLine();

        // Give native all cores for a fair comparison (det mode pins it to 1).
        BlasProvider.TrySetOpenBlasThreads(Environment.ProcessorCount);

        sb.AppendLine($"{"shape",-18} {"dt",-6} {"det GF",9} {"detPB GF",10} {"fast GF",9} {"fastPB GF",10} {"native GF",10} {"best/nat",9}  winner");
        foreach (var (m, k, n, note) in Shapes)
        {
            RunFloat(sb, m, k, n, note);
            RunDouble(sb, m, k, n, note);
        }

        Directory.CreateDirectory(Path.GetDirectoryName(ResultFile)!);
        File.WriteAllText(ResultFile, sb.ToString());
        // Always passes — this is a measurement, not a gate.
        Assert.True(true);
    }

    private static double GflopsOf(long mnk2, double ms) => ms <= 0 ? 0 : mnk2 / (ms * 1e6);

    private static unsafe void RunFloat(StringBuilder sb, int m, int k, int n, string note)
    {
        var rnd = new Random(12345);
        var a = new float[m * k]; var b = new float[k * n];
        var cM = new float[m * n]; var cN = new float[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rnd.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rnd.NextDouble() - 0.5);
        long flops = 2L * m * n * k;
        int iters = IterFor(flops);

        double det = BestMs(() =>
            ManagedBlas.Gemm<float>(a, k, false, b, n, false, cM, n, m, n, k), 3, iters);
        double detPB = BestMs(() =>
        {
            var o = new BOptF { PackingMode = PMode.ForcePackBoth };
            ManagedBlas.Gemm<float>(a, k, false, b, n, false, cM, n, m, n, k, in o);
        }, 3, iters);
        double fast = BestMs(() =>
        {
            var o = new BOptF { Mode = BMode.Fast };
            ManagedBlas.Gemm<float>(a, k, false, b, n, false, cM, n, m, n, k, in o);
        }, 3, iters);
        double fastPB = BestMs(() =>
        {
            var o = new BOptF { Mode = BMode.Fast, PackingMode = PMode.ForcePackBoth };
            ManagedBlas.Gemm<float>(a, k, false, b, n, false, cM, n, m, n, k, in o);
        }, 3, iters);

        double nat = double.PositiveInfinity;
        if (BlasProvider.HasRawSgemm)
            nat = BestMs(() =>
            {
                fixed (float* pa = a) fixed (float* pb = b) fixed (float* pc = cN)
                    BlasProvider.SgemmRaw(m, n, k, pa, k, pb, n, pc, n);
            }, 3, iters);

        Emit(sb, m, k, n, "float", note,
            GflopsOf(flops, det), GflopsOf(flops, detPB), GflopsOf(flops, fast), GflopsOf(flops, fastPB), GflopsOf(flops, nat));
    }

    private static unsafe void RunDouble(StringBuilder sb, int m, int k, int n, string note)
    {
        var rnd = new Random(54321);
        var a = new double[m * k]; var b = new double[k * n];
        var cM = new double[m * n]; var cN = new double[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = rnd.NextDouble() - 0.5;
        for (int i = 0; i < b.Length; i++) b[i] = rnd.NextDouble() - 0.5;
        long flops = 2L * m * n * k;
        int iters = IterFor(flops);

        double det = BestMs(() =>
            ManagedBlas.Gemm<double>(a, k, false, b, n, false, cM, n, m, n, k), 3, iters);
        double detPB = BestMs(() =>
        {
            var o = new BOptD { PackingMode = PMode.ForcePackBoth };
            ManagedBlas.Gemm<double>(a, k, false, b, n, false, cM, n, m, n, k, in o);
        }, 3, iters);
        double fast = BestMs(() =>
        {
            var o = new BOptD { Mode = BMode.Fast };
            ManagedBlas.Gemm<double>(a, k, false, b, n, false, cM, n, m, n, k, in o);
        }, 3, iters);
        double fastPB = BestMs(() =>
        {
            var o = new BOptD { Mode = BMode.Fast, PackingMode = PMode.ForcePackBoth };
            ManagedBlas.Gemm<double>(a, k, false, b, n, false, cM, n, m, n, k, in o);
        }, 3, iters);

        double nat = double.PositiveInfinity;
        if (BlasProvider.HasRawDgemm)
            nat = BestMs(() =>
            {
                fixed (double* pa = a) fixed (double* pb = b) fixed (double* pc = cN)
                    BlasProvider.DgemmRaw(m, n, k, pa, k, pb, n, pc, n);
            }, 3, iters);

        Emit(sb, m, k, n, "double", note,
            GflopsOf(flops, det), GflopsOf(flops, detPB), GflopsOf(flops, fast), GflopsOf(flops, fastPB), GflopsOf(flops, nat));
    }

    // More iters → better chance of catching an uncontended sample (min-of-N is the robust
    // "peak achievable" estimator on a loaded box). Large shapes were too noisy at 8 iters.
    private static int IterFor(long flops) => flops > 4_000_000_000L ? 40 : flops > 200_000_000L ? 60 : flops > 20_000_000L ? 120 : 300;

    private static double BestMs(Action op, int warm, int iters)
    {
        for (int i = 0; i < warm; i++) op();
        double best = double.PositiveInfinity;
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            op();
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            if (ms < best) best = ms;
        }
        return best;
    }

    private static void Emit(StringBuilder sb, int m, int k, int n, string dtype, string note,
        double detGf, double detPbGf, double fastGf, double fastPbGf, double nativeGf)
    {
        double bestManaged = Math.Max(Math.Max(detGf, detPbGf), Math.Max(fastGf, fastPbGf));
        double ratio = nativeGf > 0 ? bestManaged / nativeGf : 0;
        string winner = double.IsInfinity(nativeGf) || nativeGf == 0 ? "(no native)"
            : ratio >= 1.0 ? "MANAGED" : $"native (+{(1 / ratio - 1) * 100:F0}%)";
        sb.AppendLine($"{$"{m}x{k}x{n}",-18} {dtype,-6} {detGf,9:F1} {detPbGf,10:F1} {fastGf,9:F1} {fastPbGf,10:F1} {nativeGf,10:F1} {ratio,9:F2}  {winner}  [{note}]");
    }
}
