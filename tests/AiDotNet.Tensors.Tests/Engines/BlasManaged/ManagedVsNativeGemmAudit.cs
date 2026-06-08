using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using AiDotNet.Tensors.Helpers;
using Xunit;
using ManagedBlas = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

// Drop-native audit (perf/managed-blas-vs-native-audit): quantify the managed-vs-native GEMM
// gap on the real hot shapes so we can decide whether the managed kernel can fully replace
// native OpenBLAS. Writes results to a FILE (xUnit ITestOutputHelper isn't captured for
// passing tests). Native gets ALL cores (deterministic mode otherwise pins OpenBLAS to 1).
// Category=Performance so CI (Category!=Performance) skips it.
public class ManagedVsNativeGemmAudit
{
    private static readonly (int M, int K, int N, string Note)[] Shapes =
    {
        (512, 512, 512,   "square-med / attn-proj"),
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

        sb.AppendLine($"{"shape",-22} {"dtype",-6} {"managed GF",12} {"native GF",12} {"ratio",8}  winner");
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

        double mManaged = BestMs(() =>
            ManagedBlas.Gemm<float>(a, k, false, b, n, false, cM, n, m, n, k), warm: 3, iters: IterFor(flops));

        double mNative = double.PositiveInfinity;
        if (BlasProvider.HasRawSgemm)
        {
            mNative = BestMs(() =>
            {
                fixed (float* pa = a) fixed (float* pb = b) fixed (float* pc = cN)
                    BlasProvider.SgemmRaw(m, n, k, pa, k, pb, n, pc, n);
            }, warm: 3, iters: IterFor(flops));
        }

        Emit(sb, m, k, n, "float", note, GflopsOf(flops, mManaged), GflopsOf(flops, mNative));
    }

    private static unsafe void RunDouble(StringBuilder sb, int m, int k, int n, string note)
    {
        var rnd = new Random(54321);
        var a = new double[m * k]; var b = new double[k * n];
        var cM = new double[m * n]; var cN = new double[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = rnd.NextDouble() - 0.5;
        for (int i = 0; i < b.Length; i++) b[i] = rnd.NextDouble() - 0.5;
        long flops = 2L * m * n * k;

        double mManaged = BestMs(() =>
            ManagedBlas.Gemm<double>(a, k, false, b, n, false, cM, n, m, n, k), warm: 3, iters: IterFor(flops));

        double mNative = double.PositiveInfinity;
        if (BlasProvider.HasRawSgemm)
        {
            mNative = BestMs(() =>
            {
                fixed (double* pa = a) fixed (double* pb = b) fixed (double* pc = cN)
                    BlasProvider.DgemmRaw(m, n, k, pa, k, pb, n, pc, n);
            }, warm: 3, iters: IterFor(flops));
        }

        Emit(sb, m, k, n, "double", note, GflopsOf(flops, mManaged), GflopsOf(flops, mNative));
    }

    private static int IterFor(long flops) => flops > 500_000_000L ? 8 : flops > 50_000_000L ? 30 : 200;

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

    private static void Emit(StringBuilder sb, int m, int k, int n, string dtype, string note, double managedGf, double nativeGf)
    {
        double ratio = nativeGf > 0 ? managedGf / nativeGf : 0;
        string winner = double.IsInfinity(nativeGf) || nativeGf == 0 ? "(no native)"
            : ratio >= 1.0 ? "MANAGED" : $"native (+{(1 / ratio - 1) * 100:F0}%)";
        sb.AppendLine($"{$"{m}x{k}x{n}",-22} {dtype,-6} {managedGf,12:F1} {nativeGf,12:F1} {ratio,8:F2}  {winner}  [{note}]");
    }
}
