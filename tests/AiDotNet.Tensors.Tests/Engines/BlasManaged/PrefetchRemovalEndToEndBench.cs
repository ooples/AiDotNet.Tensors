using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #409 S.3: end-to-end validation that the microkernel prefetch-removal lifts
/// real GEMM (default dispatch → PackBoth → Avx2*_Run) without regressing larger
/// Kc shapes. Reports managed default-dispatch vs native OpenBLAS for FP32+FP64.
/// Gated behind <c>AIDOTNET_RUN_409_E2E=1</c>; native BLAS required.
/// </summary>
[Trait("Category", "Benchmark")]
[Collection("BlasManaged-Perf-Serial")]
public class PrefetchRemovalEndToEndBench
{
    private readonly ITestOutputHelper _output;
    public PrefetchRemovalEndToEndBench(ITestOutputHelper output) { _output = output; }

    private sealed record Shape(string Name, int M, int N, int K);

    private static readonly Shape[] Shapes =
    [
        new("FFN_up_512x2048x512",    512,  2048, 512),
        new("FFN_down_512x512x2048",  512,  512,  2048),
        new("BERT_FFN_1024x3072x768", 1024, 3072, 768),
        new("Large_1024sq",           1024, 1024, 1024),
        new("Square_2048",            2048, 2048, 2048),   // larger Kc-block stress
    ];

    [Fact]
    public void E2E_ManagedVsNative_FP64() => Run(isDouble: true);

    [Fact]
    public void E2E_ManagedVsNative_FP32() => Run(isDouble: false);

    private void Run(bool isDouble)
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_409_E2E") != "1") return;
        if (!BlasProvider.IsAvailable) { _output.WriteLine("native BLAS not loaded — skip."); return; }

        _output.WriteLine($"#409 S.3 end-to-end ({(isDouble ? "FP64" : "FP32")}) — host {HardwareFingerprint.Current}");
        _output.WriteLine($"{"Shape",-26} {"native",10} {"managed",10} {"mgd/nat",8}");
        _output.WriteLine(new string('-', 60));
        foreach (var s in Shapes)
        {
            double nat = isDouble ? NativeD(s) : NativeF(s);
            double mgd = isDouble ? ManagedD(s) : ManagedF(s);
            _output.WriteLine($"{s.Name,-26} {nat,9:F3}m {mgd,9:F3}m {mgd / nat,7:F1}x");
        }
    }

    private static int Iters(Shape s)
    {
        long w = (long)s.M * s.N * s.K;
        return w > 1_000_000_000L ? 5 : w > 100_000_000L ? 15 : 40;
    }

    private double ManagedD(Shape s)
    {
        var (a, b, c) = BufD(s);
        int it = Iters(s);
        for (int i = 0; i < 3; i++) BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K);
        var t = new double[it]; var sw = new Stopwatch();
        for (int i = 0; i < it; i++) { sw.Restart(); BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K); sw.Stop(); t[i] = sw.Elapsed.TotalMilliseconds; }
        Array.Sort(t); return t[it / 2];
    }
    private double NativeD(Shape s)
    {
        var (a, b, c) = BufD(s);
        int it = Iters(s);
        for (int i = 0; i < 3; i++) BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N);
        var t = new double[it]; var sw = new Stopwatch();
        for (int i = 0; i < it; i++) { sw.Restart(); BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N); sw.Stop(); t[i] = sw.Elapsed.TotalMilliseconds; }
        Array.Sort(t); return t[it / 2];
    }
    private double ManagedF(Shape s)
    {
        var (a, b, c) = BufF(s);
        int it = Iters(s);
        for (int i = 0; i < 3; i++) BlasManagedLib.Gemm<float>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K);
        var t = new double[it]; var sw = new Stopwatch();
        for (int i = 0; i < it; i++) { sw.Restart(); BlasManagedLib.Gemm<float>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K); sw.Stop(); t[i] = sw.Elapsed.TotalMilliseconds; }
        Array.Sort(t); return t[it / 2];
    }
    private double NativeF(Shape s)
    {
        var (a, b, c) = BufF(s);
        int it = Iters(s);
        for (int i = 0; i < 3; i++) BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N);
        var t = new double[it]; var sw = new Stopwatch();
        for (int i = 0; i < it; i++) { sw.Restart(); BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N); sw.Stop(); t[i] = sw.Elapsed.TotalMilliseconds; }
        Array.Sort(t); return t[it / 2];
    }

    private static (double[] a, double[] b, double[] c) BufD(Shape s)
    {
        var r = new Random(42); var a = new double[s.M * s.K]; var b = new double[s.K * s.N]; var c = new double[s.M * s.N];
        for (int i = 0; i < a.Length; i++) a[i] = r.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = r.NextDouble() * 2 - 1;
        return (a, b, c);
    }
    private static (float[] a, float[] b, float[] c) BufF(Shape s)
    {
        var r = new Random(42); var a = new float[s.M * s.K]; var b = new float[s.K * s.N]; var c = new float[s.M * s.N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(r.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(r.NextDouble() * 2 - 1);
        return (a, b, c);
    }
}
