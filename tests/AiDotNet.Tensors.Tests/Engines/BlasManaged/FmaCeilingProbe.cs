using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #409 follow-up: what FP64 FMA throughput can managed .NET actually sustain on
/// this host? OpenBLAS dgemm hits ~62 GFLOPS here, so the hardware does ~2
/// FMA/cycle. This probe runs pure register-only FMA chains at varying widths to
/// find where the JIT-emitted code saturates — settling whether the BlasManaged
/// microkernel's ~21 GFLOPS is a managed-codegen ceiling or kernel headroom.
///
/// <para>Gated behind <c>AIDOTNET_RUN_FMA_PROBE=1</c>.</para>
/// </summary>
[Trait("Category", "Benchmark")]
public class FmaCeilingProbe
{
    private readonly ITestOutputHelper _output;
    public FmaCeilingProbe(ITestOutputHelper output) { _output = output; }

    private static double s_sink;
    private const long Iters = 60_000_000;

    [Fact]
    public void Fp64_FmaCeiling_ByChainWidth()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_FMA_PROBE") != "1") return;
        if (!Fma.IsSupported) { _output.WriteLine("FMA not supported."); return; }

        _output.WriteLine("FP64 register-only FMA ceiling (8 flops/FMA, target ~62 GFLOPS = OpenBLAS here):");
        _output.WriteLine($"  4 chains: {Chains4():F1} GFLOPS");
        _output.WriteLine($"  8 chains: {Chains8():F1} GFLOPS");
        _output.WriteLine($" 10 chains: {Chains10():F1} GFLOPS");
        _output.WriteLine($" 12 chains: {Chains12():F1} GFLOPS");
        _output.WriteLine($"(sink={s_sink:G4})");
    }

    private static double Gf(int chains, Stopwatch sw) => 2.0 * chains * 4 * Iters / sw.Elapsed.TotalSeconds / 1e9;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double Chains4()
    {
        var a = Vector256.Create(1.0000001); var b = Vector256.Create(0.9999999);
        Vector256<double> c0 = Vector256<double>.Zero, c1 = c0, c2 = c0, c3 = c0;
        for (long i = 0; i < 1_000_000; i++) c0 = Fma.MultiplyAdd(a, b, c0);
        var sw = Stopwatch.StartNew();
        for (long i = 0; i < Iters; i++)
        {
            c0 = Fma.MultiplyAdd(a, b, c0); c1 = Fma.MultiplyAdd(a, b, c1);
            c2 = Fma.MultiplyAdd(a, b, c2); c3 = Fma.MultiplyAdd(a, b, c3);
        }
        sw.Stop();
        s_sink += Avx.Add(Avx.Add(c0, c1), Avx.Add(c2, c3)).GetElement(0);
        return Gf(4, sw);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double Chains8()
    {
        var a = Vector256.Create(1.0000001); var b = Vector256.Create(0.9999999);
        Vector256<double> c0 = Vector256<double>.Zero, c1 = c0, c2 = c0, c3 = c0,
            c4 = c0, c5 = c0, c6 = c0, c7 = c0;
        for (long i = 0; i < 1_000_000; i++) c0 = Fma.MultiplyAdd(a, b, c0);
        var sw = Stopwatch.StartNew();
        for (long i = 0; i < Iters; i++)
        {
            c0 = Fma.MultiplyAdd(a, b, c0); c1 = Fma.MultiplyAdd(a, b, c1);
            c2 = Fma.MultiplyAdd(a, b, c2); c3 = Fma.MultiplyAdd(a, b, c3);
            c4 = Fma.MultiplyAdd(a, b, c4); c5 = Fma.MultiplyAdd(a, b, c5);
            c6 = Fma.MultiplyAdd(a, b, c6); c7 = Fma.MultiplyAdd(a, b, c7);
        }
        sw.Stop();
        var s = Avx.Add(Avx.Add(Avx.Add(c0, c1), Avx.Add(c2, c3)), Avx.Add(Avx.Add(c4, c5), Avx.Add(c6, c7)));
        s_sink += s.GetElement(0);
        return Gf(8, sw);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double Chains10()
    {
        var a = Vector256.Create(1.0000001); var b = Vector256.Create(0.9999999);
        Vector256<double> c0 = Vector256<double>.Zero, c1 = c0, c2 = c0, c3 = c0,
            c4 = c0, c5 = c0, c6 = c0, c7 = c0, c8 = c0, c9 = c0;
        for (long i = 0; i < 1_000_000; i++) c0 = Fma.MultiplyAdd(a, b, c0);
        var sw = Stopwatch.StartNew();
        for (long i = 0; i < Iters; i++)
        {
            c0 = Fma.MultiplyAdd(a, b, c0); c1 = Fma.MultiplyAdd(a, b, c1);
            c2 = Fma.MultiplyAdd(a, b, c2); c3 = Fma.MultiplyAdd(a, b, c3);
            c4 = Fma.MultiplyAdd(a, b, c4); c5 = Fma.MultiplyAdd(a, b, c5);
            c6 = Fma.MultiplyAdd(a, b, c6); c7 = Fma.MultiplyAdd(a, b, c7);
            c8 = Fma.MultiplyAdd(a, b, c8); c9 = Fma.MultiplyAdd(a, b, c9);
        }
        sw.Stop();
        var s = Avx.Add(Avx.Add(Avx.Add(c0, c1), Avx.Add(c2, c3)), Avx.Add(Avx.Add(c4, c5), Avx.Add(c6, c7)));
        s = Avx.Add(s, Avx.Add(c8, c9));
        s_sink += s.GetElement(0);
        return Gf(10, sw);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double Chains12()
    {
        var a = Vector256.Create(1.0000001); var b = Vector256.Create(0.9999999);
        Vector256<double> c0 = Vector256<double>.Zero, c1 = c0, c2 = c0, c3 = c0,
            c4 = c0, c5 = c0, c6 = c0, c7 = c0, c8 = c0, c9 = c0, c10 = c0, c11 = c0;
        for (long i = 0; i < 1_000_000; i++) c0 = Fma.MultiplyAdd(a, b, c0);
        var sw = Stopwatch.StartNew();
        for (long i = 0; i < Iters; i++)
        {
            c0 = Fma.MultiplyAdd(a, b, c0); c1 = Fma.MultiplyAdd(a, b, c1);
            c2 = Fma.MultiplyAdd(a, b, c2); c3 = Fma.MultiplyAdd(a, b, c3);
            c4 = Fma.MultiplyAdd(a, b, c4); c5 = Fma.MultiplyAdd(a, b, c5);
            c6 = Fma.MultiplyAdd(a, b, c6); c7 = Fma.MultiplyAdd(a, b, c7);
            c8 = Fma.MultiplyAdd(a, b, c8); c9 = Fma.MultiplyAdd(a, b, c9);
            c10 = Fma.MultiplyAdd(a, b, c10); c11 = Fma.MultiplyAdd(a, b, c11);
        }
        sw.Stop();
        var s = Avx.Add(Avx.Add(Avx.Add(c0, c1), Avx.Add(c2, c3)), Avx.Add(Avx.Add(c4, c5), Avx.Add(c6, c7)));
        s = Avx.Add(s, Avx.Add(Avx.Add(c8, c9), Avx.Add(c10, c11)));
        s_sink += s.GetElement(0);
        return Gf(12, sw);
    }
}
