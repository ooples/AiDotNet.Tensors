using System;
using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-S (#409) Phase J2 — the JIT-emitted FP64 4×8 packed microkernel must be
/// bit-identical to the hand-written <see cref="Avx2Fp64_4x8.Run"/> (same op
/// order) and at least as fast (its reason to exist).
/// </summary>
public class Fp64MicrokernelEmitterTests
{
    private readonly ITestOutputHelper _output;
    public Fp64MicrokernelEmitterTests(ITestOutputHelper output) { _output = output; }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(7, 8)]      // odd kc
    [InlineData(32, 8)]
    [InlineData(256, 8)]
    [InlineData(37, 13)]    // ldc > Nr (padded C), odd kc
    [InlineData(64, 16)]
    public unsafe void EmittedKernel_IsBitIdenticalToHandWritten(int kc, int ldc)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return; // no AVX2 on this host

        const int Mr = 4, Nr = 8;
        var rng = new Random(123 + kc * 31 + ldc);
        var a = new double[kc * Mr];
        var b = new double[kc * Nr];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        // C is read-modify-write — start both copies from the same random state.
        int cLen = (Mr - 1) * ldc + Nr;
        var c0 = new double[cLen];
        for (int i = 0; i < cLen; i++) c0[i] = rng.NextDouble() * 2 - 1;
        var cHand = (double[])c0.Clone();
        var cJit = (double[])c0.Clone();

        Avx2Fp64_4x8.Run(a, b, cHand, ldc, kc);

        var kernel = Fp64MicrokernelEmitter.Emit(kc);
        fixed (double* pa = a) fixed (double* pb = b) fixed (double* pc = cJit)
            kernel(pa, pb, pc, ldc);

        for (int i = 0; i < cLen; i++)
            Assert.True(BitConverter.DoubleToInt64Bits(cHand[i]) == BitConverter.DoubleToInt64Bits(cJit[i]),
                $"Mismatch at {i}: hand={cHand[i]:G17}, jit={cJit[i]:G17} (kc={kc}, ldc={ldc})");
    }

    [Fact]
    public unsafe void EmittedKernel_PerfVsHandWritten()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;
        if (!Avx2.IsSupported || !Fma.IsSupported) { _output.WriteLine("no AVX2"); return; }

        const int Mr = 4, Nr = 8, kc = 256, ldc = 8;
        var rng = new Random(42);
        var a = new double[kc * Mr];
        var b = new double[kc * Nr];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        var c = new double[Mr * ldc];

        var kernel = Fp64MicrokernelEmitter.Emit(kc);
        double flopsPerCall = 2.0 * Mr * Nr * kc;
        const int iters = 2_000_000;

        // Hand-written.
        for (int i = 0; i < 2000; i++) Avx2Fp64_4x8.Run(a, b, c, ldc, kc);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) Avx2Fp64_4x8.Run(a, b, c, ldc, kc);
        sw.Stop();
        double handGf = flopsPerCall * iters / sw.Elapsed.TotalSeconds / 1e9;

        // JIT-emitted.
        fixed (double* pa = a) fixed (double* pb = b) fixed (double* pc = c)
        {
            for (int i = 0; i < 2000; i++) kernel(pa, pb, pc, ldc);
            sw.Restart();
            for (int i = 0; i < iters; i++) kernel(pa, pb, pc, ldc);
            sw.Stop();
        }
        double jitGf = flopsPerCall * iters / sw.Elapsed.TotalSeconds / 1e9;

        _output.WriteLine($"FP64 4x8 kc={kc}: hand-written {handGf:F1} GFLOPS | JIT-emitted {jitGf:F1} GFLOPS | {jitGf / handGf:F2}x");
    }
}
