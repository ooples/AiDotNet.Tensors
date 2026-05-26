using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-S (#409): validates the hand-emitted machine-code path — first the
/// executable-memory + call infrastructure (a trivial x+1 function), then the
/// 12-accumulator FMA-throughput kernel that breaks RyuJIT's 8-register cap.
/// </summary>
public class MachineCodeKernelTests
{
    private readonly ITestOutputHelper _output;
    public MachineCodeKernelTests(ITestOutputHelper output) { _output = output; }

    private static bool IsX64Windows =>
        RuntimeInformation.OSArchitecture == Architecture.X64
        && RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

    private static bool IsX64Unix =>
        RuntimeInformation.OSArchitecture == Architecture.X64
        && (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || RuntimeInformation.IsOSPlatform(OSPlatform.OSX));

    [Fact]
    public unsafe void Infra_TrivialFunction_ExecutesAndReturns()
    {
        if (!IsX64Windows && !IsX64Unix) return; // x64-only proof

        // long f(long x) => x + 1.
        // Windows x64: arg0 in RCX. SysV (Linux/macOS): arg0 in RDI.
        byte[] code = IsX64Windows
            ? new byte[] { 0x48, 0x89, 0xC8,       // mov rax, rcx
                           0x48, 0xFF, 0xC0,       // inc rax
                           0xC3 }                  // ret
            : new byte[] { 0x48, 0x89, 0xF8,       // mov rax, rdi
                           0x48, 0xFF, 0xC0,       // inc rax
                           0xC3 };                 // ret

        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) { _output.WriteLine("executable memory not supported — skipping"); return; }

        var fn = (delegate* unmanaged<long, long>)mem.Pointer;
        Assert.Equal(42L, fn(41L));
        Assert.Equal(1L, fn(0L));
        Assert.Equal(-4L, fn(-5L));
    }

    [Fact]
    public unsafe void Fp64x12_MachineCode_BreaksRyuJitCeiling()
    {
        if (!IsX64Windows) return;                 // proof encoded for Windows x64
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        byte[] code = MachineCodeFmaKernel.EmitFp64x12Windows();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null) { _output.WriteLine("executable memory not supported — skipping"); return; }

        var fn = (delegate* unmanaged<long, double*, double*, void>)mem.Pointer;
        const double a = 1.0000001, b = 0.9999999;
        var ab = new double[] { a, b };
        var res = new double[4];

        fixed (double* pab = ab, pres = res)
        {
            // Correctness: each of 12 accumulators sums `iters` copies of a*b, then
            // they're reduced lane-wise → result = 12 * iters * a*b.
            fn(1000, pres, pab);
            double expected = 12.0 * 1000 * (a * b);
            Assert.True(Math.Abs(pres[0] - expected) < expected * 1e-9,
                $"machine-code FMA wrong: got {pres[0]:G17}, expected {expected:G17}");
            // All 4 lanes identical (broadcast operands).
            Assert.Equal(pres[0], pres[1]);
            Assert.Equal(pres[0], pres[3]);

            // Perf: does our own register allocation beat RyuJIT's 8-accumulator
            // ceiling (~44 GFLOPS) and approach the hardware peak (~64)?
            const long iters = 80_000_000;
            fn(1_000_000, pres, pab); // warm
            var sw = Stopwatch.StartNew();
            fn(iters, pres, pab);
            sw.Stop();
            double flops = 12.0 * iters * 4 * 2; // 12 acc × 4 lanes × 2 flops/FMA
            double gflops = flops / sw.Elapsed.TotalSeconds / 1e9;
            _output.WriteLine($"machine-code 12-acc FP64 FMA: {gflops:F1} GFLOPS " +
                "(RyuJIT 8-acc ceiling ~44; hardware peak ~64)");
            Assert.True(gflops > 44.0,
                $"machine-code kernel {gflops:F1} GFLOPS did not beat the 44 GFLOPS RyuJIT ceiling");
        }
    }
}
